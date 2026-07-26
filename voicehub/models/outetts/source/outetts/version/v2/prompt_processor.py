from transformers import AutoTokenizer

from .tokens import SpecialTokens
from .text_processor import TextProcessor

class PromptProcessor:
    def __init__(self, tokenizer_path: str):
        if tokenizer_path:
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

        self.special_tokens = SpecialTokens()

        self.text_prompt = "{bos}\n{text_start}{words}{text_end}\n"

        if tokenizer_path:
            self.map_audio_tokens = self.get_audio_token_map()

        self.text_processor = TextProcessor()

    def get_audio_token_map(self) -> dict:
        return {
            self.tokenizer.encode(self.special_tokens.audio_code.format(i), add_special_tokens=False)[0]: i
            for i in range(4100)
        }

    def create_audio_prompt(self, processed, speaker):
        word_data = speaker["words"]
        words = []
        idx = 0  # Separate index for words

        for i in processed:
            if "<|emotion_start|>" in i["word"]:
                words.append(i["word"])
                continue

            s = word_data[idx]
            if s["word"] != i["word"]:
                raise ValueError(
                    f"Word mismatch at index {idx}:\n"
                    f"Expected word: '{s['word']}'\n"
                    f"Found word: '{i['word']}'"
                )

            word = s["word"]
            if i.get("before", []):
                word = "".join(i["before"]) + word
            if i.get("after", []):
                word += "".join(i["after"])

            tokens = "".join([self.special_tokens.audio_code.format(c) for c in s["codes"]])
            word = f"{word}{self.special_tokens.time.format(s['duration'])}{tokens}"
            
            words.append(word)
            idx += 1  # Only increment word index for non-emotion words
        
        words = (self.special_tokens.space+"\n").join(words)
        return words

    def get_completion_prompt(self, text: str, emotion: str = None, speaker: dict = None) -> str:
        words = self.text_processor.process_text(text)
        words = self.text_processor.get_text(words)
        if speaker is not None:
            speaker_text = self.text_processor.process_text(speaker['text'])
            audio = self.create_audio_prompt(speaker_text, speaker)
            words = self.text_processor.get_text(speaker_text) + self.special_tokens.space + words

        prompt = self.text_prompt.format(
            bos=self.special_tokens.bos, 
            text_start=self.special_tokens.text_start,
            words=words, 
            text_end=self.special_tokens.text_end,
        )
        if emotion is not None:
            prompt += f"{self.special_tokens.voice_characteristic_start}{emotion}{self.special_tokens.voice_characteristic_end}\n"
        prompt += self.special_tokens.audio_start + "\n"

        if speaker is not None:
            prompt += audio
            prompt += self.special_tokens.space + "\n"

        return prompt

    def get_training_prompt(self, speaker: dict) -> str:
        emotion = speaker.get("emotion", None)
        speaker_text = self.text_processor.process_text(speaker['text'])
        audio = self.create_audio_prompt(speaker_text, speaker)
        words = self.text_processor.get_text(speaker_text)
        prompt = self.text_prompt.format(
            bos=self.special_tokens.bos, 
            text_start=self.special_tokens.text_start,
            words=words, 
            text_end=self.special_tokens.text_end,
        )
        if emotion is not None:
            prompt += f"{self.special_tokens.voice_characteristic_start}{emotion}{self.special_tokens.voice_characteristic_end}\n"
        prompt += self.special_tokens.audio_start + "\n"

        prompt += audio
        prompt += "\n" + self.special_tokens.audio_end + "\n" + self.special_tokens.eos + "\n"
        return prompt

    def extract_audio_from_tokens(self, tokens: list[int]) -> list[int]:
        return [self.map_audio_tokens[i] for i in tokens if i in self.map_audio_tokens]

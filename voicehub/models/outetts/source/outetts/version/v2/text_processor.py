import re
import inflect
import uroman as ur
import MeCab
import string
from loguru import logger

from ...anyascii import anyascii
from .tokens import SpecialTokens

def clean_dashes(text: str) -> str:
    dashes = ['—', '–', '-']
    for dash in dashes:
        text = text.replace(dash, ' ')
    return text

class TextProcessor:
    def __init__(self):
        self.special_tokens = SpecialTokens()

        escaped_punct = [re.escape(p) for p in self.special_tokens.punctuation_tokens.keys()]
        self.PUNCT_PATTERN = re.compile(f"[^a-zA-Z\u0080-\uffff{''.join(escaped_punct)}\\s]")
        # Pre-compile regex patterns
        self.WHITESPACE_PATTERN = re.compile(r'(?<!\s)\s(?!\s)')
        self.MULTIPLE_SPACES_PATTERN = re.compile(r'\s+')
        self.NUMBER_PATTERN = re.compile(r'\d+(\.\d+)?')
        self.CLEAN_PATTERN = re.compile(r'[^a-z\s]')

        self.wakati = MeCab.Tagger("-Owakati")
        self.lec = inflect.engine()
        self.uroman = ur.Uroman()

    def join_punctuation(self, text: str, pun: dict):
        text = text.split()
        i = 0
        while i < len(text):
            if text[i] in pun and i > 0: 
                text[i-1] = text[i-1] + text[i] 
                text.pop(i) 
            else:
                i += 1
        # Remove standalone punctuation tokens that weren't successfully joined to words
        return [word for word in text if word not in pun and not all(c in string.punctuation for c in word)]
    
    def normalize_token_spacing(self, text: str):
        text = text.replace("'", "").replace("`", "")
        text = self.wakati.parse(text).split()
        text = [i for i in text if i not in ['', ' ']]
        result = text[0]  
        pun = self.special_tokens.punctuation_tokens
        for token in text[1:]:
            if token not in pun:
                result += '  '
            else:
                result += ' '
            result += token
        return self.WHITESPACE_PATTERN.sub('', result)
    
    def _process_text(self, text: str):
        text = self.normalize_token_spacing(text)
        text = anyascii(text)
        text = clean_dashes(text)
        text = self.NUMBER_PATTERN.sub(lambda x: self.lec.number_to_words(x.group()), text.lower())
        text = self.PUNCT_PATTERN.sub('', text)
        text = self.MULTIPLE_SPACES_PATTERN.sub(' ', text).strip().lower()
        original = text

        text = self.CLEAN_PATTERN.sub('', text)
        text = [{"word": i} for i in text.split()]
      
        pun = self.special_tokens.punctuation_tokens
        original_split = self.join_punctuation(original, pun)

        # Covers some edge cases
        for idx, orig_word in enumerate(original_split):
            processed_word = text[idx]['word']
            match = re.search(processed_word, orig_word.lower())
           
            if match:
                start, end = match.span()
                text[idx]["before"] = [pun[c] for c in orig_word[:start] if c in pun]
                text[idx]["after"] = [pun[c] for c in orig_word[end:] if c in pun]
            else:
                text[idx]["before"] = []
                text[idx]["after"] = []

        return text

    def process_text(self, text: str):
        chunks = re.split(r'(<\|emotion_start\|>.*?<\|emotion_end\|>)', text)
        final_result = []
        for chunk in chunks:
            if chunk.strip().startswith('<|emotion_start|>'):
                # Add emotion block unprocessed
                final_result.append({
                    'word': chunk,
                    'before': [],
                    'after': []
                })
            else:
                if chunk.strip():
                    processed = self._process_text(chunk)
                    final_result.extend(processed)
        return final_result

    def process_text_clean_only(self, text: str):
        text = self.normalize_token_spacing(text)
        text = anyascii(text)
        text = clean_dashes(text)
        text = self.NUMBER_PATTERN.sub(lambda x: self.lec.number_to_words(x.group()), text.lower())
        text = self.PUNCT_PATTERN.sub('', text)
        text = self.MULTIPLE_SPACES_PATTERN.sub(' ', text).strip().lower()
        text = self.CLEAN_PATTERN.sub('', text)
        return text

    def get_text(self, data: dict):
        temp = []
        for i in data:
            word = i["word"]
            if i.get("before",[]):
                word = "".join(i["before"]) + word
            if i.get("after", []):
                word += "".join(i["after"])
            temp.append(word)
        return self.special_tokens.space.join(temp)
    
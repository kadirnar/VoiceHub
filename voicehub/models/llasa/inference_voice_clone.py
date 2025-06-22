import soundfile as sf
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from xcodec2.modeling_xcodec2 import XCodec2Model


class LlasaVoiceClone:

    def __init__(self, model_path: str = "HKUSTAudio/Llasa-1B-Multilingual", device: str = "cuda"):
        self.device = device
        self.load_models(model_path, device)

    def load_models(self, model_path: str, device: str):
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(model_path).eval().to(device)
        codec_model = XCodec2Model.from_pretrained("HKUSTAudio/xcodec2").eval().to(device)

        self.tokenizer = tokenizer
        self.model = model
        self.codec_model = codec_model

    def ids_to_speech_tokens(self, speech_ids):
        speech_tokens_str = []
        for speech_id in speech_ids:
            speech_tokens_str.append(f"<|s_{speech_id}|>")
        return speech_tokens_str

    def _extract_speech_ids(self, speech_tokens_str):
        speech_ids = []
        for token_str in speech_tokens_str:
            if token_str.startswith('<|s_') and token_str.endswith('|>'):
                num_str = token_str[4:-2]
                num = int(num_str)
                speech_ids.append(num)
            else:
                print(f"Unexpected token: {token_str}")
        return speech_ids

    def __call__(
            self, prompt_wav_path: str, prompt_text: str, target_text: str, output_file: str = "gen.wav"):
        """
        Generate voice cloned speech based on a prompt audio and text.

        Args:
            prompt_wav_path: Path to the prompt audio file (16kHz)
            prompt_text: Text corresponding to the prompt audio
            target_text: Text to be synthesized with the voice from prompt
            output_file: Path to save the generated audio
        """
        # Load prompt audio
        prompt_wav, sr = sf.read(prompt_wav_path)
        prompt_wav = torch.from_numpy(prompt_wav).float().unsqueeze(0)

        # Combine prompt and target text
        input_text = prompt_text + target_text

        with torch.no_grad():
            # Encode the prompt wav
            vq_code_prompt = self.codec_model.encode_code(input_waveform=prompt_wav)
            vq_code_prompt = vq_code_prompt[0, 0, :]

            # Convert int to token <|s_number|>
            speech_ids_prefix = self.ids_to_speech_tokens(vq_code_prompt)

            # Format text for the model
            formatted_text = f"<|TEXT_UNDERSTANDING_START|>{input_text}<|TEXT_UNDERSTANDING_END|>"

            # Tokenize the text and the speech prefix
            chat = [{
                "role": "user",
                "content": "Convert the text to speech:" + formatted_text
            }, {
                "role": "assistant",
                "content": "<|SPEECH_GENERATION_START|>" + ''.join(speech_ids_prefix)
            }]

            input_ids = self.tokenizer.apply_chat_template(
                chat, tokenize=True, return_tensors='pt', continue_final_message=True)
            input_ids = input_ids.to(self.device)
            speech_end_id = self.tokenizer.convert_tokens_to_ids('<|SPEECH_GENERATION_END|>')

            # Generate the speech autoregressively
            outputs = self.model.generate(
                input_ids,
                max_length=2048,
                eos_token_id=speech_end_id,
                do_sample=True,
                top_p=1,
                temperature=0.8,
            )

            # Extract the speech tokens
            generated_ids = outputs[0][input_ids.shape[1] - len(speech_ids_prefix):-1]
            speech_tokens = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

            # Convert token <|s_number|> to int
            speech_tokens = self._extract_speech_ids(speech_tokens)
            speech_tokens = torch.tensor(speech_tokens).to(self.device).unsqueeze(0).unsqueeze(0)

            # Decode the speech tokens to speech waveform
            gen_wav = self.codec_model.decode_code(speech_tokens)

            # Save the generated audio
            sf.write(output_file, gen_wav[0, 0, :].cpu().numpy(), 16000)

            return output_file

import soundfile as sf
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from xcodec2.modeling_xcodec2 import XCodec2Model


class LlasaTTS:

    def __init__(self, model_path: str = "HKUSTAudio/Llasa-1B-Multilingual", device: str = "cuda"):
        self.load_models(model_path, device)

    def load_models(self, model_path: str, device: str):
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained().eval(model_path).to(device)
        codec_model = XCodec2Model.from_pretrained("HKUSTAudio/xcodec2").eval().to(device)

        self.tokenizer = tokenizer
        self.model = model
        self.codec_model = codec_model

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

    def __call__(self, input_text, output_file: str = "output.wav"):
        with torch.no_grad():
            formatted_text = f"<|TEXT_UNDERSTANDING_START|>{input_text}<|TEXT_UNDERSTANDING_END|>"

            chat = [{
                "role": "user",
                "content": "Convert the text to speech:" + formatted_text
            }, {
                "role": "assistant",
                "content": "<|SPEECH_GENERATION_START|>"
            }]
            input_ids = self.tokenizer.apply_chat_template(
                chat, tokenize=True, return_tensors='pt', continue_final_message=True)
            input_ids = input_ids.to('cuda')
            speech_end_id = self.tokenizer.convert_tokens_to_ids('<|SPEECH_GENERATION_END|>')
            outputs = self.model.generate(
                input_ids,
                max_length=2048,
                eos_token_id=speech_end_id,
                do_sample=True,
                top_p=1,
                temperature=0.8)
            generated_ids = outputs[0][input_ids.shape[1]:-1]
            speech_tokens = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            speech_tokens = self._extract_speech_ids(speech_tokens)
            speech_tokens = torch.tensor(speech_tokens).cuda().unsqueeze(0).unsqueeze(0)
            gen_wav = self.codec_model.decode_code(speech_tokens)
            sf.write(output_file, gen_wav[0, 0, :].cpu().numpy(), 16000)

from transformers import AutoModelForCausalLM, AutoTokenizer
from snac import SNAC
import torch
import soundfile as sf
from typing import Optional

class OrpheusTTS:
    def __init__(self,
                 model_path: str = None,
                 device: Optional[str] = None
    ):
        self.model = None
        self.snac_model = None
        self.device = None
        self.load_models(model_path, device)

    def load_models(self,
                    model_path: str = None,
                    device: Optional[str] = None):
        """
        Load the text generation model, tokenizer, and SNAC codec model.
        """
        self.model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.snac_model = SNAC.from_pretrained("hubertsiuzdak/snac_24khz").to(self.device)

    def _prepare_inputs(self,
                        prompt: str,
                        chosen_voice: str):
        """
        Tokenize, add special tokens, pad, and build attention masks.
        """
        start_token = torch.tensor([[128259]], dtype=torch.int64)  # SOH
        end_tokens = torch.tensor([[128009, 128260]], dtype=torch.int64)  # EOT, EOH

        formatted_prompt = f"{chosen_voice}: " + prompt
        ids = self.tokenizer(formatted_prompt, return_tensors="pt").input_ids
        ids = torch.cat([start_token, ids, end_tokens], dim=1)

        # Create attention mask (no padding needed for single prompt)
        attention_mask = torch.ones(1, ids.shape[1], dtype=torch.int64)
        input_ids = ids.to(self.device)
        attention_mask = attention_mask.to(self.device)
        return input_ids, attention_mask

    # _generate method has been merged into __call__

    def _decode_audio(self, generated_ids: torch.Tensor) -> list[torch.Tensor]:
        """
        Strip special tokens, rebatch into SNAC code layers, and decode to audio.
        """
        # locate last human speech token (128257), crop thereafter
        token_to_find = 128257
        token_to_remove = 128258

        idxs = (generated_ids == token_to_find).nonzero(as_tuple=True)
        if len(idxs[1]) > 0:
            last = idxs[1][-1].item()
            cropped = generated_ids[:, last + 1:]
        else:
            cropped = generated_ids

        # remove end-token padding
        rows = []
        for row in cropped:
            filtered = row[row != token_to_remove]
            rows.append(filtered)

        # group into codec code-lists of 7, remap token IDs to codebook indices
        def _redistribute(codes: list[int]):
            l1, l2, l3 = [], [], []
            for i in range((len(codes)//7)):
                base = 7 * i
                l1.append(codes[base] - 128266)
                l2.append(codes[base+1] - 128266 - 4096)
                # layer 3 accumulates several:
                l3.extend([
                    codes[base+2] - 128266 - 2*4096,
                    codes[base+3] - 128266 - 3*4096,
                    codes[base+5] - 128266 - 5*4096,
                    codes[base+6] - 128266 - 6*4096
                ])
                l2.append(codes[base+4] - 128266 - 4*4096)
            return [
                torch.tensor(l1, device=self.device).unsqueeze(0),
                torch.tensor(l2, device=self.device).unsqueeze(0),
                torch.tensor(l3, device=self.device).unsqueeze(0),
            ]

        audios = []
        for row in rows:
            codes = _redistribute(row.tolist())
            codes = [c.to(self.device) for c in codes]
            audio = self.snac_model.decode(codes)  # [1, T]
            audios.append(audio.squeeze(0))
        return audios

    def __call__(self,
                 prompt: str,
                 chosen_voice: str = "tara",
                 max_new_tokens: int = 1200,
                 temperature: float = 0.6,
                 top_p: float = 0.95,
                 repetition_penalty: float = 1.1,
                 output_path: str = "output.wav") -> str:
        """
        Main entry point. Takes a list of raw prompt strings,
        generates speech, saves .wav files, and returns file paths.
        """
        input_ids, attention_mask = self._prepare_inputs(prompt, chosen_voice)
        # Generate raw token sequences directly in __call__
        with torch.no_grad():
            generated = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                eos_token_id=128258,
            )
        audio_tensors = self._decode_audio(generated)
        # Since we're only processing one prompt, we only have one audio tensor
        audio = audio_tensors[0]
        sf.write(output_path, audio.cpu().numpy(), 24000)
        return output_path

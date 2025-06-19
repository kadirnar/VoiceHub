import re
import time

import inflect
import torch
import torch.nn.functional as F
import torchaudio
from torch import Tensor
from torch.nn.attention import SDPBackend, sdpa_kernel

from voicehub.models.vui.model import Vui
from voicehub.models.vui.sampling import multinomial, sample_top_k, sample_top_p, sample_top_p_top_k
from voicehub.models.vui.vad import detect_voice_activity as vad


def ensure_spaces_around_tags(text: str):
    # Add space before '[' if not preceded by space, '<', or '['
    text = re.sub(
        r"(?<![<\[\s])(\[)",
        lambda m: (f"\n{m.group(1)}" if m.start() > 0 and text[m.start() - 1] == "\n" else f" {m.group(1)}"),
        text,
    )
    # Add space after ']' if not preceded by digit+']' and not followed by space, '>', or ']'
    text = re.sub(
        r"(?<!\d\])(\])(?![>\]\s])",
        lambda m: (f"{m.group(1)}\n" if m.end() < len(text) and text[m.end()] == "\n" else f"{m.group(1)} "),
        text,
    )
    text = text.strip()
    return text


REPLACE = [
    ("—", ","),
    ("'", "'"),
    (":", ","),
    (";", ","),
]

engine = None
wm = None


def asr(chunk, model=None, prefix=None):
    import whisper

    global wm
    if model is not None:
        wm = model
    elif wm is None:
        wm = whisper.load_model("turbo", "cuda")

    chunk = whisper.pad_or_trim(chunk)
    mel = whisper.log_mel_spectrogram(chunk, n_mels=wm.dims.n_mels).to(wm.device)
    options = whisper.DecodingOptions(language="en", without_timestamps=True, prefix=prefix)
    result = whisper.decode(wm, mel[None], options)
    return result[0].text


def replace_numbers_with_words(text):
    global engine

    if engine is None:
        engine = inflect.engine()

    # Function to convert a number match to words
    def number_to_words(match):
        number = match.group()
        return engine.number_to_words(number) + " "

    # Replace digits with their word equivalents
    return re.sub(r"\d+", number_to_words, text)


valid_non_speech = ["breath", "sigh", "laugh", "tut", "hesitate"]
valid_non_speech = [f"[{v}]" for v in valid_non_speech]


def remove_all_invalid_non_speech(txt):
    """
    Remove all non-speech markers that are not in the valid_non_speech list.

    Only keeps valid non-speech markers like [breath], [sigh], etc.
    """
    # Find all text within square brackets
    bracket_pattern = r"\[([^\]]+)\]"
    brackets = re.findall(bracket_pattern, txt)

    # For each bracketed text, check if it's in our valid list
    for bracket in brackets:
        bracket_with_brackets = f"[{bracket}]"
        if bracket_with_brackets not in valid_non_speech and bracket != "pause":
            # If not valid, remove it from the text
            txt = txt.replace(bracket_with_brackets, "")

    return txt


def simple_clean(text):
    text = re.sub(r"(\d+)am", r"\1 AM", text)
    text = re.sub(r"(\d+)pm", r"\1 PM", text)
    text = replace_numbers_with_words(text)
    text = ensure_spaces_around_tags(text)
    text = remove_all_invalid_non_speech(text)

    text = text.replace('"', "")
    text = text.replace("”", "")
    text = text.replace("“", "")
    text = text.replace("’", "'")
    text = text.replace("%", " percent")
    text = text.replace("*", "")
    text = text.replace("(", "")
    text = text.replace(")", "")
    text = text.replace(";", "")
    text = text.replace("–", " ")
    text = text.replace("—", "")
    text = text.replace(":", "")
    text = text.replace("…", "...")
    text = text.replace("s...", "s")

    # replace repeating \n with just one \n
    text = re.sub(r"\n+", "\n", text)
    ntxt = re.sub(r" +", " ", text)

    # Ensure that ntxt ends with . or ?
    ntxt = ntxt.strip()
    if not ntxt.endswith(".") or ntxt.endswith("?"):
        ntxt += "."
    ntxt += " [pause]"
    return ntxt


@torch.inference_mode()
def generate(
        self: Vui,
        text: str,
        prompt_codes: Tensor | None = None,
        temperature: float = 0.5,
        top_k: int | None = 150,
        top_p: float | None = None,
        max_gen_len: int = int(120 * 21.53),
):
    text = simple_clean(text)
    with (
            torch.autocast("cuda", torch.bfloat16, True),
            sdpa_kernel([SDPBackend.MATH]),
    ):
        t1 = time.perf_counter()
        batch_size = 1
        device = self.device
        self.dtype
        self.decoder.allocate_inference_cache(batch_size, device, torch.bfloat16)

        texts = [text]

        encoded = self.tokenizer(
            texts,
            padding="longest",
            return_tensors="pt",
        )

        input_ids = encoded.input_ids.to(device)
        text_embeddings = self.token_emb(input_ids)

        B = batch_size
        Q = self.config.model.n_quantizers

        if prompt_codes is None:
            prompt_codes = torch.zeros((batch_size, Q, 0), dtype=torch.int64, device=device)
        else:
            prompt_codes = prompt_codes[:, :Q].repeat(batch_size, 1, 1)

        start_offset = prompt_codes.size(-1)

        pattern = self.pattern_provider.get_pattern(max_gen_len)
        # this token is used as default value for codes that are not generated yet
        unknown_token = -1
        special_token_id = self.config.model.special_token_id

        # we generate codes up to the max_gen_len that will be mapped to the pattern sequence
        codes = torch.full((B, Q, max_gen_len), unknown_token, dtype=torch.int64, device=device)
        print("codes", codes.shape)

        codes[:, :, :start_offset] = prompt_codes

        sequence, indexes, mask = pattern.build_pattern_sequence(codes, special_token_id)
        # retrieve the start_offset in the sequence:
        # it is the first sequence step that contains the `start_offset` timestep
        start_offset_sequence = pattern.get_first_step_with_timesteps(start_offset)
        assert start_offset_sequence is not None

        prev_offset = 0
        S = sequence.size(-1)

        do_prefill = True
        eos = self.config.model.audio_eos_id

        for offset in range(start_offset_sequence, S):
            # print(f"{prev_offset}:{offset}")
            curr_sequence = sequence[..., prev_offset:offset]
            audio_embeddings = (sum([self.audio_embeddings[q](curr_sequence[:, q]) for q in range(Q)]) / Q)

            if do_prefill:
                embeddings = torch.cat((text_embeddings, audio_embeddings), dim=1)
                T = embeddings.size(1)
                input_pos = torch.arange(0, T, device=device)
                do_prefill = False
            else:
                embeddings = audio_embeddings
                input_pos = torch.tensor([T], device=device)
                T += 1

            out = self.decoder(embeddings, input_pos)

            if offset == 15:
                print("TTFB", time.perf_counter() - t1)

            logits = torch.stack([self.audio_heads[q](out[:, -1]) for q in range(Q)], dim=1)

            repetition_penalty = 1.4
            history_window = 12

            # Get the history of generated tokens for each quantizer
            for q in range(Q):
                # Extract the history window for this quantizer
                history_start = max(0, offset - history_window)
                token_history = sequence[0, q, history_start:offset]

                # Only apply penalty to tokens that appear in the history
                unique_tokens = torch.unique(token_history)
                unique_tokens = unique_tokens[unique_tokens != special_token_id]
                unique_tokens = unique_tokens[unique_tokens != eos]
                unique_tokens = unique_tokens[unique_tokens != unknown_token]

                if len(unique_tokens) > 0:
                    # Apply penalty by dividing the logits for tokens that have appeared recently
                    logits[0, q, unique_tokens] = (logits[0, q, unique_tokens] / repetition_penalty)

            if offset < 24.53 * 4:
                logits[..., eos] = -float("inf")

            probs = F.softmax(logits / temperature, dim=-1)

            # print(probs.shape)
            if top_p is not None and top_k is not None:
                next_codes = sample_top_p_top_k(probs, top_p, top_k)
            elif top_p is not None and top_p > 0:
                next_codes = sample_top_p(probs, top_p)
            elif top_k is not None and top_k > 0:
                next_codes = sample_top_k(probs, top_k)
            else:
                next_codes = multinomial(probs, num_samples=1)

            next_codes = next_codes.repeat(batch_size, 1, 1)

            if (probs[..., eos] > 0.95).any():
                print("breaking at", offset)
                break

            valid_mask = mask[..., offset:offset + 1].expand(B, -1, -1)
            next_codes[~valid_mask] = special_token_id

            sequence[..., offset:offset + 1] = torch.where(
                sequence[..., offset:offset + 1] == unknown_token,
                next_codes,
                sequence[..., offset:offset + 1],
            )

            prev_offset = offset

        # print(sequence.shape)
        out_codes, out_indexes, out_mask = pattern.revert_pattern_sequence(
            sequence, special_token=unknown_token)

        # sanity checks over the returned codes and corresponding masks
        # assert (out_codes[..., :max_gen_len] != unknown_token).all()
        # assert (out_mask[..., :max_gen_len] == 1).all()
        out_codes = out_codes[..., prompt_codes.shape[-1]:offset]
        return out_codes[[0]]


@torch.inference_mode()
def render(
    self: Vui,
    text: str,
    prompt_codes: Tensor | None = None,
    temperature: float = 0.5,
    top_k: int | None = 100,
    top_p: float | None = None,
    max_secs: int = 100,
):
    """
    Render audio from text.

    Uses generate for text < 1000 characters, otherwise breaks text into sections and uses chunking with
    context.
    """
    text = remove_all_invalid_non_speech(text)
    text = simple_clean(text)
    SR = self.codec.config.sample_rate
    HZ = self.codec.hz
    max_gen_len = int(HZ * max_secs)

    if len(text) < 1000:
        codes = generate(self, text, prompt_codes, temperature, top_k, top_p, max_gen_len)
        codes = codes[..., :-10]
        audio = self.codec.from_indices(codes)
        paudio = torchaudio.functional.resample(audio[0], 22050, 16000)
        results = vad(paudio)

        if len(results):
            # Cut the audio based on VAD results, add 200ms silence at end
            s, e = results[0][0], results[-1][1]
            return audio[..., int(s * SR):int((e + 0.2) * SR)].cpu()

        raise Exception("Failed to render")

    # Otherwise we have to do some clever chaining!

    orig_codes = prompt_codes

    lines = text.split("\n")
    audios = []
    prev_codes = prompt_codes
    prev_text = ""

    for i, line in enumerate(lines):
        run = True
        while run:
            current_text = prev_text + "\n" + line if prev_text else line
            current_text = current_text.strip()
            current_text = current_text.replace("...", "")
            current_text = current_text + " [pause]"

            # Calculate max length based on text length
            maxlen = int(HZ * int(60 * len(current_text) / 500))

            try:
                print("rendering", current_text)
                with (
                        torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH),
                        torch.autocast("cuda", dtype=torch.bfloat16, enabled=True),
                ):
                    codes = generate(
                        self,
                        current_text,
                        prompt_codes=prev_codes,
                        temperature=temperature,
                        top_k=top_k,
                        top_p=top_p,
                        max_gen_len=maxlen,
                    )

                codes = codes[..., :-10]
                audio = self.codec.from_indices(codes)
                # Resample for VAD
                paudio = torchaudio.functional.resample(audio[0], 22050, 16000)

                results = vad(paudio)
                run = len(results) == 0

                if len(results):
                    prev_text = line
                    # Cut the audio based on VAD results, add 200ms silence at end
                    s, e = results[0][0], results[0][1]
                    codes = codes[..., int(s * HZ):int(e * HZ)]
                    prev_codes = codes
                    audio = audio[..., int(s * SR):int((e + 0.2) * SR)].cpu()
                    audios.append(audio)
                else:
                    prev_codes = orig_codes
                    prev_text = ""
            except KeyboardInterrupt:
                break
            except RuntimeError as e:
                prev_codes = orig_codes
                prev_text = ""
                print(e)

    return torch.cat(audios, dim=-1)

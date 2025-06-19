import soundfile as sf
import torch
from snac import SNAC
from transformers import AutoModelForCausalLM, AutoTokenizer


class OrpheusTTS:
    """
    A neural speech synthesis model that converts text to speech using SNAC audio codec.

    This class integrates a causal language model with SNAC (Speech Neural Audio Codec)
    to generate natural-sounding speech with paralinguistic features like laughter,
    sighing, and other vocal expressions.

    Example:
        >>> generator = OrpheusTTS()
        >>> generator("Hello, I'm excited to meet you!", voice="Sarah", output_prefix="greeting")
        # Creates greeting.wav file

    Attributes:
        device (str): Computation device ('cuda' or 'cpu')
        model: The causal language model for text-to-audio-token generation
        tokenizer: Tokenizer for text preprocessing
        snac_model: SNAC model for audio token decoding
    """

    def __init__(self, model_path: str = "canopylabs/orpheus-3b-0.1-ft", device: str = "cuda"):
        """
        Initialize the speech generator with specified model and device.

        Args:
            model_path: HuggingFace model path for the language model
            device: Computing device ('cuda' for GPU, 'cpu' for CPU)
        """
        self.device = device
        self._load_models(model_path)

    def _load_models(self, model_path: str):
        """Load SNAC and language models."""
        # Load SNAC audio codec model for decoding audio tokens
        self.snac_model = SNAC.from_pretrained("hubertsiuzdak/snac_24khz").to("cpu")

        # Load the main language model for text-to-audio-token generation
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=torch.bfloat16).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)

    def _prepare_inputs(self, prompt: str, voice: str):
        """
        Tokenize and format inputs for generation.

        Args:
            prompts: List of text strings to convert to speech
            voice: Voice identifier to prepend to each prompt

        Returns:
            Tuple of (input_ids, attention_mask) tensors ready for model generation
        """
        # Add voice prefix to prompt
        prompt_with_voice = f"{voice}: {prompt}"

        # Tokenize the prompt
        input_ids = self.tokenizer(prompt_with_voice, return_tensors="pt").input_ids

        # Add special tokens: Start of Human (SOH) and End of Text/Human (EOT/EOH)
        start_token = torch.tensor([[128259]], dtype=torch.int64)  # SOH
        end_tokens = torch.tensor([[128009, 128260]], dtype=torch.int64)  # EOT, EOH

        formatted_input = torch.cat([start_token, input_ids, end_tokens], dim=1)
        attention_mask = torch.ones_like(formatted_input)

        return formatted_input.to(self.device), attention_mask.to(self.device)

    def _redistribute_codes(self, codes: list):
        """
        Convert token codes to SNAC format and decode to audio.

        SNAC uses a hierarchical codec with 3 layers. This method redistributes
        the flat token sequence into the proper layer structure.

        Args:
            codes: List of integer codes from the language model

        Returns:
            Decoded audio tensor
        """
        # SNAC processes audio in groups of 7 tokens
        n_groups = (len(codes) + 1) // 7

        layer_1, layer_2, layer_3 = [], [], []

        for i in range(n_groups):
            base_idx = 7 * i
            # Layer 1: Coarse audio features (every 7th token)
            layer_1.append(codes[base_idx])
            # Layer 2: Medium resolution features
            layer_2.extend([codes[base_idx + 1] - 4096, codes[base_idx + 4] - 4 * 4096])
            # Layer 3: Fine-grained audio details
            layer_3.extend([
                codes[base_idx + 2] - 2 * 4096,
                codes[base_idx + 3] - 3 * 4096,
                codes[base_idx + 5] - 5 * 4096,
                codes[base_idx + 6] - 6 * 4096,
            ])

        snac_codes = [
            torch.tensor(layer_1).unsqueeze(0),
            torch.tensor(layer_2).unsqueeze(0),
            torch.tensor(layer_3).unsqueeze(0),
        ]

        return self.snac_model.decode(snac_codes)

    def _postprocess_tokens(self, generated_ids: torch.Tensor) -> list:
        """
        Extract and clean generated audio tokens.

        Args:
            generated_ids: Raw token output from language model

        Returns:
            List of cleaned token sequences, one per input prompt
        """
        # Find last occurrence of start token (128257) to locate audio tokens
        start_token, end_token = 128257, 128258
        start_positions = (generated_ids == start_token).nonzero(as_tuple=True)

        if len(start_positions[1]) > 0:
            last_start = start_positions[1][-1].item()
            tokens = generated_ids[:, last_start + 1:]
        else:
            tokens = generated_ids

        # Process the sequence
        row = tokens[0]  # Only process the first (and only) row

        # Remove end tokens to get clean audio token sequence
        clean_tokens = row[row != end_token]

        # Trim to multiple of 7 (SNAC requirement) and adjust token values
        trim_len = (clean_tokens.size(0) // 7) * 7
        trimmed = clean_tokens[:trim_len]
        # Adjust token IDs to SNAC's expected range
        adjusted = [t.item() - 128266 for t in trimmed]

        return adjusted

    def __call__(self, prompt: str, voice: str, output_file: str = "output.wav"):
        """
        Generate speech from text prompts.

        Args:
            prompt: Text string to convert to speech
            voice: Voice identifier (e.g., "Sarah", "John")
            output_prefix: Name for output WAV file (default: "sample")

        Example:
            >>> generator = OrpheusTTS()
            >>> generator("Hello world!", "Emma", "greeting")
            # Creates greeting.wav
        """
        input_ids, attention_mask = self._prepare_inputs(prompt, voice)

        # Generate audio tokens using the language model
        with torch.no_grad():
            generated_ids = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=1200,
                do_sample=True,
                temperature=0.6,
                top_p=0.95,
                repetition_penalty=1.1,
                num_return_sequences=1,
                eos_token_id=128258,
            )

        codes = self._postprocess_tokens(generated_ids)

        # Generate and save audio file
        audio = self._redistribute_codes(codes)
        # Save as 24kHz WAV file
        sf.write(
            f"{output_file}",
            audio.detach().squeeze().cpu().numpy(),
            24000,
        )

        return codes

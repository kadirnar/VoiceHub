from collections.abc import Iterable

import torch


class ByteTokenizer:
    """Simple byte-level tokenizer for text-to-speech."""

    def __init__(self, bos_token: int = 256) -> None:
        if bos_token < 0:
            raise ValueError(f"bos_token must be >= 0, got {bos_token}")
        self.bos_token = int(bos_token)

    @classmethod
    def for_vocab_size(cls, text_vocab_size: int) -> "ByteTokenizer":
        if text_vocab_size < 256:
            raise ValueError(
                f"text_vocab_size must be >= 256 for byte-level tokenization, got {text_vocab_size}"
            )
        # Reserve a dedicated BOS token outside UTF-8 byte range when possible.
        if text_vocab_size == 256:
            return cls(bos_token=0)
        return cls(bos_token=text_vocab_size - 1)

    def encode(self, text: str, add_bos: bool = True) -> torch.Tensor:
        tokens = list(text.encode("utf-8"))
        if add_bos:
            tokens.insert(0, self.bos_token)
        return torch.tensor(tokens, dtype=torch.long)

    def batch_encode(
        self,
        texts: Iterable[str],
        max_length: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        encoded = [self.encode(t) for t in texts]
        if max_length is None:
            max_length = max(x.numel() for x in encoded)

        batch = torch.zeros((len(encoded), max_length), dtype=torch.long)
        mask = torch.zeros((len(encoded), max_length), dtype=torch.bool)
        for i, seq in enumerate(encoded):
            n = min(max_length, seq.numel())
            batch[i, :n] = seq[:n]
            mask[i, :n] = True
        return batch, mask


class PretrainedTextTokenizer:
    """
    Hugging Face tokenizer wrapper for text conditioning.
    - right-padding for stable positional behavior
    - optional explicit BOS prepend
    """

    def __init__(self, tokenizer, add_bos: bool = True) -> None:
        self.tokenizer = tokenizer
        self.add_bos = bool(add_bos)
        # TTS collator uses fixed-length right-padding; enforce this regardless of pretrained defaults.
        self.tokenizer.padding_side = "right"

        if self.tokenizer.pad_token_id is None:
            if self.tokenizer.eos_token_id is not None and self.tokenizer.eos_token is not None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            else:
                raise ValueError(
                    "Tokenizer has no pad_token_id (and no eos_token fallback). "
                    "Set a pad token before training/inference."
                )

        if self.add_bos and self.tokenizer.bos_token_id is None:
            raise ValueError("Tokenizer has no bos_token_id but add_bos=True.")

    @classmethod
    def from_pretrained(
        cls,
        repo_id: str,
        add_bos: bool = True,
        local_files_only: bool = False,
    ) -> "PretrainedTextTokenizer":
        try:
            from transformers import AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "transformers is required for pretrained text tokenization. "
                "Install with `pip install transformers sentencepiece`."
            ) from exc

        tokenizer = AutoTokenizer.from_pretrained(
            repo_id,
            use_fast=True,
            trust_remote_code=False,
            local_files_only=local_files_only,
        )
        return cls(tokenizer=tokenizer, add_bos=add_bos)

    @property
    def vocab_size(self) -> int:
        return int(len(self.tokenizer))

    @property
    def bos_token_id(self) -> int | None:
        return self.tokenizer.bos_token_id

    @property
    def pad_token_id(self) -> int:
        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            raise RuntimeError("pad_token_id is unexpectedly None.")
        return int(pad_id)

    def encode(self, text: str, add_bos: bool | None = None) -> torch.Tensor:
        token_ids = self.tokenizer.encode(text, add_special_tokens=False)
        use_bos = self.add_bos if add_bos is None else bool(add_bos)
        if use_bos:
            bos_id = self.bos_token_id
            if bos_id is None:
                raise ValueError("Tokenizer has no bos_token_id but BOS prepend was requested.")
            token_ids.insert(0, int(bos_id))
        return torch.tensor(token_ids, dtype=torch.long)

    def batch_encode(
        self,
        texts: Iterable[str],
        max_length: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        encoded = [self.encode(t) for t in texts]
        if max_length is None:
            max_length = max(max(x.numel(), 1) for x in encoded)
        if max_length <= 0:
            raise ValueError(f"max_length must be > 0, got {max_length}")

        batch = torch.full(
            (len(encoded), max_length),
            fill_value=self.pad_token_id,
            dtype=torch.long,
        )
        mask = torch.zeros((len(encoded), max_length), dtype=torch.bool)
        for i, seq in enumerate(encoded):
            n = min(max_length, seq.numel())
            if n > 0:
                batch[i, :n] = seq[:n]
                mask[i, :n] = True
        return batch, mask

import torch
from transformers import ByT5Tokenizer


class CustomByT5Tokenizer(ByT5Tokenizer):
    """ByT5 tokenizer that returns a ``torch.Tensor`` instead of a plain list."""

    def encode(self, text, add_special_tokens=False, **kwargs):
        """
        Override the encode method.

        Args:
            text (str): Input text
            add_special_tokens (bool): Whether to add BOS/EOS tokens
        """
        # Use the parent class's encode method
        tokens = super().encode(text, add_special_tokens=add_special_tokens, **kwargs)
        return torch.tensor(tokens)

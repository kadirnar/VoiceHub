from dataclasses import dataclass, asdict, field
from typing import Dict

@dataclass
class SpecialTokens:
    """
    Dataclass containing special tokens used for text and audio processing.
    """
    bos: str = "<|im_start|>"
    eos: str = "<|im_end|>"
    audio_code: str = "<|{}|>"
    text_start: str = "<|text_start|>"
    text_end: str = "<|text_end|>"
    voice_characteristic_start: str = "<|voice_characteristic_start|>"
    voice_characteristic_end: str = "<|voice_characteristic_end|>"
    emotion_start: str = "<|emotion_start|>"
    emotion_end: str = "<|emotion_end|>"
    audio_start: str = "<|audio_start|>"
    audio_end: str = "<|audio_end|>"
    time: str = "<|t_{:.2f}|>"
    text_sep: str = "<|text_sep|>"
    space: str = "<|space|>"
    syllable_sep: str = "<|syllable_sep|>"

    punctuation_tokens: Dict[str, str] = field(default_factory=lambda: {
        # Basic English/Latin punctuation
        ".": "<|period|>",
        "!": "<|exclamation_mark|>",
        "?": "<|question_mark|>",
        ",": "<|comma|>",
        
        # Quotation marks across languages
        '"': "<|double_quote|>",
        "„": "<|low_double_quote|>",
        
        # Spanish/Italian specific
        "¡": "<|inverted_exclamation|>",
        "¿": "<|inverted_question|>",
        
        # Ellipsis
        "…": "<|ellipsis|>",
        "...": "<|ellipsis|>",
        
        # Chinese/Japanese sentence endings
        "。": "<|cjk_period|>",
        "！": "<|cjk_exclamation|>",
        "？": "<|cjk_question|>",
        "，": "<|cjk_comma|>",
        
        # Arabic sentence punctuation
        "؟": "<|arabic_question|>",
    })

    def to_dict(self) -> Dict[str, str]:
        """Convert the dataclass instance to a dictionary using asdict."""
        return asdict(self)
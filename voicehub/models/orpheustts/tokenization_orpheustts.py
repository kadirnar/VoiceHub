"""Native Llama-3 byte-BPE tokenization used by Orpheus checkpoints."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

from voicehub.tokenization import ByteBPETokenizer, Encoding, TokenizerAssetError, read_bounded_asset
from voicehub.tokenization.llama3 import LLAMA3_SPLIT_PATTERN, llama3_pretokenize

_BOS_TOKEN = "<|begin_of_text|>"
_PAD_TOKEN = "<|finetune_right_pad_id|>"


def _read_tokenizer_document(path: Path) -> dict[str, Any]:
    payload = read_bounded_asset(path)
    try:
        document = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise TokenizerAssetError(f"Invalid Orpheus tokenizer JSON: {error}.") from error
    if not isinstance(document, dict):
        raise TokenizerAssetError("Orpheus tokenizer JSON root must be an object.")
    return document


def _validate_llama3_pipeline(document: dict[str, Any]) -> None:
    if document.get("normalizer") is not None:
        raise TokenizerAssetError("Orpheus requires the unnormalized Llama-3 text pipeline.")
    pre_tokenizer = document.get("pre_tokenizer")
    if not isinstance(pre_tokenizer, dict) or pre_tokenizer.get("type") != "Sequence":
        raise TokenizerAssetError("Orpheus requires the official Llama-3 Sequence pre-tokenizer.")
    children = pre_tokenizer.get("pretokenizers")
    if not isinstance(children, list) or len(children) != 2:
        raise TokenizerAssetError("Orpheus requires a Split followed by ByteLevel pre-tokenization.")
    split, byte_level = children
    pattern = split.get("pattern") if isinstance(split, dict) else None
    regex = pattern.get("Regex") if isinstance(pattern, dict) else None
    if (not isinstance(split, dict) or split.get("type") != "Split" or regex != LLAMA3_SPLIT_PATTERN or
            split.get("behavior") != "Isolated" or split.get("invert") not in (None, False)):
        raise TokenizerAssetError("Orpheus tokenizer uses an unsupported Split expression or behavior.")
    if (not isinstance(byte_level, dict) or byte_level.get("type") != "ByteLevel" or
            byte_level.get("add_prefix_space", False) or byte_level.get("use_regex", True)):
        raise TokenizerAssetError("Orpheus requires ByteLevel(add_prefix_space=False, use_regex=False).")
    post_processor = document.get("post_processor")
    processors = (
        post_processor.get("processors")
        if isinstance(post_processor, dict) and post_processor.get("type") == "Sequence" else None)
    if not isinstance(processors, list) or len(processors) != 2:
        raise TokenizerAssetError("Orpheus requires the official ByteLevel/BOS post-processor.")
    post_byte_level, template = processors
    if (not isinstance(post_byte_level, dict) or post_byte_level.get("type") != "ByteLevel" or
            post_byte_level.get("add_prefix_space") is not True or
            post_byte_level.get("trim_offsets") is not False or post_byte_level.get("use_regex") is not True):
        raise TokenizerAssetError("Orpheus uses an unsupported ByteLevel post-processor.")
    expected_single = [
        {
            "SpecialToken": {
                "id": _BOS_TOKEN,
                "type_id": 0,
            },
        },
        {
            "Sequence": {
                "id": "A",
                "type_id": 0,
            },
        },
    ]
    if (not isinstance(template, dict) or template.get("type") != "TemplateProcessing" or
            template.get("single") != expected_single):
        raise TokenizerAssetError("Orpheus requires a single-sequence template that prepends BOS.")
    decoder = document.get("decoder")
    if (not isinstance(decoder, dict) or decoder.get("type") != "ByteLevel" or
            decoder.get("add_prefix_space") is not True or decoder.get("trim_offsets") is not True or
            decoder.get("use_regex") is not True):
        raise TokenizerAssetError("Orpheus requires the official Llama-3 ByteLevel decoder.")


class OrpheusTokenizer:
    """Exact Orpheus prompt tokenizer backed by VoiceHub byte-BPE."""

    BOS_TOKEN = _BOS_TOKEN
    PAD_TOKEN = _PAD_TOKEN

    def __init__(
        self,
        tokenizer: ByteBPETokenizer,
        *,
        bos_token_id: int,
        pad_token_id: int,
        tokenizer_path: Path,
        tokenizer_config_path: Path | None = None,
    ) -> None:
        self._tokenizer = tokenizer
        self.bos_token_id = int(bos_token_id)
        self.pad_token_id = int(pad_token_id)
        self.tokenizer_path = tokenizer_path
        self.tokenizer_config_path = tokenizer_config_path

    @classmethod
    def from_tokenizer_json(
        cls,
        path: str | Path,
        *,
        tokenizer_config_path: str | Path | None = None,
    ) -> OrpheusTokenizer:
        tokenizer_path = Path(path).expanduser().resolve()
        document = _read_tokenizer_document(tokenizer_path)
        _validate_llama3_pipeline(document)
        records = document.get("added_tokens")
        if not isinstance(records, list):
            raise TokenizerAssetError("Orpheus tokenizer has no added-token table.")
        token_ids = {
            record.get("content"): record.get("id")
            for record in records if isinstance(record, dict)
        }
        try:
            bos_token_id = token_ids[cls.BOS_TOKEN]
            pad_token_id = token_ids[cls.PAD_TOKEN]
        except KeyError as error:
            raise TokenizerAssetError(
                f"Orpheus tokenizer is missing required token {error.args[0]!r}.") from error
        if (isinstance(bos_token_id, bool) or not isinstance(bos_token_id, int) or
                isinstance(pad_token_id, bool) or not isinstance(pad_token_id, int)):
            raise TokenizerAssetError("Orpheus BOS and PAD IDs must be integers.")

        tokenizer = ByteBPETokenizer.from_tokenizer_json(
            tokenizer_path,
            pad_token_id=pad_token_id,
            use_regex=False,
            pretokenizer=llama3_pretokenize,
        )
        config_path = (
            None if tokenizer_config_path is None else Path(tokenizer_config_path).expanduser().resolve())
        return cls(
            tokenizer,
            bos_token_id=bos_token_id,
            pad_token_id=pad_token_id,
            tokenizer_path=tokenizer_path,
            tokenizer_config_path=config_path,
        )

    @property
    def vocabulary_size(self) -> int:
        return self._tokenizer.vocabulary_size

    @property
    def token_id_space_size(self) -> int:
        """Exclusive upper bound required by the causal-LM embedding."""
        return self._tokenizer.token_id_space_size

    def __len__(self) -> int:
        return self.vocabulary_size

    def encode(
        self,
        text: str,
        *,
        add_special_tokens: bool = True,
    ) -> Encoding:
        if not isinstance(add_special_tokens, bool):
            raise TypeError("`add_special_tokens` must be a boolean.")
        encoded = self._tokenizer.encode(text)
        if not add_special_tokens:
            return encoded
        return Encoding(
            input_ids=(self.bos_token_id, *encoded.input_ids),
            attention_mask=(1, *encoded.attention_mask),
            special_tokens_mask=(1, *encoded.special_tokens_mask),
        )

    def decode(
        self,
        token_ids,
        *,
        skip_special_tokens: bool = False,
    ) -> str:
        return self._tokenizer.decode(
            token_ids,
            skip_special_tokens=skip_special_tokens,
        )

    def save_pretrained(self, directory: str | Path) -> Path:
        destination = Path(directory).expanduser()
        destination.mkdir(parents=True, exist_ok=True)
        tokenizer_destination = (destination / "tokenizer.json").resolve()
        if tokenizer_destination != self.tokenizer_path.resolve():
            shutil.copy2(self.tokenizer_path, tokenizer_destination)
        if self.tokenizer_config_path is not None:
            config_destination = (destination / "tokenizer_config.json").resolve()
            if config_destination != self.tokenizer_config_path.resolve():
                shutil.copy2(
                    self.tokenizer_config_path,
                    config_destination,
                )
        return destination


__all__ = [
    "LLAMA3_SPLIT_PATTERN",
    "OrpheusTokenizer",
    "llama3_pretokenize",
]

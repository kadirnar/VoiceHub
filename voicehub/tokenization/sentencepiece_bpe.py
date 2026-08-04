"""Deterministic SentencePiece-style BPE for declarative tokenizer JSON.

This implementation covers the tokenizer graph used by published
Moonshine checkpoints: prepend/replace whitespace normalization, ordered
BPE merges, UTF-8 byte fallback, template-added BOS, and the standard
decoder chain.  It does not import Hugging Face Tokenizers or
SentencePiece.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

from voicehub.json_utils import parse_json_value
from voicehub.tokenization.assets import (
    DEFAULT_MAX_ASSET_BYTES,
    DEFAULT_MAX_JSON_DEPTH,
    DEFAULT_MAX_JSON_NODES,
    DEFAULT_MAX_MERGES,
    DEFAULT_MAX_TOKEN_BYTES,
    DEFAULT_MAX_TOKENS,
    TokenizerAssetError,
    read_bounded_asset,
)
from voicehub.tokenization.base import BatchEncoding, Encoding

_MAX_TOKEN_ID = 2**31 - 1


def _validate_json_bounds(
    value: Any,
    *,
    max_depth: int,
    max_nodes: int,
) -> None:
    stack: list[tuple[Any, int]] = [(value, 1)]
    nodes = 0
    while stack:
        item, depth = stack.pop()
        nodes += 1
        if nodes > max_nodes:
            raise TokenizerAssetError(f"Tokenizer JSON contains more than {max_nodes} values.")
        if depth > max_depth:
            raise TokenizerAssetError("Tokenizer JSON nesting exceeds the configured depth "
                                      f"{max_depth}.")
        if isinstance(item, dict):
            stack.extend((child, depth + 1) for child in item.values())
        elif isinstance(item, list):
            stack.extend((child, depth + 1) for child in item)


def _token_id(value: Any, *, context: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TokenizerAssetError(f"{context} must be an integer.")
    if not 0 <= value <= _MAX_TOKEN_ID:
        raise TokenizerAssetError(f"{context} must be between zero and {_MAX_TOKEN_ID}.")
    return value


def _parse_merge(value: Any, *, index: int) -> tuple[str, str]:
    if isinstance(value, str):
        parts = value.split(" ")
    elif isinstance(value, list):
        parts = value
    else:
        raise TokenizerAssetError(f"BPE merge {index} must be a string or pair.")
    if len(parts) != 2 or any(not isinstance(part, str) or not part for part in parts):
        raise TokenizerAssetError(f"BPE merge {index} must contain two non-empty strings.")
    return parts[0], parts[1]


@dataclass(frozen=True, slots=True)
class SentencePieceBPEAssets:
    vocabulary: Mapping[str, int]
    merges: tuple[tuple[str, str], ...]
    special_tokens: Mapping[str, int]
    added_tokens: Mapping[str, int]
    unk_token_id: int
    prefix_token_ids: tuple[int, ...]
    prepend: str
    replacement_source: str
    replacement_target: str
    byte_fallback: bool
    fuse_unk: bool
    original_document: Mapping[str, Any]


def load_sentencepiece_bpe(
    path: str | Path,
    *,
    max_bytes: int = DEFAULT_MAX_ASSET_BYTES,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    max_merges: int = DEFAULT_MAX_MERGES,
    max_token_bytes: int = DEFAULT_MAX_TOKEN_BYTES,
    max_json_depth: int = DEFAULT_MAX_JSON_DEPTH,
    max_json_nodes: int = DEFAULT_MAX_JSON_NODES,
) -> SentencePieceBPEAssets:
    """Load and strictly validate Moonshine's tokenizer JSON graph."""
    for name, value in (
        ("max_tokens", max_tokens),
        ("max_merges", max_merges),
        ("max_token_bytes", max_token_bytes),
        ("max_json_depth", max_json_depth),
        ("max_json_nodes", max_json_nodes),
    ):
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"`{name}` must be an integer.")
        if value <= 0:
            raise ValueError(f"`{name}` must be greater than zero.")
    payload = read_bounded_asset(path, max_bytes=max_bytes)
    try:
        document = parse_json_value(payload, source=Path(path).expanduser())
    except (ValueError, RecursionError) as error:
        raise TokenizerAssetError(f"Invalid tokenizer JSON: {error}.") from error
    _validate_json_bounds(
        document,
        max_depth=max_json_depth,
        max_nodes=max_json_nodes,
    )
    if not isinstance(document, dict):
        raise TokenizerAssetError("Tokenizer JSON root must be an object.")
    if document.get("pre_tokenizer") is not None:
        raise TokenizerAssetError("SentencePiece BPE currently requires `pre_tokenizer=null`.")
    if document.get("truncation") is not None or document.get("padding") is not None:
        raise TokenizerAssetError("Tokenizer-level truncation and padding policies are unsupported.")

    model = document.get("model")
    if not isinstance(model, dict) or model.get("type") != "BPE":
        raise TokenizerAssetError("Tokenizer JSON must declare a BPE model.")
    for name in (
            "continuing_subword_prefix",
            "end_of_word_suffix",
            "dropout",
    ):
        if model.get(name) not in (None, ""):
            raise TokenizerAssetError(f"SentencePiece BPE option `{name}` is unsupported.")
    if model.get("ignore_merges") not in (None, False):
        raise TokenizerAssetError("BPE `ignore_merges=True` is unsupported.")
    fuse_unk = model.get("fuse_unk", False)
    if not isinstance(fuse_unk, bool):
        raise TokenizerAssetError("BPE `fuse_unk` must be a boolean.")
    byte_fallback = model.get("byte_fallback")
    if byte_fallback is not True:
        raise TokenizerAssetError("Moonshine SentencePiece BPE requires `byte_fallback=true`.")

    raw_vocabulary = model.get("vocab")
    if not isinstance(raw_vocabulary, dict) or not raw_vocabulary:
        raise TokenizerAssetError("BPE vocabulary must be a non-empty object.")
    if len(raw_vocabulary) > max_tokens:
        raise TokenizerAssetError(f"Tokenizer vocabulary contains more than {max_tokens} tokens.")
    vocabulary: dict[str, int] = {}
    seen_ids: dict[int, str] = {}
    for token, raw_id in raw_vocabulary.items():
        if not isinstance(token, str):
            raise TokenizerAssetError("BPE vocabulary tokens must be strings.")
        if len(token.encode("utf-8")) > max_token_bytes:
            raise TokenizerAssetError(f"Vocabulary token exceeds {max_token_bytes} UTF-8 bytes.")
        token_id = _token_id(raw_id, context="Vocabulary token ID")
        previous = seen_ids.get(token_id)
        if previous is not None:
            raise TokenizerAssetError(
                f"Vocabulary tokens {previous!r} and {token!r} share ID "
                f"{token_id}.")
        vocabulary[token] = token_id
        seen_ids[token_id] = token

    raw_merges = model.get("merges")
    if not isinstance(raw_merges, list):
        raise TokenizerAssetError("BPE `merges` must be an array.")
    if len(raw_merges) > max_merges:
        raise TokenizerAssetError(f"Tokenizer contains more than {max_merges} BPE merges.")
    merges: list[tuple[str, str]] = []
    seen_merges: set[tuple[str, str]] = set()
    for index, raw_merge in enumerate(raw_merges):
        merge = _parse_merge(raw_merge, index=index)
        if merge in seen_merges:
            raise TokenizerAssetError(f"Duplicate BPE merge at index {index}.")
        if merge[0] + merge[1] not in vocabulary:
            raise TokenizerAssetError(f"BPE merge {index} produces a token absent from the vocabulary.")
        merges.append(merge)
        seen_merges.add(merge)

    raw_added = document.get("added_tokens")
    if not isinstance(raw_added, list):
        raise TokenizerAssetError("Tokenizer `added_tokens` must be an array.")
    if len(raw_added) > max_tokens:
        raise TokenizerAssetError(f"Tokenizer contains more than {max_tokens} added tokens.")
    special_tokens: dict[str, int] = {}
    added_tokens: dict[str, int] = {}
    for index, record in enumerate(raw_added):
        if not isinstance(record, dict):
            raise TokenizerAssetError(f"Added token {index} must be an object.")
        content = record.get("content")
        if not isinstance(content, str) or not content:
            raise TokenizerAssetError(f"Added token {index} must have non-empty string content.")
        if record.get("lstrip") or record.get("rstrip"):
            raise TokenizerAssetError("Whitespace-stripping added tokens are unsupported.")
        if record.get("single_word"):
            raise TokenizerAssetError("Single-word added tokens are unsupported.")
        if record.get("normalized") not in (None, False):
            raise TokenizerAssetError("Normalized added tokens are unsupported.")
        token_id = _token_id(
            record.get("id"),
            context=f"Added token {index} ID",
        )
        previous = seen_ids.get(token_id)
        if previous is not None and previous != content:
            raise TokenizerAssetError(
                f"Added token {content!r} conflicts with ID {token_id} "
                f"assigned to {previous!r}.")
        seen_ids[token_id] = content
        destination = (special_tokens if record.get("special") is True else added_tokens)
        if content in destination and destination[content] != token_id:
            raise TokenizerAssetError(f"Added token {content!r} has conflicting IDs.")
        destination[content] = token_id

    unk_token = model.get("unk_token")
    if not isinstance(unk_token, str) or unk_token not in vocabulary:
        raise TokenizerAssetError("BPE `unk_token` must name a vocabulary token.")
    unk_token_id = vocabulary[unk_token]

    normalizer = document.get("normalizer")
    if (not isinstance(normalizer, dict) or normalizer.get("type") != "Sequence" or
            not isinstance(normalizer.get("normalizers"), list) or len(normalizer["normalizers"]) != 2):
        raise TokenizerAssetError("Moonshine tokenizer requires its two-stage whitespace normalizer.")
    prepend_record, replace_record = normalizer["normalizers"]
    if (not isinstance(prepend_record, dict) or prepend_record.get("type") != "Prepend" or
            not isinstance(prepend_record.get("prepend"), str)):
        raise TokenizerAssetError("Invalid tokenizer Prepend normalizer.")
    pattern = (
        replace_record.get("pattern")
        if isinstance(replace_record, dict) and replace_record.get("type") == "Replace" else None)
    if (not isinstance(pattern, dict) or set(pattern) != {"String"} or
            not isinstance(pattern["String"], str) or not isinstance(replace_record.get("content"), str)):
        raise TokenizerAssetError("Invalid tokenizer Replace normalizer.")

    post_processor = document.get("post_processor")
    if (not isinstance(post_processor, dict) or post_processor.get("type") != "TemplateProcessing"):
        raise TokenizerAssetError("Moonshine tokenizer requires TemplateProcessing.")
    single = post_processor.get("single")
    if not isinstance(single, list) or len(single) != 2:
        raise TokenizerAssetError("Moonshine tokenizer requires a BOS-plus-sequence template.")
    first, second = single
    if (not isinstance(first, dict) or not isinstance(first.get("SpecialToken"), dict) or
            not isinstance(second, dict) or not isinstance(second.get("Sequence"), dict) or
            second["Sequence"].get("id") != "A"):
        raise TokenizerAssetError("Unsupported tokenizer single-sequence template.")
    special_name = first["SpecialToken"].get("id")
    template_specials = post_processor.get("special_tokens")
    if (not isinstance(special_name, str) or not isinstance(template_specials, dict) or
            not isinstance(template_specials.get(special_name), dict)):
        raise TokenizerAssetError("Tokenizer template BOS token is invalid.")
    prefix_ids = template_specials[special_name].get("ids")
    if (not isinstance(prefix_ids, list) or not prefix_ids or
            any(isinstance(value, bool) or not isinstance(value, int) for value in prefix_ids)):
        raise TokenizerAssetError("Tokenizer template BOS IDs must be a non-empty integer array.")
    for token_id in prefix_ids:
        if token_id not in seen_ids:
            raise TokenizerAssetError(f"Tokenizer template refers to unknown ID {token_id}.")

    decoder = document.get("decoder")
    expected_decoder_types = ("Replace", "ByteFallback", "Fuse", "Strip")
    if (not isinstance(decoder, dict) or decoder.get("type") != "Sequence" or
            not isinstance(decoder.get("decoders"), list) or
            tuple(item.get("type") if isinstance(item, dict) else None
                  for item in decoder["decoders"]) != expected_decoder_types):
        raise TokenizerAssetError("Unsupported SentencePiece BPE decoder chain.")
    decoder_replace = decoder["decoders"][0]
    decoder_pattern = decoder_replace.get("pattern")
    decoder_strip = decoder["decoders"][3]
    if (not isinstance(decoder_pattern, dict) or decoder_pattern.get("String") != replace_record["content"] or
            decoder_replace.get("content") != replace_record["pattern"]["String"] or
            decoder_strip.get("content") != replace_record["pattern"]["String"] or
            decoder_strip.get("start") != 1 or decoder_strip.get("stop") != 0):
        raise TokenizerAssetError("SentencePiece BPE decoder does not invert its whitespace marker.")

    return SentencePieceBPEAssets(
        vocabulary=MappingProxyType(vocabulary),
        merges=tuple(merges),
        special_tokens=MappingProxyType(special_tokens),
        added_tokens=MappingProxyType(added_tokens),
        unk_token_id=unk_token_id,
        prefix_token_ids=tuple(prefix_ids),
        prepend=prepend_record["prepend"],
        replacement_source=replace_record["pattern"]["String"],
        replacement_target=replace_record["content"],
        byte_fallback=byte_fallback,
        fuse_unk=fuse_unk,
        original_document=MappingProxyType(document),
    )


class SentencePieceBPETokenizer:
    """Ordered BPE tokenizer with SentencePiece whitespace and byte
    fallback."""

    def __init__(
        self,
        assets: SentencePieceBPEAssets,
        *,
        pad_token_id: int,
        bos_token_id: int,
        eos_token_id: int,
        max_input_chars: int = 1_000_000,
    ) -> None:
        if not isinstance(assets, SentencePieceBPEAssets):
            raise TypeError("`assets` must be SentencePieceBPEAssets.")
        if (isinstance(max_input_chars, bool) or not isinstance(max_input_chars, int) or
                max_input_chars <= 0):
            raise ValueError("`max_input_chars` must be a positive integer.")
        self._assets = assets
        self._vocabulary = assets.vocabulary
        self._id_to_token = MappingProxyType({
            token_id: token
            for token, token_id in (
                *assets.vocabulary.items(),
                *assets.special_tokens.items(),
                *assets.added_tokens.items(),
            )
        })
        self._merge_ranks = MappingProxyType({pair: rank for rank, pair in enumerate(assets.merges)})
        self._special_ids = frozenset(assets.special_tokens.values())
        self._recognized_added = tuple(
            sorted(
                (
                    *assets.special_tokens,
                    *assets.added_tokens,
                ),
                key=lambda value: (-len(value), value),
            ))
        self.pad_token_id = self._known_id(pad_token_id, "pad_token_id")
        self.bos_token_id = self._known_id(bos_token_id, "bos_token_id")
        self.eos_token_id = self._known_id(eos_token_id, "eos_token_id")
        self.unk_token_id = assets.unk_token_id
        self.max_input_chars = max_input_chars

    @classmethod
    def from_tokenizer_json(
        cls,
        path: str | Path,
        *,
        pad_token_id: int,
        bos_token_id: int,
        eos_token_id: int,
        **limits: Any,
    ) -> SentencePieceBPETokenizer:
        return cls(
            load_sentencepiece_bpe(path, **limits),
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
        )

    def _known_id(self, value: Any, name: str) -> int:
        token_id = _token_id(value, context=name)
        if token_id not in self._id_to_token:
            raise ValueError(f"`{name}` {token_id} is absent from the tokenizer.")
        return token_id

    @property
    def vocabulary_size(self) -> int:
        return len(set(self._id_to_token))

    @property
    def token_id_space_size(self) -> int:
        return max(self._id_to_token) + 1

    @property
    def special_tokens(self) -> Mapping[str, int]:
        return self._assets.special_tokens

    def _initial_tokens(self, text: str) -> list[str]:
        normalized = (self._assets.prepend + text).replace(
            self._assets.replacement_source,
            self._assets.replacement_target,
        )
        pieces: list[str] = []
        for character in normalized:
            if character in self._vocabulary:
                pieces.append(character)
                continue
            fallback = tuple(f"<0x{byte:02X}>" for byte in character.encode("utf-8"))
            if self._assets.byte_fallback and all(token in self._vocabulary for token in fallback):
                pieces.extend(fallback)
            else:
                unknown = self._id_to_token[self.unk_token_id]
                if not (self._assets.fuse_unk and pieces and pieces[-1] == unknown):
                    pieces.append(unknown)
        return pieces

    def _merge(self, pieces: list[str]) -> tuple[int, ...]:
        while len(pieces) > 1:
            best_rank = None
            best_index = None
            for index in range(len(pieces) - 1):
                rank = self._merge_ranks.get((pieces[index], pieces[index + 1]))
                if rank is not None and (best_rank is None or rank < best_rank):
                    best_rank = rank
                    best_index = index
            if best_index is None:
                break
            pieces[best_index:best_index + 2] = [pieces[best_index] + pieces[best_index + 1]]
        return tuple(self._vocabulary.get(piece, self.unk_token_id) for piece in pieces)

    def _plain_segments(self, text: str) -> tuple[tuple[bool, str], ...]:
        """Split literal added tokens before normalizing ordinary text."""
        if not self._recognized_added:
            return ((False, text), )
        segments: list[tuple[bool, str]] = []
        start = 0
        index = 0
        while index < len(text):
            match = next(
                (token for token in self._recognized_added if text.startswith(token, index)),
                None,
            )
            if match is None:
                index += 1
                continue
            if index > start:
                segments.append((False, text[start:index]))
            segments.append((True, match))
            index += len(match)
            start = index
        if start < len(text) or not segments:
            segments.append((False, text[start:]))
        return tuple(segments)

    def encode(
        self,
        text: str,
        *,
        add_special_tokens: bool = True,
    ) -> Encoding:
        if not isinstance(text, str):
            raise TypeError("`text` must be a string.")
        if len(text) > self.max_input_chars:
            raise ValueError(f"Input contains more than {self.max_input_chars} characters.")
        ids: list[int] = []
        for is_added, segment in self._plain_segments(text):
            if is_added:
                if segment in self._assets.special_tokens:
                    ids.append(self._assets.special_tokens[segment])
                else:
                    ids.append(self._assets.added_tokens[segment])
            elif segment:
                ids.extend(self._merge(self._initial_tokens(segment)))
        if add_special_tokens:
            ids[:0] = self._assets.prefix_token_ids
        return Encoding(
            input_ids=tuple(ids),
            attention_mask=tuple(1 for _ in ids),
        )

    def encode_batch(
        self,
        texts: Sequence[str],
        *,
        add_special_tokens: bool = True,
        pad: bool = False,
    ) -> BatchEncoding:
        if isinstance(texts, (str, bytes)) or not isinstance(texts, Sequence):
            raise TypeError("`texts` must be a sequence of strings.")
        encodings = tuple(self.encode(text, add_special_tokens=add_special_tokens) for text in texts)
        maximum = (max((len(item.input_ids) for item in encodings), default=0) if pad else None)
        input_rows: list[tuple[int, ...]] = []
        attention_rows: list[tuple[int, ...]] = []
        special_rows: list[tuple[int, ...]] = []
        for item in encodings:
            amount = 0 if maximum is None else maximum - len(item.input_ids)
            input_rows.append(item.input_ids + (self.pad_token_id, ) * amount)
            attention_rows.append(item.attention_mask + (0, ) * amount)
            special_rows.append(item.special_tokens_mask + (1, ) * amount)
        return BatchEncoding(
            input_ids=tuple(input_rows),
            attention_mask=tuple(attention_rows),
            special_tokens_mask=tuple(special_rows),
        )

    @staticmethod
    def _byte_value(token: str) -> int | None:
        if len(token) != 6 or not token.startswith("<0x") or token[-1] != ">":
            return None
        try:
            return int(token[3:5], 16)
        except ValueError:
            return None

    def decode(
        self,
        token_ids: Iterable[int],
        *,
        skip_special_tokens: bool = True,
    ) -> str:
        tokens: list[str] = []
        for raw_id in token_ids:
            token_id = _token_id(raw_id, context="Token ID")
            try:
                token = self._id_to_token[token_id]
            except KeyError as error:
                raise ValueError(f"Token ID {token_id} is absent from the tokenizer.") from error
            if skip_special_tokens and token_id in self._special_ids:
                continue
            tokens.append(token.replace(
                self._assets.replacement_target,
                self._assets.replacement_source,
            ))

        fragments: list[str] = []
        buffered_bytes = bytearray()

        def flush_bytes() -> None:
            if buffered_bytes:
                fragments.append(buffered_bytes.decode("utf-8", errors="replace"))
                buffered_bytes.clear()

        for token in tokens:
            byte_value = self._byte_value(token)
            if byte_value is not None:
                buffered_bytes.append(byte_value)
            else:
                flush_bytes()
                fragments.append(token)
        flush_bytes()
        decoded = "".join(fragments)
        if decoded.startswith(self._assets.replacement_source):
            decoded = decoded[len(self._assets.replacement_source):]
        return decoded

    def batch_decode(
        self,
        sequences: Iterable[Iterable[int]],
        *,
        skip_special_tokens: bool = True,
    ) -> list[str]:
        return [self.decode(
            sequence,
            skip_special_tokens=skip_special_tokens,
        ) for sequence in sequences]

    def save_pretrained(self, directory: str | Path) -> Path:
        destination = Path(directory).expanduser()
        destination.mkdir(parents=True, exist_ok=True)
        path = destination / "tokenizer.json"
        path.write_text(
            json.dumps(
                dict(self._assets.original_document),
                ensure_ascii=False,
                separators=(",", ":"),
            ),
            encoding="utf-8",
        )
        return path


__all__ = [
    "SentencePieceBPEAssets",
    "SentencePieceBPETokenizer",
    "load_sentencepiece_bpe",
]

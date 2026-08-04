"""Bounded parsers for byte-BPE tokenizer assets.

The loaders in this module intentionally understand data files rather
than executing tokenizer code from a model repository. This keeps
checkpoint loading deterministic and makes malformed or unexpectedly
large assets fail before they can consume unbounded memory.
"""

from __future__ import annotations

import base64
import binascii
from collections.abc import Mapping
from dataclasses import dataclass, field
from numbers import Integral
from pathlib import Path
from types import MappingProxyType
from typing import Any

from voicehub.json_utils import parse_json_value

DEFAULT_MAX_ASSET_BYTES = 64 * 1024 * 1024
DEFAULT_MAX_TOKENS = 2_000_000
DEFAULT_MAX_MERGES = 2_000_000
DEFAULT_MAX_TOKEN_BYTES = 16 * 1024
DEFAULT_MAX_JSON_DEPTH = 64
DEFAULT_MAX_JSON_NODES = 4_000_000
_MAX_TOKEN_ID = 2**31 - 1
_UNICODE_NORMALIZATIONS = frozenset({"NFC", "NFD", "NFKC", "NFKD"})


class TokenizerAssetError(ValueError):
    """Raised when a tokenizer asset is malformed or outside safe bounds."""


@dataclass(frozen=True, slots=True)
class ByteBPEAssets:
    """Validated inputs needed to construct a byte-level BPE tokenizer."""

    vocabulary: Mapping[bytes, int]
    merges: tuple[tuple[bytes, bytes], ...] = ()
    special_tokens: Mapping[str, int] = field(default_factory=dict)
    added_tokens: Mapping[str, int] = field(default_factory=dict)
    unk_token_id: int | None = None
    add_prefix_space: bool = False
    use_regex: bool = True
    normalization: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "vocabulary",
            MappingProxyType(dict(self.vocabulary)),
        )
        object.__setattr__(self, "merges", tuple(self.merges))
        object.__setattr__(
            self,
            "special_tokens",
            MappingProxyType(dict(self.special_tokens)),
        )
        object.__setattr__(
            self,
            "added_tokens",
            MappingProxyType(dict(self.added_tokens)),
        )


def read_bounded_asset(
    path: str | Path,
    *,
    max_bytes: int = DEFAULT_MAX_ASSET_BYTES,
) -> bytes:
    """Read one regular file while enforcing a strict byte limit."""
    if isinstance(max_bytes, bool) or not isinstance(max_bytes, Integral):
        raise TypeError("`max_bytes` must be an integer.")
    if max_bytes <= 0:
        raise ValueError("`max_bytes` must be greater than zero.")

    asset_path = Path(path).expanduser()
    if not asset_path.is_file():
        raise FileNotFoundError(f"Tokenizer asset was not found: {asset_path}")
    size = asset_path.stat().st_size
    if size > max_bytes:
        raise TokenizerAssetError(f"Tokenizer asset is {size} bytes; the configured limit is {max_bytes}.")
    with asset_path.open("rb") as stream:
        payload = stream.read(max_bytes + 1)
    if len(payload) > max_bytes:
        raise TokenizerAssetError(f"Tokenizer asset exceeds the configured {max_bytes}-byte limit.")
    return payload


def load_tiktoken_ranks(
    path: str | Path,
    *,
    max_bytes: int = DEFAULT_MAX_ASSET_BYTES,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    max_token_bytes: int = DEFAULT_MAX_TOKEN_BYTES,
) -> dict[bytes, int]:
    """Load OpenAI's base64-token/rank text format without ``tiktoken``."""
    _positive_limit(max_tokens, name="max_tokens")
    _positive_limit(max_token_bytes, name="max_token_bytes")
    payload = read_bounded_asset(path, max_bytes=max_bytes)
    ranks: dict[bytes, int] = {}
    seen_ranks: set[int] = set()
    for line_number, raw_line in enumerate(payload.splitlines(), start=1):
        if not raw_line.strip():
            continue
        if len(ranks) >= max_tokens:
            raise TokenizerAssetError(f"TikToken asset contains more than {max_tokens} tokens.")
        fields = raw_line.split()
        if len(fields) != 2:
            raise TokenizerAssetError(f"Invalid TikToken record on line {line_number}: expected two fields.")
        encoded_token, encoded_rank = fields
        try:
            token = base64.b64decode(encoded_token, validate=True)
        except (binascii.Error, ValueError) as error:
            raise TokenizerAssetError(f"Invalid base64 token on line {line_number}.") from error
        if not token:
            raise TokenizerAssetError(f"TikToken record on line {line_number} contains an empty token.")
        if len(token) > max_token_bytes:
            raise TokenizerAssetError(f"Token on line {line_number} exceeds {max_token_bytes} bytes.")
        try:
            rank = int(encoded_rank.decode("ascii"))
        except (UnicodeDecodeError, ValueError) as error:
            raise TokenizerAssetError(f"Invalid token rank on line {line_number}.") from error
        _validate_token_id(rank, context=f"rank on line {line_number}")
        if token in ranks:
            raise TokenizerAssetError(f"Duplicate token in TikToken asset on line {line_number}.")
        if rank in seen_ranks:
            raise TokenizerAssetError(f"Duplicate rank {rank} in TikToken asset on line {line_number}.")
        ranks[token] = rank
        seen_ranks.add(rank)
    if not ranks:
        raise TokenizerAssetError("TikToken asset does not contain any tokens.")
    return ranks


def load_huggingface_byte_bpe(
    path: str | Path,
    *,
    max_bytes: int = DEFAULT_MAX_ASSET_BYTES,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    max_merges: int = DEFAULT_MAX_MERGES,
    max_token_bytes: int = DEFAULT_MAX_TOKEN_BYTES,
    max_json_depth: int = DEFAULT_MAX_JSON_DEPTH,
    max_json_nodes: int = DEFAULT_MAX_JSON_NODES,
) -> ByteBPEAssets:
    """Load a Hugging Face ``tokenizer.json`` containing a byte-level BPE.

    Only declarative tokenizer data is read. Repository Python modules,
    ``auto_map`` entries, and remote code are never imported or
    executed.
    """
    for name, value in (
        ("max_tokens", max_tokens),
        ("max_merges", max_merges),
        ("max_token_bytes", max_token_bytes),
        ("max_json_depth", max_json_depth),
        ("max_json_nodes", max_json_nodes),
    ):
        _positive_limit(value, name=name)
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
    model = document.get("model")
    if not isinstance(model, dict) or model.get("type") != "BPE":
        raise TokenizerAssetError("Tokenizer JSON must contain a model with type 'BPE'.")
    for unsupported_option in (
            "continuing_subword_prefix",
            "end_of_word_suffix",
            "dropout",
    ):
        if model.get(unsupported_option) not in (None, ""):
            raise TokenizerAssetError(f"Byte-BPE option `{unsupported_option}` is not supported.")
    if model.get("byte_fallback") is True:
        raise TokenizerAssetError("SentencePiece-style `byte_fallback` is not a ByteLevel BPE asset.")

    raw_vocabulary = model.get("vocab")
    if not isinstance(raw_vocabulary, dict) or not raw_vocabulary:
        raise TokenizerAssetError("Byte-BPE model must contain a non-empty vocabulary.")
    if len(raw_vocabulary) > max_tokens:
        raise TokenizerAssetError(f"Tokenizer vocabulary contains more than {max_tokens} tokens.")

    vocabulary: dict[bytes, int] = {}
    vocabulary_strings: dict[str, int] = {}
    seen_ids: set[int] = set()
    for token_string, token_id in raw_vocabulary.items():
        if not isinstance(token_string, str):
            raise TokenizerAssetError("Tokenizer vocabulary keys must be strings.")
        normalized_id = _validate_token_id(token_id, context="vocabulary token ID")
        token = decode_gpt2_token(token_string)
        if not token:
            raise TokenizerAssetError("Byte-BPE vocabulary cannot contain an empty token.")
        if len(token) > max_token_bytes:
            raise TokenizerAssetError(f"Vocabulary token exceeds {max_token_bytes} decoded bytes.")
        if token in vocabulary:
            raise TokenizerAssetError("Byte-BPE vocabulary contains duplicate byte tokens.")
        if normalized_id in seen_ids:
            raise TokenizerAssetError(f"Byte-BPE vocabulary contains duplicate ID {normalized_id}.")
        vocabulary[token] = normalized_id
        vocabulary_strings[token_string] = normalized_id
        seen_ids.add(normalized_id)

    raw_merges = model.get("merges", [])
    if not isinstance(raw_merges, list):
        raise TokenizerAssetError("Byte-BPE `merges` must be an array.")
    if len(raw_merges) > max_merges:
        raise TokenizerAssetError(f"Tokenizer contains more than {max_merges} BPE merges.")
    merges: list[tuple[bytes, bytes]] = []
    seen_pairs: set[tuple[bytes, bytes]] = set()
    for index, raw_merge in enumerate(raw_merges):
        left_string, right_string = _parse_json_merge(raw_merge, index=index)
        pair = (decode_gpt2_token(left_string), decode_gpt2_token(right_string))
        if not pair[0] or not pair[1]:
            raise TokenizerAssetError(f"BPE merge {index} contains an empty token.")
        if pair in seen_pairs:
            raise TokenizerAssetError(f"Duplicate BPE merge at index {index}.")
        combined = pair[0] + pair[1]
        if combined not in vocabulary:
            raise TokenizerAssetError(f"BPE merge {index} produces a token absent from the vocabulary.")
        merges.append(pair)
        seen_pairs.add(pair)

    special_tokens: dict[str, int] = {}
    added_tokens: dict[str, int] = {}
    seen_added_ids: dict[int, str] = {}
    raw_added_tokens = document.get("added_tokens", [])
    if not isinstance(raw_added_tokens, list):
        raise TokenizerAssetError("Tokenizer `added_tokens` must be an array.")
    if len(raw_added_tokens) > max_tokens:
        raise TokenizerAssetError(f"Tokenizer contains more than {max_tokens} added tokens.")
    for index, record in enumerate(raw_added_tokens):
        if not isinstance(record, dict):
            raise TokenizerAssetError(f"Added token {index} must be an object.")
        content = record.get("content")
        token_id = _validate_token_id(
            record.get("id"),
            context=f"added token {index} ID",
        )
        if not isinstance(content, str) or not content:
            raise TokenizerAssetError(f"Added token {index} must have non-empty string content.")
        if record.get("lstrip") or record.get("rstrip"):
            raise TokenizerAssetError("Whitespace-stripping added tokens are not byte-BPE compatible.")
        destination = special_tokens if record.get("special") is True else added_tokens
        existing = destination.get(content)
        if existing is not None and existing != token_id:
            kind = "Special" if destination is special_tokens else "Added"
            raise TokenizerAssetError(f"{kind} token {content!r} is assigned more than one ID.")
        previous_content = seen_added_ids.get(token_id)
        if previous_content is not None and previous_content != content:
            raise TokenizerAssetError(
                f"Added tokens {previous_content!r} and {content!r} share ID "
                f"{token_id}.")
        seen_added_ids[token_id] = content
        if record.get("special") is True:
            if existing is not None and existing != token_id:
                raise TokenizerAssetError(f"Special token {content!r} is assigned more than one ID.")
            special_tokens[content] = token_id
        else:
            added_tokens[content] = token_id

    unk_token_id: int | None = None
    unk_token = model.get("unk_token")
    if unk_token is not None:
        if not isinstance(unk_token, str) or unk_token not in vocabulary_strings:
            raise TokenizerAssetError("BPE `unk_token` is absent from the vocabulary.")
        unk_token_id = vocabulary_strings[unk_token]

    add_prefix_space, use_regex = _find_byte_level_options(document.get("pre_tokenizer"))
    return ByteBPEAssets(
        vocabulary=vocabulary,
        merges=tuple(merges),
        special_tokens=special_tokens,
        added_tokens=added_tokens,
        unk_token_id=unk_token_id,
        add_prefix_space=add_prefix_space,
        use_regex=use_regex,
        normalization=_find_unicode_normalization(document.get("normalizer")),
    )


def gpt2_byte_encoder() -> Mapping[int, str]:
    """Return GPT-2's reversible mapping from bytes to Unicode characters."""
    visible = (
        list(range(ord("!"),
                   ord("~") + 1)) + list(range(ord("¡"),
                                               ord("¬") + 1)) + list(range(ord("®"),
                                                                           ord("ÿ") + 1)))
    byte_values = list(visible)
    code_points = list(visible)
    extra_index = 0
    visible_set = set(visible)
    for byte_value in range(256):
        if byte_value in visible_set:
            continue
        byte_values.append(byte_value)
        code_points.append(256 + extra_index)
        extra_index += 1
    return MappingProxyType({
        byte_value: chr(code_point)
        for byte_value, code_point in zip(byte_values, code_points)
    })


_GPT2_BYTE_ENCODER = gpt2_byte_encoder()
_GPT2_BYTE_DECODER = MappingProxyType({
    character: byte_value
    for byte_value, character in _GPT2_BYTE_ENCODER.items()
})


def encode_gpt2_token(token: bytes) -> str:
    """Represent raw bytes using GPT-2's JSON-safe token alphabet."""
    if not isinstance(token, bytes):
        raise TypeError("`token` must be bytes.")
    return "".join(_GPT2_BYTE_ENCODER[byte] for byte in token)


def decode_gpt2_token(token: str) -> bytes:
    """Convert one GPT-2/Roberta byte-level vocabulary token to raw bytes."""
    if not isinstance(token, str):
        raise TypeError("`token` must be a string.")
    try:
        return bytes(_GPT2_BYTE_DECODER[character] for character in token)
    except KeyError as error:
        raise TokenizerAssetError(
            f"Vocabulary token contains non-byte-level character {error.args[0]!r}.") from error


def _parse_json_merge(value: Any, *, index: int) -> tuple[str, str]:
    if isinstance(value, str):
        fields = value.split(" ")
    elif isinstance(value, list):
        fields = value
    else:
        raise TokenizerAssetError(f"BPE merge {index} must be a string or pair.")
    if len(fields) != 2 or not all(isinstance(field, str) for field in fields):
        raise TokenizerAssetError(f"BPE merge {index} must contain two token strings.")
    return fields[0], fields[1]


def _find_byte_level_options(value: Any) -> tuple[bool, bool]:
    if value is None:
        return False, True
    if not isinstance(value, dict):
        raise TokenizerAssetError("Tokenizer `pre_tokenizer` must be an object.")
    if value.get("type") == "ByteLevel":
        add_prefix_space = value.get("add_prefix_space", False)
        use_regex = value.get("use_regex", True)
        if not isinstance(add_prefix_space, bool):
            raise TokenizerAssetError("ByteLevel `add_prefix_space` must be a boolean.")
        if not isinstance(use_regex, bool):
            raise TokenizerAssetError("ByteLevel `use_regex` must be a boolean.")
        return add_prefix_space, use_regex
    if value.get("type") == "Sequence":
        children = value.get("pretokenizers")
        if not isinstance(children, list):
            raise TokenizerAssetError("Sequence pre-tokenizer must contain a `pretokenizers` array.")
        byte_level_children = tuple(
            child for child in children if isinstance(child, dict) and child.get("type") == "ByteLevel")
        if len(byte_level_children) > 1:
            raise TokenizerAssetError("A tokenizer cannot contain more than one ByteLevel pre-tokenizer.")
        if byte_level_children:
            return _find_byte_level_options(byte_level_children[0])
    return False, True


def _find_unicode_normalization(value: Any) -> str | None:
    if value is None:
        return None
    if not isinstance(value, dict):
        raise TokenizerAssetError("Tokenizer `normalizer` must be an object.")
    normalizer_type = value.get("type")
    if normalizer_type in _UNICODE_NORMALIZATIONS:
        return str(normalizer_type)
    if normalizer_type == "Sequence":
        children = value.get("normalizers")
        if not isinstance(children, list):
            raise TokenizerAssetError("Sequence normalizer must contain a `normalizers` array.")
        normalizations = tuple(
            normalization for child in children
            if (normalization := _find_unicode_normalization(child)) is not None)
        if len(normalizations) > 1:
            raise TokenizerAssetError("More than one Unicode normalization pass is not supported.")
        return normalizations[0] if normalizations else None
    return None


def _validate_json_bounds(value: Any, *, max_depth: int, max_nodes: int) -> None:
    stack: list[tuple[Any, int]] = [(value, 1)]
    nodes = 0
    while stack:
        item, depth = stack.pop()
        nodes += 1
        if nodes > max_nodes:
            raise TokenizerAssetError(f"Tokenizer JSON contains more than {max_nodes} values.")
        if depth > max_depth:
            raise TokenizerAssetError(f"Tokenizer JSON nesting exceeds the configured depth {max_depth}.")
        if isinstance(item, dict):
            stack.extend((child, depth + 1) for child in item.values())
        elif isinstance(item, list):
            stack.extend((child, depth + 1) for child in item)


def _validate_token_id(value: Any, *, context: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TokenizerAssetError(f"{context.capitalize()} must be an integer.")
    normalized = int(value)
    if not 0 <= normalized <= _MAX_TOKEN_ID:
        raise TokenizerAssetError(f"{context.capitalize()} must be in [0, {_MAX_TOKEN_ID}].")
    return normalized


def _positive_limit(value: Any, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"`{name}` must be an integer.")
    normalized = int(value)
    if normalized <= 0:
        raise ValueError(f"`{name}` must be greater than zero.")
    return normalized

"""Dependency-free contracts and immutable results for text tokenizers."""

from __future__ import annotations

from collections.abc import Collection, Iterable, Iterator, Sequence
from dataclasses import dataclass
from numbers import Integral
from typing import Literal, Protocol, TypeAlias, runtime_checkable

SpecialTokenSelection: TypeAlias = Literal["all", "none"] | Collection[str]
PaddingStrategy: TypeAlias = bool | Literal["longest", "max_length"]
TruncationStrategy: TypeAlias = bool | Literal["left", "right"]


def _token_ids(values: Iterable[int], *, field_name: str) -> tuple[int, ...]:
    try:
        normalized = tuple(values)
    except TypeError as error:
        raise TypeError(f"`{field_name}` must be an iterable of token IDs.") from error
    for value in normalized:
        if isinstance(value, bool) or not isinstance(value, Integral):
            raise TypeError(f"`{field_name}` must contain only integer token IDs.")
        if value < 0:
            raise ValueError(f"`{field_name}` cannot contain negative token IDs.")
    return tuple(int(value) for value in normalized)


def _binary_mask(
    values: Iterable[int] | None,
    *,
    field_name: str,
    length: int,
    default: int,
) -> tuple[int, ...]:
    if values is None:
        return (default, ) * length
    try:
        normalized = tuple(values)
    except TypeError as error:
        raise TypeError(f"`{field_name}` must be an iterable of zeros and ones.") from error
    if len(normalized) != length:
        raise ValueError(f"`{field_name}` has length {len(normalized)}, expected {length}.")
    if any(isinstance(value, bool) or not isinstance(value, Integral) or value not in (0, 1)
           for value in normalized):
        raise ValueError(f"`{field_name}` must contain only integer zeros and ones.")
    return tuple(int(value) for value in normalized)


@dataclass(frozen=True, slots=True)
class Encoding:
    """One immutable token sequence and its aligned masks.

    ``attention_mask`` uses one for tokens visible to the model and zero
    for padding. ``special_tokens_mask`` uses one for a special or
    padding token. Tuples are used deliberately so a cached encoding
    cannot be mutated by a caller or a collator.
    """

    input_ids: tuple[int, ...] | Sequence[int]
    attention_mask: tuple[int, ...] | Sequence[int] | None = None
    special_tokens_mask: tuple[int, ...] | Sequence[int] | None = None

    def __post_init__(self) -> None:
        input_ids = _token_ids(self.input_ids, field_name="input_ids")
        object.__setattr__(self, "input_ids", input_ids)
        object.__setattr__(
            self,
            "attention_mask",
            _binary_mask(
                self.attention_mask,
                field_name="attention_mask",
                length=len(input_ids),
                default=1,
            ),
        )
        object.__setattr__(
            self,
            "special_tokens_mask",
            _binary_mask(
                self.special_tokens_mask,
                field_name="special_tokens_mask",
                length=len(input_ids),
                default=0,
            ),
        )

    @property
    def ids(self) -> tuple[int, ...]:
        """Return the token IDs using the concise name used by tokenizers."""
        return self.input_ids

    def __len__(self) -> int:
        return len(self.input_ids)

    def __iter__(self) -> Iterator[int]:
        return iter(self.input_ids)

    def __getitem__(self, index: int | slice) -> int | tuple[int, ...]:
        return self.input_ids[index]


@dataclass(frozen=True, slots=True)
class BatchEncoding:
    """An immutable batch of token sequences.

    Rows may be ragged when padding is disabled. With either padding
    strategy, every row has the same width and can be converted to a
    tensor by a training or inference boundary without making the
    tokenizer depend on a tensor library.
    """

    input_ids: tuple[tuple[int, ...], ...] | Sequence[Sequence[int]]
    attention_mask: tuple[tuple[int, ...], ...] | Sequence[Sequence[int]]
    special_tokens_mask: tuple[tuple[int, ...], ...] | Sequence[Sequence[int]]

    def __post_init__(self) -> None:
        rows = tuple(
            _token_ids(row, field_name=f"input_ids[{index}]") for index, row in enumerate(self.input_ids))
        try:
            attention_rows = tuple(self.attention_mask)
            special_rows = tuple(self.special_tokens_mask)
        except TypeError as error:
            raise TypeError("Batch masks must be iterables of mask rows.") from error
        if len(attention_rows) != len(rows) or len(special_rows) != len(rows):
            raise ValueError("Batch IDs and masks must contain the same number of rows.")

        normalized_attention = tuple(
            _binary_mask(
                row,
                field_name=f"attention_mask[{index}]",
                length=len(rows[index]),
                default=1,
            ) for index, row in enumerate(attention_rows))
        normalized_special = tuple(
            _binary_mask(
                row,
                field_name=f"special_tokens_mask[{index}]",
                length=len(rows[index]),
                default=0,
            ) for index, row in enumerate(special_rows))
        object.__setattr__(self, "input_ids", rows)
        object.__setattr__(self, "attention_mask", normalized_attention)
        object.__setattr__(self, "special_tokens_mask", normalized_special)

    @property
    def ids(self) -> tuple[tuple[int, ...], ...]:
        """Return the token ID rows using the concise tokenizer convention."""
        return self.input_ids

    @property
    def is_padded(self) -> bool:
        """Whether all rows have the same width."""
        return len({len(row) for row in self.input_ids}) <= 1

    def __len__(self) -> int:
        return len(self.input_ids)

    def __iter__(self) -> Iterator[tuple[int, ...]]:
        return iter(self.input_ids)


def pad_encodings(
    encodings: Iterable[Encoding],
    *,
    pad_token_id: int,
    length: int | None = None,
    pad_to_multiple_of: int | None = None,
    padding_side: Literal["left", "right"] = "right",
) -> BatchEncoding:
    """Pad encodings to a common width without importing a tensor library."""
    rows = tuple(encodings)
    if isinstance(pad_token_id, bool) or not isinstance(pad_token_id, Integral):
        raise TypeError("`pad_token_id` must be an integer.")
    if pad_token_id < 0:
        raise ValueError("`pad_token_id` must be non-negative.")
    if padding_side not in ("left", "right"):
        raise ValueError("`padding_side` must be either 'left' or 'right'.")
    if length is not None:
        if isinstance(length, bool) or not isinstance(length, Integral):
            raise TypeError("Padding `length` must be an integer or None.")
        if length < 0:
            raise ValueError("Padding `length` must be non-negative.")
        width = int(length)
    else:
        width = max((len(encoding) for encoding in rows), default=0)
    if pad_to_multiple_of is not None:
        if (isinstance(pad_to_multiple_of, bool) or not isinstance(pad_to_multiple_of, Integral)):
            raise TypeError("`pad_to_multiple_of` must be an integer or None.")
        if pad_to_multiple_of <= 0:
            raise ValueError("`pad_to_multiple_of` must be greater than zero.")
        multiple = int(pad_to_multiple_of)
        width = ((width + multiple - 1) // multiple) * multiple

    padded_ids: list[tuple[int, ...]] = []
    padded_attention: list[tuple[int, ...]] = []
    padded_special: list[tuple[int, ...]] = []
    for encoding in rows:
        amount = width - len(encoding)
        if amount < 0:
            raise ValueError(f"Cannot pad an encoding of length {len(encoding)} to length {width}.")
        padding_ids = (int(pad_token_id), ) * amount
        padding_attention = (0, ) * amount
        padding_special = (1, ) * amount
        if padding_side == "right":
            padded_ids.append(encoding.input_ids + padding_ids)
            padded_attention.append(encoding.attention_mask + padding_attention)
            padded_special.append(encoding.special_tokens_mask + padding_special)
        else:
            padded_ids.append(padding_ids + encoding.input_ids)
            padded_attention.append(padding_attention + encoding.attention_mask)
            padded_special.append(padding_special + encoding.special_tokens_mask)
    return BatchEncoding(
        input_ids=tuple(padded_ids),
        attention_mask=tuple(padded_attention),
        special_tokens_mask=tuple(padded_special),
    )


@runtime_checkable
class Tokenizer(Protocol):
    """Behavior shared by VoiceHub-native text tokenizers."""

    @property
    def vocabulary_size(self) -> int:
        """Number of regular and special tokens."""
        ...

    @property
    def pad_token_id(self) -> int | None:
        """Token used for padded positions, if configured."""
        ...

    def encode(
        self,
        text: str,
        *,
        allowed_special: SpecialTokenSelection = "none",
        disallowed_special: SpecialTokenSelection = "all",
        max_length: int | None = None,
        truncation: TruncationStrategy = False,
    ) -> Encoding:
        """Encode one text value."""
        ...

    def encode_batch(
        self,
        texts: Iterable[str],
        *,
        padding: PaddingStrategy = False,
        max_length: int | None = None,
        truncation: TruncationStrategy = False,
        pad_to_multiple_of: int | None = None,
        allowed_special: SpecialTokenSelection = "none",
        disallowed_special: SpecialTokenSelection = "all",
    ) -> BatchEncoding:
        """Encode and optionally pad a collection of texts."""
        ...

    def decode(
        self,
        token_ids: Iterable[int] | Encoding,
        *,
        skip_special_tokens: bool = False,
        errors: str = "replace",
    ) -> str:
        """Decode token IDs into text."""
        ...

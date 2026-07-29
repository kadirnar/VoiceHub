"""Dependency-free tokenizer for the released English Chatterbox model."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import torch

# Special tokens
SOT = "[START]"
EOT = "[STOP]"
UNK = "[UNK]"
SPACE = "[SPACE]"
SPECIAL_TOKENS = (
    SOT,
    EOT,
    UNK,
    SPACE,
    "[PAD]",
    "[SEP]",
    "[CLS]",
    "[MASK]",
)

_MAX_ASSET_BYTES = 4 * 1024 * 1024
_MAX_VOCABULARY = 100_000
_MAX_MERGES = 500_000
_WHITESPACE_PATTERN = re.compile(r"\w+|[^\w\s]+", re.UNICODE)


def _reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Chatterbox tokenizer JSON contains duplicate key {key!r}.")
        result[key] = value
    return result


class EnTokenizer:
    """Source-compatible BPE over Chatterbox's ``tokenizer.json`` asset.

    The asset uses Hugging Face Tokenizers' BPE model with the
    ``Whitespace`` pre-tokenizer and explicit added tokens.  This
    implementation keeps those exact semantics in standard-library
    Python and bounds every untrusted structure before use.
    """

    def __init__(self, vocab_file_path: str | Path):
        self.asset_path = Path(vocab_file_path).expanduser().resolve()
        if not self.asset_path.is_file():
            raise FileNotFoundError(f"Chatterbox tokenizer asset was not found: {self.asset_path}.")
        size = self.asset_path.stat().st_size
        if size <= 0 or size > _MAX_ASSET_BYTES:
            raise ValueError(
                "Chatterbox tokenizer asset size is outside the allowed "
                f"range 1..{_MAX_ASSET_BYTES} bytes.")
        try:
            payload = json.loads(
                self.asset_path.read_text(encoding="utf-8"),
                object_pairs_hook=_reject_duplicates,
            )
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError(f"Could not parse Chatterbox tokenizer JSON: {error}.") from error
        if not isinstance(payload, dict):
            raise ValueError("Chatterbox tokenizer asset must contain an object.")
        model = payload.get("model")
        if not isinstance(model, dict) or model.get("type") != "BPE":
            raise ValueError("Chatterbox requires a tokenizer.json BPE model.")
        if model.get("dropout") not in (None, 0, 0.0):
            raise ValueError("Stochastic tokenizer BPE dropout is not supported.")
        vocabulary = model.get("vocab")
        if (not isinstance(vocabulary, dict) or not vocabulary or len(vocabulary) > _MAX_VOCABULARY):
            raise ValueError("Chatterbox tokenizer vocabulary is invalid or too large.")
        normalized_vocabulary: dict[str, int] = {}
        used_ids: dict[int, str] = {}
        for token, token_id in vocabulary.items():
            if (not isinstance(token, str) or not token or isinstance(token_id, bool) or
                    not isinstance(token_id, int) or token_id < 0):
                raise ValueError(
                    "Chatterbox tokenizer vocabulary must map non-empty "
                    "strings to non-negative integer IDs.")
            previous = used_ids.get(token_id)
            if previous is not None and previous != token:
                raise ValueError(f"Chatterbox tokenizer ID {token_id} is duplicated.")
            normalized_vocabulary[token] = token_id
            used_ids[token_id] = token
        self.vocabulary = normalized_vocabulary
        self.id_to_token = used_ids

        raw_merges = model.get("merges", [])
        if (not isinstance(raw_merges, list) or len(raw_merges) > _MAX_MERGES):
            raise ValueError("Chatterbox tokenizer merge table is invalid or too large.")
        pair_ranks: dict[tuple[str, str], int] = {}
        for rank, item in enumerate(raw_merges):
            if isinstance(item, str):
                pair = item.split(" ")
            elif isinstance(item, list):
                pair = item
            else:
                raise ValueError("Every Chatterbox BPE merge must be a pair.")
            if (len(pair) != 2 or not all(isinstance(value, str) and value for value in pair)):
                raise ValueError("Every Chatterbox BPE merge must contain two tokens.")
            key = pair[0], pair[1]
            if key in pair_ranks:
                raise ValueError(f"Duplicate Chatterbox BPE merge {key!r}.")
            if pair[0] + pair[1] not in self.vocabulary:
                raise ValueError(f"Chatterbox BPE merge {key!r} produces an unknown token.")
            pair_ranks[key] = rank
        self.pair_ranks = pair_ranks

        added_tokens = payload.get("added_tokens", [])
        if not isinstance(added_tokens, list):
            raise ValueError("Chatterbox added_tokens must be a list.")
        added: dict[str, int] = {}
        for item in added_tokens:
            if not isinstance(item, dict):
                raise ValueError("Each Chatterbox added token must be an object.")
            content = item.get("content")
            token_id = item.get("id")
            if (not isinstance(content, str) or not content or isinstance(token_id, bool) or
                    not isinstance(token_id, int) or self.vocabulary.get(content) != token_id):
                raise ValueError("Chatterbox added tokens must match the BPE vocabulary.")
            if any(bool(item.get(name, False)) for name in (
                    "single_word",
                    "lstrip",
                    "rstrip",
            )):
                raise ValueError(
                    "This Chatterbox runtime only accepts the released "
                    "non-stripping added-token contract.")
            added[content] = token_id
        self.added_tokens = added
        self._added_spellings = tuple(sorted(added, key=lambda value: (-len(value), value)))
        unknown = model.get("unk_token")
        if not isinstance(unknown, str) or unknown not in self.vocabulary:
            raise ValueError("Chatterbox BPE must declare a known unk_token.")
        self.unk_token_id = self.vocabulary[unknown]
        self.check_vocabset_sot_eot()

    def check_vocabset_sot_eot(self) -> None:
        if SOT not in self.vocabulary or EOT not in self.vocabulary:
            raise ValueError("Chatterbox tokenizer vocabulary lacks [START] or [STOP].")

    def _added_match(self, text: str, start: int) -> tuple[int, str] | None:
        best: tuple[int, str] | None = None
        for spelling in self._added_spellings:
            position = text.find(spelling, start)
            if position < 0:
                continue
            candidate = position, spelling
            if best is None or candidate[0] < best[0] or (candidate[0] == best[0] and
                                                          len(candidate[1]) > len(best[1])):
                best = candidate
        return best

    def _bpe(self, piece: str) -> list[int]:
        symbols = list(piece)
        while len(symbols) > 1:
            candidates = ((self.pair_ranks.get((symbols[index], symbols[index + 1])), index)
                          for index in range(len(symbols) - 1))
            ranked = [(rank, index) for rank, index in candidates if rank is not None]
            if not ranked:
                break
            selected_rank = min(rank for rank, _ in ranked)
            merged: list[str] = []
            index = 0
            while index < len(symbols):
                can_merge = index + 1 < len(symbols)
                if can_merge:
                    pair = (symbols[index], symbols[index + 1])
                    can_merge = self.pair_ranks.get(pair) == selected_rank
                if can_merge:
                    merged.append(symbols[index] + symbols[index + 1])
                    index += 2
                else:
                    merged.append(symbols[index])
                    index += 1
            symbols = merged
        return [self.vocabulary.get(symbol, self.unk_token_id) for symbol in symbols]

    def _ordinary(self, text: str) -> list[int]:
        token_ids: list[int] = []
        for match in _WHITESPACE_PATTERN.finditer(text):
            token_ids.extend(self._bpe(match.group(0)))
        return token_ids

    def encode(self, text: str, verbose: bool = False) -> list[int]:
        del verbose
        if not isinstance(text, str):
            raise TypeError("Chatterbox tokenizer input must be a string.")
        text = text.replace(" ", SPACE)
        output: list[int] = []
        cursor = 0
        while cursor < len(text):
            match = self._added_match(text, cursor)
            if match is None:
                output.extend(self._ordinary(text[cursor:]))
                break
            position, spelling = match
            if position > cursor:
                output.extend(self._ordinary(text[cursor:position]))
            output.append(self.added_tokens[spelling])
            cursor = position + len(spelling)
        return output

    def text_to_tokens(self, text: str) -> torch.Tensor:
        return torch.tensor(
            self.encode(text),
            dtype=torch.int32,
        ).unsqueeze(0)

    def decode(self, sequence) -> str:
        if isinstance(sequence, torch.Tensor):
            values = sequence.detach().cpu().reshape(-1).tolist()
        else:
            values = list(sequence)
        tokens = [self.id_to_token.get(int(token_id), UNK) for token_id in values]
        text = "".join(tokens)
        return (text.replace(" ", "").replace(SPACE, " ").replace(EOT, "").replace(UNK, ""))


__all__ = [
    "EOT",
    "EnTokenizer",
    "SPACE",
    "SPECIAL_TOKENS",
    "SOT",
    "UNK",
]

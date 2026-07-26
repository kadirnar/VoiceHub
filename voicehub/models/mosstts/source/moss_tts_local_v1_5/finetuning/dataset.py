from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

from common import normalize_audio_path_list


USER_MESSAGE_KEYS = (
    "text",
    "instruction",
    "tokens",
    "quality",
    "sound_event",
    "ambient_sound",
    "language",
)


def normalize_audio_codes(value: Any, field_name: str) -> torch.Tensor:
    tensor = torch.as_tensor(value, dtype=torch.long)
    if tensor.ndim != 2:
        raise ValueError(f"`{field_name}` must have shape (T, n_vq), got {tuple(tensor.shape)}.")
    return tensor.cpu().contiguous()


def normalize_audio_code_list(
    value: Any,
    field_name: str,
    *,
    allow_none: bool = False,
) -> Optional[List[Optional[torch.Tensor]]]:
    if value in (None, "", []):
        return None
    if torch.is_tensor(value):
        return [normalize_audio_codes(value, field_name)]
    if isinstance(value, list):
        if not value:
            return None
        if allow_none and any(item is None for item in value):
            return [
                None if item is None else normalize_audio_codes(item, f"{field_name}[{index}]")
                for index, item in enumerate(value)
            ]
        first_item = value[0]
        if torch.is_tensor(first_item):
            return [normalize_audio_codes(item, f"{field_name}[{index}]") for index, item in enumerate(value)]
        if isinstance(first_item, list):
            if first_item and isinstance(first_item[0], list):
                return [normalize_audio_codes(item, f"{field_name}[{index}]") for index, item in enumerate(value)]
            return [normalize_audio_codes(value, field_name)]
    raise TypeError(f"Unsupported `{field_name}` type: {type(value)}")


class MossTTSLocalV15SFTDataset(Dataset):
    def __init__(
        self,
        records: Iterable[Dict[str, Any]],
        processor,
        n_vq: Optional[int] = None,
    ) -> None:
        self.records = list(records)
        self.processor = processor
        self.n_vq = n_vq
        self._audio_cache: Dict[str, torch.Tensor] = {}

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        return self._pack_record(self.records[index], index=index)

    def _audio_pad_token_id(self) -> int:
        return int(
            getattr(
                self.processor.model_config,
                "audio_pad_token_id",
                getattr(self.processor.model_config, "audio_pad_code"),
            )
        )

    def _encode_audio_paths(self, paths: List[str], target_n_vq: int) -> List[torch.Tensor]:
        uncached_paths = [path for path in paths if path not in self._audio_cache]
        if uncached_paths:
            if getattr(self.processor, "audio_tokenizer", None) is None:
                raise ValueError(
                    "Found reference audio paths but processor.audio_tokenizer is not available. "
                    "Run prepare_data.py with reference encoding enabled, or keep the codec loaded during training."
                )
            encoded = self.processor.encode_audios_from_path(uncached_paths, n_vq=target_n_vq)
            for path, codes in zip(uncached_paths, encoded):
                self._audio_cache[path] = codes.cpu()
        return [self._audio_cache[path] for path in paths]

    def _validate_code_list(
        self,
        codes_list: Optional[List[Optional[torch.Tensor]]],
        target_n_vq: int,
        field_name: str,
    ) -> Optional[List[Optional[torch.Tensor]]]:
        if codes_list is None:
            return None
        for codes in codes_list:
            if codes is None:
                continue
            if int(codes.shape[1]) != target_n_vq:
                raise ValueError(
                    f"`{field_name}` n_vq={codes.shape[1]} does not match target n_vq={target_n_vq}."
                )
        return codes_list

    def _resolve_reference_codes(
        self,
        record: Dict[str, Any],
        target_n_vq: int,
    ) -> Optional[List[Optional[torch.Tensor]]]:
        if record.get("reference_audio_codes") is not None:
            return self._validate_code_list(
                normalize_audio_code_list(
                    record["reference_audio_codes"],
                    "reference_audio_codes",
                    allow_none=True,
                ),
                target_n_vq,
                "reference_audio_codes",
            )
        if record.get("ref_audio_codes") is not None:
            return self._validate_code_list(
                normalize_audio_code_list(record["ref_audio_codes"], "ref_audio_codes"),
                target_n_vq,
                "ref_audio_codes",
            )

        for path_field in ("reference", "reference_audio", "ref_audio"):
            paths = normalize_audio_path_list(
                record.get(path_field),
                path_field,
                allow_none=(path_field == "reference"),
            )
            if paths is None:
                continue
            encoded_paths = self._encode_audio_paths([path for path in paths if path is not None], target_n_vq)
            encoded_iter = iter(encoded_paths)
            return [None if path is None else next(encoded_iter) for path in paths]
        return None

    def _pack_record(self, record: Dict[str, Any], *, index: int) -> Dict[str, torch.Tensor]:
        if "audio_codes" not in record:
            raise ValueError(f"Record {index} is missing `audio_codes`. Run prepare_data.py first.")

        target_codes = normalize_audio_codes(record["audio_codes"], "audio_codes")
        target_n_vq = int(target_codes.shape[1])
        if self.n_vq is not None and target_n_vq != int(self.n_vq):
            raise ValueError(f"Expected n_vq={self.n_vq}, but got {target_n_vq} in record {index}.")

        config_n_vq = int(getattr(self.processor.model_config, "n_vq", target_n_vq))
        if target_n_vq != config_n_vq:
            raise ValueError(
                f"MOSS-TTS Local v1.5 uses fixed n_vq={config_n_vq}; record {index} has n_vq={target_n_vq}."
            )

        reference_codes = self._resolve_reference_codes(record, target_n_vq)
        user_kwargs: Dict[str, Any] = {"reference": reference_codes}
        for key in USER_MESSAGE_KEYS:
            if record.get(key) is not None:
                user_kwargs[key] = record[key]

        user_message = self.processor.build_user_message(**user_kwargs)
        prompt = self.processor([[user_message]], mode="generation", n_vq=target_n_vq)
        assistant_message = self.processor.build_assistant_message(audio_codes_list=[target_codes])
        conversation = self.processor(
            [[user_message, assistant_message]],
            mode="computing_loss",
            n_vq=target_n_vq,
        )

        full_input_ids = conversation["input_ids"][0].cpu()
        prompt_length = int(prompt["input_ids"][0].shape[0])
        if prompt_length >= int(full_input_ids.shape[0]):
            raise ValueError(
                f"Record {index} prompt length must be shorter than the packed teacher-forcing sequence."
            )

        loss_mask = torch.zeros(full_input_ids.shape[0] - 1, dtype=torch.bool)
        loss_mask[prompt_length - 1 :] = True
        return {
            "input_ids": full_input_ids,
            "loss_mask": loss_mask,
            "record_id": str(record.get("id", index)),
        }

    def collate_fn(self, batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        input_ids_list = [item["input_ids"] for item in batch]
        padded = self.processor._pad(input_ids_list)

        full_input_ids = padded["input_ids"].to(torch.long)
        full_attention_mask = padded["attention_mask"].bool()
        loss_masks = pad_sequence(
            [item["loss_mask"] for item in batch],
            batch_first=True,
            padding_value=False,
            padding_side="left",
        )

        labels = full_input_ids[:, 1:, :].clone()
        labels = labels.masked_fill(~loss_masks.unsqueeze(-1), -100)
        labels = labels.masked_fill(~full_attention_mask[:, 1:].unsqueeze(-1), -100)
        audio_pad_token_id = self._audio_pad_token_id()
        labels[:, :, 1:] = labels[:, :, 1:].masked_fill(
            labels[:, :, 1:] == audio_pad_token_id,
            -100,
        )

        return {
            "input_ids": full_input_ids[:, :-1, :].contiguous(),
            "attention_mask": full_attention_mask[:, :-1].contiguous(),
            "labels": labels.contiguous(),
            "record_ids": [str(item.get("record_id", "")) for item in batch],
        }


MossTTSSFTDataset = MossTTSLocalV15SFTDataset

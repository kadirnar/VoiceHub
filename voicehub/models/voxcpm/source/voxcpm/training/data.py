import math
from typing import Dict, List, Optional, Tuple

import argbind
import torch
from datasets import Audio, Dataset, DatasetDict, load_dataset
from torch.utils.data import Dataset as TorchDataset

from ..model.voxcpm import VoxCPMConfig
from ..modules.audiovae import AudioVAE
from .packers import AudioFeatureProcessingPacker

DEFAULT_TEXT_COLUMN = "text"
DEFAULT_AUDIO_COLUMN = "audio"
DEFAULT_REF_AUDIO_COLUMN = "ref_audio"
DEFAULT_ID_COLUMN = "dataset_id"


@argbind.bind()
def load_audio_text_datasets(
    train_manifest: str,
    val_manifest: str = "",
    text_column: str = DEFAULT_TEXT_COLUMN,
    audio_column: str = DEFAULT_AUDIO_COLUMN,
    ref_audio_column: str = DEFAULT_REF_AUDIO_COLUMN,
    dataset_id_column: str = DEFAULT_ID_COLUMN,
    sample_rate: int = 16_000,
    num_proc: int = 1,
) -> Tuple[Dataset, Optional[Dataset]]:
    data_files = {"train": train_manifest}
    if val_manifest:
        data_files["validation"] = val_manifest

    dataset_dict: DatasetDict = load_dataset("json", data_files=data_files)

    def prepare(ds: Dataset) -> Dataset:
        if audio_column not in ds.column_names:
            raise ValueError(f"Expected '{audio_column}' column in manifest.")
        ds = ds.cast_column(audio_column, Audio(sampling_rate=sample_rate))
        if audio_column != DEFAULT_AUDIO_COLUMN:
            ds = ds.rename_column(audio_column, DEFAULT_AUDIO_COLUMN)
        if text_column != DEFAULT_TEXT_COLUMN:
            ds = ds.rename_column(text_column, DEFAULT_TEXT_COLUMN)

        # ref_audio is optional — cast to Audio if the column exists
        ref_col = ref_audio_column if ref_audio_column in ds.column_names else DEFAULT_REF_AUDIO_COLUMN
        if ref_col in ds.column_names:
            ds = ds.cast_column(ref_col, Audio(sampling_rate=sample_rate))
            if ref_col != DEFAULT_REF_AUDIO_COLUMN:
                ds = ds.rename_column(ref_col, DEFAULT_REF_AUDIO_COLUMN)

        if dataset_id_column and dataset_id_column in ds.column_names:
            if dataset_id_column != DEFAULT_ID_COLUMN:
                ds = ds.rename_column(dataset_id_column, DEFAULT_ID_COLUMN)
        else:
            ds = ds.add_column(DEFAULT_ID_COLUMN, [0] * len(ds))
        return ds

    train_ds = prepare(dataset_dict["train"])
    val_ds = prepare(dataset_dict["validation"]) if "validation" in dataset_dict else None
    return train_ds, val_ds


def compute_sample_lengths(
    ds: Dataset,
    audio_vae_fps: int = 25,
    patch_size: int = 1,
) -> List[int]:
    """
    预估每个样本经过 packer 之后的大致序列长度（text+audio），用于过滤超长样本。

    逻辑与 AudioFeatureProcessingPacker / AudioVAE 一致：
    - 文本长度: len(text_ids)
    - 音频长度:
        duration(s) * audio_vae_fps -> 近似 VAE 帧数 t_vae
        t_seq = ceil(t_vae / patch_size)
    - 无 ref_audio: text_len + t_seq + 2
    - 有 ref_audio: text_len + t_seq + ref_seq + 4

    Optimized: Use batch column access instead of iterating item by item.
    """
    text_ids_list = ds["text_ids"]
    text_lens = [len(t) for t in text_ids_list]

    has_duration = "duration" in ds.column_names
    if has_duration:
        durations = ds["duration"]
    else:
        durations = []
        for i in range(len(ds)):
            audio = ds[i][DEFAULT_AUDIO_COLUMN]
            durations.append(len(audio["array"]) / float(audio["sampling_rate"]))

    has_ref_audio = DEFAULT_REF_AUDIO_COLUMN in ds.column_names
    if has_ref_audio:
        ref_duration_col = "ref_duration" if "ref_duration" in ds.column_names else None

    lengths = []
    for i, (text_len, duration) in enumerate(zip(text_lens, durations)):
        t_vae = math.ceil(float(duration) * audio_vae_fps)
        t_seq = math.ceil(t_vae / patch_size)

        ref_seq = 0
        if has_ref_audio:
            # Estimate ref_audio length; ref_audio is None for samples without it
            if ref_duration_col:
                ref_dur = ds[i].get(ref_duration_col)
            else:
                ref_item = ds[i].get(DEFAULT_REF_AUDIO_COLUMN)
                ref_dur = len(ref_item["array"]) / float(ref_item["sampling_rate"]) if ref_item else None
            if ref_dur is not None and float(ref_dur) > 0:
                ref_vae = math.ceil(float(ref_dur) * audio_vae_fps)
                ref_seq = math.ceil(ref_vae / patch_size)

        # +2 for 101/102; +2 more for 103/104 when ref_audio present
        overhead = 4 if ref_seq > 0 else 2
        total_len = text_len + t_seq + ref_seq + overhead
        lengths.append(total_len)

    return lengths


class HFVoxCPMDataset(TorchDataset):
    """
    Thin wrapper around a tokenized HuggingFace dataset that returns
    PyTorch-friendly samples.
    """

    _SENTINEL = [-100.0]

    def __init__(self, dataset: Dataset):
        self.dataset = dataset
        self.has_ref_audio = DEFAULT_REF_AUDIO_COLUMN in dataset.column_names

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx: int):
        item = self.dataset[idx]
        audio = item[DEFAULT_AUDIO_COLUMN]
        sample = {
            "text_ids": item["text_ids"],
            "audio_array": audio["array"],
            "audio_sampling_rate": audio["sampling_rate"],
            "dataset_id": item.get(DEFAULT_ID_COLUMN, 0),
            "is_prompt": item.get("is_prompt", False),
        }
        if self.has_ref_audio:
            ref = item.get(DEFAULT_REF_AUDIO_COLUMN)
            sample["ref_audio_array"] = ref["array"] if ref else self._SENTINEL
        return sample

    @staticmethod
    def pad_sequences(seqs: List[torch.Tensor], pad_value: float):
        if not seqs:
            return torch.empty(0)
        max_len = max(seq.shape[0] for seq in seqs)
        padded = []
        for seq in seqs:
            if seq.shape[0] < max_len:
                pad_width = (0, max_len - seq.shape[0])
                seq = torch.nn.functional.pad(seq, pad_width, value=pad_value)
            padded.append(seq)
        return torch.stack(padded)

    @classmethod
    def collate_fn(cls, batch: List[Dict]):
        text_tensors = [torch.tensor(sample["text_ids"], dtype=torch.int32) for sample in batch]
        audio_tensors = [torch.tensor(sample["audio_array"], dtype=torch.float32) for sample in batch]
        dataset_ids = torch.tensor([sample["dataset_id"] for sample in batch], dtype=torch.int32)
        is_prompts = [bool(sample.get("is_prompt", False)) for sample in batch]

        text_padded = cls.pad_sequences(text_tensors, pad_value=-100)
        audio_padded = cls.pad_sequences(audio_tensors, pad_value=-100.0)
        task_ids = torch.ones(text_padded.size(0), dtype=torch.int32)

        result = {
            "text_tokens": text_padded,
            "audio_tokens": audio_padded,
            "task_ids": task_ids,
            "dataset_ids": dataset_ids,
            "is_prompts": is_prompts,
        }

        if "ref_audio_array" in batch[0]:
            ref_tensors = [torch.tensor(s["ref_audio_array"], dtype=torch.float32) for s in batch]
            result["ref_audio_tokens"] = cls.pad_sequences(ref_tensors, pad_value=-100.0)

        return result


class BatchProcessor:
    """
    Wraps ``AudioFeatureProcessingPacker`` so the training loop can mirror
    the minicpm-audio mechanics.
    """

    def __init__(
        self,
        *,
        config: VoxCPMConfig,
        audio_vae: AudioVAE,
        dataset_cnt: int,
        device: torch.device,
    ):
        self.device = device
        self.dataset_cnt = dataset_cnt
        self.audio_vae = audio_vae
        self.audio_vae.to(device)
        self.packer = AudioFeatureProcessingPacker(
            dataset_cnt=dataset_cnt,
            max_len=config.max_length,
            patch_size=config.patch_size,
            feat_dim=config.feat_dim,
            audio_vae=self.audio_vae,
        )

    def __call__(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        audio_tokens = batch["audio_tokens"].to(self.device)
        text_tokens = batch["text_tokens"].to(self.device)
        task_ids = batch["task_ids"].to(self.device)
        dataset_ids = batch["dataset_ids"].to(self.device)

        ref_audio_tokens = None
        if "ref_audio_tokens" in batch:
            ref_audio_tokens = batch["ref_audio_tokens"].to(self.device)

        packed = self.packer(
            audio_tokens=audio_tokens,
            text_tokens=text_tokens,
            task_ids=task_ids,
            dataset_ids=dataset_ids,
            is_prompts=batch["is_prompts"],
            ref_audio_tokens=ref_audio_tokens,
        )
        return packed


def build_dataloader(
    hf_dataset: Dataset,
    *,
    accelerator,
    batch_size: int,
    num_workers: int,
    drop_last: bool = False,
) -> torch.utils.data.DataLoader:
    torch_dataset = HFVoxCPMDataset(hf_dataset)
    # Standard padding-based batching; Accelerator will attach DistributedSampler if needed.
    return accelerator.prepare_dataloader(
        torch_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True,
        collate_fn=HFVoxCPMDataset.collate_fn,
        drop_last=drop_last,
    )

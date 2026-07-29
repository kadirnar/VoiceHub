"""Inference, fine-tuning, and export lifecycle for native CosyVoice."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from voicehub.architectures.cosyvoice_native.artifacts import resolve_cosyvoice_artifacts
from voicehub.architectures.cosyvoice_native.checkpoint import export_cosyvoice_checkpoint, load_cosyvoice_checkpoint
from voicehub.architectures.cosyvoice_native.configuration import CosyVoiceArchitectureConfig
from voicehub.architectures.cosyvoice_native.modeling import CosyVoiceNativeModel, CosyVoiceSynthesisOutput
from voicehub.architectures.cosyvoice_native.tokenization import CosyVoiceTextTokenizer
from voicehub.hub import read_json_file, write_json_file


class CosyVoiceNativeRuntime:
    """Own one native graph and its immutable text tokenizer assets."""

    def __init__(
        self,
        model: CosyVoiceNativeModel,
        tokenizer: CosyVoiceTextTokenizer,
    ) -> None:
        if not isinstance(model, CosyVoiceNativeModel):
            raise TypeError("`model` must be CosyVoiceNativeModel.")
        if not isinstance(tokenizer, CosyVoiceTextTokenizer):
            raise TypeError("`tokenizer` must be CosyVoiceTextTokenizer.")
        self.model = model
        self.tokenizer = tokenizer

    @property
    def sample_rate(self) -> int:
        return self.model.config.sample_rate

    def prepare_for_training(self, component: str) -> torch.nn.Module:
        normalized = str(component).strip().lower().replace("-", "_")
        if normalized.startswith("hifigan"):
            self.model.attach_discriminator(tiny=(self.model.config.hift.base_channels < 64), )
            if normalized == "hifigan_discriminator":
                selected = self.model.hifigan.discriminator
                for parameter in self.model.parameters():
                    parameter.requires_grad_(False)
                for parameter in selected.parameters():
                    parameter.requires_grad_(True)
                return selected
        return self.model.freeze_except(normalized)

    def prepare_for_inference(self) -> None:
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

    def prepare_language_batch(
        self,
        records: Sequence[Mapping[str, Any]],
        *,
        device: str | torch.device | None = None,
    ) -> dict[str, Tensor]:
        if not records:
            raise ValueError("CosyVoice language batch cannot be empty.")
        text_items: list[Tensor] = []
        instruction_items: list[Tensor] = []
        speech_items: list[Tensor] = []
        for record in records:
            if not isinstance(record, Mapping):
                raise TypeError("Every CosyVoice training record must be a mapping.")
            supplied_text = record.get("text_tokens")
            if supplied_text is None:
                supplied_text = self.tokenizer.encode_tensor(
                    record["text"],
                    device=device,
                )[0]
            text_items.append(torch.as_tensor(
                supplied_text,
                dtype=torch.long,
                device=device,
            ).flatten())
            supplied_instruction = record.get("instruction_tokens")
            if supplied_instruction is None:
                supplied_instruction = self.tokenizer.instruction_tokens(
                    record.get("instruction"),
                    device=device,
                )[0]
            instruction_items.append(
                torch.as_tensor(
                    supplied_instruction,
                    dtype=torch.long,
                    device=device,
                ).flatten())
            if "speech_tokens" not in record:
                raise ValueError(
                    "Native CosyVoice SFT requires pre-encoded `speech_tokens`; "
                    "the published speech-tokenizer remains frozen.")
            speech_items.append(
                torch.as_tensor(
                    record["speech_tokens"],
                    dtype=torch.long,
                    device=device,
                ).flatten())

        def pad(items: list[Tensor], value: int = 0) -> tuple[Tensor, Tensor]:
            lengths = torch.tensor(
                [item.numel() for item in items],
                dtype=torch.long,
                device=device,
            )
            maximum = int(lengths.max().item())
            result = items[0].new_full((len(items), maximum), value)
            for index, item in enumerate(items):
                result[index, :item.numel()] = item
            return result, lengths

        text, text_lengths = pad(text_items, self.tokenizer.pad_token_id)
        instruction, instruction_lengths = pad(
            instruction_items,
            self.tokenizer.pad_token_id,
        )
        speech, speech_lengths = pad(speech_items)
        return {
            "text_tokens": text,
            "text_lengths": text_lengths,
            "instruction_tokens": instruction,
            "instruction_lengths": instruction_lengths,
            "speech_tokens": speech,
            "speech_lengths": speech_lengths,
        }

    @torch.inference_mode()
    def generate(
        self,
        text: str,
        *,
        speaker_embedding: Tensor,
        instruction: str | None = None,
        prompt_speech_tokens: Tensor | None = None,
        prompt_features: Tensor | None = None,
        min_new_tokens: int = 0,
        max_new_tokens: int = 1_024,
        top_k: int = 25,
        top_p: float = 0.8,
        temperature: float = 1.0,
        flow_steps: int = 10,
        seed: int | None = None,
    ) -> CosyVoiceSynthesisOutput:
        device = next(self.model.parameters()).device
        text_tokens = self.tokenizer.encode_tensor(text, device=device)
        instruction_tokens = self.tokenizer.instruction_tokens(
            instruction,
            device=device,
        )
        speaker_embedding = torch.as_tensor(
            speaker_embedding,
            device=device,
            dtype=next(self.model.flow.parameters()).dtype,
        )
        if speaker_embedding.ndim == 1:
            speaker_embedding = speaker_embedding[None]
        expected = self.model.config.flow.speaker_embedding_dim
        if tuple(speaker_embedding.shape) != (1, expected):
            raise ValueError(f"`speaker_embedding` must have shape [{expected}] or [1, {expected}].")
        if prompt_speech_tokens is not None:
            prompt_speech_tokens = prompt_speech_tokens.to(
                device=device,
                dtype=torch.long,
            )
            if prompt_speech_tokens.ndim == 1:
                prompt_speech_tokens = prompt_speech_tokens[None]
        if prompt_features is not None:
            prompt_features = prompt_features.to(
                device=device,
                dtype=speaker_embedding.dtype,
            )
        generator = None
        if seed is not None:
            generator = torch.Generator(device=device)
            generator.manual_seed(seed)
        self.prepare_for_inference()
        return self.model.synthesize(
            text_tokens,
            instruction_tokens,
            speaker_embedding,
            prompt_speech_tokens=prompt_speech_tokens,
            prompt_features=prompt_features,
            min_new_tokens=min_new_tokens,
            max_new_tokens=max_new_tokens,
            top_k=top_k,
            top_p=top_p,
            temperature=temperature,
            flow_steps=flow_steps,
            generator=generator,
        )

    def save_pretrained(self, directory: str | Path) -> Path:
        target = Path(directory).expanduser()
        target.mkdir(parents=True, exist_ok=True)
        occupied = [path for path in target.iterdir()]
        if occupied:
            raise FileExistsError("CosyVoice export destination must be empty: "
                                  f"{target}.")
        write_json_file(
            target / "cosyvoice_config.json",
            self.model.config.to_dict(),
        )
        export_cosyvoice_checkpoint(
            self.model.llm,
            target / "llm.safetensors",
            component="llm",
        )
        export_cosyvoice_checkpoint(
            self.model.flow,
            target / "flow.safetensors",
            component="flow",
        )
        export_cosyvoice_checkpoint(
            self.model.hift,
            target / "hift.safetensors",
            component="hift",
        )
        self.tokenizer.save_pretrained(target)
        return target


def load_cosyvoice_runtime(
    source: str | Path,
    *,
    revision: str | None = None,
    device: str | torch.device = "cpu",
    dtype: torch.dtype | None = None,
    cache_dir: str | None = None,
    token: str | bool | None = None,
    local_files_only: bool = False,
    require_official_inventory: bool = False,
) -> CosyVoiceNativeRuntime:
    """Strictly load one complete pickle-free native artifact."""
    artifacts = resolve_cosyvoice_artifacts(
        source,
        revision=revision,
        cache_dir=cache_dir,
        token=token,
        local_files_only=local_files_only,
    )
    config = CosyVoiceArchitectureConfig.from_dict(read_json_file(artifacts.config))
    with torch.device("meta"):
        model = CosyVoiceNativeModel(
            config,
            initialize=False,
            dtype=dtype,
        )
    for component, module, path in (
        ("llm", model.llm, artifacts.llm),
        ("flow", model.flow, artifacts.flow),
        ("hift", model.hift, artifacts.hift),
    ):
        load_cosyvoice_checkpoint(
            module,
            path,
            component=component,
            device=device,
            dtype=dtype,
            require_official_inventory=require_official_inventory,
        )
    tokenizer = CosyVoiceTextTokenizer.from_files(
        artifacts.vocab,
        artifacts.merges,
        artifacts.tokenizer_config,
        validate_published_ids=(config.generation == 3 and config.language.text_vocab_size == 151_936),
    )
    return CosyVoiceNativeRuntime(model, tokenizer)


__all__ = [
    "CosyVoiceNativeRuntime",
    "load_cosyvoice_runtime",
]

"""Source-compatible fine-tuning utilities for Higgs Audio.

The released Higgs model exposes both text and audio logits but leaves
its loss fields unset.  This module supplies the causal objectives and
reuses the vendored ChatML preparation/collation path so audio
boundaries retain the author-defined masking semantics.
"""

from __future__ import annotations

import base64
import math
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass, fields
from io import BytesIO
from pathlib import Path
from typing import Any

from voicehub.dependencies import import_optional
from voicehub.modeling_outputs import TTSTrainingOutput
from voicehub.training.adapters import CausalLMTrainingAdapter
from voicehub.training.contracts import TrainingContext


def _resolve_training_dtype(torch: Any, dtype_name: str, device: str) -> Any:
    """Resolve the configured dtype without using reduced precision on CPU."""
    try:
        dtype = getattr(torch, dtype_name)
    except AttributeError as exc:
        raise ValueError(f"Unsupported Higgs Audio torch_dtype {dtype_name!r}.") from exc
    if device == "cpu" and dtype in {torch.float16, torch.bfloat16}:
        return torch.float32
    return dtype


def _freeze_audio_tokenizer(audio_tokenizer: Any) -> None:
    """Keep the codec as preprocessing state, never optimizer state."""
    requires_grad = getattr(audio_tokenizer, "requires_grad_", None)
    if callable(requires_grad):
        requires_grad(False)
    else:
        parameters = getattr(audio_tokenizer, "parameters", None)
        if callable(parameters):
            for parameter in parameters():
                parameter.requires_grad = False
    evaluate = getattr(audio_tokenizer, "eval", None)
    if callable(evaluate):
        evaluate()


@dataclass(slots=True)
class HiggsTrainingBackend:
    """Minimal Higgs runtime used by fine-tuning.

    The serving engine is deliberately not a dependency of this backend.
    In particular, there is no ``kv_caches`` field and no CUDA graph
    capture: training only needs the differentiable model plus frozen
    preprocessing components.
    """

    model: Any
    tokenizer: Any
    audio_tokenizer: Any
    collator: Any
    sample_rate: int

    def prepare_for_training(self) -> None:
        """Disable generation state and restore source training collation."""
        config = getattr(self.model, "config", None)
        for candidate in (
                config,
                getattr(config, "text_config", None),
        ):
            if candidate is not None and hasattr(candidate, "use_cache"):
                candidate.use_cache = False

        graph_runners = getattr(self.model, "decode_graph_runners", None)
        if graph_runners is not None:
            clear = getattr(graph_runners, "clear", None)
            if callable(clear):
                clear()
        if hasattr(self.model, "current_past_key_values_bucket"):
            self.model.current_past_key_values_bucket = None

        _freeze_audio_tokenizer(self.audio_tokenizer)
        self.collator.round_to = 8
        self.collator.pad_left = False
        self.collator.return_audio_in_tokens = bool(getattr(config, "encode_audio_in_tokens", False))

    def save_pretrained(self, save_directory: str | Path) -> Path:
        """Export model/tokenizer files in the native safetensors layout."""
        destination = Path(save_directory).expanduser()
        destination.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(
            destination,
            safe_serialization=True,
        )
        save_tokenizer = getattr(self.tokenizer, "save_pretrained", None)
        if callable(save_tokenizer):
            save_tokenizer(destination)
        return destination

    def build_inference_runtime(
            self,
            *,
            device: str,
            kv_cache_lengths: tuple[int, ...] = (1024, 4096, 8192),
    ) -> Any:
        """Convert this trained backend into a serving engine in place.

        Static caches are allocated only at this explicit inference
        boundary. The serving shell reuses the trained model and
        preprocessing objects; it does not reload the base checkpoint or
        retain graphs from an older model instance.
        """
        cache_module = import_optional(
            "transformers.cache_utils",
            model_type="higgstts",
            install_extra="higgstts",
        )
        serve_module = import_optional(
            "voicehub.models.higgstts.source.boson_multimodal.serve."
            "serve_engine",
            model_type="higgstts",
            install_extra="higgstts",
        )

        config = self.model.config
        for candidate in (
                config,
                getattr(config, "text_config", None),
        ):
            if candidate is not None and hasattr(candidate, "use_cache"):
                candidate.use_cache = True

        graph_runners = getattr(self.model, "decode_graph_runners", None)
        if graph_runners is not None:
            clear = getattr(graph_runners, "clear", None)
            if callable(clear):
                clear()
        if hasattr(self.model, "current_past_key_values_bucket"):
            self.model.current_past_key_values_bucket = None
        self.model.eval()

        engine_class = serve_module.HiggsAudioServeEngine
        engine = engine_class.__new__(engine_class)
        engine.device = device
        engine.model_name_or_path = getattr(config, "_name_or_path", "")
        engine.torch_dtype = self.model.dtype
        engine.model = self.model
        engine.tokenizer = self.tokenizer
        engine.audio_tokenizer = self.audio_tokenizer
        engine.audio_num_codebooks = config.audio_num_codebooks
        engine.audio_codebook_size = config.audio_codebook_size
        engine.audio_tokenizer_tps = self.audio_tokenizer.tps
        engine.samples_per_token = int(self.audio_tokenizer.sampling_rate // self.audio_tokenizer.tps)
        engine.hamming_window_len = (2 * engine.audio_num_codebooks * engine.samples_per_token)
        self.model.set_audio_special_tokens(self.tokenizer)

        cache_config = deepcopy(config.text_config)
        cache_config.num_hidden_layers = config.text_config.num_hidden_layers
        if config.audio_dual_ffn_layers:
            cache_config.num_hidden_layers += len(config.audio_dual_ffn_layers)
        engine.kv_caches = {
            length:
            cache_module.StaticCache(
                config=cache_config,
                max_batch_size=1,
                max_cache_len=length,
                device=self.model.device,
                dtype=self.model.dtype,
            )
            for length in sorted(set(kv_cache_lengths))
        }

        self.collator.round_to = 1
        self.collator.pad_left = False
        self.collator.return_audio_in_tokens = False
        engine.collator = self.collator
        if device == "cuda":
            self.model.capture_model(engine.kv_caches.values())
        return engine


def load_higgs_training_backend(
    model_name_or_path: str,
    audio_tokenizer_name_or_path: str,
    *,
    device: str,
    torch_dtype: str = "bfloat16",
) -> HiggsTrainingBackend:
    """Load Higgs safetensors without constructing its serving engine.

    ``HiggsAudioServeEngine`` preallocates static KV-cache buckets and
    captures CUDA graphs. Neither is used by teacher-forced training, so
    this loader imports the exact underlying components directly.
    """
    torch = import_optional(
        "torch",
        model_type="higgstts",
        install_extra="higgstts",
    )
    transformers = import_optional(
        "transformers",
        model_type="higgstts",
        install_extra="higgstts",
    )
    model_module = import_optional(
        "voicehub.models.higgstts.source.boson_multimodal.model."
        "higgs_audio",
        model_type="higgstts",
        install_extra="higgstts",
    )
    audio_tokenizer_module = import_optional(
        "voicehub.models.higgstts.source.boson_multimodal."
        "audio_processing.higgs_audio_tokenizer",
        model_type="higgstts",
        install_extra="higgstts",
    )
    collator_module = import_optional(
        "voicehub.models.higgstts.source.boson_multimodal."
        "data_collator.higgs_audio_collator",
        model_type="higgstts",
        install_extra="higgstts",
    )

    dtype = _resolve_training_dtype(torch, torch_dtype, device)
    model = model_module.HiggsAudioModel.from_pretrained(
        model_name_or_path,
        torch_dtype=dtype,
        use_safetensors=True,
    )
    model.to(device)
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name_or_path, )
    audio_tokenizer = (
        audio_tokenizer_module.load_higgs_audio_tokenizer(
            audio_tokenizer_name_or_path,
            device=device,
        ))
    model.set_audio_special_tokens(tokenizer)

    config = model.config
    if config.encode_whisper_embed:
        whisper_processor = transformers.AutoProcessor.from_pretrained(
            "openai/whisper-large-v3-turbo",
            trust_remote_code=True,
        )
    else:
        whisper_processor = None

    collator = collator_module.HiggsAudioSampleCollator(
        whisper_processor=whisper_processor,
        encode_whisper_embed=config.encode_whisper_embed,
        audio_in_token_id=config.audio_in_token_idx,
        audio_out_token_id=config.audio_out_token_idx,
        audio_stream_bos_id=config.audio_stream_bos_id,
        audio_stream_eos_id=config.audio_stream_eos_id,
        pad_token_id=config.pad_token_id,
        return_audio_in_tokens=bool(config.encode_audio_in_tokens),
        use_delay_pattern=config.use_delay_pattern,
        audio_num_codebooks=config.audio_num_codebooks,
        round_to=8,
        pad_left=False,
    )
    backend = HiggsTrainingBackend(
        model=model,
        tokenizer=tokenizer,
        audio_tokenizer=audio_tokenizer,
        collator=collator,
        sample_rate=int(audio_tokenizer.sampling_rate),
    )
    backend.prepare_for_training()
    return backend


class HiggsSFTDataset:
    """Convert ChatML records to the vendored ``ChatMLDatasetSample`` type.

    Records may already be prepared samples, ``ChatMLSample`` objects,
    or mappings accepted by ``prepare_chatml_sample``. Audio content is
    encoded once per item with the frozen Higgs tokenizer.
    """

    def __init__(
        self,
        records: Sequence[Any],
        *,
        tokenizer,
        audio_tokenizer,
    ):
        self.records = records
        self.tokenizer = tokenizer
        self.audio_tokenizer = audio_tokenizer

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int):
        torch = import_optional(
            "torch",
            model_type="higgstts",
            install_extra="higgstts",
        )
        dataset_module = import_optional(
            "voicehub.models.higgstts.source.boson_multimodal.dataset."
            "chatml_dataset",
            model_type="higgstts",
            install_extra="higgstts",
        )
        record = self.records[index]
        if isinstance(record, dataset_module.ChatMLDatasetSample):
            return record
        if isinstance(record, Mapping):
            prepared_fields = {item.name for item in fields(dataset_module.ChatMLDatasetSample)}
            if prepared_fields.issubset(record):
                return dataset_module.ChatMLDatasetSample(**{name: record[name] for name in prepared_fields})
            record = dict(record)

        input_tokens, label_tokens, audio_contents, _ = (
            dataset_module.prepare_chatml_sample(record, self.tokenizer))
        if input_tokens is None or label_tokens is None:
            raise ValueError(f"Higgs Audio could not prepare ChatML record {index}.")

        codes = []
        waveforms = []
        sample_rate = int(self.audio_tokenizer.sampling_rate)
        for content in audio_contents:
            waveform = self._load_audio(content, sample_rate)
            encoded = self.audio_tokenizer.encode(waveform, sample_rate)
            encoded = torch.as_tensor(encoded).squeeze(0).cpu().long()
            if encoded.ndim != 2:
                raise ValueError(
                    "Higgs audio tokenizer must return [codebooks, time] "
                    f"codes after removing the batch dimension; got "
                    f"{tuple(encoded.shape)}.")
            codes.append(encoded)
            waveforms.append(torch.as_tensor(waveform).reshape(-1).float())

        num_codebooks = int(getattr(self.audio_tokenizer, "num_codebooks", 0))
        if codes:
            audio_ids_start = torch.tensor(
                self._offsets([item.shape[-1] for item in codes]),
                dtype=torch.long,
            )
            audio_ids = torch.cat(codes, dim=-1)
        else:
            audio_ids_start = torch.empty(0, dtype=torch.long)
            audio_ids = torch.empty((num_codebooks, 0), dtype=torch.long)

        if waveforms:
            waveform_starts = torch.tensor(
                self._offsets([item.numel() for item in waveforms]),
                dtype=torch.long,
            )
            waveform = torch.cat(waveforms)
            rates = torch.full(
                (len(waveforms), ),
                sample_rate,
                dtype=torch.long,
            )
        else:
            waveform_starts = torch.empty(0, dtype=torch.long)
            waveform = torch.empty(0, dtype=torch.float32)
            rates = torch.empty(0, dtype=torch.long)

        return dataset_module.ChatMLDatasetSample(
            input_ids=torch.tensor(input_tokens, dtype=torch.long),
            label_ids=torch.tensor(label_tokens, dtype=torch.long),
            audio_ids_concat=audio_ids,
            audio_ids_start=audio_ids_start,
            audio_waveforms_concat=waveform,
            audio_waveforms_start=waveform_starts,
            audio_sample_rate=rates,
            audio_speaker_indices=torch.full(
                (len(codes), ),
                -1,
                dtype=torch.long,
            ),
            audio_label_ids_concat=audio_ids.clone(),
        )

    @staticmethod
    def _offsets(lengths: list[int]) -> list[int]:
        offsets = []
        current = 0
        for length in lengths:
            offsets.append(current)
            current += int(length)
        return offsets

    @staticmethod
    def _load_audio(content, sample_rate: int):
        librosa = import_optional(
            "librosa",
            model_type="higgstts",
            install_extra="higgstts",
        )
        if getattr(content, "raw_audio", None):
            payload = str(content.raw_audio)
            if "," in payload and payload.lstrip().startswith("data:"):
                payload = payload.split(",", 1)[1]
            waveform, _ = librosa.load(
                BytesIO(base64.b64decode(payload)),
                sr=sample_rate,
                mono=True,
            )
            return waveform
        audio_url = str(getattr(content, "audio_url", ""))
        if not audio_url or audio_url == "placeholder":
            raise ValueError("Every Higgs training audio content requires audio_url or "
                             "raw_audio.")
        waveform, _ = librosa.load(
            audio_url,
            sr=sample_rate,
            mono=True,
        )
        return waveform


class _HiggsCollator:
    """Lazily expose the loaded training backend's source collator."""

    def __init__(self, adapter: HiggsTrainingAdapter):
        self.adapter = adapter

    def __call__(self, features):
        self.adapter.setup()
        output = self.adapter.model.model.collator(features)
        return {item.name: getattr(output, item.name) for item in fields(output)}

    def resume_fingerprint(self) -> dict[str, Any]:
        config = getattr(self.adapter.model, "config", None)
        return {
            "round_to": 8,
            "pad_left": False,
            "return_audio_in_tokens": bool(getattr(config, "encode_audio_in_tokens", False)),
            "base_model": getattr(config, "name_or_path", None),
            "audio_tokenizer": getattr(
                config,
                "audio_tokenizer_name_or_path",
                None,
            ),
        }


class HiggsTrainingAdapter(CausalLMTrainingAdapter):
    """Reconstructed joint text/audio objective for Higgs safetensors.

    Boson AI has not published its training loop. This recipe follows
    the released model and collator contracts, but is intentionally
    identified as reconstructed rather than author-verified.
    """

    supports_custom_recipe = True

    def __init__(self, model, spec):
        super().__init__(model, spec)
        self.data_collator = _HiggsCollator(self)

    def setup(self):
        backend = getattr(self.model, "_training_backend", None)
        if (backend is not None and getattr(self.model, "model", None) is not backend):
            self.model._prepare_for_training()
        super().setup()
        runtime = self.model.model
        tokenizer = getattr(runtime, "audio_tokenizer", None)
        if tokenizer is not None:
            if hasattr(tokenizer, "eval"):
                tokenizer.eval()
            if hasattr(tokenizer, "parameters"):
                for parameter in tokenizer.parameters():
                    parameter.requires_grad_(False)
        collator = getattr(runtime, "collator", None)
        if collator is not None:
            collator.round_to = 8
            collator.pad_left = False
            collator.return_audio_in_tokens = bool(
                getattr(self.primary_model.config, "encode_audio_in_tokens", False))
        return self

    def create_dataset(self, records, **kwargs):
        if kwargs:
            unexpected = ", ".join(sorted(kwargs))
            raise TypeError(f"Unexpected Higgs dataset option(s): {unexpected}.")
        self.setup()
        runtime = self.model.model
        return HiggsSFTDataset(
            records,
            tokenizer=runtime.tokenizer,
            audio_tokenizer=runtime.audio_tokenizer,
        )

    def execute_training_phase(
        self,
        context: TrainingContext,
    ) -> TTSTrainingOutput:
        self.setup()
        batch = dict(context.inputs)
        required = ("input_ids", "attention_mask", "label_ids", "label_audio_ids")
        missing = [name for name in required if name not in batch]
        if missing:
            raise ValueError("Higgs Audio fine-tuning requires source-collated fields: " + ", ".join(missing))
        allowed = {
            "input_ids",
            "attention_mask",
            "audio_features",
            "audio_feature_attention_mask",
            "audio_in_ids",
            "audio_in_ids_start",
            "audio_out_ids",
            "audio_out_ids_start",
            "audio_out_ids_start_group_loc",
            "label_ids",
            "label_audio_ids",
            "reward",
        }
        outputs = self.primary_model(
            **{
                name: value
                for name, value in batch.items() if name in allowed
            },
            use_cache=False,
            return_dict=True,
        )
        losses = self.compute_causal_losses(
            outputs,
            batch["label_audio_ids"],
        )
        text_weight = float(getattr(self.model.config, "training_text_loss_weight", 1.0))
        audio_weight = float(getattr(self.model.config, "training_audio_loss_weight", 1.0))
        if (not math.isfinite(text_weight) or not math.isfinite(audio_weight) or text_weight < 0 or
                audio_weight < 0):
            raise ValueError("Higgs training loss weights must be finite and "
                             "non-negative.")
        weighted = []
        if "text_loss" in losses and text_weight:
            weighted.append(text_weight * losses["text_loss"])
        if "audio_loss" in losses and audio_weight:
            weighted.append(audio_weight * losses["audio_loss"])
        if not weighted:
            raise ValueError("Higgs Audio received no supervised text or audio tokens.")
        loss = sum(weighted)
        return TTSTrainingOutput(
            loss=loss,
            logits=(outputs.logits, outputs.audio_logits),
            losses={
                "loss": loss,
                **losses,
            },
            metadata={
                "model_type": self.model_type,
                "training_family": self.spec.family_name,
                "training_support": self.spec.support.value,
                "training_phase": context.phase.name,
                "optimizer_names": context.phase.optimizer_names,
                "source_native_recipe": False,
                "recipe_provenance": "reconstructed-from-forward-contract",
            },
            training_phase=context.phase.name,
            optimizer_names=context.phase.optimizer_names,
        )

    def compute_causal_losses(self, outputs, label_audio_ids):
        """Return token-normalized, causally shifted text/audio losses."""
        torch = import_optional(
            "torch",
            model_type="higgstts",
            install_extra="higgstts",
        )
        functional = torch.nn.functional
        losses: dict[str, Any] = {}

        labels = outputs.expanded_labels
        if labels is not None and labels.shape[-1] > 1:
            text_targets = labels[:, 1:].reshape(-1)
            valid = text_targets.ne(-100)
            if valid.any():
                text_logits = outputs.logits[:, :-1].reshape(-1, outputs.logits.shape[-1])
                losses["text_loss"] = functional.cross_entropy(
                    text_logits.float(),
                    text_targets,
                    ignore_index=-100,
                    reduction="sum",
                ) / valid.sum()

        audio_logits = outputs.audio_logits
        audio_labels = label_audio_ids
        if (audio_logits is not None and audio_labels is not None and audio_logits.shape[0] > 1 and
                audio_labels.shape[-1] > 1):
            if audio_logits.shape[1] != audio_labels.shape[0]:
                raise ValueError(
                    "Higgs audio logits/codebook labels disagree: "
                    f"{audio_logits.shape[1]} != {audio_labels.shape[0]}.")
            length = min(audio_logits.shape[0], audio_labels.shape[-1])
            codebook_weights = torch.as_tensor(
                self.primary_model.audio_codebook_weights,
                dtype=torch.float32,
                device=audio_logits.device,
            )
            if codebook_weights.numel() != audio_logits.shape[1]:
                raise ValueError(
                    "Higgs audio_codebook_weights must contain one value per "
                    "audio codebook.")
            weight_sum = codebook_weights.sum()
            if (not torch.isfinite(codebook_weights).all() or (codebook_weights < 0).any() or
                    weight_sum.item() <= 0):
                raise ValueError(
                    "Higgs audio_codebook_weights must be finite, "
                    "non-negative, and sum to a positive value.")
            codebook_weights = codebook_weights / weight_sum
            audio_loss = torch.zeros(
                (),
                dtype=torch.float32,
                device=audio_logits.device,
            )
            active_weight = torch.zeros_like(audio_loss)
            for codebook in range(audio_logits.shape[1]):
                targets = audio_labels[codebook, 1:length]
                valid = targets.ne(-100)
                if not valid.any():
                    continue
                codebook_loss = functional.cross_entropy(
                    audio_logits[:length - 1, codebook].float(),
                    targets,
                    ignore_index=-100,
                    reduction="sum",
                ) / valid.sum()
                losses[f"audio_loss_{codebook}"] = codebook_loss
                audio_loss = (audio_loss + codebook_weights[codebook] * codebook_loss)
                active_weight = active_weight + codebook_weights[codebook]
            if active_weight.item() > 0:
                losses["audio_loss"] = audio_loss / active_weight
        return losses

    def save_pretrained(self, save_directory) -> None:
        self.setup()
        backend = getattr(self.model, "training_backend", None)
        if backend is not None:
            backend.save_pretrained(save_directory)
            return
        destination = Path(save_directory)
        self.primary_model.save_pretrained(
            destination,
            safe_serialization=True,
        )
        tokenizer = getattr(self.model.model, "tokenizer", None)
        if tokenizer is not None and hasattr(tokenizer, "save_pretrained"):
            tokenizer.save_pretrained(destination)


__all__ = [
    "HiggsSFTDataset",
    "HiggsTrainingAdapter",
    "HiggsTrainingBackend",
    "load_higgs_training_backend",
]

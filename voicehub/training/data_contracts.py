"""Inspectable source-data contracts for TTS training."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, replace
from enum import Enum
from types import MappingProxyType
from typing import Any

from voicehub.dependencies import resolve_import_path


def _field_names(values: Iterable[str], *, owner: str) -> tuple[str, ...]:
    if isinstance(values, str):
        values = (values, )
    normalized = []
    for value in values:
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{owner} must contain non-empty field names.")
        normalized.append(value.strip())
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"{owner} must not contain duplicate field names.")
    return tuple(normalized)


def _field_aliases(values: Mapping[str, str] | Iterable[tuple[str, str]], ) -> tuple[tuple[str, str], ...]:
    items = tuple(values.items()) if isinstance(values, Mapping) else tuple(values)
    normalized = []
    seen = set()
    for item in items:
        if not isinstance(item, (tuple, list)) or len(item) != 2:
            raise TypeError("TTS field_aliases entries must be source/target pairs.")
        source, target = item
        if not isinstance(source, str) or not source.strip():
            raise ValueError("TTS field alias sources must be non-empty strings.")
        if not isinstance(target, str) or not target.strip():
            raise ValueError("TTS field alias targets must be non-empty strings.")
        source = source.strip()
        target = target.strip()
        if source in seen:
            raise ValueError(f"TTS field_aliases repeats source {source!r}.")
        seen.add(source)
        normalized.append((source, target))
    return tuple(normalized)


def _field_present(record: Mapping[str, Any], name: str) -> bool:
    if name not in record or record[name] is None:
        return False
    value = record[name]
    return not isinstance(value, (str, bytes, bytearray)) or bool(value.strip())


class TTSDataArchitecture(str, Enum):
    """Canonical source-data layouts used by TTS training recipes."""

    CODEC_LM = "codec-lm"
    SEQUENCE_TO_SEQUENCE = "sequence-to-sequence"
    DIFFUSION = "diffusion"
    VITS = "vits"
    ACOUSTIC = "acoustic"
    HYBRID = "hybrid"

    def __str__(self) -> str:
        return self.value

    @classmethod
    def coerce(cls, value: TTSDataArchitecture | str) -> TTSDataArchitecture:
        if isinstance(value, cls):
            return value
        if not isinstance(value, str) or not value.strip():
            raise TypeError("TTS data architecture must be a non-empty string or enum value.")
        normalized = value.strip().lower().replace("_", "-")
        aliases = {
            "causal-lm": cls.CODEC_LM.value,
            "llm": cls.CODEC_LM.value,
            "seq2seq": cls.SEQUENCE_TO_SEQUENCE.value,
            "flow": cls.DIFFUSION.value,
            "flow-matching": cls.DIFFUSION.value,
            "gan": cls.VITS.value,
            "acoustic-regression": cls.ACOUSTIC.value,
            "composite": cls.HYBRID.value,
        }
        normalized = aliases.get(normalized, normalized)
        try:
            return cls(normalized)
        except ValueError as exc:
            choices = ", ".join(member.value for member in cls)
            raise ValueError(f"Unknown TTS data architecture {value!r}. Expected one of: {choices}.") from exc


class TTSDataReadiness(str, Enum):
    """How far VoiceHub owns a model's dataset preparation path."""

    INTEGRATED = "integrated-raw"
    PREPROCESSED = "preprocessed"
    CUSTOM = "custom"
    UNAVAILABLE = "unavailable"

    def __str__(self) -> str:
        return self.value

    @classmethod
    def coerce(cls, value: TTSDataReadiness | str) -> TTSDataReadiness:
        if isinstance(value, cls):
            return value
        if not isinstance(value, str) or not value.strip():
            raise TypeError("TTS data readiness must be a non-empty string or enum value.")
        normalized = value.strip().lower().replace("_", "-")
        try:
            return cls(normalized)
        except ValueError as exc:
            choices = ", ".join(member.value for member in cls)
            raise ValueError(
                f"Unknown TTS data readiness {value!r}. Expected one of: "
                f"{choices}.") from exc


@dataclass(frozen=True)
class TTSRecordVariant:
    """One accepted source or preprocessed record shape.

    Every field in ``required_fields`` must be present.  Each tuple in
    ``one_of`` describes an alternative group for which at least one
    field must be present. ``at_most_one_of`` rejects ambiguous aliases,
    ``forbidden_fields`` excludes incompatible source forms, and the
    conditional requirement fields express dependent metadata. Values
    that are ``None`` or empty strings count as absent.
    """

    name: str
    required_fields: tuple[str, ...] = ()
    one_of: tuple[tuple[str, ...], ...] = ()
    at_most_one_of: tuple[tuple[str, ...], ...] = ()
    forbidden_fields: tuple[str, ...] = ()
    requires: tuple[tuple[str, tuple[str, ...]], ...] = ()
    requires_one_of: tuple[tuple[str, tuple[str, ...]], ...] = ()
    description: str = ""
    preprocessed: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("TTS record variant names must be non-empty strings.")
        object.__setattr__(self, "name", self.name.strip())
        object.__setattr__(
            self,
            "required_fields",
            _field_names(self.required_fields, owner="required_fields"),
        )
        groups = []
        for group in self.one_of:
            normalized = _field_names(group, owner="one_of")
            if not normalized:
                raise ValueError("TTS record one_of groups cannot be empty.")
            groups.append(normalized)
        object.__setattr__(self, "one_of", tuple(groups))
        exclusive_groups = []
        for group in self.at_most_one_of:
            normalized = _field_names(group, owner="at_most_one_of")
            if len(normalized) < 2:
                raise ValueError("TTS record at_most_one_of groups require at least two fields.")
            exclusive_groups.append(normalized)
        object.__setattr__(
            self,
            "at_most_one_of",
            tuple(exclusive_groups),
        )
        object.__setattr__(
            self,
            "forbidden_fields",
            _field_names(
                self.forbidden_fields,
                owner="forbidden_fields",
            ),
        )
        object.__setattr__(
            self,
            "requires",
            self._normalize_dependencies(
                self.requires,
                owner="requires",
            ),
        )
        object.__setattr__(
            self,
            "requires_one_of",
            self._normalize_dependencies(
                self.requires_one_of,
                owner="requires_one_of",
            ),
        )
        if not isinstance(self.description, str):
            raise TypeError("TTS record variant descriptions must be strings.")
        if not isinstance(self.preprocessed, bool):
            raise TypeError("TTS record variant preprocessed must be a boolean.")

    @staticmethod
    def _normalize_dependencies(
        values: Iterable[tuple[str, Iterable[str]]],
        *,
        owner: str,
    ) -> tuple[tuple[str, tuple[str, ...]], ...]:
        normalized = []
        for value in values:
            if (not isinstance(value, (tuple, list)) or len(value) != 2):
                raise TypeError(f"{owner} entries must be (trigger, required_fields) pairs.")
            trigger, required = value
            trigger_fields = _field_names((trigger, ), owner=f"{owner} trigger")
            dependencies = _field_names(required, owner=owner)
            if not dependencies:
                raise ValueError(f"{owner} dependency groups cannot be empty.")
            normalized.append((trigger_fields[0], dependencies))
        triggers = [trigger for trigger, _ in normalized]
        if len(set(triggers)) != len(triggers):
            raise ValueError(f"{owner} must not repeat trigger fields.")
        return tuple(normalized)

    def missing(self, record: Mapping[str, Any]) -> tuple[str, ...]:
        """Return human-readable contract issues found in ``record``."""
        missing = [name for name in self.required_fields if not _field_present(record, name)]
        for group in self.one_of:
            if not any(_field_present(record, name) for name in group):
                missing.append("(" + " or ".join(group) + ")")
        for group in self.at_most_one_of:
            present = [name for name in group if _field_present(record, name)]
            if len(present) > 1:
                missing.append("at most one of (" + ", ".join(group) + ")")
        for name in self.forbidden_fields:
            if _field_present(record, name):
                missing.append(f"forbidden field {name}")
        for trigger, required in self.requires:
            if not _field_present(record, trigger):
                continue
            absent = [name for name in required if not _field_present(record, name)]
            if absent:
                missing.append(f"{trigger} requires " + ", ".join(absent))
        for trigger, alternatives in self.requires_one_of:
            if (_field_present(record, trigger) and
                    not any(_field_present(record, name) for name in alternatives)):
                missing.append(f"{trigger} requires one of (" + " or ".join(alternatives) + ")")
        return tuple(missing)

    def matches(self, record: Mapping[str, Any]) -> bool:
        return not self.missing(record)


@dataclass(frozen=True)
class TTSDatasetSpec:
    """Inspectable source-data contract for one TTS architecture or model."""

    architecture: TTSDataArchitecture
    variants: tuple[TTSRecordVariant, ...]
    model_type: str | None = None
    sample_rate: int | None = None
    description: str = ""
    readiness: TTSDataReadiness | None = None
    training_support: str | None = None
    field_aliases: Mapping[str, str] | tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "architecture",
            TTSDataArchitecture.coerce(self.architecture),
        )
        variants = tuple(self.variants)
        if not variants or any(not isinstance(variant, TTSRecordVariant) for variant in variants):
            raise ValueError("TTSDatasetSpec.variants must contain TTSRecordVariant values.")
        names = [variant.name for variant in variants]
        if len(set(names)) != len(names):
            raise ValueError("TTS record variant names must be unique within a dataset spec.")
        object.__setattr__(self, "variants", variants)
        if self.model_type is not None:
            if not isinstance(self.model_type, str) or not self.model_type.strip():
                raise ValueError("TTSDatasetSpec.model_type must be a non-empty string or None.")
            object.__setattr__(self, "model_type", self.model_type.strip().lower())
        if self.sample_rate is not None:
            if isinstance(self.sample_rate, bool) or not isinstance(self.sample_rate, int):
                raise TypeError("TTSDatasetSpec.sample_rate must be an integer or None.")
            if self.sample_rate <= 0:
                raise ValueError("TTSDatasetSpec.sample_rate must be positive.")
        if not isinstance(self.description, str):
            raise TypeError("TTSDatasetSpec.description must be a string.")
        if self.readiness is not None:
            object.__setattr__(
                self,
                "readiness",
                TTSDataReadiness.coerce(self.readiness),
            )
        if self.training_support is not None:
            if (not isinstance(self.training_support, str) or not self.training_support.strip()):
                raise ValueError("TTSDatasetSpec.training_support must be a non-empty "
                                 "string or None.")
            object.__setattr__(
                self,
                "training_support",
                self.training_support.strip().lower().replace("_", "-"),
            )
        object.__setattr__(
            self,
            "field_aliases",
            _field_aliases(self.field_aliases),
        )

    def match_variant(self, record: Mapping[str, Any], *, index: int | None = None) -> str:
        """Validate a record and return the matching variant name."""
        if not isinstance(record, Mapping):
            location = "" if index is None else f" {index}"
            raise TypeError(f"TTS record{location} must be a mapping, received "
                            f"{type(record).__name__}.")
        for variant in self.variants:
            if variant.matches(record):
                return variant.name
        location = "" if index is None else f" {index}"
        alternatives = "; ".join(
            f"{variant.name}: {', '.join(variant.missing(record)) or 'valid'}" for variant in self.variants)
        target = f" for {self.model_type!r}" if self.model_type else ""
        raise ValueError(
            f"TTS record{location} does not match any {self.architecture.value} "
            f"dataset variant{target}. Missing requirements — {alternatives}.")

    @property
    def raw_variants(self) -> tuple[TTSRecordVariant, ...]:
        return tuple(variant for variant in self.variants if not variant.preprocessed)

    @property
    def preprocessed_variants(self) -> tuple[TTSRecordVariant, ...]:
        return tuple(variant for variant in self.variants if variant.preprocessed)

    @property
    def accepts_raw_records(self) -> bool:
        """Whether a model contract includes an integrated source record."""
        return bool(self.raw_variants)

    @property
    def requires_preprocessing(self) -> bool:
        return self.readiness is TTSDataReadiness.PREPROCESSED


def _variant(
    name: str,
    *,
    required: tuple[str, ...] = (),
    one_of: tuple[tuple[str, ...], ...] = (),
    at_most_one_of: tuple[tuple[str, ...], ...] = (),
    forbidden: tuple[str, ...] = (),
    requires: tuple[tuple[str, tuple[str, ...]], ...] = (),
    requires_one_of: tuple[tuple[str, tuple[str, ...]], ...] = (),
    description: str = "",
    preprocessed: bool = False,
) -> TTSRecordVariant:
    return TTSRecordVariant(
        name=name,
        required_fields=required,
        one_of=one_of,
        at_most_one_of=at_most_one_of,
        forbidden_fields=forbidden,
        requires=requires,
        requires_one_of=requires_one_of,
        description=description,
        preprocessed=preprocessed,
    )


_ARCHITECTURE_SPECS: Mapping[TTSDataArchitecture, TTSDatasetSpec] = MappingProxyType({
    TTSDataArchitecture.CODEC_LM:
    TTSDatasetSpec(
        architecture=TTSDataArchitecture.CODEC_LM,
        description="Autoregressive text/audio-token or codec-language-model data.",
        variants=(
            _variant(
                "raw-audio",
                required=("text", ),
                one_of=(("audio", "audio_codes"), ),
                description="Text paired with audio or frozen-codec tokens.",
            ),
            _variant(
                "conversation",
                one_of=(("conversation", "messages"), ),
                description="Structured multi-turn text/audio conversation.",
            ),
            _variant(
                "tokenized",
                required=("input_ids", "labels"),
                description="Model-ready causal token sequence.",
                preprocessed=True,
            ),
            _variant(
                "model-ready",
                required=("model_inputs", ),
                one_of=(("labels", "targets", "target"), ),
                description="Backend-shaped tensors in the model_inputs namespace.",
                preprocessed=True,
            ),
        ),
    ),
    TTSDataArchitecture.SEQUENCE_TO_SEQUENCE:
    TTSDatasetSpec(
        architecture=TTSDataArchitecture.SEQUENCE_TO_SEQUENCE,
        description="Encoder text plus teacher-forced acoustic or codec targets.",
        variants=(
            _variant(
                "raw-audio",
                required=("text", "audio"),
                description="Text/audio pair processed by the selected model.",
            ),
            _variant(
                "processor-ready",
                required=("input_ids", "labels"),
                description="Processor-produced encoder inputs and decoder labels.",
                preprocessed=True,
            ),
            _variant(
                "model-ready",
                required=("model_inputs", ),
                one_of=(("labels", "targets", "target"), ),
                preprocessed=True,
            ),
        ),
    ),
    TTSDataArchitecture.DIFFUSION:
    TTSDatasetSpec(
        architecture=TTSDataArchitecture.DIFFUSION,
        description="Conditional flow-matching, rectified-flow, or diffusion data.",
        variants=(
            _variant(
                "raw-audio",
                required=("text", "audio"),
                description="Source pair for a model-owned acoustic preprocessor.",
            ),
            _variant(
                "acoustic-features",
                one_of=(("text", "input_ids", "text_tokens"), ),
                description="Conditioning plus model-specific clean acoustic features.",
                preprocessed=True,
                required=("target_latents", ),
            ),
            _variant(
                "model-ready",
                required=("model_inputs", ),
                one_of=(("labels", "velocity_target", "noise_target", "target"), ),
                description="Noisy state, time, conditioning, mask, and source target.",
                preprocessed=True,
            ),
        ),
    ),
    TTSDataArchitecture.VITS:
    TTSDatasetSpec(
        architecture=TTSDataArchitecture.VITS,
        description="VITS/GAN text, waveform, spectrogram, and adversarial data.",
        variants=(
            _variant(
                "raw-audio",
                one_of=(("text", "phonemes", "input_ids"), ("audio", "audio_values")),
                description="Text or phonemes paired with a waveform.",
            ),
            _variant(
                "acoustic-features",
                one_of=(("input_ids", "phoneme_ids"), ("spectrogram", "mel_spec"), ("audio", "audio_values")),
                description="Prepared VITS text, spectrogram, and waveform values.",
                preprocessed=True,
            ),
            _variant(
                "phase-ready",
                required=("model_inputs", "training_phase"),
                one_of=(("labels", "target", "audio_values"), ),
                description="Generator/discriminator-specific source batch.",
                preprocessed=True,
            ),
        ),
    ),
    TTSDataArchitecture.ACOUSTIC:
    TTSDatasetSpec(
        architecture=TTSDataArchitecture.ACOUSTIC,
        description="Direct acoustic, mel, codec, or waveform regression data.",
        variants=(
            _variant(
                "raw-audio",
                required=("text", "audio"),
            ),
            _variant(
                "model-ready",
                one_of=(("input_ids", "model_inputs"), ("labels", "target", "audio_values")),
                preprocessed=True,
            ),
        ),
    ),
    TTSDataArchitecture.HYBRID:
    TTSDatasetSpec(
        architecture=TTSDataArchitecture.HYBRID,
        description="Multi-component language-model, diffusion, acoustic, or GAN data.",
        variants=(
            _variant(
                "raw-audio",
                required=("text", "audio"),
            ),
            _variant(
                "conversation",
                one_of=(("conversation", "messages"), ),
            ),
            _variant(
                "model-ready",
                required=("model_inputs", ),
                one_of=(("labels", "targets", "target", "training_phase"), ),
                preprocessed=True,
            ),
            _variant(
                "tokenized",
                required=("input_ids", "labels"),
                preprocessed=True,
            ),
        ),
    ),
})


def _model_spec(
    architecture: TTSDataArchitecture,
    values: Mapping[str, Any],
) -> TTSDatasetSpec:
    base = _ARCHITECTURE_SPECS[architecture]
    return TTSDatasetSpec(
        architecture=architecture,
        variants=tuple(values.get("variants", base.preprocessed_variants)),
        sample_rate=values.get("sample_rate", base.sample_rate),
        description=str(values.get("description", base.description)),
        readiness=values.get("readiness", base.readiness),
        field_aliases=tuple(values.get("field_aliases", base.field_aliases)),
    )


def build_orpheustts_dataset_spec() -> TTSDatasetSpec:
    """Return the OrpheusTTS source-data contract."""
    return _model_spec(
        TTSDataArchitecture.CODEC_LM, {
            "sample_rate":
            24_000,
            "variants": (
                _variant("raw-audio", required=("text", ), one_of=(("audio", "audio_codes"), )),
                _ARCHITECTURE_SPECS[TTSDataArchitecture.CODEC_LM].variants[2],
            ),
        })


def build_dia_dataset_spec() -> TTSDatasetSpec:
    """Return the Dia source-data contract."""
    return _model_spec(
        TTSDataArchitecture.SEQUENCE_TO_SEQUENCE, {
            "sample_rate":
            44_100,
            "variants": (
                _variant("raw-audio", required=("text", "audio")),
                _variant(
                    "processor-ready",
                    required=(
                        "input_ids",
                        "attention_mask",
                        "decoder_input_ids",
                        "decoder_attention_mask",
                        "labels",
                    ),
                    preprocessed=True,
                ),
            ),
        })


def build_chatterbox_dataset_spec() -> TTSDatasetSpec:
    """Return the Chatterbox source-data contract."""
    return _model_spec(
        TTSDataArchitecture.HYBRID, {
            "sample_rate":
            24_000,
            "variants": (
                _variant(
                    "t3-raw",
                    required=("text", ),
                    one_of=(("audio", "audio_path"), ),
                    description="Text plus source audio for the native T3 objective.",
                ),
                _variant(
                    "flow-raw",
                    one_of=(("audio", "audio_path"), ),
                    description="Source audio for the native flow objective.",
                ),
                _variant(
                    "t3-precomputed",
                    required=(
                        "text_tokens",
                        "speech_tokens",
                        "speaker_emb",
                    ),
                    preprocessed=True,
                ),
                _variant(
                    "flow-precomputed",
                    required=(
                        "speech_token",
                        "speech_feat",
                        "embedding",
                    ),
                    preprocessed=True,
                ),
            ),
        })


def build_kokoro_dataset_spec() -> TTSDatasetSpec:
    """Return the Kokoro source-data contract."""
    return _model_spec(
        TTSDataArchitecture.ACOUSTIC, {
            "sample_rate":
            24_000,
            "variants": (
                _variant(
                    "full-preprocessed",
                    required=("durations", ),
                    one_of=(
                        ("input_ids", "phonemes"),
                        ("ref_s", "voice"),
                        ("audio_values", "audio", "labels"),
                    ),
                    description=(
                        "Phoneme IDs, prepared style, integer durations, and "
                        "waveform targets for the alternating duration/acoustic "
                        "recipe."),
                    preprocessed=True,
                ),
                _variant(
                    "duration-only",
                    required=("durations", "training_phase"),
                    one_of=(
                        ("input_ids", "phonemes"),
                        ("ref_s", "voice"),
                    ),
                    description="Explicit duration-only phase supervision.",
                    preprocessed=True,
                ),
            ),
        })


def build_vui_dataset_spec() -> TTSDatasetSpec:
    """Return the VUI source-data contract."""
    return _model_spec(
        TTSDataArchitecture.CODEC_LM, {
            "sample_rate":
            44_100,
            "variants": (
                _variant(
                    "codec-batch",
                    required=("input_ids", "audio_codes"),
                    description=(
                        "Text tokens and codebook-first Fluac targets; text masks "
                        "and audio-code lengths are optional."),
                    preprocessed=True,
                ), ),
        })


def build_echo_dataset_spec() -> TTSDatasetSpec:
    """Return the Echo source-data contract."""
    return _model_spec(
        TTSDataArchitecture.DIFFUSION, {
            "variants": (
                _variant(
                    "flow-batch",
                    required=(
                        "target_latents",
                        "text_input_ids",
                        "text_mask",
                        "speaker_latents",
                        "speaker_mask",
                    ),
                    description="Frozen-codec latents and aligned conditioning.",
                    preprocessed=True,
                ), ),
        })


def build_conversationtts_dataset_spec() -> TTSDatasetSpec:
    """Return the ConversationTTS source-data contract."""
    return _model_spec(
        TTSDataArchitecture.CODEC_LM, {
            "sample_rate":
            24_000,
            "variants": (
                _variant(
                    "raw-text-audio",
                    one_of=(
                        ("text", "texts"),
                        ("audio", "audio_values"),
                    ),
                    at_most_one_of=(
                        ("text", "texts"),
                        ("audio", "audio_values"),
                    ),
                    forbidden=(
                        "text_token_ids",
                        "text_ids",
                        "audio_codes",
                        "codes",
                    ),
                    description="Text plus raw waveform for frozen-Mimi encoding.",
                ),
                _variant(
                    "raw-text-code",
                    one_of=(
                        ("text", "texts"),
                        ("audio_codes", "codes"),
                    ),
                    at_most_one_of=(
                        ("text", "texts"),
                        ("audio_codes", "codes"),
                    ),
                    forbidden=(
                        "text_token_ids",
                        "text_ids",
                        "audio",
                        "audio_values",
                    ),
                    preprocessed=True,
                ),
                _variant(
                    "tokenized-text-audio",
                    one_of=(
                        ("text_token_ids", "text_ids"),
                        ("audio", "audio_values"),
                    ),
                    at_most_one_of=(
                        ("text_token_ids", "text_ids"),
                        ("audio", "audio_values"),
                    ),
                    forbidden=("text", "texts", "audio_codes", "codes"),
                    preprocessed=True,
                ),
                _variant(
                    "tokenized-text-code",
                    one_of=(
                        ("text_token_ids", "text_ids"),
                        ("audio_codes", "codes"),
                    ),
                    at_most_one_of=(
                        ("text_token_ids", "text_ids"),
                        ("audio_codes", "codes"),
                    ),
                    forbidden=("text", "texts", "audio", "audio_values"),
                    preprocessed=True,
                ),
                _variant(
                    "multi-codebook-batch",
                    required=("tokens", "labels", "tokens_mask"),
                    description="Aligned token/codebook tensors and validity mask.",
                    preprocessed=True,
                ),
            ),
        })


def build_cosyvoice_dataset_spec() -> TTSDatasetSpec:
    """Return the CosyVoice source-data contract."""
    return _model_spec(
        TTSDataArchitecture.HYBRID,
        {
            "sample_rate":
            24_000,
            "readiness":
            TTSDataReadiness.INTEGRATED,
            # An identity alias preserves the model-canonical path field instead
            # of applying the shared ``audio_path`` -> ``audio`` spelling.
            "field_aliases": (("audio_path", "audio_path"), ),
            "variants": (
                _variant(
                    "llm-raw-audio",
                    required=("text", ),
                    one_of=((
                        "speech_audio",
                        "audio",
                        "waveform",
                        "audio_path",
                    ), ),
                    at_most_one_of=((
                        "speech_audio",
                        "audio",
                        "waveform",
                        "audio_path",
                    ), ),
                    forbidden=("speech_tokens", ),
                    requires_one_of=(
                        (
                            "speech_audio",
                            (
                                "speech_sampling_rate",
                                "sampling_rate",
                                "sample_rate",
                            ),
                        ),
                        (
                            "audio",
                            (
                                "speech_sampling_rate",
                                "sampling_rate",
                                "sample_rate",
                            ),
                        ),
                        (
                            "waveform",
                            (
                                "speech_sampling_rate",
                                "sampling_rate",
                                "sample_rate",
                            ),
                        ),
                    ),
                    description=(
                        "Text plus raw audio for the frozen native S3Tokenizer. "
                        "Tensor-like audio declares its source rate; PCM WAVE "
                        "paths use `audio_path` and carry their own rate."),
                ),
                _variant(
                    "llm-record",
                    required=("text", "speech_tokens"),
                    forbidden=(
                        "speech_audio",
                        "audio",
                        "waveform",
                        "audio_path",
                    ),
                    description=(
                        "Text plus precomputed speech tokens for the native "
                        "CosyVoice language-model dataset."),
                    preprocessed=True,
                ),
            ),
        })


def build_llasa_dataset_spec() -> TTSDatasetSpec:
    """Return the LLaSA source-data contract."""
    return _model_spec(
        TTSDataArchitecture.CODEC_LM, {
            "sample_rate":
            16_000,
            "variants": (
                _variant("raw-audio", required=("text", ), one_of=(("audio", "audio_codes"), )),
                _ARCHITECTURE_SPECS[TTSDataArchitecture.CODEC_LM].variants[2],
            ),
        })


def build_f5tts_dataset_spec() -> TTSDatasetSpec:
    """Return the F5-TTS source-data contract."""
    return _model_spec(
        TTSDataArchitecture.DIFFUSION, {
            "sample_rate":
            24_000,
            "variants": (
                _variant(
                    "waveform-vocab",
                    required=("input_values", "input_ids"),
                    description=(
                        "Waveform values plus caller-supplied vocabulary IDs; "
                        "sequence lengths are optional."),
                    preprocessed=True,
                ),
                _variant(
                    "mel-features",
                    required=("input_ids", ),
                    one_of=(("mel", "mel_spec"), ),
                    description=(
                        "Prepared 100-bin mel features plus vocabulary IDs; "
                        "sequence lengths are optional."),
                    preprocessed=True,
                ),
                _variant(
                    "native-ready",
                    required=("inp", "text"),
                    description="Native waveform/mel input and encoded text conditioning.",
                    preprocessed=True,
                ),
            ),
        })


def build_gptsovits_dataset_spec() -> TTSDatasetSpec:
    """Return the GPT-SoVITS source-data contract."""
    return _model_spec(
        TTSDataArchitecture.HYBRID, {
            "sample_rate":
            32_000,
            "variants": (
                _variant(
                    "s1-preprocessed",
                    required=(
                        "phoneme_ids",
                        "semantic_ids",
                        "bert_features",
                    ),
                    preprocessed=True,
                ),
                _variant(
                    "s2-preprocessed",
                    required=(
                        "ssl_features",
                        "spectrogram",
                        "audio_values",
                        "phoneme_ids",
                    ),
                    preprocessed=True,
                ),
                _variant(
                    "s2-pro-preprocessed",
                    required=(
                        "ssl_features",
                        "spectrogram",
                        "audio_values",
                        "phoneme_ids",
                        "speaker_embedding",
                    ),
                    preprocessed=True,
                ),
            ),
        })


def build_melotts_dataset_spec() -> TTSDatasetSpec:
    """Return the MeloTTS source-data contract."""
    return _model_spec(
        TTSDataArchitecture.VITS, {
            "sample_rate":
            44_100,
            "variants": (
                _variant(
                    "explicit-features",
                    required=(
                        "input_ids",
                        "tone_ids",
                        "language_ids",
                        "bert_features",
                        "ja_bert_features",
                        "spectrogram",
                        "audio_values",
                        "speaker_id",
                    ),
                    description=(
                        "Exact phone, tone, language, BERT, spectrogram, waveform, "
                        "and speaker supervision consumed by the native collator."),
                    preprocessed=True,
                ), ),
        })


def build_openvoice_dataset_spec() -> TTSDatasetSpec:
    """Return the OpenVoice source-data contract."""
    return _model_spec(
        TTSDataArchitecture.VITS, {
            "sample_rate":
            22_050,
            "variants": (
                _variant(
                    "paired-waveforms",
                    required=("source_audio", "target_audio"),
                    description=(
                        "Linguistically paired source and target waveforms for "
                        "the explicit reconstructed converter objective."),
                ),
                _variant(
                    "paired-waveform-aliases",
                    required=("audio", "target_waveform"),
                    description="Canonical source and target waveform aliases.",
                ),
            ),
        })


def build_outetts_dataset_spec() -> TTSDatasetSpec:
    """Return the OuteTTS source-data contract."""
    return _model_spec(
        TTSDataArchitecture.CODEC_LM, {
            "sample_rate":
            24_000,
            "variants": (
                _variant(
                    "v3-profile",
                    one_of=(("speaker_profile", "speaker", "profile"), ),
                    description=(
                        "Aligned V3 speaker profile with word timestamps, DAC "
                        "codebooks, and acoustic features."),
                    preprocessed=True,
                ),
                _variant(
                    "inline-v3-profile",
                    required=("text", "words", "global_features"),
                    preprocessed=True,
                ),
                _variant(
                    "tokenized",
                    required=("input_ids", "labels"),
                    preprocessed=True,
                ),
            ),
        })


def build_parlertts_dataset_spec() -> TTSDatasetSpec:
    """Return the Parler-TTS source-data contract."""
    return _model_spec(
        TTSDataArchitecture.SEQUENCE_TO_SEQUENCE, {
            "sample_rate":
            44_100,
            "variants": (
                _variant(
                    "waveform-teacher-forcing",
                    one_of=(
                        ("description", "input_ids"),
                        ("audio_values", "input_values"),
                    ),
                    at_most_one_of=(
                        ("description", "input_ids"),
                        ("audio_values", "input_values"),
                    ),
                    forbidden=("audio_codes", "labels"),
                    description=(
                        "Description or prepared text IDs plus waveform tensors "
                        "for frozen-DAC teacher forcing."),
                ),
                _variant(
                    "dac-codes",
                    required=("audio_codes", ),
                    one_of=(("description", "input_ids"), ),
                    at_most_one_of=(("description", "input_ids"), ),
                    forbidden=("audio_values", "input_values", "labels"),
                    preprocessed=True,
                ),
                _variant(
                    "delayed-labels",
                    required=("labels", ),
                    one_of=(("description", "input_ids"), ),
                    at_most_one_of=(("description", "input_ids"), ),
                    forbidden=("audio_values", "input_values", "audio_codes"),
                    preprocessed=True,
                ),
            ),
        })


def build_mosstts_dataset_spec() -> TTSDatasetSpec:
    """Return the MossTTS source-data contract."""
    return _model_spec(
        TTSDataArchitecture.CODEC_LM, {
            "variants": (
                _variant(
                    "raw-audio",
                    required=("text", ),
                    one_of=(("audio", "waveform", "audio_path"), ),
                    at_most_one_of=(("audio", "waveform", "audio_path"), ),
                    forbidden=("speech_tokens", ),
                    description=(
                        "Text plus waveform for the selected runtime's frozen "
                        "codec; model sample rates vary by checkpoint family."),
                ),
                _variant(
                    "preencoded-rvq",
                    required=("text", "speech_tokens"),
                    forbidden=("audio", "waveform", "audio_path"),
                    preprocessed=True,
                ),
            ),
        })


def build_qwen3tts_dataset_spec() -> TTSDatasetSpec:
    """Return the Qwen3-TTS source-data contract."""
    return _model_spec(
        TTSDataArchitecture.CODEC_LM, {
            "sample_rate":
            24_000,
            "field_aliases": (
                ("reference_audio", "ref_audio"),
                ("reference_audio_path", "ref_audio"),
                ("speaker_audio", "ref_audio"),
            ),
            "variants": (
                _variant(
                    "single-speaker-sft",
                    required=("text", "audio_codes", "ref_audio"),
                    description="Precomputed 16-codebook targets and reference audio.",
                    preprocessed=True,
                ),
                _variant(
                    "model-ready",
                    required=(
                        "input_ids",
                        "codec_ids",
                        "ref_mels",
                        "text_embedding_mask",
                        "codec_embedding_mask",
                        "attention_mask",
                        "codec_0_labels",
                        "codec_mask",
                    ),
                    preprocessed=True,
                ),
            ),
        })


def build_irodoritts_dataset_spec() -> TTSDatasetSpec:
    """Return the IrodoriTTS source-data contract."""
    return _model_spec(
        TTSDataArchitecture.DIFFUSION, {
            "sample_rate":
            48_000,
            "variants": (
                _variant(
                    "raw-waveform",
                    required=("text", ),
                    one_of=(("waveform", "audio"), ),
                    at_most_one_of=(("waveform", "audio"), ),
                    forbidden=("target_latent", "latent"),
                    description="Text plus an in-memory target waveform.",
                ),
                _variant(
                    "preencoded-latent",
                    required=("text", ),
                    one_of=(("target_latent", "latent"), ),
                    at_most_one_of=(("target_latent", "latent"), ),
                    forbidden=("waveform", "audio"),
                    description="Text plus a precomputed codec target latent.",
                    preprocessed=True,
                ),
            ),
        })


def build_zonos_dataset_spec() -> TTSDatasetSpec:
    """Return the Zonos source-data contract."""
    return _model_spec(
        TTSDataArchitecture.CODEC_LM, {
            "variants": (
                _variant(
                    "codec-batch",
                    required=("prefix_conditioning", "audio_codes"),
                    description="Prefix conditioning and delayed DAC codebooks.",
                    preprocessed=True,
                ), ),
        })


def build_zonos2_dataset_spec() -> TTSDatasetSpec:
    """Return the Zonos2 source-data contract."""
    return _model_spec(
        TTSDataArchitecture.CODEC_LM, {
            "sample_rate":
            44_100,
            "variants": (
                _variant(
                    "raw-audio",
                    one_of=(
                        ("text", "texts"),
                        ("audio", "audio_values"),
                    ),
                    at_most_one_of=(
                        ("text", "texts"),
                        ("audio", "audio_values"),
                    ),
                    forbidden=("audio_codes", "input_ids", "labels"),
                    description="Text plus waveform for frozen native DAC encoding.",
                ),
                _variant(
                    "cached-dac",
                    required=("audio_codes", ),
                    one_of=(("text", "texts"), ),
                    at_most_one_of=(("text", "texts"), ),
                    forbidden=("audio", "audio_values", "input_ids", "labels"),
                    description="Text plus undelayed cached DAC frames.",
                    preprocessed=True,
                ),
                _variant(
                    "model-ready",
                    required=("input_ids", "labels"),
                    forbidden=(
                        "text",
                        "texts",
                        "audio",
                        "audio_values",
                        "audio_codes",
                    ),
                    preprocessed=True,
                ),
            ),
        })


def build_voxcpm_dataset_spec() -> TTSDatasetSpec:
    """Return the VoxCPM source-data contract."""
    return _model_spec(
        TTSDataArchitecture.DIFFUSION, {
            "sample_rate":
            16_000,
            "variants": (
                _variant(
                    "raw-waveform",
                    required=("text", ),
                    one_of=(("audio", "waveform"), ),
                    at_most_one_of=(("audio", "waveform"), ),
                    forbidden=("audio_features", ),
                    description="Text plus an in-memory 16 kHz target waveform.",
                ),
                _variant(
                    "audio-features",
                    required=("text", "audio_features"),
                    forbidden=("audio", "waveform"),
                    description="Text plus pre-encoded AudioVAE latent patches.",
                    preprocessed=True,
                ),
            ),
        })


def build_omnivoice_dataset_spec() -> TTSDatasetSpec:
    """Return the OmniVoice source-data contract."""
    return _model_spec(
        TTSDataArchitecture.HYBRID, {
            "sample_rate":
            24_000,
            "variants": (
                _variant(
                    "raw-audio",
                    required=("text", ),
                    one_of=(("audio", "waveform"), ),
                    at_most_one_of=(("audio", "waveform"), ),
                    forbidden=("audio_tokens", ),
                    description="Text plus raw 24 kHz audio for frozen-codec encoding.",
                ),
                _variant(
                    "audio-tokens",
                    required=("text", "audio_tokens"),
                    forbidden=("audio", "waveform"),
                    description="Text plus pre-encoded eight-codebook audio tokens.",
                    preprocessed=True,
                ),
            ),
        })


def build_higgstts_dataset_spec() -> TTSDatasetSpec:
    """Return the Higgs Audio source-data contract."""
    return _model_spec(
        TTSDataArchitecture.CODEC_LM, {
            "sample_rate":
            24_000,
            "variants": (
                _variant(
                    "raw-audio",
                    required=("text", ),
                    one_of=(("audio", "target_audio"), ),
                    at_most_one_of=(
                        ("audio", "target_audio"),
                        ("reference_audio", "reference_codes"),
                    ),
                    forbidden=("audio_codes", ),
                    requires=(
                        ("reference_audio", ("reference_text", )),
                        ("reference_codes", ("reference_text", )),
                    ),
                    requires_one_of=(("reference_text", ("reference_audio", "reference_codes")), ),
                    description="Text plus raw target audio and optional aligned reference.",
                ),
                _variant(
                    "audio-codes",
                    required=("text", "audio_codes"),
                    at_most_one_of=(("reference_audio", "reference_codes"), ),
                    forbidden=("audio", "target_audio"),
                    requires=(
                        ("reference_audio", ("reference_text", )),
                        ("reference_codes", ("reference_text", )),
                    ),
                    requires_one_of=(("reference_text", ("reference_audio", "reference_codes")), ),
                    description="Text plus pre-encoded target and optional reference codes.",
                    preprocessed=True,
                ),
            ),
        })


def build_xtts_dataset_spec() -> TTSDatasetSpec:
    """Return the XTTS source-data contract."""
    return _model_spec(
        TTSDataArchitecture.HYBRID, {
            "sample_rate":
            22_050,
            "variants": (
                _variant(
                    "native-gpt-tokens",
                    required=(
                        "text_inputs",
                        "text_lengths",
                        "audio_codes",
                        "wav_lengths",
                    ),
                    one_of=(("cond_mels", "cond_latents"), ),
                    description=(
                        "Precomputed native XTTS GPT inputs; this remains the "
                        "zero-DVAE-overhead training path."),
                    preprocessed=True,
                ),
                _variant(
                    "native-gpt-waveform",
                    required=(
                        "text_inputs",
                        "text_lengths",
                    ),
                    one_of=(
                        ("wav", "audio_values"),
                        ("cond_mels", "cond_latents"),
                    ),
                    at_most_one_of=(("wav", "audio_values"), ),
                    forbidden=("audio_codes", ),
                    description=(
                        "Tokenized text and conditioning plus waveform audio; "
                        "the separately loaded native frozen DVAE produces targets."),
                    preprocessed=True,
                ),
            ),
        })


def build_vibevoice_dataset_spec() -> TTSDatasetSpec:
    """Return the VibeVoice TTS source-data contract."""
    return _model_spec(
        TTSDataArchitecture.HYBRID, {
            "sample_rate":
            24_000,
            "variants": (
                _variant(
                    "lm-diffusion-batch",
                    required=(
                        "input_ids",
                        "attention_mask",
                        "speech_tensors",
                        "speech_masks",
                        "speeches_loss_input",
                        "speech_semantic_tensors",
                        "acoustic_input_mask",
                        "acoustic_loss_mask",
                    ),
                    preprocessed=True,
                ), ),
        })


def build_fishtts_dataset_spec() -> TTSDatasetSpec:
    """Return the Fish Speech source-data contract."""
    return _model_spec(
        TTSDataArchitecture.CODEC_LM, {
            "sample_rate":
            44_100,
            "variants": (
                _variant(
                    "semantic-tokens",
                    required=("labels", ),
                    one_of=(("tokens", "inputs"), ),
                    description="Offline semantic-codec tokens and labels.",
                    preprocessed=True,
                ), ),
        })


def build_csm_dataset_spec() -> TTSDatasetSpec:
    """Return the CSM source-data contract."""
    return _model_spec(
        TTSDataArchitecture.CODEC_LM, {
            "sample_rate":
            24_000,
            "variants": (
                _variant("conversation", one_of=(("conversation", "messages"), )),
                _variant(
                    "grouped-audios",
                    required=("texts", "speaker_ids", "audios"),
                ),
                _variant(
                    "grouped-concatenated",
                    required=(
                        "texts",
                        "speaker_ids",
                        "audio",
                        "audio_cut_idxs",
                    ),
                ),
                _variant("utterance", required=("text", "audio")),
                _ARCHITECTURE_SPECS[TTSDataArchitecture.CODEC_LM].variants[2],
            ),
        })


def build_neutts_dataset_spec() -> TTSDatasetSpec:
    """Return the NeuTTS source-data contract."""
    return _model_spec(
        TTSDataArchitecture.CODEC_LM, {
            "variants": (
                _variant("raw-audio", required=("text", ), one_of=(("audio", "audio_codes"), )),
                _ARCHITECTURE_SPECS[TTSDataArchitecture.CODEC_LM].variants[2],
            ),
        })


def build_speecht5_dataset_spec() -> TTSDatasetSpec:
    """Return the SpeechT5 source-data contract."""
    return _model_spec(
        TTSDataArchitecture.SEQUENCE_TO_SEQUENCE, {
            "sample_rate":
            16_000,
            "variants": (
                _variant("raw-audio", required=("text", "audio")),
                _ARCHITECTURE_SPECS[TTSDataArchitecture.SEQUENCE_TO_SEQUENCE].variants[1],
            ),
        })


def build_styletts2_dataset_spec() -> TTSDatasetSpec:
    """Return the StyleTTS2 source-data contract."""
    return _model_spec(
        TTSDataArchitecture.VITS, {
            "sample_rate":
            24_000,
            "variants": (
                _variant(
                    "explicit-features",
                    required=(
                        "input_ids",
                        "alignments",
                        "normalized_mel",
                        "reference_mel",
                        "f0_targets",
                        "noise_targets",
                        "audio_values",
                    ),
                    description=(
                        "Exact phoneme, monotonic-alignment, mel, prosody, noise, "
                        "and waveform supervision consumed by the native collator."),
                    preprocessed=True,
                ), ),
        })


def build_supertonic_dataset_spec() -> TTSDatasetSpec:
    """Return the Supertonic source-data contract."""
    return _model_spec(
        TTSDataArchitecture.DIFFUSION, {
            "sample_rate":
            44_100,
            "variants": (
                _variant(
                    "text-style-object",
                    required=("text", "style"),
                    one_of=((
                        "target_duration",
                        "duration",
                        "duration_seconds",
                        "target_latent",
                        "latent",
                        "latents",
                    ), ),
                    preprocessed=True,
                ),
                _variant(
                    "text-style-tensors",
                    required=("text", "style_ttl", "style_dp"),
                    one_of=((
                        "target_duration",
                        "duration",
                        "duration_seconds",
                        "target_latent",
                        "latent",
                        "latents",
                    ), ),
                    preprocessed=True,
                ),
                _variant(
                    "tokenized-style-object",
                    required=("text_ids", "style"),
                    one_of=(
                        ("text_mask", "text_lengths"),
                        (
                            "target_duration",
                            "duration",
                            "duration_seconds",
                            "target_latent",
                            "latent",
                            "latents",
                        ),
                    ),
                    preprocessed=True,
                ),
                _variant(
                    "tokenized-style-tensors",
                    required=("text_ids", "style_ttl", "style_dp"),
                    one_of=(
                        ("text_mask", "text_lengths"),
                        (
                            "target_duration",
                            "duration",
                            "duration_seconds",
                            "target_latent",
                            "latent",
                            "latents",
                        ),
                    ),
                    preprocessed=True,
                ),
            ),
        })


def build_inflecttts_dataset_spec() -> TTSDatasetSpec:
    """Return the InflectTTS source-data contract."""
    return _model_spec(
        TTSDataArchitecture.VITS, {
            "sample_rate":
            24_000,
            "variants": (
                _variant(
                    "explicit-features",
                    required=("input_ids", "spectrogram", "audio_values"),
                    description=(
                        "Checkpoint-compatible phoneme IDs, 513-bin magnitude "
                        "spectrogram, and aligned waveform."),
                    preprocessed=True,
                ), ),
        })


def build_bark_dataset_spec() -> TTSDatasetSpec:
    """Return the Bark source-data contract."""
    return _model_spec(
        TTSDataArchitecture.HYBRID, {
            "sample_rate":
            24_000,
            "variants": (
                _variant(
                    "causal-stage",
                    required=("input_ids", "labels", "training_phase"),
                    description="Prepared semantic- or coarse-stage tokens.",
                    preprocessed=True,
                ),
                _variant(
                    "fine-stage",
                    required=(
                        "input_ids",
                        "labels",
                        "codebook_idx",
                        "training_phase",
                    ),
                    description="Prepared fine-stage codec tokens.",
                    preprocessed=True,
                ),
                _variant(
                    "all-stages",
                    required=(
                        "semantic_input_ids",
                        "semantic_labels",
                        "coarse_input_ids",
                        "coarse_labels",
                        "fine_input_ids",
                        "fine_labels",
                        "codebook_idx",
                    ),
                    description="One batch carrying prefixed inputs for all Bark phases.",
                    preprocessed=True,
                ),
            ),
        })


def build_vits_dataset_spec() -> TTSDatasetSpec:
    """Return the VITS source-data contract."""
    return _model_spec(
        TTSDataArchitecture.VITS, {
            "variants": (
                _variant(
                    "raw-adversarial",
                    required=("text", ),
                    one_of=(("audio", "audio_values"), ),
                    description=(
                        "Raw text/waveform training requires "
                        "enable_native_adversarial_training=True and an explicit "
                        "training_acoustic_config."),
                ),
                _variant(
                    "tokenized-raw-adversarial",
                    required=("input_ids", ),
                    one_of=(("audio", "audio_values"), ),
                    description=(
                        "Tokenized raw-waveform training requires "
                        "enable_native_adversarial_training=True and an explicit "
                        "training_acoustic_config."),
                ),
                _variant(
                    "precomputed-spectrogram",
                    required=("spectrogram", ),
                    one_of=(
                        ("text", "input_ids"),
                        ("audio", "audio_values"),
                    ),
                    description="Generator warm-start with an explicit spectrogram.",
                    preprocessed=True,
                ),
            ),
        })


_TRAINING_FAMILY_TO_DATA_ARCHITECTURE = MappingProxyType({
    "causal-lm": TTSDataArchitecture.CODEC_LM,
    "sequence-to-sequence": TTSDataArchitecture.SEQUENCE_TO_SEQUENCE,
    "flow-matching": TTSDataArchitecture.DIFFUSION,
    "vits": TTSDataArchitecture.VITS,
    "acoustic-regression": TTSDataArchitecture.ACOUSTIC,
    "composite": TTSDataArchitecture.HYBRID,
})


def _load_model_dataset_spec(training_spec: Any) -> TTSDatasetSpec | None:
    factory_path = training_spec.dataset_spec_factory
    if factory_path is None:
        return None
    try:
        factory = resolve_import_path(factory_path)
    except (AttributeError, ImportError) as exc:
        raise ImportError(
            f"Could not resolve TTS dataset spec factory {factory_path!r} "
            f"for {training_spec.model_type!r}.") from exc
    if not callable(factory):
        raise TypeError(
            f"TTS dataset spec factory {factory_path!r} for "
            f"{training_spec.model_type!r} must be callable.")
    spec = factory()
    if not isinstance(spec, TTSDatasetSpec):
        raise TypeError(
            f"TTS dataset spec factory {factory_path!r} for "
            f"{training_spec.model_type!r} returned {type(spec).__name__}; "
            "expected TTSDatasetSpec.")
    if spec.model_type not in (None, training_spec.model_type):
        raise ValueError(
            f"TTS dataset spec factory {factory_path!r} returned a contract for "
            f"{spec.model_type!r}, not {training_spec.model_type!r}.")
    if spec.training_support not in (None, training_spec.support.value):
        raise ValueError(
            f"TTS dataset spec factory {factory_path!r} declares training support "
            f"{spec.training_support!r}, not {training_spec.support.value!r}.")
    return spec


def get_tts_dataset_spec(
    model_type: str | None = None,
    *,
    architecture: TTSDataArchitecture | str | None = None,
) -> TTSDatasetSpec:
    """Return the inspectable dataset contract for a model or architecture."""
    canonical_model_type = None
    training_support = None
    model_spec = None
    if model_type is not None:
        if not isinstance(model_type, str) or not model_type.strip():
            raise ValueError("model_type must be a non-empty string or None.")
        from voicehub.training.specs import get_training_spec

        training_spec = get_training_spec(model_type)
        canonical_model_type = training_spec.model_type
        training_support = training_spec.support.value
        if training_spec.task.value != "text-to-speech":
            raise ValueError(
                f"{canonical_model_type!r} is registered for {training_spec.task.value}, not TTS.")
        model_spec = _load_model_dataset_spec(training_spec)
        if model_spec is not None:
            resolved_architecture = model_spec.architecture
        else:
            try:
                resolved_architecture = _TRAINING_FAMILY_TO_DATA_ARCHITECTURE[training_spec.family_name]
            except KeyError as exc:
                raise ValueError(
                    f"TTS training family {training_spec.family_name!r} has no "
                    "registered source-data architecture.") from exc
        if architecture is not None and TTSDataArchitecture.coerce(architecture) is not resolved_architecture:
            raise ValueError(
                f"{canonical_model_type!r} uses {resolved_architecture.value!r} "
                f"data, not {TTSDataArchitecture.coerce(architecture).value!r}.")
    elif architecture is None:
        raise ValueError("Pass either model_type or architecture.")
    else:
        resolved_architecture = TTSDataArchitecture.coerce(architecture)

    base = model_spec or _ARCHITECTURE_SPECS[resolved_architecture]
    if canonical_model_type is None:
        variants = base.variants
        readiness = None
    else:
        variants = base.variants if model_spec is not None else base.preprocessed_variants
        if training_support == "inference-only":
            readiness = TTSDataReadiness.UNAVAILABLE
        elif base.readiness is not None:
            readiness = base.readiness
        elif any(not variant.preprocessed for variant in variants):
            readiness = TTSDataReadiness.INTEGRATED
        elif training_support == "custom":
            readiness = TTSDataReadiness.CUSTOM
        else:
            readiness = TTSDataReadiness.PREPROCESSED
    return replace(
        base,
        variants=variants,
        model_type=canonical_model_type,
        readiness=readiness,
        training_support=training_support,
    )


def list_tts_dataset_specs() -> tuple[TTSDatasetSpec, ...]:
    """Return one model-specific dataset contract for every TTS profile."""
    from voicehub.training.specs import list_training_specs

    return tuple(get_tts_dataset_spec(spec.model_type) for spec in list_training_specs())


__all__ = [
    "TTSDataArchitecture",
    "TTSDataReadiness",
    "TTSDatasetSpec",
    "TTSRecordVariant",
    "get_tts_dataset_spec",
    "list_tts_dataset_specs",
]

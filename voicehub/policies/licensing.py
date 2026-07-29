"""Machine-readable model license and commercial-use metadata."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping


@dataclass(frozen=True)
class ModelLicenseSpec:
    """License conditions attached to source or checkpoint artifacts."""

    model_type: str
    license_id: str
    commercial_use: bool | None
    upstream: str
    notice: str


_MODEL_LICENSES = (
    ModelLicenseSpec(
        "conversationtts",
        "CC-BY-NC-4.0",
        False,
        "https://github.com/Audio-Foundation-Models/ConversationTTS",
        "Source, checkpoints, datasets, and evaluation tools are non-commercial.",
    ),
    ModelLicenseSpec(
        "fishtts",
        "Fish-Audio-Research-License",
        False,
        "https://github.com/fishaudio/fish-speech",
        "Fine-tuned checkpoints are derivative works. Commercial use "
        "requires a separate written Fish Audio license. Distribution must "
        "include the Fish Audio Research License, retain its exact copyright "
        "notice, and prominently display “Built with Fish Audio”. The "
        "license also restricts using materials, derivatives, or outputs to "
        "create or improve non-Fish foundational generative-AI models.",
    ),
    ModelLicenseSpec(
        "llasa",
        "CC-BY-NC-4.0",
        False,
        "https://huggingface.co/HKUSTAudio/xcodec2",
        "The vendored XCodec2 component is restricted to non-commercial use.",
    ),
    ModelLicenseSpec(
        "asr_medasr",
        "health-ai-developer-foundations",
        None,
        "https://huggingface.co/google/medasr",
        "Access requires accepting Google's Health AI Developer Foundations "
        "terms. Review the healthcare-specific use restrictions before "
        "fine-tuning or deployment.",
    ),
    ModelLicenseSpec(
        "asr_nemotron",
        "OpenMDW-1.1",
        True,
        "https://huggingface.co/nvidia/nemotron-3.5-asr-streaming-0.6b",
        "Use of the checkpoint and derivatives is governed by the "
        "OpenMDW-1.1 license.",
    ),
    ModelLicenseSpec(
        "asr_parakeet_tdt",
        "CC-BY-4.0",
        True,
        "https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3",
        "The pinned Parakeet TDT checkpoint and derivatives require "
        "CC-BY-4.0 attribution. The VoiceHub-owned architecture port is "
        "audited against Apache-2.0 Transformers and NeMo source.",
    ),
    ModelLicenseSpec(
        "asr_seamless_m4t_v2",
        "CC-BY-NC-4.0",
        False,
        "https://huggingface.co/facebook/seamless-m4t-v2-large",
        "The pinned SeamlessM4T-v2 Large checkpoint and fine-tuned "
        "derivatives are non-commercial under CC-BY-NC-4.0. The "
        "VoiceHub-native S2T architecture port is audited against "
        "Apache-2.0 Transformers source.",
    ),
    ModelLicenseSpec(
        "asr_nemo",
        "NVIDIA-NGC-Terms",
        None,
        "https://catalog.ngc.nvidia.com/orgs/nvidia/teams/nemo/models/stt_en_quartznet15x5",
        "The QuartzNet checkpoint is governed by the NVIDIA NGC Terms of "
        "Use; the VoiceHub-owned architecture code is Apache-2.0.",
    ),
    ModelLicenseSpec(
        "asr_speechbrain",
        "Apache-2.0",
        True,
        "https://huggingface.co/speechbrain/asr-crdnn-rnnlm-librispeech",
        "The pinned CRDNN, RNNLM, tokenizer, and source implementation are "
        "Apache-2.0. The original pickle files cross a strict one-time "
        "conversion boundary; steady-state artifacts are Safetensors.",
    ),
    ModelLicenseSpec(
        "asr_wenet",
        "NOT DECLARED",
        None,
        "http://mobvoi-speech-public.ufile.ucloud.cn/public/wenet/gigaspeech/20210728_u2pp_conformer_exp.tar.gz",
        "The published GigaSpeech checkpoint archive does not declare a "
        "checkpoint license. The VoiceHub-owned architecture port is "
        "Apache-2.0, but that source license is not assumed for the weights.",
    ),
    ModelLicenseSpec(
        "neutts",
        "NeuTTS-Open-License-1.0",
        None,
        "https://github.com/neuphonic/neutts",
        "NeuTTS Air is Apache-2.0. Other registered variants use the NeuTTS "
        "Open License v1.0, which allows limited commercial use below its "
        "USD 5,000,000 annual-revenue threshold and requires a separate "
        "license at or above that threshold.",
    ),
    ModelLicenseSpec(
        "outetts",
        "CC-BY-NC-SA-4.0",
        False,
        "https://huggingface.co/OuteAI/Llama-OuteTTS-1.0-1B",
        "The default Llama-OuteTTS-1.0-1B checkpoint is non-commercial. "
        "The supported OuteTTS 0.6B checkpoint is Apache-2.0; "
        "review the selected artifact before training or deployment.",
    ),
    ModelLicenseSpec(
        "vad_sherpa_onnx",
        "LicenseRef-TEN-VAD-Open-Source-License",
        None,
        "https://github.com/TEN-framework/ten-vad",
        "The provider's optional TEN family is governed by a non-standard "
        "license with additional deployment restrictions, including limits "
        "on competing with Agora. Review the bundled THIRD_PARTY_LICENSE "
        "before conversion, fine-tuning, distribution, or deployment. The "
        "default Silero family retains its own checkpoint terms.",
    ),
    ModelLicenseSpec(
        "xtts",
        "CPML",
        None,
        "https://huggingface.co/coqui/XTTS-v2",
        "XTTS checkpoint terms are separate from the MPL-2.0 runtime source.",
    ),
)

MODEL_LICENSES: Mapping[str, ModelLicenseSpec] = MappingProxyType(
    {spec.model_type: spec
     for spec in _MODEL_LICENSES})


def get_model_license(model_type: str) -> ModelLicenseSpec | None:
    """Return special license metadata, if a model has additional terms."""
    return MODEL_LICENSES.get(model_type)

"""Built-in native architecture catalogue.

Only lightweight registration modules are imported here. Model graphs,
processors, checkpoint readers, and objectives remain lazy component
references until a runtime explicitly resolves them.
"""

from __future__ import annotations

from importlib import import_module

from voicehub.architectures.registry import ARCHITECTURE_REGISTRY, ArchitectureRegistry
from voicehub.architectures.specifications import ArchitectureSpec

BUILTIN_ARCHITECTURE_REGISTRARS = (
    (
        "voicehub.architectures.asr_dispatch.registration",
        "register_asr_dispatch_architecture",
    ),
    (
        "voicehub.architectures.whisper.registration",
        "register_whisper_architecture",
    ),
    (
        "voicehub.architectures.wav2vec2.registration",
        "register_wav2vec2_architecture",
    ),
    (
        "voicehub.architectures.hubert.registration",
        "register_hubert_architecture",
    ),
    (
        "voicehub.architectures.wavlm.registration",
        "register_wavlm_architecture",
    ),
    (
        "voicehub.architectures.moonshine.registration",
        "register_moonshine_architecture",
    ),
    (
        "voicehub.architectures.qwen3_asr.registration",
        "register_qwen3_asr_architecture",
    ),
    (
        "voicehub.architectures.granite_speech.registration",
        "register_granite_speech_architecture",
    ),
    (
        "voicehub.architectures.parakeet_tdt.registration",
        "register_parakeet_tdt_architecture",
    ),
    (
        "voicehub.architectures.nemotron_asr.registration",
        "register_nemotron_asr_architecture",
    ),
    (
        "voicehub.architectures.cohere_asr.registration",
        "register_cohere_asr_architecture",
    ),
    (
        "voicehub.architectures.seamless_m4t_v2.registration",
        "register_seamless_m4t_v2_architecture",
    ),
    (
        "voicehub.architectures.vibevoice.registration",
        "register_vibevoice_asr_architecture",
    ),
    (
        "voicehub.architectures.medasr.registration",
        "register_medasr_architecture",
    ),
    (
        "voicehub.architectures.sensevoice.registration",
        "register_sensevoice_architecture",
    ),
    (
        "voicehub.architectures.nemo_ctc.registration",
        "register_nemo_ctc_architecture",
    ),
    (
        "voicehub.architectures.wenet_u2pp.registration",
        "register_wenet_u2pp_architecture",
    ),
    (
        "voicehub.architectures.espnet_transformer.registration",
        "register_espnet_architecture",
    ),
    (
        "voicehub.architectures.vits.registration",
        "register_vits_architecture",
    ),
    (
        "voicehub.architectures.vui.registration",
        "register_vui_architecture",
    ),
    (
        "voicehub.architectures.chatterbox.registration",
        "register_chatterbox_architecture",
    ),
    (
        "voicehub.architectures.csm.registration",
        "register_csm_architecture",
    ),
    (
        "voicehub.architectures.conversationtts.registration",
        "register_conversationtts_architecture",
    ),
    (
        "voicehub.architectures.llasa.registration",
        "register_llasa_architecture",
    ),
    (
        "voicehub.architectures.neutts.registration",
        "register_neutts_architecture",
    ),
    (
        "voicehub.architectures.outetts.registration",
        "register_outetts_architecture",
    ),
    (
        "voicehub.architectures.speecht5.registration",
        "register_speecht5_architecture",
    ),
    (
        "voicehub.architectures.kokoro.registration",
        "register_kokoro_architecture",
    ),
    (
        "voicehub.architectures.f5tts.registration",
        "register_f5tts_architecture",
    ),
    (
        "voicehub.architectures.gptsovits.registration",
        "register_gptsovits_architecture",
    ),
    (
        "voicehub.architectures.mosstts.registration",
        "register_mosstts_architecture",
    ),
    (
        "voicehub.architectures.melotts.registration",
        "register_melotts_architecture",
    ),
    (
        "voicehub.architectures.openvoice.registration",
        "register_openvoice_architecture",
    ),
    (
        "voicehub.architectures.parlertts.registration",
        "register_parlertts_architecture",
    ),
    (
        "voicehub.architectures.styletts2.registration",
        "register_styletts2_architecture",
    ),
    (
        "voicehub.architectures.qwen3_tts.registration",
        "register_qwen3_tts_architecture",
    ),
    (
        "voicehub.architectures.vibevoice.registration",
        "register_vibevoice_tts_architecture",
    ),
    (
        "voicehub.architectures.voxcpm2.registration",
        "register_voxcpm2_architecture",
    ),
    (
        "voicehub.architectures.omnivoice.registration",
        "register_omnivoice_architecture",
    ),
    (
        "voicehub.architectures.higgs_audio_v2.registration",
        "register_higgs_audio_v2_architecture",
    ),
    (
        "voicehub.architectures.irodoritts.registration",
        "register_irodori_architecture",
    ),
    (
        "voicehub.architectures.cosyvoice_native.registration",
        "register_cosyvoice_architecture",
    ),
    (
        "voicehub.architectures.xtts2.registration",
        "register_xtts2_architecture",
    ),
    (
        "voicehub.architectures.fishtts.registration",
        "register_fish_s2_architecture",
    ),
    (
        "voicehub.architectures.zonos.registration",
        "register_zonos_architecture",
    ),
    (
        "voicehub.architectures.zonos2.registration",
        "register_zonos2_architecture",
    ),
    (
        "voicehub.architectures.supertonic.registration",
        "register_supertonic_architecture",
    ),
    (
        "voicehub.architectures.bark.registration",
        "register_bark_architecture",
    ),
    (
        "voicehub.architectures.inflecttts.registration",
        "register_inflect_architecture",
    ),
    (
        "voicehub.architectures.echo_flow.registration",
        "register_echo_architecture",
    ),
    (
        "voicehub.architectures.vad_dispatch.registration",
        "register_vad_dispatch_architecture",
    ),
    (
        "voicehub.architectures.silero_vad.registration",
        "register_silero_vad_architecture",
    ),
    (
        "voicehub.architectures.fsmn_vad.registration",
        "register_fsmn_vad_architecture",
    ),
    (
        "voicehub.architectures.speechbrain_asr.registration",
        "register_speechbrain_asr_architecture",
    ),
    (
        "voicehub.architectures.speechbrain_vad.registration",
        "register_speechbrain_vad_architecture",
    ),
    (
        "voicehub.architectures.ten_vad.registration",
        "register_ten_vad_architecture",
    ),
    (
        "voicehub.architectures.webrtc_vad.registration",
        "register_webrtc_vad_architecture",
    ),
    (
        "voicehub.architectures.marblenet_vad.registration",
        "register_marblenet_vad_architecture",
    ),
    (
        "voicehub.architectures.pyannet.registration",
        "register_pyannet_architecture",
    ),
    (
        "voicehub.architectures.energy_vad.registration",
        "register_energy_vad_architecture",
    ),
    (
        "voicehub.architectures.causal_lm.registration",
        "register_causal_lm_architecture",
    ),
    (
        "voicehub.architectures.dac.registration",
        "register_dac_architecture",
    ),
    (
        "voicehub.architectures.encodec.registration",
        "register_encodec_architecture",
    ),
    (
        "voicehub.architectures.dia.registration",
        "register_dia_architecture",
    ),
)


def register_builtin_architectures(
    *,
    registry: ArchitectureRegistry | None = None,
    exist_ok: bool = True,
) -> tuple[ArchitectureSpec, ...]:
    """Register every built-in declaration without importing model graphs."""
    target = ARCHITECTURE_REGISTRY if registry is None else registry
    if not isinstance(target, ArchitectureRegistry):
        raise TypeError("`registry` must be an ArchitectureRegistry or None.")
    if not isinstance(exist_ok, bool):
        raise TypeError("`exist_ok` must be a boolean.")

    registered = []
    for module_name, function_name in BUILTIN_ARCHITECTURE_REGISTRARS:
        registrar = getattr(import_module(module_name), function_name)
        spec = registrar(
            registry=target,
            exist_ok=exist_ok,
        )
        if not isinstance(spec, ArchitectureSpec):
            raise TypeError(
                f"Built-in registrar {module_name}:{function_name} returned "
                f"{type(spec).__name__}, not ArchitectureSpec.")
        registered.append(spec)
    return tuple(registered)


__all__ = [
    "BUILTIN_ARCHITECTURE_REGISTRARS",
    "register_builtin_architectures",
]

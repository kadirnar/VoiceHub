"""Small, dependency-lazy helpers shared by source-integrated backends."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from voicehub.dependencies import import_optional
from voicehub.modeling_outputs import TTSOutput


def finish_audio_output(
    audio: Any,
    sample_rate: int,
    *,
    output_file: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> TTSOutput:
    """Build and optionally persist the normalized VoiceHub output."""
    output = TTSOutput(
        audio=audio,
        sample_rate=int(sample_rate),
        metadata=metadata or {},
    )
    if output_file:
        output.save(output_file)
    return output


def resolve_model_directory(
    name_or_path: str,
    *,
    model_type: str,
    install_extra: str | None = None,
    **download_kwargs: Any,
) -> Path:
    """Resolve a local directory or download one complete Hub snapshot."""
    source = Path(name_or_path).expanduser()
    if source.is_dir():
        return source.resolve()
    hub = import_optional(
        "huggingface_hub",
        model_type=model_type,
        install_extra=install_extra or model_type,
    )
    return Path(
        hub.snapshot_download(
            repo_id=name_or_path,
            **{
                key: value
                for key, value in download_kwargs.items() if value is not None
            },
        ))


def resolve_torch_dtype(torch: Any, dtype_name: str, device: str) -> Any:
    """Map a configured dtype name while keeping CPU defaults safe."""
    if device == "cpu" and dtype_name in {"float16", "bfloat16"}:
        return torch.float32
    try:
        return getattr(torch, dtype_name)
    except AttributeError as exc:
        raise ValueError(f"Unknown torch dtype: {dtype_name!r}.") from exc

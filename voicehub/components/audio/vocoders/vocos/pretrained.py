from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import torch
from torch import nn

from voicehub.checkpointing import SafeTensorReader, save_safetensors
from voicehub.components.audio.vocoders.vocos.configuration import (
    load_vocos_config,
    require_component_config,
)
from voicehub.components.audio.vocoders.vocos.feature_extractors import (
    EncodecFeatures,
    FeatureExtractor,
    MelSpectrogramFeatures,
)
from voicehub.components.audio.vocoders.vocos.heads import (
    FourierHead,
    IMDCTCosHead,
    IMDCTSymExpHead,
    ISTFTHead,
)
from voicehub.components.audio.vocoders.vocos.models import (
    Backbone,
    VocosBackbone,
    VocosResNetBackbone,
)
from voicehub.hub_transport import download_hugging_face_snapshot
from voicehub.path_utils import is_explicit_local_path

_OFFICIAL_REVISIONS = {
    "charactr/vocos-mel-24khz": "0feb3fdd929bcd6649e0e7c5a688cf7dd012ef21",
    "charactr/vocos-encodec-24khz": "4e61d082c08045a4c11e5b148ad93b1d0c591a14",
}
_COMPONENTS: dict[str, Callable[..., nn.Module]] = {}


def register_vocos_component(
    class_path: str,
    component: Callable[..., nn.Module],
    *,
    replace: bool = False,
) -> None:
    """Register one reviewed component without arbitrary module imports."""
    if not isinstance(class_path, str) or not class_path:
        raise ValueError("`class_path` must be a non-empty string.")
    if not callable(component):
        raise TypeError("A Vocos component must be callable.")
    if class_path in _COMPONENTS and not replace:
        raise ValueError(f"Vocos component {class_path!r} is already registered.")
    _COMPONENTS[class_path] = component


def _register_builtin_components() -> None:
    for class_path, component in {
        "vocos.feature_extractors.MelSpectrogramFeatures": MelSpectrogramFeatures,
        "vocos.feature_extractors.EncodecFeatures": EncodecFeatures,
        "vocos.models.VocosBackbone": VocosBackbone,
        "vocos.models.VocosResNetBackbone": VocosResNetBackbone,
        "vocos.heads.ISTFTHead": ISTFTHead,
        "vocos.heads.IMDCTSymExpHead": IMDCTSymExpHead,
        "vocos.heads.IMDCTCosHead": IMDCTCosHead,
    }.items():
        _COMPONENTS[class_path] = component
        native_path = f"{component.__module__}.{component.__name__}"
        _COMPONENTS[native_path] = component


_register_builtin_components()


def instantiate_class(
    args: Any | tuple[Any, ...],
    init: Mapping[str, Any],
) -> nn.Module:
    """Instantiates a class with the given args and init.

    Args:
        args: Positional arguments required for instantiation.
        init: Dict of the form {"class_path":...,"init_args":...}.

    Returns:
        The instantiated class object.
    """
    if not isinstance(init, Mapping):
        raise TypeError("Vocos component configuration must be a mapping.")
    class_path = init.get("class_path")
    if not isinstance(class_path, str) or not class_path:
        raise ValueError("Vocos component configuration requires `class_path`.")
    kwargs = init.get("init_args", {})
    if not isinstance(kwargs, Mapping):
        raise ValueError("Vocos component `init_args` must be a mapping.")
    if not isinstance(args, tuple):
        args = (args,)
    try:
        component = _COMPONENTS[class_path]
    except KeyError as error:
        raise ValueError(
            f"Vocos component {class_path!r} is not registered. Register "
            "VoiceHub-owned extensions explicitly with "
            "`register_vocos_component`."
        ) from error
    instance = component(*args, **dict(kwargs))
    if not isinstance(instance, nn.Module):
        raise TypeError(f"Vocos component {class_path!r} did not create a module.")
    return instance


def _load_config(path: Path) -> dict[str, Any]:
    if path.suffix.lower() == ".json":
        if path.stat().st_size > 256 * 1024:
            raise ValueError("Vocos configuration exceeds the 256 KiB safety limit.")
        config = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(config, dict):
            raise ValueError("Vocos JSON configuration must contain an object.")
        return config
    return load_vocos_config(path)


def _configuration_path(root: Path) -> Path:
    candidates = tuple(
        path for path in (root / "config.json", root / "config.yaml")
        if path.is_file()
    )
    if len(candidates) != 1:
        found = ", ".join(path.name for path in candidates) or "none"
        raise FileNotFoundError(
            "Vocos artifact must contain exactly one config.json or "
            f"config.yaml; found {found}."
        )
    return candidates[0]


def _weight_path(root: Path) -> Path:
    safe_candidates = tuple(sorted(root.glob("*.safetensors")))
    if len(safe_candidates) > 1:
        raise ValueError(
            "Vocos artifact contains multiple Safetensors files without an "
            "index."
        )
    if safe_candidates:
        return safe_candidates[0]
    legacy_candidates = tuple(
        path for path in (
            root / "pytorch_model.bin",
            root / "model.bin",
            root / "model.pt",
            root / "model.pth",
        )
        if path.is_file()
    )
    if len(legacy_candidates) != 1:
        found = ", ".join(path.name for path in legacy_candidates) or "none"
        raise FileNotFoundError(
            "Vocos artifact must contain exactly one Safetensors or legacy "
            f"weight file; found {found}."
        )
    return legacy_candidates[0]


def _legacy_tensor_state(path: Path) -> dict[str, torch.Tensor]:
    try:
        payload = torch.load(
            path,
            map_location="cpu",
            weights_only=True,
        )
    except TypeError as error:
        raise RuntimeError(
            "Legacy Vocos checkpoints require PyTorch with restricted "
            "`weights_only=True` loading."
        ) from error
    if not isinstance(payload, Mapping):
        raise TypeError("Legacy Vocos checkpoint must contain a tensor mapping.")
    for wrapper in ("state_dict", "model_state_dict"):
        nested = payload.get(wrapper)
        if isinstance(nested, Mapping):
            payload = nested
            break
    if not payload or any(
        not isinstance(name, str) or not isinstance(value, torch.Tensor)
        for name, value in payload.items()
    ):
        raise TypeError(
            "Legacy Vocos checkpoint must map string names to tensors only."
        )
    return dict(payload)


def _tensor_state(path: Path) -> dict[str, torch.Tensor]:
    if path.suffix.lower() == ".safetensors":
        with SafeTensorReader(path) as reader:
            return reader.state_dict(device="cpu")
    return _legacy_tensor_state(path)


def _load_validated_state(
    model: nn.Module,
    state: Mapping[str, torch.Tensor],
    *,
    allow_external_codec_omission: bool,
) -> None:
    expected = model.state_dict()
    codec_names = frozenset(
        name for name in expected
        if name.startswith("feature_extractor.encodec.")
    )
    supplied_codec_names = codec_names.intersection(state)
    allowed_missing = frozenset()
    if allow_external_codec_omission and not supplied_codec_names:
        allowed_missing = codec_names
    missing = tuple(
        name for name in expected
        if name not in state and name not in allowed_missing
    )
    unexpected = tuple(name for name in state if name not in expected)
    mismatched = tuple(
        (name, tuple(state[name].shape), tuple(expected[name].shape))
        for name in state.keys() & expected.keys()
        if tuple(state[name].shape) != tuple(expected[name].shape)
    )
    if missing or unexpected or mismatched:
        details: list[str] = []
        if missing:
            details.append(f"missing={list(missing[:20])!r}")
        if unexpected:
            details.append(f"unexpected={list(unexpected[:20])!r}")
        if mismatched:
            details.append(f"shape_mismatches={list(mismatched[:20])!r}")
        raise ValueError("Vocos checkpoint inventory mismatch: " + "; ".join(details))
    incompatible = model.load_state_dict(dict(state), strict=False)
    unresolved_missing = tuple(
        name for name in incompatible.missing_keys
        if name not in allowed_missing
    )
    if unresolved_missing or incompatible.unexpected_keys:
        raise RuntimeError("Vocos checkpoint changed during validated loading.")
    feature_extractor = getattr(model, "feature_extractor", None)
    if supplied_codec_names and isinstance(feature_extractor, EncodecFeatures):
        feature_extractor.mark_encodec_weights_loaded()


class Vocos(nn.Module):
    """
    The Vocos class represents a Fourier-based neural vocoder for audio synthesis.
    This class is primarily designed for inference, with support for loading from pretrained
    model checkpoints. It consists of three main components: a feature extractor,
    a backbone, and a head.
    """

    def __init__(
        self,
        feature_extractor: FeatureExtractor,
        backbone: Backbone,
        head: FourierHead,
    ):
        super().__init__()
        self.feature_extractor = feature_extractor
        self.backbone = backbone
        self.head = head
        self._component_config: dict[str, Any] | None = None

    @classmethod
    def from_hparams(cls, config_path: str | Path) -> Vocos:
        """
        Class method to create a new Vocos model instance from hyperparameters stored in a yaml configuration file.
        """
        config = _load_config(Path(config_path).expanduser())
        expected_sections = {"feature_extractor", "backbone", "head"}
        extra_sections = set(config) - expected_sections
        missing_sections = expected_sections - set(config)
        if missing_sections or extra_sections:
            raise ValueError(
                "Vocos configuration sections mismatch: "
                f"missing={sorted(missing_sections)!r}, "
                f"unknown={sorted(extra_sections)!r}."
            )
        feature_config = require_component_config(config, "feature_extractor")
        backbone_config = require_component_config(config, "backbone")
        head_config = require_component_config(config, "head")
        feature_extractor = instantiate_class(args=(), init=feature_config)
        backbone = instantiate_class(args=(), init=backbone_config)
        head = instantiate_class(args=(), init=head_config)
        if not isinstance(feature_extractor, FeatureExtractor):
            raise TypeError("Configured Vocos feature extractor has the wrong type.")
        if not isinstance(backbone, Backbone):
            raise TypeError("Configured Vocos backbone has the wrong type.")
        if not isinstance(head, FourierHead):
            raise TypeError("Configured Vocos head has the wrong type.")
        model = cls(feature_extractor=feature_extractor, backbone=backbone, head=head)
        model._component_config = {
            "feature_extractor": feature_config,
            "backbone": backbone_config,
            "head": head_config,
        }
        return model

    @classmethod
    def from_pretrained(
        cls,
        repo_id: str | Path,
        revision: str | None = None,
        *,
        cache_dir: str | Path | None = None,
        token: str | bool | None = None,
        local_files_only: bool = False,
        trust_legacy_checkpoint: bool = False,
        load_encodec_weights: bool = False,
        encodec_checkpoint: str | Path | None = None,
        trust_official_encodec_pickle: bool = False,
    ) -> Vocos:
        """
        Class method to create a new Vocos model instance from a pre-trained model stored in the Hugging Face model hub.
        """
        source = Path(repo_id).expanduser()
        pinned_revision = _OFFICIAL_REVISIONS.get(str(repo_id))
        official_repo = pinned_revision is not None
        official_checkpoint = official_repo and revision in {None, pinned_revision}
        if source.exists():
            root = source.parent if source.is_file() else source
        else:
            if is_explicit_local_path(repo_id):
                raise FileNotFoundError(f"Local Vocos path was not found: {source}.")
            resolved_revision = (
                revision
                or pinned_revision
            )
            root = download_hugging_face_snapshot(
                str(repo_id),
                revision=resolved_revision,
                cache_dir=cache_dir,
                token=token,
                local_files_only=local_files_only,
                allow_patterns=(
                    "config.json",
                    "config.yaml",
                    "*.safetensors",
                    "pytorch_model.bin",
                    "model.bin",
                    "model.pt",
                    "model.pth",
                ),
            )
        config_path = (
            source
            if source.is_file() and source.suffix.lower() in {".json", ".yaml", ".yml"}
            else _configuration_path(root)
        )
        weight_path = (
            source
            if source.is_file() and source.suffix.lower() not in {".json", ".yaml", ".yml"}
            else _weight_path(root)
        )
        if (
            weight_path.suffix.lower() != ".safetensors"
            and not trust_legacy_checkpoint
            and not official_checkpoint
        ):
            raise ValueError(
                "Loading a local or unpinned legacy Vocos pickle requires "
                "`trust_legacy_checkpoint=True`; convert it to Safetensors "
                "for steady-state use."
            )
        for name, value in (
            ("load_encodec_weights", load_encodec_weights),
            ("trust_official_encodec_pickle", trust_official_encodec_pickle),
        ):
            if not isinstance(value, bool):
                raise TypeError(f"`{name}` must be a boolean.")
        if encodec_checkpoint is not None and not isinstance(
            encodec_checkpoint,
            (str, Path),
        ):
            raise TypeError("`encodec_checkpoint` must be path-like or None.")
        if encodec_checkpoint is not None:
            load_encodec_weights = True
        if trust_official_encodec_pickle and not load_encodec_weights:
            raise ValueError(
                "`trust_official_encodec_pickle=True` requires "
                "`load_encodec_weights=True` or `encodec_checkpoint`."
            )
        if (
            load_encodec_weights
            and encodec_checkpoint is None
            and not trust_official_encodec_pickle
        ):
            raise ValueError(
                "The official Encodec release is a legacy `.th` archive. "
                "Set `trust_official_encodec_pickle=True` explicitly, or "
                "provide a converted Safetensors `encodec_checkpoint`."
            )

        model = cls.from_hparams(config_path)
        if (
            load_encodec_weights
            and isinstance(model.feature_extractor, EncodecFeatures)
        ):
            from voicehub.components.audio.codecs.encodec import (
                load_encodec_model,
            )

            encodec = load_encodec_model(
                model.feature_extractor.encodec_model_name,
                checkpoint=encodec_checkpoint,
                cache_dir=cache_dir,
                local_files_only=local_files_only,
                trust_official_pickle=trust_official_encodec_pickle,
            )
            model.feature_extractor.attach_encodec(encodec)
        state_dict = _tensor_state(weight_path)
        _load_validated_state(
            model,
            state_dict,
            allow_external_codec_omission=isinstance(
                model.feature_extractor,
                EncodecFeatures,
            ),
        )
        model.eval()
        return model

    def export_safetensors(self, path: str | Path) -> Path:
        """Export a complete, dependency-free Vocos state."""
        feature_extractor = self.feature_extractor
        include_encodec = (
            not isinstance(feature_extractor, EncodecFeatures)
            or feature_extractor.encodec_weights_available
        )
        state = {
            name: tensor.detach().cpu().contiguous()
            for name, tensor in self.state_dict().items()
            if (
                include_encodec
                or not name.startswith("feature_extractor.encodec.")
            )
        }
        return save_safetensors(
            state,
            path,
            metadata={
                "format": "voicehub-native-vocos-v1",
                "source_revision": "c859e3b7b534f3776a357983029d34170ddd6fc3",
                "source_license": "MIT",
                "encodec_weights": (
                    "included"
                    if include_encodec
                    else "external-code-decoding-only"
                ),
            },
        )

    def forward(self, audio_input: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        """
        Method to run a copy-synthesis from audio waveform. The feature extractor first processes the audio input,
        which is then passed through the backbone and the head to reconstruct the audio output.

        Args:
            audio_input (Tensor): The input tensor representing the audio waveform of shape (B, T),
                                        where B is the batch size and L is the waveform length.


        Returns:
            Tensor: The output tensor representing the reconstructed audio waveform of shape (B, T).
        """
        features = self.feature_extractor(audio_input, **kwargs)
        audio_output = self.decode(features, **kwargs)
        return audio_output

    def decode(self, features_input: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        """
        Method to decode audio waveform from already calculated features. The features input is passed through
        the backbone and the head to reconstruct the audio output.

        Args:
            features_input (Tensor): The input tensor of features of shape (B, C, L), where B is the batch size,
                                     C denotes the feature dimension, and L is the sequence length.

        Returns:
            Tensor: The output tensor representing the reconstructed audio waveform of shape (B, T).
        """
        x = self.backbone(features_input, **kwargs)
        audio_output = self.head(x)
        return audio_output

    def codes_to_features(self, codes: torch.Tensor) -> torch.Tensor:
        """
        Transforms an input sequence of discrete tokens (codes) into feature embeddings using the feature extractor's
        codebook weights.

        Args:
            codes (Tensor): The input tensor. Expected shape is (K, L) or (K, B, L),
                            where K is the number of codebooks, B is the batch size and L is the sequence length.

        Returns:
            Tensor: Features of shape (B, C, L), where B is the batch size, C denotes the feature dimension,
                    and L is the sequence length.
        """
        if not isinstance(self.feature_extractor, EncodecFeatures):
            raise TypeError(
                "Code conversion requires an EncodecFeatures extractor."
            )

        if codes.dim() == 2:
            codes = codes.unsqueeze(1)
        if codes.dim() != 3:
            raise ValueError("Encodec codes must have shape [codebook, batch, frames].")
        if codes.shape[0] > self.feature_extractor.num_q:
            raise ValueError("Encodec codes use more codebooks than this Vocos model.")

        n_bins = self.feature_extractor.encodec.quantizer.bins
        if codes.numel() and (
            int(codes.min().item()) < 0 or int(codes.max().item()) >= n_bins
        ):
            raise ValueError("Encodec code index is outside the codec vocabulary.")
        offsets = torch.arange(0, n_bins * len(codes), n_bins, device=codes.device)
        embeddings_idxs = codes + offsets.view(-1, 1, 1)
        features = torch.nn.functional.embedding(embeddings_idxs, self.feature_extractor.codebook_weights).sum(dim=0)
        features = features.transpose(1, 2)

        return features


__all__ = [
    "Vocos",
    "instantiate_class",
    "register_vocos_component",
]

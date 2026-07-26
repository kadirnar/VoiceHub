"""CosyVoice 1/2/3 integration backed by vendored upstream source."""

from __future__ import annotations

from functools import wraps
from pathlib import Path

from voicehub.configuration_utils import VoiceHubConfig
from voicehub.dependencies import import_optional
from voicehub.modeling_outputs import TTSOutput
from voicehub.modeling_utils import PreTrainedTTSModel


def _install_vendored_yaml_loader(runtime) -> None:
    """Make upstream checkpoint YAML resolve VoiceHub's vendored modules."""
    current_loader = runtime.load_hyperpyyaml
    if getattr(current_loader, "__voicehub_vendored_yaml__", False):
        return

    from voicehub.models.cosyvoice.training import _load_vendored_hyperpyyaml

    @wraps(current_loader)
    def vendored_loader(yaml_stream, *args, **kwargs):
        return _load_vendored_hyperpyyaml(
            current_loader,
            yaml_stream,
            *args,
            **kwargs,
        )

    vendored_loader.__voicehub_vendored_yaml__ = True
    runtime.load_hyperpyyaml = vendored_loader


class CosyVoiceConfig(VoiceHubConfig):
    """Configuration shared by CosyVoice checkpoint generations."""

    model_type = "cosyvoice"

    def __init__(
        self,
        *,
        load_jit: bool = False,
        load_trt: bool = False,
        load_vllm: bool = False,
        fp16: bool = False,
        training_component: str = "llm",
        sample_rate: int = 24000,
        **kwargs,
    ):
        super().__init__(sample_rate=sample_rate, **kwargs)
        self.load_jit = load_jit
        self.load_trt = load_trt
        self.load_vllm = load_vllm
        self.fp16 = fp16
        self.training_component = training_component

    def validate(self) -> None:
        component = self.training_component.lower().replace("-", "_")
        allowed = {
            "llm",
            "language_model",
            "flow",
            "hifigan_generator",
            "hifigan_discriminator",
        }
        if component not in allowed:
            raise ValueError(
                "training_component must select llm, flow, "
                "hifigan_generator, or hifigan_discriminator.")


class CosyVoiceForTextToSpeech(PreTrainedTTSModel):
    """Unified source-native CosyVoice synthesis interface."""

    config_class = CosyVoiceConfig
    default_model_name_or_path = "FunAudioLLM/Fun-CosyVoice3-0.5B-2512"

    def __init__(
        self,
        config: CosyVoiceConfig | str | None = None,
        *,
        model_path: str | None = None,
        device: str = "auto",
        lazy_load: bool = True,
        **config_overrides,
    ):
        config = self._coerce_config(
            config,
            model_path=model_path,
            **config_overrides,
        )
        self._cosyvoice_training_backend = None
        super().__init__(config, device=device, lazy_load=lazy_load)

    def _load_pretrained_model(self) -> None:
        if self.is_training_load:
            self._load_training_model()
            return
        self._load_full_inference_model()

    def _load_training_model(self) -> None:
        from voicehub.models.cosyvoice.training import load_cosyvoice_training_backend

        backend = load_cosyvoice_training_backend(
            self.config.name_or_path,
            self.config.training_component,
        )
        self.model = backend
        self._cosyvoice_training_backend = backend
        self.config.name_or_path = str(backend.artifacts.model_directory)
        self.config.sample_rate = backend.sample_rate

    def _load_full_inference_model(self) -> None:
        runtime = import_optional(
            "voicehub.models.cosyvoice.source.cosyvoice.cli.cosyvoice",
            model_type="cosyvoice",
            install_extra="cosyvoice",
        )
        _install_vendored_yaml_loader(runtime)
        model_directory = Path(self.config.name_or_path).expanduser()
        if not model_directory.is_dir():
            model_directory = Path(runtime.snapshot_download(self.config.name_or_path))

        common = {
            "model_dir": str(model_directory),
            "load_trt": self.config.load_trt,
            "fp16": self.config.fp16,
        }
        if (model_directory / "cosyvoice3.yaml").is_file():
            model_class = runtime.CosyVoice3
            common["load_vllm"] = self.config.load_vllm
        elif (model_directory / "cosyvoice2.yaml").is_file():
            model_class = runtime.CosyVoice2
            common["load_jit"] = self.config.load_jit
            common["load_vllm"] = self.config.load_vllm
        elif (model_directory / "cosyvoice.yaml").is_file():
            model_class = runtime.CosyVoice
            common["load_jit"] = self.config.load_jit
        else:
            raise ValueError(f"{model_directory} does not contain a CosyVoice model config.")

        self.model = model_class(**common)
        self._cosyvoice_training_backend = None
        self.config.name_or_path = str(model_directory)
        self.config.sample_rate = self.model.sample_rate

    def _validate_training_runtime(self) -> None:
        if self.config.load_jit or self.config.load_trt or self.config.load_vllm:
            raise ValueError(
                "CosyVoice fine-tuning requires unfused PyTorch modules; "
                "disable JIT, TensorRT, and vLLM.")

    @staticmethod
    def _component_state_on_cpu(component):
        return {
            name: (value.detach().cpu().clone() if hasattr(value, "detach") else value)
            for name, value in component.state_dict().items()
        }

    @staticmethod
    def _restore_component_state(component, state_dict) -> None:
        try:
            incompatible = component.load_state_dict(
                state_dict,
                strict=True,
            )
        except TypeError:
            incompatible = component.load_state_dict(state_dict)
        missing = tuple(getattr(incompatible, "missing_keys", ()))
        unexpected = tuple(getattr(incompatible, "unexpected_keys", ()))
        if missing or unexpected:
            raise RuntimeError(
                "The fine-tuned CosyVoice component is incompatible with "
                "the full inference runtime "
                f"(missing={missing}, unexpected={unexpected}).")

    def _prepare_for_training(self) -> None:
        from voicehub.models.cosyvoice.training import _canonical_training_component

        selected = _canonical_training_component(self.config.training_component, )
        backend = self._cosyvoice_training_backend
        if backend is not None and backend.component_name == selected:
            # Generation temporarily attaches a full serving wrapper. Keep
            # the original backend as the stable owner of parameters and
            # optimizer references, then reattach it for continued training.
            self.model = backend
            move_component = getattr(
                backend.selected_component,
                "to",
                None,
            )
            if callable(move_component):
                move_component(self.device)
            return

        preserved_state = None
        runtime = getattr(self.model, "model", None)
        component = getattr(runtime, selected, None) if runtime is not None else None
        if component is not None and hasattr(component, "state_dict"):
            preserved_state = self._component_state_on_cpu(component)

        previous_model = self.model
        previous_backend = backend
        self.model = None
        self._cosyvoice_training_backend = None
        try:
            self._load_training_model()
            if preserved_state is not None:
                self._restore_component_state(
                    self._cosyvoice_training_backend.selected_component,
                    preserved_state,
                )
        except BaseException:
            self.model = previous_model
            self._cosyvoice_training_backend = previous_backend
            raise

    def _ensure_full_inference_runtime(self) -> None:
        """Rebuild inference modules before synthesizing a Trainer artifact."""
        backend = self._cosyvoice_training_backend
        if backend is None or self.model is not backend:
            return

        previous_model = self.model
        self.model = None
        self._cosyvoice_training_backend = None
        try:
            self._load_full_inference_model()
            runtime = getattr(self.model, "model", None)
            component = (getattr(runtime, backend.component_name, None) if runtime is not None else None)
            if component is None:
                raise TypeError(
                    "The full CosyVoice inference runtime does not expose "
                    f"the fine-tuned {backend.component_name!r} component.")
            selected_component = backend.selected_component
            move_component = getattr(selected_component, "to", None)
            if callable(move_component):
                move_component(getattr(runtime, "device", self.device))
            evaluate_component = getattr(selected_component, "eval", None)
            if callable(evaluate_component):
                evaluate_component()
            setattr(runtime, backend.component_name, selected_component)
            # Retain the component-only backend off the serving path. A
            # prepared adapter and its optimizer continue to own these exact
            # parameter objects if training resumes after synthesis. Replacing
            # the freshly loaded copy also avoids retaining the selected model
            # twice for the lifetime of the serving wrapper.
            self._cosyvoice_training_backend = backend
        except BaseException:
            self.model = previous_model
            self._cosyvoice_training_backend = backend
            raise

    def _select_inference(
        self,
        *,
        mode: str,
        text: str,
        speaker: str | None,
        prompt_text: str,
        speaker_audio_path: str | None,
        instruct_text: str,
        stream: bool,
        speed: float,
    ):
        if mode == "auto":
            mode = "zero_shot" if speaker_audio_path else "sft"

        common = {"stream": stream, "speed": speed}
        if mode == "sft":
            if speaker is None:
                speakers = self.model.list_available_spks()
                if not speakers:
                    raise ValueError(
                        "This checkpoint has no built-in speaker; provide "
                        "speaker_audio_path for zero-shot synthesis.")
                speaker = speakers[0]
            return self.model.inference_sft(text, speaker, **common)
        if mode == "zero_shot":
            if not speaker_audio_path or not prompt_text:
                raise ValueError("zero_shot requires speaker_audio_path and prompt_text.")
            return self.model.inference_zero_shot(
                text,
                prompt_text,
                speaker_audio_path,
                **common,
            )
        if mode == "cross_lingual":
            if not speaker_audio_path:
                raise ValueError("cross_lingual requires speaker_audio_path.")
            return self.model.inference_cross_lingual(
                text,
                speaker_audio_path,
                **common,
            )
        if mode == "instruct":
            if hasattr(self.model, "inference_instruct2"):
                if not speaker_audio_path:
                    raise ValueError("CosyVoice 2/3 instruct mode requires speaker_audio_path.")
                return self.model.inference_instruct2(
                    text,
                    instruct_text,
                    speaker_audio_path,
                    **common,
                )
            if speaker is None:
                raise ValueError("CosyVoice 1 instruct mode requires speaker.")
            return self.model.inference_instruct(
                text,
                speaker,
                instruct_text,
                **common,
            )
        raise ValueError("mode must be one of: auto, sft, zero_shot, cross_lingual, instruct.")

    def _generate(
        self,
        text: str,
        *,
        output_file: str | None = None,
        mode: str = "auto",
        speaker: str | None = None,
        prompt_text: str = "",
        speaker_audio_path: str | None = None,
        instruct_text: str = "",
        stream: bool = False,
        speed: float = 1.0,
    ) -> TTSOutput:
        self.load()
        self._ensure_full_inference_runtime()
        generator = self._select_inference(
            mode=mode,
            text=text,
            speaker=speaker,
            prompt_text=prompt_text,
            speaker_audio_path=speaker_audio_path,
            instruct_text=instruct_text,
            stream=stream,
            speed=speed,
        )
        chunks = [item["tts_speech"].reshape(-1) for item in generator]
        if not chunks:
            raise RuntimeError("CosyVoice returned no audio chunks.")

        torch = import_optional(
            "torch",
            model_type="cosyvoice",
            install_extra="cosyvoice",
        )
        output = TTSOutput(
            audio=torch.cat(chunks),
            sample_rate=self.sample_rate,
            metadata={"mode": mode},
        )
        if output_file:
            output.save(output_file)
        return output


CosyVoiceTTS = CosyVoiceForTextToSpeech

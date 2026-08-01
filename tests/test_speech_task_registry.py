import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import ModuleType
from unittest.mock import patch

from voicehub.auto import (
    AutoConfig,
    AutoModel,
    AutoModelForSpeechRecognition,
    AutoModelForTextToSpeech,
    AutoModelForVoiceActivityDetection,
)
from voicehub.automodel import MODEL_TYPE_TO_MODEL_CLASS_NAME, AutoInferenceModel
from voicehub.configuration_utils import VoiceHubConfig
from voicehub.registry import (
    MODEL_ALIASES,
    MODEL_REGISTRY,
    ModelRegistry,
    ModelSpec,
    get_model_spec,
    list_model_specs,
    register_model_alias,
    register_model_spec,
    unregister_model_alias,
    unregister_model_spec,
)
from voicehub.tasks import SpeechTask


class _FakeSpeechModel:

    def __init__(self, config, **kwargs):
        self.config = config
        self.init_kwargs = kwargs
        self.loaded = False
        self.inference_strategy = None

    def set_inference_strategy(self, strategy):
        self.inference_strategy = strategy

    def load(self):
        self.loaded = True

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path,
        *,
        config,
        inference_strategy=None,
        **kwargs,
    ):
        model = cls(config, **kwargs)
        model.pretrained_model_name_or_path = pretrained_model_name_or_path
        model.inference_strategy = inference_strategy
        return model


class _FakeASRModel(_FakeSpeechModel):
    pass


class _FakeVADModel(_FakeSpeechModel):
    pass


class _ExtensionASRConfig(VoiceHubConfig):
    model_type = "test-auto-register-asr"


class SpeechTaskRegistryTests(unittest.TestCase):
    ASR_MODEL_TYPE = "test-speech-registry-asr"
    ASR_ALIAS = "test-speech-registry-stt"
    VAD_MODEL_TYPE = "test-speech-registry-vad"
    VAD_ALIAS = "test-speech-registry-activity"
    FAKE_MODULE = "tests._voicehub_fake_speech_backends"

    def tearDown(self):
        for model_type in (self.ASR_MODEL_TYPE, self.VAD_MODEL_TYPE):
            unregister_model_spec(model_type, missing_ok=True)

    def _spec(
        self,
        model_type,
        class_name,
        task,
        *,
        module=None,
    ):
        return ModelSpec(
            model_type=model_type,
            module=module or self.FAKE_MODULE,
            class_name=class_name,
            default_model_path=f"acme/{model_type}",
            install_extra=model_type,
            task=task,
            architecture="test-speech-family",
        )

    def _register_task_backends(self):
        register_model_spec(
            self._spec(
                self.ASR_MODEL_TYPE,
                "_FakeASRModel",
                SpeechTask.AUTOMATIC_SPEECH_RECOGNITION,
            ),
            aliases=(self.ASR_ALIAS, ),
        )
        register_model_spec(
            self._spec(
                self.VAD_MODEL_TYPE,
                "_FakeVADModel",
                "vad",
            ),
            aliases=(self.VAD_ALIAS, ),
        )

    def test_model_spec_normalizes_task_metadata(self):
        tts_spec = self._spec(
            self.ASR_MODEL_TYPE,
            "_FakeASRModel",
            SpeechTask.TEXT_TO_SPEECH,
        )
        asr_spec = self._spec(
            self.ASR_MODEL_TYPE,
            "_FakeASRModel",
            "stt",
        )

        self.assertIs(tts_spec.task, SpeechTask.TEXT_TO_SPEECH)
        self.assertIs(
            asr_spec.task,
            SpeechTask.AUTOMATIC_SPEECH_RECOGNITION,
        )
        self.assertEqual(
            asr_spec.capabilities,
            (SpeechTask.AUTOMATIC_SPEECH_RECOGNITION.value, ),
        )
        self.assertEqual(asr_spec.architecture, "test-speech-family")
        self.assertTrue(asr_spec.supports_task("asr"))
        self.assertFalse(asr_spec.supports_task("vad"))

    def test_registration_aliases_and_task_filters_are_live(self):
        self._register_task_backends()

        self.assertIs(
            get_model_spec(self.ASR_ALIAS),
            MODEL_REGISTRY[self.ASR_MODEL_TYPE],
        )
        self.assertEqual(
            MODEL_ALIASES[self.VAD_ALIAS],
            self.VAD_MODEL_TYPE,
        )
        self.assertEqual(
            MODEL_TYPE_TO_MODEL_CLASS_NAME[self.ASR_MODEL_TYPE],
            "_FakeASRModel",
        )
        self.assertIn(
            self.ASR_MODEL_TYPE,
            {spec.model_type
             for spec in list_model_specs(task="speech-to-text")},
        )
        self.assertNotIn(
            self.VAD_MODEL_TYPE,
            {spec.model_type
             for spec in list_model_specs(task="speech-to-text")},
        )

    def test_isolated_model_registry_keeps_live_read_only_views(self):
        spec = ModelSpec.from_classes(
            model_type=self.ASR_MODEL_TYPE,
            model_class=_FakeASRModel,
            config_class=_ExtensionASRConfig,
            task="asr",
        )
        registry = ModelRegistry()
        specs = registry.specs
        aliases = registry.aliases

        registry.register(spec, aliases=(self.ASR_ALIAS, ))

        self.assertIs(specs[self.ASR_MODEL_TYPE], spec)
        self.assertEqual(aliases[self.ASR_ALIAS], self.ASR_MODEL_TYPE)
        self.assertIs(registry.get(self.ASR_ALIAS), spec)
        self.assertEqual(tuple(registry), (self.ASR_MODEL_TYPE, ))
        with self.assertRaises(TypeError):
            specs["other"] = spec
        registry.clear()
        self.assertEqual(dict(specs), {})
        self.assertEqual(dict(aliases), {})

    def test_auto_factory_registers_and_dispatches_extension_models(self):
        model_type = _ExtensionASRConfig.model_type
        alias = "test-auto-register-stt"
        try:
            spec = AutoModelForSpeechRecognition.register(
                _ExtensionASRConfig,
                _FakeASRModel,
                default_model_path="acme/extension-asr",
                aliases=(alias, ),
            )
            model = AutoModel.from_pretrained(
                "acme/checkpoint",
                model_type=alias,
                config_kwargs={"sample_rate": 22_050},
                marker="extension",
            )
        finally:
            AutoModel.unregister(
                model_type,
                missing_ok=True,
            )

        self.assertEqual(spec.model_type, model_type)
        self.assertIsInstance(model, _FakeASRModel)
        self.assertEqual(model.init_kwargs["marker"], "extension")
        self.assertEqual(model.config.sample_rate, 22_050)

    def test_native_filter_resolves_owned_architecture_contracts(self):
        native_asr = list_model_specs(
            task="asr",
            native=True,
        )

        self.assertIn(
            "asr_whisper",
            {spec.model_type
             for spec in native_asr},
        )
        whisper = get_model_spec("asr_whisper")
        self.assertTrue(whisper.is_voicehub_native)
        self.assertEqual(
            whisper.native_architecture.architecture_id,
            "whisper",
        )
        generic = get_model_spec("asr_transformers")
        self.assertTrue(generic.is_voicehub_native)
        self.assertEqual(
            generic.native_architecture.architecture_id,
            "native-asr-dispatch",
        )
        sherpa = get_model_spec("vad_sherpa_onnx")
        self.assertTrue(sherpa.is_voicehub_native)
        self.assertEqual(
            sherpa.native_architecture.architecture_id,
            "native-vad-dispatch",
        )
        with self.assertRaisesRegex(TypeError, "native"):
            list_model_specs(native="yes")

    def test_alias_registration_is_idempotent_only_when_requested(self):
        register_model_spec(self._spec(
            self.ASR_MODEL_TYPE,
            "_FakeASRModel",
            "asr",
        ))
        register_model_alias(self.ASR_ALIAS, self.ASR_MODEL_TYPE)

        with self.assertRaisesRegex(ValueError, "already registered"):
            register_model_alias(self.ASR_ALIAS, self.ASR_MODEL_TYPE)
        register_model_alias(
            self.ASR_ALIAS.upper(),
            self.ASR_MODEL_TYPE,
            exist_ok=True,
        )
        self.assertEqual(
            unregister_model_alias(self.ASR_ALIAS),
            self.ASR_MODEL_TYPE,
        )
        self.assertNotIn(self.ASR_ALIAS, MODEL_ALIASES)

    def test_task_factories_load_only_matching_backends(self):
        self._register_task_backends()
        fake_module = ModuleType(self.FAKE_MODULE)
        fake_module._FakeASRModel = _FakeASRModel
        fake_module._FakeVADModel = _FakeVADModel

        with patch.dict(sys.modules, {self.FAKE_MODULE: fake_module}):
            config = AutoConfig.for_model(
                self.ASR_ALIAS,
                sample_rate=16000,
            )
            asr_model = AutoModelForSpeechRecognition.from_config(config)
            vad_model = AutoModelForVoiceActivityDetection.from_pretrained(
                "acme/vad-checkpoint",
                model_type=self.VAD_ALIAS,
                marker="vad",
            )

        self.assertIsInstance(asr_model, _FakeASRModel)
        self.assertEqual(asr_model.config.sample_rate, 16000)
        self.assertIsInstance(vad_model, _FakeVADModel)
        self.assertEqual(
            vad_model.pretrained_model_name_or_path,
            "acme/vad-checkpoint",
        )
        self.assertEqual(vad_model.init_kwargs["marker"], "vad")

    def test_explicit_model_type_cannot_override_a_different_typed_config(self):
        self._register_task_backends()
        config = AutoConfig.for_model(self.ASR_MODEL_TYPE)

        with self.assertRaisesRegex(ValueError, "supplied config targets"):
            AutoModelForSpeechRecognition.from_pretrained(
                "acme/other-asr-checkpoint",
                model_type="asr_transformers",
                config=config,
            )

    def test_explicit_model_type_can_specialize_a_generic_config(self):
        register_model_spec(self._spec(
            self.ASR_MODEL_TYPE,
            "_FakeASRModel",
            "asr",
        ))
        fake_module = ModuleType(self.FAKE_MODULE)
        fake_module._FakeASRModel = _FakeASRModel
        config = VoiceHubConfig(sample_rate=16_000)

        with patch.dict(sys.modules, {self.FAKE_MODULE: fake_module}):
            model = AutoModelForSpeechRecognition.from_pretrained(
                "acme/asr-checkpoint",
                model_type=self.ASR_MODEL_TYPE,
                config=config,
            )

        self.assertIsInstance(model, _FakeASRModel)
        self.assertEqual(config.model_type, self.ASR_MODEL_TYPE)

    def test_task_mismatch_is_rejected_before_runtime_import(self):
        missing_module = "tests._voicehub_module_that_must_not_import"
        register_model_spec(
            self._spec(
                self.ASR_MODEL_TYPE,
                "_MissingASRModel",
                "asr",
                module=missing_module,
            ))
        config = VoiceHubConfig()
        config.model_type = self.ASR_MODEL_TYPE

        with self.assertRaisesRegex(
                ValueError,
                "Use AutoModelForSpeechRecognition",
        ):
            AutoModelForTextToSpeech.from_config(config)
        with self.assertRaisesRegex(
                ValueError,
                "Use AutoModelForSpeechRecognition",
        ):
            AutoInferenceModel.from_pretrained(self.ASR_MODEL_TYPE)
        self.assertNotIn(missing_module, sys.modules)

    def test_legacy_inference_discovery_remains_tts_only(self):
        self._register_task_backends()

        available = {spec.model_type for spec in AutoInferenceModel.available_models()}
        self.assertIn("dia", available)
        self.assertNotIn(self.ASR_MODEL_TYPE, available)
        self.assertNotIn(self.VAD_MODEL_TYPE, available)

    def test_empty_task_factory_sources_use_provider_defaults_without_hub_lookup(self):
        with patch.object(
                AutoConfig,
                "from_pretrained",
                side_effect=AssertionError("empty sources must not query config"),
        ):
            asr = AutoModelForSpeechRecognition.from_pretrained("   ")
            vad = AutoModelForVoiceActivityDetection.from_pretrained()

        self.assertEqual(asr.config.name_or_path, "openai/whisper-small")
        self.assertEqual(vad.config.name_or_path, "")
        self.assertFalse(asr.is_loaded)
        self.assertFalse(vad.is_loaded)
        with self.assertRaisesRegex(ValueError, "checkpoint-family provider"):
            vad.load()

    def test_upstream_model_type_aliases_are_resolved_in_task_context(self):
        with tempfile.TemporaryDirectory() as directory:
            config_path = Path(directory) / "config.json"
            config_path.write_text(
                json.dumps({
                    "model_type": "wav2vec2",
                    "architectures": ["Wav2Vec2ForAudioFrameClassification"],
                }),
                encoding="utf-8",
            )

            asr = AutoModelForSpeechRecognition.from_pretrained(directory)
            vad = AutoModelForVoiceActivityDetection.from_pretrained(directory)

        self.assertEqual(asr.config.model_type, "asr_transformers")
        self.assertEqual(vad.config.model_type, "vad_transformers")
        resolved_directory = str(Path(directory).resolve())
        self.assertEqual(asr.config.name_or_path, resolved_directory)
        self.assertEqual(vad.config.name_or_path, resolved_directory)

    def test_factory_config_overrides_cannot_bypass_provider_validation(self):
        invalid_factories = (
            lambda: AutoModelForSpeechRecognition.from_pretrained(architecture_family="not-asr", ),
            lambda: AutoModelForVoiceActivityDetection.from_pretrained(architecture_family="not-vad", ),
            lambda: AutoModelForVoiceActivityDetection.from_pretrained(threshold=1.5, ),
            lambda: AutoModelForVoiceActivityDetection.from_pretrained(
                model_kwargs={
                    "api_key": "must-not-be-persisted",
                }, ),
        )
        for factory in invalid_factories:
            with self.subTest(factory=factory), self.assertRaises((TypeError, ValueError)):
                factory()

    def test_provider_overrides_rebuild_without_mutating_the_caller_config(self):
        config = AutoConfig.for_model("vad_transformers")

        model = AutoModelForVoiceActivityDetection.from_config(
            config,
            threshold=0.75,
        )

        self.assertIsNot(model.config, config)
        self.assertEqual(config.inference_config["threshold"], 0.5)
        self.assertEqual(model.config.inference_config["threshold"], 0.75)


if __name__ == "__main__":
    unittest.main()

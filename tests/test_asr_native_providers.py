import json
import subprocess
import sys
import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import numpy as np

from voicehub.modeling_outputs import ASROutput
from voicehub.models.asr_native._shared import normalize_asr_result
from voicehub.models.asr_native.configuration import (
    ESPnetASRConfig,
    FasterWhisperConfig,
    FunASRConfig,
    NeMoASRConfig,
    OpenAIWhisperConfig,
    SpeechBrainASRConfig,
    WeNetASRConfig,
    WhisperXConfig,
)
from voicehub.models.asr_native.espnet import ESPnetASRForSpeechRecognition
from voicehub.models.asr_native.faster_whisper import FasterWhisperForSpeechRecognition
from voicehub.models.asr_native.funasr import FunASRForSpeechRecognition
from voicehub.models.asr_native.nemo import NeMoASRForSpeechRecognition
from voicehub.models.asr_native.openai_whisper import OpenAIWhisperForSpeechRecognition
from voicehub.models.asr_native.speechbrain import SpeechBrainASRForSpeechRecognition
from voicehub.models.asr_native.wenet import WeNetASRForSpeechRecognition
from voicehub.models.asr_native.whisperx import WhisperXForSpeechRecognition
from voicehub.registry import get_model_spec

PROJECT_ROOT = Path(__file__).resolve().parents[1]


@contextmanager
def _temporary_modules(modules):
    missing = object()
    originals = {name: sys.modules.get(name, missing) for name in modules}
    sys.modules.update(modules)
    try:
        yield
    finally:
        for name, original in originals.items():
            if original is missing:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = original


class _InferenceRuntime:

    def __init__(self):
        self.training = True
        self.eval_calls = 0

    def eval(self):
        self.training = False
        self.eval_calls += 1
        return self

    def train(self):
        self.training = True
        return self


class NativeASRProviderContractTests(unittest.TestCase):

    def test_importing_wrappers_does_not_import_provider_packages(self):
        code = """
import json
import sys
from voicehub.models.asr_native.espnet import ESPnetASRForSpeechRecognition
from voicehub.models.asr_native.faster_whisper import FasterWhisperForSpeechRecognition
from voicehub.models.asr_native.funasr import FunASRForSpeechRecognition
from voicehub.models.asr_native.nemo import NeMoASRForSpeechRecognition
from voicehub.models.asr_native.openai_whisper import OpenAIWhisperForSpeechRecognition
from voicehub.models.asr_native.speechbrain import SpeechBrainASRForSpeechRecognition
from voicehub.models.asr_native.wenet import WeNetASRForSpeechRecognition
from voicehub.models.asr_native.whisperx import WhisperXForSpeechRecognition
providers = (
    "espnet2",
    "faster_whisper",
    "funasr",
    "nemo",
    "speechbrain",
    "wenet",
    "whisper",
    "whisperx",
)
print(json.dumps({name: name in sys.modules for name in providers}))
"""
        result = subprocess.run(
            [sys.executable, "-c", code],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )

        self.assertEqual(
            json.loads(result.stdout.strip().splitlines()[-1]),
            {
                "espnet2": False,
                "faster_whisper": False,
                "funasr": False,
                "nemo": False,
                "speechbrain": False,
                "wenet": False,
                "whisper": False,
                "whisperx": False,
            },
        )

    def test_normalizer_preserves_segment_word_and_provider_metadata(self):
        result = {
            "text":
            " hello world ",
            "language":
            "en",
            "segments": [
                {
                    "id":
                    7,
                    "start":
                    100,
                    "end":
                    700,
                    "text":
                    "hello world",
                    "confidence":
                    0.8,
                    "words": [
                        {
                            "word": "hello",
                            "start": 100,
                            "end": 300,
                            "probability": 0.9,
                            "speaker": "SPEAKER_00",
                        },
                        {
                            "word": "world",
                            "start": 400,
                            "end": 700,
                            "score": 0.75,
                        },
                    ],
                },
            ],
        }

        output = normalize_asr_result(
            result,
            backend="test-provider",
            duration=1.0,
            timestamp_scale=0.001,
        )

        self.assertEqual(output.text, "hello world")
        self.assertEqual(output.metadata["backend"], "test-provider")
        self.assertEqual(output.segments[0].start, 0.1)
        self.assertEqual(output.segments[0].end, 0.7)
        self.assertEqual(output.segments[0].metadata["id"], 7)
        self.assertEqual(output.segments[0].words[0].speaker, "SPEAKER_00")
        self.assertEqual(output.segments[0].words[1].confidence, 0.75)

    def test_inference_only_and_upstream_training_boundaries_are_explicit(self):
        cases = (
            (
                FasterWhisperForSpeechRecognition(FasterWhisperConfig()),
                "CTranslate2",
            ),
            (
                OpenAIWhisperForSpeechRecognition(OpenAIWhisperConfig()),
                "asr_transformers",
            ),
            (
                WhisperXForSpeechRecognition(WhisperXConfig()),
                "inference/alignment",
            ),
            (
                NeMoASRForSpeechRecognition(NeMoASRConfig()),
                "Lightning/Hydra",
            ),
            (
                SpeechBrainASRForSpeechRecognition(SpeechBrainASRConfig()),
                "Brain",
            ),
            (
                FunASRForSpeechRecognition(FunASRConfig()),
                "upstream",
            ),
            (
                ESPnetASRForSpeechRecognition(ESPnetASRConfig()),
                "ASRTask",
            ),
            (
                WeNetASRForSpeechRecognition(WeNetASRConfig()),
                "distributed",
            ),
        )

        for model, message in cases:
            with (
                    self.subTest(model=model.config.model_type),
                    self.assertRaisesRegex(ValueError, message),
            ):
                model._validate_training_runtime()

    def test_runtime_tokens_are_not_serialized(self):
        models = (
            WhisperXForSpeechRecognition(
                WhisperXConfig(),
                token="whisperx-secret",
            ),
            NeMoASRForSpeechRecognition(
                NeMoASRConfig(),
                token="nemo-secret",
            ),
            SpeechBrainASRForSpeechRecognition(
                SpeechBrainASRConfig(),
                token="speechbrain-secret",
            ),
        )

        for model in models:
            with self.subTest(model=model.config.model_type):
                serialized = json.dumps(model.config.to_dict())
                self.assertNotIn("secret", serialized)
                self.assertNotIn("token", serialized)

    def test_configs_reject_nested_credentials_and_managed_loader_options(self):
        with self.assertRaisesRegex(ValueError, "credentials"):
            FasterWhisperConfig(
                inference_config={
                    "provider": {
                        "api_key": "secret",
                    },
                }, )
        with self.assertRaisesRegex(ValueError, "managed"):
            NeMoASRConfig(
                model_kwargs={
                    "map_location": "cuda",
                }, )

    def test_funasr_wrapper_and_registry_use_the_same_default_checkpoint(self):
        self.assertEqual(
            FunASRForSpeechRecognition.default_model_name_or_path,
            get_model_spec("asr_funasr").default_model_path,
        )

    def test_cpu_cuda_native_runtimes_never_auto_select_mps(self):
        fake_torch = ModuleType("torch")
        fake_torch.cuda = SimpleNamespace(is_available=lambda: False)
        fake_torch.backends = SimpleNamespace(mps=SimpleNamespace(is_available=lambda: True), )
        model_classes = (
            FasterWhisperForSpeechRecognition,
            WhisperXForSpeechRecognition,
            NeMoASRForSpeechRecognition,
            SpeechBrainASRForSpeechRecognition,
            FunASRForSpeechRecognition,
            ESPnetASRForSpeechRecognition,
            WeNetASRForSpeechRecognition,
        )

        with _temporary_modules({"torch": fake_torch}):
            for model_class in model_classes:
                with self.subTest(model=model_class.__name__):
                    self.assertEqual(model_class._resolve_device("auto"), "cpu")
                    with self.assertRaisesRegex(ValueError, "CPU and CUDA"):
                        model_class._resolve_device("mps")

    def test_ctranslate2_wrappers_reject_half_precision_on_cpu(self):
        models = (
            FasterWhisperForSpeechRecognition(
                FasterWhisperConfig(compute_type="float16"),
                device="cpu",
            ),
            WhisperXForSpeechRecognition(
                WhisperXConfig(compute_type="float16"),
                device="cpu",
            ),
        )
        for model in models:
            with (
                    self.subTest(model=model.config.model_type),
                    self.assertRaisesRegex(ValueError, "cannot use.*CPU"),
            ):
                model._load_pretrained_model()


class WhisperProviderInferenceTests(unittest.TestCase):

    def test_faster_whisper_normalizes_generator_segments_and_words(self):
        captured = {}

        class Runtime(_InferenceRuntime):

            def transcribe(
                self,
                audio,
                *,
                language=None,
                task=None,
                word_timestamps=False,
                chunk_length=None,
                beam_size=None,
                max_new_tokens=None,
                hotwords=None,
            ):
                captured["audio"] = audio
                captured["options"] = {
                    "language": language,
                    "task": task,
                    "word_timestamps": word_timestamps,
                    "chunk_length": chunk_length,
                    "beam_size": beam_size,
                    "max_new_tokens": max_new_tokens,
                    "hotwords": hotwords,
                }
                segments = iter([
                    SimpleNamespace(
                        text=" hello ",
                        start=0.1,
                        end=0.4,
                        words=[SimpleNamespace(
                            word="hello",
                            start=0.1,
                            end=0.4,
                            probability=0.92,
                        )],
                    )
                ])
                info = SimpleNamespace(
                    language="en",
                    language_probability=0.98,
                    duration_after_vad=0.8,
                )
                return segments, info

        runtime = Runtime()
        module = ModuleType("faster_whisper")

        def load_model(source, **kwargs):
            captured["load"] = (source, kwargs)
            return runtime

        module.WhisperModel = load_model
        model = FasterWhisperForSpeechRecognition(
            FasterWhisperConfig(
                name_or_path="small.en",
                compute_type="int8",
            ),
            device="cpu",
        )

        with _temporary_modules({"faster_whisper": module}):
            output = model.transcribe(
                np.zeros(8_000, dtype=np.float32),
                sampling_rate=8_000,
                language="en",
                return_timestamps="word",
                num_beams=3,
                hotwords=("VoiceHub", "speech"),
            )

        self.assertEqual(captured["load"][0], "small.en")
        self.assertEqual(captured["load"][1]["compute_type"], "int8")
        self.assertEqual(captured["audio"].shape, (16_000, ))
        self.assertEqual(captured["options"]["hotwords"], "VoiceHub speech")
        self.assertTrue(captured["options"]["word_timestamps"])
        self.assertEqual(output.text, "hello")
        self.assertEqual(output.segments[0].words[0].confidence, 0.92)
        self.assertEqual(output.language, "en")
        self.assertAlmostEqual(output.duration, 1.0)
        self.assertEqual(output.metadata["artifact_format"], "ctranslate2")

    def test_openai_whisper_normalizes_dictionary_output(self):
        captured = {}

        class Runtime(_InferenceRuntime):

            def transcribe(
                self,
                audio,
                *,
                language=None,
                task=None,
                word_timestamps=False,
                beam_size=None,
            ):
                captured["options"] = {
                    "language": language,
                    "task": task,
                    "word_timestamps": word_timestamps,
                    "beam_size": beam_size,
                }
                return {
                    "text": "merhaba",
                    "language": "tr",
                    "segments": [
                        {
                            "start": 0.0,
                            "end": 0.5,
                            "text": "merhaba",
                        },
                    ],
                }

        runtime = Runtime()
        module = ModuleType("whisper")

        def load_model(source, *, device, **kwargs):
            captured["load"] = (source, device, kwargs)
            return runtime

        module.load_model = load_model
        model = OpenAIWhisperForSpeechRecognition(
            OpenAIWhisperConfig(name_or_path="medium"),
            device="cpu",
        )

        with _temporary_modules({"whisper": module}):
            output = model.transcribe(
                np.zeros(16_000, dtype=np.float32),
                sampling_rate=16_000,
                language="tr",
                return_timestamps=True,
                num_beams=2,
            )

        self.assertEqual(captured["load"][:2], ("medium", "cpu"))
        self.assertFalse(captured["options"]["word_timestamps"])
        self.assertEqual(captured["options"]["beam_size"], 2)
        self.assertEqual(output.text, "merhaba")
        self.assertEqual(output.segments[0].end, 0.5)
        self.assertEqual(output.metadata["backend"], "openai-whisper")

    def test_whisperx_runs_alignment_only_when_requested(self):
        captured = {}

        class Runtime(_InferenceRuntime):

            def transcribe(self, audio, *, batch_size=16, language=None):
                captured["transcribe"] = (audio, batch_size, language)
                return {
                    "text": "aligned speech",
                    "language": "en",
                    "segments": [
                        {
                            "start": 0.0,
                            "end": 1.0,
                            "text": "aligned speech",
                        },
                    ],
                }

        runtime = Runtime()
        module = ModuleType("whisperx")

        def load_model(source, device, **kwargs):
            captured["load"] = (source, device, kwargs)
            return runtime

        def load_align_model(*, language_code, device):
            captured["align_load"] = (language_code, device)
            return "align-model", {"dictionary": "metadata"}

        def align(
            segments,
            align_model,
            metadata,
            audio,
            device,
            *,
            return_char_alignments,
        ):
            captured["align"] = (
                segments,
                align_model,
                metadata,
                audio,
                device,
                return_char_alignments,
            )
            return {
                "text":
                "aligned speech",
                "language":
                "en",
                "segments": [
                    {
                        "start": 0.0,
                        "end": 1.0,
                        "text": "aligned speech",
                        "words": [
                            {
                                "word": "aligned",
                                "start": 0.0,
                                "end": 0.4,
                                "score": 0.91,
                            },
                        ],
                    },
                ],
            }

        module.load_model = load_model
        module.load_align_model = load_align_model
        module.align = align
        model = WhisperXForSpeechRecognition(
            WhisperXConfig(
                name_or_path="large-v3",
                compute_type="float32",
            ),
            device="cpu",
        )

        with _temporary_modules({"whisperx": module}):
            output = model.transcribe(
                np.zeros(16_000, dtype=np.float32),
                sampling_rate=16_000,
                return_timestamps="word",
                batch_size=4,
            )

        self.assertEqual(captured["load"][:2], ("large-v3", "cpu"))
        self.assertEqual(captured["transcribe"][1], 4)
        self.assertEqual(captured["align_load"], ("en", "cpu"))
        self.assertEqual(output.segments[0].words[0].confidence, 0.91)
        self.assertTrue(output.metadata["aligned"])


class NativeToolkitInferenceTests(unittest.TestCase):

    def test_unsupported_non_default_options_fail_instead_of_disappearing(self):
        cases = (
            (
                FasterWhisperForSpeechRecognition(FasterWhisperConfig()),
                {
                    "stride_length_s": 1.0
                },
                "stride_length_s",
            ),
            (
                OpenAIWhisperForSpeechRecognition(OpenAIWhisperConfig()),
                {
                    "chunk_length_s": 15.0
                },
                "chunk_length_s",
            ),
            (
                WhisperXForSpeechRecognition(WhisperXConfig()),
                {
                    "hotwords": ["VoiceHub"]
                },
                "hotwords",
            ),
            (
                NeMoASRForSpeechRecognition(NeMoASRConfig()),
                {
                    "task": "translate"
                },
                "task",
            ),
            (
                SpeechBrainASRForSpeechRecognition(SpeechBrainASRConfig()),
                {
                    "num_beams": 4
                },
                "num_beams",
            ),
            (
                FunASRForSpeechRecognition(FunASRConfig()),
                {
                    "batch_size": 12
                },
                "batch_size",
            ),
            (
                ESPnetASRForSpeechRecognition(ESPnetASRConfig()),
                {
                    "batch_size": 2
                },
                "batch_size",
            ),
            (
                WeNetASRForSpeechRecognition(WeNetASRConfig()),
                {
                    "hotwords": "VoiceHub"
                },
                "hotwords",
            ),
        )

        for model, options, option_name in cases:
            with (
                    self.subTest(model=model.config.model_type),
                    self.assertRaisesRegex(ValueError, option_name),
            ):
                model._transcribe(
                    np.zeros(160, dtype=np.float32),
                    sampling_rate=16_000,
                    **options,
                )

    def test_nemo_uses_supported_path_argument_and_normalizes_hypothesis(self):
        captured = {}

        class Runtime(_InferenceRuntime):

            def to(self, device):
                captured["device"] = device
                return self

            def transcribe(
                self,
                *,
                paths2audio_files,
                batch_size,
                return_hypotheses,
                timestamps,
            ):
                path = Path(paths2audio_files[0])
                captured["transcribe"] = {
                    "path": path,
                    "exists": path.is_file(),
                    "batch_size": batch_size,
                    "return_hypotheses": return_hypotheses,
                    "timestamps": timestamps,
                }
                return [
                    SimpleNamespace(
                        text="parakeet result",
                        language="en",
                        timestamp={"segment": [
                            {
                                "start": 0.0,
                                "end": 0.5,
                                "text": "parakeet result",
                            },
                        ]},
                    )
                ]

        runtime = Runtime()

        class ASRModel:

            @classmethod
            def from_pretrained(
                cls,
                *,
                model_name,
                map_location,
                token=None,
            ):
                captured["load"] = (model_name, map_location, token)
                return runtime

        nemo = ModuleType("nemo")
        nemo.__path__ = []
        collections = ModuleType("nemo.collections")
        collections.__path__ = []
        asr = ModuleType("nemo.collections.asr")
        asr.__path__ = []
        asr.models = SimpleNamespace(ASRModel=ASRModel)
        modules = {
            "nemo": nemo,
            "nemo.collections": collections,
            "nemo.collections.asr": asr,
        }
        model = NeMoASRForSpeechRecognition(
            NeMoASRConfig(name_or_path="nvidia/parakeet-test"),
            device="cpu",
            token="private",
        )

        with _temporary_modules(modules):
            output = model.transcribe(
                np.zeros(8_000, dtype=np.float32),
                sampling_rate=8_000,
                language="en",
                return_timestamps=True,
                batch_size=2,
            )

        self.assertEqual(
            captured["load"],
            ("nvidia/parakeet-test", "cpu", "private"),
        )
        self.assertEqual(captured["device"], "cpu")
        self.assertTrue(captured["transcribe"]["exists"])
        self.assertFalse(captured["transcribe"]["path"].exists())
        self.assertEqual(captured["transcribe"]["batch_size"], 2)
        self.assertTrue(captured["transcribe"]["timestamps"])
        self.assertEqual(output.text, "parakeet result")
        self.assertEqual(output.segments[0].end, 0.5)
        self.assertEqual(output.metadata["decoder"], "SimpleNamespace")

    def test_speechbrain_materializes_one_short_lived_audio_file(self):
        captured = {}

        class Runtime(_InferenceRuntime):

            def transcribe_file(self, audio_path):
                path = Path(audio_path)
                captured["path"] = path
                captured["exists"] = path.is_file()
                return "speech brain"

        runtime = Runtime()

        class EncoderDecoderASR:

            @classmethod
            def from_hparams(cls, **kwargs):
                captured["load"] = kwargs
                return runtime

        speechbrain = ModuleType("speechbrain")
        speechbrain.__path__ = []
        inference = ModuleType("speechbrain.inference")
        inference.__path__ = []
        asr_module = ModuleType("speechbrain.inference.ASR")
        asr_module.EncoderDecoderASR = EncoderDecoderASR
        modules = {
            "speechbrain": speechbrain,
            "speechbrain.inference": inference,
            "speechbrain.inference.ASR": asr_module,
        }
        model = SpeechBrainASRForSpeechRecognition(
            SpeechBrainASRConfig(savedir="cache/speechbrain"),
            device="cpu",
            token="private",
        )

        with _temporary_modules(modules):
            output = model.transcribe(
                np.zeros(16_000, dtype=np.float32),
                sampling_rate=16_000,
                language="en",
            )

        self.assertEqual(captured["load"]["source"], model.default_model_name_or_path)
        self.assertEqual(captured["load"]["run_opts"], {"device": "cpu"})
        self.assertEqual(captured["load"]["savedir"], "cache/speechbrain")
        self.assertEqual(captured["load"]["use_auth_token"], "private")
        self.assertTrue(captured["exists"])
        self.assertFalse(captured["path"].exists())
        self.assertEqual(output.text, "speech brain")
        self.assertEqual(output.metadata["backend"], "speechbrain")
        with self.assertRaisesRegex(ValueError, "timestamps"):
            model.transcribe(
                np.zeros(16, dtype=np.float32),
                sampling_rate=16_000,
                return_timestamps=True,
            )

    def test_funasr_maps_millisecond_timestamps(self):
        captured = {}

        class Runtime(_InferenceRuntime):

            def generate(
                self,
                *,
                input,
                batch_size_s=None,
                hotword=None,
                language=None,
            ):
                path = Path(input)
                captured["generate"] = {
                    "path": path,
                    "exists": path.is_file(),
                    "batch_size_s": batch_size_s,
                    "hotword": hotword,
                    "language": language,
                }
                return [{
                    "text": "ni hao",
                    "language": "zh",
                    "timestamp": [
                        [100, 400, "ni"],
                        [450, 800, "hao"],
                    ],
                }]

        runtime = Runtime()
        module = ModuleType("funasr")

        def auto_model(**kwargs):
            captured["load"] = kwargs
            return runtime

        module.AutoModel = auto_model
        model = FunASRForSpeechRecognition(
            FunASRConfig(
                name_or_path="paraformer-test",
                vad_model="fsmn-vad",
                generate_kwargs={
                    "batch_size_s": 12,
                },
            ),
            device="cpu",
        )

        with _temporary_modules({"funasr": module}):
            output = model.transcribe(
                np.zeros(16_000, dtype=np.float32),
                sampling_rate=16_000,
                language="zh",
                return_timestamps=True,
                hotwords=("VoiceHub", ),
            )

        self.assertEqual(captured["load"]["model"], "paraformer-test")
        self.assertEqual(captured["load"]["vad_model"], "fsmn-vad")
        self.assertTrue(captured["generate"]["exists"])
        self.assertFalse(captured["generate"]["path"].exists())
        self.assertEqual(captured["generate"]["batch_size_s"], 12)
        self.assertEqual(captured["generate"]["hotword"], ("VoiceHub", ))
        self.assertEqual(output.segments[0].start, 0.1)
        self.assertEqual(output.segments[1].end, 0.8)
        self.assertEqual(output.metadata["raw_keys"], ("language", "text", "timestamp"))

    def test_espnet_normalizes_first_beam_without_promising_timestamps(self):
        captured = {}

        class Runtime(_InferenceRuntime):

            def __call__(self, waveform):
                captured["waveform"] = waveform
                return [
                    ("espnet hypothesis", ["token"], [1], "hypothesis"),
                ]

        runtime = Runtime()

        class Speech2Text:

            @classmethod
            def from_pretrained(cls, **kwargs):
                captured["load"] = kwargs
                return runtime

        espnet = ModuleType("espnet2")
        espnet.__path__ = []
        binary = ModuleType("espnet2.bin")
        binary.__path__ = []
        inference = ModuleType("espnet2.bin.asr_inference")
        inference.Speech2Text = Speech2Text
        modules = {
            "espnet2": espnet,
            "espnet2.bin": binary,
            "espnet2.bin.asr_inference": inference,
        }
        model = ESPnetASRForSpeechRecognition(
            ESPnetASRConfig(
                name_or_path="espnet/test",
                beam_size=4,
                ctc_weight=0.2,
            ),
            device="cpu",
        )

        with _temporary_modules(modules):
            output = model.transcribe(
                np.zeros(8_000, dtype=np.float32),
                sampling_rate=8_000,
                language="en",
            )

        self.assertEqual(captured["load"]["model_tag"], "espnet/test")
        self.assertEqual(captured["load"]["beam_size"], 4)
        self.assertEqual(captured["waveform"].shape, (16_000, ))
        self.assertEqual(output.text, "espnet hypothesis")
        self.assertEqual(output.duration, 1.0)
        with self.assertRaisesRegex(ValueError, "timestamps"):
            model.transcribe(
                np.zeros(16, dtype=np.float32),
                sampling_rate=16_000,
                return_timestamps=True,
            )

    def test_wenet_normalizes_mapping_and_keeps_language_default(self):
        captured = {}

        class Runtime(_InferenceRuntime):

            def transcribe(self, audio_path):
                path = Path(audio_path)
                captured["path"] = path
                captured["exists"] = path.is_file()
                return {
                    "text": "wenet result",
                    "confidence": 0.9,
                }

        runtime = Runtime()
        vendored_runtime = ModuleType("voicehub.models.asr_native._wenet", )

        def load_model(source, *, device, **kwargs):
            captured["load"] = (source, device, kwargs)
            return runtime

        vendored_runtime.load_model = load_model
        model = WeNetASRForSpeechRecognition(
            WeNetASRConfig(
                name_or_path="english",
                language="en",
            ),
            device="cpu",
        )

        with patch(
                "voicehub.models.asr_native.wenet.import_optional",
                return_value=vendored_runtime,
        ) as import_runtime:
            output = model.transcribe(
                np.zeros(16_000, dtype=np.float32),
                sampling_rate=16_000,
            )

        import_runtime.assert_called_once_with(
            "voicehub.models.asr_native._wenet",
            model_type="asr_wenet",
            install_extra=None,
        )
        self.assertEqual(captured["load"][:2], ("english", "cpu"))
        self.assertTrue(captured["exists"])
        self.assertFalse(captured["path"].exists())
        self.assertEqual(output.text, "wenet result")
        self.assertEqual(output.language, "en")
        self.assertEqual(output.metadata["backend"], "wenet")


if __name__ == "__main__":
    unittest.main()

import unittest
from contextlib import nullcontext
from enum import Enum
from types import SimpleNamespace
from unittest.mock import Mock, patch

from voicehub.models.csm.inference import CSMForTextToSpeech
from voicehub.models.fishtts.inference import FishTTSForTextToSpeech
from voicehub.models.higgstts.inference import HiggsTTSForTextToSpeech
from voicehub.models.llasa.inference import LlasaForTextToSpeech
from voicehub.models.mosstts.inference import MossTTSConfig, MossTTSForTextToSpeech
from voicehub.models.neutts.inference import NeuTTSForTextToSpeech
from voicehub.models.orpheustts.inference import OrpheusTTSForTextToSpeech
from voicehub.models.outetts.inference import OuteTTSConfig, OuteTTSForTextToSpeech
from voicehub.models.qwen3tts.inference import Qwen3TTSConfig, Qwen3TTSForTextToSpeech


class _TokenRow:

    def __init__(self, values):
        self.values = values

    def detach(self):
        return self

    def cpu(self):
        return self

    def tolist(self):
        return list(self.values)


class _TokenBatch:

    def __init__(self, values):
        self.row = _TokenRow(values)

    def __getitem__(self, index):
        if index != 0:
            raise IndexError(index)
        return self.row


class _InferenceMode:

    def __init__(self, torch):
        self.torch = torch

    def __enter__(self):
        self.torch.inference_depth += 1

    def __exit__(self, exc_type, exc_value, traceback):
        self.torch.inference_depth -= 1


class _CodeMatrix:
    ndim = 2

    def numel(self):
        return 4

    def transpose(self, first, second):
        if (first, second) != (0, 1):
            raise AssertionError((first, second))
        return self


class _AudioTensor:

    def __init__(self, values):
        self.values = values
        if values and isinstance(values[0], list):
            self.shape = (len(values), len(values[0]))
        else:
            self.shape = (len(values), )
        self.ndim = len(self.shape)

    def numel(self):
        result = 1
        for dimension in self.shape:
            result *= dimension
        return result

    def squeeze(self, dimension):
        if dimension != 0 or self.shape[0] != 1:
            raise AssertionError((dimension, self.shape))
        return _AudioTensor(list(self.values[0]))

    def mean(self, *, dim):
        if dim != 0 or self.ndim != 2:
            raise AssertionError((dim, self.shape))
        return _AudioTensor([
            sum(channel[index] for channel in self.values) / len(self.values)
            for index in range(self.shape[1])
        ])

    def detach(self):
        return self

    def float(self):
        return self

    def cpu(self):
        return self


class _FakeTorch:
    long = object()

    def __init__(self):
        self.inference_depth = 0

    @property
    def inference_active(self):
        return self.inference_depth > 0

    def inference_mode(self):
        return _InferenceMode(self)

    def as_tensor(self, value, **kwargs):
        del kwargs
        if isinstance(value, _AudioTensor):
            return value
        return _CodeMatrix()

    def cat(self, values, *, dim=0):
        return ("concatenated", tuple(values), dim)


class InferencePreflightTests(unittest.TestCase):

    def test_required_conditioning_fails_before_runtime_load(self):
        requests = [
            (
                OrpheusTTSForTextToSpeech(device="cpu"),
                {},
                "non-empty `voice`",
            ),
            (
                LlasaForTextToSpeech(device="cpu"),
                {
                    "speaker_audio_path": "reference.wav"
                },
                "speaker_audio_path.*reference_text",
            ),
            (
                NeuTTSForTextToSpeech(device="cpu"),
                {},
                "speaker_audio_path.*reference_text",
            ),
            (
                FishTTSForTextToSpeech(device="cpu"),
                {
                    "speaker_audio_path": "reference.wav"
                },
                "non-empty `reference_text`",
            ),
            (
                CSMForTextToSpeech(device="cpu"),
                {
                    "reference_text": "orphaned transcript"
                },
                "speaker_audio_path.*reference_text",
            ),
        ]

        for model, options, message in requests:
            with self.subTest(model=model.config.model_type):
                with self.assertRaisesRegex(ValueError, message):
                    model.generate("hello", **options)
                self.assertFalse(model.is_loaded)

    def test_oute_rejects_ambiguous_speaker_sources_before_load(self):
        model = OuteTTSForTextToSpeech(device="cpu")

        with self.assertRaisesRegex(ValueError, "only one"):
            model.generate(
                "hello",
                speaker_audio_path="reference.wav",
                speaker_profile_path="speaker.json",
            )

        self.assertFalse(model.is_loaded)

    def test_invalid_moss_variant_fails_before_importing_runtime(self):
        model = MossTTSForTextToSpeech(
            MossTTSConfig(variant="unknown"),
            device="cpu",
        )

        with self.assertRaisesRegex(ValueError, "Unsupported MOSS-TTS variant"):
            model.generate("hello")

        self.assertFalse(model.is_loaded)

    def test_higgs_token_limit_fails_before_runtime_load(self):
        model = HiggsTTSForTextToSpeech(device="cpu")

        with self.assertRaisesRegex(
                ValueError,
                "greater than zero|positive integer",
        ):
            model.generate("hello", max_new_tokens=0)

        self.assertFalse(model.is_loaded)

    def test_higgs_rejects_invalid_finite_backend_options_before_load(self):
        invalid_options = (
            ({
                "stop_strings": 3
            }, "stop_strings"),
            ({
                "ras_win_max_num_repeat": "2"
            }, "ras_win_max_num_repeat"),
        )
        for options, message in invalid_options:
            with self.subTest(options=options):
                model = HiggsTTSForTextToSpeech(device="cpu")
                with self.assertRaisesRegex((TypeError, ValueError), message):
                    model.generate("hello", **options)
                self.assertFalse(model.is_loaded)

    def test_csm_rejects_unrepresentable_duration_and_top_k(self):
        requests = (
            ({
                "max_audio_length_ms": 79
            }, "at least 80"),
            ({
                "top_k": 2_052
            }, "audio vocabulary size"),
        )
        for options, message in requests:
            with self.subTest(options=options):
                model = CSMForTextToSpeech(device="cpu")
                with self.assertRaisesRegex(ValueError, message):
                    model.generate("hello", **options)
                self.assertFalse(model.is_loaded)

    def test_fish_rejects_multiple_samples_and_unknown_options(self):
        model = FishTTSForTextToSpeech(device="cpu")
        with self.assertRaisesRegex(ValueError, "num_samples=1"):
            model.generate("hello", num_samples=2)
        self.assertFalse(model.is_loaded)

        with self.assertRaisesRegex(ValueError, "Unsupported generation option"):
            model.generate("hello", topp=0.9)
        self.assertFalse(model.is_loaded)

    def test_qwen_rejects_unknown_options_before_runtime_load(self):
        model = Qwen3TTSForTextToSpeech(device="cpu")

        with self.assertRaisesRegex(
                ValueError,
                "Unsupported Qwen3-TTS generation option.*top_pp",
        ):
            model.generate(
                "hello",
                mode="voice_design",
                top_pp=0.9,
            )

        self.assertFalse(model.is_loaded)

    def test_oute_validates_version_and_backend_contracts_before_load(self):
        legacy = OuteTTSForTextToSpeech(
            OuteTTSConfig(interface_version="V2"),
            device="cpu",
        )
        with self.assertRaisesRegex(ValueError, "has no bundled default speaker"):
            legacy.generate("hello")
        self.assertFalse(legacy.is_loaded)

        batch = OuteTTSForTextToSpeech(
            OuteTTSConfig(backend="VLLM"),
            device="cpu",
        )
        with self.assertRaisesRegex(ValueError, "only supports batch"):
            batch.generate("hello", generation_type="chunked")
        self.assertFalse(batch.is_loaded)

        synchronous = OuteTTSForTextToSpeech(device="cpu")
        with self.assertRaisesRegex(ValueError, "asynchronous backend"):
            synchronous.generate("hello", generation_type="batch")
        self.assertFalse(synchronous.is_loaded)

    def test_oute_validates_context_limit_and_sampler_options(self):
        model = OuteTTSForTextToSpeech(
            OuteTTSConfig(max_seq_length=4_096),
            device="cpu",
        )
        with self.assertRaisesRegex(ValueError, "exceeds.*max_seq_length"):
            model.generate("hello", max_length=4_097)
        self.assertFalse(model.is_loaded)

        with self.assertRaisesRegex(ValueError, "sampler option.*top_pp"):
            model.generate("hello", sampler={"top_pp": 0.9})
        self.assertFalse(model.is_loaded)

        model._validate_sampler({
            "temperature": 0,
            "top_k": 0,
        })


class OrpheusTokenParsingTests(unittest.TestCase):

    def test_extracts_only_complete_valid_snac_frames(self):
        offset = OrpheusTTSForTextToSpeech._AUDIO_TOKEN_OFFSET
        size = OrpheusTTSForTextToSpeech._SNAC_CODEBOOK_SIZE
        frame = [offset + channel * size + channel for channel in range(7)]
        tokens = [
            10,
            OrpheusTTSForTextToSpeech._START_SPEECH_TOKEN_ID,
            *frame,
            offset,
            OrpheusTTSForTextToSpeech._END_SPEECH_TOKEN_ID,
        ]

        codes = OrpheusTTSForTextToSpeech._extract_audio_codes(_TokenBatch(tokens))

        self.assertEqual(
            codes,
            [channel * size + channel for channel in range(7)],
        )

    def test_rejects_invalid_codebook_channel(self):
        offset = OrpheusTTSForTextToSpeech._AUDIO_TOKEN_OFFSET
        tokens = [
            OrpheusTTSForTextToSpeech._START_SPEECH_TOKEN_ID,
            *([offset] * 7),
            OrpheusTTSForTextToSpeech._END_SPEECH_TOKEN_ID,
        ]

        with self.assertRaisesRegex(RuntimeError, "channel 1"):
            OrpheusTTSForTextToSpeech._extract_audio_codes(_TokenBatch(tokens))


class WrapperHelperTests(unittest.TestCase):

    def test_llasa_malformed_speech_token_has_actionable_error(self):
        with self.assertRaisesRegex(RuntimeError, "malformed speech token"):
            LlasaForTextToSpeech._extract_speech_ids(["<|s_not-an-integer|>"])

    def test_oute_enum_errors_list_supported_values(self):

        class Backend(Enum):
            HF = "hf"
            VLLM = "vllm"

        with self.assertRaisesRegex(
                ValueError,
                "Choose one of: hf, vllm",
        ):
            OuteTTSForTextToSpeech._enum_member(
                Backend,
                "missing",
                option_name="backend",
            )

    def test_moss_variant_aliases_are_canonical(self):
        model = MossTTSForTextToSpeech(
            MossTTSConfig(variant="local-v1.5"),
            device="cpu",
        )

        self.assertEqual(model._resolve_variant(), "local_v1_5")

    def test_moss_default_codec_tracks_variant(self):
        expected = {
            "delay": "OpenMOSS-Team/MOSS-Audio-Tokenizer",
            "local": "OpenMOSS-Team/MOSS-Audio-Tokenizer",
            "local_v1_5": "OpenMOSS-Team/MOSS-Audio-Tokenizer-v2",
            "realtime": "OpenMOSS-Team/MOSS-Audio-Tokenizer",
        }
        model = MossTTSForTextToSpeech(device="cpu")

        for variant, codec_name in expected.items():
            with self.subTest(variant=variant):
                self.assertEqual(
                    model._resolve_codec_name_or_path(variant),
                    codec_name,
                )

        configured = MossTTSForTextToSpeech(
            MossTTSConfig(codec_name_or_path="custom/codec"),
            device="cpu",
        )
        self.assertEqual(
            configured._resolve_codec_name_or_path("delay"),
            "custom/codec",
        )

    def test_moss_standard_loader_uses_codec_rate_not_tts_config_rate(self):

        class Module:

            def __init__(self):
                self.training = True

            @classmethod
            def from_pretrained(cls, *args, **kwargs):
                del args, kwargs
                return cls()

            def eval(self):
                self.training = False
                return self

            def to(self, *args, **kwargs):
                del args, kwargs
                return self

        codec = Module()
        codec.config = SimpleNamespace(sampling_rate=48_000)

        class ProcessorFactory:
            codec_path = None

            @classmethod
            def from_pretrained(cls, model_path, *, codec_path):
                del model_path
                cls.codec_path = codec_path
                return SimpleNamespace(
                    audio_tokenizer=codec,
                    model_config=SimpleNamespace(sampling_rate=24_000),
                )

        configuration = SimpleNamespace(MossTTSDelayConfig=type("Config", (), {}))
        modeling = SimpleNamespace(MossTTSDelayModel=Module)
        processing = SimpleNamespace(MossTTSDelayProcessor=ProcessorFactory)
        module_path = "voicehub.models.mosstts.source.moss_tts_delay"
        modules = {
            f"{module_path}.configuration_moss_tts": configuration,
            f"{module_path}.modeling_moss_tts": modeling,
            f"{module_path}.processing_moss_tts": processing,
        }
        transformers = SimpleNamespace(AutoConfig=SimpleNamespace(register=Mock()), )
        model = MossTTSForTextToSpeech(device="cpu")
        model._variant = "delay"
        model._codec_name_or_path = model._resolve_codec_name_or_path("delay")

        with patch(
                "voicehub.models.mosstts.inference.import_optional",
                side_effect=lambda name, **kwargs: modules[name],
        ):
            model._load_standard_variant(
                transformers,
                dtype=object(),
            )

        self.assertEqual(
            ProcessorFactory.codec_path,
            "OpenMOSS-Team/MOSS-Audio-Tokenizer",
        )
        self.assertEqual(model.sample_rate, 48_000)
        self.assertFalse(model.model.training)
        self.assertFalse(codec.training)

    def test_moss_inference_transition_preserves_runtime_identities(self):

        class Module:

            def __init__(self):
                self.training = True

            def eval(self):
                self.training = False
                return self

        standard_model = Module()
        standard_codec = Module()
        standard = MossTTSForTextToSpeech(device="cpu")
        standard.model = standard_model
        standard._processor = SimpleNamespace(audio_tokenizer=standard_codec)

        standard._prepare_for_inference()

        self.assertIs(standard.model, standard_model)
        self.assertIs(standard._processor.audio_tokenizer, standard_codec)
        self.assertFalse(standard_model.training)
        self.assertFalse(standard_codec.training)

        realtime_model = Module()
        realtime_codec = Module()
        realtime_runtime = SimpleNamespace(
            model=realtime_model,
            codec=realtime_codec,
        )
        realtime = MossTTSForTextToSpeech(device="cpu")
        realtime.model = realtime_runtime
        realtime._processor = realtime_codec

        realtime._prepare_for_inference()

        self.assertIs(realtime.model, realtime_runtime)
        self.assertIs(realtime.model.model, realtime_model)
        self.assertIs(realtime.model.codec, realtime_codec)
        self.assertFalse(realtime_model.training)
        self.assertFalse(realtime_codec.training)

    def test_moss_decodes_inside_inference_mode(self):
        torch = _FakeTorch()

        class Realtime:

            def generate(self, *args, **kwargs):
                del args, kwargs
                if not torch.inference_active:
                    raise AssertionError("generation must use inference mode")
                return [[[1, 2], [3, 4]]]

        class Codec:

            def decode(self, codes):
                del codes
                if not torch.inference_active:
                    raise AssertionError("codec decode must use inference mode")
                return SimpleNamespace(audio=["realtime-audio"])

        realtime = MossTTSForTextToSpeech(device="cpu")
        realtime._torch = torch
        realtime.model = Realtime()
        realtime._processor = Codec()
        self.assertEqual(
            realtime._generate_realtime(
                "hello",
                speaker_audio_path=None,
                max_new_tokens=4,
                generation_options={},
            ),
            "realtime-audio",
        )

        class BatchTensor:

            def to(self, device):
                del device
                return self

        class StandardProcessor:

            def build_user_message(self, **kwargs):
                del kwargs
                return "message"

            def __call__(self, messages, *, mode):
                del messages, mode
                return {
                    "input_ids": BatchTensor(),
                    "attention_mask": BatchTensor(),
                }

            def decode(self, generated):
                del generated
                if not torch.inference_active:
                    raise AssertionError("processor decode must use inference mode")
                return [SimpleNamespace(audio_codes_list=["standard-audio"])]

        class StandardModel:

            def generate(self, **kwargs):
                del kwargs
                if not torch.inference_active:
                    raise AssertionError("generation must use inference mode")
                return "tokens"

        standard = MossTTSForTextToSpeech(device="cpu")
        standard._torch = torch
        standard.model = StandardModel()
        standard._processor = StandardProcessor()
        self.assertEqual(
            standard._generate_standard(
                "hello",
                speaker_audio_path=None,
                language=None,
                instruction=None,
                duration_tokens=None,
                max_new_tokens=4,
                generation_options={},
            ),
            "standard-audio",
        )

    def test_moss_standard_decode_concatenates_every_audio_segment(self):
        torch = _FakeTorch()

        class BatchTensor:

            def to(self, device):
                del device
                return self

        class Processor:

            def build_user_message(self, **kwargs):
                del kwargs
                return "message"

            def __call__(self, messages, *, mode):
                del messages, mode
                return {
                    "input_ids": BatchTensor(),
                    "attention_mask": BatchTensor(),
                }

            def decode(self, generated):
                del generated
                return [
                    SimpleNamespace(audio_codes_list=["first", "second"]),
                    SimpleNamespace(audio_codes_list=["third"]),
                ]

        model = MossTTSForTextToSpeech(device="cpu")
        model._torch = torch
        model.model = SimpleNamespace(generate=lambda **kwargs: "tokens")
        model._processor = Processor()

        audio = model._generate_standard(
            "hello",
            speaker_audio_path=None,
            language=None,
            instruction=None,
            duration_tokens=None,
            max_new_tokens=4,
            generation_options={},
        )

        self.assertEqual(
            audio,
            ("concatenated", ("first", "second", "third"), -1),
        )

    def test_moss_stereo_is_downmixed_to_mono(self):
        model = MossTTSForTextToSpeech(device="cpu")
        model._torch = _FakeTorch()
        stereo = _AudioTensor([
            [1.0, 3.0, 5.0],
            [3.0, 5.0, 7.0],
        ])

        waveform, source_channels = model._normalize_mono_audio(stereo)

        self.assertEqual(source_channels, 2)
        self.assertEqual(waveform.ndim, 1)
        self.assertEqual(waveform.values, [2.0, 4.0, 6.0])

    def test_qwen_auto_mode_uses_loaded_checkpoint_role(self):
        backend = SimpleNamespace(
            model=SimpleNamespace(tts_model_type="voice_design"),
            generate_voice_design=Mock(return_value=([[0.1, -0.1]], 24_000)),
            generate_custom_voice=Mock(),
            generate_voice_clone=Mock(),
        )
        model = Qwen3TTSForTextToSpeech(
            Qwen3TTSConfig(name_or_path="test/base-named-directory"),
            device="cpu",
        )
        model.model = backend

        output = model.generate("hello", mode="auto", instruct="calm")

        backend.generate_voice_design.assert_called_once()
        backend.generate_custom_voice.assert_not_called()
        backend.generate_voice_clone.assert_not_called()
        self.assertEqual(output.metadata["mode"], "voice_design")

    def test_higgs_scopes_generation_seed_and_reports_effective_value(self):
        response = SimpleNamespace(
            audio=[0.1, -0.1],
            sampling_rate=24_000,
            generated_text="hello",
            usage={
                "completion_tokens": 2,
            },
        )
        model = HiggsTTSForTextToSpeech(device="cpu")
        model.device = "cpu"
        model.model = SimpleNamespace(generate=Mock(return_value=response))
        model._ensure_serving_runtime = Mock()
        model._build_chat_sample = Mock(return_value=object())

        with patch(
                "voicehub.models.higgstts.inference.seeded_inference",
                return_value=nullcontext(43),
        ) as seeded:
            output = model._generate("hello", seed=None)

        seeded.assert_called_once_with(
            None,
            device="cpu",
            model_type="higgstts",
        )
        self.assertEqual(
            model.model.generate.call_args.kwargs["seed"],
            43,
        )
        self.assertEqual(output.metadata["seed"], 43)
        self.assertIsNone(output.metadata["requested_seed"])

    def test_qwen_voice_design_allows_an_empty_instruction(self):
        backend = SimpleNamespace(
            model=SimpleNamespace(tts_model_type="voice_design"),
            generate_voice_design=Mock(return_value=([[0.1, -0.1]], 24_000)),
            generate_custom_voice=Mock(),
            generate_voice_clone=Mock(),
        )
        model = Qwen3TTSForTextToSpeech(device="cpu")
        model.model = backend

        output = model.generate(
            "hello",
            mode="voice_design",
            instruct="",
            seed=7,
        )

        backend.generate_voice_design.assert_called_once()
        self.assertEqual(
            backend.generate_voice_design.call_args.kwargs["instruct"],
            "",
        )
        self.assertEqual(output.metadata["seed"], 7)

    def test_higgs_defaults_to_forced_audio_generation(self):

        class Message:

            def __init__(self, **kwargs):
                self.values = kwargs

        class ChatMLSample:

            def __init__(self, **kwargs):
                self.values = kwargs

        backend = Mock()
        backend.generate.return_value = SimpleNamespace(
            audio=[0.1, -0.1],
            sampling_rate=24_000,
            generated_text="hello",
            usage={},
        )
        model = HiggsTTSForTextToSpeech(device="cpu")
        model.model = backend
        model._types = SimpleNamespace(
            ChatMLSample=ChatMLSample,
            Message=Message,
        )

        model.generate("hello")

        self.assertTrue(backend.generate.call_args.kwargs["force_audio_gen"])

    def test_csm_processor_batch_to_must_preserve_mapping_contract(self):
        batch = SimpleNamespace(to=Mock(return_value=object()))

        with self.assertRaisesRegex(TypeError, "must return a mapping"):
            CSMForTextToSpeech._move_processor_output_to_device(
                batch,
                "cpu",
            )


if __name__ == "__main__":
    unittest.main()

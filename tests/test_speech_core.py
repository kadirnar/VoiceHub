import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import soundfile

from voicehub import (
    ASRInferenceConfig,
    ASROutput,
    ASRSegment,
    ASRWord,
    AudioInput,
    AudioProcessor,
    PreTrainedASRModel,
    SpeechSegment,
    SpeechTask,
    VADInferenceConfig,
    VADOutput,
    load_audio,
)
from voicehub.configuration_utils import VoiceHubConfig
from voicehub.vad_utils import frame_probabilities_to_segments, merge_speech_segments

PROJECT_ROOT = Path(__file__).resolve().parents[1]


class _CoreASRConfig(VoiceHubConfig):
    model_type = "test-speech-core-asr"

    def __init__(self, **kwargs):
        super().__init__(sample_rate=16_000, **kwargs)


class _CoreASRRuntime:

    def __init__(self):
        self.training = True

    def eval(self):
        self.training = False
        return self

    def train(self):
        self.training = True
        return self


class _CoreASRModel(PreTrainedASRModel):
    config_class = _CoreASRConfig

    def __init__(self):
        self.load_count = 0
        self.calls = []
        super().__init__(_CoreASRConfig(), device="cpu")

    def _load_pretrained_model(self):
        self.load_count += 1
        self.model = _CoreASRRuntime()

    def _transcribe(self, audio, *, sampling_rate=None, language=None, **kwargs):
        materialized = load_audio(
            audio,
            sampling_rate=sampling_rate,
            target_sampling_rate=self.sample_rate,
        )
        self.calls.append((materialized.waveform.clone(), language, kwargs))
        return ASROutput(
            text=f"request {len(self.calls)}",
            language=language,
            duration=materialized.duration,
        )


class _ArtifactASRConfig(VoiceHubConfig):
    model_type = "test-artifact-asr"


class _ArtifactAwareASRModel(PreTrainedASRModel):
    config_class = _ArtifactASRConfig

    def __init__(self, config, **kwargs):
        self.settings_seen_during_load = None
        super().__init__(config, **kwargs)

    def _load_pretrained_model(self):
        self.settings_seen_during_load = (
            self.processor.to_dict(),
            getattr(self.inference_config, "language", None),
        )
        self.model = _CoreASRRuntime()

    def _transcribe(self, audio, **kwargs):
        del audio, kwargs
        return ASROutput(text="")


class SpeechCoreImportTests(unittest.TestCase):

    def test_public_import_does_not_import_optional_speech_runtimes(self):
        code = """
import json
import sys
import voicehub
optional = (
    "torch",
    "transformers",
    "faster_whisper",
    "whisperx",
    "whisper",
    "nemo",
    "speechbrain",
    "funasr",
    "espnet2",
    "wenet",
    "silero_vad",
    "webrtcvad",
)
print(json.dumps({name: name in sys.modules for name in optional}))
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
                "torch": False,
                "transformers": False,
                "faster_whisper": False,
                "whisperx": False,
                "whisper": False,
                "nemo": False,
                "speechbrain": False,
                "funasr": False,
                "espnet2": False,
                "wenet": False,
                "silero_vad": False,
                "webrtcvad": False,
            },
        )

    def test_task_aliases_are_normalized(self):
        self.assertIs(
            SpeechTask.coerce("speech_to_text"),
            SpeechTask.AUTOMATIC_SPEECH_RECOGNITION,
        )
        self.assertIs(
            SpeechTask.coerce("vad"),
            SpeechTask.VOICE_ACTIVITY_DETECTION,
        )
        with self.assertRaisesRegex(ValueError, "Unknown speech task"):
            SpeechTask.coerce("speaker-diarization")


class SpeechInferenceConfigurationTests(unittest.TestCase):

    def test_asr_config_round_trips_provider_options(self):
        config = ASRInferenceConfig(
            language="tr",
            return_timestamps="word",
            chunk_length_s=30.0,
            stride_length_s=(4.0, 2.0),
            hotwords=["VoiceHub", "Ankara"],
            provider_temperature=0.1,
        )

        with tempfile.TemporaryDirectory() as directory:
            path = config.save_pretrained(directory)
            restored = ASRInferenceConfig.from_pretrained(directory)

        self.assertEqual(path.name, "transcription_config.json")
        self.assertEqual(restored.language, "tr")
        self.assertEqual(restored.return_timestamps, "word")
        self.assertEqual(restored.stride_length_s, (4.0, 2.0))
        self.assertEqual(restored.provider_temperature, 0.1)

    def test_task_configs_reject_invalid_controls(self):
        invalid_asr = (
            {
                "task": "summarize"
            },
            {
                "return_timestamps": "character"
            },
            {
                "return_timestamps": 1
            },
            {
                "batch_size": True
            },
            {
                "stride_length_s": -0.1
            },
        )
        for values in invalid_asr:
            with self.subTest(values=values), self.assertRaises((TypeError, ValueError)):
                ASRInferenceConfig(**values)

        invalid_vad = (
            {
                "threshold": 1.1
            },
            {
                "speech_pad_ms": -1
            },
            {
                "window_size_samples": 0
            },
            {
                "return_frames": 1
            },
        )
        for values in invalid_vad:
            with self.subTest(values=values), self.assertRaises((TypeError, ValueError)):
                VADInferenceConfig(**values)

    def test_asr_config_accepts_segment_timestamps(self):
        config = ASRInferenceConfig(return_timestamps="segment")

        self.assertEqual(config.return_timestamps, "segment")

    def test_serializable_speech_options_reject_nested_credentials(self):
        with self.assertRaisesRegex(ValueError, "runtime secrets"):
            ASRInferenceConfig(provider_options={
                "api_key": "must-not-be-persisted",
            })
        with self.assertRaisesRegex(ValueError, "runtime secrets"):
            AudioProcessor(request_headers={
                "authorization": "Bearer secret",
            })

        config = VADInferenceConfig()
        config.token = "added-after-validation"
        with self.assertRaisesRegex(ValueError, "runtime secrets"):
            config.to_dict()

    def test_vad_config_round_trips_without_losing_extensions(self):
        config = VADInferenceConfig(
            threshold=0.62,
            onset=0.7,
            offset=0.4,
            max_speech_duration_s=12.5,
            provider_mode="balanced",
        )

        with tempfile.TemporaryDirectory() as directory:
            config.save_pretrained(directory)
            restored = VADInferenceConfig.from_pretrained(directory)

        self.assertEqual(restored.threshold, 0.62)
        self.assertEqual(restored.onset, 0.7)
        self.assertEqual(restored.offset, 0.4)
        self.assertEqual(restored.provider_mode, "balanced")


class SpeechOutputTests(unittest.TestCase):

    def test_asr_output_preserves_typed_words_segments_and_mapping_protocol(self):
        word = ASRWord(
            text=" VoiceHub ",
            start=0.1,
            end=0.4,
            confidence=0.95,
            speaker="speaker-0",
        )
        segment = ASRSegment(
            text=" VoiceHub works ",
            start=0.1,
            end=0.8,
            confidence=0.9,
            language="en",
            words=(word, ),
        )
        output = ASROutput(
            text=" VoiceHub works ",
            segments=[segment],
            language="en",
            duration=1.0,
            metadata={"backend": "fake"},
        )

        self.assertEqual(output.text, "VoiceHub works")
        self.assertEqual(output[0], "VoiceHub works")
        self.assertEqual(output["segments"][0].words[0].text, "VoiceHub")
        self.assertEqual(output.to_tuple(), ("VoiceHub works", (segment, )))
        self.assertEqual(output.to_dict()["metadata"]["backend"], "fake")

    def test_asr_optional_text_fields_are_trimmed_and_validated(self):
        word = ASRWord(text="word", speaker=" speaker-1 ")
        segment = ASRSegment(
            text="text",
            language=" en ",
            speaker=" speaker-2 ",
        )
        output = ASROutput(text="text", language=" tr ")

        self.assertEqual(word.speaker, "speaker-1")
        self.assertEqual(segment.language, "en")
        self.assertEqual(segment.speaker, "speaker-2")
        self.assertEqual(output.language, "tr")
        with self.assertRaisesRegex(ValueError, "language"):
            ASRSegment(text="text", language=" ")

    def test_output_validation_rejects_invalid_time_and_confidence_contracts(self):
        invalid_factories = (
            lambda: ASRWord(text="", start=0.0, end=0.1),
            lambda: ASRWord(text="word", start=0.2, end=0.1),
            lambda: ASRWord(text="word", confidence=1.01),
            lambda: SpeechSegment(start=0.5, end=0.5),
            lambda: VADOutput(
                segments=(
                    SpeechSegment(start=0.4, end=0.8),
                    SpeechSegment(start=0.7, end=0.9),
                )),
            lambda: VADOutput(
                segments=(SpeechSegment(start=0.4, end=0.8), ),
                duration=0.5,
            ),
        )
        for factory in invalid_factories:
            with self.subTest(factory=factory), self.assertRaises((TypeError, ValueError)):
                factory()

    def test_vad_output_exposes_duration_membership_and_sample_bounds(self):
        first = SpeechSegment(start=0.1, end=0.3, score=0.8)
        second = SpeechSegment(start=0.5, end=0.9, score=0.6)
        output = VADOutput(
            segments=[first, second],
            duration=1.0,
            sample_rate=16_000,
        )

        self.assertAlmostEqual(output.speech_duration, 0.6)
        self.assertTrue(output.contains(0.1))
        self.assertFalse(output.contains(0.3))
        self.assertEqual(second.sample_bounds(16_000), (8_000, 14_400))


class AudioLoadingTests(unittest.TestCase):

    def test_array_audio_is_downmixed_and_resampled_deterministically(self):
        stereo = np.stack([
            np.linspace(-1.0, 1.0, 8, dtype=np.float32),
            np.linspace(1.0, -1.0, 8, dtype=np.float32),
        ])

        loaded = load_audio(
            stereo,
            sampling_rate=8,
            target_sampling_rate=16,
        )

        self.assertIsInstance(loaded, AudioInput)
        self.assertEqual(loaded.sampling_rate, 16)
        self.assertEqual(loaded.waveform.shape, (16, ))
        self.assertAlmostEqual(loaded.duration, 1.0)
        np.testing.assert_allclose(loaded.waveform, 0.0, atol=1e-7)

    def test_integer_pcm_is_normalized_before_audio_processing(self):
        signed = load_audio(
            np.asarray([-32768, 0, 32767], dtype=np.int16),
            sampling_rate=16_000,
        )
        unsigned = load_audio(
            np.asarray([0, 128, 255], dtype=np.uint8),
            sampling_rate=16_000,
        )

        np.testing.assert_allclose(
            signed.waveform,
            [-1.0, 0.0, 32767 / 32768],
            atol=1e-7,
        )
        np.testing.assert_allclose(
            unsigned.waveform,
            [-1.0, 0.0, 127 / 128],
            atol=1e-7,
        )

    def test_boolean_audio_is_not_interpreted_as_pcm(self):
        with self.assertRaisesRegex(TypeError, "real numeric"):
            load_audio(
                np.asarray([True, False]),
                sampling_rate=16_000,
            )

    def test_mapping_and_audio_input_rates_cannot_be_silently_overridden(self):
        with self.assertRaisesRegex(ValueError, "require a positive"):
            load_audio(np.zeros(16, dtype=np.float32))
        with self.assertRaisesRegex(ValueError, "conflicts"):
            load_audio(
                {
                    "array": np.zeros(16, dtype=np.float32),
                    "sampling_rate": 16_000,
                },
                sampling_rate=8_000,
            )
        with self.assertRaisesRegex(ValueError, "conflicts"):
            load_audio(
                AudioInput(np.zeros(16, dtype=np.float32), 16_000),
                sampling_rate=8_000,
            )

    def test_file_audio_is_loaded_with_path_rate_and_mono_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "speech.wav"
            soundfile.write(
                path,
                np.linspace(-0.2, 0.2, 80, dtype=np.float32),
                8_000,
            )

            loaded = load_audio(path, target_sampling_rate=16_000)

        self.assertEqual(loaded.path, path)
        self.assertEqual(loaded.sampling_rate, 16_000)
        self.assertEqual(loaded.waveform.shape, (160, ))
        self.assertAlmostEqual(loaded.duration, 0.01)

    def test_audio_processor_keeps_decoding_lazy(self):
        processor = AudioProcessor(normalize=False)
        value = processor("not-yet-read.wav", sampling_rate=16_000, language="en")

        self.assertEqual(value["audio"], "not-yet-read.wav")
        self.assertEqual(value["sampling_rate"], 16_000)
        self.assertEqual(value["language"], "en")
        self.assertEqual(processor.to_dict(), {"normalize": False})


class SpeechArtifactLifecycleTests(unittest.TestCase):

    def test_eager_loading_restores_artifact_settings_before_model_load(self):
        with tempfile.TemporaryDirectory() as directory:
            _ArtifactASRConfig().save_pretrained(directory)
            ASRInferenceConfig(language="tr").save_pretrained(directory)
            AudioProcessor(normalize=False).save_pretrained(directory)

            model = _ArtifactAwareASRModel.from_pretrained(
                directory,
                device="cpu",
                lazy_load=False,
            )

        self.assertEqual(
            model.settings_seen_during_load,
            ({
                "normalize": False
            }, "tr"),
        )


class VADUtilityValidationTests(unittest.TestCase):

    def test_short_regions_are_joined_before_minimum_duration_filtering(self):
        config = VADInferenceConfig(
            min_speech_duration_ms=350,
            min_silence_duration_ms=100,
            speech_pad_ms=0,
        )

        segments = frame_probabilities_to_segments(
            [0.9, 0.8, 0.7, 0.1, 0.6],
            sampling_rate=1_000,
            frame_hop_samples=100,
            frame_length_samples=100,
            config=config,
        )

        self.assertEqual(
            [(segment.start, segment.end) for segment in segments],
            [(0.0, 0.5)],
        )
        self.assertAlmostEqual(segments[0].score, 0.75)

    def test_frame_geometry_requires_integral_sample_counts(self):
        invalid_values = (
            {
                "sampling_rate": True
            },
            {
                "sampling_rate": 16_000.5
            },
            {
                "frame_hop_samples": 1.5
            },
            {
                "frame_length_samples": False
            },
        )
        for overrides in invalid_values:
            options = {
                "sampling_rate": 16_000,
                "frame_hop_samples": 160,
                "frame_length_samples": 160,
                **overrides,
            }
            with self.subTest(overrides=overrides), self.assertRaises((TypeError, ValueError)):
                frame_probabilities_to_segments([0.8], **options)

    def test_subsample_maximum_duration_is_rejected_without_looping(self):
        config = VADInferenceConfig(
            min_speech_duration_ms=0,
            min_silence_duration_ms=0,
            speech_pad_ms=0,
            max_speech_duration_s=0.0001,
        )

        with self.assertRaisesRegex(ValueError, "at least one audio sample"):
            frame_probabilities_to_segments(
                [0.9],
                sampling_rate=1_000,
                frame_hop_samples=10,
                config=config,
            )

    def test_merge_gap_must_be_a_finite_real_number(self):
        segment = SpeechSegment(start=0.0, end=0.1)
        for max_gap in (True, float("nan"), float("inf"), "0.1"):
            with self.subTest(max_gap=max_gap), self.assertRaises((TypeError, ValueError)):
                merge_speech_segments((segment, ), max_gap=max_gap)


class BufferedSpeechSessionTests(unittest.TestCase):

    def test_sessions_are_isolated_flush_once_and_can_be_reset(self):
        model = _CoreASRModel()
        first = model.stream(sampling_rate=16_000, language="en")
        second = model.stream(sampling_rate=16_000, language="tr")
        first.push(np.zeros(4, dtype=np.float32))
        first.push(np.ones(4, dtype=np.float32))
        second.push(np.full(3, 2.0, dtype=np.float32))

        first_output = first.flush()
        self.assertIs(first.flush(), first_output)
        second_output = second.flush()

        self.assertEqual(model.load_count, 1)
        self.assertEqual(first_output.language, "en")
        self.assertEqual(second_output.language, "tr")
        np.testing.assert_array_equal(model.calls[0][0], np.r_[np.zeros(4), np.ones(4)])
        np.testing.assert_array_equal(model.calls[1][0], np.full(3, 2.0))

        first.reset()
        first.push(np.zeros(2, dtype=np.float32))
        self.assertEqual(first.flush().text, "request 3")

    def test_stream_state_errors_are_explicit(self):
        model = _CoreASRModel()
        session = model.stream(sampling_rate=16_000)
        with self.assertRaisesRegex(ValueError, "no audio"):
            session.flush()

        session.push(np.zeros(2, dtype=np.float32))
        session.flush()
        with self.assertRaisesRegex(RuntimeError, "Reset"):
            session.push(np.zeros(2, dtype=np.float32))

        session.close()
        self.assertTrue(session.is_closed)
        with self.assertRaisesRegex(RuntimeError, "closed"):
            session.push(np.zeros(2, dtype=np.float32))
        with self.assertRaisesRegex(RuntimeError, "closed"):
            session.reset()


if __name__ == "__main__":
    unittest.main()

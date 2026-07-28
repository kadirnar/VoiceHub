import importlib.util
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from voicehub.models.cosyvoice.inference import CosyVoiceConfig, CosyVoiceForTextToSpeech
from voicehub.models.cosyvoice.training import (
    CosyVoiceTrainingAdapter,
    CosyVoiceTrainingArtifacts,
    CosyVoiceTrainingBackend,
    load_cosyvoice_training_backend,
)
from voicehub.trainer import Trainer
from voicehub.training.specs import get_training_spec
from voicehub.training_args import TrainingArguments

TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None
if TORCH_AVAILABLE:
    import torch


class _HyperPyYamlStub:

    def __init__(self, configs):
        self.configs = configs
        self.overrides = None

    def load_hyperpyyaml(self, config_file, *, overrides):
        self.config_name = Path(config_file.name).name
        self.contents = config_file.read()
        self.overrides = dict(overrides)
        return self.configs


class CosyVoicePackagingTests(unittest.TestCase):

    def test_default_and_training_installs_cover_cosyvoice_runtime(self):
        pyproject = (Path(__file__).resolve().parents[1] / "pyproject.toml").read_text(encoding="utf-8")
        project_dependencies = pyproject.split("dependencies = [", 1)[1].split("\n]", 1)[0]
        training_dependencies = pyproject.split("training = [", 1)[1].split("\n]", 1)[0]
        inference_required = {
            "gdown",
            "hydra-core",
            "librosa",
            "lightning",
            "matplotlib",
            "omegaconf",
            "openai-whisper",
            "regex",
            "rich",
            "safetensors",
            "scipy",
            "tiktoken",
            "tqdm",
            "wget",
        }
        training_required = {"pyarrow", "pyworld"}

        self.assertEqual(
            sorted(
                dependency for dependency in inference_required
                if f'"{dependency}' not in project_dependencies),
            [],
        )
        self.assertEqual(
            sorted(
                dependency for dependency in training_required
                if f'"{dependency}' not in training_dependencies),
            [],
        )


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch is an optional training extra")
class CosyVoiceTrainingRuntimeTests(unittest.TestCase):

    def test_training_loader_builds_only_the_selected_yaml_component(self):
        component = torch.nn.Linear(2, 1)
        checkpoint_state = {
            "weight": torch.full_like(component.weight, 3.0),
            "bias": torch.full_like(component.bias, -2.0),
        }
        hyperpyyaml = _HyperPyYamlStub({
            "llm": component,
            "train_conf": {
                "optim": "adamw",
                "optim_conf": {
                    "lr": 2e-4,
                },
                "scheduler": "warmuplr",
                "scheduler_conf": {
                    "warmup_steps": 10,
                },
            },
            "sample_rate": 22_050,
        })

        with tempfile.TemporaryDirectory() as directory:
            model_directory = Path(directory)
            (model_directory / "cosyvoice3.yaml").write_text(
                "llm: !new:cosyvoice.llm.llm.CosyVoice3LM\n"
                "mel: !name:matcha.utils.audio.mel_spectrogram\n",
                encoding="utf-8",
            )
            (model_directory / "llm.pt").touch()

            def import_stub(module_name, **_kwargs):
                if module_name == "hyperpyyaml":
                    return hyperpyyaml
                if module_name == "torch":
                    return torch
                raise AssertionError(f"Unexpected import: {module_name}")

            with (
                    patch(
                        "voicehub.models.cosyvoice.training.import_optional",
                        side_effect=import_stub,
                    ),
                    patch.object(torch, "load", return_value=checkpoint_state),
            ):
                backend = load_cosyvoice_training_backend(
                    model_directory,
                    "language_model",
                )

        self.assertEqual(backend.component_name, "llm")
        self.assertIs(backend.selected_component, component)
        self.assertTrue(hasattr(backend.model, "llm"))
        self.assertFalse(hasattr(backend.model, "flow"))
        self.assertFalse(hasattr(backend.model, "hift"))
        self.assertEqual(hyperpyyaml.config_name, "cosyvoice3.yaml")
        self.assertIn(
            "!new:voicehub.models.cosyvoice.source.cosyvoice.llm.llm.CosyVoice3LM",
            hyperpyyaml.contents,
        )
        self.assertIn(
            "!name:voicehub.models.cosyvoice.source.matcha.utils.audio.mel_spectrogram",
            hyperpyyaml.contents,
        )
        self.assertNotIn("llm", hyperpyyaml.overrides)
        self.assertEqual(hyperpyyaml.overrides["flow"], None)
        self.assertEqual(hyperpyyaml.overrides["hift"], None)
        self.assertEqual(hyperpyyaml.overrides["hifigan"], None)
        self.assertEqual(
            hyperpyyaml.overrides["qwen_pretrain_path"],
            str(model_directory.resolve() / "CosyVoice-BlankEN"),
        )
        self.assertEqual(backend.sample_rate, 22_050)
        torch.testing.assert_close(component.weight, checkpoint_state["weight"])
        torch.testing.assert_close(component.bias, checkpoint_state["bias"])

    def test_training_loader_accepts_component_safetensors(self):
        component = torch.nn.Linear(1, 1, bias=False)
        checkpoint_state = {
            "weight": torch.full_like(component.weight, 4.0),
        }
        hyperpyyaml = _HyperPyYamlStub({
            "flow": component,
            "train_conf": {
                "optim": "adam",
                "optim_conf": {
                    "lr": 1e-3,
                },
                "scheduler": "constantlr",
                "scheduler_conf": {},
            },
        })
        safetensors = SimpleNamespace(load_file=lambda path, device: checkpoint_state, )

        with tempfile.TemporaryDirectory() as directory:
            model_directory = Path(directory)
            (model_directory / "cosyvoice.yaml").write_text(
                "# test config\n",
                encoding="utf-8",
            )
            (model_directory / "flow.safetensors").touch()

            def import_stub(module_name, **_kwargs):
                if module_name == "hyperpyyaml":
                    return hyperpyyaml
                if module_name == "safetensors.torch":
                    return safetensors
                raise AssertionError(f"Unexpected import: {module_name}")

            with patch(
                    "voicehub.models.cosyvoice.training.import_optional",
                    side_effect=import_stub,
            ):
                backend = load_cosyvoice_training_backend(
                    model_directory,
                    "flow",
                )

        self.assertEqual(
            backend.artifacts.checkpoint_path.name,
            "flow.safetensors",
        )
        self.assertFalse(hasattr(backend.model, "llm"))
        torch.testing.assert_close(component.weight, checkpoint_state["weight"])

    def test_generate_rebuilds_full_runtime_and_overlays_trained_weights(self):
        model = CosyVoiceForTextToSpeech(
            CosyVoiceConfig(name_or_path="unused", training_component="llm"),
            device="cpu",
            lazy_load=True,
        )
        trained_component = torch.nn.Linear(1, 1, bias=False)
        trained_component.weight.data.fill_(7.0)
        artifacts = CosyVoiceTrainingArtifacts(
            model_directory=Path("/model"),
            config_path=Path("/model/cosyvoice3.yaml"),
            checkpoint_path=Path("/model/llm.pt"),
            component_name="llm",
            train_conf={},
            sample_rate=24_000,
        )
        backend = CosyVoiceTrainingBackend(trained_component, artifacts)
        model.model = backend
        model._cosyvoice_training_backend = backend

        full_component = torch.nn.Linear(1, 1, bias=False)
        full_component.weight.data.zero_()

        def load_full_runtime():
            model.model = SimpleNamespace(
                model=SimpleNamespace(
                    llm=full_component,
                    flow=object(),
                    hift=object(),
                ),
                sample_rate=24_000,
            )
            model._cosyvoice_training_backend = None

        with (
                patch.object(
                    model,
                    "_load_full_inference_model",
                    side_effect=load_full_runtime,
                ) as load_full,
                patch.object(
                    model,
                    "_select_inference",
                    return_value=iter([{
                        "tts_speech": torch.tensor([[1.0, 2.0]]),
                    }]),
                ),
        ):
            output = model._generate("portable artifact")

        load_full.assert_called_once_with()
        self.assertIs(model.model.model.llm, trained_component)
        torch.testing.assert_close(
            full_component.weight,
            torch.zeros_like(full_component.weight),
        )
        torch.testing.assert_close(
            output.audio,
            torch.tensor([1.0, 2.0]),
        )
        self.assertIs(model._cosyvoice_training_backend, backend)
        self.assertIsNot(model.model, backend)

    def test_same_adapter_resumes_on_its_optimizer_owned_backend_after_generate(self, ):

        class SourceComponent(torch.nn.Module):

            def __init__(self, scale):
                super().__init__()
                self.scale = torch.nn.Parameter(torch.tensor(float(scale)))

            def forward(self, batch, device):
                return {
                    "loss": batch["values"].to(device).mean() * self.scale,
                }

        model = CosyVoiceForTextToSpeech(
            CosyVoiceConfig(name_or_path="unused", training_component="llm"),
            device="cpu",
            lazy_load=True,
        )
        trained_component = SourceComponent(2.0)
        artifacts = CosyVoiceTrainingArtifacts(
            model_directory=Path("/model"),
            config_path=Path("/model/cosyvoice3.yaml"),
            checkpoint_path=Path("/model/llm.pt"),
            component_name="llm",
            train_conf={},
            sample_rate=24_000,
        )
        backend = CosyVoiceTrainingBackend(trained_component, artifacts)
        model.model = backend
        model._cosyvoice_training_backend = backend
        adapter = CosyVoiceTrainingAdapter(
            model,
            get_training_spec("cosyvoice"),
        )
        inference_component = SourceComponent(0.0)

        def load_full_runtime():
            model.model = SimpleNamespace(
                model=SimpleNamespace(
                    llm=inference_component,
                    flow=object(),
                    hift=object(),
                ),
                sample_rate=24_000,
            )
            model._cosyvoice_training_backend = None

        with tempfile.TemporaryDirectory() as directory:
            trainer = Trainer(
                model=model,
                args=TrainingArguments(
                    output_dir=directory,
                    max_steps=1,
                    per_device_train_batch_size=1,
                    max_grad_norm=0,
                    logging_strategy="no",
                    save_strategy="no",
                    use_cpu=True,
                ),
                train_dataset=[{
                    "batch": {
                        "values": torch.tensor([2.0]),
                    }
                }],
                data_collator=lambda features: features[0],
                training_adapter=adapter,
                optimizer_factory=lambda _name, parameters, _args: torch.optim.SGD(
                    [parameter for _, parameter in parameters],
                    lr=0.1,
                ),
                scheduler_factory=lambda _name, optimizer, _steps, _args:
                (torch.optim.lr_scheduler.LambdaLR(
                    optimizer,
                    lambda _step: 1.0,
                )),
            )
            trainer._move_model_to_device()
            trainer.create_optimizer_and_scheduler(1)
            optimizer_parameter = trainer.optimizer.optimizers["language_model"].param_groups[0]["params"][0]

            with (
                    patch.object(
                        model,
                        "_load_full_inference_model",
                        side_effect=load_full_runtime,
                    ),
                    patch.object(
                        model,
                        "_select_inference",
                        return_value=iter([{
                            "tts_speech": torch.tensor([[1.0]]),
                        }]),
                    ),
            ):
                model._generate("inspect, then continue training")

            self.assertIs(model._cosyvoice_training_backend, backend)
            self.assertIsNot(model.model, backend)
            self.assertIs(optimizer_parameter, trained_component.scale)
            self.assertIs(model.model.model.llm, trained_component)
            torch.testing.assert_close(
                inference_component.scale,
                torch.tensor(0.0),
            )

            train_output = trainer.train()

        self.assertEqual(train_output.global_step, 1)
        self.assertIs(model.model, backend)
        self.assertIs(adapter.primary_model, trained_component)
        self.assertIs(optimizer_parameter, model.model.model.llm.scale)
        torch.testing.assert_close(
            trained_component.scale,
            torch.tensor(1.8),
        )
        torch.testing.assert_close(
            inference_component.scale,
            torch.tensor(0.0),
        )

    def test_prepare_for_training_replaces_an_existing_full_runtime(self):
        model = CosyVoiceForTextToSpeech(
            CosyVoiceConfig(name_or_path="unused", training_component="flow"),
            device="cpu",
            lazy_load=True,
        )
        inference_component = torch.nn.Linear(1, 1, bias=False)
        inference_component.weight.data.fill_(5.0)
        model.model = SimpleNamespace(
            model=SimpleNamespace(
                llm=object(),
                flow=inference_component,
                hift=object(),
            ))

        training_component = torch.nn.Linear(1, 1, bias=False)
        training_component.weight.data.zero_()
        artifacts = CosyVoiceTrainingArtifacts(
            model_directory=Path("/model"),
            config_path=Path("/model/cosyvoice3.yaml"),
            checkpoint_path=Path("/model/flow.pt"),
            component_name="flow",
            train_conf={},
            sample_rate=24_000,
        )
        backend = CosyVoiceTrainingBackend(training_component, artifacts)

        def load_training_runtime():
            model.model = backend
            model._cosyvoice_training_backend = backend

        with patch.object(
                model,
                "_load_training_model",
                side_effect=load_training_runtime,
        ) as load_training:
            model._prepare_for_training()

        load_training.assert_called_once_with()
        self.assertIs(model.model, backend)
        self.assertFalse(hasattr(backend.model, "llm"))
        self.assertFalse(hasattr(backend.model, "hift"))
        torch.testing.assert_close(
            training_component.weight,
            inference_component.weight,
        )

    def test_source_optimizer_scheduler_and_resume_signature_are_preserved(self):
        component = torch.nn.Linear(1, 1)
        train_conf = {
            "optim": "adamw",
            "optim_conf": {
                "lr": 0.012,
                "betas": (0.8, 0.95),
            },
            "scheduler": "constantlr",
            "scheduler_conf": {
                "ignored_by_source": True,
            },
        }
        wrapper = SimpleNamespace(
            config=SimpleNamespace(
                model_type="cosyvoice",
                training_component="llm",
            ),
            model=SimpleNamespace(model=SimpleNamespace(llm=component), ),
            _cosyvoice_training_backend=SimpleNamespace(train_conf=train_conf, ),
        )
        adapter = CosyVoiceTrainingAdapter(
            wrapper,
            get_training_spec("cosyvoice"),
        )
        parameters = list(component.named_parameters())

        optimizer = adapter.create_optimizer(
            "language_model",
            parameters,
            SimpleNamespace(),
        )
        scheduler = adapter.create_scheduler(
            "language_model",
            optimizer,
            100,
            SimpleNamespace(),
        )
        resume_configuration = adapter.recipe_resume_configuration()

        self.assertIsInstance(optimizer, torch.optim.AdamW)
        self.assertEqual(optimizer.param_groups[0]["lr"], 0.012)
        self.assertEqual(optimizer.param_groups[0]["betas"], (0.8, 0.95))
        self.assertEqual(type(scheduler).__name__, "ConstantLR")
        self.assertEqual(
            resume_configuration["source_optimization"]["scheduler"],
            "constantlr",
        )
        self.assertEqual(
            resume_configuration["source_optimization"]["optim_conf"]["lr"],
            0.012,
        )

    def test_hifigan_selection_fails_before_loading_inference_graphs(self):
        calls = []
        wrapper = SimpleNamespace(
            config=SimpleNamespace(
                model_type="cosyvoice",
                training_component="hifigan_generator",
            ),
            load_for_training=lambda: calls.append("load"),
        )
        adapter = CosyVoiceTrainingAdapter(
            wrapper,
            get_training_spec("cosyvoice"),
        )

        with self.assertRaisesRegex(ValueError, "training-only HiFiGan"):
            adapter.setup()

        self.assertEqual(calls, [])


if __name__ == "__main__":
    unittest.main()

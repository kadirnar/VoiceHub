import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from voicehub import AutoModelForTextToSpeech
from voicehub.inference_strategy import (
    EagerInferenceStrategy,
    InferenceStrategy,
    TorchCompileInferenceStrategy,
    get_inference_strategy,
    list_inference_strategies,
    register_inference_strategy,
    unregister_inference_strategy,
)
from voicehub.optimization import OptimizationCompatibilityError
from voicehub.optimization.protocols import OptimizationCompileTarget
from voicehub.optimization.torch_compile import (
    TorchCompileCapabilityReport,
    TorchCompileUnavailableError,
)


class RecordingInferenceStrategy(InferenceStrategy):
    name = "recording"

    def __init__(self):
        self.events = []

    def validate(self, wrapper):
        self.events.append(("validate", wrapper))

    def prepare(self, model, *, wrapper):
        prepared = {
            "runtime": model,
        }
        self.events.append(("prepare", model, wrapper))
        return prepared

    def restore_for_training(self, model, *, wrapper):
        self.events.append(("restore", model, wrapper))
        return model["runtime"]


class InferenceStrategyHooksTest(unittest.TestCase):

    def test_eager_strategy_is_a_no_op(self):
        strategy = EagerInferenceStrategy()
        wrapper = object()
        model = object()

        self.assertIsNone(strategy.validate(wrapper))
        self.assertIs(strategy.prepare(model, wrapper=wrapper), model)
        self.assertIs(strategy.restore_for_training(model, wrapper=wrapper), model)

    def test_strategy_can_wrap_and_restore_a_runtime(self):
        strategy = RecordingInferenceStrategy()
        wrapper = object()
        model = object()

        strategy.validate(wrapper)
        runtime = strategy.prepare(model, wrapper=wrapper)
        restored = strategy.restore_for_training(runtime, wrapper=wrapper)

        self.assertIs(restored, model)
        self.assertEqual(
            strategy.events,
            [
                ("validate", wrapper),
                ("prepare", model, wrapper),
                ("restore", runtime, wrapper),
            ],
        )

    def test_torch_compile_strategy_patches_and_restores_declared_targets(self):

        class Runtime(torch.nn.Module):

            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(3, 2)

            def forward(self, values):
                return self.linear(values)

            def optimization_compile_targets(self, mode):
                self.compile_mode = mode
                return (
                    OptimizationCompileTarget(
                        label="runtime",
                        owner=self,
                        attribute="forward",
                    ), )

        runtime = Runtime().eval()
        wrapper = SimpleNamespace(device="cpu")
        strategy = TorchCompileInferenceStrategy(
            backend="eager",
            dynamic=True,
        )
        expected_keys = tuple(runtime.state_dict())
        eager = runtime(torch.ones(1, 3))

        strategy.validate(wrapper)
        prepared = strategy.prepare(runtime, wrapper=wrapper)
        actual = prepared(torch.ones(1, 3))

        self.assertIs(prepared, runtime)
        self.assertEqual(runtime.compile_mode, "inference")
        self.assertIn("forward", runtime.__dict__)
        self.assertEqual(tuple(runtime.state_dict()), expected_keys)
        torch.testing.assert_close(actual, eager)
        self.assertEqual(
            strategy.runtime_metadata(wrapper)["outcome"],
            "compiled",
        )

        restored = strategy.restore_for_training(
            prepared,
            wrapper=wrapper,
        )
        self.assertIs(restored, runtime)
        self.assertNotIn("forward", runtime.__dict__)
        self.assertIsNone(strategy.runtime_metadata(wrapper))

    def test_required_torch_compile_strategy_fails_before_model_loading(self):
        report = TorchCompileCapabilityReport(
            available=False,
            backend="missing",
            backend_available=False,
            torch_version=torch.__version__,
            available_backends=(),
            reason="unit-test backend is unavailable",
        )
        strategy = TorchCompileInferenceStrategy(
            backend="missing",
            requirement="required",
        )

        with (
            patch(
                "voicehub.optimization.torch_compile.inspect_torch_compile",
                return_value=report,
            ),
            self.assertRaisesRegex(
                TorchCompileUnavailableError,
                "unit-test backend is unavailable",
            ),
        ):
            strategy.validate(SimpleNamespace(device="cpu"))

    def test_quality_rejected_tts_compile_strategies_fail_before_loading(self):
        for model_type in ("neutts", "vits", "vui"):
            with self.subTest(model_type=model_type):
                model = AutoModelForTextToSpeech.from_pretrained(
                    "",
                    model_type=model_type,
                    device="cpu",
                    lazy_load=True,
                    inference_strategy=TorchCompileInferenceStrategy(
                        backend="eager",
                        requirement="required",
                    ),
                )
                with self.assertRaisesRegex(
                    OptimizationCompatibilityError,
                    rf"{model_type}.*real-checkpoint",
                ):
                    model.load()
                self.assertIsNone(model.model)

    def test_runtime_context_binds_the_registered_architecture(self):
        wrapper = SimpleNamespace(
            config=SimpleNamespace(model_type="vits"),
            device="cpu",
        )
        context = TorchCompileInferenceStrategy._runtime_context(
            torch.nn.Linear(2, 2),
            wrapper,
        )

        self.assertEqual(context.architecture, "vits")

    def test_optional_torch_compile_strategy_reports_eager_fallback(self):
        report = TorchCompileCapabilityReport(
            available=False,
            backend="missing",
            backend_available=False,
            torch_version=torch.__version__,
            available_backends=(),
            reason="unit-test backend is unavailable",
        )
        runtime = torch.nn.Linear(2, 2)
        wrapper = SimpleNamespace(device="cpu")
        strategy = TorchCompileInferenceStrategy(
            backend="missing",
            requirement="auto",
        )

        with patch(
            "voicehub.optimization.torch_compile.inspect_torch_compile",
            return_value=report,
        ):
            strategy.validate(wrapper)
            prepared = strategy.prepare(runtime, wrapper=wrapper)

        self.assertIs(prepared, runtime)
        metadata = strategy.runtime_metadata(wrapper)
        self.assertEqual(metadata["outcome"], "eager-fallback")
        self.assertIn("unavailable", metadata["reason"])
        self.assertIs(
            strategy.restore_for_training(prepared, wrapper=wrapper),
            runtime,
        )


class InferenceStrategyRegistryTest(unittest.TestCase):

    def setUp(self):
        self.registered_names = []

    def tearDown(self):
        for name in self.registered_names:
            if name in list_inference_strategies():
                unregister_inference_strategy(name)

    def register(self, name, factory, **kwargs):
        register_inference_strategy(name, factory, **kwargs)
        normalized = name.strip().lower()
        if normalized not in self.registered_names:
            self.registered_names.append(normalized)

    def test_default_and_named_eager_resolution_create_fresh_instances(self):
        default = get_inference_strategy()
        named = get_inference_strategy(" EAGER ")

        self.assertIsInstance(default, EagerInferenceStrategy)
        self.assertIsInstance(named, EagerInferenceStrategy)
        self.assertIsNot(default, named)

    def test_named_torch_compile_strategy_is_opt_in_and_required(self):
        strategy = get_inference_strategy("torch-compile")

        self.assertIsInstance(strategy, TorchCompileInferenceStrategy)
        self.assertEqual(strategy.config.requirement.value, "required")

    def test_existing_strategy_instance_is_returned_unchanged(self):
        strategy = RecordingInferenceStrategy()

        self.assertIs(get_inference_strategy(strategy), strategy)

    def test_custom_factory_is_lazy_and_names_are_normalized(self):
        calls = []

        def factory():
            calls.append("called")
            return RecordingInferenceStrategy()

        self.register(" Unit-Test-Recording ", factory)
        self.assertEqual(calls, [])
        self.assertIn("unit-test-recording", list_inference_strategies())

        first = get_inference_strategy("UNIT-TEST-RECORDING")
        second = get_inference_strategy("unit-test-recording")

        self.assertIsInstance(first, RecordingInferenceStrategy)
        self.assertIsInstance(second, RecordingInferenceStrategy)
        self.assertIsNot(first, second)
        self.assertEqual(calls, ["called", "called"])

    def test_strategy_names_are_listed_deterministically(self):
        self.register("unit-test-zeta", RecordingInferenceStrategy)
        self.register("unit-test-alpha", RecordingInferenceStrategy)

        names = list_inference_strategies()

        self.assertEqual(names, tuple(sorted(names)))
        self.assertIn("eager", names)

    def test_duplicate_registration_requires_exist_ok(self):
        self.register("unit-test-replace", RecordingInferenceStrategy)

        with self.assertRaisesRegex(ValueError, "already registered"):
            register_inference_strategy("unit-test-replace", RecordingInferenceStrategy)

        self.register(
            "unit-test-replace",
            EagerInferenceStrategy,
            exist_ok=True,
        )
        self.assertIsInstance(
            get_inference_strategy("unit-test-replace"),
            EagerInferenceStrategy,
        )

    def test_builtin_registration_and_removal_are_protected(self):
        register_inference_strategy(
            "EAGER",
            EagerInferenceStrategy,
            exist_ok=True,
        )
        with self.assertRaisesRegex(ValueError, "cannot be replaced"):
            register_inference_strategy(
                "eager",
                RecordingInferenceStrategy,
                exist_ok=True,
            )
        with self.assertRaisesRegex(ValueError, "cannot be unregistered"):
            unregister_inference_strategy(" eager ")
        with self.assertRaisesRegex(ValueError, "cannot be replaced"):
            register_inference_strategy(
                "torch-compile",
                RecordingInferenceStrategy,
                exist_ok=True,
            )
        with self.assertRaisesRegex(ValueError, "cannot be unregistered"):
            unregister_inference_strategy("torch-compile")

    def test_unregister_removes_custom_registration(self):
        self.register("unit-test-remove", RecordingInferenceStrategy)

        unregister_inference_strategy("UNIT-TEST-REMOVE")
        self.registered_names.remove("unit-test-remove")

        with self.assertRaisesRegex(KeyError, "Unknown inference strategy"):
            get_inference_strategy("unit-test-remove")

    def test_invalid_names_and_factories_are_rejected(self):
        with self.assertRaisesRegex(TypeError, "names must be strings"):
            register_inference_strategy(123, RecordingInferenceStrategy)
        with self.assertRaisesRegex(ValueError, "cannot be empty"):
            register_inference_strategy("  ", RecordingInferenceStrategy)
        with self.assertRaisesRegex(TypeError, "must be callable"):
            register_inference_strategy("unit-test-invalid", object())
        with self.assertRaisesRegex(TypeError, "must inherit"):
            register_inference_strategy("unit-test-invalid", object)

    def test_invalid_lookup_values_have_actionable_errors(self):
        with self.assertRaisesRegex(TypeError, "name or InferenceStrategy"):
            get_inference_strategy(object())
        with self.assertRaisesRegex(ValueError, "cannot be empty"):
            get_inference_strategy(" ")
        with self.assertRaisesRegex(KeyError, "Available strategies: eager"):
            get_inference_strategy("unit-test-missing")

    def test_factory_return_type_is_checked_at_resolution(self):
        self.register("unit-test-bad-return", lambda: object())

        with self.assertRaisesRegex(
                TypeError,
                "not an InferenceStrategy instance",
        ):
            get_inference_strategy("unit-test-bad-return")

    def test_unregister_rejects_invalid_and_unknown_names(self):
        with self.assertRaisesRegex(TypeError, "names must be strings"):
            unregister_inference_strategy(None)
        with self.assertRaisesRegex(ValueError, "cannot be empty"):
            unregister_inference_strategy("")
        with self.assertRaisesRegex(KeyError, "No inference strategy"):
            unregister_inference_strategy("unit-test-never-registered")


if __name__ == "__main__":
    unittest.main()

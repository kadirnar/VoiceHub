from __future__ import annotations

import json
import unittest

import torch
from torch import nn

from voicehub.optimization import (
    DiffusionGuidanceStrategy,
    DiffusionPredictionCacheMethod,
    DiffusionSamplingCompatibilityError,
    DiffusionSamplingConfig,
    DiffusionSamplingController,
    DiffusionSamplingMixin,
    DiffusionSamplingPass,
    DiffusionSamplingPolicy,
    DiffusionScheduleStrategy,
    DiffusionSolverStrategy,
    DiffusionStepContext,
    OptimizationContext,
    STORK2FlowSolver,
    STORKFlowConfig,
    TTSOptimizationConfig,
    resolve_tts_optimization,
)


def _context() -> OptimizationContext:
    return OptimizationContext(
        mode="inference",
        device="cpu",
        dtype="float32",
    )


def _step(
    index: int,
    total: int,
    *,
    lane: str = "default",
) -> DiffusionStepContext:
    return DiffusionStepContext(
        index=index,
        total_steps=total,
        timestep=float(total - index),
        next_timestep=float(total - index - 1),
        lane=lane,
    )


class _TinySampler(DiffusionSamplingMixin, nn.Module):

    def __init__(self):
        super().__init__()
        self.projection = nn.Linear(2, 2)
        self._initialize_diffusion_sampling()


class _TinySTORKSampler(_TinySampler):
    diffusion_sampling_capabilities = (_TinySampler.diffusion_sampling_capabilities | {"stork2"})


class DiffusionSamplingConfigurationTests(unittest.TestCase):

    def test_config_round_trips_as_strict_json(self):
        config = DiffusionSamplingConfig(
            target_steps=12,
            schedule="trailing",
            guidance="adaptive",
            guidance_start=0.1,
            guidance_end=0.8,
            adaptive_guidance_threshold=0.02,
            adaptive_guidance_warmup_steps=3,
            adaptive_guidance_patience=2,
            prediction_cache="taylorseer",
            cache_interval=3,
            cache_warmup_steps=1,
            cache_max_consecutive_steps=2,
            cache_rel_l1_threshold=0.04,
            cache_error_budget=0.15,
            taylor_order=2,
        )

        payload = config.to_dict()
        encoded = json.dumps(payload, allow_nan=False, sort_keys=True)

        self.assertEqual(config.schedule, DiffusionScheduleStrategy.TRAILING)
        self.assertEqual(config.guidance, DiffusionGuidanceStrategy.ADAPTIVE)
        self.assertEqual(
            config.prediction_cache,
            DiffusionPredictionCacheMethod.TAYLOR,
        )
        self.assertEqual(json.loads(encoded), payload)
        self.assertEqual(DiffusionSamplingConfig.from_dict(payload), config)
        self.assertEqual(
            DiffusionSamplingPolicy.coerce(True),
            DiffusionSamplingPolicy.REQUIRED,
        )

    def test_calibrated_methods_fail_closed_without_calibration(self):
        with self.assertRaisesRegex(ValueError, "teacache_coefficients"):
            DiffusionSamplingConfig(prediction_cache="teacache")
        with self.assertRaisesRegex(
                ValueError,
                "smoothcache_compute_step_mask",
        ):
            DiffusionSamplingConfig(prediction_cache="smoothcache")
        with self.assertRaisesRegex(ValueError, "guidance_start"):
            DiffusionSamplingConfig(
                guidance_start=0.9,
                guidance_end=0.1,
            )
        with self.assertRaisesRegex(ValueError, "taylor_order"):
            DiffusionSamplingConfig(taylor_order=3)
        with self.assertRaisesRegex(ValueError, "cannot be combined"):
            DiffusionSamplingConfig(
                solver="stork-2",
                prediction_cache="fora",
            )
        self.assertEqual(
            DiffusionSamplingConfig(solver="stork").solver,
            DiffusionSolverStrategy.STORK2,
        )


class DiffusionScheduleTests(unittest.TestCase):

    def test_fifty_step_schedule_is_rebuilt_before_integration(self):
        native = torch.linspace(1, 0, 51)
        for strategy in DiffusionScheduleStrategy:
            with self.subTest(strategy=strategy.value):
                controller = DiffusionSamplingController(
                    DiffusionSamplingConfig(
                        target_steps=8,
                        schedule=strategy,
                    ))
                prepared = controller.prepare_schedule(native)

                self.assertEqual(prepared.numel(), 9)
                self.assertEqual(prepared[0], native[0])
                self.assertEqual(prepared[-1], native[-1])
                self.assertTrue(torch.all(prepared[:-1] > prepared[1:]))
                self.assertEqual(controller.stats()["native_steps"], 50)
                self.assertEqual(controller.stats()["prepared_steps"], 8)

    def test_invalid_schedule_fails_before_a_solver_runs(self):
        controller = DiffusionSamplingController(DiffusionSamplingConfig())
        invalid = (
            torch.tensor([0.0]),
            torch.tensor([0.0, 1.0, 0.5]),
            torch.tensor([0.0, float("nan")]),
        )
        for schedule in invalid:
            with self.subTest(schedule=schedule):
                with self.assertRaises(ValueError):
                    controller.prepare_schedule(schedule)

    def test_interpolated_schedule_requires_floating_point_values(self):
        controller = DiffusionSamplingController(
            DiffusionSamplingConfig(
                target_steps=2,
                schedule="quadratic",
            ))

        with self.assertRaisesRegex(
                DiffusionSamplingCompatibilityError,
                "floating-point",
        ):
            controller.prepare_schedule(torch.arange(5))


class DiffusionPredictionCacheTests(unittest.TestCase):

    def test_fora_reduces_model_evaluations_and_isolates_lanes(self):
        controller = DiffusionSamplingController(
            DiffusionSamplingConfig(
                prediction_cache="fora",
                cache_interval=4,
                cache_warmup_steps=1,
                cache_max_consecutive_steps=2,
            ))
        calls = {"conditional": 0, "unconditional": 0}

        def evaluate(index: int, lane: str) -> torch.Tensor:

            def compute() -> torch.Tensor:
                calls[lane] += 1
                return torch.full((1, 2), float(index))

            return controller.evaluate(
                _step(index, 6, lane=lane),
                torch.full((1, 2), float(index)),
                compute,
            )

        conditional = [evaluate(index, "conditional") for index in range(6)]
        unconditional = [evaluate(index, "unconditional") for index in range(6)]

        self.assertLess(calls["conditional"], 6)
        self.assertEqual(calls["conditional"], calls["unconditional"])
        self.assertEqual(conditional[0].shape, unconditional[0].shape)
        self.assertEqual(controller.stats()["predicted_calls"], 6)

    def test_teacache_uses_explicit_rescaling_coefficients(self):
        controller = DiffusionSamplingController(
            DiffusionSamplingConfig(
                prediction_cache="teacache",
                teacache_coefficients=(0.0, 1.0),
                cache_warmup_steps=1,
                cache_rel_l1_threshold=0.5,
                cache_error_budget=1.0,
                cache_max_consecutive_steps=4,
            ))
        calls = 0

        def run(index: int, value: float) -> torch.Tensor:

            def compute() -> torch.Tensor:
                nonlocal calls
                calls += 1
                return torch.full((1, 2), float(calls))

            return controller.evaluate(
                _step(index, 3),
                torch.full((1, 2), value),
                compute,
            )

        first = run(0, 1.0)
        second = run(1, 1.01)
        third = run(2, 2.0)

        self.assertTrue(torch.equal(first, second))
        self.assertFalse(torch.equal(second, third))
        self.assertEqual(calls, 2)

    def test_teacache_accumulates_change_between_adjacent_probes(self):
        controller = DiffusionSamplingController(
            DiffusionSamplingConfig(
                prediction_cache="teacache",
                teacache_coefficients=(0.0, 1.0),
                cache_warmup_steps=1,
                cache_rel_l1_threshold=1.0,
                cache_error_budget=0.25,
                cache_max_consecutive_steps=5,
            ))
        calls = 0

        def run(index: int, value: float) -> torch.Tensor:

            def compute() -> torch.Tensor:
                nonlocal calls
                calls += 1
                return torch.tensor([float(calls)])

            return controller.evaluate(
                _step(index, 3),
                torch.tensor([value]),
                compute,
            )

        first = run(0, 1.0)
        second = run(1, 1.1)
        third = run(2, 1.2)

        self.assertEqual(calls, 1)
        self.assertTrue(torch.equal(first, second))
        self.assertTrue(torch.equal(second, third))

    def test_teacache_incompatible_probe_fails_closed(self):
        controller = DiffusionSamplingController(
            DiffusionSamplingConfig(
                prediction_cache="teacache",
                teacache_coefficients=(0.0, ),
                cache_warmup_steps=1,
                cache_rel_l1_threshold=1.0,
                cache_error_budget=1.0,
                cache_max_consecutive_steps=5,
            ))
        calls = 0

        def run(index: int, size: int) -> torch.Tensor:

            def compute() -> torch.Tensor:
                nonlocal calls
                calls += 1
                return torch.full((size, ), float(calls))

            return controller.evaluate(
                _step(index, 2),
                torch.ones(size),
                compute,
            )

        first = run(0, 1)
        second = run(1, 2)

        self.assertEqual(calls, 2)
        self.assertEqual(first.shape, (1, ))
        self.assertEqual(second.shape, (2, ))

    def test_prediction_caches_refresh_when_probe_metadata_changes(self):
        configs = (
            DiffusionSamplingConfig(
                prediction_cache="fora",
                cache_interval=4,
                cache_warmup_steps=1,
            ),
            DiffusionSamplingConfig(
                prediction_cache="taylor",
                cache_interval=4,
                cache_warmup_steps=1,
            ),
            DiffusionSamplingConfig(
                prediction_cache="smoothcache",
                smoothcache_compute_step_mask=(True, False),
                cache_warmup_steps=1,
            ),
        )
        for config in configs:
            with self.subTest(method=config.prediction_cache.value):
                controller = DiffusionSamplingController(config)
                calls = 0

                def run(index: int, size: int) -> torch.Tensor:

                    def compute() -> torch.Tensor:
                        nonlocal calls
                        calls += 1
                        return torch.full((size, ), float(calls))

                    return controller.evaluate(
                        _step(index, 2),
                        torch.ones(size),
                        compute,
                    )

                first = run(0, 1)
                second = run(1, 2)

                self.assertEqual(calls, 2)
                self.assertEqual(first.shape, (1, ))
                self.assertEqual(second.shape, (2, ))

    def test_smoothcache_mask_must_match_prepared_step_count(self):
        controller = DiffusionSamplingController(
            DiffusionSamplingConfig(
                prediction_cache="smoothcache",
                smoothcache_compute_step_mask=(True, False),
            ))
        with self.assertRaisesRegex(
                DiffusionSamplingCompatibilityError,
                "mask length",
        ):
            controller.evaluate(
                _step(0, 3),
                torch.ones(1),
                lambda: torch.ones(1),
            )

    def test_taylor_extrapolates_from_computed_outputs(self):
        controller = DiffusionSamplingController(
            DiffusionSamplingConfig(
                prediction_cache="taylor",
                cache_interval=2,
                cache_warmup_steps=1,
                cache_max_consecutive_steps=1,
                taylor_order=1,
            ))
        outputs = []
        calls = 0
        for index in range(4):

            def compute(index=index) -> torch.Tensor:
                nonlocal calls
                calls += 1
                return torch.tensor([float(index)])

            outputs.append(controller.evaluate(
                _step(index, 4),
                torch.tensor([float(index)]),
                compute,
            ))

        self.assertEqual(calls, 2)
        self.assertEqual(outputs[3].item(), 3.0)


class STORKFlowSolverTests(unittest.TestCase):

    def test_constant_velocity_is_exact_for_forward_and_reverse_time(self):
        for start, end, velocity_value in (
            (0.0, 1.0, 2.0),
            (1.0, 0.0, 2.0),
        ):
            with self.subTest(start=start, end=end):
                solver = STORK2FlowSolver(STORKFlowConfig(stages=9))
                schedule = torch.linspace(start, end, 11)
                state = torch.zeros(2, 3, 4, dtype=torch.float16)
                velocity = torch.full_like(state, velocity_value)
                for index in range(10):
                    state = solver.advance(
                        state,
                        velocity,
                        timestep=schedule[index],
                        next_timestep=schedule[index + 1],
                    )

                expected = velocity_value * (end - start)
                self.assertTrue(
                    torch.allclose(
                        state.float(),
                        torch.full_like(state.float(), expected),
                        atol=2e-3,
                        rtol=0,
                    ))
                self.assertEqual(solver.stats()["model_evaluations"], 10)
                self.assertEqual(solver.stats()["startup_steps"], 1)
                self.assertEqual(solver.stats()["stabilized_steps"], 9)

    def test_solver_is_shape_generic_and_resets_history(self):
        solver = STORK2FlowSolver()
        first = solver.advance(
            torch.zeros(1, 5, 3),
            torch.ones(1, 5, 3),
            timestep=0.0,
            next_timestep=0.1,
        )
        second = solver.advance(
            first,
            torch.ones_like(first),
            timestep=0.1,
            next_timestep=0.2,
            discontinuity=True,
        )

        self.assertEqual(second.shape, (1, 5, 3))
        self.assertEqual(solver.stats()["startup_steps"], 2)

    def test_controller_runs_stork_after_schedule_compaction(self):
        controller = DiffusionSamplingController(
            DiffusionSamplingConfig(
                target_steps=20,
                solver="stork2",
                stork_stages=7,
            ))
        schedule = controller.prepare_schedule(torch.linspace(0, 1, 51))
        state = torch.zeros(1, 2)
        calls = 0
        for index in range(schedule.numel() - 1):
            calls += 1
            context = DiffusionStepContext(
                index=index,
                total_steps=schedule.numel() - 1,
                timestep=schedule[index],
                next_timestep=schedule[index + 1],
                lane="solver",
                solver="stork2",
            )
            state = controller.advance(
                context,
                state,
                torch.ones_like(state),
            )

        self.assertEqual(calls, 20)
        self.assertTrue(torch.allclose(state, torch.ones_like(state)))
        self.assertEqual(controller.stats()["solver_steps"], 20)


class DiffusionGuidanceTests(unittest.TestCase):

    def test_limited_interval_only_narrows_native_guidance(self):
        controller = DiffusionSamplingController(
            DiffusionSamplingConfig(
                guidance="limited_interval",
                guidance_start=0.25,
                guidance_end=0.75,
            ))
        decisions = [controller.should_use_guidance(_step(index, 5), native=True) for index in range(5)]

        self.assertEqual(decisions, [False, True, True, True, False])
        self.assertFalse(controller.should_use_guidance(_step(2, 5), native=False))

    def test_adaptive_guidance_stops_after_convergence_patience(self):
        controller = DiffusionSamplingController(
            DiffusionSamplingConfig(
                guidance="adaptive",
                adaptive_guidance_threshold=0.05,
                adaptive_guidance_warmup_steps=0,
                adaptive_guidance_patience=2,
            ))
        conditional = torch.ones(2, 3)
        unconditional = conditional * 1.01

        for index in range(2):
            context = _step(index, 4, lane="guidance")
            self.assertTrue(controller.should_use_guidance(context, native=True))
            controller.observe_guidance(
                context,
                conditional,
                unconditional,
            )

        self.assertFalse(controller.should_use_guidance(
            _step(2, 4, lane="guidance"),
            native=True,
        ))


class DiffusionSamplingPassTests(unittest.TestCase):

    def test_pass_applies_and_restores_explicit_sampler_surface(self):
        model = _TinySampler()
        config = DiffusionSamplingConfig(
            target_steps=6,
            prediction_cache="fora",
        )
        optimization_pass = DiffusionSamplingPass(config)

        result = optimization_pass.apply(model, _context())

        self.assertEqual(model.diffusion_sampling_config, config)
        self.assertEqual(result.metadata["targets"], ["model"])
        self.assertEqual(
            optimization_pass.restore(model, result.state, _context()),
            model,
        )
        self.assertIsNone(model.diffusion_sampling_config)

    def test_stork_requires_an_explicit_direct_velocity_adapter(self):
        unsupported = _TinySampler()
        supported = _TinySTORKSampler()
        config = DiffusionSamplingConfig(solver="stork2", target_steps=20)

        with self.assertRaisesRegex(
                DiffusionSamplingCompatibilityError,
                "stork2",
        ):
            unsupported.enable_diffusion_sampling(config)
        self.assertEqual(
            supported.enable_diffusion_sampling(config),
            config,
        )

    def test_universal_tts_config_serializes_diffusion_sampling(self):
        config = TTSOptimizationConfig(
            compile=False,
            attn_implementation="native",
            kernel_backend="native",
            diffusion_sampling=True,
            diffusion_sampling_config={
                "target_steps": 8,
                "prediction_cache": "fora",
            },
        )
        payload = config.to_dict()

        self.assertEqual(payload["diffusion_sampling"], "required")
        self.assertEqual(
            payload["diffusion_sampling_config"]["target_steps"],
            8,
        )
        plan = resolve_tts_optimization(
            "f5tts",
            config,
            context=_context(),
        )
        self.assertIn(
            "voicehub.diffusion-sampling@1",
            [optimization_pass.qualified_id for optimization_pass in plan],
        )


if __name__ == "__main__":
    unittest.main()

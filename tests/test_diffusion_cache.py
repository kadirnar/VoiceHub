from __future__ import annotations

import json
import unittest

import torch
from torch import nn

from voicehub.optimization import (
    DiffusionBlockResidualCache,
    DiffusionCacheConfig,
    DiffusionCacheMethod,
    DiffusionCacheMixin,
    DiffusionCachePass,
    DiffusionCachePolicy,
    DiffusionCachePredictor,
    OptimizationContext,
    TTSOptimizationCompatibilityError,
    TTSOptimizationConfig,
    resolve_tts_optimization,
)


def _context(mode: str = "inference") -> OptimizationContext:
    return OptimizationContext(
        mode=mode,
        device="cpu",
        dtype="float32",
    )


class _CountingBlock(nn.Module):

    def __init__(self, operation: str = "add", value: float = 1.0):
        super().__init__()
        self.operation = operation
        self.value = nn.Parameter(torch.tensor(float(value)))
        self.calls = 0

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        self.calls += 1
        if self.operation == "add":
            return hidden_states + self.value
        if self.operation == "multiply":
            return hidden_states * self.value
        raise AssertionError(f"Unknown test operation: {self.operation}")


def _run(
    cache: DiffusionBlockResidualCache,
    blocks: tuple[_CountingBlock, ...],
    hidden_states: torch.Tensor,
    **kwargs,
) -> torch.Tensor:
    return cache.run(
        blocks,
        hidden_states,
        lambda block, value: block(value),
        **kwargs,
    )


class _TinyCachedDiT(DiffusionCacheMixin, nn.Module):

    def __init__(self):
        super().__init__()
        self.blocks = nn.ModuleList([
            _CountingBlock("add", 1.0),
            _CountingBlock("add", 2.0),
            _CountingBlock("add", 3.0),
            _CountingBlock("add", 4.0),
        ], )
        self._initialize_diffusion_cache()

    def forward(
        self,
        hidden_states: torch.Tensor,
        *,
        lane: str = "default",
        valid_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self._run_diffusion_blocks(
            self.blocks,
            hidden_states,
            lambda block, value: block(value),
            cache_lane=lane,
            valid_mask=valid_mask,
        )


class DiffusionCacheConfigurationTests(unittest.TestCase):

    def test_config_round_trips_as_strict_json(self):
        config = DiffusionCacheConfig.from_dict({
            "front_blocks": 2,
            "back_blocks": 1,
            "residual_diff_threshold": 0.125,
            "warmup_steps": 3,
            "max_cached_steps": 11,
            "max_consecutive_cached_steps": 2,
            "max_accumulated_relative_error": 0.4,
            "predictor": "taylorseer",
            "compute_step_mask": [True, False, True],
            "synchronize_distributed": False,
            "epsilon": 1e-5,
        })

        payload = config.to_dict()
        encoded = json.dumps(payload, allow_nan=False, sort_keys=True)

        self.assertEqual(config.predictor, DiffusionCachePredictor.TAYLOR)
        self.assertEqual(config.method, DiffusionCacheMethod.DBCACHE)
        self.assertEqual(config.compute_step_mask, (True, False, True))
        self.assertEqual(json.loads(encoded), payload)
        self.assertEqual(DiffusionCacheConfig.from_dict(payload), config)
        self.assertEqual(
            DiffusionCachePolicy.coerce(True),
            DiffusionCachePolicy.REQUIRED,
        )
        self.assertEqual(
            DiffusionCachePolicy.coerce(False),
            DiffusionCachePolicy.DISABLED,
        )

    def test_invalid_configurations_fail_closed(self):
        invalid = (
            ({
                "front_blocks": 0
            }, ValueError, "front_blocks"),
            ({
                "back_blocks": -1
            }, ValueError, "back_blocks"),
            ({
                "residual_diff_threshold": float("nan")
            }, ValueError, "residual_diff_threshold"),
            ({
                "max_cached_steps": -2
            }, ValueError, "max_cached_steps"),
            ({
                "max_consecutive_cached_steps": -2
            }, ValueError, "max_consecutive_cached_steps"),
            ({
                "max_accumulated_relative_error": 0.0
            }, ValueError, "max_accumulated_relative_error"),
            ({
                "compute_step_mask": [True, 1]
            }, TypeError, "compute_step_mask"),
            ({
                "epsilon": 0.0
            }, ValueError, "epsilon"),
            ({
                "predictor": "not-a-predictor"
            }, ValueError, "predictor"),
        )
        for values, error_type, message in invalid:
            with self.subTest(values=values):
                with self.assertRaisesRegex(error_type, message):
                    DiffusionCacheConfig(**values)
        with self.assertRaisesRegex(ValueError, "Unknown diffusion-cache"):
            DiffusionCachePolicy.coerce("sometimes")
        with self.assertRaisesRegex(ValueError, "First-block cache"):
            DiffusionCacheConfig(
                method="fbcache",
                front_blocks=2,
            )


class DiffusionBlockResidualCacheTests(unittest.TestCase):

    @staticmethod
    def _additive_blocks(count: int = 4) -> tuple[_CountingBlock, ...]:
        return tuple(_CountingBlock("add", float(index + 1)) for index in range(count))

    def test_cache_executes_first_and_last_but_skips_middle_on_hit(self):
        blocks = self._additive_blocks()
        cache = DiffusionBlockResidualCache(
            DiffusionCacheConfig(
                front_blocks=1,
                back_blocks=1,
                warmup_steps=0,
                max_consecutive_cached_steps=-1,
            ), )
        value = torch.ones(1, 2, 3)

        with torch.inference_mode():
            eager = _run(cache, blocks, value)
            cached = _run(cache, blocks, value)

        self.assertTrue(torch.equal(cached, eager))
        self.assertEqual([block.calls for block in blocks], [2, 1, 1, 2])

    def test_warmup_computes_middle_before_first_cache_hit(self):
        blocks = self._additive_blocks()
        cache = DiffusionBlockResidualCache(
            DiffusionCacheConfig(
                front_blocks=1,
                back_blocks=1,
                warmup_steps=2,
                max_consecutive_cached_steps=-1,
            ), )
        value = torch.ones(1, 2, 3)

        with torch.inference_mode():
            for _ in range(3):
                _run(cache, blocks, value)

        self.assertEqual([block.calls for block in blocks], [3, 2, 2, 3])
        self.assertEqual(cache.stats()["warmup_misses"], 1)
        self.assertEqual(cache.stats()["cached_steps"], 1)

    def test_threshold_selects_hits_and_recomputes_changed_probes(self):
        blocks = self._additive_blocks(3)
        cache = DiffusionBlockResidualCache(
            DiffusionCacheConfig(
                front_blocks=1,
                residual_diff_threshold=0.1,
                warmup_steps=0,
                max_consecutive_cached_steps=-1,
            ), )
        baseline = torch.ones(1, 2, 2)

        with torch.inference_mode():
            _run(cache, blocks, baseline)
            _run(cache, blocks, baseline.clone())
            _run(cache, blocks, baseline * 4)

        self.assertEqual([block.calls for block in blocks], [3, 2, 2])
        stats = cache.stats()
        self.assertEqual(stats["cached_steps"], 1)
        self.assertEqual(stats["computed_steps"], 2)
        self.assertEqual(stats["threshold_misses"], 1)
        self.assertGreater(stats["last_relative_difference"], 0.1)

    def test_max_consecutive_hits_forces_a_periodic_refresh(self):
        blocks = self._additive_blocks(3)
        cache = DiffusionBlockResidualCache(
            DiffusionCacheConfig(
                front_blocks=1,
                warmup_steps=0,
                max_consecutive_cached_steps=1,
            ), )
        value = torch.ones(1, 2, 2)

        with torch.inference_mode():
            for _ in range(4):
                _run(cache, blocks, value)

        self.assertEqual([block.calls for block in blocks], [4, 2, 2])
        stats = cache.stats()
        self.assertEqual(stats["computed_steps"], 2)
        self.assertEqual(stats["cached_steps"], 2)
        self.assertEqual(stats["limit_misses"], 1)

    def test_cfg_lanes_do_not_share_probe_or_residual_state(self):
        blocks = self._additive_blocks(3)
        cache = DiffusionBlockResidualCache(
            DiffusionCacheConfig(
                front_blocks=1,
                warmup_steps=0,
                max_consecutive_cached_steps=-1,
            ), )

        with torch.inference_mode():
            for lane, value in (
                ("conditional", torch.ones(1, 2, 2)),
                ("unconditional", torch.full((1, 2, 2), 3.0)),
                ("conditional", torch.ones(1, 2, 2)),
                ("unconditional", torch.full((1, 2, 2), 3.0)),
            ):
                _run(cache, blocks, value, lane=lane)

        self.assertEqual([block.calls for block in blocks], [4, 2, 2])
        self.assertEqual(cache.stats()["computed_steps"], 2)
        self.assertEqual(cache.stats()["cached_steps"], 2)

    def test_request_session_cleans_state_even_after_an_exception(self):
        blocks = self._additive_blocks(3)
        cache = DiffusionBlockResidualCache(
            DiffusionCacheConfig(
                front_blocks=1,
                warmup_steps=0,
                max_consecutive_cached_steps=-1,
            ), )
        value = torch.ones(1, 2, 2)

        with self.assertRaisesRegex(RuntimeError, "request failed"):
            with cache.session():
                with torch.inference_mode():
                    _run(cache, blocks, value)
                    _run(cache, blocks, value)
                raise RuntimeError("request failed")
        self.assertEqual(cache._states, {})

        with cache.session():
            with torch.inference_mode():
                _run(cache, blocks, value)

        self.assertEqual(cache._states, {})
        self.assertEqual([block.calls for block in blocks], [3, 2, 2])
        self.assertEqual(cache.stats()["sessions"], 2)

    def test_valid_mask_ignores_padding_but_detects_valid_changes(self):
        blocks = self._additive_blocks(2)
        cache = DiffusionBlockResidualCache(
            DiffusionCacheConfig(
                front_blocks=1,
                warmup_steps=0,
                residual_diff_threshold=0.1,
                max_consecutive_cached_steps=-1,
            ), )
        mask = torch.tensor([[True, True, False]])
        baseline = torch.ones(1, 3, 1)
        padding_changed = baseline.clone()
        padding_changed[:, 2] = 100.0
        valid_changed = padding_changed.clone()
        valid_changed[:, 0] = 3.0

        with torch.inference_mode():
            _run(cache, blocks, baseline, valid_mask=mask)
            _run(cache, blocks, padding_changed, valid_mask=mask)
            _run(cache, blocks, valid_changed, valid_mask=mask)

        self.assertEqual([block.calls for block in blocks], [3, 2])
        self.assertEqual(cache.stats()["cached_steps"], 1)
        self.assertEqual(cache.stats()["threshold_misses"], 1)

    def test_gradient_and_training_calls_bypass_cache(self):
        gradient_blocks = self._additive_blocks(2)
        gradient_cache = DiffusionBlockResidualCache(DiffusionCacheConfig(front_blocks=1, warmup_steps=0), )
        first = torch.ones(1, 2, 2, requires_grad=True)
        second = torch.ones(1, 2, 2, requires_grad=True)

        with torch.enable_grad():
            output = _run(gradient_cache, gradient_blocks, first)
            output.sum().backward()
            _run(gradient_cache, gradient_blocks, second)

        self.assertIsNotNone(first.grad)
        self.assertEqual(
            [block.calls for block in gradient_blocks],
            [2, 2],
        )
        self.assertEqual(gradient_cache.stats()["bypassed_steps"], 2)
        self.assertEqual(gradient_cache.stats()["computed_steps"], 0)

        training_blocks = self._additive_blocks(2)
        training_cache = DiffusionBlockResidualCache(DiffusionCacheConfig(front_blocks=1, warmup_steps=0), )
        with torch.no_grad():
            _run(
                training_cache,
                training_blocks,
                torch.ones(1, 2, 2),
                training=True,
            )
            _run(
                training_cache,
                training_blocks,
                torch.ones(1, 2, 2),
                training=True,
            )
        self.assertEqual(
            [block.calls for block in training_blocks],
            [2, 2],
        )
        self.assertEqual(training_cache.stats()["bypassed_steps"], 2)
        self.assertEqual(training_cache.stats()["cached_steps"], 0)

    def test_taylor_predictor_uses_computed_step_distance(self):
        front = _CountingBlock("add", 0.0)
        middle = _CountingBlock("multiply", 2.0)
        blocks = (front, middle)
        cache = DiffusionBlockResidualCache(
            DiffusionCacheConfig(
                front_blocks=1,
                warmup_steps=2,
                residual_diff_threshold=1.0,
                max_consecutive_cached_steps=-1,
                predictor="taylor",
            ), )

        with torch.inference_mode():
            first = _run(cache, blocks, torch.tensor([[[1.0]]]))
            second = _run(cache, blocks, torch.tensor([[[2.0]]]))
            predicted = _run(cache, blocks, torch.tensor([[[3.0]]]))

        self.assertEqual(first.item(), 2.0)
        self.assertEqual(second.item(), 4.0)
        self.assertEqual(predicted.item(), 6.0)
        self.assertEqual(middle.calls, 2)
        self.assertEqual(cache.stats()["cached_steps"], 1)

    def test_runtime_stats_report_cache_effectiveness(self):
        blocks = self._additive_blocks(3)
        cache = DiffusionBlockResidualCache(
            DiffusionCacheConfig(
                front_blocks=1,
                warmup_steps=0,
                max_consecutive_cached_steps=-1,
            ), )
        value = torch.ones(1, 2, 2)

        with torch.inference_mode():
            _run(cache, blocks, value)
            _run(cache, blocks, value)

        self.assertEqual(
            cache.stats(),
            {
                "calls": 2,
                "computed_steps": 1,
                "cached_steps": 1,
                "bypassed_steps": 0,
                "warmup_misses": 0,
                "threshold_misses": 0,
                "limit_misses": 0,
                "invalidations": 0,
                "sessions": 0,
                "hit_rate": 0.5,
                "last_relative_difference": 0.0,
                "maximum_relative_difference": 0.0,
            },
        )


class DiffusionCachePassTests(unittest.TestCase):

    def test_pass_is_reversible_and_preserves_state_dict_keys(self):
        model = _TinyCachedDiT().eval()
        original_keys = tuple(model.state_dict())
        previous = DiffusionCacheConfig(
            front_blocks=1,
            warmup_steps=1,
            residual_diff_threshold=0.2,
        )
        model.enable_diffusion_cache(previous)
        optimization_pass = DiffusionCachePass({
            "front_blocks": 1,
            "back_blocks": 1,
            "warmup_steps": 0,
            "max_consecutive_cached_steps": -1,
        })
        context = _context()

        optimization_pass.validate(model, context)
        result = optimization_pass.apply(model, context)

        self.assertIs(result.model, model)
        self.assertEqual(
            model.diffusion_cache_config,
            optimization_pass.config,
        )
        self.assertEqual(tuple(model.state_dict()), original_keys)
        self.assertEqual(result.metadata["targets"], ["model"])

        with torch.inference_mode():
            model(torch.ones(1, 2, 2))
            model(torch.ones(1, 2, 2))
        runtime = optimization_pass.runtime_manifest_status(result)
        self.assertEqual(runtime["model"]["calls"], 2)
        self.assertEqual(runtime["model"]["cached_steps"], 1)

        restored = optimization_pass.restore(model, result.state, context)

        self.assertIs(restored, model)
        self.assertEqual(model.diffusion_cache_config, previous)
        self.assertEqual(tuple(model.state_dict()), original_keys)


class NativeDiffusionCacheAdapterTests(unittest.TestCase):

    @staticmethod
    def _cache_config() -> DiffusionCacheConfig:
        return DiffusionCacheConfig(
            front_blocks=1,
            warmup_steps=0,
            residual_diff_threshold=1.0,
            max_consecutive_cached_steps=-1,
        )

    def test_f5_and_cosyvoice_flat_dit_adapters_take_cache_hits(self):
        from voicehub.architectures.cosyvoice_native.configuration import CosyVoiceArchitectureConfig
        from voicehub.architectures.cosyvoice_native.flow import DiTEstimator
        from voicehub.architectures.f5tts.configuration import F5TTSArchitectureConfig
        from voicehub.architectures.f5tts.modeling import F5DiT

        f5_config = F5TTSArchitectureConfig(
            model_name="cache-test",
            mel_dim=8,
            dim=32,
            depth=2,
            heads=4,
            dim_head=8,
            text_dim=16,
            text_num_embeds=12,
            conv_layers=1,
            n_fft=32,
            win_length=32,
            hop_length=8,
            sample_rate=8_000,
            dropout=0.0,
        )
        f5 = F5DiT(f5_config).eval()
        f5.enable_diffusion_cache(self._cache_config())
        hidden = torch.randn(1, 4, 8)
        conditioning = torch.randn_like(hidden)
        token_ids = torch.ones(1, 4, dtype=torch.long)
        timestep = torch.tensor([0.5])
        mask = torch.ones(1, 4, dtype=torch.bool)

        cosy_config = CosyVoiceArchitectureConfig.tiny().flow
        cosy = DiTEstimator(cosy_config).eval()
        cosy.enable_diffusion_cache(self._cache_config())
        values = torch.randn(1, cosy_config.mel_channels, 4)
        flow_mask = torch.ones(1, 1, 4)
        means = torch.randn_like(values)
        speakers = torch.randn(1, cosy_config.mel_channels)
        flow_conditioning = torch.randn_like(values)

        with torch.inference_mode():
            first_f5 = f5(
                hidden,
                conditioning,
                token_ids,
                timestep,
                mask=mask,
            )
            second_f5 = f5(
                hidden,
                conditioning,
                token_ids,
                timestep,
                mask=mask,
            )
            first_cosy = cosy(
                values,
                flow_mask,
                means,
                timestep,
                speakers,
                flow_conditioning,
                diffusion_cache_lane="conditional",
            )
            second_cosy = cosy(
                values,
                flow_mask,
                means,
                timestep,
                speakers,
                flow_conditioning,
                diffusion_cache_lane="conditional",
            )

        torch.testing.assert_close(second_f5, first_f5)
        torch.testing.assert_close(second_cosy, first_cosy)
        self.assertEqual(f5.diffusion_cache_stats()["cached_steps"], 1)
        self.assertEqual(cosy.diffusion_cache_stats()["cached_steps"], 1)

    def test_irodori_and_vibevoice_custom_adapters_take_cache_hits(self):
        from voicehub.architectures.irodoritts.configuration import IrodoriModelConfig
        from voicehub.architectures.irodoritts.modeling import TextToLatentRFDiT
        from voicehub.architectures.vibevoice.configuration import VibeVoiceDiffusionConfig
        from voicehub.architectures.vibevoice.diffusion import VibeVoiceDiffusionHead

        irodori_config = IrodoriModelConfig(
            adaln_rank=8,
            latent_dim=4,
            mlp_ratio=2.0,
            model_dim=16,
            num_heads=2,
            num_layers=2,
            speaker_dim=16,
            speaker_heads=2,
            speaker_layers=1,
            speaker_mlp_ratio=2.0,
            text_dim=16,
            text_heads=2,
            text_layers=1,
            text_mlp_ratio=2.0,
            text_vocab_size=266,
            timestep_embed_dim=8,
            variant="custom",
        )
        irodori = TextToLatentRFDiT(irodori_config).eval()
        irodori.enable_diffusion_cache(self._cache_config())
        latent = torch.randn(1, 3, 4)
        timestep = torch.tensor([0.5])
        text_ids = torch.ones(1, 3, dtype=torch.long)
        text_mask = torch.ones(1, 3, dtype=torch.bool)
        reference = torch.randn(1, 2, 4)
        reference_mask = torch.ones(1, 2, dtype=torch.bool)
        latent_mask = torch.ones(1, 3, dtype=torch.bool)

        vibe_config = VibeVoiceDiffusionConfig(
            hidden_size=8,
            head_layers=2,
            head_ffn_ratio=2.0,
            latent_size=2,
            ddpm_num_steps=10,
            ddpm_num_inference_steps=2,
            ddpm_batch_mul=1,
        )
        vibe = VibeVoiceDiffusionHead(vibe_config).eval()
        vibe.enable_diffusion_cache(self._cache_config())
        noisy_latents = torch.randn(2, 2)
        vibe_timesteps = torch.ones(2)
        vibe_condition = torch.randn(2, 8)

        with torch.inference_mode():
            first_irodori = irodori(
                latent,
                timestep,
                text_ids,
                text_mask,
                reference,
                reference_mask,
                latent_mask=latent_mask,
            )
            second_irodori = irodori(
                latent,
                timestep,
                text_ids,
                text_mask,
                reference,
                reference_mask,
                latent_mask=latent_mask,
            )
            first_vibe = vibe(
                noisy_latents,
                vibe_timesteps,
                vibe_condition,
            )
            second_vibe = vibe(
                noisy_latents,
                vibe_timesteps,
                vibe_condition,
            )

        torch.testing.assert_close(second_irodori, first_irodori)
        torch.testing.assert_close(second_vibe, first_vibe)
        self.assertEqual(irodori.diffusion_cache_stats()["cached_steps"], 1)
        self.assertEqual(vibe.diffusion_cache_stats()["cached_steps"], 1)


class UniversalDiffusionCacheResolutionTests(unittest.TestCase):

    def test_supported_cache_is_ordered_before_compile(self):
        plan = resolve_tts_optimization(
            "f5tts",
            TTSOptimizationConfig(
                attn_implementation="native",
                kernel_backend="native",
                diffusion_cache="required",
                diffusion_cache_config={
                    "front_blocks": 2,
                    "predictor": "taylor",
                },
                compile=True,
                compile_config={"backend": "eager"},
            ),
            context=_context(),
        )

        self.assertEqual(
            [item.compatibility_kind for item in plan.passes],
            ["diffusion-cache", "compile"],
        )
        self.assertEqual(
            [decision.feature for decision in plan.decisions],
            [
                "kernels",
                "attention",
                "diffusion_sampling",
                "diffusion_cache",
                "compile",
            ],
        )
        cache_decision = plan.decisions[3]
        self.assertEqual(cache_decision.requested, "required")
        self.assertEqual(
            cache_decision.selected,
            "block-residual-cache:dbcache:taylor",
        )

    def test_unsupported_auto_falls_back_and_required_fails(self):
        automatic = resolve_tts_optimization(
            "vits",
            TTSOptimizationConfig(
                attn_implementation="native",
                kernel_backend="native",
                diffusion_cache="auto",
                compile=False,
            ),
            context=_context(),
        )
        decision = next(item for item in automatic.decisions if item.feature == "diffusion_cache")

        self.assertEqual(automatic.passes, ())
        self.assertEqual(decision.selected, "unsupported")
        self.assertIn("does not declare", decision.reason)
        with self.assertRaisesRegex(
                TTSOptimizationCompatibilityError,
                "does not declare an architecture-owned diffusion-cache",
        ):
            resolve_tts_optimization(
                "vits",
                TTSOptimizationConfig(
                    attn_implementation="native",
                    kernel_backend="native",
                    diffusion_cache="required",
                    compile=False,
                ),
                context=_context(),
            )


if __name__ == "__main__":
    unittest.main()

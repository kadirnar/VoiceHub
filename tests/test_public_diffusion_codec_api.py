import json
import subprocess
import sys


def test_optimization_facade_exports_complete_diffusion_and_codec_apis():
    import voicehub.optimization as optimization
    from voicehub.optimization import codecs, diffusion, diffusion_cache, diffusion_sampling, diffusion_solvers

    modules = (
        codecs,
        diffusion,
        diffusion_cache,
        diffusion_sampling,
        diffusion_solvers,
    )
    expected = set().union(*(set(module.__all__) for module in modules))
    assert expected <= set(optimization.__all__)
    assert set(optimization.__all__) <= set(dir(optimization))
    assert len(optimization.__all__) == len(set(optimization.__all__))
    for module in modules:
        for name in module.__all__:
            assert getattr(optimization, name) is getattr(module, name)


def test_codec_package_exports_complete_structural_api():
    import voicehub.components.audio.codecs as codecs
    from voicehub.components.audio.codecs import base, catalog

    assert set(codecs.__all__) == set(base.__all__) | set(catalog.__all__)
    assert set(codecs.__all__) <= set(dir(codecs))
    assert len(codecs.__all__) == len(set(codecs.__all__))
    for module in (base, catalog):
        for name in module.__all__:
            assert getattr(codecs, name) is getattr(module, name)


def test_root_facade_exports_practical_diffusion_and_codec_apis():
    import voicehub
    from voicehub.components.audio.codecs import base, catalog
    from voicehub.optimization import codecs, diffusion, diffusion_cache, diffusion_sampling, diffusion_solvers

    expected = {
        "AudioAutoencoderView": base.AudioAutoencoderView,
        "AudioCodec": base.AudioCodec,
        "AudioCodecComponentView": base.AudioCodecComponentView,
        "CODEC_CATALOG": catalog.CODEC_CATALOG,
        "CodecCatalogEntry": catalog.CodecCatalogEntry,
        "CodecCodeBatch": base.CodecCodeBatch,
        "CodecIntegration": catalog.CodecIntegration,
        "CodecRepresentation": catalog.CodecRepresentation,
        "CodecStageAvailability": catalog.CodecStageAvailability,
        "DenseCodecCodes": base.DenseCodecCodes,
        "RaggedCodecCodes": base.RaggedCodecCodes,
        "codec_is_stochastic_vae": base.codec_is_stochastic_vae,
        "codec_target_is_stochastic": base.codec_target_is_stochastic,
        "coerce_codec_codes": base.coerce_codec_codes,
        "get_codec_entries_for_model": catalog.get_codec_entries_for_model,
        "get_codec_entry": catalog.get_codec_entry,
        "list_codec_entries": catalog.list_codec_entries,
        "separate_audio_codec": base.separate_audio_codec,
        "CodecCUDAGraphCaptureError": codecs.CodecCUDAGraphCaptureError,
        "CodecCUDAGraphRunner": codecs.CodecCUDAGraphRunner,
        "CodecCompileComponent": codecs.CodecCompileComponent,
        "CodecCompilePolicy": codecs.CodecCompilePolicy,
        "CodecKernelBackend": codecs.CodecKernelBackend,
        "CodecKernelPass": codecs.CodecKernelPass,
        "CodecOptimizationCompatibilityError": codecs.CodecOptimizationCompatibilityError,
        "CodecOptimizationConfig": codecs.CodecOptimizationConfig,
        "CodecOptimizationPlan": codecs.CodecOptimizationPlan,
        "CodecOptimizationPolicy": codecs.CodecOptimizationPolicy,
        "CodecOptimizationResult": codecs.CodecOptimizationResult,
        "capture_codec_cuda_graph": codecs.capture_codec_cuda_graph,
        "discover_codec_compile_targets": codecs.discover_codec_compile_targets,
        "optimize_codec": codecs.optimize_codec,
        "resolve_codec_optimization": codecs.resolve_codec_optimization,
        "DiffusionArchitectureKind": diffusion.DiffusionArchitectureKind,
        "DiffusionCacheMethod": diffusion_cache.DiffusionCacheMethod,
        "DiffusionGuidanceStrategy": diffusion_sampling.DiffusionGuidanceStrategy,
        "DiffusionModelOptimizationSupport": diffusion.DiffusionModelOptimizationSupport,
        "DiffusionOperation": diffusion.DiffusionOperation,
        "DiffusionPredictionCacheMethod": diffusion_sampling.DiffusionPredictionCacheMethod,
        "DiffusionSamplingConfig": diffusion_sampling.DiffusionSamplingConfig,
        "DiffusionSamplingController": diffusion_sampling.DiffusionSamplingController,
        "DiffusionSamplingPass": diffusion_sampling.DiffusionSamplingPass,
        "DiffusionSamplingPolicy": diffusion_sampling.DiffusionSamplingPolicy,
        "DiffusionScheduleStrategy": diffusion_sampling.DiffusionScheduleStrategy,
        "DiffusionSolverStrategy": diffusion_sampling.DiffusionSolverStrategy,
        "DiffusionStepContext": diffusion_sampling.DiffusionStepContext,
        "STORK2FlowSolver": diffusion_solvers.STORK2FlowSolver,
        "STORKFlowConfig": diffusion_solvers.STORKFlowConfig,
        "get_diffusion_model_optimization_support": diffusion.get_diffusion_model_optimization_support,
        "list_diffusion_model_optimization_support": diffusion.list_diffusion_model_optimization_support,
    }

    assert set(expected) <= set(voicehub.__all__)
    assert set(voicehub.__all__) <= set(dir(voicehub))
    assert len(voicehub.__all__) == len(set(voicehub.__all__))
    for name, value in expected.items():
        assert getattr(voicehub, name) is value


def test_public_facades_do_not_eagerly_load_compilers_or_model_graphs():
    code = """
import json
import sys

import voicehub
import voicehub.components.audio.codecs as codec_api
import voicehub.optimization as optimization

before = {
    "codec_base": "voicehub.components.audio.codecs.base" in sys.modules,
    "codec_optimization": "voicehub.optimization.codecs" in sys.modules,
    "diffusion": "voicehub.optimization.diffusion" in sys.modules,
}
discoverable = all(
    name in dir(module) and name in module.__all__
    for module, names in (
        (
            voicehub,
            (
                "CODEC_CATALOG",
                "CodecOptimizationConfig",
                "DiffusionArchitectureKind",
                "DenseCodecCodes",
            ),
        ),
        (
            optimization,
            (
                "CodecOptimizationConfig",
                "DiffusionArchitectureKind",
            ),
        ),
        (codec_api, ("AudioCodecComponentView", "CodecCodeBatch")),
    )
    for name in names
)

_ = voicehub.CodecOptimizationConfig
_ = voicehub.DiffusionArchitectureKind
_ = voicehub.DenseCodecCodes
_ = voicehub.CODEC_CATALOG

blocked = (
    "torch.utils.cpp_extension",
    "voicehub.kernels.triton_activations",
    "voicehub.architectures.f5tts.modeling",
    "voicehub.architectures.irodoritts.modeling",
    "voicehub.architectures.vibevoice.diffusion",
    "voicehub.models.echo.autoencoder",
    "voicehub.models.echo.model",
)
after = {
    "blocked": sorted(name for name in blocked if name in sys.modules),
    "triton": sorted(
        name
        for name in sys.modules
        if name == "triton" or name.startswith("triton.")
    ),
}
print(json.dumps({
    "after": after,
    "before": before,
    "discoverable": discoverable,
}))
"""
    completed = subprocess.run(
        (sys.executable, "-c", code),
        check=True,
        capture_output=True,
        text=True,
    )

    assert json.loads(completed.stdout) == {
        "after": {
            "blocked": [],
            "triton": [],
        },
        "before": {
            "codec_base": False,
            "codec_optimization": False,
            "diffusion": False,
        },
        "discoverable": True,
    }

"""
llama.cpp inference backend for MOSS-TTS-Delay.

This package provides a torch-free (or torch-optional) end-to-end TTS
pipeline using:
  - llama.cpp for the Qwen3 backbone (GGUF format)
  - NumPy for embeddings, LM heads, delay state machine, and sampling
  - ONNX Runtime or TensorRT for the audio tokenizer

When PyTorch is available, LM heads can optionally be GPU-accelerated
via ``heads_backend: torch`` in the config.

Set ``low_memory: true`` in the config to enable staged loading for
GPUs with limited VRAM (8-16 GB).

Quick start::

    from voicehub.models.mosstts.source.moss_tts_delay.llama_cpp import LlamaCppPipeline, PipelineConfig

    config = PipelineConfig.from_yaml("configs/llama_cpp/default.yaml")
    with LlamaCppPipeline(config) as pipeline:
        waveform = pipeline.generate(text="Hello, world!")
"""

from .pipeline import LlamaCppPipeline, PipelineConfig

__all__ = ["LlamaCppPipeline", "PipelineConfig"]

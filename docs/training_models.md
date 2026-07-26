# Training model matrix

Every VoiceHub registry entry has a mandatory training profile. Profiles select
the objective family, label aliases, source component paths, loss weights, and
whether the vendored upstream snapshot includes its own training recipe.

| Model             | Training family       | Native recipe in snapshot |
| ----------------- | --------------------- | ------------------------- |
| Orpheus-TTS       | Causal LM             | —                         |
| Dia               | Sequence-to-sequence  | —                         |
| Vui               | Acoustic regression   | —                         |
| Chatterbox        | Flow matching         | —                         |
| Kokoro            | Acoustic regression   | —                         |
| Echo-TTS          | Flow matching         | —                         |
| ConversationTTS   | Causal LM             | —                         |
| LLaSA             | Causal LM             | —                         |
| CosyVoice         | Composite             | `cosyvoice/bin/train.py`   |
| F5-TTS            | Flow matching         | `f5_tts/train/train.py`    |
| GPT-SoVITS        | Composite             | S1 and S2 trainers         |
| MeloTTS           | Acoustic regression   | `melo/train.py`            |
| OpenVoice         | Acoustic regression   | —                         |
| OuteTTS           | Causal LM             | —                         |
| Parler-TTS        | Sequence-to-sequence  | —                         |
| StyleTTS 2        | Composite             | fine-tune trainer          |
| MOSS-TTS          | Causal LM             | SFT recipe                 |
| Qwen3-TTS         | Causal LM             | —                         |
| Irodori-TTS       | Flow matching         | —                         |
| Zonos 1           | Causal LM             | —                         |
| ZONOS2            | Causal LM             | —                         |
| VoxCPM            | Flow matching         | training components        |
| OmniVoice         | Composite             | native trainer             |
| Higgs Audio       | Causal LM             | —                         |
| XTTS v2           | Composite             | GPT trainer                |
| VibeVoice         | Sequence-to-sequence  | —                         |
| Fish Speech S2    | Composite             | `fish_speech/train.py`      |
| Sesame CSM        | Causal LM             | —                         |
| NeuTTS            | Causal LM             | —                         |
| Supertonic        | Acoustic regression   | ONNX inference snapshot    |
| Inflect           | Acoustic regression   | —                         |

“Native recipe” means the exact vendored upstream revision includes an
executable training entry point. A dash does not mean the model is absent from
Trainer: the family adapter trains the differentiable source `forward()` with
portable, preprocessed batch fields.

## Batch boundary

VoiceHub standardizes the boundary between a dataset and an upstream model:

```text
input_ids / attention_mask       token models
input_values                     mel, codec, latent, or waveform inputs
labels                           portable training target
model_inputs                     optional mapping of source-specific tensors
```

The adapter forwards matching fields unchanged. `model_inputs` is useful for a
source model with several conditioning tensors:

```python
sample = {
    "model_inputs": {
        "text_tokens": text_tokens,
        "speaker_embedding": speaker_embedding,
        "noisy_latents": noisy_latents,
        "timesteps": timesteps,
    },
    "labels": clean_latents,
}
```

## Source limitations

An open checkpoint and an inference graph are not automatically a trainable
source model. Supertonic's published VoiceHub snapshot is ONNX-only. Its
profile is discoverable and the Trainer accepts a custom PyTorch training
adapter, but the ONNX session itself cannot receive gradients. VoiceHub reports
that boundary explicitly rather than claiming a successful fine-tune.

For any specialized objective, subclass `BaseTrainingAdapter`, override
`compute_objective()` or `__call__()`, and register it:

```python
AutoTrainingAdapter.register(
    "supertonic",
    SupertonicPyTorchTrainingAdapter,
)
```

This extension replaces only training behavior. The existing inference
adapter, model configuration, source provenance, and `generate()` API remain
unchanged.

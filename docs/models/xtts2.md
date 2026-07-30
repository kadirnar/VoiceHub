# XTTS v2: native DVAE data preparation

VoiceHub owns the XTTS v2 GPT, conditioning stack, HiFi-GAN decoder, and
standalone discrete autoencoder as PyTorch code. The released inference
checkpoint and the fine-tuning tokenizer are intentionally separate:

- `model.pth` contains the GPT, conditioning modules, speaker encoder, and
  vocoder. It does **not** contain DVAE weights.
- `dvae.pth` contains the 53-tensor acoustic autoencoder and codebook.
- `mel_stats.pth` contains the 80 normalization values used before the DVAE.

This split follows Coqui's [pinned XTTS-v2 artifact
tree](https://huggingface.co/coqui/XTTS-v2/tree/6c2b0d75eae4b7047358e3b6bd9325f857d43f77)
and [GPT trainer
source](https://github.com/coqui-ai/TTS/blob/dbf1a08a0d4e47fdad6172e433eeb34bc6b13b4e/TTS/tts/layers/xtts/trainer/gpt_trainer.py).
VoiceHub does not invent or derive DVAE weights from `model.pth`.

## Native graph

`voicehub.architectures.xtts2.dvae.XTTS2DVAE` exposes three stable
boundaries:

```text
22,050 Hz waveform
  -> 80-bin normalized log mel (hop 256)
  -> encoder (two stride-2 stages + residual blocks)
  -> 1,024-entry nearest-neighbour codebook
  -> acoustic target codes (one code per 1,024 waveform samples)

codebook embedding
  -> decoder (residual blocks + two upsample stages)
  -> reconstructed mel
```

The top-level `encoder.*`, `codebook.*`, and `decoder.*` state names exactly
match the separately published `dvae.pth`. GPT fine-tuning freezes this graph
and only calls the encoder and quantizer. The decoder remains available as a
separate autoencoder boundary for validation and codec optimization.

## Safe artifact conversion

Published artifacts are legacy PyTorch pickle containers. Runtime and training
never load them automatically. Review the source files, verify their pinned
digests, and perform the explicit one-time conversion:

```python
from voicehub.architectures.xtts2 import (
    convert_trusted_legacy_xtts2_dvae_checkpoint,
    convert_trusted_legacy_xtts2_mel_stats,
)

convert_trusted_legacy_xtts2_dvae_checkpoint(
    "dvae.pth",
    "dvae.safetensors",
    trust_legacy_pickle=True,
)
convert_trusted_legacy_xtts2_mel_stats(
    "mel_stats.pth",
    "mel_stats.safetensors",
    trust_legacy_pickle=True,
)
```

The converters default to the immutable XTTS-v2 revision and fail unless these
SHA-256 digests agree:

| Artifact | Bytes | SHA-256 |
| --- | ---: | --- |
| `dvae.pth` | 210,514,388 | `b29bc227d410d4991e0a8c09b858f77415013eeb9fba9650258e96095557d97a` |
| `mel_stats.pth` | 1,067 | `1f69422a8a8f344c4fca2f0c6b8d41d2151d6615b7321e48e6bb15ae949b119c` |

Conversion also requires the exact 53-key tensor namespace, expected shapes,
finite floating-point values, and architecture metadata. Native loading
requires `.safetensors` and rejects missing, extra, or shape-incompatible
tensors.

## Fine-tuning from waveform

Point the XTTS configuration at both converted artifacts:

```python
from voicehub.models.xtts_native.configuration_xtts import XTTSConfig

config = XTTSConfig(
    training_dvae_checkpoint="dvae.safetensors",
    training_mel_stats_checkpoint="mel_stats.safetensors",
)
```

A collated GPT batch can then provide `wav` or `audio_values` instead of
`audio_codes`:

```python
batch = {
    "text_inputs": text_token_ids,       # [batch, text]
    "text_lengths": text_lengths,        # [batch]
    "wav": waveform,                     # [batch, time] or [batch, 1, time]
    "wav_lengths": waveform_lengths,     # optional; derived for unpadded audio
    "cond_mels": conditioning_mels,      # or cond_latents
}
```

The training adapter computes `audio_codes` under `torch.no_grad()`, removes
the waveform from the GPT inputs, and keeps the GPT as the only trainable
component. Existing records with precomputed `audio_codes` are unchanged and
avoid repeated DVAE work.

For reproducible large training runs, offline precomputation remains the
highest-throughput option. The integrated waveform path is useful for dataset
creation, validation, and smaller fine-tuning jobs, and it can be compiled at
the encoder boundary without changing the inference model checkpoint.

# Training model matrix

This table describes what the **current VoiceHub integration** can fine-tune.
It does not describe whether a model was trainable in its original research
repository.

| Status | Meaning |
| --- | --- |
| **End-to-end / raw data** | VoiceHub provides the model-specific objective, dataset/collator, and text/audio or conversation preprocessing needed to start from ordinary training records. |
| **Preprocessed recipe** | VoiceHub provides the verified differentiable objective and training route, but the caller must supply model-ready tokens, codec codes, features, or source-shaped batches. |
| **Specialized / partial** | A component/checkpoint variant is covered, or an explicitly experimental reconstructed objective is available, but the complete author-verified family recipe is not. |
| **Unavailable** | The integrated runtime has no verified training graph, or the declared custom recipe is intentionally blocked until its model-specific adapter is implemented. |

`native` and `preprocessed` in the Python training specification are capability
checks used by the trainer. The table uses the more practical categories above:
a native scalar loss is still a **preprocessed recipe** when VoiceHub does not
provide raw-data preparation.

## Model-by-model support

| Model (`model_type`) | Status | Fine-tuning boundary |
| --- | --- | --- |
| Orpheus-TTS (`orpheustts`) | **End-to-end / raw data** | Text plus 24 kHz audio, or precomputed SNAC codes, is framed as the author-style causal codec sequence. SNAC remains frozen; train the unquantized language model. |
| Dia (`dia`) | **End-to-end / raw data** | Uses the official Transformers `DiaForConditionalGeneration` loss and `DiaProcessor` labels from text/audio records. Only the converted `Dia-1.6B-0626`-style Transformers checkpoint is trainable; the original Nari runtime is inference-only. |
| Vui (`vui`) | **Unavailable** | The integrated generation runtime has no verified current 300M training objective or target schema. |
| Chatterbox (`chatterbox`) | **Unavailable** | T3 and S3Gen flow require distinct source-native objectives. The declared custom recipe is gated because the inference graph is not a faithful trainer for either phase. |
| Kokoro (`kokoro`) | **Unavailable** | The released inference path does not expose the alignment, duration, and acoustic training graph. |
| Echo-TTS (`echo`) | **Unavailable** | No preserved and verified flow-training graph is integrated; the serving checkpoint path is inference-oriented. |
| ConversationTTS (`conversationtts`) | **Preprocessed recipe** | Implements the exact codebook-zero and residual masked cross-entropies. Batches must provide `tokens`, `labels`, and `tokens_mask`. |
| LLaSA (`llasa`) | **End-to-end / raw data** | Text plus audio, or precomputed XCodec2 codes, is converted to completion-only codec-LM labels. XCodec2 is frozen. |
| CosyVoice (`cosyvoice`) | **Specialized / partial** | One source-native component is trained per run. LLM and flow objectives are integrated for source-shaped batches; HiFT/HiFi-GAN still requires the upstream training-only generator, discriminator, and mel graph. JIT, TensorRT, and vLLM serving paths are rejected. |
| F5-TTS (`f5tts`) | **Preprocessed recipe** | Implements the source conditional-flow-matching call, optimizer route, EMA lifecycle, and EMA safetensors export. The caller supplies the source-shaped audio/mel, text, and optional length batch. |
| GPT-SoVITS (`gptsovits`) | **Unavailable** | The custom S1 semantic and S2 VITS/GAN phases need their stage-specific data graphs, discriminator, losses, and optimizer cadence. |
| MeloTTS (`melotts`) | **Unavailable** | The complete VITS recipe needs generator, waveform and duration discriminators, KL/duration/mel/adversarial losses, and feature matching; the inference synthesizer is insufficient. |
| OpenVoice (`openvoice`) | **Unavailable** | The released conversion/acoustic graph omits the verified training objective and training-only checkpoint topology. |
| OuteTTS (`outetts`) | **End-to-end / raw data** | Source prompts and codec tokens are built from text/audio or a prepared speaker record. Fine-tuning requires the unquantized HF backend; llama.cpp/GGUF and 4-bit/8-bit generic training are rejected. |
| Parler-TTS (`parlertts`) | **Preprocessed recipe** | The Transformers model supplies its teacher-forced loss. Description/prompt tokenization, codec targets, masks, and audio preprocessing remain dataset responsibilities. |
| StyleTTS 2 (`styletts2`) | **Unavailable** | The custom multi-module diffusion/GAN recipe is recorded but not implemented by a valid adapter. MPD, MSD, WD, diffusion, style, duration, and acoustic updates must remain source-faithful. |
| MOSS-TTS (`mosstts`) | **Preprocessed recipe** | Delay, Local, and Realtime use their native LM losses. Local v1.5 uses the integrated channel-wise text/audio objective and source dataset, but target `audio_codes` must be prepared first. |
| Qwen3-TTS (`qwen3tts`) | **Preprocessed recipe** | Implements the official 12 Hz Base single-speaker SFT objective for talker and code predictor. The registered training default is `Qwen/Qwen3-TTS-12Hz-1.7B-Base`, distinct from the inference-oriented default. Records require text, 16-codebook target `audio_codes`, and 24 kHz reference audio. CustomVoice and VoiceDesign are export/inference targets, not SFT starting checkpoints. |
| Irodori-TTS (`irodoritts`) | **Specialized / partial** | The flow objective is available for model-ready velocity targets and conditioning. Duration-model training is a separate, unimplemented objective. |
| Zonos 1 (`zonos`) | **Unavailable** | No verified causal multi-codebook training objective, delay layout, and conditioning recipe is integrated. |
| ZONOS2 (`zonos2`) | **Unavailable** | The fused raw-tensor generation engine is not a differentiable PyTorch training graph. |
| VoxCPM (`voxcpm`) | **Preprocessed recipe** | The training-safe runtime exposes the source diffusion and stop losses, disables inference optimization, and freezes the AudioVAE according to source policy. Batch construction remains VoxCPM-specific. |
| OmniVoice (`omnivoice`) | **Preprocessed recipe** | The direct training model supplies its masked text/audio objective. Callers provide codebook-first/time-last inputs and labels; VoiceHub owns their padding schema. |
| Higgs Audio (`higgstts`) | **Specialized / partial** | ChatML text/audio records use the vendored preparation and collator with a frozen audio tokenizer. VoiceHub implements token-normalized causal text and per-codebook audio losses, but Boson AI has not published an author-verified fine-tuning loop, so this reconstructed recipe is experimental. |
| XTTS v2 (`xtts`) | **Specialized / partial** | The author-supported GPT fine-tune is integrated from raw XTTS metadata/audio, including frozen DVAE preprocessing, weighted text/mel-code losses, optimizer, and scheduler. DVAE and vocoder/adversarial training are not supported. |
| VibeVoice (`vibevoice`) | **Unavailable** | The integrated realtime 0.5B runtime raises from training `forward`; the separate non-streaming community recipe is not wired as a verified VoiceHub path. |
| Fish Speech S2 (`fishtts`) | **Preprocessed recipe** | The semantic transformer supports the exact base and residual codebook losses from Fish protobuf data or pretokenized channel-first records. The codec is an offline frozen tokenizer, not a trainable phase. |
| Sesame CSM (`csm`) | **End-to-end / raw data** | Uses the official Transformers CSM model, processor-generated audio-frame labels, and native backbone/depth-decoder loss from conversation or text/audio records. Mimi is frozen. |
| NeuTTS (`neutts`) | **End-to-end / raw data** | Text plus audio, or precomputed NeuCodec codes, is converted to completion-only labels for the HF backbone. GGUF backbones are inference-only; an attached frozen ONNX decoder does not block HF-backbone training. |
| Supertonic (`supertonic`) | **Unavailable** | The published integration is ONNX-only and cannot receive gradients. |
| Inflect (`inflecttts`) | **Unavailable** | The released artifact is inference-only and omits the posterior/discriminator state required for a complete VITS recipe. |

## Safetensors, GGUF, and resume semantics

A file format is not a training capability:

- **Safetensors** can be a safe weight warm start when it belongs to the
  unfused PyTorch/Transformers graph expected by the adapter. It does not by
  itself contain optimizer, scheduler, scaler, sampler, RNG, or recipe state.
- A **VoiceHub checkpoint** is the exact-resume artifact. It stores model
  components together with optimizer/scheduler and trainer runtime state.
  Model-specific adapters may also write `native_export/`. Its recipe manifest
  states whether that directory is a complete inference export or only a
  component weight warm start; a safetensors filename alone is not evidence
  that the full upstream loader topology is present.
- **GGUF**, ONNX, JIT, TensorRT, vLLM, and other fused or serving-only artifacts
  are not generic gradient-bearing checkpoints. If a repository publishes both
  safetensors and GGUF, select the compatible unquantized safetensors/source
  checkpoint for fine-tuning. The base adapter rejects recognizable serving,
  compiled, and quantized artifacts before loading and validates the resolved
  training graph again afterward; family-specific loaders add stricter checks.
- Quantized adapter training is a separate capability. It requires a
  PEFT/LoRA-aware adapter and must not be inferred from support for full-precision
  fine-tuning.

## Before a run

The [training workflow](../guides/training.md) and
[companion notebook](../guides/notebook.md)
demonstrate these checks with Dia's raw text/audio training path.

Verify that:

1. the selected checkpoint variant uses the differentiable backend named in
   the table;
2. one small batch returns a finite scalar loss with `requires_grad=True`;
3. the intended parameters receive gradients and frozen codecs/vocoders do not;
4. every adversarial or auxiliary phase has its source training graph, loss,
   detach boundary, optimizer, and update cadence;
5. saving and resuming restores the same component and optimizer topology.

See the [trainer architecture](../concepts/trainer.md) for phase scheduling,
data collation, optimizer routing, strategy integration, and checkpoint
semantics.

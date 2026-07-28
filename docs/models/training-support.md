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
| Vui (`vui`) | **Preprocessed recipe** | Implements delayed-codebook, teacher-forced causal cross-entropy for Fluac codec IDs. Callers provide text token IDs, codec codes, and optional sequence masks/lengths; the codec stays frozen. |
| Chatterbox (`chatterbox`) | **Unavailable** | T3 and S3Gen flow require distinct source-native objectives. The declared custom recipe is gated because the inference graph is not a faithful trainer for either phase. |
| Kokoro (`kokoro`) | **Unavailable** | The released inference path does not expose the alignment, duration, and acoustic training graph. |
| Echo-TTS (`echo`) | **Preprocessed recipe** | Implements the released rectified-flow velocity objective over target/noise latents with text and speaker conditioning. Callers provide source-shaped codec latents and masks; the Fish codec remains frozen. |
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
| Zonos 1 (`zonos`) | **Preprocessed recipe** | Implements the released delay pattern and teacher-forced multi-codebook causal cross-entropy. Callers provide prefix conditioning and DAC code tensors; the autoencoder remains frozen. |
| ZONOS2 (`zonos2`) | **Unavailable** | The fused raw-tensor generation engine is not a differentiable PyTorch training graph. |
| VoxCPM (`voxcpm`) | **Preprocessed recipe** | The training-safe runtime exposes the source diffusion and stop losses, disables inference optimization, and freezes the AudioVAE according to source policy. Batch construction remains VoxCPM-specific. |
| OmniVoice (`omnivoice`) | **Preprocessed recipe** | The direct training model supplies its masked text/audio objective. Callers provide codebook-first/time-last inputs and labels; VoiceHub owns their padding schema. |
| Higgs Audio (`higgstts`) | **Specialized / partial** | ChatML text/audio records use the vendored preparation and collator with a frozen audio tokenizer. VoiceHub implements token-normalized causal text and per-codebook audio losses, but Boson AI has not published an author-verified fine-tuning loop, so this reconstructed recipe is experimental. |
| XTTS v2 (`xtts`) | **Specialized / partial** | The author-supported GPT fine-tune is integrated from raw XTTS metadata/audio, including frozen DVAE preprocessing, weighted text/mel-code losses, optimizer, and scheduler. DVAE and vocoder/adversarial training are not supported. |
| VibeVoice (`vibevoice`) | **Preprocessed recipe** | The non-streaming `microsoft/VibeVoice-1.5B` checkpoint uses its native masked language-model plus diffusion graph and frozen acoustic/semantic tokenizers. Callers provide the source recipe's token, speech-latent, and acoustic-mask tensors. The default realtime 0.5B checkpoint remains inference-only. |
| Fish Speech S2 (`fishtts`) | **Preprocessed recipe** | The semantic transformer supports the exact base and residual codebook losses from Fish protobuf data or pretokenized channel-first records. The codec is an offline frozen tokenizer, not a trainable phase. |
| Sesame CSM (`csm`) | **End-to-end / raw data** | Uses the official Transformers CSM model, processor-generated audio-frame labels, and native backbone/depth-decoder loss from conversation or text/audio records. Mimi is frozen. |
| NeuTTS (`neutts`) | **End-to-end / raw data** | Text plus audio, or precomputed NeuCodec codes, is converted to completion-only labels for the HF backbone. GGUF backbones are inference-only; an attached frozen ONNX decoder does not block HF-backbone training. |
| Supertonic (`supertonic`) | **Unavailable** | The published integration is ONNX-only and cannot receive gradients. |
| Inflect (`inflecttts`) | **Unavailable** | The released artifact is inference-only and omits the posterior/discriminator state required for a complete VITS recipe. |
| Bark (`bark`) | **Preprocessed recipe** | VoiceHub computes stage-aligned cross-entropy from the semantic, coarse-acoustic, and fine-acoustic Transformers submodel logits. The caller must provide stage-aligned token IDs, labels, masks, and the fine-codebook index; audio tokenization remains an offline dataset step, and this is not end-to-end raw-audio fine-tuning. |
| SpeechT5 (`speecht5`) | **End-to-end / raw data** | Uses the native Transformers spectrogram and stop-token losses. Text/audio records are processed into labels and an optional speaker embedding; the HiFi-GAN vocoder remains frozen. |
| VITS / MMS-TTS (`vits`) | **Specialized / partial** | The shared Trainer can run VoiceHub's waveform-only reconstruction experiment when `enable_experimental_reconstruction_training=True`. This explicit opt-in is not full VITS fine-tuning: Transformers does not expose the source posterior, duration, KL, discriminator, feature-matching, or adversarial recipe. |
| Transformers ASR (`asr_transformers`) | **End-to-end / raw data** | Dynamically dispatches compatible conventional CTC, speech sequence-to-sequence, RNN-T, or TDT checkpoints. Prompted audio-language models must use a dedicated model type so their native request and labels are preserved. |
| Whisper ASR (`asr_whisper`) | **End-to-end / raw data** | Defaults to Whisper large-v3-turbo and uses the native processor plus teacher-forced sequence-to-sequence loss. |
| Tiron (`asr_tiron`) | **End-to-end / raw data** | Fine-tunes the differentiable Whisper checkpoint while preserving its added speaker/timestamp vocabulary. Whole-meeting chunking and cross-window speaker linking are inference orchestration, not training phases. |
| Qwen3-ASR (`asr_qwen3`) | **End-to-end / raw data** | Builds the native audio user turn and assistant target, including the validated language prefix and `<asr_text>` boundary. VoiceHub derives completion-only vocabulary labels from the rendered assistant tokens and masks prompt, audio, and padding positions, including on cached processor batches. |
| VibeVoice-ASR-HF (`asr_vibevoice`) | **End-to-end / raw data** | Uses the checkpoint's multimodal chat template and processor-generated labels. Fine-tuning requires the unquantized Safetensors model; BitNet/GGML serving artifacts are rejected. |
| Granite Speech 4.1 (`asr_granite_speech`) | **End-to-end / raw data** | Implements IBM's published supervised collator: the tokenizer renders the configured `<\|audio\|>` instruction, the processor builds acoustic inputs, transcript plus EOS tokens are appended, and prompt/padding labels are masked with `-100`. The native causal objective remains responsible for shifting labels. |
| Parakeet TDT v3 (`asr_parakeet_tdt`) | **End-to-end / raw data** | Processes audio and transcript jointly so native labels and decoder inputs reach the token-and-duration transducer loss. |
| Nemotron 3.5 ASR (`asr_nemotron`) | **End-to-end / raw data** | Joint processing retains transcript labels, decoder inputs, language prompt IDs, and lookahead controls. VoiceHub installs the native RNN-T loss explicitly and normalizes the released processor's blank prefix to the model vocabulary. |
| Cohere Transcribe (`asr_cohere`) | **End-to-end / raw data** | Routes language and punctuation through the processor, combines its decoder prompt with teacher-forced transcript tokens, masks prompt/padding targets, and applies unshifted token cross-entropy. Long recordings must be pre-segmented with one aligned transcript per segment because the inference-only chunk reassembler does not create chunk-level training labels. |
| MedASR (`asr_medasr`) | **End-to-end / raw data** | Uses the joint LASR audio/text processor and native CTC loss. Checkpoint access and use remain governed by Google's Health AI Developer Foundations terms. |
| Wav2Vec2 ASR (`asr_wav2vec2`) | **End-to-end / raw data** | Uses the native Transformers CTC graph and processor with correctly padded labels. |
| HuBERT ASR (`asr_hubert`) | **End-to-end / raw data** | Uses the native Transformers CTC graph and processor with correctly padded labels. |
| WavLM ASR (`asr_wavlm`) | **End-to-end / raw data** | Uses the native Transformers CTC graph and processor with correctly padded labels. |
| Moonshine ASR (`asr_moonshine`) | **End-to-end / raw data** | Uses the native teacher-forced speech sequence-to-sequence loss and processor. |
| SeamlessM4T v2 ASR (`asr_seamless_m4t_v2`) | **End-to-end / raw data** | Uses the native multilingual speech sequence-to-sequence loss, processor, and explicit target-language contract. |
| Auditok VAD (`vad_auditok`) | **Unavailable** | Deterministic energy detector with no trainable graph. |
| Sherpa-ONNX VAD (`vad_sherpa_onnx`) | **Unavailable** | Silero and TEN artifacts are loaded as ONNX inference graphs; fine-tune their source checkpoints before export. |
| pyannote segmentation (`vad_pyannote_segmentation`) | **Specialized / partial** | Inference is normalized by VoiceHub; fine-tuning remains owned by pyannote's task, protocol, data, and trainer stack. |
| pyannote Brouhaha (`vad_pyannote_brouhaha`) | **Specialized / partial** | The multi-task speech/SNR/C50 objective and data recipe remain owned by pyannote upstream. |

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

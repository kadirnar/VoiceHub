# ConversationTTS source status

The upstream repository
[`Audio-Foundation-Models/ConversationTTS`](https://github.com/Audio-Foundation-Models/ConversationTTS)
declares the source, checkpoints, datasets, and evaluation tools under
CC BY-NC 4.0 at revision `b3851f7`.

VoiceHub vendors the executable model, inference, text-tokenizer, and
MimiCodec runtime source. Checkpoint weights remain external. Commercial use
is not granted by CC BY-NC 4.0; this restriction is exposed through
`get_model_spec("conversationtts").license`.

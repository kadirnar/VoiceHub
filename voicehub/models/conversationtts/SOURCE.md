# ConversationTTS source status

The upstream repository
[`Audio-Foundation-Models/ConversationTTS`](https://github.com/Audio-Foundation-Models/ConversationTTS)
did not contain a source-code license when VoiceHub audited revision
`b3851f7`.

VoiceHub therefore registers the architecture and its configuration, but does
not copy or execute the upstream implementation. Calling `load()` raises
`SourceLicenseError`. The integration can be completed as soon as upstream
publishes a redistribution-compatible license.

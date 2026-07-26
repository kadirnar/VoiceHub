from transformers import PretrainedConfig

class BigCodecConfig(PretrainedConfig):
    model_type = "bigcodec"

    def __init__(
        self,
        # 下面这些只是示例超参
        semantic_hidden_size=1024,
        codec_encoder_hidden_size=1024,
        codec_decoder_hidden_size=1024,
        use_vocos=True,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.semantic_hidden_size = semantic_hidden_size
        self.codec_encoder_hidden_size = codec_encoder_hidden_size
        self.codec_decoder_hidden_size = codec_decoder_hidden_size
        self.use_vocos = use_vocos

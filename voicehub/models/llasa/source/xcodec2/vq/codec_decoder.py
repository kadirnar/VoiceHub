import sys
 
import numpy as np
import torch
import torch.nn as nn
from voicehub.models.llasa.source.xcodec2.vq.residual_vq import ResidualVQ
from voicehub.models.llasa.source.xcodec2.vq.module import WNConv1d, DecoderBlock, ResLSTM
from voicehub.models.llasa.source.xcodec2.vq.alias_free_torch import *
from voicehub.models.llasa.source.xcodec2.vq import activations
import voicehub.models.llasa.source.xcodec2.vq.blocks as blocks
from torch.nn import utils

from voicehub.models.llasa.source.xcodec2.vq.bs_roformer5 import TransformerBlock
 
from torchtune.modules import RotaryPositionalEmbeddings

def init_weights(m):
    if isinstance(m, nn.Conv1d):
        nn.init.trunc_normal_(m.weight, std=0.02)
        nn.init.constant_(m.bias, 0)

class CodecDecoder(nn.Module):
    def __init__(self,
                 in_channels=1024,
                 upsample_initial_channel=1536,
                 ngf=48,
                 use_rnn=True,
                 rnn_bidirectional=False,
                 rnn_num_layers=2,
                 up_ratios=(5, 4, 4, 4, 2),
                 dilations=(1, 3, 9),
                 vq_num_quantizers=1,
                 vq_dim=2048,
                 vq_commit_weight=0.25,
                 vq_weight_init=False,
                 vq_full_commit_loss=False,
                 codebook_size=16384,
                 codebook_dim=32,
                ):
        super().__init__()
        self.hop_length = np.prod(up_ratios)
        self.ngf = ngf
        self.up_ratios = up_ratios
        
        self.quantizer = ResidualVQ(
            num_quantizers=vq_num_quantizers,
            dim=vq_dim, # double the dim for acousitc and semantic
            codebook_size=codebook_size,
            codebook_dim=codebook_dim,
            threshold_ema_dead_code=2,
            commitment=vq_commit_weight,
            weight_init=vq_weight_init,
            full_commit_loss=vq_full_commit_loss,
        )
        channels = upsample_initial_channel
        layers = [WNConv1d(in_channels, channels, kernel_size=7, padding=3)]
        
        if use_rnn:
            layers += [
                ResLSTM(channels,
                        num_layers=rnn_num_layers,
                        bidirectional=rnn_bidirectional
                    )
            ]
        
        for i, stride in enumerate(up_ratios):
            input_dim = channels // 2**i
            output_dim = channels // 2 ** (i + 1)
            layers += [DecoderBlock(input_dim, output_dim, stride, dilations)]

        layers += [
            Activation1d(activation=activations.SnakeBeta(output_dim, alpha_logscale=True)),
            WNConv1d(output_dim, 1, kernel_size=7, padding=3),
            nn.Tanh(),
        ]

        self.model = nn.Sequential(*layers)
        
        self.reset_parameters()

    def forward(self, x, vq=True):
        if vq is True:
            x, q, commit_loss = self.quantizer(x)
            return x, q, commit_loss
        x = self.model(x)
        return x

    def vq2emb(self, vq):
        self.quantizer = self.quantizer.eval()
        x = self.quantizer.vq2emb(vq)
        return x

    def get_emb(self):
        self.quantizer = self.quantizer.eval()
        embs = self.quantizer.get_emb()
        return embs

    def inference_vq(self, vq):
        x = vq[None,:,:]
        x = self.model(x)
        return x

    def inference_0(self, x):
        x, q, loss, perp = self.quantizer(x)
        x = self.model(x)
        return x, None
    
    def inference(self, x):
        x = self.model(x)
        return x, None


    def remove_weight_norm(self):
        """Remove weight normalization module from all of the layers."""

        def _remove_weight_norm(m):
            try:
                torch.nn.utils.remove_weight_norm(m)
            except ValueError:  # this module didn't have weight norm
                return

        self.apply(_remove_weight_norm)

    def apply_weight_norm(self):
        """Apply weight normalization module from all of the layers."""

        def _apply_weight_norm(m):
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.ConvTranspose1d):
                torch.nn.utils.weight_norm(m)

        self.apply(_apply_weight_norm)

    def reset_parameters(self):
        self.apply(init_weights)


class CodecDecoder_oobleck_Transformer(nn.Module):
    def __init__(self,
                ngf=32,
                up_ratios=(5, 4, 4, 4, 2),
                dilations=(1, 3, 9),
                vq_num_quantizers=1,
                vq_dim=1024,
                vq_commit_weight=0.25,
                vq_weight_init=False,
                vq_full_commit_loss=False,
                codebook_size=16384,
                codebook_dim=16,
                hidden_dim=1024,
                depth=12,
                heads=16,
                pos_meb_dim=64,
                ):
        super().__init__()
        self.hop_length = np.prod(up_ratios)
        self.capacity = ngf
        self.up_ratios = up_ratios
        self.hidden_dim = hidden_dim
        self.quantizer = ResidualVQ(
            num_quantizers=vq_num_quantizers,
            dim=vq_dim, # double the dim for acousitc and semantic
            codebook_size=codebook_size,
            codebook_dim=codebook_dim,
            threshold_ema_dead_code=2,
            commitment=vq_commit_weight,
            weight_init=vq_weight_init,
            full_commit_loss=vq_full_commit_loss,
        )

        time_rotary_embed = RotaryPositionalEmbeddings(dim=pos_meb_dim)
         
        transformer_blocks = [
            TransformerBlock(dim=hidden_dim, n_heads=heads, rotary_embed=time_rotary_embed)
            for _ in range(depth)
        ]        
 
        self.transformers = nn.Sequential(*transformer_blocks)

        self.final_layer_norm = nn.LayerNorm(hidden_dim, eps=1e-6)         
         
        self.conv_blocks = blocks.DilatedResidualDecoder(
            capacity=self.capacity,
            dilated_unit=self.dilated_unit,
            upsampling_unit=self.upsampling_unit,
            ratios=up_ratios,  # 逆转编码器的下采样比率
            dilations=dilations,
            pre_network_conv=self.pre_conv,
            post_network_conv=self.post_conv,
            )
            
 
        
        self.reset_parameters()

    def forward(self, x, vq=True):
        if vq is True:
            x, q, commit_loss = self.quantizer(x)
            return x, q, commit_loss
        x= self.transformers(x)
        x = self.final_layer_norm(x)
        x = x.permute(0, 2, 1)
        x = self.conv_blocks(x)
        return x

    def vq2emb(self, vq):
        self.quantizer = self.quantizer.eval()
        x = self.quantizer.vq2emb(vq)
        return x

    def get_emb(self):
        self.quantizer = self.quantizer.eval()
        embs = self.quantizer.get_emb()
        return embs

    def inference_vq(self, vq):
        x = vq[None,:,:]
        x = self.model(x)
        return x

    def inference_0(self, x):
        x, q, loss, perp = self.quantizer(x)
        x = self.model(x)
        return x, None
    
    def inference(self, x):
        x = self.model(x)
        return x, None


    def remove_weight_norm(self):
        """Remove weight normalization module from all of the layers."""

        def _remove_weight_norm(m):
            try:
                torch.nn.utils.remove_weight_norm(m)
            except ValueError:  # this module didn't have weight norm
                return

        self.apply(_remove_weight_norm)

    def apply_weight_norm(self):
        """Apply weight normalization module from all of the layers."""

        def _apply_weight_norm(m):
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.ConvTranspose1d):
                torch.nn.utils.weight_norm(m)

        self.apply(_apply_weight_norm)

    def reset_parameters(self):
        self.apply(init_weights)

    def pre_conv(self, out_channels):
        return nn.Conv1d(in_channels=self.hidden_dim, out_channels=out_channels, kernel_size=1)

    # 定义后处理卷积层，将模型的输出映射到最终的输出通道数
    def post_conv(self,in_channels):
        return nn.Conv1d(in_channels=in_channels, out_channels=1, kernel_size=1)

    def dilated_unit(self, hidden_dim, dilation):
        return blocks.DilatedConvolutionalUnit(
            hidden_dim=hidden_dim,
            dilation=dilation,
            kernel_size=3,
            activation=nn.ReLU ,
            normalization=utils.weight_norm
        )

    # 定义上采样单元
    def upsampling_unit(self,input_dim, output_dim, stride):
        return blocks.UpsamplingUnit(
            input_dim=input_dim,
            output_dim=output_dim,
            stride=stride,
            activation=nn.ReLU ,
            normalization=utils.weight_norm
        )

def main():
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 初始化模型
    model = CodecDecoder_oobleck_Transformer().to(device)
    print("Model initialized.")

    # 创建测试输入: batch_size x in_channels x sequence_length
    batch_size = 2
    in_channels = 1024
    sequence_length = 100  # 示例长度，可以根据需要调整
    dummy_input = torch.randn(batch_size, sequence_length, in_channels).to(device)
    print(f"Dummy input shape: {dummy_input.shape}")

    # 将模型设为评估模式
    model.eval()

 
 
    output_no_vq = model(dummy_input, vq=False)
    c=1
 
if __name__ == "__main__":
    main()
import sys
 
import torch
from torch import nn
import numpy as np
from voicehub.models.llasa.source.xcodec2.vq.module import WNConv1d, EncoderBlock, ResLSTM
from voicehub.models.llasa.source.xcodec2.vq.alias_free_torch import *
from voicehub.models.llasa.source.xcodec2.vq import activations
from voicehub.models.llasa.source.xcodec2.vq.bs_roformer5 import TransformerBlock
 
from torchtune.modules import RotaryPositionalEmbeddings
import voicehub.models.llasa.source.xcodec2.vq.blocks as blocks
from torch.nn import utils
def init_weights(m):
    if isinstance(m, nn.Conv1d):
        nn.init.trunc_normal_(m.weight, std=0.02)
        nn.init.constant_(m.bias, 0)

class CodecEncoder(nn.Module):
    def __init__(self,
                ngf=48,
                use_rnn=True,
                rnn_bidirectional=False,
                rnn_num_layers=2,
                up_ratios=(2, 2, 4, 4, 5),
                dilations=(1, 3, 9),
                out_channels=1024):
        super().__init__()
        self.hop_length = np.prod(up_ratios)
        self.ngf = ngf
        self.up_ratios = up_ratios

        # Create first convolution
        d_model = ngf
        self.block = [WNConv1d(1, d_model, kernel_size=7, padding=3)]

        # Create EncoderBlocks that double channels as they downsample by `stride`
        for i, stride in enumerate(up_ratios):
            d_model *= 2
            self.block += [EncoderBlock(d_model, stride=stride, dilations=dilations)]
        # RNN
        if use_rnn:
            self.block += [
                ResLSTM(d_model,
                        num_layers=rnn_num_layers,
                        bidirectional=rnn_bidirectional
                    )
            ]
        # Create last convolution
        self.block += [
            Activation1d(activation=activations.SnakeBeta(d_model, alpha_logscale=True)),
            WNConv1d(d_model, out_channels, kernel_size=3, padding=1),
        ]

        # Wrap black into nn.Sequential
        self.block = nn.Sequential(*self.block)
        self.enc_dim = d_model
        
        self.reset_parameters()

    def forward(self, x):
        out = self.block(x)
        return out

    def inference(self, x):
        return self.block(x)

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
            if isinstance(m, nn.Conv1d):
                torch.nn.utils.weight_norm(m)

        self.apply(_apply_weight_norm)

    def reset_parameters(self):
        self.apply(init_weights)


class Transpose(nn.Module):
    def __init__(self, dim1, dim2):
        super(Transpose, self).__init__()
        self.dim1 = dim1
        self.dim2 = dim2

    def forward(self, x):
        return x.transpose(self.dim1, self.dim2)

class CodecEncoder_Transformer(nn.Module):
    def __init__(self,
                ngf=48,
                up_ratios=[2, 2, 4, 4, 5],
                dilations=(1, 3, 9),
                hidden_dim=1024,
                depth=12,
                heads=12,
                pos_meb_dim=64,
                ):
        super().__init__()
        self.hop_length = np.prod(up_ratios)
        self.ngf =ngf
        self.up_ratios = up_ratios
 
        d_model = ngf
        self.conv_blocks = [WNConv1d(1, d_model, kernel_size=7, padding=3)]

 
        for i, stride in enumerate(up_ratios): 
            d_model *= 2 
            self.conv_blocks += [EncoderBlock(d_model, stride=stride, dilations=dilations)]
 
        self.conv_blocks = nn.Sequential(*self.conv_blocks)
 
        
        # time_rotary_embed = RotaryPositionalEmbeddings(dim=pos_meb_dim)
        
 
        # transformer_blocks = [
        #     TransformerBlock(dim=hidden_dim, n_heads=heads, rotary_embed=time_rotary_embed)
        #     for _ in range(depth)
        # ]
        
 
        # self.transformers = nn.Sequential(*transformer_blocks)

        # self.final_layer_norm = nn.LayerNorm(hidden_dim, eps=1e-6)

        self.conv_final_block  = [
        Activation1d(activation=activations.SnakeBeta(d_model, alpha_logscale=True)),
        WNConv1d(d_model, hidden_dim, kernel_size=3, padding=1),
        ]
        self.conv_final_block = nn.Sequential(*self.conv_final_block)
        
        self.reset_parameters()

    def forward(self, x):
        x = self.conv_blocks(x)
        # x = x.permute(0, 2, 1)
        # x= self.transformers(x)
        # x = self.final_layer_norm(x)
        # x = x.permute(0, 2, 1)
        x =  self.conv_final_block (x)
        x = x.permute(0, 2, 1)
        return x

    def inference(self, x):
        return self.block(x)

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
            if isinstance(m, nn.Conv1d):
                torch.nn.utils.weight_norm(m)

        self.apply(_apply_weight_norm)

    def reset_parameters(self):
        self.apply(init_weights)



class Codec_oobleck_Transformer(nn.Module):
    def __init__(self,
                ngf=32,
                up_ratios=(2, 2,4,4, 5),
                dilations=(1, 3, 9),
                hidden_dim=1024,
                depth=12,
                heads=16,
                pos_meb_dim=64,
                ):
        super().__init__()
        self.hop_length = np.prod(up_ratios)
        self.ngf =ngf
        self.up_ratios = up_ratios
        self.hidden_dim = hidden_dim
         

        self.conv_blocks =  blocks.DilatedResidualEncoder(
        capacity=ngf,
        dilated_unit=self.dilated_unit,
        downsampling_unit=self.downsampling_unit,
        ratios=up_ratios,
        dilations=dilations,
        pre_network_conv=self.pre_conv,
        post_network_conv=self.post_conv,
    )
 
        
        time_rotary_embed = RotaryPositionalEmbeddings(dim=pos_meb_dim)
         
        transformer_blocks = [
            TransformerBlock(dim=hidden_dim, n_heads=heads, rotary_embed=time_rotary_embed)
            for _ in range(depth)
        ]        
 
        self.transformers = nn.Sequential(*transformer_blocks)

        self.final_layer_norm = nn.LayerNorm(hidden_dim, eps=1e-6)
 
        
        self.reset_parameters()

    def forward(self, x):
        x = self.conv_blocks(x)
        x = x.permute(0, 2, 1)
        x= self.transformers(x)
        x = self.final_layer_norm(x)
        return x

    def inference(self, x):
        return self.block(x)

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
            if isinstance(m, nn.Conv1d):
                torch.nn.utils.weight_norm(m)

        self.apply(_apply_weight_norm)

    def reset_parameters(self):
        self.apply(init_weights)

    def dilated_unit(self,hidden_dim, dilation):
        return blocks.DilatedConvolutionalUnit(hidden_dim,
                                               dilation,
                                               kernel_size=3,
                                               activation=nn.ReLU,
                                               normalization=utils.weight_norm)

    def downsampling_unit(self, input_dim: int, output_dim: int, stride: int):
        return blocks.DownsamplingUnit(input_dim,
                                       output_dim,
                                       stride,
                                       nn.ReLU,
                                       normalization=utils.weight_norm)

    def pre_conv(self,out_channels):
        return nn.Conv1d(1, out_channels, 1)

    def post_conv(self,in_channels):
        return nn.Conv1d(in_channels, self.hidden_dim, 1)





class CodecEncoder_only_Transformer(nn.Module):
    def __init__(self,hidden_dim=1024,depth=12,heads=16,pos_meb_dim=64):
        super().__init__()
        # self.embed = nn.Linear(input_dim, hidden_dim )input_dim=300,

        depth = depth
        time_rotary_embed = RotaryPositionalEmbeddings(dim=pos_meb_dim)
        
 
        transformer_blocks = [
            TransformerBlock(dim=hidden_dim, n_heads=heads, rotary_embed=time_rotary_embed)
            for _ in range(depth)
        ]
        
 
        self.transformers = nn.Sequential(*transformer_blocks)

        self.final_layer_norm = nn.LayerNorm(hidden_dim, eps=1e-6)

    def forward(self, x: torch.Tensor ) -> torch.Tensor:
        # x = self.embed(x)
 
 
        x= self.transformers(x)
        x = self.final_layer_norm(x)
 
        return x




 


def get_model_size(model):
    # 计算总参数数
    total_params = sum(p.numel() for p in model.parameters())
    
    # 假设每个参数都是32位浮点数，计算模型大小（以字节为单位）
    model_size_bytes = total_params    # 每个参数4字节
    
    # 转换为更易读的单位（例如，MB）
    model_size_mb = model_size_bytes / (1024 ** 2)
    
    return total_params, model_size_mb

if __name__ == '__main__':
    model = Codec_oobleck_Transformer()
    x = torch.randn(1, 1, 16000)  # example input tensor
    output = model(x)
    print("Output shape:", output.shape)        

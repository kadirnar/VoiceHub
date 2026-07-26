import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import numpy as np


class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(3, 3)):
        super(EncoderBlock, self).__init__()

        self.pool_size = 2

        self.conv_block = ConvBlock(in_channels, out_channels, kernel_size)

    def forward(self, x):
        latent = self.conv_block(x)
        output = F.avg_pool2d(latent, kernel_size=self.pool_size)
        return output, latent 

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(3, 3)):
        super(DecoderBlock, self).__init__()

        stride = 2

        self.upsample = nn.ConvTranspose2d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=stride,
            stride=stride,
            padding=(0, 0),
            bias=False,
        )

        self.conv_block = ConvBlock(in_channels * 2, out_channels, kernel_size)

    def forward(self, x, latent):
        x = self.upsample(x)
        x = torch.cat((x, latent), dim=1)
        output = self.conv_block(x)
        return output


class UNet(nn.Module):
    def __init__(self,freq_dim=1281,out_channel=1024):
        super(UNet, self).__init__()

        self.downsample_ratio = 16
        
 
        in_channels = 1 #self.audio_channels * self.cmplx_num

        self.encoder_block1 = EncoderBlock(in_channels, 16)
        self.encoder_block2 = EncoderBlock(16, 64)
        self.encoder_block3 = EncoderBlock(64, 256)
        self.encoder_block4 = EncoderBlock(256, 1024)
        self.middle = EncoderBlock(1024, 1024)
        self.decoder_block1 = DecoderBlock(1024, 256)
        self.decoder_block2 = DecoderBlock(256, 64)
        self.decoder_block3 = DecoderBlock(64, 16)
        self.decoder_block4 = DecoderBlock(16, 16)

        self.fc = nn.Linear(freq_dim*16, out_channel)

    def forward(self, x_ori):
        """
        Args:
            complex_sp: (batch_size, channels_num, time_steps, freq_bins)，复数张量

        Returns:
            output: (batch_size, channels_num, time_steps, freq_bins)，复数张量
        """

 
        x= self.process_image(x_ori)
        x1, latent1 = self.encoder_block1(x)
        x2, latent2 = self.encoder_block2(x1)
        x3, latent3 = self.encoder_block3(x2)
        x4, latent4 = self.encoder_block4(x3)
        _, h = self.middle(x4)
        x5 = self.decoder_block1(h, latent4)
        x6 = self.decoder_block2(x5, latent3)
        x7 = self.decoder_block3(x6, latent2)
        x8 = self.decoder_block4(x7, latent1)
        x= self.unprocess_image(x8,x_ori.shape[2])
        x = x.permute(0, 2, 1, 3).contiguous()  # 将形状变为 [6, 256, 16, 1024]
        x = x.view(x.size(0), x.size(1), -1) 
        x= self.fc(x)
 
        return x

    def process_image(self, x):
        """
        处理频谱以便可以被 downsample_ratio 整除。

        Args:
            x: (B, C, T, F)
        
        Returns:
            output: (B, C, T_padded, F_reduced)
        """

        B, C, T, Freq = x.shape

        pad_len = (
            int(np.ceil(T / self.downsample_ratio)) * self.downsample_ratio
            - T
        )
        x = F.pad(x, pad=(0, 0, 0, pad_len))

        output = x[:, :, :, 0 : Freq - 1]

        return output

    def unprocess_image(self, x,time_steps):
        """
        恢复频谱到原始形状。

        Args:
            x: (B, C, T_padded, F_reduced)
        
        Returns:
            output: (B, C, T_original, F_original)
        """
        x = F.pad(x, pad=(0, 1))

        output = x[:, :,0:time_steps, :]

        return output

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(3, 3)):
        super(ConvBlock, self).__init__()

        padding = [kernel_size[0] // 2, kernel_size[1] // 2]

        self.bn1 = nn.BatchNorm2d(in_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=padding,
            bias=False,
        )

        self.conv2 = nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=padding,
            bias=False,
        )

        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=(1, 1),
                padding=(0, 0),
            )
            self.is_shortcut = True
        else:
            self.is_shortcut = False

    def forward(self, x):
        h = self.conv1(F.leaky_relu_(self.bn1(x)))
        h = self.conv2(F.leaky_relu_(self.bn2(h)))

        if self.is_shortcut:
            return self.shortcut(x) + h
        else:
            return x + h


def test_unet():
    # 定义输入参数
    batch_size = 6
    channels = 1  # 音频通道数
    time_steps = 256  # 时间步数
    freq_bins = 1024  # 频率 bins 数

    # 创建一个随机的复数张量作为输入
    real_part = torch.randn(batch_size, channels, time_steps, freq_bins)
    imag_part = torch.randn(batch_size, channels, time_steps, freq_bins)
    complex_sp = real_part #torch.complex(real_part, imag_part)

    # 实例化 UNet 模型
    model = UNet()

    # 前向传播
    output = model(complex_sp)

    # 输出输入和输出的形状
    print("输入形状:", complex_sp.shape)
    print("输出形状:", output.shape)

    # 检查输出是否为复数张量
    assert torch.is_complex(output), "输出不是复数张量"

    # 检查输出形状是否与输入形状一致
    assert output.shape == complex_sp.shape, "输出形状与输入形状不一致"

    print("测试通过，模型正常工作。")

# 运行测试函数
if __name__ == "__main__":
    test_unet()
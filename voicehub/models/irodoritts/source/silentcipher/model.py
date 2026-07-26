import torch
import torch.nn as nn
import numpy as np


class Layer(nn.Module):
	def __init__(self, dim_in, dim_out, kernel_size, stride, padding):
		super(Layer, self).__init__()
		self.conv = nn.Conv2d(dim_in, dim_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=True)
		self.gate = nn.Conv2d(dim_in, dim_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=True)
		self.bn = nn.BatchNorm2d(dim_out)

	def forward(self, x):
		return self.bn(self.conv(x) * torch.sigmoid(self.gate(x)))

class Encoder(nn.Module):
	def __init__(self, out_dim=32, n_layers=3, message_dim=0, message_band_size=None, n_fft=None):
		super(Encoder, self).__init__()
		assert message_band_size is not None
		assert n_fft is not None
		self.message_band_size = message_band_size
		main = [Layer(dim_in=1, dim_out=32, kernel_size=3, stride=1, padding=1)]

		for i in range(n_layers-2):
			main.append(Layer(dim_in=32, dim_out=32, kernel_size=3, stride=1, padding=1))
		main.append(Layer(dim_in=32, dim_out=out_dim, kernel_size=3, stride=1, padding=1))

		self.main = nn.Sequential(*main)
		self.linear = nn.Linear(message_dim, message_band_size)
		self.n_fft = n_fft

	def forward(self, x):
		h = self.main(x)
		return h
	
	def transform_message(self, msg):
		output = self.linear(msg.transpose(2, 3)).transpose(2, 3)
		if self.message_band_size != self.n_fft // 2 + 1:
			output = torch.nn.functional.pad(output, (0, 0, 0, self.n_fft // 2 + 1 - self.message_band_size))
		return output

class CarrierDecoder(nn.Module):
	def __init__(self, config, conv_dim, n_layers=4, message_band_size=1024):
		super(CarrierDecoder, self).__init__()
		self.config = config
		self.message_band_size = message_band_size
		layers = [Layer(dim_in=conv_dim, dim_out=96, kernel_size=3, stride=1, padding=1)]

		for i in range(n_layers-2):
			layers.append(Layer(dim_in=96, dim_out=96, kernel_size=3, stride=1, padding=1))

		layers.append(Layer(dim_in=96, dim_out=1, kernel_size=1, stride=1, padding=0))

		self.main = nn.Sequential(*layers)

	def forward(self, x, message_sdr):
		h = self.main(x)
  
		if self.config.ensure_negative_message:
			h = torch.abs(h)
  
		h[:, :, self.message_band_size:, :] = 0

		if not self.config.no_normalization:
			h = h / torch.mean(h**2, dim=2, keepdim=True)**0.5 / (10**(message_sdr/20))
		
		return h

class MsgDecoder(nn.Module):
	def __init__(self, message_dim=0, message_band_size=None, channel_dim=128, num_layers=10):
		super(MsgDecoder, self).__init__()
		assert message_band_size is not None
		self.message_band_size = message_band_size

		main = [
			nn.Dropout(0),
			Layer(dim_in=1, dim_out=channel_dim, kernel_size=3, stride=1, padding=1)
		]
		for l in range(num_layers - 2):
			main += [
				nn.Dropout(0),
				Layer(dim_in=channel_dim, dim_out=channel_dim, kernel_size=3, stride=1, padding=1),
			]
		main += [
			nn.Dropout(0),
			Layer(dim_in=channel_dim, dim_out=message_dim, kernel_size=3, stride=1, padding=1)
		]
		self.main = nn.Sequential(*main)
		self.linear = nn.Linear(self.message_band_size, 1)

	def forward(self, x):
   
		h = self.main(x[:, :, :self.message_band_size])
		h = self.linear(h.transpose(2, 3)).squeeze(3).unsqueeze(1)
		return h

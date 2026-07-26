import torch

class Singleton(type):
    _instances = {}
    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]

class STFT(torch.nn.Module, metaclass=Singleton):
    def __init__(self, filter_length=1024, hop_length=512):
        super(STFT, self).__init__()

        self.filter_length = filter_length
        self.hop_len = hop_length
        self.win_len = filter_length
        self.window = torch.hann_window(self.win_len)
        self.num_samples = -1

    def transform(self, x):
        x = torch.nn.functional.pad(x, (0, self.win_len - x.shape[1]%self.win_len))
        fft = torch.stft(x, self.filter_length, self.hop_len, self.win_len, window=self.window.to(x.device), return_complex=True)
    
        real_part, imag_part = fft.real, fft.imag
        
        squared = real_part**2 + imag_part**2
        additive_epsilon = torch.ones_like(squared) * (squared == 0).float() * 1e-24
        magnitude = torch.sqrt(squared + additive_epsilon) - torch.sqrt(additive_epsilon)
        
        phase = torch.autograd.Variable(torch.atan2(imag_part.data, real_part.data)).float()
        return magnitude, phase

    def inverse(self, magnitude, phase):
        
        recombine_magnitude_phase = magnitude*torch.cos(phase) + 1j*magnitude*torch.sin(phase)
        inverse_transform = torch.istft(recombine_magnitude_phase, self.filter_length, hop_length=self.hop_len, win_length=self.win_len, window=self.window.to(magnitude.device)).unsqueeze(1)  # , length=self.num_samples
        padding = self.win_len - (self.num_samples % self.win_len)
        inverse_transform = inverse_transform[:, :, :-padding]
        return inverse_transform


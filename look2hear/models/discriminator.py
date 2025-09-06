import inspect
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import math
from .base_model import BaseModel
#from ..layers import activations, normalizations
from asteroid.utils.torch_utils import pad_x_to_y
import torch
import torch.nn as nn

class SimpleSTFTClassifier(nn.Module):
    def __init__(self, num_classes=1):
        super(SimpleSTFTClassifier, self).__init__()
        # 입력: (batch, 2, 321, 401) - 2채널(real, imag)
        self.conv1 = nn.Conv2d(2, 32, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):
        # x: (batch, 321, 401), 복소수 tensor
        x_real = x.real.unsqueeze(1)  # (batch, 1, 321, 401)
        x_imag = x.imag.unsqueeze(1)  # (batch, 1, 321, 401)
        x = torch.cat([x_real, x_imag], dim=1)  # (batch, 2, 321, 401)

        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.pool(x)  # (batch, 64, 1, 1)
        x = x.view(x.size(0), -1)  # (batch, 64)
        out = self.fc(x)  # (batch, num_classes)
        out=torch.sigmoid(out)  # Sigmoid activation for binary classification
        return out
class LightSTFTClassifier(nn.Module):
    def __init__(self, num_classes=1):
        super(LightSTFTClassifier, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(2, 16, kernel_size=3, padding=1),   # (B, 16, F, T)
            nn.BatchNorm2d(16),
            nn.ReLU(),

            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),  # (B, 32, F/2, T/2)
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # (B, 64, F/4, T/4)
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )

        self.pool = nn.AdaptiveAvgPool2d((1, 1))  # (B, 64, 1, 1)
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):
        x_real = x.real.unsqueeze(1)
        x_imag = x.imag.unsqueeze(1)
        x = torch.cat([x_real, x_imag], dim=1)  # (B, 2, F, T)
        x = self.conv_layers(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        return torch.sigmoid(self.fc(x))

class DISCRIMINATOR(BaseModel):
    def __init__(
        self,
        out_channels=128,
        in_channels=512,
        num_blocks=16,
        upsampling_depth=4,
        att_n_head=4,
        att_hid_chan=4,
        att_kernel_size=8, 
        att_stride=1,
        win=2048, 
        stride=512,
        num_sources=2,
        sample_rate=44100,
    ):
        super(DISCRIMINATOR, self).__init__(sample_rate=sample_rate)
        
        self.sample_rate = sample_rate
        self.win = win
        self.stride = stride
        self.group = self.win // 2
        self.enc_dim = self.win // 2 + 1
        self.feature_dim = out_channels
        self.num_output = num_sources
        self.eps = torch.finfo(torch.float32).eps
        self.classifier = LightSTFTClassifier()
    def pad_input(self, input, window, stride):
        """
        Zero-padding input according to window/stride size.
        """
        batch_size, nsample = input.shape

        # pad the signals at the end for matching the window/stride size
        rest = window - (stride + nsample % window) % window
        if rest > 0:
            pad = torch.zeros(batch_size, rest).type(input.type())
            input = torch.cat([input, pad], 1)
        pad_aux = torch.zeros(batch_size, stride).type(input.type())
        input = torch.cat([pad_aux, input, pad_aux], 1)

        return input, rest
    
    def forward(self, input,intermediate=False,istest=False):
        # input shape: (B, C, T)
        was_one_d = False
        if input.ndim == 1:
            was_one_d = True
            input = input.unsqueeze(0).unsqueeze(1)
        if input.ndim == 2:
            was_one_d = True
            input = input.unsqueeze(1)
        if input.ndim == 3:
            input = input
        batch_size, nch, nsample = input.shape
        input = input.view(batch_size*nch, -1) #원래가 nfft 2048, hop=512이고, n_filter 512, n_kernel 512였네..
        # spec1= torch.stft(input, n_fft=640, hop_length=160, 
        #                   window=torch.hann_window(640).to(input.device).type(input.type()),
        #                   return_complex=True) #(batch,321,401)
        # frequency-domain separation
        spec = torch.stft(input, n_fft=self.win, hop_length=self.stride, 
                          window=torch.hann_window(self.win).to(input.device).type(input.type()),
                          return_complex=True) #(batch,321,401)
        logits=self.classifier(spec)
        return logits

    def get_model_args(self):
        model_args = {"n_sample_rate": 2}
        return model_args
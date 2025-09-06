import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchlibrosa.stft import STFT, ISTFT, magphase
#from .seq import BiGRU, BiLSTM
#from .constants import N_CLASS
N_CLASS=120
FREQ_DIM=480 #원래는 320
import inspect
from torch import nn
from einops.layers.torch import Rearrange
from look2hear.models.layers import activations, normalizations
def GlobLN(nOut):
    return nn.GroupNorm(1, nOut, eps=1e-8)


class ConvNormAct(nn.Module):
    """
    This class defines the convolution layer with normalization and a PReLU
    activation
    """

    def __init__(self, nIn, nOut, kSize, stride=1, groups=1):
        """
        :param nIn: number of input channels
        :param nOut: number of output channels
        :param kSize: kernel size
        :param stride: stride rate for down-sampling. Default is 1
        """
        super().__init__()
        padding = int((kSize - 1) / 2)
        self.conv = nn.Conv1d(
            nIn, nOut, kSize, stride=stride, padding=padding, bias=True, groups=groups
        )
        self.norm = GlobLN(nOut)
        self.act = nn.PReLU()

    def forward(self, input):
        output = self.conv(input)
        output = self.norm(output)
        return self.act(output)


class ConvNorm(nn.Module):
    """
    This class defines the convolution layer with normalization and PReLU activation
    """

    def __init__(self, nIn, nOut, kSize, stride=1, groups=1, bias=True):
        """
        :param nIn: number of input channels
        :param nOut: number of output channels
        :param kSize: kernel size
        :param stride: stride rate for down-sampling. Default is 1
        """
        super().__init__()
        padding = int((kSize - 1) / 2)
        self.conv = nn.Conv1d(
            nIn, nOut, kSize, stride=stride, padding=padding, bias=bias, groups=groups
        )
        self.norm = GlobLN(nOut)

    def forward(self, input):
        output = self.conv(input)
        return self.norm(output)

class ATTConvActNorm(nn.Module):
    def __init__(
        self,
        in_chan: int = 1,
        out_chan: int = 1,
        kernel_size: int = -1,
        stride: int = 1,
        groups: int = 1,
        dilation: int = 1,
        padding: int = None,
        norm_type: str = None,
        act_type: str = None,
        n_freqs: int = -1,
        xavier_init: bool = False,
        bias: bool = True,
        is2d: bool = False,
        *args,
        **kwargs,
    ):
        super(ATTConvActNorm, self).__init__()
        self.in_chan = in_chan
        self.out_chan = out_chan
        self.kernel_size = kernel_size
        self.stride = stride
        self.groups = groups
        self.dilation = dilation
        self.padding = padding
        self.norm_type = norm_type
        self.act_type = act_type
        self.n_freqs = n_freqs
        self.xavier_init = xavier_init
        self.bias = bias

        if self.padding is None:
            self.padding = 0 if self.stride > 1 else "same"

        if kernel_size > 0:
            conv = nn.Conv2d if is2d else nn.Conv1d

            self.conv = conv(
                in_channels=self.in_chan,
                out_channels=self.out_chan,
                kernel_size=self.kernel_size,
                stride=self.stride,
                padding=self.padding,
                dilation=self.dilation,
                groups=self.groups,
                bias=self.bias,
            )
            if self.xavier_init:
                nn.init.xavier_uniform_(self.conv.weight)
        else:
            self.conv = nn.Identity()

        self.act = activations.get(self.act_type)()
        self.norm = normalizations.get(self.norm_type)(
            (self.out_chan, self.n_freqs) if self.norm_type == "LayerNormalization4D" else self.out_chan
        )

    def forward(self, x: torch.Tensor):
        output = self.conv(x)
        output = self.act(output)
        output = self.norm(output)
        return output

    def get_config(self):
        encoder_args = {}

        for k, v in (self.__dict__).items():
            if not k.startswith("_") and k != "training":
                if not inspect.ismethod(v):
                    encoder_args[k] = v

        return encoder_args


class BiGRU(nn.Module):
    def __init__(self, image_size, patch_size, channels, depth):
        super(BiGRU, self).__init__()
        image_width, image_height = pair(image_size)
        patch_width, patch_height = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0
        patch_dim = channels * patch_height * patch_width
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (w p1) (h p2) -> b (w h) (p1 p2 c)', p1=patch_width, p2=patch_height),
        )
        self.gru = nn.GRU(patch_dim, patch_dim // 2, num_layers=depth, batch_first=True, bidirectional=True)

    def forward(self, x):
        x = self.to_patch_embedding(x)
        return self.gru(x)[0]


class BiLSTM(nn.Module):
    def __init__(self, image_size, patch_size, channels, depth):
        super(BiLSTM, self).__init__()
        image_width, image_height = pair(image_size)
        patch_width, patch_height = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0
        patch_dim = channels * patch_height * patch_width
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (w p1) (h p2) -> b (w h) (p1 p2 c)', p1=patch_width, p2=patch_height),
        )
        self.lstm = nn.LSTM(patch_dim, patch_dim // 2, num_layers=depth, batch_first=True, bidirectional=True)

    def forward(self, x):
        return self.lstm(x)[0]


# helpers
def pair(t):
    return t if isinstance(t, tuple) else (t, t)

def init_layer(layer: nn.Module):
    r"""Initialize a Linear or Convolutional layer."""
    nn.init.xavier_uniform_(layer.weight)

    if hasattr(layer, "bias"):
        if layer.bias is not None:
            layer.bias.data.fill_(0.0)


def init_bn(bn: nn.Module):
    r"""Initialize a Batchnorm layer."""
    bn.bias.data.fill_(0.0)
    bn.weight.data.fill_(1.0)
    bn.running_mean.data.fill_(0.0)
    bn.running_var.data.fill_(1.0)


class Wav2Spec(nn.Module):
    def __init__(self, hop_length, window_size):
        super(Wav2Spec, self).__init__()
        self.hop_length = hop_length
        self.stft = STFT(window_size, hop_length, window_size)

    def forward(self, audio):
        bs, c, segment_samples = audio.shape
        audio = audio.reshape(bs * c, segment_samples)
        real, imag = self.stft(audio[:, :-1])
        mag = torch.clamp(real ** 2 + imag ** 2, 1e-10, np.inf) ** 0.5
        cos = real / mag
        sin = imag / mag
        _, _, time_steps, freq_bins = mag.shape
        mag = mag.reshape(bs, c, time_steps, freq_bins)
        cos = cos.reshape(bs, c, time_steps, freq_bins)
        sin = sin.reshape(bs, c, time_steps, freq_bins)
        return mag, cos, sin


class Spec2Wav(nn.Module):
    def __init__(self, hop_length, window_size):
        super(Spec2Wav, self).__init__()
        self.istft = ISTFT(window_size, hop_length, window_size)

    def forward(self, x, spec_m, cos_m, sin_m, audio_len):
        bs, c, time_steps, freqs_steps = x.shape
        x = x.reshape(bs, c // 4, 4, time_steps, freqs_steps)
        mask_spec = torch.sigmoid(x[:, :, 0, :, :])
        _mask_real = torch.tanh(x[:, :, 1, :, :])
        _mask_imag = torch.tanh(x[:, :, 2, :, :])
        _, mask_cos, mask_sin = magphase(_mask_real, _mask_imag)
        linear_spec = x[:, :, 3, :, :]
        out_cos = cos_m * mask_cos - sin_m * mask_sin
        out_sin = sin_m * mask_cos + cos_m * mask_sin
        out_spec = F.relu(spec_m.detach() * mask_spec + linear_spec)
        out_real = (out_spec * out_cos).reshape(bs * c // 4, 1, time_steps, freqs_steps)
        out_imag = (out_spec * out_sin).reshape(bs * c // 4, 1, time_steps, freqs_steps)
        audio = self.istft(out_real, out_imag, audio_len).reshape(bs, c // 4, audio_len)
        return audio, out_spec
    
    # def forward(self, x, spec_m):
    #     bs, c, time_steps, freqs_steps = x.shape
    #     x = x.reshape(bs, c // 4, 4, time_steps, freqs_steps)
    #     mask_spec = torch.sigmoid(x[:, :, 0, :, :])
    #     linear_spec = x[:, :, 3, :, :]
    #     out_spec = F.relu(spec_m.detach() * mask_spec + linear_spec)
    #     return out_spec



class ResConvBlock(nn.Module):
    def __init__(self, in_planes, planes, bias=False):
        super(ResConvBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes, momentum=0.01)
        self.bn2 = nn.BatchNorm2d(planes, momentum=0.01)
        self.act1 = nn.PReLU()
        self.act2 = nn.PReLU()
        self.conv1 = nn.Conv2d(in_planes, planes, (3, 3), padding=(1, 1), bias=bias)
        self.conv2 = nn.Conv2d(planes, planes, (3, 3), padding=(1, 1), bias=bias)
        if in_planes != planes:
            self.shortcut = nn.Conv2d(in_planes, planes, (1, 1))
            self.is_shortcut = True
        else:
            self.is_shortcut = False
        self.init_weights()

    def init_weights(self):
        r"""Initialize weights."""
        init_bn(self.bn1)
        init_bn(self.bn2)
        init_layer(self.conv1)
        init_layer(self.conv2)

        if self.is_shortcut:
            init_layer(self.shortcut)

    def forward(self, x):
        out = self.conv1(self.act1(self.bn1(x))) #(1,1,401,321)->(1,1,401,321)-
        out = self.conv2(self.act2(self.bn2(out)))
        if self.is_shortcut:
            return self.shortcut(x) + out
        else:
            return out + x


class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, n_blocks, kernel_size, bias):
        super(EncoderBlock, self).__init__()
        self.conv = nn.ModuleList([
            ResConvBlock(in_channels, out_channels, bias)
        ])
        for i in range(n_blocks - 1):
            self.conv.append(ResConvBlock(out_channels, out_channels, bias))
        if kernel_size is not None:
            self.pool = nn.MaxPool2d(kernel_size)
        else:
            self.pool = None

    def forward(self, x):
        for each_layer in self.conv:
            x = each_layer(x)
        if self.pool is not None:
            return x, self.pool(x)
        return x


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, n_blocks, stride, bias, gate=False):
        super(DecoderBlock, self).__init__()
        self.gate = gate
        if self.gate:
            self.W_g = nn.Sequential(
                nn.Conv2d(out_channels, out_channels // 2, (1, 1)),
                nn.BatchNorm2d(out_channels // 2)
            )
            self.W_x = nn.Sequential(
                nn.Conv2d(out_channels, out_channels // 2, (1, 1)),
                nn.BatchNorm2d(out_channels // 2)
            )
            self.psi = nn.Sequential(
                nn.Conv2d(out_channels // 2, 1, (1, 1)),
                nn.BatchNorm2d(1),
                nn.Sigmoid()
            )
        self.conv1 = nn.ConvTranspose2d(in_channels, out_channels, stride, stride, (0, 0), bias=bias)
        self.bn1 = nn.BatchNorm2d(in_channels, momentum=0.01)
        self.conv = nn.ModuleList([
            ResConvBlock(out_channels * 2, out_channels, bias)
        ])
        for i in range(n_blocks - 1):
            self.conv.append(ResConvBlock(out_channels, out_channels, bias))
        self.init_weights()
        self.pitch_proj = nn.Conv2d(1, out_channels, kernel_size=1)  # 1x1 conv for channel up
        self.fusion_conv = nn.Conv2d(3 * out_channels, 2 * out_channels, kernel_size=1)

    def init_weights(self):
        init_bn(self.bn1)
        init_layer(self.conv1)

    def forward(self, x, concat,mixed_salience=None):
        x = self.conv1(F.relu_(self.bn1(x))) #(1,128,401,40)->(1,64,401,80)
        #mixed_saliece : (b,t=diff,f=120)
        b,c,t,f=x.shape
        mixed_salience = self.pitch_proj(mixed_salience.unsqueeze(1))  # (b, 1, t, f)
        pitch_time=mixed_salience.shape[2]
        if self.gate: #()
            concat = x * self.psi(F.relu_(self.W_g(x) + self.W_x(concat)))
        x=torch.nn.functional.interpolate(x, size=(pitch_time, f), mode='bilinear', align_corners=True)  # (b, c, t, f)
        concat=torch.nn.functional.interpolate(concat, size=(pitch_time, f), mode='bilinear', align_corners=True)  # (b, c, t, f)
        mixed_salience = torch.nn.functional.interpolate(mixed_salience, size=(pitch_time, f), mode='bilinear', align_corners=True)  # (b, c, t, f)
        x = torch.cat((x, concat,mixed_salience), dim=1)
        x = self.fusion_conv(x)  # (b, 3*128, t, f) -> (b, 2*128, t, f)
        for each_layer in self.conv:
            x = each_layer(x) #
        return x


class DJCMEncoder(nn.Module):
    def __init__(self, in_channels, n_blocks):
        super(DJCMEncoder, self).__init__()
        self.en_blocks = nn.ModuleList([
            EncoderBlock(in_channels, 32, n_blocks, (1, 2), False),
            EncoderBlock(32, 32, n_blocks, (1, 2), False),
            #EncoderBlock(32, 32, n_blocks, (1, 2), False),
            #EncoderBlock(128, 128, n_blocks, (1, 2), False),
            # EncoderBlock(256, 384, n_blocks, (1, 2), False),
            # EncoderBlock(384, 384, n_blocks, (1, 2), False)
        ])

    def forward(self, x):
        concat_tensors = []
        for layer in self.en_blocks:
            _, x = layer(x)
            concat_tensors.append(_)
        return x, concat_tensors


class Decoder(nn.Module):
    def __init__(self, n_blocks, gate=False):
        super(Decoder, self).__init__()
        self.de_blocks = nn.ModuleList([
            # DecoderBlock(384, 384, n_blocks, (1, 2), False, gate),
            # DecoderBlock(384, 384, n_blocks, (1, 2), False, gate),
            #DecoderBlock(128, 128, n_blocks, (1, 2), False, gate),
            #DecoderBlock(32, 32, n_blocks, (1, 2), False, gate),
            DecoderBlock(32, 32, n_blocks, (1, 2), False, gate),
            DecoderBlock(32, 32, n_blocks, (1, 2), False, gate),
        ])

    def forward(self, x, concat_tensors,mixed_salience):
        for i, layer in enumerate(self.de_blocks):
            x = layer(x, concat_tensors[-1-i],mixed_salience) #x:(1,128,401,20)
        return x


class DJCMLatentBlocks(nn.Module):
    def __init__(self, n_blocks, latent_layers):
        super(DJCMLatentBlocks, self).__init__()
        self.latent_blocks = nn.ModuleList([])
        for i in range(latent_layers):
            self.latent_blocks.append(EncoderBlock(32, 32, n_blocks, None, False))

    def forward(self, x):
       
        for layer in self.latent_blocks:
            x = layer(x) #(1,128,401->249,40)
        return x
#MySpeakerAttention(out_channels//2, 1, n_head=n_head, hid_chan=att_hid_chan, act_type="prelu", norm_type="LayerNormalization4D", dim=4)
class MyCrossAttention(nn.Module):
    def __init__(
        self,
        in_chan: int,
        n_freqs: int=1,
        n_head: int = 4,
        hid_chan: int = 4,
        act_type: str = "prelu",
        norm_type: str = "LayerNormalization4D",
        dim: int = 4,
        *args,
        **kwargs,
    ):
        super(MyCrossAttention, self).__init__()
        self.in_chan = in_chan
        self.n_freqs = n_freqs
        self.n_head = n_head
        self.hid_chan = hid_chan
        self.act_type = act_type
        self.norm_type = norm_type
        self.dim = dim

        assert self.in_chan % self.n_head == 0

        self.Queries = nn.ModuleList()
        self.Keys = nn.ModuleList()
        self.Values = nn.ModuleList()

        for _ in range(self.n_head):
            self.Queries.append(
                ATTConvActNorm(
                    in_chan=self.in_chan,
                    out_chan=self.hid_chan,
                    kernel_size=1,
                    act_type=self.act_type,
                    norm_type=self.norm_type,
                    n_freqs=self.n_freqs,
                    is2d=True,
                )
            )
            self.Keys.append(
                ATTConvActNorm(
                    in_chan=self.in_chan,
                    out_chan=self.hid_chan,
                    kernel_size=1,
                    act_type=self.act_type,
                    norm_type=self.norm_type,
                    n_freqs=self.n_freqs,
                    is2d=True,
                )
            )
            self.Values.append(
                ATTConvActNorm(
                    in_chan=self.in_chan,
                    out_chan=self.in_chan // self.n_head,
                    kernel_size=1,
                    act_type=self.act_type,
                    norm_type=self.norm_type,
                    n_freqs=self.n_freqs,
                    is2d=True,
                )
            )

        self.attn_concat_proj = ATTConvActNorm(
            in_chan=self.in_chan,
            out_chan=self.in_chan,
            kernel_size=1,
            act_type=self.act_type,
            norm_type=self.norm_type,
            n_freqs=self.n_freqs,
            is2d=True,
        )

    def forward(self, x1: torch.Tensor,x2: torch.Tensor):
        #(128,401,20)
        batch_size, ch, time, freq = x1.size()#(batch,64,249,80)
        residual1 = x1
        residual2 = x2
        x=torch.cat([x1.unsqueeze(1),x2.unsqueeze(1)],dim=1) # (batch,2,128,401,67)
        x=x.view(2*batch_size, ch, time, freq).contiguous()  # [B*2, 64,249,80)
        all_Q = [q(x) for q in self.Queries]  # [B*2, E=4, T=249, F=80]
        all_K = [k(x) for k in self.Keys]  # [B*2, E, T, F]
        all_V = [v(x) for v in self.Values]  # [B*2, C/n_head=16, T, F]

        Q = torch.cat(all_Q, dim=0)  # [B'*2*num_head, E, T, F]    B' = B*n_head
        K = torch.cat(all_K, dim=0)  # [B'*2*num_head, E, T, F]
        V = torch.cat(all_V, dim=0)  # [B'*2*num_head, C/n_head, T, F]
        Q=Q.view(batch_size*self.n_head ,2, self.hid_chan, time, freq).contiguous()  # [B, 2, E, T, F]
        K=K.view(batch_size*self.n_head ,2, self.hid_chan, time, freq).contiguous()  # [B, 2, E, T, F]
        V=V.view(batch_size*self.n_head ,2, ch//self.n_head, time, freq).contiguous()  # [B, 2, C/n_head, T, F]
        Q=Q.permute(0,3,1,2,4).contiguous()  # [B, T, 2, E, F]
        K=K.permute(0,3,1,2,4).contiguous()
        V=V.permute(0,3,1,2,4).contiguous()  # [B, T, 2, C/n_head, F]
        Q=Q.view(batch_size*time*self.n_head, 2, self.hid_chan, freq).contiguous()  # [B*t=996, 2, E, F]
        K=K.view(batch_size*time*self.n_head, 2, self.hid_chan, freq).contiguous()  # [B*t, 2, E, F]
        V=V.view(batch_size*time*self.n_head, 2, ch//self.n_head, freq).contiguous()  # [B*t, 2, C/n_head, F]
        
        Q = Q.flatten(start_dim=2)  # [B*t, 2, E*F]
        K = K.flatten(start_dim=2)  # [B*t, 2, E*F]
        old_shape = V.shape #old_shape: [B*t, 2, C/n_head, F]
        V = V.flatten(start_dim=2)  # [B*t, 2, C*F/n_head]
        emb_dim = Q.shape[-1]  # C*F/n_head

        attn_mat = torch.matmul(Q, K.transpose(1, 2)) / (emb_dim**0.5)  # [B*t, 2, 2]
        attn_mat = F.softmax(attn_mat, dim=2)  # [B*t, 2, 2]
        V = torch.matmul(attn_mat, V)  # [B*t, 2, C*F/n_head]
        V = V.reshape(old_shape)  # [B*t, 2, C/n_head, F]
        V = V.transpose(1, 2)  # [B*t, C/n_head, 2, F]
        emb_dim = V.shape[1]  # C/n_head
        #(batch,2,128,401,67)
        x = V.reshape([self.n_head, batch_size, 2, emb_dim, time, freq])  # [n_head, B, C/n_head, T, F]
        x = x.transpose(0, 1).contiguous()  # [B, n_head, 2, C/n_head, T, F]
        x=x.permute(0,2,1,3,4,5).contiguous()  # [B, 2, n_head, C/n_head, T, F]
        x = x.reshape([batch_size, 2, self.n_head * emb_dim, time, freq])  # [B, 2, C, T, F]
        
        x1=x[:,0,:,:,:]  # [B, C, T, F]
        x2=x[:,1,:,:,:]
        x1 = self.attn_concat_proj(x1)  # [B, C, T, F]
        x2 = self.attn_concat_proj(x2)

        x1 = x1 + residual1
        x2 = x2 + residual2

        return x1,x2 #[batch_size, ch, time, freq]
    def forward(self, x1,x2: torch.Tensor):
        batch_size, ch, time, freq = x1.size()
        residual1 = x1
        residual2 = x2
        x1=x1.permute(0,2,1,3).contiguous() # (b,time,64,nband)
        x2=x2.permute(0,2,1,3).contiguous() # (b,time,64,nband)
        x1=x1.view(batch_size*time, self.in_chan, freq).contiguous() # (b*time,64,nband)
        x2=x2.view(batch_size*time, self.in_chan, freq).contiguous()
        
        x= torch.cat([x1.unsqueeze(2), x2.unsqueeze(2)], dim=2)  # (b*time,64,2,nband)
        _batch_size,_ch,_time,_freq=x.shape
        all_Q = [q(x) for q in self.Queries]  # [B, E, T, F]
        all_K = [k(x) for k in self.Keys]  # [B, E, T, F]
        all_V = [v(x) for v in self.Values]  # [B, C/n_head, T, F]

        Q = torch.cat(all_Q, dim=0)  # [B', E, T=2, F=67]    B' = B*n_head
        K = torch.cat(all_K, dim=0)  # [B', E, T, F]
        V = torch.cat(all_V, dim=0)  # [B', C/n_head, T, F]

        Q = Q.transpose(1, 2).flatten(start_dim=2)  # [B', T, E*F]
        K = K.transpose(1, 2).flatten(start_dim=2)  # [B', T, E*F]
        V = V.transpose(1, 2)  # [B', T, C/n_head, F]
        old_shape = V.shape
        V = V.flatten(start_dim=2)  # [B', T, C*F/n_head]
        emb_dim = Q.shape[-1]  # C*F/n_head
        
        attn_mat = torch.matmul(Q, K.transpose(1, 2)) / (emb_dim**0.5)  # [B', T, T]
        attn_mat = F.softmax(attn_mat, dim=2)  # [B', T, T]
        V = torch.matmul(attn_mat, V)  # [B', T, C*F/n_head]
        V = V.reshape(old_shape)  # [B', T, C/n_head, F]
        V = V.transpose(1, 2)  # [B'=b*T, C=64/n_head=4, T=2, F=67]
        emb_dim = V.shape[1]  # C/n_head 
        #(b,128,time,nband)
        x = V.view([self.n_head, _batch_size, emb_dim, _time, _freq])  # [n_head, B, C/n_head, T, F]
        
        x = x.transpose(0, 1).contiguous()  # [B=344, n_head=4, C/n_head=16, T=2, F=67]

        x = x.view([_batch_size, self.n_head * emb_dim, _time, _freq])  # [B, C, T, F]
        x = self.attn_concat_proj(x)  # [B=344, C=64, T=2, F=67]
        x1=x[:,:,0,:] # (b*t,64,nband)
        x2=x[:,:,1,:] # (b*t,64,nband)
        x1=x1.view(batch_size, time, self.in_chan, freq).contiguous()  # [B, T, 64, F]
        x2=x2.view(batch_size, time, self.in_chan, freq).contiguous()  # [B, T, 64, F]
        x1=x1.permute(0,2,1,3).contiguous()  # 
        x2=x2.permute(0,2,1,3).contiguous()
        x1 = x1 + residual1
        x2 = x2 + residual2

        return x1,x2


class DecoderCrossAttention(nn.Module):
    def __init__(self, n_blocks, gate=False):
        super(DecoderCrossAttention, self).__init__()
        self.de_blocks = nn.ModuleList([
            #DecoderBlock(128, 128, n_blocks, (1, 2), False, gate),
            #DecoderBlock(32, 32, n_blocks, (1, 2), False, gate),
            DecoderBlock(32, 32, n_blocks, (1, 2), False, gate),
            DecoderBlock(32, 32, n_blocks, (1, 2), False, gate),
        ])
        self.cross_attention =nn.ModuleList()
        self.cross_attention.append(MyCrossAttention(32))
        self.cross_attention.append(MyCrossAttention(32))
        #self.cross_attention.append(MyCrossAttention(32))

    def forward(self,x1,x2,concat_tensors1,concat_tensors2, mixed_salience):
        for i, layer in enumerate(self.de_blocks):
            x1 = layer(x1, concat_tensors1[-1-i],mixed_salience) #x:(1,64,249,80)->
            x2 = layer(x2, concat_tensors2[-1-i],mixed_salience) #
            x1, x2 = self.cross_attention[i](x1,x2)#(1,64=dim,249=time,80)
        return x1, x2 #(1,32,249,320)
class DJCMPE_DecoderCrossAttention(nn.Module):
    def __init__(self, n_blocks, seq_frames, seq='gru', seq_layers=1, gate=False):
        super(DJCMPE_DecoderCrossAttention, self).__init__()
        self.de_blocks = DecoderCrossAttention(n_blocks, gate)
        self.after_conv1 = EncoderBlock(32, 32, n_blocks, None, False)
        self.after_conv2 = nn.Conv2d(32, 1, (1, 1))
        init_layer(self.after_conv2)
        if seq.lower() == 'gru':
            self.fc = nn.Sequential(
                BiGRU((seq_frames, FREQ_DIM), (1, FREQ_DIM), 1, seq_layers),
                nn.Linear(FREQ_DIM, N_CLASS),
                nn.Sigmoid()
            )
        elif seq.lower() == 'lstm':
            self.fc = nn.Sequential(
                BiLSTM((seq_frames, FREQ_DIM), (1, FREQ_DIM), 1, seq_layers),
                nn.Linear(FREQ_DIM, N_CLASS),
                nn.Sigmoid()
            )
        else:
            self.fc = nn.Sequential(
                nn.Linear(FREQ_DIM, N_CLASS),
                nn.Sigmoid()
            )
    
    def forward(self,x1,x2,concat_tensors1,concat_tensors2,mixed_salience=None):
        x1, x2 = self.de_blocks(x1,x2,concat_tensors1,concat_tensors2,mixed_salience) #(1,128,401,40) (32->64->128)
        x1 = self.after_conv2(self.after_conv1(x1)) #(1,1,249,512)
        x2 = self.after_conv2(self.after_conv1(x2))
        x1 = self.fc(x1).squeeze(1) #(1,249,120)
        x2 = self.fc(x2).squeeze(1) #(1,249,120)
        return x1, x2
        

import torch
import torch.nn as nn
import torch.nn.functional as fn
from models.slstm import SLSTM


def get_activate(act_name: str, ):
    pass


class CausalConv1d(nn.Conv1d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.causal_padding = self.dilation[0] * (self.kernel_size[0] - 1)

    def forward(self, x):
        return self._conv_forward(fn.pad(x, [self.causal_padding, 0]), self.weight, self.bias)


class CausalConvTranspose1d(nn.ConvTranspose1d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.causal_padding = self.dilation[0] * (self.kernel_size[0] - 1) + self.output_padding[0] + 1 - self.stride[0]

    def forward(self, x, output_size=None):
        if self.padding_mode != 'zeros':
            raise ValueError('Only `zeros` padding mode is supported for ConvTranspose1d')

        assert isinstance(self.padding, tuple)
        output_padding = self._output_padding(
            x, output_size, self.stride, self.padding, self.kernel_size, self.dilation)
        return fn.conv_transpose1d(
            x, self.weight, self.bias, self.stride, self.padding,
            output_padding, self.groups, self.dilation)[..., :-self.causal_padding]


class ResConv1D(nn.Module):
    def __init__(self, in_channels, out_channels, activate_fn, kernel_size=7, dilation=1, dropout=0.0):
        super().__init__()

        self.dilation = dilation

        self.layers = nn.Sequential(
            CausalConv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, dilation=dilation),
            activate_fn if activate_fn is not None else nn.ELU(),
            nn.Conv1d(in_channels=out_channels, out_channels=out_channels, kernel_size=1),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return x + self.layers(x)


class TemporalEncodeBlock(nn.Module):
    def __init__(self, n_latent, n_layer, n_cycle, n_kernel: int = 7, n_stride: int = 1, n_lstm: int = 0, scale_mode: str = 'exp', activate_fn=None, dropout=0.0):
        super().__init__()
        self.enc = nn.Sequential()
        self.n_stride = n_stride
        for i in range(n_layer):
            n_dia_skip = (i + 1) % n_cycle + 1
            if scale_mode == 'exp':
                n_dia_skip = (2 ** i) % n_cycle + 1
            self.enc.append(ResConv1D(in_channels=n_latent, out_channels=n_latent, activate_fn=activate_fn, kernel_size=n_kernel, dilation=n_dia_skip, dropout=dropout))
        self.lstm = SLSTM(n_latent, num_layers=2, ) if n_lstm > 0 else nn.Identity()
        self.enc.append(CausalConv1d(in_channels=n_latent, out_channels=n_latent, kernel_size=2 * n_stride, stride=n_stride))

    def forward(self, x, mask):
        mask = mask[:, :, ::self.n_stride]
        hidden = self.enc(x)
        hidden = self.lstm(hidden)
        return hidden * mask, mask


class TemporalDecodeBlock(nn.Module):
    def __init__(self, n_latent, n_layer, n_cycle, n_kernel: int = 7, n_stride: int = 1, n_lstm: int = 0, scale_mode: str = 'exp', activate_fn=None, dropout=0.0):
        super().__init__()
        self.dec = nn.Sequential()
        self.lstm = SLSTM(n_latent, num_layers=2, ) if n_lstm > 0 else nn.Identity()
        self.n_stride = n_stride
        self.dec.append(CausalConvTranspose1d(in_channels=n_latent, out_channels=n_latent, kernel_size=2 * n_stride, stride=n_stride))
        for i in range(n_layer):
            n_dia_skip = (i + 1) % n_cycle + 1
            if scale_mode == 'exp':
                n_dia_skip = (2 ** i) % n_cycle + 1
            self.dec.append(ResConv1D(in_channels=n_latent, out_channels=n_latent, activate_fn=activate_fn, kernel_size=n_kernel, dilation=n_dia_skip, dropout=dropout))

    def forward(self, x):
        hidden = self.lstm(x)
        return self.dec(hidden)


class Encoder(nn.Module):
    def __init__(self, in_channels, n_latent, strides, n_kernel: int = 7, n_layer: int = 4, n_cycle: int = 12, dropout=0.1, n_lstm=0):
        super().__init__()
        self.conv = CausalConv1d(in_channels=in_channels, out_channels=n_latent, kernel_size=n_kernel)
        self.act_fn = nn.ELU()
        self.global_encoder = nn.ModuleList()
        self.local_encoder = nn.ModuleList()

        for isd in strides:
            self.global_encoder.append(
                TemporalEncodeBlock(n_latent=n_latent, n_layer=n_layer, n_cycle=n_cycle, n_lstm=n_lstm, n_kernel=n_kernel, n_stride=isd, scale_mode='exp', dropout=dropout),
            )
            self.local_encoder.append(
                TemporalEncodeBlock(n_latent=n_latent, n_layer=n_layer, n_cycle=1, n_kernel=n_kernel, n_stride=isd, scale_mode='linear', dropout=dropout)
            )

    def forward(self, x, mask):
        hidden = self.act_fn(self.conv(x)) * mask
        for gtb, ltb in zip(self.global_encoder, self.local_encoder):
            hidden_g, mask_g = gtb(hidden, mask)
            hidden_l, mask_l = ltb(hidden, mask)
            hidden = hidden_g - hidden_l
            mask = mask_g
        return hidden_g, hidden_l, mask


class Decoder(nn.Module):
    def __init__(self, out_channels, n_latent, strides, n_kernel: int = 7, n_layer: int = 4, n_cycle: int = 12, dropout=0.1, n_lstm: int = 0):
        super().__init__()
        self.conv = CausalConv1d(in_channels=n_latent, out_channels=out_channels, kernel_size=n_kernel)
        self.act_fn = nn.ELU()
        self.global_encoder = nn.ModuleList()
        self.local_encoder = nn.ModuleList()
        strides.reverse()
        for isd in strides:
            self.global_encoder.append(
                TemporalDecodeBlock(n_latent=n_latent, n_layer=n_layer, n_cycle=n_cycle, n_lstm=n_lstm, n_kernel=n_kernel, n_stride=isd, scale_mode='exp', dropout=dropout)
            )
            self.local_encoder.append(
                TemporalDecodeBlock(n_latent=n_latent, n_layer=n_layer, n_cycle=1, n_lstm=0, n_kernel=n_kernel, n_stride=isd, scale_mode='linear', dropout=dropout)
            )

    def forward(self, x, mask=None):
        hidden = x
        for gtb, ltb in zip(self.global_encoder, self.local_encoder):
            hidden_g = gtb(hidden)
            hidden_l = ltb(hidden)
            hidden = hidden_g + hidden_l
        output = self.act_fn(self.conv(hidden))
        if mask is not None:
            output = output * mask
        return output


def main():
    b_tmp_e_block = False
    b_tmp_d_block = False
    b_tmp_encoder = False
    b_tmp_decoder = True

    batch = 4
    n_seq = 320
    n_dim = 256
    n_fet = 51

    if b_tmp_e_block:
        net = TemporalEncodeBlock(n_latent=n_dim, n_layer=3, n_cycle=12, n_kernel=7, n_stride=2, scale_mode='exp')
        _ = net.eval()

        x = torch.rand(size=(batch, n_dim, n_seq))
        mask = torch.rand(size=(batch, 1, n_seq))
        y, y_mask = net(x, mask)
        print("x shape : {}, y : shape {}, y_mask : shape {}".format(x.shape, y.shape, y_mask.shape))

    if b_tmp_d_block:
        net = TemporalDecodeBlock(n_latent=n_dim, n_layer=3, n_cycle=12, n_kernel=7, n_stride=2, scale_mode='exp')
        _ = net.eval()

        x = torch.rand(size=(batch, n_dim, n_seq // 2))
        y = net(x)
        print("x shape : {}, y : shape {}".format(x.shape, y.shape))

    if b_tmp_encoder:
        net = Encoder(in_channels=n_fet, n_latent=256, strides=[2, 2, 2, 2], n_layer=3, n_lstm=2)
        _ = net.eval()

        x = torch.rand(size=(batch, n_fet, n_seq))
        mask = torch.rand(size=(batch, 1, n_seq))
        y_global, y_local, y_mask = net(x, mask)
        print("x shape : {}, y_g : {}, y_l : {}, y_mask : shape {}".format(x.shape, y_global.shape, y_local.shape, y_mask.shape))

    if b_tmp_decoder:
        net = Decoder(out_channels=n_fet, n_latent=256, strides=[2, 2, 2, 2], n_layer=3)
        _ = net.eval()

        n_seq = 320
        n_decode_seq = 20

        x = torch.rand(size=(batch, 256, n_decode_seq))
        mask = torch.rand(size=(batch, 1, n_seq))
        y = net(x, mask)
        print("x shape : {}, y : {}".format(x.shape, y.shape))


if __name__ == '__main__':
    print()
    main()

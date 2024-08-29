from typing import List
import torch
import torch.nn as nn
from models.tcn import Encoder, Decoder
from models.vector_quantize import GroupedResidualVQ


class LSCodec(nn.Module):
    def __init__(self,
                 n_feat: int, n_latent: int, strides: List[int],
                 n_kernel: int = 7, n_layer: int = 4, n_lstm: int = 2, n_cycle: int = 12, dropout=0.1):
        super().__init__()

        self.encoder = Encoder(in_channels=n_feat, n_latent=n_latent, strides=strides, n_kernel=n_kernel, n_layer=n_layer, n_lstm=n_lstm, n_cycle=n_cycle, dropout=dropout)
        self.decoder = Decoder(out_channels=n_feat, n_latent=n_latent, strides=strides, n_kernel=n_kernel, n_layer=n_layer, n_lstm=n_lstm, n_cycle=n_cycle, dropout=dropout)
        self.quantizer = GroupedResidualVQ(dim=n_latent * 2, groups=2, num_quantizers=8, codebook_dim=256, codebook_size=512, kmeans_init=True,
                                           shared_codebook=False, ema_update=True, threshold_ema_dead_code=2, quantize_dropout=True)

    def forward(self, x, mask=None):
        #############################################################################
        # param x : tensor shape like (batch, n_dim, n_len)
        # param mask : tensor shape like (batch, 1, n_len) with 1 keep and 0 ignore
        if mask is None:
            mask = torch.ones(size=(x.shape[0], 1, x.shape[2]), dtype=x.dtype, device=x.device)
        hidden_g, hidden_l, mask_ = self.encoder(x, mask)

        #########################
        # quantizer
        hidden_g = torch.transpose(hidden_g, 2, 1)
        hidden_l = torch.transpose(hidden_l, 2, 1)
        hidden, all_indices, commit_losses = self.quantizer(torch.cat([hidden_g, hidden_l], dim=-1), mask=mask_.squeeze(1).bool())
        hidden_g, hidden_l = torch.transpose(hidden, 2, 1).chunk(2, dim=1)
        index_g, index_l = all_indices[0, ...].permute(0, 2, 1), all_indices[1, ...].permute(0, 2, 1)

        hidden = hidden_g + hidden_l
        output = self.decoder(hidden) * mask
        output_g = self.decoder(hidden_g) * mask
        output_l = self.decoder(hidden_l) * mask
        return output, output_g, output_l, commit_losses

    def encode(self, x, mask=None, b_split: bool = True):
        if mask is None:
            mask = torch.ones(size=(x.shape[0], 1, x.shape[2]), dtype=x.dtype, device=x.device)
        #########################
        # encode
        hidden_g, hidden_l, mask_ = self.encoder(x, mask)

        #########################
        # quantizer
        hidden_g = torch.transpose(hidden_g, 2, 1)
        hidden_l = torch.transpose(hidden_l, 2, 1)
        hidden, all_indices, commit_losses = self.quantizer(torch.cat([hidden_g, hidden_l], dim=-1), mask=mask_.squeeze(1).bool())
        hidden_g, hidden_l = torch.transpose(hidden, 2, 1).chunk(2, dim=1)

        if b_split:
            return hidden_g, hidden_l
        else:
            return hidden_g + hidden_l

    def decode(self, h, mask=None):
        if mask is None:
            return self.decoder(h)
        return self.decoder(h) * mask

    def encode_idx(self, x, mask=None, b_split: bool = True):
        if mask is None:
            mask = torch.ones(size=(x.shape[0], 1, x.shape[2]), dtype=x.dtype, device=x.device)

        hidden_g, hidden_l, mask_ = self.encoder(x, mask)

        #########################
        # quantizer
        hidden_g = torch.transpose(hidden_g, 2, 1)
        hidden_l = torch.transpose(hidden_l, 2, 1)
        hidden, all_indices, commit_losses = self.quantizer(torch.cat([hidden_g, hidden_l], dim=-1), mask=mask_.squeeze(1).bool())
        index_g, index_l = all_indices[0, ...].permute(0, 2, 1), all_indices[1, ...].permute(0, 2, 1)
        if b_split:
            return index_g, index_l
        else:
            return torch.cat([index_g, index_l], dim=1)

    def decode_idx(self, idx, mask=None):
        index_g, index_l = idx[:, :8, :], idx[:, 8:, :]
        all_indices = torch.cat([index_g.unsqueeze(0), index_l.unsqueeze(0)], dim=0)
        all_indices = torch.transpose(all_indices, 3, 2)
        hidden_quantize = self.quantizer.get_output_from_indices(all_indices)
        hidden_g, hidden_l = torch.transpose(hidden_quantize, 2, 1).chunk(2, dim=1)
        hidden = hidden_g + hidden_l
        if mask is None:
            return self.decoder(hidden)
        return self.decoder(hidden) * mask


def main():
    n_feat = 51
    n_latent = 256
    n_seq = 320
    n_bat = 4
    net = LSCodec(n_feat=n_feat, n_latent=n_latent, strides=[2, 5])
    _ = net.eval()

    #####################################
    # fwd io interface
    x = torch.rand(size=(n_bat, n_feat, n_seq))
    output, index_g, index_l = net(x)
    print("[fwd] x shape {}, y shape {}".format(x.shape, output.shape))
    print("[fwd] index_g shape {}, index_l shape {}".format(index_g.shape, index_l.shape))

    #####################################
    # encode
    hidden = net.encode(x, b_split=False)
    output = net.decode(hidden)
    print("[encode] : hidden shape {}, output shape {}".format(hidden.shape, output.shape))

    #####################################
    # encode_idx
    hidden_idx = net.encode_idx(x, b_split=False)
    output = net.decode_idx(hidden_idx)
    print("[encode_idx] : hidden shape {}, output shape {}".format(hidden_idx.shape, output.shape))


if __name__ == '__main__':
    main()

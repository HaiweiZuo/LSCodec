import os
import re
import time
import toml

import torch as th
import torch.nn as nn
from torchinfo import summary
from models.lscodec import LSCodec


class Enc(nn.Module):
    def __init__(self, net: LSCodec):
        super().__init__()
        self.enc = net.encoder
        self.q = net.quantizer

    def forward(self, x, mask=None):
        if mask is None:
            mask = th.ones(size=(x.shape[0], 1, x.shape[2]), dtype=x.dtype, device=x.device)
        #########################
        # encode
        hidden_g, hidden_l, mask_ = self.enc(x, mask)

        #########################
        # quantizer
        hidden_g = th.transpose(hidden_g, 2, 1)
        hidden_l = th.transpose(hidden_l, 2, 1)
        hidden, _, _ = self.q(th.cat([hidden_g, hidden_l], dim=-1), mask=mask_.squeeze(1).bool())
        hidden_g, hidden_l = th.transpose(hidden, 2, 1).chunk(2, dim=1)
        return hidden_g + hidden_l


class Dec(nn.Module):
    def __init__(self, net: LSCodec):
        super().__init__()
        self.dec = net.decoder
        self.q = net.quantizer

    def forward(self, h, mask=None):
        if mask is None:
            return self.dec(h)
        return self.dec(h) * mask


def analysis_flops(sum_lns):
    result = {
        'flops': None,
        'param': None,
    }
    for ln in sum_lns:
        if 'Total mult-adds' in ln:
            m = re.search('Total mult-adds \(G\):\\s+([.\\d]+)', ln)
            if m:
                v = float(m.group(1))
                macs = v * (10 ** 9)  # Multiply-Accumulates
                flops = macs * 2
                flops = flops / (10 ** 9)
                result['flops'] = flops
        if 'Params size (MB):' in ln:
            m = re.search('Params size \(MB\):\\s+([.\\d]+)', ln)
            if m:
                v = float(m.group(1))
                result['param'] = v

    return result


@th.no_grad()
def eval_flops(net: LSCodec, x, fps=30.0):
    x_shp = x.shape

    ######################
    # 1. total flops
    summary_result = summary(net, input_size=x_shp, verbose=0)
    total_lines = str(summary_result).split('\n')
    total_result = analysis_flops(total_lines)
    total_flop = total_result['flops']
    total_param = total_result['param']

    ######################
    # 2. encoding flops
    sub_enc = Enc(net=net)
    _ = sub_enc.eval()
    z_enc = sub_enc(x.to('cuda'))
    z_shp = z_enc.shape

    summary_enc = summary(sub_enc, input_size=x_shp, verbose=0)
    total_lines = str(summary_enc).split('\n')
    enc_result = analysis_flops(total_lines)
    enc_flop = enc_result['flops']
    enc_param = enc_result['param']

    ######################
    # 3. decoding flops
    sub_dec = Dec(net=net)
    _ = sub_dec.eval()

    summary_dec = summary(sub_dec, input_size=z_shp, verbose=0)
    total_lines = str(summary_dec).split('\n')
    dec_result = analysis_flops(total_lines)
    dec_flop = dec_result['flops']
    dec_param = dec_result['param']

    ######################
    # 4. rtf checking
    raw_duration = x.shape[1] / fps
    raw_repeat = 5
    rtf_encs, rtf_decs = [], []
    for ri in range(raw_repeat):
        start_time = time.time()
        z_idx = net.encode_idx(x.to('cuda'), b_split=False)
        rtf_enc = (time.time() - start_time) / raw_duration
        if ri > 0:
            rtf_encs.append(rtf_enc)

        start_time = time.time()
        y = net.decode_idx(z_idx)
        rtf_dec = (time.time() - start_time) / raw_duration
        if ri > 0:
            rtf_decs.append(rtf_dec)
    rtf_enc = sum(rtf_encs) / len(rtf_encs)
    rtf_dec = sum(rtf_decs) / len(rtf_decs)

    result = {
        'all_flops': total_flop, 'enc_flops': enc_flop, 'dec_flops': dec_flop,
        'all_param': total_param, 'enc_param': enc_param, 'dec_param': dec_param,
        'enc_rtf': round(1.0 / rtf_enc, 2), 'dec_rtf': round(1.0 / rtf_dec, 2)
    }
    return result


def load_model(conf_path):
    with open(conf_path, 'r') as fin:
        conf = toml.load(fin)
        fin.close()
    model_conf = conf['LSCodec']
    net = LSCodec(**model_conf)
    _ = net.eval()
    return net, model_conf


def main():
    conf_path = r'../config'
    conf_list = []
    for root, _, fns in os.walk(conf_path):
        for fn in fns:
            if not fn.lower().startswith('config_lscodec_smap_'):
                continue
            conf = os.path.join(root, fn)
            conf_list.append(conf)

    batch = 4
    nseq = 320
    repeat = 10

    for conf_path in conf_list:
        conf_name = os.path.basename(conf_path).replace('.toml', '')
        net, model_conf = load_model(conf_path)
        x = th.randn(size=(batch, model_conf['n_feat'], nseq))
        results = {}
        for r in range(repeat):
            result = eval_flops(net, x, fps=60.0)
            for ky, vl in result.items():
                if ky not in results.keys():
                    results[ky] = []
                results[ky].append(vl)
        for ky, vl in results.items():
            results[ky] = round(sum(vl) / len(vl), 2)
        print("conf : {}, result : {}".format(conf_name, results))


if __name__ == '__main__':
    main()

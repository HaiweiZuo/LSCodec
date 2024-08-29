import os
import random
from typing import Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


def exponential_moving_average(data, alpha):
    """
    EMA:
    param:
    data : numpy array as time series
    alpha : float between (0, 1). close to 1 means focus on local info

    returns:
    ema : numpy array
    """
    ema = np.zeros_like(data)
    ema[0] = data[0]

    for i in range(1, len(data)):
        ema[i] = alpha * data[i] + (1 - alpha) * ema[i - 1]
    return ema


class DatasetSmall(Dataset):
    def __init__(self, data_root: str, data_mode: str = 'train', data_range: Tuple[int, int] = (64, 320), data_alpha: float = 0.1):
        assert data_mode in ['train', 'test']

        self.data_root = data_root
        self.data_mode = data_mode
        self.data_range = data_range
        self.data_alpha = data_alpha

        self.data = None
        self.data_ema = None
        self.data_flc = None
        self.label = None
        if not self.load():
            raise RuntimeError

    def load(self):
        try:
            data_path = os.path.join(self.data_root, "{}.npy".format(self.data_mode))
            data = np.load(data_path)

            label_path = os.path.join(self.data_root, "{}.npy".format("labels"))
            label = np.load(label_path)
        except Exception as e:
            self.data = None
            self.label = None
            self.data_ema = None
            self.data_flc = None
            print("error load data, cause : {}".format(e))
            return False

        self.data = data
        self.label = label
        self.data_ema = exponential_moving_average(data, self.data_alpha)
        self.data_flc = data - self.data_ema
        return True

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        str_idx = idx
        end_idx = str_idx + random.randint(a=self.data_range[0], b=self.data_range[1])

        if end_idx > self.data.shape[0]:
            res_idx = end_idx - self.data.shape[0]
            str_idx = str_idx - res_idx
            end_idx = end_idx - res_idx

        tseq = torch.from_numpy(self.data[str_idx: end_idx, :]).to(torch.float32)
        cond = {
            "len": end_idx - str_idx,
            "avg": torch.from_numpy(self.data_ema[str_idx: end_idx, :]).to(torch.float32),
            "flc": torch.from_numpy(self.data_flc[str_idx: end_idx, :]).to(torch.float32),
        }
        return tseq, cond

    def get_data_scale(self):
        nlen, ndim = self.data.shape[0], self.data.shape[1]
        nano = (np.sum(self.label) / (self.label.shape[0] * self.label.shape[1]))
        return (nlen, ndim, nano)


class DatasetLarge(Dataset):
    def __init__(self, data_root: str, data_mode: str = 'train', data_range: Tuple[int, int] = (224, 320), data_alpha: float = 0.1, data_overlap: float = 0.4):
        assert data_mode in ['train', 'test']

        self.data_root = data_root
        self.data_mode = data_mode
        self.data_range = data_range
        self.data_alpha = data_alpha
        self.data_overlap = data_overlap

        self.data_list = []
        self.data_index = []
        if not self.load():
            raise RuntimeError

    def load(self):
        data_list = []
        data_index = []
        data_length = 0
        data_labels_pos = 0
        data_labels_neg = 0
        for root, _, fns in os.walk(self.data_root):
            for fn in fns:
                if not fn.lower().endswith('_{}.npy'.format('train')):
                    continue
                ####################
                # data loading
                lfn = fn.replace('_{}.npy'.format('train'), '_labels.npy')
                data_path = os.path.join(root, fn)
                data = np.load(data_path)
                data_length += data.shape[0]
                if 1 == len(data.shape):
                    data = data[:, np.newaxis]

                label_path = os.path.join(root, lfn)
                if not os.path.exists(label_path):
                    label_path = label_path.replace('_labels.npy', '_label.npy')

                label = np.load(label_path)
                n_neg = int(np.sum(label))
                data_labels_pos += (label.shape[0] - n_neg)
                data_labels_neg += n_neg

                data_ema = exponential_moving_average(data, self.data_alpha)
                data_flc = data - data_ema
                data_len = data.shape[0]
                data_list.append({
                    "data": data,
                    "label": label,
                    "avg": data_ema,
                    "flc": data_flc,
                    "len": data_len
                })

                ####################
                # data indexing
                hop_length = int(self.data_range[0] * (1 - self.data_overlap))
                hop_block = int(data_len // hop_length)
                hop_id = len(data_list) - 1
                hop_index = [hop_length * i for i in range(hop_block)]
                for hopi in hop_index:
                    data_index.append([hop_id, hopi])

        self.data_list = data_list
        self.data_index = data_index
        return 0 < len(self.data_index)

    def __len__(self):
        return len(self.data_index)

    def __getitem__(self, idx):
        hop_id, hopi = self.data_index[idx]
        data_block: dict = self.data_list[hop_id]
        str_idx = hopi
        end_idx = str_idx + random.randint(a=self.data_range[0], b=self.data_range[1])

        data = data_block['data']
        data_ema = data_block['avg']
        data_flc = data_block['flc']

        if end_idx > data.shape[0]:
            res_idx = end_idx - data.shape[0]
            str_idx = str_idx - res_idx
            end_idx = end_idx - res_idx

        tseq = torch.from_numpy(data[str_idx: end_idx, :]).to(torch.float32)
        cond = {
            "len": end_idx - str_idx,
            "avg": torch.from_numpy(data_ema[str_idx: end_idx, :]).to(torch.float32),
            "flc": torch.from_numpy(data_flc[str_idx: end_idx, :]).to(torch.float32),
        }
        return tseq, cond

    def get_data_scale(self):
        nlen, ndim = 0, 0
        nano = 0.0
        for dat in self.data_list:
            ilen, idim = dat['data'].shape[0], dat['data'].shape[1]
            nlen += ilen
            ndim = idim
            nano += np.sum(dat['label']) / (dat['label'].shape[0] * dat['label'].shape[1])
        nano = nano / len(self.data_list)
        return (nlen, ndim, nano)


class Collate:
    def __init__(self, align: int = -1, keep_len_last: bool = True):
        self.align = align
        self.keep_len_last = keep_len_last

    def batch_2_tensor(self, batch, tdim=0):
        dims = batch[0].dim()
        max_size = [max([b.size(i) for b in batch]) for i in range(dims)]
        if self.align > 0:
            res_size = max_size[tdim] % self.align
            max_size[tdim] += (self.align - res_size) if res_size > 0 else 0

        size = (len(batch),) + tuple(max_size)
        canvas = torch.zeros(size=size, dtype=batch[0].dtype)  # batch[0].new_zeros(size=size)

        for i, b in enumerate(batch):
            sub_tensor = canvas[i]
            for d in range(dims):
                sub_tensor = sub_tensor.narrow(d, 0, b.size(d))
            sub_tensor.add_(b)
        return canvas

    def adapt(self, batch):
        items = []
        for bat in batch:
            tseq, cond = bat

            block = {
                "inp": tseq,
                "len": cond["len"],
                "avg": cond["avg"],
                "flc": cond["flc"],
            }
            items.append(block)
        return items

    def collate(self, batch):
        notnone_batches = [b for b in batch if b is not None]

        ###########################################################
        # to batch
        b_tseq = [b["inp"] for b in notnone_batches]
        b_len = [b["len"] for b in notnone_batches]
        b_avg = [b["avg"] for b in notnone_batches]
        b_flc = [b["flc"] for b in notnone_batches]

        ###########################################################
        # to tensor
        t_tseq = self.batch_2_tensor(b_tseq, tdim=0)
        t_mask = torch.ones(size=(t_tseq.shape[0], t_tseq.shape[1], 1), dtype=torch.float32, device=t_tseq.device)
        for ti, tl in enumerate(b_len):
            t_mask[ti, tl:, :] = 0.0

        t_len = torch.as_tensor(b_len)
        t_avg = self.batch_2_tensor(b_avg, tdim=0)
        t_flc = self.batch_2_tensor(b_flc, tdim=0)

        if self.keep_len_last:
            t_tseq = torch.transpose(t_tseq, 1, 2)
            t_mask = torch.transpose(t_mask, 1, 2)
            t_avg = torch.transpose(t_avg, 1, 2)
            t_flc = torch.transpose(t_flc, 1, 2)

        tseq = t_tseq
        condition = {
            "avg": t_avg,
            "flc": t_flc,
            "len": t_len,
            "mask": t_mask
        }
        return tseq, condition

    def __call__(self, batch):
        adapted_batch = self.adapt(batch)
        return self.collate(adapted_batch)


def main_small():
    data_root = r'D:\workspace\project_tcn\proj_dtaad\processed\MBA'
    dss = DatasetSmall(data_root=data_root, data_mode='train')
    print("data size :", len(dss))
    for seq, cond in dss:
        print("seq {}, cond : {}".format(seq.shape, cond.keys()))
        break

    dls = DataLoader(dataset=dss, batch_size=4, shuffle=False, collate_fn=Collate(align=8, keep_len_last=True))
    for batch in dls:
        tseq, cond = batch
        print("tseq :", tseq.shape, "cond :", cond.keys())
        break


def main_large():
    data_root = r'D:\workspace\project_tcn\proj_dtaad\processed\MSL'
    dss = DatasetLarge(data_root=data_root, data_mode='train', data_overlap=0.5)
    print("data large size :", len(dss))

    dls = DataLoader(dataset=dss, batch_size=4, shuffle=False, collate_fn=Collate(align=8, keep_len_last=True))
    for batch in dls:
        tseq, cond = batch
        print("tseq :", tseq.shape, "cond :", cond.keys())
        break


def test_dataset_scale():
    from collections import OrderedDict
    ds_path_mba = r'../processed/MBA'
    ds_path_msl = r'../processed/MSL'
    ds_path_smap = r'../processed/SMAP'
    ds_path_smd = r'../processed/SMD'
    ds_path_synthetic = r'../processed/synthetic'

    def stat(dpth: str, is_larger: bool):
        if is_larger:
            ds_train = DatasetLarge(data_root=dpth, data_mode='train')
            ds_test = DatasetLarge(data_root=dpth, data_mode='test')
        else:
            ds_train = DatasetSmall(data_root=dpth, data_mode='train')
            ds_test = DatasetSmall(data_root=dpth, data_mode='test')
        train_n_len, train_n_dim, train_a_ratio = ds_train.get_data_scale()
        test_n_len, _, _ = ds_test.get_data_scale()
        return (train_n_len, test_n_len, train_n_dim, train_a_ratio)

    dict_state = OrderedDict()
    dict_state["MBA"] = stat(ds_path_mba, False)
    dict_state["MSL"] = stat(ds_path_msl, True)
    dict_state["SMAP"] = stat(ds_path_smap, True)
    dict_state["SMD"] = stat(ds_path_smd, True)
    dict_state["SYNTHETIC"] = stat(ds_path_synthetic, False)

    for ky, vl in dict_state.items():
        print("{}  -->  {}".format(ky, vl))





def main():
    b_small = False
    b_large = False
    b_state = True

    if b_small:
        main_small()

    if b_large:
        main_large()

    if b_state:
        test_dataset_scale()


if __name__ == '__main__':
    main()

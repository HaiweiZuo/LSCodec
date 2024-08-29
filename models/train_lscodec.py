import os
import time
from typing import Tuple

import tqdm
import numpy as np
import torch
from accelerate import Accelerator

from models.lscodec import LSCodec
from models.balancer import Balancer
from datasets.dataset_ts1 import DatasetSmall, DatasetLarge, Collate, DataLoader
from utils.trainer import ITrainer
from utils.tools import IModel
from utils.optim import Noam_Scheduler


class Trainer(ITrainer):
    def __init__(self, conf, accl: Accelerator):
        super().__init__(conf, accl)
        self.log(fn='__init__', msg="[Train] Init trainer. Device: {}".format(self.accelerator.device))

        self.batch_size = self.config.Train.batch_size
        self.valid_size = self.config.Train.valid_size
        self.dataset_scale = self.config.Data.data_scale
        self.dataset_stride = self.config.LSCodec.strides

        #####################################
        # build dataset
        result = self.build_dataset()
        self.train_ds = result["train_ds"]
        self.valid_ds = result["valid_ds"]
        self.train_dl = self.accelerator.prepare_data_loader(result["train_dl"])
        self.valid_dl = self.accelerator.prepare_data_loader(result["valid_dl"])
        self.log(fn='__init__', msg="[Train] Data size : {}, Data batch : {} | [Valid] Data size : {}, Data batch : {}. Batch size {}. Device {}".format(
            len(self.train_ds), len(self.train_dl), len(self.valid_ds), len(self.valid_dl), self.batch_size, self.accelerator.device
        ))

        #####################################
        # build lscodec
        net = self.build_net()
        self.model = self.accelerator.prepare_model(net)

        #####################################
        # build optim
        self.lr = self.config.Train.lr
        self.weight_decay = self.config.Train.weight_decay
        self.enable_schedule = self.config.Train.enable_schedule
        self.grad_norm = self.config.Train.grad_norm

        optim, scheduler = self.build_optimize(use_schedule=self.enable_schedule)
        self.optim = optim
        self.scheduler = scheduler
        self.progress = self.config.Train.progress

        #####################################
        # build loss
        self.lambda_rec = self.config.Loss.lambda_rec
        self.lambda_rec_g = self.config.Loss.lambda_rec_g
        self.lambda_rec_l = self.config.Loss.lambda_rec_l
        self.lambda_commit = self.config.Loss.lambda_commit

        self.balancer = Balancer(weights={"all": self.lambda_rec,  "g": self.lambda_rec_g, "l": self.lambda_rec_l}, rescale_grads=True, total_norm=1.)
        self.enable_balancer = self.config.Loss.get("enable_balancer", False)

        ######################################################################
        # resume setting
        if self.resume:
            max_ep, max_it = self._find_lastest()
            self.cur_epoch = max_ep
            self.cur_iters = max_it
        self.log(fn='__init__', msg="end construction.")

    def build_dataset(self):
        data_root = self.config.Data.data_root
        data_alpha = self.config.Data.data_alpha
        data_overlap = self.config.Data.data_overlap
        data_range: Tuple[int, int] = tuple(self.config.Data.data_range)
        num_workers = self.config.Data.num_work

        align = 1
        for it in self.dataset_stride:
            align *= it

        if self.dataset_scale == 'small':
            trn_ds = DatasetSmall(data_root=data_root, data_mode='train', data_range=data_range, data_alpha=data_alpha)
            val_ds = DatasetSmall(data_root=data_root, data_mode='test', data_range=data_range, data_alpha=data_alpha)
            trn_dl = DataLoader(dataset=trn_ds, batch_size=self.batch_size, shuffle=True, collate_fn=Collate(align=align, keep_len_last=True), num_workers=num_workers)
            val_dl = DataLoader(dataset=val_ds, batch_size=1, shuffle=False, collate_fn=Collate(align=align, keep_len_last=True), num_workers=num_workers)
        elif self.dataset_scale == 'large':
            trn_ds = DatasetLarge(data_root=data_root, data_mode='train', data_range=data_range, data_alpha=data_alpha, data_overlap=data_overlap)
            val_ds = DatasetLarge(data_root=data_root, data_mode='test', data_range=data_range, data_alpha=data_alpha, data_overlap=data_overlap)
            trn_dl = DataLoader(dataset=trn_ds, batch_size=self.batch_size, shuffle=True, collate_fn=Collate(align=align, keep_len_last=True), num_workers=num_workers)
            val_dl = DataLoader(dataset=val_ds, batch_size=1, shuffle=False, collate_fn=Collate(align=align, keep_len_last=True), num_workers=num_workers)
        else:
            raise NotImplementedError
        return {"train_ds": trn_ds, "train_dl": trn_dl, "valid_ds": val_ds, "valid_dl": val_dl}

    def build_net(self):
        net = LSCodec(
            n_feat=self.config.LSCodec.n_feat,
            n_latent=self.config.LSCodec.n_latent,
            strides=list(self.config.LSCodec.strides),
            n_kernel=self.config.LSCodec.n_kernel,
            n_layer=self.config.LSCodec.n_layer,
            n_lstm=self.config.LSCodec.n_lstm,
            n_cycle=self.config.LSCodec.n_cycle,
            dropout=self.config.LSCodec.dropout
        )
        net_size = IModel.calc_model_size(net)
        self.log(fn='build_net', msg="net size {:.2f} mb".format(net_size))
        return net

    def build_optimize(self, use_schedule=False):
        optim = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        scheduler = None
        if use_schedule:
            scheduler = Noam_Scheduler(optimizer=optim, warmup_steps=self.config.Train.warmup_steps)
        optim = self.accelerator.prepare_optimizer(optim)
        if use_schedule:
            scheduler = self.accelerator.prepare_scheduler(scheduler)
        return optim, scheduler

    def main(self):
        self.log(fn='main', msg="start train main loop ...")
        if self.cur_epoch < 0:
            self.cur_epoch = 0
            self.cur_iters = 0

        while (self.cur_epoch < self.max_epoch) and (self.cur_iters < self.max_iters):
            bar = tqdm.tqdm(self.train_dl)
            vlr = self.lr if self.scheduler is None else self.scheduler.get_last_lr()[0]

            _ = self.model.train()
            for idx, batch in enumerate(bar):
                tseq, cond = batch
                loss, desc_loss = self.train_batch(tseq, cond, b_detail_loss=True)

                dlos = ["{}: {:.4f}".format(ky, vl) for ky, vl in desc_loss.items()]
                desc = 'Train Epoch [{}]/[{}]-[iteration: {}]. loss : {:.6f}. lr : {:.6f}'.format(
                    self.cur_epoch + 1, self.max_epoch + 1, self.cur_iters + 1, loss, vlr
                )
                desc += "[detail loss :"
                for dls in dlos:
                    desc += dls + " "
                desc += "]"
                mem_cost = torch.cuda.memory_reserved() / 1E9
                mem_info = "[GPU Memo: {:.2f}mb]".format(mem_cost)
                desc += mem_info
                bar.set_description(desc=desc)
                self.cur_iters += 1

                if self.cur_iters > 0 and (self.cur_iters % self.print_every_iter == 0):
                    self.log(fn="main", msg=desc)

                if self.cur_iters > 0 and (self.cur_iters % self.save_every_iter == 0):
                    self.save_state_checkpoint(eid=self.cur_epoch, iid=self.cur_iters)
                    self.log(fn="main", msg="save checkpoint.{}.{}".format(self.cur_epoch, self.cur_iters))

            self.cur_epoch += 1
            if self.scheduler is not None:
                self.scheduler.step()

            _ = self.model.eval()
            with torch.no_grad():
                bar = tqdm.tqdm(self.valid_dl)
                avg_succ = True
                avg_loss = []
                avg_time = []
                eval_pth = os.path.join(self.run_folder, "eval_{}".format(self.cur_epoch))
                os.makedirs(eval_pth, exist_ok=True)
                max_valid_size = self.valid_size
                try:
                    for idx, batch in enumerate(bar):

                        tseq, cond = batch
                        val_loss, sample, val_time = self.test_batch(tseq, cond)
                        val_loss = val_loss.cpu().item()
                        avg_loss.append(val_loss)
                        avg_time.append(val_time)
                        bar.set_description(desc="val [{}/{}] rec loss {:.4f}, time : {:.4f}".format(self.cur_epoch, self.max_epoch, val_loss, val_time))

                        np.save(os.path.join(eval_pth, 'seq.gt.{}.npy'.format(idx)), np.transpose(tseq[0, :, :].cpu().numpy(), (1, 0)))
                        np.save(os.path.join(eval_pth, 'seq.pred.{}.npy'.format(idx)), np.transpose(sample['output'], (1, 0)))

                        np.save(os.path.join(eval_pth, 'seq.gt_g.{}.npy'.format(idx)), np.transpose(sample['gt_g'], (1, 0)))
                        np.save(os.path.join(eval_pth, 'seq.pred_g.{}.npy'.format(idx)), np.transpose(sample['pred_g'], (1, 0)))

                        np.save(os.path.join(eval_pth, 'seq.gt_l.{}.npy'.format(idx)), np.transpose(sample['gt_l'], (1, 0)))
                        np.save(os.path.join(eval_pth, 'seq.pred_l.{}.npy'.format(idx)), np.transpose(sample['pred_l'], (1, 0)))

                        if idx > max_valid_size:
                            break

                except Exception as e:
                    avg_succ = False
                    self.log(fn="main", msg="validating in {}, error cause {}".format(self.cur_epoch, e))
                if avg_succ:
                    self.log(fn="main",
                             msg="validating in {}, average reconstruction loss : {:.4f}, average time : {}".format(self.cur_epoch, np.mean(avg_loss), np.mean(avg_time)))

        net = self.accelerator.unwrap_model(self.model)
        _ = net.eval()
        _ = net.cpu()
        torch.save(net.state_dict(), os.path.join(self.run_folder, "checkpoint.pt"))
        self.log(fn='main', msg="training succ!")
        return True

    def l2loss_with_mask(self, pred, gt, mask):
        rec_loss = ((pred - gt) ** 2) * mask
        rec_loss = torch.sum(rec_loss) / torch.sum(mask)
        return rec_loss

    def train_batch(self, tseq, cond, b_detail_loss=True):

        self.optim.zero_grad()
        t_avg = cond['avg']
        t_flc = cond['flc']
        mask = cond['mask']
        if self.enable_balancer:
            tseq.requires_grad = True

        output, output_g, output_l, commit_losses = self.model(tseq, mask)
        commit_losses = torch.mean(commit_losses)

        rec_loss = self.l2loss_with_mask(output, tseq, mask)
        rec_loss_g = self.l2loss_with_mask(output_g, t_avg, mask)  # ((output_g - t_avg) ** 2) * mask
        rec_loss_l = self.l2loss_with_mask(output_l, t_flc, mask)  # ((output_l - t_flc) ** 2) * mask
        g_loss_final = (self.lambda_rec * rec_loss +
                        self.lambda_rec_g * rec_loss_g +
                        self.lambda_rec_l * rec_loss_l +
                        self.lambda_commit * commit_losses)

        if self.enable_balancer:
            loss_group = {"all": rec_loss, "g": rec_loss_g, "l": rec_loss_l}
            self.balancer.backward(loss_group, tseq)

        self.accelerator.backward(g_loss_final)
        if self.grad_norm > 0.0:
            self.accelerator.clip_grad_norm_(self.model.parameters(), max_norm=self.grad_norm, norm_type=2)
        self.optim.step()
        desc = {'loss': g_loss_final.cpu().item()}
        if b_detail_loss:
            desc['rec_loss'] = rec_loss.cpu().item()
            desc['rec_loss_g'] = rec_loss_g.cpu().item()
            desc['rec_loss_l'] = rec_loss_l.cpu().item()
            desc['commit'] = commit_losses.cpu().item()

        return g_loss_final.cpu().item(), desc

    def test_batch(self, tseq, cond):
        t_avg = cond['avg']
        t_flc = cond['flc']
        mask = cond['mask']

        t1 = time.time()
        output, output_g, output_l, _ = self.model(tseq, mask)
        rec_loss = self.l2loss_with_mask(output, tseq, mask)
        rec_loss_g = self.l2loss_with_mask(output_g, t_avg, mask)  # ((output_g - t_avg) ** 2) * mask
        rec_loss_l = self.l2loss_with_mask(output_l, t_flc, mask)  # ((output_l - t_flc) ** 2) * mask
        g_loss_final = (self.lambda_rec * rec_loss +
                        self.lambda_rec_g * rec_loss_g +
                        self.lambda_rec_l * rec_loss_l)
        t2 = time.time()

        sample = {
            "output": output[0, :, :].cpu().numpy(),
            "pred_g": output_g[0, :, :].cpu().numpy(),
            "pred_l": output_l[0, :, :].cpu().numpy(),
            "gt_g": t_avg[0, :, :].cpu().numpy(),
            "gt_l": t_flc[0, :, :].cpu().numpy(),
        }
        return g_loss_final, sample, t2 - t1

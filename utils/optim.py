import torch
import torch.optim as opt
from torch.optim.lr_scheduler import _LRScheduler as lrscheduler


def sigmoid_schedule(t, start=-3.0, end=3, tau=1.0, clip_min=1e-4):
    # A gamma function based on sigmoid function.
    start = torch.tensor(start)
    end = torch.tensor(end)
    tau = torch.tensor(tau)

    v_start = (start / tau).sigmoid()
    v_end = (end / tau).sigmoid()
    output = ((t * (end - start) + start) / tau).sigmoid()
    output = (v_end - output) / (v_end - v_start)

    return output.clamp(clip_min, 1.)


class Noam_Scheduler(lrscheduler):
    # https://github.com/tugstugi/pytorch-saltnet/blob/master/utils/lr_scheduler.py
    def __init__(self, optimizer, warmup_steps):
        self.warmup_steps = warmup_steps
        super(Noam_Scheduler, self).__init__(optimizer)

    def get_lr(self):
        last_epoch = max(1, self.last_epoch)

        scale = self.warmup_steps ** 0.5 * min(last_epoch ** (-0.5), last_epoch * self.warmup_steps ** -1.5)
        return [base_lr * scale for base_lr in self.base_lrs]


class Modified_Noam_Scheduler(lrscheduler):
    def __init__(self, optimizer, base):
        self.base = base
        super(Modified_Noam_Scheduler, self).__init__(optimizer)

    def get_lr(self):
        last_epoch = max(1, self.last_epoch)

        scale = self.base ** 0.5 * (last_epoch + self.base) ** (-0.5)
        return [base_lr * scale for base_lr in self.base_lrs]


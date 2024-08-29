import os
import re

from accelerate import Accelerator
from utils.tools import IToml, ILogger


class ITrainer:
    def __init__(self, config, accl: Accelerator):
        ##############################
        # config
        self.config = IToml(config=config).config if isinstance(config, str) else config
        self.accelerator = accl
        self.device = self.accelerator.device

        ##############################
        # folder
        self.run_name: str = self.config.Basic.run_name
        self.run_folder: str = os.path.join(self.config.Basic.run_folder, self.run_name)
        os.makedirs(self.run_folder, exist_ok=True)
        self.logger = ILogger(log_id=self.run_name, log_path=self.run_folder)
        self.log(fn='__init__', msg="Start trainer-pitch-{}, run name : {}, run folder : {}".format(self.device, self.run_name, self.run_folder))

        ##############################
        # state
        self.no_adv_epoch = self.config.Train.get("no_adv_epoch", 9999)
        self.max_iters = self.config.Train.max_iters
        self.max_epoch = self.config.Train.max_epoch
        self.save_every_iter = self.config.Train.save_every_iter
        self.print_every_iter = self.config.Train.print_every_iter
        self.cur_epoch = -1
        self.cur_iters = -1
        self.resume = self.config.Train.resume

    def save_state_checkpoint(self, eid, iid):
        resume_folder = os.path.join(self.run_folder, "ckpt.{}.{}".format(eid, iid))
        self.accelerator.save_state(resume_folder, safe_serialization=False)

    def load_state_checkpoint(self, eid, iid):
        resume_folder = os.path.join(self.run_folder, "ckpt.{}.{}".format(eid, iid))
        self.accelerator.load_state(input_dir=resume_folder)

    def _find_lastest(self):
        max_ep, max_it = -1, -1
        for root, dirs, _ in os.walk(self.run_folder):
            for dr in dirs:
                m = re.search("ckpt.(\\d+)\\.(\\d+)", dr)
                if not m:
                    continue
                ep, it = int(m.group(1)), int(m.group(2))
                max_ep = max(max_ep, ep)
                max_it = max(max_it, it)
        if (max_ep >= 0) and (max_it >= 0):
            try:
                self.load_state_checkpoint(eid=max_ep, iid=max_it)
                self.log(fn='__init__', msg="resume succ from checkpoint ckpt.{}.{}".format(max_ep, max_it))
                return max_ep, max_it
            except Exception as e:
                self.log(fn='__init__', msg="resume fail from checkpoint ckpt.{}.{}, cause {}".format(max_ep, max_it, e))
        return -1, -1

    def log(self, fn, msg, level='info', use_main: bool = True):
        if use_main:
            if self.accelerator.is_main_process:
                self.logger.log(func=self.loc(fn), info=msg, level=level)
        else:
            self.logger.log(func=self.loc(fn), info=msg, level=level)

    def loc(self, fn):
        return "[{}][{}]".format(self.run_name, fn)

    def get_run_id(self):
        return self.run_name

    def get_run_folder(self):
        return self.run_folder

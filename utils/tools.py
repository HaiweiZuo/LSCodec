import toml
import munch
from munch import Munch

import os
import logging
from logging.handlers import RotatingFileHandler
from typing import Dict


class IToml:
    def __init__(self, config, use_dict: bool = False):
        try:
            fin = open(config, 'r')
            conf: dict = toml.load(fin)
            fin.close()
        except Exception as e:
            raise RuntimeError("Toml File Parser Failed, Cause: {}".format(e))
        self.config = self.recursive_munch(conf)
        if use_dict:
            self.config = munch.unmunchify(self.config)

    def empty(self):
        return 0 == len(self.config)

    def recursive_munch(self, d):
        if isinstance(d, dict):
            return Munch((k, self.recursive_munch(v)) for k, v in d.items())
        elif isinstance(d, list):
            return [self.recursive_munch(v) for v in d]
        else:
            return d


class ILogger:
    class _LoggerInfo:
        def __init__(self):
            self.id: str = ""
            self.path: str = ""
            self.level: int = 0
            self.dump_to_file: bool = False
            self.dump_to_file_r: bool = False
            self.dump_to_console: bool = False
            self.log_object = None

        def setter(self, logid, logpath, level, b_console: bool = True, b_file: bool = True, b_rotate: bool = True):
            self.id = logid
            self.path = logpath
            self.level = level
            self.dump_to_console = b_console
            self.dump_to_file = b_file
            self.dump_to_file_r = b_file and b_rotate

        def setlogger(self, logger):
            self.log_object = logger

    def __init__(self, log_id="default", log_path=None):
        self.maxbites = 10  # unit Mb
        self.maxcount = 5
        self.formater = logging.Formatter(
            fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        self.log_pool: Dict[str, ILogger._LoggerInfo] = {}
        self.gen_logger(logid=log_id, logpath=log_path)

        self.curren_log_id = log_id
        self.curren_log_obj = self.log_pool[log_id].log_object
        self.curren_log_map = {"info": self.curren_log_obj.info,
                               "error": self.curren_log_obj.error,
                               "warning": self.curren_log_obj.warning,
                               "fatal": self.curren_log_obj.fatal,
                               "debug": self.curren_log_obj.debug}
        self.force_silence: bool = False

    def gen_logger(self, logid: str, logpath: str, level: int = logging.INFO, b_console: bool = True, b_file: bool = True, b_filerotate: bool = True) -> bool:
        if logid in self.log_pool.keys():
            self.log(func=self.loc("gen_logger"), info="log id : {} conflict".format(logid), level='warning')
            return False
        if logpath is None:
            b_file = False
        else:
            os.makedirs(logpath, exist_ok=True)

        extLogger = logging.getLogger(name=logid)
        extInfo = ILogger._LoggerInfo()
        extInfo.setter(logid, logpath, level=level, b_console=b_console, b_file=b_file, b_rotate=b_filerotate)

        if b_file:
            if not b_filerotate:
                hnd_file = logging.FileHandler(filename=os.path.join(logpath, "{}.log".format(logid)), encoding='utf-8')
                hnd_file.setFormatter(self.formater)
                hnd_file.setLevel(level)
                extLogger.addHandler(hnd_file)
            else:
                hnd_rotf = RotatingFileHandler(filename=os.path.join(logpath, "{}.log".format(logid)),
                                               maxBytes=self.maxbites * 1024 * 1024,
                                               backupCount=self.maxcount,
                                               encoding='utf-8')
                hnd_rotf.setFormatter(self.formater)
                hnd_rotf.setLevel(level)
                extLogger.addHandler(hnd_rotf)

        if b_console:
            hnd_stream = logging.StreamHandler()
            hnd_stream.setFormatter(self.formater)
            hnd_stream.setLevel(level)
            extLogger.addHandler(hnd_stream)

        extLogger.setLevel(level)
        extInfo.setlogger(extLogger)
        self.log_pool[logid] = extInfo
        return True

    def add_logger(self, logid, logobject, set_current: bool = True) -> bool:
        if logid in self.log_pool.keys():
            self.log(func=self.loc("add_logger"), level='debug', info="add logger failed, cause id {} conflict".format(logid))
            return False
        extInfo = ILogger._LoggerInfo()
        extInfo.setter(logid, None, level=-1, b_console=False, b_file=False, b_rotate=False)
        extInfo.setlogger(logobject)
        self.log_pool[logid] = extInfo
        if set_current:
            self.set_logger(logid)
        return True

    def set_logger(self, log_id):
        if log_id == self.curren_log_id:
            return
        if log_id in self.log_pool.keys():
            self.curren_log_id = log_id
            self.curren_log_obj = self.log_pool[log_id].log_object
        else:
            self.gen_logger(logid=log_id, logpath="", b_console=True, b_file=False, b_filerotate=False)
            self.curren_log_id = log_id
            self.curren_log_obj = self.log_pool[log_id].log_object
        self.curren_log_map = {"info": self.curren_log_obj.info,
                               "error": self.curren_log_obj.error,
                               "warning": self.curren_log_obj.warning,
                               "fatal": self.curren_log_obj.fatal,
                               "debug": self.curren_log_obj.debug}

    def set_force_silence(self, flag: bool):
        self.force_silence = flag

    def log(self, func: str, info: str, level: str = "info"):
        if self.force_silence:
            return

        if level in self.curren_log_map.keys():
            self.curren_log_map[level](msg="{} ---> {}".format(func, info))
        else:
            self.curren_log_map['info'](msg="{} ---> {}".format(func, info))


class IModel:

    @staticmethod
    def calc_model_size(net):
        total_trainable_params_size = sum(p.numel() * p.element_size() for p in net.parameters() if p.requires_grad)
        net_size_in_mb = total_trainable_params_size / (1024 ** 2)
        return net_size_in_mb



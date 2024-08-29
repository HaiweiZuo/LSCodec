from argparse import ArgumentParser
from accelerate import Accelerator, DistributedDataParallelKwargs

from models.train_lscodec import Trainer as LSCodecTrainer


def task_train_lscodec(args, accl):
    conf = args.config
    trainer = LSCodecTrainer(conf, accl)
    ret = trainer.main()
    print("training motion representation {}".format("succ" if ret else "fail"))


def main(args):
    ununsed = args.force_unused
    kwgs = DistributedDataParallelKwargs(find_unused_parameters=ununsed)

    if args.force_cpu:
        accl = Accelerator(cpu=True, kwargs_handlers=[kwgs])
    else:
        accl = Accelerator(kwargs_handlers=[kwgs])

    if args.task == 'lscodec':
        task_train_lscodec(args, accl)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--task", choices=['lscodec'])

    parser.add_argument("--config", type=str, default="./config/config_lscodec_kpi.toml")
    parser.add_argument("--force_cpu", action='store_true')
    parser.add_argument("--force_unused", action='store_true')

    args = parser.parse_args()
    main(args)

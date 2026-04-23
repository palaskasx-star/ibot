import argparse
import os
import os.path as osp

from mmengine.config import Config, DictAction
from mmengine.runner import Runner
from mmdet.utils import register_all_modules

def parse_args():
    parser = argparse.ArgumentParser(description='Train a detector')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--work-dir', help='the dir to save logs and models')
    parser.add_argument(
        '--amp',
        action='store_true',
        default=False,
        help='enable automatic mixed precision training')
    parser.add_argument(
        '--auto-scale-lr',
        action='store_true',
        help='enable automatically scaling LR based on batch size')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    # When using PyTorch launcher, local_rank is provided by the environment
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    return args

def main():
    args = parse_args()

    # 1. Register all mmdet modules (Required in 3.x)
    register_all_modules()

    # 2. Load Config
    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    # 3. Handle Work Directory
    if args.work_dir is not None:
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        cfg.work_dir = osp.join('./work_dirs',
                                osp.splitext(osp.basename(args.config))[0])

    # 4. Handle Mixed Precision (AMP)
    if args.amp is True:
        cfg.optim_wrapper.type = 'AmpOptimWrapper'
        cfg.optim_wrapper.setdefault('loss_scale', 'dynamic')

    # 5. Build the Runner
    # In 3.x, the Runner is the single entry point for training/testing
    runner = Runner.from_cfg(cfg)

    # 6. Start Training
    runner.train()

if __name__ == '__main__':
    main()

import argparse
import os
import os.path as osp

from mmengine.config import Config, DictAction
from mmengine.runner import Runner
from mmdet.utils import register_all_modules

def parse_args():
    parser = argparse.ArgumentParser(description='Test (and eval) a model')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('--work-dir', help='the dir to save logs and models')
    parser.add_argument(
        '--show', action='store_true', help='show prediction results')
    parser.add_argument(
        '--show-dir',
        help='directory where painted images will be saved')
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
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    return args

def main():
    args = parse_args()

    # 1. Register all modules (important for custom backbones)
    register_all_modules()

    # 2. Load Config
    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    # 3. Setup work directory
    if args.work_dir is not None:
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        cfg.work_dir = osp.join('./work_dirs',
                                osp.splitext(osp.basename(args.config))[0])

    # 4. Update config for checkpoint loading
    cfg.load_from = args.checkpoint

    # 5. Handle visualization
    if args.show or args.show_dir:
        cfg.visualizer.vis_backends.append(dict(type='LocalVisBackend'))
        if args.show_dir:
            os.makedirs(args.show_dir, exist_ok=True)
            # Add a hook to save the images
            cfg.default_hooks.visualization = dict(
                type='DetVisualizationHook',
                draw=True,
                test_out_dir=args.show_dir)

    # 6. Build the Runner
    runner = Runner.from_cfg(cfg)

    # 7. Start Testing
    # This will use the test_dataloader and test_evaluator from your config
    runner.test()

if __name__ == '__main__':
    main()

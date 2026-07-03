# python imports
import argparse
import os
import glob
import time
from pprint import pprint

# torch imports
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.utils.data

# our code
from libs.core import load_config
from libs.datasets import make_dataset, make_data_loader
from libs.modeling import make_meta_arch
from libs.utils import valid_one_epoch, ANETdetection, fix_random_seed


################################################################################
def main(args):
    """0. load config"""
    # sanity check
    if os.path.isfile(args.config):
        cfg = load_config(args.config)
    else:
        raise ValueError("Config file does not exist.")
    assert len(cfg['val_split']) > 0, "Test set must be specified!"
    if ".pth.tar" in args.ckpt:
        assert os.path.isfile(args.ckpt), "CKPT file does not exist!"
        ckpt_file = args.ckpt
    else:
        # [CHANGED] 将断言替换为友好错误提示，并检测空目录
        if not os.path.isdir(args.ckpt):
            raise FileNotFoundError(
                f"Checkpoint 目录不存在: {args.ckpt}\n"
                "请先运行训练（python train.py ...）生成模型权重，\n"
                "或指定正确的 checkpoint 文件路径。"
            )
        ckpt_file_list = sorted(glob.glob(os.path.join(args.ckpt, '*.pth.tar')))
        if len(ckpt_file_list) == 0:
            raise FileNotFoundError(
                f"Checkpoint 目录中未找到 .pth.tar 文件: {args.ckpt}\n"
                "该目录下没有训练好的模型权重。请确认训练已完成并生成了 checkpoint。"
            )
        ckpt_file = ckpt_file_list[-1]

    if args.topk > 0:
        cfg['model']['test_cfg']['max_seg_num'] = args.topk
    pprint(cfg)

    """1. fix all randomness"""
    # fix the random seeds (this will fix everything)
    _ = fix_random_seed(0, include_cuda=True)

    """2. create dataset / dataloader"""
    val_dataset = make_dataset(
        cfg['dataset_name'], False, cfg['val_split'], **cfg['dataset']
    )
    # set bs = 1, and disable shuffle
    # [CHANGED] num_workers=0 避免 Windows 下 DataLoader 多进程卡死
    val_loader = make_data_loader(
        val_dataset, False, None, 1, 0  # was: cfg['loader']['num_workers']
    )

    """3. create model and evaluator"""
    # model
    model = make_meta_arch(cfg['model_name'], **cfg['model'])
    # [CHANGED] 单卡时跳过 DataParallel，避免 Windows 下死锁
    if len(cfg['devices']) > 1:
        model = nn.DataParallel(model, device_ids=[torch.device(d).index for d in cfg['devices']])
    else:
        model = model.cuda()

    """4. load ckpt"""
    print("=> loading checkpoint '{}'".format(ckpt_file))
    # load ckpt, reset epoch / best rmse
    checkpoint = torch.load(ckpt_file, map_location=cfg['devices'][0])
    # load ema model instead
    print("Loading from EMA model ...")
    # [CHANGED] 单卡时去除 DataParallel 的 'module.' 前缀
    state_dict = checkpoint['state_dict_ema']
    if len(cfg['devices']) == 1:
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    del checkpoint

    # set up evaluator
    det_eval, output_file = None, None
    if not args.saveonly:
        val_db_vars = val_dataset.get_attributes()
        det_eval = ANETdetection(
            val_dataset.json_file,
            val_dataset.split[0],
            tiou_thresholds=val_db_vars['tiou_thresholds']
        )
    else:
        output_file = os.path.join(os.path.split(ckpt_file)[0], 'eval_results.pkl')

    """5. Test the model"""
    print("\nStart testing model {:s} ...".format(cfg['model_name']))
    start = time.time()
    mAP = valid_one_epoch(
        val_loader,
        model,
        -1,
        evaluator=det_eval,
        output_file=output_file,
        ext_score_file=cfg['test_cfg']['ext_score_file'],
        tb_writer=None,
        print_freq=args.print_freq
    )
    end = time.time()
    print("All done! Total time: {:0.2f} sec".format(end - start))
    return


################################################################################
if __name__ == '__main__':
    """Entry Point"""
    # the arg parser
    parser = argparse.ArgumentParser(
        description='Train a point-based transformer for action localization')
    parser.add_argument('config', type=str, metavar='DIR',
                        help='path to a config file')
    parser.add_argument('ckpt', type=str, metavar='DIR',
                        help='path to a checkpoint')
    parser.add_argument('-t', '--topk', default=-1, type=int,
                        help='max number of output actions (default: -1)')
    parser.add_argument('--saveonly', action='store_true',
                        help='Only save the ouputs without evaluation (e.g., for test set)')
    parser.add_argument('-p', '--print-freq', default=10, type=int,
                        help='print frequency (default: 10 iterations)')
    args = parser.parse_args()
    main(args)

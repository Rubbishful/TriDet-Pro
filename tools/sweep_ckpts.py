"""
Sweep all checkpoints in a directory and evaluate mAP on the test set.

Usage:
  C:/Developer/Anaconda/envs/PatternRecognition/python.exe tools/sweep_ckpts.py ^
      configs/thumos_i3d_se.yaml ckpt/thumos_i3d_se_test
"""

import os as _os
_os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')

import argparse
import csv
import glob
import json
import os
import re
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn

PROJ_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJ_ROOT))

from libs.core import load_config
from libs.datasets import make_dataset, make_data_loader
from libs.modeling import make_meta_arch
from libs.utils import valid_one_epoch, ANETdetection, fix_random_seed

EVAL_OUTPUT = PROJ_ROOT / 'eval_output'
PYTHON = r"C:\Developer\Anaconda\envs\PatternRecognition\python.exe"


def evaluate_checkpoint(cfg, ckpt_path, val_loader, det_eval, device, print_freq):
    """Load a checkpoint and evaluate mAP. Returns (mAP, elapsed_seconds)."""
    model = make_meta_arch(cfg['model_name'], **cfg['model'])
    model = nn.DataParallel(model, device_ids=[torch.device(d).index for d in cfg['devices']])

    checkpoint = torch.load(ckpt_path, map_location=device)
    epoch = checkpoint.get('epoch', '?')
    model.load_state_dict(checkpoint['state_dict_ema'])
    del checkpoint

    t0 = time.time()
    mAP = valid_one_epoch(
        val_loader, model, -1,
        evaluator=det_eval,
        output_file=None,
        ext_score_file=cfg['test_cfg']['ext_score_file'],
        tb_writer=None,
        print_freq=print_freq,
    )
    elapsed = time.time() - t0

    # free model memory
    del model
    torch.cuda.empty_cache()

    return epoch, mAP, elapsed


def main(args):
    cfg = load_config(args.config)
    device = cfg['devices'][0]

    # override top-k
    if args.topk > 0:
        cfg['model']['test_cfg']['max_seg_num'] = args.topk

    # find checkpoint files
    ckpt_dir = Path(args.ckpt_dir)
    ckpt_files = sorted(
        ckpt_dir.glob('epoch_*.pth.tar'),
        key=lambda p: int(re.search(r'epoch_(\d+)', p.name).group(1))
    )
    if not ckpt_files:
        print(f"[ERROR] No epoch_*.pth.tar files found in {ckpt_dir}")
        sys.exit(1)

    # filter by epoch range
    if args.start_epoch > 0 or args.end_epoch > 0:
        start = args.start_epoch if args.start_epoch > 0 else 0
        end = args.end_epoch if args.end_epoch > 0 else float('inf')
        ckpt_files = [f for f in ckpt_files if start <= int(
            re.search(r'epoch_(\d+)', f.name).group(1)) <= end]

    print(f"Found {len(ckpt_files)} checkpoints: "
          f"{', '.join(p.name for p in ckpt_files)}")

    # build dataset and evaluator ONCE
    _ = fix_random_seed(0, include_cuda=True)
    val_dataset = make_dataset(
        cfg['dataset_name'], False, cfg['val_split'], **cfg['dataset']
    )
    val_loader = make_data_loader(
        val_dataset, False, None, 1, cfg['loader']['num_workers']
    )
    val_db_vars = val_dataset.get_attributes()
    det_eval = ANETdetection(
        val_dataset.json_file,
        val_dataset.split[0],
        tiou_thresholds=val_db_vars['tiou_thresholds']
    )

    # evaluate each checkpoint
    results = []
    cfg_name = os.path.basename(args.config).replace('.yaml', '')
    tag = f"{cfg_name}_{ckpt_dir.name}"

    print(f"\n{'='*60}")
    print(f"  Sweeping {len(ckpt_files)} checkpoints on {cfg['val_split']}")
    print(f"  Config: {args.config}")
    print(f"{'='*60}\n")

    for i, ckpt_path in enumerate(ckpt_files):
        print(f"\n[{i+1:3d}/{len(ckpt_files):3d}] {ckpt_path.name}")

        epoch, mAP, elapsed = evaluate_checkpoint(
            cfg, str(ckpt_path), val_loader, det_eval, device, args.print_freq
        )

        results.append({
            'epoch': int(epoch) if isinstance(epoch, int) else epoch,
            'ckpt': ckpt_path.name,
            'mAP': round(float(mAP * 100), 2),
            'time_s': round(elapsed, 1),
        })

        # incremental save
        EVAL_OUTPUT.mkdir(parents=True, exist_ok=True)

        # save CSV
        csv_path = EVAL_OUTPUT / f'sweep_{tag}.csv'
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['epoch', 'ckpt', 'mAP', 'time_s'])
            writer.writeheader()
            writer.writerows(results)

        # save JSON
        json_path = EVAL_OUTPUT / f'sweep_{tag}.json'
        with open(json_path, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"  mAP={results[-1]['mAP']:.2f}%  time={elapsed:.0f}s  "
              f"[saved to {EVAL_OUTPUT}]")

    # final summary
    best = max(results, key=lambda r: r['mAP'])
    worst = min(results, key=lambda r: r['mAP'])

    print(f"\n{'='*60}")
    print(f"  Sweep Complete — {len(results)} checkpoints")
    print(f"  Best:  epoch {best['epoch']:3d}  mAP = {best['mAP']:.2f}%")
    print(f"  Worst: epoch {worst['epoch']:3d}  mAP = {worst['mAP']:.2f}%")
    print(f"  Output: {EVAL_OUTPUT}")
    print(f"{'='*60}")

    # print full table
    print(f"\n  {'Epoch':>6s}  {'mAP':>8s}  {'Time':>8s}")
    print(f"  {'-'*6}  {'-'*8}  {'-'*8}")
    for r in results:
        marker = ' <-- BEST' if r['epoch'] == best['epoch'] else ''
        print(f"  {r['epoch']:6d}  {r['mAP']:7.2f}%  {r['time_s']:7.1f}s{marker}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Sweep all checkpoints and evaluate mAP')
    parser.add_argument('config', help='path to config YAML')
    parser.add_argument('ckpt_dir', help='directory containing epoch_*.pth.tar files')
    parser.add_argument('--start-epoch', default=-1, type=int,
                        help='minimum epoch to evaluate (default: all)')
    parser.add_argument('--end-epoch', default=-1, type=int,
                        help='maximum epoch to evaluate (default: all)')
    parser.add_argument('-t', '--topk', default=-1, type=int,
                        help='max number of output actions (default: config value)')
    parser.add_argument('-p', '--print-freq', default=20, type=int,
                        help='print frequency during eval (default: 20)')
    args = parser.parse_args()
    main(args)

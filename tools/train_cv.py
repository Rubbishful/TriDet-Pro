import os as _os
_os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')

"""
Train with 90/10 train/val split and progressive epoch search.

Train in blocks of --step epochs. After each block, evaluate validation loss
and check the stopping criterion: stop when loss improvement is < threshold
for --patience consecutive blocks, or loss regresses.

Outputs loss curves and CSV logs to ./train_output/.

Usage:
  python tools/train_cv.py ./configs/thumos_i3d_se.yaml --step 10 --batch-size 4 --grad-accum 1
"""

import argparse
import csv
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Subset

PROJ_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJ_ROOT))

from libs.core import load_config
from libs.datasets import make_dataset, make_data_loader
from libs.modeling import make_meta_arch
from libs.utils import (
    AverageMeter, ModelEma,
    fix_random_seed, make_optimizer, make_scheduler,
    train_one_epoch,
)

TRAIN_OUTPUT = PROJ_ROOT / 'train_output'


def train_val_split(n_samples, val_ratio=0.1, seed=42):
    """Return (train_indices, val_indices) with stratified shuffle."""
    rng = np.random.RandomState(seed)
    perm = rng.permutation(n_samples)
    val_n = max(int(n_samples * val_ratio), 1)
    val_idx = perm[:val_n].tolist()
    train_idx = perm[val_n:].tolist()
    return train_idx, val_idx


@torch.no_grad()
def compute_val_loss(model, val_loader):
    """Compute average final_loss on validation set (model stays in train mode)."""
    model.train()
    meter = AverageMeter()
    for video_list in val_loader:
        losses = model(video_list)
        meter.update(losses['final_loss'].item(), len(video_list))
    return meter.avg


def check_early_stop(block_losses, threshold=0.01, patience=3):
    """
    Return True if loss improvement < threshold for `patience` consecutive blocks,
    or if loss increased (regressed) in all of the last `patience` blocks.
    """
    if len(block_losses) < patience + 1:
        return False
    recent = block_losses[-(patience + 1):]
    stagnation_count = 0
    for i in range(1, len(recent)):
        improvement = (recent[i - 1] - recent[i]) / max(recent[i - 1], 1e-8)
        if improvement < threshold:
            stagnation_count += 1
    return stagnation_count >= patience


def save_loss_csv(all_records, filepath):
    """Save per-iteration training loss records to CSV."""
    if not all_records:
        return
    keys = ['epoch', 'iteration', 'final_loss', 'cls_loss', 'reg_loss']
    with open(filepath, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=keys, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(all_records)


def plot_loss_curves(loss_records, block_epochs, block_losses, save_dir, tag=''):
    """Generate loss curve plots and save to save_dir."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Training loss per iteration
    ax1 = axes[0]
    if loss_records:
        iters = [r['iteration'] for r in loss_records]
        final = [r['final_loss'] for r in loss_records]
        ax1.plot(iters, final, 'b-', alpha=0.5, linewidth=0.5, label='final_loss')
        if 'cls_loss' in loss_records[0]:
            cls_vals = [r['cls_loss'] for r in loss_records]
            ax1.plot(iters, cls_vals, 'g-', alpha=0.3, linewidth=0.3, label='cls_loss')
        if 'reg_loss' in loss_records[0]:
            reg_vals = [r['reg_loss'] for r in loss_records]
            ax1.plot(iters, reg_vals, 'r-', alpha=0.3, linewidth=0.3, label='reg_loss')
        ax1.set_xlabel('Global Iteration')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training Loss' + (f' [{tag}]' if tag else ''))
        ax1.legend(fontsize=7)
        ax1.grid(True, alpha=0.3)

    # Plot 2: Validation loss per block
    ax2 = axes[1]
    if block_epochs and block_losses:
        ax2.plot(block_epochs, block_losses, 'ro-', markersize=6, label='val_loss')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Val Loss')
        ax2.set_title('Validation Loss' + (f' [{tag}]' if tag else ''))
        ax2.legend()
        ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    fig_path = os.path.join(save_dir, f'loss_curves_{tag}.png' if tag else 'loss_curves.png')
    plt.savefig(fig_path, dpi=150)
    plt.close()
    return fig_path


def train_with_search(cfg, train_indices, val_indices, args):
    """Train with block-wise validation and early stopping."""
    print(f"\n{'='*60}")
    print(f"  Train samples: {len(train_indices)}, Val samples: {len(val_indices)}")
    print(f"  Batch size: {cfg['loader']['batch_size']}, Grad accum: {args.grad_accum}")
    print(f"  Effective batch size: {cfg['loader']['batch_size'] * args.grad_accum}")
    print(f"{'='*60}")

    rng_gen = fix_random_seed(cfg['init_rand_seed'], include_cuda=True)

    full_dataset = make_dataset(
        cfg['dataset_name'], True, cfg['train_split'], **cfg['dataset']
    )
    train_ds = Subset(full_dataset, train_indices)
    val_ds = Subset(full_dataset, val_indices)

    train_loader = make_data_loader(
        train_ds, True, rng_gen, cfg['loader']['batch_size'], cfg['loader']['num_workers']
    )
    val_loader = make_data_loader(
        val_ds, False, None,
        cfg['loader']['batch_size'], max(cfg['loader']['num_workers'] // 2, 1)
    )

    model = make_meta_arch(cfg['model_name'], **cfg['model'])
    model = nn.DataParallel(model, device_ids=[torch.device(d).index for d in cfg['devices']])
    model_ema = ModelEma(model)

    optimizer = make_optimizer(model, cfg['opt'])
    num_iters_per_epoch = len(train_loader)
    scheduler = make_scheduler(optimizer, cfg['opt'], num_iters_per_epoch)

    block_losses = []
    block_epochs = []
    best_val_loss = float('inf')
    best_epoch = 0
    total_epochs = cfg['opt']['epochs'] + cfg['opt']['warmup_epochs']
    all_loss_records = []

    # build output tag
    cfg_name = os.path.basename(args.config).replace('.yaml', '')
    tag = f"{cfg_name}_b{cfg['loader']['batch_size']}_ga{args.grad_accum}"

    for block_start in range(0, total_epochs, args.step):
        block_end = min(block_start + args.step, total_epochs)

        for epoch in range(block_start, block_end):
            records = train_one_epoch(
                train_loader, model, optimizer, scheduler, epoch,
                model_ema=model_ema,
                clip_grad_l2norm=cfg['train_cfg']['clip_grad_l2norm'],
                print_freq=args.print_freq,
                grad_accum=args.grad_accum,
                return_losses=True,
            )
            if records:
                all_loss_records.extend(records)

        val_loss = compute_val_loss(model_ema.module, val_loader)
        block_losses.append(val_loss)
        block_epochs.append(block_end)
        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
            best_epoch = block_end

        n_blocks = len(block_losses)
        if n_blocks > 1:
            prev = block_losses[-2]
            imp_pct = (prev - val_loss) / max(prev, 1e-8) * 100
            imp_str = f"improvement: {imp_pct:+.2f}%"
        else:
            imp_str = "improvement: N/A (first block)"
        marker = " *BEST*" if is_best else ""
        print(f"Block {n_blocks} (epoch {block_end:3d}): "
              f"val_loss={val_loss:.4f}  {imp_str}{marker}")

        # save intermediate CSV & plot after each block
        TRAIN_OUTPUT.mkdir(parents=True, exist_ok=True)
        csv_path = TRAIN_OUTPUT / f'loss_records_{tag}.csv'
        save_loss_csv(all_loss_records, str(csv_path))
        plot_loss_curves(all_loss_records, block_epochs, block_losses, str(TRAIN_OUTPUT), tag)
        print(f"  [Loss plot & CSV updated] -> {TRAIN_OUTPUT}")

        if np.isnan(val_loss) or val_loss > 1e6:
            print("Loss unstable, stopping early.")
            break

        if check_early_stop(block_losses, args.threshold, args.patience):
            print(f"Early stopped: improvement < {args.threshold*100:.0f}% "
                  f"for {args.patience} consecutive blocks.")
            break

    return {
        'block_epochs': block_epochs,
        'block_losses': [float(v) for v in block_losses],
        'best_epoch': best_epoch,
        'best_val_loss': float(best_val_loss),
        'stopped_at_epoch': block_epochs[-1] if block_epochs else 0,
        'total_blocks': len(block_losses),
        'train_n': len(train_indices),
        'val_n': len(val_indices),
    }


def main(args):
    cfg = load_config(args.config)

    # --- batch size override ---
    if args.batch_size > 0:
        old_bs = cfg['loader'].get('batch_size', 8)
        cfg['loader']['batch_size'] = args.batch_size
        print(f"[Setup] Batch size override: {old_bs} -> {args.batch_size}")
        if args.grad_accum > 1:
            effective = args.batch_size * args.grad_accum
            print(f"[Setup] Gradient accumulation: {args.grad_accum} steps, "
                  f"effective batch size = {effective}")

    if args.max_epochs > 0:
        total = args.max_epochs + cfg['opt']['warmup_epochs']
        print(f"[Setup] Overriding epochs: warmup={cfg['opt']['warmup_epochs']} + "
              f"train={args.max_epochs} = {total} total (step={args.step})")
    else:
        args.max_epochs = cfg['opt']['epochs']
    cfg['opt']['epochs'] = args.max_epochs

    cfg['opt']['learning_rate'] *= len(cfg['devices'])
    cfg['loader']['num_workers'] *= len(cfg['devices'])

    ts = datetime.fromtimestamp(int(time.time()))
    cfg_name = os.path.basename(args.config).replace('.yaml', '')
    out_dir = Path(cfg.get('output_folder', './ckpt'))
    exp_folder = out_dir / f"{cfg_name}_search_{ts.strftime('%m%d_%H%M')}"
    exp_folder.mkdir(parents=True, exist_ok=True)

    with open(exp_folder / 'config.txt', 'w') as f:
        from pprint import pprint
        pprint(cfg, stream=f)

    full_dataset = make_dataset(
        cfg['dataset_name'], True, cfg['train_split'], **cfg['dataset']
    )
    n_samples = len(full_dataset)
    train_idx, val_idx = train_val_split(
        n_samples, val_ratio=args.val_ratio, seed=cfg['init_rand_seed']
    )

    result = train_with_search(cfg, train_idx, val_idx, args)

    result['batch_size'] = cfg['loader']['batch_size']
    result['grad_accum'] = args.grad_accum
    with open(exp_folder / 'result.json', 'w') as f:
        json.dump(result, f, indent=2)

    print(f"\n{'='*60}")
    print(f"  Epoch search complete")
    print(f"  Best epoch:  {result['best_epoch']}")
    print(f"  Best loss:   {result['best_val_loss']:.4f}")
    print(f"  Stopped at:  {result['stopped_at_epoch']}")
    print(f"  Results saved to: {exp_folder}")
    print(f"  Loss plots & CSV saved to: {TRAIN_OUTPUT}")
    print(f"{'='*60}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Train with 90/10 split and progressive epoch search')
    parser.add_argument('config', help='path to config YAML')
    parser.add_argument('--step', default=10, type=int,
                        help='epoch step size for validation check (default: 10)')
    parser.add_argument('--max-epochs', default=-1, type=int,
                        help='override training epochs, -1 to use config value')
    parser.add_argument('--val-ratio', default=0.1, type=float,
                        help='validation split ratio (default: 0.1)')
    parser.add_argument('--threshold', default=0.01, type=float,
                        help='minimum improvement threshold (default: 0.01 = 1%%)')
    parser.add_argument('--patience', default=3, type=int,
                        help='consecutive stagnant blocks before stopping (default: 3)')
    parser.add_argument('--print-freq', default=10, type=int,
                        help='print frequency in iterations (default: 10)')
    parser.add_argument('--batch-size', default=-1, type=int,
                        help='override batch size from config, -1 to use config value')
    parser.add_argument('--grad-accum', default=1, type=int,
                        help='gradient accumulation steps (default: 1 = no accumulation)')
    args = parser.parse_args()
    main(args)

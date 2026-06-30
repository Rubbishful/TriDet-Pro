"""
K-fold cross-validation training with progressive epoch search.

For each fold, train in blocks of --step epochs. After each block, evaluate
validation loss and check the stopping criterion: stop when validation loss
improvement is < threshold for --patience consecutive blocks, or loss regresses.

Usage:
  python tools/train_cv.py ./configs/thumos_i3d_se.yaml --k 10 --step 10
  python tools/train_cv.py ./configs/thumos_i3d_se.yaml --k 5 --step 10 --max-epochs 80 --patience 3 --threshold 0.01
"""

import argparse
import copy
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


def kfold_split(n_samples, k=10, seed=42):
    """Return list of (train_indices, val_indices) for each fold."""
    rng = np.random.RandomState(seed)
    perm = rng.permutation(n_samples)
    fold_size = n_samples // k
    folds = []
    for i in range(k):
        start = i * fold_size
        end = start + fold_size if i < k - 1 else n_samples
        val_idx = perm[start:end].tolist()
        train_idx = np.concatenate([perm[:start], perm[end:]]).tolist()
        folds.append((train_idx, val_idx))
    return folds


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


def train_fold(cfg, train_indices, val_indices, fold_id, args, device):
    """Train a single fold with block-wise validation and early stopping."""
    print(f"\n{'='*60}")
    print(f"  Fold {fold_id + 1}/{args.k}")
    print(f"  Train samples: {len(train_indices)}, Val samples: {len(val_indices)}")
    print(f"{'='*60}")

    fold_seed = cfg['init_rand_seed'] + fold_id * 100
    rng_gen = fix_random_seed(fold_seed, include_cuda=True)

    # --- Create full dataset, then subset ---
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

    # --- Model, optimizer, scheduler ---
    model = make_meta_arch(cfg['model_name'], **cfg['model'])
    model = nn.DataParallel(model, device_ids=[torch.device(d).index for d in cfg['devices']])
    model_ema = ModelEma(model)

    optimizer = make_optimizer(model, cfg['opt'])
    num_iters_per_epoch = len(train_loader)
    scheduler = make_scheduler(optimizer, cfg['opt'], num_iters_per_epoch)

    # --- Training blocks ---
    block_losses = []       # best val_loss per block
    block_epochs = []       # ending epoch per block
    best_val_loss = float('inf')
    best_epoch = 0
    total_epochs = cfg['opt']['epochs'] + cfg['opt']['warmup_epochs']

    for block_start in range(0, total_epochs, args.step):
        block_end = min(block_start + args.step, total_epochs)

        # Train this block
        for epoch in range(block_start, block_end):
            train_one_epoch(
                train_loader, model, optimizer, scheduler, epoch,
                model_ema=model_ema,
                clip_grad_l2norm=cfg['train_cfg']['clip_grad_l2norm'],
                print_freq=args.print_freq,
            )

        # Validate
        val_loss = compute_val_loss(model_ema.module, val_loader)
        block_losses.append(val_loss)
        block_epochs.append(block_end)
        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
            best_epoch = block_end

        # Log
        n_blocks = len(block_losses)
        if n_blocks > 1:
            prev = block_losses[-2]
            imp_pct = (prev - val_loss) / max(prev, 1e-8) * 100
            imp_str = f"improvement: {imp_pct:+.2f}%"
        else:
            imp_str = "improvement: N/A (first block)"
        marker = " *BEST*" if is_best else ""
        print(f"[Fold {fold_id+1}] Block {n_blocks} (epoch {block_end:3d}): "
              f"val_loss={val_loss:.4f}  {imp_str}{marker}")

        # Check NaN
        if np.isnan(val_loss) or val_loss > 1e6:
            print(f"[Fold {fold_id+1}] Loss unstable, stopping fold early.")
            break

        # Early stop
        if check_early_stop(block_losses, args.threshold, args.patience):
            print(f"[Fold {fold_id+1}] Early stopped: "
                  f"improvement < {args.threshold*100:.0f}% for {args.patience} consecutive blocks.")
            break

    result = {
        'fold': fold_id,
        'train_n': len(train_indices),
        'val_n': len(val_indices),
        'block_epochs': block_epochs,
        'block_losses': [float(v) for v in block_losses],
        'best_epoch': best_epoch,
        'best_val_loss': float(best_val_loss),
        'stopped_at_epoch': block_epochs[-1] if block_epochs else 0,
        'total_blocks': len(block_losses),
    }
    return result


def main(args):
    cfg = load_config(args.config)

    # Override epochs for search mode
    if args.max_epochs > 0:
        total = args.max_epochs + cfg['opt']['warmup_epochs']
        print(f"[Setup] Overriding epochs: warmup={cfg['opt']['warmup_epochs']} + "
              f"train={args.max_epochs} = {total} total (step={args.step})")
    else:
        args.max_epochs = cfg['opt']['epochs']
    cfg['opt']['epochs'] = args.max_epochs

    # Scale LR / workers
    cfg['opt']['learning_rate'] *= len(cfg['devices'])
    cfg['loader']['num_workers'] *= len(cfg['devices'])

    # Output folder
    ts = datetime.fromtimestamp(int(time.time()))
    cfg_name = os.path.basename(args.config).replace('.yaml', '')
    out_dir = Path(cfg.get('output_folder', './ckpt'))
    cv_folder = out_dir / f"{cfg_name}_cv_{ts.strftime('%m%d_%H%M')}"
    cv_folder.mkdir(parents=True, exist_ok=True)

    # Save merged config
    with open(cv_folder / 'config.txt', 'w') as f:
        from pprint import pprint
        pprint(cfg, stream=f)

    device = cfg['devices'][0]

    # --- Create full dataset for index-based K-fold split ---
    full_dataset = make_dataset(
        cfg['dataset_name'], True, cfg['train_split'], **cfg['dataset']
    )
    n_samples = len(full_dataset)
    if n_samples < args.k:
        print(f"[Warning] Dataset has {n_samples} samples, fewer than k={args.k}. "
              f"Using k={n_samples} instead.")
        args.k = n_samples

    folds = kfold_split(n_samples, k=args.k, seed=cfg['init_rand_seed'])

    # --- Train each fold ---
    all_results = []
    for fold_id in range(args.k):
        result = train_fold(
            cfg,
            folds[fold_id][0],
            folds[fold_id][1],
            fold_id,
            args,
            device,
        )
        all_results.append(result)

        # Save per-fold result
        with open(cv_folder / f'fold_{fold_id:02d}.json', 'w') as f:
            json.dump(result, f, indent=2)

    # --- Summary ---
    best_epochs = [r['best_epoch'] for r in all_results]
    best_losses = [r['best_val_loss'] for r in all_results]
    stopped_epochs = [r['stopped_at_epoch'] for r in all_results]

    summary = {
        'config': args.config,
        'k': args.k,
        'step': args.step,
        'threshold': args.threshold,
        'patience': args.patience,
        'max_epochs': args.max_epochs,
        'warmup_epochs': cfg['opt']['warmup_epochs'],
        'folds': all_results,
        'mean_best_epoch': float(np.mean(best_epochs)),
        'std_best_epoch': float(np.std(best_epochs)),
        'mean_best_val_loss': float(np.mean(best_losses)),
        'std_best_val_loss': float(np.std(best_losses)),
        'mean_stopped_epoch': float(np.mean(stopped_epochs)),
    }

    with open(cv_folder / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*60}")
    print(f"  Cross-validation complete ({args.k} folds)")
    print(f"  Best epoch:  {summary['mean_best_epoch']:.1f} ± {summary['std_best_epoch']:.1f}")
    print(f"  Best loss:   {summary['mean_best_val_loss']:.4f} ± {summary['std_best_val_loss']:.4f}")
    print(f"  Stopped at:  {summary['mean_stopped_epoch']:.1f} (mean)")
    print(f"  Results saved to: {cv_folder}")
    print(f"{'='*60}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='K-fold CV training with progressive epoch search')
    parser.add_argument('config', help='path to config YAML')
    parser.add_argument('--k', default=10, type=int,
                        help='number of folds (default: 10)')
    parser.add_argument('--step', default=10, type=int,
                        help='epoch step size for validation check (default: 10)')
    parser.add_argument('--max-epochs', default=100, type=int,
                        help='maximum training epochs, excluding warmup (default: 100)')
    parser.add_argument('--threshold', default=0.01, type=float,
                        help='minimum improvement threshold (default: 0.01 = 1%%)')
    parser.add_argument('--patience', default=3, type=int,
                        help='consecutive stagnant blocks before stopping (default: 3)')
    parser.add_argument('--print-freq', default=10, type=int,
                        help='print frequency in iterations (default: 10)')
    args = parser.parse_args()
    main(args)

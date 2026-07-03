import os as _os
_os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')

import argparse
import csv
import json
import os
import time
import datetime
from pathlib import Path
from pprint import pprint

import torch
import torch.nn as nn
import torch.utils.data

from libs.core import load_config
from libs.datasets import make_dataset, make_data_loader
from libs.modeling import make_meta_arch
from libs.utils import (train_one_epoch, valid_one_epoch, ANETdetection,
                        save_checkpoint, make_optimizer, make_scheduler,
                        fix_random_seed, ModelEma)


def evaluate_checkpoint(cfg, ckpt_path, val_loader, det_eval, devices):
    """Load an EMA checkpoint and run validation, returning average mAP (float)."""
    model = make_meta_arch(cfg['model_name'], **cfg['model'])
    model = nn.DataParallel(model, device_ids=[torch.device(d).index for d in devices])
    checkpoint = torch.load(ckpt_path, map_location=devices[0])
    model.load_state_dict(checkpoint['state_dict_ema'])
    del checkpoint
    mAP = valid_one_epoch(
        val_loader, model, -1,
        evaluator=det_eval, output_file=None,
        ext_score_file=cfg['test_cfg'].get('ext_score_file'),
        tb_writer=None, print_freq=1000,
    )
    del model
    return mAP


def check_convergence(mAPs, threshold=0.001):
    """
    Check if the last 3 step-to-step improvements satisfy convergence.
    mAPs: chronologically ordered mAP values from step evaluations.
    Convergence: 3 consecutive epoch increases each yield < threshold mAP
    improvement, or any step shows regression.
    Returns (converged: bool, reason: str).
    """
    if len(mAPs) < 4:
        return False, "insufficient_data"
    deltas = [mAPs[i + 1] - mAPs[i] for i in range(len(mAPs) - 3, len(mAPs) - 1)]
    # Regression check: any step shows mAP decrease
    if any(d < 0 for d in deltas):
        return True, "regression"
    # Plateau check: each of the last 3 step improvements < threshold
    if all(d < threshold for d in deltas):
        return True, "plateau"
    return False, "continuing"


def save_loss_csv(all_records, filepath):
    if not all_records:
        return
    keys = ['epoch', 'iteration', 'final_loss', 'cls_loss', 'reg_loss']
    with open(filepath, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=keys, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(all_records)


def plot_loss_curves(loss_records, save_dir, tag=''):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    if loss_records:
        iters = [r['iteration'] for r in loss_records]
        final = [r['final_loss'] for r in loss_records]
        ax.plot(iters, final, 'b-', alpha=0.5, linewidth=0.5, label='final_loss')
        if 'cls_loss' in loss_records[0]:
            cls_vals = [r['cls_loss'] for r in loss_records]
            ax.plot(iters, cls_vals, 'g-', alpha=0.3, linewidth=0.3, label='cls_loss')
        if 'reg_loss' in loss_records[0]:
            reg_vals = [r['reg_loss'] for r in loss_records]
            ax.plot(iters, reg_vals, 'r-', alpha=0.3, linewidth=0.3, label='reg_loss')
        ax.set_xlabel('Global Iteration')
        ax.set_ylabel('Loss')
        ax.set_title('Training Loss' + (f' [{tag}]' if tag else ''))
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig_path = os.path.join(save_dir, f'loss_curves_{tag}.png' if tag else 'loss_curves.png')
    plt.savefig(fig_path, dpi=150)
    plt.close()
    return fig_path


def run_convergence_train(cfg, args):
    """Single-run convergence test with periodic mAP evaluation."""
    warmup = cfg['opt']['warmup_epochs']
    start_eval_epochs = args.start_epochs
    step_size = args.step_size
    max_epochs = args.max_epochs
    threshold = args.threshold

    # Override config to train for max possible epochs
    cfg['opt']['epochs'] = max_epochs
    total_epochs = max_epochs + warmup

    # Output folder
    cfg_filename = os.path.basename(args.config).replace('.yaml', '')
    if args.output_dir:
        ckpt_folder = args.output_dir
    else:
        ts = datetime.datetime.fromtimestamp(int(time.time()))
        ckpt_folder = os.path.join(
            cfg['output_folder'], cfg_filename + '_conv_' + str(ts))
    os.makedirs(ckpt_folder, exist_ok=True)
    os.makedirs('train_output', exist_ok=True)

    # Fix seed
    rng_generator = fix_random_seed(cfg['init_rand_seed'], include_cuda=True)

    # LR scaling
    cfg['opt']["learning_rate"] *= len(cfg['devices'])
    cfg['loader']['num_workers'] *= len(cfg['devices'])

    # Train dataset & loader
    train_dataset = make_dataset(cfg['dataset_name'], True, cfg['train_split'], **cfg['dataset'])
    train_db_vars = train_dataset.get_attributes()
    cfg['model']['train_cfg']['head_empty_cls'] = train_db_vars['empty_label_ids']
    train_loader = make_data_loader(train_dataset, True, rng_generator, **cfg['loader'])

    # Val dataset & loader (batch_size=1 for evaluation)
    val_dataset = make_dataset(cfg['dataset_name'], False, cfg['val_split'], **cfg['dataset'])
    val_loader = make_data_loader(val_dataset, False, None, 1, max(cfg['loader']['num_workers'], 1))
    val_db_vars = val_dataset.get_attributes()
    det_eval = ANETdetection(
        val_dataset.json_file, val_dataset.split[0],
        tiou_thresholds=val_db_vars['tiou_thresholds'])

    # Model, optimizer, scheduler, EMA
    model = make_meta_arch(cfg['model_name'], **cfg['model'])
    model = nn.DataParallel(model, device_ids=[torch.device(d).index for d in cfg['devices']])
    optimizer = make_optimizer(model, cfg['opt'])
    num_iters_per_epoch = len(train_loader)
    scheduler = make_scheduler(optimizer, cfg['opt'], num_iters_per_epoch)
    model_ema = ModelEma(model)

    # Save config
    with open(os.path.join(ckpt_folder, 'config.txt'), 'w') as fid:
        pprint(cfg, stream=fid)

    tag = f"{cfg_filename}_b{cfg['loader']['batch_size']}_ga{args.grad_accum}"
    all_loss_records = []
    results = []  # list of (non_warmup_epoch, mAP, time_s)

    print(f"\nConvergence test: start_eval={start_eval_epochs}, step={step_size}, "
          f"max={max_epochs}, warmup={warmup}, threshold={threshold}")
    print(f"Training up to {total_epochs} total epochs, checkpoints -> {ckpt_folder}\n")

    for epoch in range(total_epochs):
        t0 = time.time()
        records = train_one_epoch(
            train_loader, model, optimizer, scheduler, epoch,
            model_ema=model_ema,
            clip_grad_l2norm=cfg['train_cfg']['clip_grad_l2norm'],
            print_freq=args.print_freq,
            grad_accum=args.grad_accum,
            return_losses=True,
        )
        t1 = time.time()
        if records:
            all_loss_records.extend(records)

        # Save loss tracking
        train_out = Path('train_output')
        train_out.mkdir(parents=True, exist_ok=True)
        save_loss_csv(all_loss_records, str(train_out / f'loss_records_{tag}.csv'))
        plot_loss_curves(all_loss_records, str(train_out), tag)

        # Save checkpoint at specified frequency and at the last epoch
        if (epoch == total_epochs - 1) or (args.ckpt_freq > 0 and epoch > 0 and epoch % args.ckpt_freq == 0):
            save_states = {
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'scheduler': scheduler.state_dict(),
                'optimizer': optimizer.state_dict(),
                'state_dict_ema': model_ema.module.state_dict(),
            }
            save_checkpoint(save_states, False, file_folder=ckpt_folder,
                            file_name='epoch_{:03d}.pth.tar'.format(epoch))

        # Evaluate at step boundaries (after warmup, at multiples of step_size, starting from start_epochs)
        non_warmup_epoch = epoch - warmup + 1
        should_eval = (
            non_warmup_epoch >= start_eval_epochs and
            non_warmup_epoch % step_size == 0
        )
        if should_eval:
            ckpt_path = os.path.join(ckpt_folder, 'epoch_{:03d}.pth.tar'.format(epoch))
            print(f"\n[Eval] epoch {non_warmup_epoch} (total {epoch}) — evaluating...")
            mAP = evaluate_checkpoint(cfg, ckpt_path, val_loader, det_eval, cfg['devices'])
            elapsed = time.time() - t0
            results.append({
                'non_warmup_epoch': non_warmup_epoch,
                'total_epoch': epoch,
                'mAP': mAP,
                'time_s': round(elapsed, 1),
            })
            print(f"[Eval] epoch {non_warmup_epoch}: mAP = {mAP*100:.2f}%")

            # Write intermediate results
            csv_path = os.path.join(ckpt_folder, 'convergence_results.csv')
            with open(csv_path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=['non_warmup_epoch', 'total_epoch', 'mAP', 'time_s'])
                writer.writeheader()
                writer.writerows(results)

            # Check convergence
            mAPs = [r['mAP'] for r in results]
            converged, reason = check_convergence(mAPs, threshold)
            if converged:
                print(f"\n[Converged] epoch {non_warmup_epoch}: {reason}")
                print(f"  mAP history: {[f'{m*100:.2f}%' for m in mAPs]}")
                break

    # Write final summary
    summary = {
        'config': args.config,
        'start_epochs': start_eval_epochs,
        'step_size': step_size,
        'max_epochs': max_epochs,
        'warmup_epochs': warmup,
        'threshold': threshold,
        'results': results,
    }
    if results:
        mAPs = [r['mAP'] for r in results]
        converged, reason = check_convergence(mAPs, threshold)
        summary['converged'] = converged
        summary['convergence_reason'] = reason
        summary['best_mAP'] = max(mAPs)
        summary['best_epoch'] = results[mAPs.index(max(mAPs))]['non_warmup_epoch']
        summary['final_mAP'] = mAPs[-1]
    else:
        summary['converged'] = False
        summary['convergence_reason'] = 'no_evaluation_done'

    summary_path = os.path.join(ckpt_folder, 'convergence_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\nResults saved to {ckpt_folder}")
    print(f"  CSV:  convergence_results.csv")
    print(f"  JSON: convergence_summary.json")
    if results:
        print(f"  Best mAP: {summary['best_mAP']*100:.2f}% at epoch {summary['best_epoch']}")
    return summary


def run_eval_only(cfg, args):
    """Evaluate all checkpoints in a directory and check convergence (post-hoc analysis)."""
    ckpt_dir = args.eval_only
    ckpt_files = sorted([
        f for f in os.listdir(ckpt_dir)
        if f.startswith('epoch_') and f.endswith('.pth.tar')
    ])
    if not ckpt_files:
        print(f"No epoch_*.pth.tar checkpoints found in {ckpt_dir}")
        return

    # Val dataset
    val_dataset = make_dataset(cfg['dataset_name'], False, cfg['val_split'], **cfg['dataset'])
    val_loader = make_data_loader(val_dataset, False, None, 1, max(cfg['loader']['num_workers'], 1))
    val_db_vars = val_dataset.get_attributes()
    det_eval = ANETdetection(
        val_dataset.json_file, val_dataset.split[0],
        tiou_thresholds=val_db_vars['tiou_thresholds'])

    results = []
    for ckpt_file in ckpt_files:
        ckpt_path = os.path.join(ckpt_dir, ckpt_file)
        epoch_str = ckpt_file.replace('epoch_', '').replace('.pth.tar', '')
        total_epoch = int(epoch_str)
        non_warmup_epoch = total_epoch - cfg['opt']['warmup_epochs'] + 1
        print(f"\nEvaluating {ckpt_file} (non-warmup ~{non_warmup_epoch})...")
        mAP = evaluate_checkpoint(cfg, ckpt_path, val_loader, det_eval, cfg['devices'])
        results.append({
            'non_warmup_epoch': non_warmup_epoch,
            'total_epoch': total_epoch,
            'mAP': mAP,
            'time_s': 0,
        })
        print(f"  mAP = {mAP*100:.2f}%")

    # Write results
    csv_path = os.path.join(ckpt_dir, 'convergence_results.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['non_warmup_epoch', 'total_epoch', 'mAP', 'time_s'])
        writer.writeheader()
        writer.writerows(results)

    # Check convergence using all eval points
    mAPs = [r['mAP'] for r in results]
    if len(mAPs) >= 4:
        converged, reason = check_convergence(mAPs, args.threshold)
        if converged:
            sub = mAPs[-4:]
            print(f"\nConverged: {reason}")
            print(f"  Last 4 mAPs: {[f'{m*100:.2f}%' for m in sub]}")
            deltas = [sub[i+1] - sub[i] for i in range(3)]
            print(f"  Deltas: {[f'{d*100:.3f}%' for d in deltas]}")
        else:
            print(f"\nNot converged yet")
    else:
        print(f"\nNeed at least 4 evaluation points for convergence check, got {len(results)}")

    print(f"\nResults saved to {csv_path}")


def main(args):
    if os.path.isfile(args.config):
        cfg = load_config(args.config)
    else:
        raise ValueError("Config file does not exist.")

    if args.batch_size > 0:
        cfg['loader']['batch_size'] = args.batch_size

    # Eval-only mode: analyze existing checkpoints
    if args.eval_only:
        run_eval_only(cfg, args)
        return

    # Train + evaluate mode
    run_convergence_train(cfg, args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Convergence test: train with periodic mAP evaluation')
    parser.add_argument('config', type=str, help='path to config file')
    parser.add_argument('--start-epochs', default=60, type=int,
                        help='non-warmup epochs before first evaluation (default: 60)')
    parser.add_argument('--step-size', default=5, type=int,
                        help='epochs between evaluations (default: 5)')
    parser.add_argument('--max-epochs', default=100, type=int,
                        help='hard cap on non-warmup training epochs (default: 100)')
    parser.add_argument('--threshold', default=0.001, type=float,
                        help='mAP plateau threshold — absolute mAP (default: 0.001 = 0.1%%)')
    parser.add_argument('--output-dir', default='', type=str,
                        help='output directory (default: auto-generated under ckpt/)')
    parser.add_argument('--eval-only', default='', type=str,
                        help='evaluate existing checkpoints in DIR without training')
    parser.add_argument('--batch-size', default=-1, type=int,
                        help='override batch size (default: use config value)')
    parser.add_argument('--grad-accum', default=1, type=int,
                        help='gradient accumulation steps (default: 1)')
    parser.add_argument('--ckpt-freq', default=5, type=int,
                        help='checkpoint save frequency in epochs (default: 5)')
    parser.add_argument('-p', '--print-freq', default=10, type=int,
                        help='training print frequency (default: 10)')
    args = parser.parse_args()
    main(args)

# python imports
import argparse
import csv
import os
import time
import datetime
from pathlib import Path
from pprint import pprint

# torch imports
import torch
import torch.nn as nn
import torch.utils.data

# our code
from libs.core import load_config
from libs.datasets import make_dataset, make_data_loader
from libs.modeling import make_meta_arch
from libs.utils import (train_one_epoch, valid_one_epoch, ANETdetection,
                        save_checkpoint, make_optimizer, make_scheduler,
                        fix_random_seed, ModelEma)

TRAIN_OUTPUT = Path(__file__).resolve().parent / 'train_output'


def save_loss_csv(all_records, filepath):
    """Save per-iteration training loss records to CSV."""
    if not all_records:
        return
    keys = ['epoch', 'iteration', 'final_loss', 'cls_loss', 'reg_loss']
    with open(filepath, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=keys, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(all_records)


def plot_loss_curves(loss_records, save_dir, tag=''):
    """Generate training loss curve plot."""
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


################################################################################
def main(args):
    """main function that handles training / inference"""

    """1. setup parameters / folders"""
    # parse args
    args.start_epoch = 0
    if os.path.isfile(args.config):
        cfg = load_config(args.config)
    else:
        raise ValueError("Config file does not exist.")
    pprint(cfg)

    # batch size override
    if args.batch_size > 0:
        old_bs = cfg['loader'].get('batch_size', 8)
        cfg['loader']['batch_size'] = args.batch_size
        print(f"[Setup] Batch size override: {old_bs} -> {args.batch_size}")

    # prep for output folder (based on time stamp)
    if not os.path.exists(cfg['output_folder']):
        os.mkdir(cfg['output_folder'])
    cfg_filename = os.path.basename(args.config).replace('.yaml', '')
    if len(args.output) == 0:
        ts = datetime.datetime.fromtimestamp(int(time.time()))
        ckpt_folder = os.path.join(
            cfg['output_folder'], cfg_filename + '_' + str(ts))
    else:
        ckpt_folder = os.path.join(
            cfg['output_folder'], cfg_filename + '_' + str(args.output))
    if not os.path.exists(ckpt_folder):
        os.mkdir(ckpt_folder)

    # fix the random seeds (this will fix everything)
    rng_generator = fix_random_seed(cfg['init_rand_seed'], include_cuda=True)

    # re-scale learning rate / # workers based on number of GPUs
    cfg['opt']["learning_rate"] *= len(cfg['devices'])
    cfg['loader']['num_workers'] *= len(cfg['devices'])

    """2. create dataset / dataloader"""
    train_dataset = make_dataset(
        cfg['dataset_name'], True, cfg['train_split'], **cfg['dataset']
    )
    # update cfg based on dataset attributes (fix to epic-kitchens)
    train_db_vars = train_dataset.get_attributes()
    cfg['model']['train_cfg']['head_empty_cls'] = train_db_vars['empty_label_ids']

    # data loaders
    train_loader = make_data_loader(
        train_dataset, True, rng_generator, **cfg['loader'])

    """3. create model, optimizer, and scheduler"""
    # model
    model = make_meta_arch(cfg['model_name'], **cfg['model'])
    # not ideal for multi GPU training, ok for now
    model = nn.DataParallel(model, device_ids=[torch.device(d).index for d in cfg['devices']])
    # optimizer
    optimizer = make_optimizer(model, cfg['opt'])
    # schedule
    num_iters_per_epoch = len(train_loader)
    scheduler = make_scheduler(optimizer, cfg['opt'], num_iters_per_epoch)

    # enable model EMA
    print("Using model EMA ...")
    model_ema = ModelEma(model)

    """4. Resume from model / Misc"""
    # resume from a checkpoint?
    if args.resume:
        if os.path.isfile(args.resume):
            # load ckpt, reset epoch / best rmse
            checkpoint = torch.load(args.resume, map_location=cfg['devices'][0])
            args.start_epoch = checkpoint['epoch'] + 1
            model.load_state_dict(checkpoint['state_dict'])
            model_ema.module.load_state_dict(checkpoint['state_dict_ema'])
            # also load the optimizer / scheduler if necessary
            optimizer.load_state_dict(checkpoint['optimizer'])
            scheduler.load_state_dict(checkpoint['scheduler'])
            print("=> loaded checkpoint '{:s}' (epoch {:d}".format(
                args.resume, checkpoint['epoch']
            ))
            del checkpoint
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
            return

    # save the current config
    with open(os.path.join(ckpt_folder, 'config.txt'), 'w') as fid:
        pprint(cfg, stream=fid)
        fid.flush()

    """4. training / validation loop"""
    print("\nStart training model {:s} ...".format(cfg['model_name']))

    # start training
    max_epochs = cfg['opt'].get(
        'early_stop_epochs',
        cfg['opt']['epochs'] + cfg['opt']['warmup_epochs']
    )
    tag = f"{cfg_filename}_b{cfg['loader']['batch_size']}_ga{args.grad_accum}"
    all_loss_records = []

    for epoch in range(args.start_epoch, max_epochs):
        # train for one epoch
        records = train_one_epoch(
            train_loader,
            model,
            optimizer,
            scheduler,
            epoch,
            model_ema=model_ema,
            clip_grad_l2norm=cfg['train_cfg']['clip_grad_l2norm'],
            print_freq=args.print_freq,
            grad_accum=args.grad_accum,
            return_losses=True,
        )
        if records:
            all_loss_records.extend(records)

        # save loss CSV & plot periodically
        TRAIN_OUTPUT.mkdir(parents=True, exist_ok=True)
        save_loss_csv(all_loss_records, str(TRAIN_OUTPUT / f'loss_records_{tag}.csv'))
        plot_loss_curves(all_loss_records, str(TRAIN_OUTPUT), tag)

        # save ckpt once in a while
        if (
                (epoch == max_epochs - 1) or
                (
                        (args.ckpt_freq > 0) and
                        (epoch % args.ckpt_freq == 0) and
                        (epoch > 0)
                )
        ):
            save_states = {
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'scheduler': scheduler.state_dict(),
                'optimizer': optimizer.state_dict(),
            }

            save_states['state_dict_ema'] = model_ema.module.state_dict()
            save_checkpoint(
                save_states,
                False,
                file_folder=ckpt_folder,
                file_name='epoch_{:03d}.pth.tar'.format(epoch)
            )

    print(f"Loss plots & CSV saved to: {TRAIN_OUTPUT}")
    print("All done!")
    return

################################################################################
if __name__ == '__main__':
    """Entry Point"""
    # the arg parser
    parser = argparse.ArgumentParser(
        description='Train a point-based transformer for action localization')
    parser.add_argument('config', metavar='DIR',
                        help='path to a config file')
    parser.add_argument('-p', '--print-freq', default=10, type=int,
                        help='print frequency (default: 10 iterations)')
    parser.add_argument('-c', '--ckpt-freq', default=5, type=int,
                        help='checkpoint frequency (default: every 5 epochs)')
    parser.add_argument('--output', default='', type=str,
                        help='name of exp folder (default: none)')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to a checkpoint (default: none)')
    parser.add_argument('--batch-size', default=-1, type=int,
                        help='override batch size from config, -1 to use config value')
    parser.add_argument('--grad-accum', default=1, type=int,
                        help='gradient accumulation steps (default: 1)')
    args = parser.parse_args()
    main(args)

"""
Inspect learned channel attention weights from a trained SE/ECA checkpoint.

Usage:
  C:/Developer/Anaconda/envs/PatternRecognition/python.exe tools/inspect_se_weights.py ^
      configs/thumos_i3d_se.yaml ckpt/thumos_i3d_se_test/epoch_069.pth.tar
"""

import os as _os
_os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')

import argparse
import json
import os
import sys
from collections import defaultdict
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
from libs.modeling.blocks import SELayer, ECALayer
from libs.utils import fix_random_seed

PYTHON = r"C:\Developer\Anaconda\envs\PatternRecognition\python.exe"


def inspect_static_params(checkpoint, cfg):
    """Directly inspect the learned SE/ECA conv weights from the state dict."""
    state = checkpoint['state_dict_ema']
    layers = defaultdict(dict)

    for key, tensor in state.items():
        # Look for attention layer params
        if 'att.fc' in key or 'att.conv' in key:
            parts = key.split('.')
            # Find the layer index pattern: module.backbone.stem.X.att.fc.Y.weight
            layer_name = '.'.join(parts[:parts.index('att') + 1])
            param_name = '.'.join(parts[parts.index('att') + 1:])
            layers[layer_name][param_name] = tensor.cpu()

    return dict(layers)


def inspect_runtime_weights(cfg, checkpoint, args):
    """Run validation samples through model and capture SE attention weights."""
    device = torch.device(cfg['devices'][0])

    # build dataset
    full_dataset = make_dataset(
        cfg['dataset_name'], True, cfg['train_split'], **cfg['dataset']
    )
    rng = np.random.RandomState(42)
    indices = rng.choice(len(full_dataset), min(args.num_samples, len(full_dataset)),
                         replace=False).tolist()
    subset = Subset(full_dataset, indices)
    loader = make_data_loader(subset, False, None, 1, 1)

    # build model
    model = make_meta_arch(cfg['model_name'], **cfg['model'])
    model = nn.DataParallel(model, device_ids=[torch.device(d).index for d in cfg['devices']])
    model.load_state_dict(checkpoint['state_dict_ema'])
    model.eval()
    model = model.to(device)

    # register hooks
    att_stats = defaultdict(lambda: {'weights': [], 'type': 'SE'})
    handles = []

    for name, module in model.named_modules():
        if isinstance(module, SELayer):
            # Hook on fc (Sequential ending with Sigmoid) — output = channel weights [0,1]
            h = module.fc.register_forward_hook(
                lambda m, i, o, n=name: att_stats[n]['weights'].append(o.detach().cpu())
            )
            handles.append(h)
            att_stats[name]['type'] = 'SE'
        elif isinstance(module, ECALayer):
            # Reconstruct attention: pool → conv → sigmoid → transpose
            h = module.register_forward_hook(
                lambda m, i, o, n=name: att_stats[n]['weights'].append(
                    m.conv(m.pool(i[0]).transpose(1, 2)).sigmoid().transpose(1, 2).detach().cpu()
                )
            )
            handles.append(h)
            att_stats[name]['type'] = 'ECA'

    # forward pass
    print(f"\nRunning {len(subset)} samples through model to capture attention weights...")
    with torch.no_grad():
        for i, video_list in enumerate(loader):
            _ = model(video_list)
            if (i + 1) % 5 == 0:
                print(f"  processed {i + 1}/{len(loader)} samples")

    # clean up hooks
    for h in handles:
        h.remove()

    # aggregate stats
    results = {}
    for layer_name, data in sorted(att_stats.items()):
        if not data['weights']:
            continue
        # stack all captured weights: each is (B, C, 1)
        all_w = torch.cat([w.flatten() for w in data['weights']])  # total elements
        w_np = all_w.numpy()
        results[layer_name] = {
            'type': data['type'],
            'num_channels': data['weights'][0].shape[1],
            'num_samples': len(data['weights']),
            'mean': float(w_np.mean()),
            'std': float(w_np.std()),
            'min': float(w_np.min()),
            'max': float(w_np.max()),
            'p25': float(np.percentile(w_np, 25)),
            'p50': float(np.percentile(w_np, 50)),
            'p75': float(np.percentile(w_np, 75)),
            'frac_below_01': float((w_np < 0.1).mean()),
            'frac_below_05': float((w_np < 0.5).mean()),
            'frac_above_09': float((w_np > 0.9).mean()),
            'frac_above_05': float((w_np > 0.5).mean()),
        }

    return results


def print_summary(static_params, runtime_stats):
    """Print a human-readable summary."""
    print("\n" + "=" * 70)
    print("  SE/ECA Channel Attention — Learned Parameter Summary")
    print("=" * 70)

    if static_params:
        for layer_name, params in sorted(static_params.items()):
            print(f"\n  [{layer_name}]")
            for pname, tensor in sorted(params.items()):
                t = tensor.float()
                print(f"    {pname}: shape={list(tensor.shape)}  "
                      f"mean={t.mean():+.4f}  std={t.std():.4f}  "
                      f"min={t.min():+.4f}  max={t.max():+.4f}")

    if runtime_stats:
        print("\n" + "=" * 70)
        print("  SE/ECA Channel Attention — Runtime Weight Distribution")
        print("  (Sigmoid output, 0=suppress, 1=pass-through)")
        print("=" * 70)
        print(f"  {'Layer':<50s} {'Mean':>6s} {'Std':>6s} {'<0.1':>6s} {'>0.9':>6s} "
              f"{'Channels':>8s}")
        print(f"  {'-'*50} {'-'*6} {'-'*6} {'-'*6} {'-'*6} {'-'*8}")

        for layer_name, s in sorted(runtime_stats.items()):
            short_name = layer_name.replace('module.backbone.', '')
            print(f"  {short_name:<50s} {s['mean']:6.3f} {s['std']:6.3f} "
                  f"{s['frac_below_01']:5.1%} {s['frac_above_09']:5.1%} "
                  f"{s['num_channels']:>8d}")

        # overall assessment
        all_means = [s['mean'] for s in runtime_stats.values()]
        all_below_01 = [s['frac_below_01'] for s in runtime_stats.values()]
        all_above_09 = [s['frac_above_09'] for s in runtime_stats.values()]

        print(f"\n  Overall: mean weight = {np.mean(all_means):.3f}, "
              f"avg suppressed (<0.1) = {np.mean(all_below_01):.1%}, "
              f"avg pass-through (>0.9) = {np.mean(all_above_09):.1%}")

        if np.mean(all_means) > 0.85 and np.mean(all_below_01) < 0.05:
            print("  ⚠ Assessment: SE weights are SATURATED near 1.0 — attention is barely "
                  "modulating channels.")
            print("    This means the SE layers are essentially acting as identity, "
                  "adding parameters without benefit.")
            print("    Possible causes: insufficient training, low LR, or SE not needed "
                  "at this capacity.")
        elif np.mean(all_means) < 0.3:
            print("  ⚠ Assessment: SE weights are SUPPRESSED — attention is aggressively "
                  "killing channels, which may hurt performance.")
        else:
            print("  ✓ Assessment: SE weights show meaningful variance across channels — "
                  "attention is learning non-trivial channel importance.")


def main(args):
    cfg = load_config(args.config)
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    epoch = checkpoint.get('epoch', '?')
    print(f"Loaded checkpoint from epoch {epoch}")

    # 1. Static parameter inspection
    print("\n--- Static Parameter Inspection ---")
    static_params = inspect_static_params(checkpoint, cfg)
    print_summary(static_params, {})

    # 2. Runtime weight inspection (forward pass with hooks)
    if not args.params_only:
        print("\n--- Runtime Weight Inspection ---")
        _ = fix_random_seed(0, include_cuda=True)
        runtime_stats = inspect_runtime_weights(cfg, checkpoint, args)

        # save detailed stats
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        stats_file = out_dir / f"se_stats_epoch{epoch:03d}.json"
        with open(stats_file, 'w') as f:
            json.dump(runtime_stats, f, indent=2)
        print(f"\nDetailed stats saved to: {stats_file}")

        print_summary({}, runtime_stats)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Inspect SE/ECA channel attention weights')
    parser.add_argument('config', help='path to config YAML')
    parser.add_argument('checkpoint', help='path to .pth.tar checkpoint')
    parser.add_argument('--num-samples', default=20, type=int,
                        help='number of samples for runtime inspection (default: 20)')
    parser.add_argument('--params-only', action='store_true',
                        help='only inspect static parameters, skip forward pass')
    parser.add_argument('--output-dir', default='./train_output',
                        help='output directory for stats JSON')
    args = parser.parse_args()
    main(args)

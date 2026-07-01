import os as _os
_os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')

"""
Hyperparameter random/grid search for TriDet model.

Randomly samples from the search space, trains each trial with short epochs
and progressive early stopping (same logic as train_cv.py), then ranks all
trials by best validation loss.

Usage:
  # Random search with built-in search space
  python tools/search_hyperparams.py configs/thumos_i3d.yaml \\
      --trials 30 --max-epochs 10 --step 5

  # Grid search over specific params
  python tools/search_hyperparams.py configs/thumos_i3d.yaml \\
      --trials 0 --grid "sgp_mlp_dim:384,768" --grid "k:1.5,5.0" \\
      --max-epochs 10 --step 5

  # Resume from a previous output folder
  python tools/search_hyperparams.py configs/thumos_i3d.yaml \\
      --trials 30 --max-epochs 10 --resume ./output/search_xxx

Output structure:
  output_folder/
    search_MMDD_HHMM/
      summary.csv        # all trials ranked by val_loss
      trial_000/          # per-trial config + result
      trial_001/
      ...
"""

import argparse
import csv
import itertools
import json
import os
import random
import sys
import time
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from pprint import pformat

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

# ---------------------------------------------------------------------------
# Default search space — the most performance-critical macro parameters.
# Each entry: (type, [candidates])  or  (type, (lo, hi)) for uniform sampling.
# type = 'choice' | 'uniform_int' | 'uniform_float' | 'log_uniform'
# ---------------------------------------------------------------------------
DEFAULT_SEARCH_SPACE = {
    # --- backbone / SGP ---
    "sgp_mlp_dim":         {"type": "choice",       "values": [256, 384, 512, 768, 1024]},
    "k":                   {"type": "choice",       "values": [1.3, 1.5, 2.0, 3.0, 5.0, 7.0]},
    "init_conv_vars":      {"type": "choice",       "values": [0.0, 0.1, 0.2, 0.5, 1.0]},
    "n_sgp_win_size":      {"type": "choice",       "values": [1, 3, 5, 7, -1]},
    "input_noise":         {"type": "choice",       "values": [0.0, 0.0001, 0.0005, 0.001, 0.005]},
    # --- FPN / head ---
    "head_dim":            {"type": "choice",       "values": [256, 384, 512, 768]},
    "fpn_dim":             {"type": "choice",       "values": [256, 384, 512, 768]},
    "embd_dim":            {"type": "choice",       "values": [256, 384, 512, 768]},
    # --- Trident-head ---
    "num_bins":            {"type": "choice",       "values": [8, 12, 16, 20, 24]},
    "iou_weight_power":    {"type": "choice",       "values": [0.1, 0.2, 0.5, 1.0, 1.5]},
    # --- channel attention ---
    "use_att":             {"type": "choice",       "values": [False, True]},
    "att_type":            {"type": "choice",       "values": ["SE", "ECA"]},
    "att_reduction":       {"type": "choice",       "values": [8, 16, 32]},
    # --- training ---
    "learning_rate":       {"type": "choice",       "values": [5e-5, 1e-4, 2e-4, 5e-4, 1e-3]},
    "weight_decay":        {"type": "choice",       "values": [0.01, 0.025, 0.05, 0.1]},
    "center_sample_radius": {"type": "choice",      "values": [1.0, 1.5, 2.0, 2.5]},
    "label_smoothing":     {"type": "choice",       "values": [0.0, 0.05, 0.1]},
    "droppath":            {"type": "choice",       "values": [0.0, 0.05, 0.1, 0.15, 0.2]},
}


def sample_param(param_def, rng):
    """Draw one value from a search-space definition."""
    t = param_def["type"]
    if t == "choice":
        return rng.choice(param_def["values"])
    elif t == "uniform_int":
        return rng.randint(param_def["low"], param_def["high"])
    elif t == "uniform_float":
        return rng.uniform(param_def["low"], param_def["high"])
    elif t == "log_uniform":
        lo, hi = param_def["low"], param_def["high"]
        return 10 ** rng.uniform(lo, hi)
    else:
        raise ValueError(f"Unknown param type: {t}")


def sample_config(search_space, rng):
    """Draw a full hyperparameter set (flat dict) from the search space."""
    sampled = {}
    for key, pdef in search_space.items():
        sampled[key] = sample_param(pdef, rng)
    # cross-constraint: att_type/att_reduction only relevant when use_att=True
    # (these are still sampled but will be ignored by the model when use_att=False)
    return sampled


def apply_params(cfg, params):
    """Override config values from a flat params dict (in-place)."""
    model_keys = {
        "sgp_mlp_dim", "k", "init_conv_vars", "n_sgp_win_size", "input_noise",
        "head_dim", "fpn_dim", "embd_dim", "num_bins", "iou_weight_power",
        "use_att", "att_type", "att_reduction",
    }
    train_cfg_keys = {"center_sample_radius", "label_smoothing", "droppath"}
    opt_keys = {"learning_rate", "weight_decay"}

    for k, v in params.items():
        if k in model_keys:
            cfg["model"][k] = v
        elif k in train_cfg_keys:
            cfg["train_cfg"][k] = v
        elif k in opt_keys:
            cfg["opt"][k] = v

    # also ensure n_sgp_win_size is per-level if needed
    return cfg


def build_search_space(args):
    """Build the effective search space: default overridden by --grid entries."""
    space = deepcopy(DEFAULT_SEARCH_SPACE)
    if args.tune:
        # --tune limits to the params explicitly listed (comma-separated)
        keep = set(k.strip() for k in args.tune.split(","))
        space = {k: v for k, v in space.items() if k in keep}
        if not space:
            print("[ERROR] --tune filtered all params. Check your list.")
            sys.exit(1)
    if args.preset:
        # --preset replaces the space entirely with a named subset
        presets = _build_presets()
        if args.preset not in presets:
            print(f"[ERROR] Unknown preset '{args.preset}'. Available: {list(presets.keys())}")
            sys.exit(1)
        space = presets[args.preset]
    return space


def _build_presets():
    """Named subsets of parameters for targeted searches."""
    return {
        "sgp": {
            "sgp_mlp_dim":    {"type": "choice", "values": [256, 384, 512, 768, 1024]},
            "k":              {"type": "choice", "values": [1.3, 1.5, 2.0, 3.0, 5.0, 7.0]},
            "init_conv_vars": {"type": "choice", "values": [0.0, 0.1, 0.2, 0.5, 1.0]},
            "n_sgp_win_size": {"type": "choice", "values": [1, 3, 5, 7, -1]},
        },
        "detection": {
            "num_bins":         {"type": "choice", "values": [8, 12, 16, 20, 24]},
            "iou_weight_power": {"type": "choice", "values": [0.1, 0.2, 0.5, 1.0, 1.5]},
            "center_sample_radius": {"type": "choice", "values": [1.0, 1.5, 2.0, 2.5]},
            "label_smoothing":  {"type": "choice", "values": [0.0, 0.05, 0.1]},
        },
        "attention": {
            "use_att":       {"type": "choice", "values": [False, True]},
            "att_type":      {"type": "choice", "values": ["SE", "ECA"]},
            "att_reduction": {"type": "choice", "values": [8, 16, 32]},
        },
        "optimization": {
            "learning_rate": {"type": "choice", "values": [5e-5, 1e-4, 2e-4, 5e-4, 1e-3]},
            "weight_decay":  {"type": "choice", "values": [0.01, 0.025, 0.05, 0.1]},
            "droppath":      {"type": "choice", "values": [0.0, 0.05, 0.1, 0.15, 0.2]},
        },
        "capacity": {
            "head_dim": {"type": "choice", "values": [256, 384, 512, 768]},
            "fpn_dim":  {"type": "choice", "values": [256, 384, 512, 768]},
            "embd_dim": {"type": "choice", "values": [256, 384, 512, 768]},
            "sgp_mlp_dim": {"type": "choice", "values": [256, 384, 512, 768, 1024]},
        },
        "noise": {
            "input_noise":    {"type": "choice", "values": [0.0, 0.0001, 0.0005, 0.001, 0.005]},
            "init_conv_vars": {"type": "choice", "values": [0.0, 0.1, 0.2, 0.5, 1.0]},
        },
    }


def generate_trials(search_space, args, rng):
    """Generate the list of trial parameter dicts (random or grid)."""
    if args.trials > 0:
        # random search
        return [sample_config(search_space, rng) for _ in range(args.trials)]
    else:
        # grid search over --grid entries only
        grid_space = {}
        for entry in args.grid:
            key, vals = entry.split(":", 1)
            key = key.strip()
            vals = [v.strip() for v in vals.split(",")]
            # infer type from default space or guess
            if key in search_space and search_space[key]["type"] == "choice":
                # try to keep the same type
                pdef = search_space[key]
                if all(v.replace(".", "").replace("-", "").isdigit() for v in vals if v not in ("True", "False")):
                    # numeric
                    if any("." in v for v in vals):
                        vals = [float(v) for v in vals]
                    else:
                        vals = [int(v) for v in vals]
                grid_space[key] = vals
            else:
                # guess numeric
                numeric_vals = []
                for v in vals:
                    if v in ("True", "False"):
                        numeric_vals.append(v == "True")
                    else:
                        numeric_vals.append(float(v) if "." in v else int(v))
                grid_space[key] = numeric_vals

        keys = list(grid_space.keys())
        combos = list(itertools.product(*grid_space.values()))
        print(f"[Grid] {len(combos)} combinations from {keys}")
        return [dict(zip(keys, combo)) for combo in combos]


@torch.no_grad()
def compute_val_loss(model, val_loader):
    model.train()
    meter = AverageMeter()
    for video_list in val_loader:
        losses = model(video_list)
        meter.update(losses["final_loss"].item(), len(video_list))
    return meter.avg


def check_early_stop(block_losses, threshold=0.01, patience=3):
    if len(block_losses) < patience + 1:
        return False
    recent = block_losses[-(patience + 1):]
    stag = sum(
        (recent[i - 1] - recent[i]) / max(recent[i - 1], 1e-8) < threshold
        for i in range(1, len(recent))
    )
    return stag >= patience


def train_one_trial(cfg, train_indices, val_indices, trial_dir, args, rng):
    """Run one trial, return (best_val_loss, best_epoch, stopped_epoch)."""
    full_dataset = make_dataset(
        cfg["dataset_name"], True, cfg["train_split"], **cfg["dataset"]
    )
    train_ds = Subset(full_dataset, train_indices)
    val_ds = Subset(full_dataset, val_indices)

    train_loader = make_data_loader(
        train_ds, True, rng, cfg["loader"]["batch_size"], cfg["loader"]["num_workers"]
    )
    val_loader = make_data_loader(
        val_ds, False, None,
        cfg["loader"]["batch_size"], max(cfg["loader"]["num_workers"] // 2, 1)
    )

    model = make_meta_arch(cfg["model_name"], **cfg["model"])
    model = nn.DataParallel(model, device_ids=[torch.device(d).index for d in cfg["devices"]])
    model_ema = ModelEma(model)

    optimizer = make_optimizer(model, cfg["opt"])
    num_iters_per_epoch = len(train_loader)
    scheduler = make_scheduler(optimizer, cfg["opt"], num_iters_per_epoch)

    block_losses = []
    best_val_loss = float("inf")
    best_epoch = 0
    total_epochs = cfg["opt"]["epochs"] + cfg["opt"]["warmup_epochs"]

    for block_start in range(0, total_epochs, args.step):
        block_end = min(block_start + args.step, total_epochs)

        for epoch in range(block_start, block_end):
            train_one_epoch(
                train_loader, model, optimizer, scheduler, epoch,
                model_ema=model_ema,
                clip_grad_l2norm=cfg["train_cfg"]["clip_grad_l2norm"],
                print_freq=args.print_freq,
            )

        val_loss = compute_val_loss(model_ema.module, val_loader)
        block_losses.append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = block_end
            torch.save(model_ema.module.state_dict(), trial_dir / "best_model.pth")

        if np.isnan(val_loss) or val_loss > 1e6:
            print(f"  [WARN] Loss unstable ({val_loss:.2f}), stopping trial.")
            break
        if check_early_stop(block_losses, args.threshold, args.patience):
            break

    return {
        "best_val_loss": float(best_val_loss),
        "best_epoch": best_epoch,
        "stopped_epoch": block_end,
        "block_losses": [float(v) for v in block_losses],
    }


def main(args):
    base_cfg = load_config(args.config)

    # override epochs
    if args.max_epochs > 0:
        base_cfg["opt"]["epochs"] = args.max_epochs
        total = args.max_epochs + base_cfg["opt"]["warmup_epochs"]
        print(f"[Setup] Training epochs: {args.max_epochs} + {base_cfg['opt']['warmup_epochs']} warmup = {total}")
    else:
        args.max_epochs = base_cfg["opt"]["epochs"]

    # scale lr by devices
    base_cfg["opt"]["learning_rate"] *= len(base_cfg["devices"])
    base_cfg["loader"]["num_workers"] *= len(base_cfg["devices"])

    # build search space
    search_space = build_search_space(args)
    print(f"[Search space] {len(search_space)} parameters: {list(search_space.keys())}")

    # setup output
    ts = datetime.fromtimestamp(int(time.time()))
    cfg_name = os.path.basename(args.config).replace(".yaml", "")
    out_dir = Path(base_cfg.get("output_folder", "./ckpt"))
    exp_folder = out_dir / f"{cfg_name}_hparam_{ts.strftime('%m%d_%H%M')}"
    exp_folder.mkdir(parents=True, exist_ok=True)

    # save search-space
    with open(exp_folder / "search_space.json", "w") as f:
        json.dump(search_space, f, indent=2, default=str)

    # generate trials
    rng = random.Random(base_cfg["init_rand_seed"])
    trials = generate_trials(search_space, args, rng)
    if len(trials) == 0:
        print("[ERROR] No trials generated. Use --trials N for random search or --grid for grid search.")
        sys.exit(1)

    print(f"[Trials] {len(trials)} total")
    if args.limit > 0:
        trials = trials[:args.limit]
        print(f"[Trials] limited to first {args.limit}")

    # split data once (same split for all trials)
    full_dataset = make_dataset(
        base_cfg["dataset_name"], True, base_cfg["train_split"], **base_cfg["dataset"]
    )
    n_samples = len(full_dataset)
    perm = np.random.RandomState(base_cfg["init_rand_seed"]).permutation(n_samples)
    val_n = max(int(n_samples * args.val_ratio), 1)
    val_indices = perm[:val_n].tolist()
    train_indices = perm[val_n:].tolist()
    print(f"[Data] Train={len(train_indices)}, Val={len(val_indices)}")

    # run trials
    results = []
    summary_path = exp_folder / "summary.csv"
    fieldnames = list(search_space.keys()) + [
        "trial", "best_val_loss", "best_epoch", "stopped_epoch", "duration_s"
    ]

    for i, params in enumerate(trials):
        trial_dir = exp_folder / f"trial_{i:03d}"
        trial_dir.mkdir(parents=True, exist_ok=True)

        # skip already-done trials (for resume)
        if (trial_dir / "result.json").exists():
            with open(trial_dir / "result.json") as f:
                result = json.load(f)
            print(f"[{i+1:3d}/{len(trials):3d}] SKIP (already done)  "
                  f"best_loss={result['best_val_loss']:.4f}")
            row = {**params, "trial": i, **result}
            results.append(row)
            continue

        # build per-trial config
        trial_cfg = deepcopy(base_cfg)
        apply_params(trial_cfg, params)

        # log config
        with open(trial_dir / "config.txt", "w", encoding="utf-8") as f:
            f.write(pformat(trial_cfg))
        with open(trial_dir / "params.json", "w") as f:
            json.dump(params, f, indent=2)

        print(f"\n{'='*60}")
        print(f"[{i+1:3d}/{len(trials):3d}] {params}")
        print(f"{'='*60}")

        trial_rng = fix_random_seed(base_cfg["init_rand_seed"] + i, include_cuda=True)
        t_start = time.time()
        result = train_one_trial(trial_cfg, train_indices, val_indices, trial_dir, args, trial_rng)
        result["duration_s"] = round(time.time() - t_start, 1)

        print(f"  => best_val_loss={result['best_val_loss']:.4f}  "
              f"best_epoch={result['best_epoch']}  stopped={result['stopped_epoch']}  "
              f"time={result['duration_s']:.0f}s")

        with open(trial_dir / "result.json", "w") as f:
            json.dump(result, f, indent=2)

        row = {**params, "trial": i, **result}
        results.append(row)

        # write incremental summary
        sorted_results = sorted(results, key=lambda r: r["best_val_loss"])
        with open(summary_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(sorted_results)

        # print current top 3
        print(f"  [Top-3 so far]")
        for rank, r in enumerate(sorted_results[:3]):
            print(f"    #{rank+1} trial={r['trial']:03d}  val_loss={r['best_val_loss']:.4f}  "
                  f"epoch={r['best_epoch']}")

    # final summary
    sorted_results = sorted(results, key=lambda r: r["best_val_loss"])
    print(f"\n{'='*60}")
    print(f"  Search complete — {len(results)} trials")
    print(f"  Best: trial={sorted_results[0]['trial']:03d}  "
          f"val_loss={sorted_results[0]['best_val_loss']:.4f}")
    print(f"  Results: {exp_folder}")
    print(f"{'='*60}")

    # print best params
    print("\n  Best params:")
    for k in search_space:
        print(f"    {k}: {sorted_results[0].get(k, 'N/A')}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Hyperparameter search for TriDet model")
    parser.add_argument("config", help="path to base config YAML")
    parser.add_argument("--trials", default=30, type=int,
                        help="number of random trials (0 = use --grid for grid search)")
    parser.add_argument("--limit", default=-1, type=int,
                        help="limit to first N trials (-1 = all)")
    parser.add_argument("--max-epochs", default=-1, type=int,
                        help="training epochs per trial, -1 uses config value")
    parser.add_argument("--step", default=5, type=int,
                        help="validation check every N epochs")
    parser.add_argument("--val-ratio", default=0.1, type=float,
                        help="validation split ratio")
    parser.add_argument("--threshold", default=0.01, type=float,
                        help="early-stop improvement threshold")
    parser.add_argument("--patience", default=3, type=int,
                        help="early-stop patience in blocks")
    parser.add_argument("--print-freq", default=10, type=int,
                        help="print frequency in iterations")
    parser.add_argument("--grid", action="append", default=[],
                        help="grid search entry: 'param:val1,val2,...' (repeatable)")
    parser.add_argument("--tune", default="", type=str,
                        help="comma-separated param names to restrict search (others use config defaults)")
    parser.add_argument("--preset", default="", type=str,
                        help="use a named preset space: sgp|detection|attention|optimization|capacity|noise")
    parser.add_argument("--output", default="", type=str,
                        help="override output folder")
    args = parser.parse_args()
    main(args)

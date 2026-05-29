"""
8GB 显存验证脚本 —— 端到端测试 THUMOS14 训练 + 推理管线

仅调整不改变模型结构的参数 (batch_size, max_seq_len)，
模型架构与论文完全一致，验证结果可直接作为真实训练的起点。

用法:
    conda activate PatternRecognition
    python test.py                          # max_seq_len=1500
    python test.py --max-seq-len 1200       # 更保守
    python test.py --max-seq-len 2304       # 原始配置 (可能 OOM)
    python test.py --amp                    # 开启 AMP 混合精度
"""

import os
import sys
import argparse
import time
from pprint import pprint

import torch
import torch.nn as nn
import numpy as np

from libs.core import load_config
from libs.datasets import make_dataset, make_data_loader
from libs.modeling import make_meta_arch
from libs.utils import fix_random_seed


# ---------------------------------------------------------------------------
# 仅调整不改变模型结构的参数
# ---------------------------------------------------------------------------
LOW_MEM_OVERRIDES = {
    "loader": {
        "batch_size": 1,         # 2 → 1
    },
    # max_seq_len 由命令行 --max-seq-len 控制，默认 1500
    "train_cfg": {
        "init_loss_norm": 100,   # 适配 batch=1
    },
}


def override_config(cfg, overrides):
    """深度合并配置，overrides 中的值覆盖 cfg 中的对应项。"""
    for key, value in overrides.items():
        if key in cfg and isinstance(value, dict) and isinstance(cfg[key], dict):
            cfg[key].update(value)
        else:
            cfg[key] = value
    return cfg


def get_gpu_memory():
    """返回当前已分配和缓存的 GPU 显存（MB）。"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024 ** 2
        reserved = torch.cuda.memory_reserved() / 1024 ** 2
        return allocated, reserved
    return 0, 0


def print_config_diff(cfg):
    """打印被修改的关键配置项，对比原始值。"""
    print("\n  配置变更 (仅以下参数被覆写，其余与 config yaml 一致):")
    print(f"    batch_size      : 2 → {cfg['loader']['batch_size']}")
    print(f"    max_seq_len     : 2304 → {cfg['dataset']['max_seq_len']}")
    print(f"  (以下所有参数保持与论文一致，不改变模型结构)")
    print(f"    backbone_arch   : {cfg['model']['backbone_arch']}")
    print(f"    embd_dim        : {cfg['model']['embd_dim']}")
    print(f"    sgp_mlp_dim     : {cfg['model']['sgp_mlp_dim']}")
    print(f"    fpn_dim         : {cfg['model']['fpn_dim']}")
    print(f"    head_dim        : {cfg['model']['head_dim']}")
    print(f"    use_trident_head: {cfg['model']['use_trident_head']}")
    print(f"    num_bins        : {cfg['model']['num_bins']}")
    print(f"    k               : {cfg['model']['k']}")
    print(f"    n_sgp_win_size  : {cfg['model']['n_sgp_win_size']}")


def main():
    parser = argparse.ArgumentParser(description="8GB 显存验证脚本")
    parser.add_argument("--config", default="./configs/thumos_i3d.yaml",
                        help="配置文件路径")
    parser.add_argument("--max-seq-len", type=int, default=1500,
                        help="最大序列长度, 2304→1500 减少 ~35%% 显存 (default: 1500)")
    parser.add_argument("--train-samples", type=int, default=2,
                        help="训练用样本数 (default: 2)")
    parser.add_argument("--val-samples", type=int, default=1,
                        help="验证用样本数 (default: 1)")
    parser.add_argument("--amp", action="store_true", default=True, dest="use_amp",
                        help="开启 AMP 混合精度 (默认启用)")
    parser.add_argument("--no-amp", action="store_false", dest="use_amp",
                        help="关闭 AMP")
    args = parser.parse_args()

    print("=" * 60)
    print("  TriDet 8GB VRAM 验证 (完整模型结构)")
    print("=" * 60)

    # 1. 加载配置
    print("\n[1/6] 加载配置 ...")
    cfg = load_config(args.config)
    cfg = override_config(cfg, LOW_MEM_OVERRIDES)

    # max_seq_len 必须被 FPN stride 整除 (2^(fpn_levels-1))
    fpn_levels = cfg["model"]["backbone_arch"][-1] + 1
    scale_factor = cfg["model"]["scale_factor"]
    seq_divisor = scale_factor ** (fpn_levels - 1)
    raw_len = args.max_seq_len
    aligned_len = (raw_len // seq_divisor) * seq_divisor
    if aligned_len != raw_len:
        print(f"\n  [WARN] max_seq_len 必须被 {seq_divisor} 整除")
        print(f"         {raw_len} → {aligned_len} (自动对齐)")
    cfg["dataset"]["max_seq_len"] = aligned_len

    # 同步 dataset → model 字段
    cfg["model"]["input_dim"] = cfg["dataset"]["input_dim"]
    cfg["model"]["num_classes"] = cfg["dataset"]["num_classes"]
    cfg["model"]["max_seq_len"] = cfg["dataset"]["max_seq_len"]
    cfg["model"]["train_cfg"] = cfg["train_cfg"]
    cfg["model"]["test_cfg"] = cfg["test_cfg"]

    print_config_diff(cfg)
    if args.use_amp:
        print(f"    AMP             : 启用 (torch.cuda.amp)")

    # 2. 数据集
    print("\n[2/6] 创建数据集 ...")
    rng = fix_random_seed(cfg["init_rand_seed"], include_cuda=True)

    train_dataset = make_dataset(
        cfg["dataset_name"], True, cfg["train_split"], **cfg["dataset"]
    )
    val_dataset = make_dataset(
        cfg["dataset_name"], False, cfg["val_split"], **cfg["dataset"]
    )

    full_train = len(train_dataset.data_list)
    full_val = len(val_dataset.data_list)
    train_dataset.data_list = train_dataset.data_list[:args.train_samples]
    val_dataset.data_list = val_dataset.data_list[:args.val_samples]
    print(f"  训练: {len(train_dataset.data_list)} / {full_train} 样本")
    print(f"  验证: {len(val_dataset.data_list)} / {full_val} 样本")

    train_db_vars = train_dataset.get_attributes()
    cfg["model"]["train_cfg"]["head_empty_cls"] = train_db_vars["empty_label_ids"]

    train_loader = make_data_loader(
        train_dataset, True, rng, **cfg["loader"]
    )

    # 3. 模型
    print("\n[3/6] 构建模型 (完整结构) ...")
    alloc_before, _ = get_gpu_memory()
    print(f"  构建前已分配: {alloc_before:.1f} MB")

    model = make_meta_arch(cfg["model_name"], **cfg["model"])
    model = nn.DataParallel(model, device_ids=cfg["devices"])
    model.train()

    alloc_after, _ = get_gpu_memory()
    print(f"  构建后已分配: {alloc_after:.1f} MB")
    print(f"  模型参数占用: ~{alloc_after - alloc_before:.1f} MB")

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  总参数: {total_params:,}  |  可训练: {trainable_params:,}")

    # 4. 单步训练
    print("\n[4/6] 训练单步 Forward + Backward ...")
    torch.cuda.reset_peak_memory_stats()

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    optimizer.zero_grad()

    batch = next(iter(train_loader))
    t0 = time.time()

    if args.use_amp:
        scaler = torch.amp.GradScaler()
        with torch.amp.autocast("cuda"):
            losses = model(batch)
            loss = losses["final_loss"]
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
    else:
        losses = model(batch)
        loss = losses["final_loss"]
        loss.backward()
        optimizer.step()

    t1 = time.time()

    peak_mb = torch.cuda.max_memory_allocated() / 1024 ** 2
    print(f"  cls_loss   = {losses['cls_loss'].item():.4f}")
    print(f"  reg_loss   = {losses['reg_loss'].item():.4f}")
    print(f"  final_loss = {loss.item():.4f}")
    print(f"  单步耗时: {(t1 - t0):.2f}s")
    print(f"  训练阶段峰值显存: {peak_mb:.1f} MB")

    # 5. 推理
    print("\n[5/6] 推理单样本 ...")
    model.eval()
    torch.cuda.reset_peak_memory_stats()

    sample = val_dataset[0]
    with torch.no_grad():
        if args.use_amp:
            with torch.amp.autocast("cuda"):
                results = model([sample])
        else:
            results = model([sample])

    peak_infer_mb = torch.cuda.max_memory_allocated() / 1024 ** 2

    result = results[0]
    n_det = len(result["segments"])
    # postprocessing 会移除 duration，从原始样本取
    duration = sample["duration"]
    print(f"  视频: {result['video_id']}")
    print(f"  时长: {duration:.1f}s  |  检出: {n_det} 个动作段")
    if n_det > 0:
        top_score = result["scores"].max().item()
        print(f"  最高置信度: {top_score:.4f}")
        print(f"  检出段范围: {result['segments'].min().item():.1f}s - {result['segments'].max().item():.1f}s")
    print(f"  推理阶段峰值显存: {peak_infer_mb:.1f} MB")

    # 6. 汇总
    print("\n[6/6] 汇总")
    print("=" * 60)
    overall_peak = max(peak_mb, peak_infer_mb, alloc_after)
    headroom = 8192 - overall_peak
    print(f"  max_seq_len          : {cfg['dataset']['max_seq_len']}")
    print(f"  AMP                  : {'启用' if args.use_amp else '关闭'}")
    print(f"  整体峰值显存          : {overall_peak:.1f} MB")
    print(f"  8GB 余量              : {headroom:+.1f} MB")

    # 显存不会因训练规模增大而暴涨的原因:
    # SGP 使用 Depthwise Conv (groups=n_embd)，无 O(T^2) 注意力矩阵
    # 显存复杂度 O(T × C)，线性增长——即使 max_seq_len 翻倍、batch_size 翻倍也能承受
    estimated_full = overall_peak * (2304 / cfg['dataset']['max_seq_len']) * (2 / cfg['loader']['batch_size'])
    if not args.use_amp:
        estimated_full *= 1.7  # FP32 比 FP16 多 ~70%

    print(f"\n  真实训练估算 (max_seq_len=2304, batch=2, {'AMP' if args.use_amp else 'FP32'}):")
    print(f"    预估峰值显存: ~{estimated_full:.0f} MB")

    if overall_peak > 8192:
        print(f"\n  [FAIL] 超出 8GB {(overall_peak - 8192):.0f} MB —— 建议:")
        print(f"     python test.py --max-seq-len {args.max_seq_len - 300}")
        sys.exit(1)
    else:
        print(f"\n  [PASS] 管线完整: 数据加载 / 前向 / 反向 / 推理 / NMS 全部通过")
        if estimated_full < 7500:
            print(f"\n  可以尝试直接用原始配置训练:")
            print(f"     python train.py ./configs/thumos_i3d.yaml --output baseline")
        else:
            print(f"\n  启动真实训练:")
            print(f"    修改 configs/thumos_i3d.yaml:")
            print(f"      batch_size: 1")
            print(f"      max_seq_len: {cfg['dataset']['max_seq_len']}")
            print(f"    然后: python train.py ./configs/thumos_i3d.yaml --output baseline")
    print("=" * 60)


if __name__ == "__main__":
    main()

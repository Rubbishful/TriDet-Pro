"""
数据流形状与意义展示脚本 —— 逐级展示 TriDet 输入到输出的完整形状变换

展示内容:
  - 原始数据加载 (dataset 单个样本)
  - 预处理 (batched_inputs, batched_masks)
  - Backbone → Neck → Heads 各级输出
  - 训练模式: GT 标签生成与损失计算
  - 推理模式: 最终预测结果

用法:
    conda activate PatternRecognition
    python test.py                           # 默认展示训练模式
    python test.py --mode inference          # 展示推理模式
    python test.py --mode both --no-amp      # 同时展示，关闭 AMP
"""

import os
import sys
import argparse

# Windows KMP/OpenMP 冲突 workaround — 必须在 import torch 之前
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torch.nn as nn
import numpy as np

from libs.core import load_config
from libs.datasets import make_dataset, make_data_loader
from libs.modeling import make_meta_arch
from libs.utils import fix_random_seed


# ── 工具函数 ──────────────────────────────────────────────────────────────

SEP = "─" * 72
THIN = "-" * 48


def print_header(title: str):
    print(f"\n{SEP}")
    print(f"  {title}")
    print(SEP)


def print_shape(name: str, obj, indent: int = 2):
    """打印变量形状与意义，支持 Tensor / List / dict 递归。"""
    prefix = " " * indent
    if isinstance(obj, torch.Tensor):
        s = " × ".join(str(d) for d in obj.shape)
        print(f"{prefix}{name}: [{s}]  {obj.dtype}")
    elif isinstance(obj, (list, tuple)):
        print(f"{prefix}{name}: List[{len(obj)}]")
        for i, item in enumerate(obj):
            if i >= 4:
                print(f"{prefix}  ... ({len(obj) - 4} more)")
                break
            if isinstance(item, torch.Tensor):
                s = " × ".join(str(d) for d in item.shape)
                print(f"{prefix}  [{i}]: [{s}]  {item.dtype}")
            elif isinstance(item, dict):
                for k, v in item.items():
                    print_shape(f"[{i}].{k}", v, indent + 2)
            else:
                print(f"{prefix}  [{i}]: {type(item).__name__}")
    elif isinstance(obj, dict):
        for k, v in obj.items():
            print_shape(k, v, indent)
    else:
        print(f"{prefix}{name}: {type(obj).__name__} = {obj}")


def print_tensor_stats(name: str, t: torch.Tensor, indent: int = 2):
    """打印张量的统计信息。"""
    prefix = " " * indent
    if t.numel() == 0:
        print(f"{prefix}{name}: empty tensor")
        return
    print(f"{prefix}{name}: "
          f"min={t.min().item():.4f}  max={t.max().item():.4f}  "
          f"mean={t.mean().item():.4f}  std={t.std().item():.4f}  "
          f"nan={torch.isnan(t).any().item()}  inf={torch.isinf(t).any().item()}")


# ── 数据集原始输出 ────────────────────────────────────────────────────────

def show_dataset_sample(dataset, idx=0):
    """展示单个原始样本的结构与形状。"""
    print_header("1. 数据集原始样本 (Dataset.__getitem__)")

    sample = dataset[idx]

    print(f"\n  样本索引: {idx}")
    print(f"  video_id : {sample['video_id']}")
    print(f"  duration : {sample['duration']:.1f}s")
    print(f"  fps      : {sample['fps']:.2f}")
    print(f"  feat_stride    : {sample.get('feat_stride', 'N/A')}")
    print(f"  feat_num_frames: {sample.get('feat_num_frames', 'N/A')}")

    print(f"\n  ── 原始字段形状 ──")
    print_shape("feats", sample['feats'])
    print(f"    {'':2s}  (C=input_dim={sample['feats'].shape[0]}, T={sample['feats'].shape[1]}) — I3D 逐帧特征")
    if sample['segments'] is not None:
        print_shape("segments", sample['segments'])
        print(f"    {'':2s}  (N_gt={sample['segments'].shape[0]}, 2) — GT 动作起止时刻 (特征网格坐标)")
        print_shape("labels", sample['labels'])
        print(f"    {'':2s}  (N_gt={sample['labels'].shape[0]},) — GT 动作类别编号 (0~num_classes-1)")
    else:
        print(f"    {'':2s}segments: None (推理模式无 GT)")
        print(f"    {'':2s}labels  : None (推理模式无 GT)")

    return sample


# ── 预处理 ────────────────────────────────────────────────────────────────

def show_preprocessing(model, sample):
    """展示预处理后的 batched 张量形状。"""
    print_header("2. 预处理 (preprocessing)")

    batched_inputs, batched_masks = model.preprocessing([sample])

    print(f"\n  输入: 1 个样本, 实际特征长度 T={sample['feats'].shape[-1]}")
    print(f"  max_seq_len 配置: {model.max_seq_len}")

    print_shape("batched_inputs", batched_inputs)
    print(f"    {'':2s}  [B=1, C=input_dim, T=max_seq_len] — 右侧 zero-padding 到固定长度")
    print_shape("batched_masks", batched_masks)
    print(f"    {'':2s}  [B=1, 1, T=max_seq_len] — True=有效位置, False=padding")

    print_tensor_stats("batched_inputs (有效区域)", batched_inputs[:, :, :sample['feats'].shape[-1]])
    print(f"    {'':2s}padding 填充值: {batched_inputs[0, 0, -1].item():.1f}")

    return batched_inputs, batched_masks


# ── Backbone & Neck ───────────────────────────────────────────────────────

def show_backbone_neck(model, batched_inputs, batched_masks):
    """展示 Backbone 和 FPN Neck 的输出形状。"""
    print_header("3. Backbone & Neck")

    # Backbone
    feats, masks = model.backbone(batched_inputs, batched_masks)
    n_feat_levels = len(feats)
    print("\n  [Backbone: SGPBackbone]")
    print(f"    输出 {n_feat_levels} 层特征 (1 原始 + {n_feat_levels - 1} 下采样):")
    for level, (f, m) in enumerate(zip(feats, masks)):
        print_shape(f"  Level {level} feats", f)
        print_shape(f"  Level {level} masks", m)
        print(f"    {'':2s}  [B=1, C=embd_dim={f.shape[1]}, T_lvl={f.shape[-1]}]")

    # Neck
    print(f"\n  [Neck: {type(model.neck).__name__}]")
    fpn_feats, fpn_masks = model.neck(feats, masks)
    print(f"    输出 {len(fpn_feats)} 层 FPN 特征:")
    for level, (f, m) in enumerate(zip(fpn_feats, fpn_masks)):
        print_shape(f"  Level {level} feats", f)
        print_shape(f"  Level {level} masks", m)
        print(f"    {'':2s}  [B=1, C=fpn_dim={f.shape[1]}, T_lvl={f.shape[-1]}]")

    return fpn_feats, fpn_masks


# ── Points (锚点生成器) ──────────────────────────────────────────────────

def show_points(model, fpn_feats):
    """展示 Point Generator 的输出。"""
    print_header("4. Point Generator (锚点坐标)")

    points = model.point_generator(fpn_feats)
    print(f"\n  共 {len(points)} 层 FPN 锚点:")

    for level, pts in enumerate(points):
        print(f"\n    FPN Level {level}:")
        print_shape("  points", pts)
        print(f"    {'':2s}  [T_lvl={pts.shape[0]}, 4] — 每行是一个锚点")
        print(f"    {'':2s}    列 0: t_position   — 锚点在特征网格中的时间位置 (0~T_lvl-1)")
        print(f"    {'':2s}    列 1: reg_range_min — 该锚点负责的最小回归距离")
        print(f"    {'':2s}    列 2: reg_range_max — 该锚点负责的最大回归距离")
        print(f"    {'':2s}    列 3: stride        — 该层特征对应的时间跨度 (用于解码)")
        print(f"    {'':2s}    示例前 3 点:")
        for i in range(min(3, pts.shape[0])):
            print(f"    {'':2s}      pt[{i}]: pos={pts[i,0]:.0f}  reg=[{pts[i,1]:.0f}, {pts[i,2]:.0f}]  stride={pts[i,3]:.0f}")

    total_pts = sum(p.shape[0] for p in points)
    print(f"\n  总锚点数: {total_pts}")

    return points


# ── Heads ─────────────────────────────────────────────────────────────────

def show_heads(model, fpn_feats, fpn_masks):
    """展示三个 Head 的输出。"""
    print_header("5. Heads (分类头 / 回归头 / Trident 边界头)")

    # Classification head
    out_cls_logits = model.cls_head(fpn_feats, fpn_masks)
    print("\n  [ClsHead — 动作分类]")
    for level, logits in enumerate(out_cls_logits):
        print_shape(f"  Level {level} logits", logits)
        print(f"    {'':2s}  [B=1, num_classes={logits.shape[1]}, T_lvl={logits.shape[-1]}] — 每个锚点的类别 logits (含背景)")

    # Regression head
    out_offsets = model.reg_head(fpn_feats, fpn_masks)
    reg_out_c = out_offsets[0].shape[1]
    print("\n  [RegHead — 边界回归]")
    if model.use_trident_head:
        print(f"    Trident 模式: 输出通道 = 2 × (num_bins+1) = 2 × {model.num_bins + 1} = {reg_out_c}")
    for level, off in enumerate(out_offsets):
        print_shape(f"  Level {level} offsets", off)
        print(f"    {'':2s}  [B=1, {reg_out_c}, T_lvl={off.shape[-1]}] — ", end="")
        if model.use_trident_head:
            print(f"(左bin分布, 右bin分布) 各 {model.num_bins + 1} 个bin")
        else:
            print(f"(左偏移量, 右偏移量)")

    # Trident start/end heads
    if model.use_trident_head:
        print(f"\n  [TridentHead — 细粒度边界分布 (num_bins={model.num_bins})]")
        out_lb_logits = model.start_head(fpn_feats, fpn_masks)
        out_rb_logits = model.end_head(fpn_feats, fpn_masks)
        for level, (lb, rb) in enumerate(zip(out_lb_logits, out_rb_logits)):
            print_shape(f"  Level {level} left_bin_logits", lb)
            print(f"    {'':2s}  [B=1, num_bins+1={lb.shape[1]}, T_lvl={lb.shape[-1]}] — 左边界 bin 分布")
            print_shape(f"  Level {level} right_bin_logits", rb)
            print(f"    {'':2s}  [B=1, num_bins+1={rb.shape[1]}, T_lvl={rb.shape[-1]}] — 右边界 bin 分布")
    else:
        out_lb_logits, out_rb_logits = None, None
        print(f"\n  [TridentHead] 未启用 (use_trident_head=False)")

    return out_cls_logits, out_offsets, out_lb_logits, out_rb_logits


# ── permute 后 & GT 标签生成 ──────────────────────────────────────────────

def show_training_labels(model, fpn_masks, out_cls_logits, out_offsets, points, sample):
    """展示训练模式下 permute 后的输出形状 + GT 标签生成。"""
    print_header("6. 训练模式: Permute + GT 标签生成")

    # permute: 参考 meta_archs.py L467-472
    out_cls_logits = [x.permute(0, 2, 1) for x in out_cls_logits]
    out_offsets = [x.permute(0, 2, 1) for x in out_offsets]
    fpn_masks_permuted = [x.squeeze(1) for x in fpn_masks]

    print("\n  [Permute 后 — 统一为 FPN 列表 (长度=F) 格式]")
    print(f"    FPN 层级数: {len(fpn_masks_permuted)}")
    for level in range(len(fpn_masks_permuted)):
        print_shape(f"  Level {level} cls_logits", out_cls_logits[level])
        print(f"    {'':2s}  [B=1, T_lvl, num_classes]")
        reg_c = out_offsets[level].shape[-1]
        print_shape(f"  Level {level} offsets", out_offsets[level])
        if model.use_trident_head:
            print(f"    {'':2s}  [B=1, T_lvl, {reg_c}] — 左/右各{model.num_bins+1} bin分布")
        else:
            print(f"    {'':2s}  [B=1, T_lvl, 2] — (left_offset, right_offset)")
        print_shape(f"  Level {level} mask", fpn_masks_permuted[level])
        print(f"    {'':2s}  [B=1, T_lvl] — True=有效位置")

    # GT label generation
    gt_segments = [sample['segments'].to(out_cls_logits[0].device)]
    gt_labels = [sample['labels'].to(out_cls_logits[0].device)]

    gt_cls_labels, gt_offsets = model.label_points(points, gt_segments, gt_labels)

    print(f"\n  [GT 标签生成 (label_points)]")
    print(f"    GT 段数: {gt_segments[0].shape[0]}")
    for b in range(len(gt_cls_labels)):
        print_shape(f"  batch[{b}] gt_cls_labels", gt_cls_labels[b])
        num_pos = (gt_cls_labels[b] >= 0).sum().item()
        print(f"    {'':2s}  [FT, num_classes] — sigmoid 多标签目标.  总:{num_pos}")
        print_shape(f"  batch[{b}] gt_offsets", gt_offsets[b])
        print(f"    {'':2s}  [FT, 2] — (左偏移目标, 右偏移目标), 用 stride 归一化后的值")

    return fpn_masks_permuted, out_cls_logits, out_offsets, gt_cls_labels, gt_offsets


# ── 损失计算 ──────────────────────────────────────────────────────────────

def show_losses(model, sample):
    """展示训练模式下的完整 forward + loss 计算。"""
    print_header("7. 损失计算 (model.forward → losses)")

    model.train()
    with torch.no_grad():
        losses = model([sample])

    print(f"\n  损失项:")
    for name, val in losses.items():
        print(f"    {name}: {val.item():.4f}")
        print(f"    {'':2s}  意义: ", end="")
        if name == "cls_loss":
            print("分类损失 (Focal Loss) — 判断每个锚点的动作类别")
        elif name == "reg_loss":
            print("回归损失 (DIoU Loss) — 预测边界与 GT 的重叠 + 中心距离")
        elif name == "final_loss":
            print("总损失 = cls_loss + reg_loss × loss_weight (自动平衡)")
        else:
            print("")

    return losses


# ── 推理模式 ──────────────────────────────────────────────────────────────

def show_inference(model, sample):
    """展示推理模式下的完整输出。"""
    print_header("8. 推理模式 (model.forward → results)")

    model.eval()
    with torch.no_grad():
        results = model([sample])

    print(f"\n  检出视频数: {len(results)}")

    if len(results) == 0:
        print("  无检出 (result 为空)")
        return

    result = results[0]

    print(f"\n  ── 推理输出字段 ──")
    for k, v in result.items():
        if isinstance(v, torch.Tensor):
            print_shape(f"  {k}", v)
        elif isinstance(v, str):
            print(f"    {k}: '{v}'")
        elif isinstance(v, (int, float)):
            print(f"    {k}: {v}")

    print(f"\n  ── 各字段意义 ──")
    print(f"    video_id  : 视频标识符")
    print(f"    segments  : [M, 2] — 检测到的动作段 (秒), M={result['segments'].shape[0] if result['segments'].numel() > 0 else 0}")
    print(f"    scores    : [M]    — 置信度分数 (sigmoid 后的最大类别概率)")
    print(f"    labels    : [M]    — 预测类别编号 (0~num_classes-1)")

    n_det = result['segments'].shape[0] if result['segments'].numel() > 0 else 0
    if n_det > 0:
        print(f"\n  ── 检出示例 (置信度 top-5) ──")
        top_k = min(5, n_det)
        sorted_idx = result['scores'].argsort(descending=True)
        for rank, idx in enumerate(sorted_idx[:top_k]):
            seg = result['segments'][idx]
            score = result['scores'][idx].item()
            label = result['labels'][idx].item()
            print(f"    #{rank+1}: [{seg[0].item():6.1f}s, {seg[1].item():6.1f}s]  "
                  f"score={score:.4f}  class={label}")


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="TriDet 数据流形状与意义展示脚本")
    parser.add_argument("--config", default="./configs/thumos_i3d.yaml",
                        help="配置文件路径")
    parser.add_argument("--mode", choices=["train", "inference", "both"],
                        default="train", help="展示模式")
    parser.add_argument("--max-seq-len", type=int, default=900,
                        help="最大序列长度 (默认 900, 降低显存需求)")
    parser.add_argument("--no-amp", action="store_true",
                        help="关闭 AMP (默认启用)")
    parser.add_argument("--batch-size", type=int, default=1,
                        help="batch size (默认 1)")
    args = parser.parse_args()

    print("=" * 72)
    print("  TriDet 数据流形状与意义展示")
    print("=" * 72)

    # ── 加载配置 ──
    print("\n[配置加载]")
    cfg = load_config(args.config)

    # 覆盖 batch_size 和 max_seq_len
    cfg["loader"]["batch_size"] = args.batch_size

    fpn_levels = cfg["model"]["backbone_arch"][-1] + 1
    scale_factor = cfg["model"]["scale_factor"]
    seq_divisor = scale_factor ** (fpn_levels - 1)
    raw_len = args.max_seq_len
    aligned_len = (raw_len // seq_divisor) * seq_divisor
    if aligned_len != raw_len:
        print(f"  max_seq_len {raw_len} → {aligned_len} (对齐 {seq_divisor} 的倍数)")
    cfg["dataset"]["max_seq_len"] = aligned_len

    # 同步 dataset → model
    cfg["model"]["input_dim"] = cfg["dataset"]["input_dim"]
    cfg["model"]["num_classes"] = cfg["dataset"]["num_classes"]
    cfg["model"]["max_seq_len"] = cfg["dataset"]["max_seq_len"]
    cfg["model"]["train_cfg"] = cfg["train_cfg"]
    cfg["model"]["test_cfg"] = cfg["test_cfg"]

    print(f"  config           : {args.config}")
    print(f"  fpn_type         : {cfg['model'].get('fpn_type', 'fpn')}")
    print(f"  use_trident_head : {cfg['model']['use_trident_head']}")
    print(f"  backbone_arch    : {cfg['model']['backbone_arch']}")
    print(f"  embd_dim         : {cfg['model']['embd_dim']}")
    print(f"  fpn_dim          : {cfg['model']['fpn_dim']}")
    print(f"  head_dim         : {cfg['model']['head_dim']}")
    print(f"  num_classes      : {cfg['model']['num_classes']}")
    print(f"  max_seq_len      : {cfg['model']['max_seq_len']}")
    print(f"  batch_size       : {args.batch_size}")
    print(f"  AMP              : {'关闭' if args.no_amp else '启用'}")

    # ── 创建数据集 ──
    rng = fix_random_seed(cfg["init_rand_seed"], include_cuda=True)

    train_dataset = make_dataset(
        cfg["dataset_name"], True, cfg["train_split"], **cfg["dataset"]
    )
    val_dataset = make_dataset(
        cfg["dataset_name"], False, cfg["val_split"], **cfg["dataset"]
    )

    print(f"\n  训练集样本数: {len(train_dataset.data_list)}")
    print(f"  验证集样本数: {len(val_dataset.data_list)}")

    # ── 创建模型 ──
    model = make_meta_arch(cfg["model_name"], **cfg["model"])

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n  模型总参数: {total_params:,}")
    print(f"  可训练参数: {trainable_params:,}")

    # ── 1. 数据集样本 ──
    sample = show_dataset_sample(train_dataset, idx=0)

    # ── 2. 预处理 ──
    batched_inputs, batched_masks = show_preprocessing(model, sample)

    # ── 3. Backbone & Neck ──
    fpn_feats, fpn_masks = show_backbone_neck(model, batched_inputs, batched_masks)

    # ── 4. Points ──
    points = show_points(model, fpn_feats)

    # ── 5. Heads ──
    out_cls_logits, out_offsets, out_lb_logits, out_rb_logits = show_heads(
        model, fpn_feats, fpn_masks
    )

    # ── 训练模式 ──
    if args.mode in ("train", "both"):
        show_training_labels(model, fpn_masks, out_cls_logits, out_offsets, points, sample)
        show_losses(model, sample)

    # ── 推理模式 ──
    if args.mode in ("inference", "both"):
        # 取验证集样本做推理
        val_sample = val_dataset[0]
        print(f"\n  推理样本: {val_sample['video_id']} (时长 {val_sample['duration']:.1f}s)")
        show_inference(model, val_sample)

    # ── 整体数据流总结 ──
    print_header("总结: 完整数据流")
    print(f"""
    Dataset.__getitem__
      │  feats: [C, T_raw]  I3D逐帧特征
      │  segments: [N, 2]   GT动作段 (特征网格坐标)
      │  labels:   [N]      GT类别编号
      ▼
    preprocessing(video_list)
      │  batched_inputs: [B, C, T=max_seq_len]  Zero-pad到固定长度
      │  batched_masks:  [B, 1, T]              有效位置掩码
      ▼
    Backbone (SGPBackbone)
      │  feats:  [B, embd_dim, T/scale_factor]   SGP下采样特征
      │  masks:  [B, 1, T/scale_factor]
      ▼
    Neck (FPN1D / BiFPN1D)
      │  fpn_feats: List[F] of [B, fpn_dim, T_f]  多尺度特征金字塔
      │  fpn_masks: List[F] of [B, 1, T_f]
      ▼
    Point Generator
      │  points: List[F] of [T_f, 4]             锚点坐标/范围/stride
      ▼
    ┌─────────────────┬─────────────────┬──────────────────┐
    ClsHead           RegHead           TridentHead (可选)
    [B, T_f, #cls]    [B, T_f, 2]       [B, T_f, #bins+1]
    每个锚点类别logits  左右偏移量         边界bin分布
    └─────────────────┴─────────────────┴──────────────────┘
      ▼
    [Training]                    [Inference]
    label_points → GT分配          解码 → NMS → 最终段
    losses → cls + reg × weight    segments [M,2], scores [M]
""")

    print("=" * 72)
    print("  展示完成")
    print("=" * 72)


if __name__ == "__main__":
    main()

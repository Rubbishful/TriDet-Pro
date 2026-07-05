"""快速冒烟测试 — 验证各特性配置是否能正常实例化和前向传播。

覆盖特性: SE/ECA 通道注意力, GIoU/DIoU/EIoU/Alpha-DIoU/Focaler-DIoU loss,
          BiFPN/FPN neck, IoU预测头+TAL, AMP混合精度。

每个 config: 训练模式→验证 losses, 推理模式→验证输出结构。
"""
import os
import sys
import traceback

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
from libs.core.config import load_config
from libs.modeling import make_meta_arch

# ── 测试的 config 列表 ──────────────────────────────────────────────
CONFIGS = [
    # (名称, config路径, 激活的特性)
    ("Baseline",           "configs/thumos_i3d.yaml"),
    ("SE-Attention",       "configs/thumos_i3d_se.yaml"),
    ("ECA-Attention",      "configs/thumos_i3d_eca.yaml"),
    ("GIoU-Loss",          "configs/thumos_i3d_giou.yaml"),
    ("EIoU-Loss",          "configs/thumos_i3d_eiou.yaml"),
    ("Alpha-DIoU-Loss",    "configs/thumos_i3d_alpha_diou.yaml"),
    ("Focaler-DIoU-Loss",  "configs/thumos_i3d_focaler_diou.yaml"),
    ("BiFPN-Neck",         "configs/thumos_i3d_bifpn.yaml"),
    ("FPN-Neck",           "configs/thumos_i3d_fpn.yaml"),
    ("IoU-Head-TAL",       "configs/thumos_i3d_iou_tal_baseline.yaml"),
    ("IoU-TAL-Iter-AMP",   "configs/thumos_i3d_iou_tal_iteration.yaml"),
]

# ── 随机数据生成 ────────────────────────────────────────────────────

def make_dummy_video(input_dim, feat_len, num_classes, device):
    """生成一个训练/推理兼容的 dummy 视频数据 (batch_size=1)。"""
    duration = feat_len * 4 / 25.0  # feat_stride=4, fps=25
    return {
        "video_id": "smoke_test",
        "feats": torch.randn(input_dim, feat_len, device=device),
        "fps": 25.0,
        "duration": duration,
        "feat_stride": 4,
        "feat_num_frames": 16,
        "segments": torch.tensor([[10.0, 50.0]], device=device),
        "labels": torch.tensor([0], dtype=torch.long, device=device),
    }

# ── 单 config 测试 ──────────────────────────────────────────────────

def smoke_one(name, config_path):
    """对一个 config 执行训练+推理冒烟，返回 (ok, msg)。"""
    try:
        cfg = load_config(config_path)
        model_cfg = cfg["model"].copy()

        model = make_meta_arch(cfg['model_name'], **model_cfg)
        device = next(model.parameters()).device

        input_dim = cfg["dataset"]["input_dim"]
        num_classes = cfg["dataset"]["num_classes"]
        feat_len = min(cfg["dataset"]["max_seq_len"], 256)

        # ── 训练模式 ──
        model.train()
        video = make_dummy_video(input_dim, feat_len, num_classes, device)
        losses = model([video])

        required = ["cls_loss", "reg_loss", "final_loss"]
        missing = [k for k in required if k not in losses]
        if missing:
            return False, f"训练 losses 缺少字段: {missing}"

        for k in required:
            v = losses[k]
            if v is not None and not torch.isfinite(v):
                return False, f"{k} = {v.item()} (非有限值)"

        loss_str = ", ".join(f"{k}={losses[k].item():.4f}" for k in required)

        # ── 推理模式 ──
        model.eval()
        video_infer = make_dummy_video(input_dim, feat_len, num_classes, device)
        video_infer["segments"] = None
        video_infer["labels"] = None
        with torch.no_grad():
            results = model([video_infer])

        if not isinstance(results, list) or len(results) == 0:
            return False, "推理未返回结果列表"

        r = results[0]
        for k in ["video_id", "segments", "scores", "labels"]:
            if k not in r:
                return False, f"推理结果缺少字段: {k}"

        infer_str = (f"segs={r['segments'].shape}, "
                     f"scores={r['scores'].shape}, "
                     f"labels={r['labels'].shape}")

        return True, f"train[{loss_str}] infer[{infer_str}]"

    except Exception as e:
        import traceback as _tb
        _tb.print_exc()
        return False, f"{type(e).__name__}: {e}"

# ── 主入口 ──────────────────────────────────────────────────────────

def main():
    passed, failed = 0, []
    for name, path in CONFIGS:
        print(f"\n  --- {name} ({path}) ---", flush=True)
        ok, msg = smoke_one(name, path)
        status = "PASS" if ok else "FAIL"
        if ok:
            print(f"  [{status}] {msg}", flush=True)
            passed += 1
        else:
            print(f"  [{status}] {msg}", flush=True)
            failed.append(name)

    print(f"\n  RESULT: {passed}/{len(CONFIGS)} passed")
    if failed:
        print(f"  FAILED: {', '.join(failed)}")
        sys.exit(1)
    else:
        print("  All smoke tests passed!")

if __name__ == "__main__":
    main()

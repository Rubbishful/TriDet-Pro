"""
可视化脚本：对比 ActivityNet GT 与模型预测的时间轴。

用法:
  # 生成所有可用视频的预测 pkl（在终端执行）
  python eval.py configs/anet_tsp.yaml ckpt/anet_tsp_baseline/ --saveonly

  # 随机采样 10 个视频生成可视化
  python visualize.py --pkl ckpt/anet_tsp_baseline/eval_results.pkl --num-samples 10

  # 指定视频 ID
  python visualize.py --pkl ckpt/anet_tsp_baseline/eval_results.pkl --video-id XXXXXXXXXXX

  # 指定输出目录和置信度阈值
  python visualize.py --pkl ckpt/anet_tsp_baseline/eval_results.pkl --num-samples 5 --score-thresh 0.3 --output-dir vis_output
"""

import argparse
import json
import os
import pickle
import random
import sys
from collections import defaultdict

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

# ── 路径常量 ───────────────────────────────────────────────
ANNOTATION_FILE = r"D:\Code\ActivityNet\anet_1.3\annotations\anet1.3_tsp_filtered.json"
VIDEO_DIR = r"D:\Code\ActivityNet\anet_video\activitynet-100\validation\data"
FEATURE_DIR = r"D:\Code\ActivityNet\anet_1.3\tsp_features"

# 可视化配色
GT_COLOR = "#2ecc71"  # 绿色 — Ground Truth
PRED_COLOR = "#e74c3c"  # 红色 — 预测
GT_ALPHA = 0.55
PRED_ALPHA = 0.50
NUM_FRAMES = 5  # 每个视频抽取的帧数


def build_label_dict():
    """从标注文件构建 label_id -> label_name 映射。"""
    with open(ANNOTATION_FILE, "r") as f:
        data = json.load(f)
    label_map = {}
    for v in data["database"].values():
        for ann in v["annotations"]:
            label_map[ann["label_id"]] = ann["label"]
    return label_map


def find_video_path(video_id):
    """根据 youtube_id (不带 v_ 前缀) 定位原始视频文件路径。"""
    for prefix in ("v_", "v__"):
        path = os.path.join(VIDEO_DIR, prefix + video_id + ".mp4")
        if os.path.isfile(path):
            return path
    return None


def build_triple_overlap():
    """
    构建 视频 + 特征 + 标注 三者重叠的 video_id 集合。
    返回 {youtube_id: {"video": path, "feature": path, "anno": entry}}
    """
    with open(ANNOTATION_FILE, "r") as f:
        anno_data = json.load(f)
    db = anno_data["database"]

    # 扫描视频
    video_ids = {}
    for fname in os.listdir(VIDEO_DIR):
        if not fname.endswith(".mp4"):
            continue
        name = fname[:-4]
        if name.startswith("v__"):
            yt_id = name[3:]
        elif name.startswith("v_"):
            yt_id = name[2:]
        else:
            yt_id = name
        video_ids[yt_id] = os.path.join(VIDEO_DIR, fname)

    # 扫描特征
    feat_ids = set()
    for fname in os.listdir(FEATURE_DIR):
        if fname.endswith(".npy"):
            name = fname[:-4]
            if name.startswith("v_"):
                feat_ids.add(name[2:])
            else:
                feat_ids.add(name)

    # 三者重叠
    overlap = {}
    for yt_id, vpath in video_ids.items():
        if yt_id not in feat_ids:
            continue
        if yt_id not in db:
            continue
        if db[yt_id]["subset"].lower() != "validation":
            continue
        overlap[yt_id] = {
            "video": vpath,
            "feature": os.path.join(FEATURE_DIR, "v_" + yt_id + ".npy"),
            "anno": db[yt_id],
        }
    return overlap


def extract_frames(video_path, duration, num_frames=NUM_FRAMES):
    """从视频中等间隔抽取帧 (RGB numpy arrays)。"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        cap.release()
        return None

    fps = total_frames / duration if duration > 0 else 30.0
    frames = []
    for i in range(num_frames):
        t = duration * (i + 0.5) / num_frames
        frame_idx = int(t * fps)
        frame_idx = min(frame_idx, total_frames - 1)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append((t, frame))

    cap.release()
    return frames


def load_predictions(pkl_path, score_thresh=0.0):
    """
    加载 eval_results.pkl，按 video_id 分组。
    返回 {video_id: [(t_start, t_end, label_id, score), ...]}
    """
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)

    preds = defaultdict(list)
    for i in range(len(data["video-id"])):
        score = float(data["score"][i])
        if score < score_thresh:
            continue
        preds[data["video-id"][i]].append(
            (
                float(data["t-start"][i]),
                float(data["t-end"][i]),
                int(data["label"][i]),
                score,
            )
        )
    # 按 score 降序排列
    for vid in preds:
        preds[vid].sort(key=lambda x: x[3], reverse=True)
    return dict(preds)


def draw_visualization(video_id, info, predictions, label_dict, score_thresh, output_path):
    """为单个视频生成 GT vs Pred 对比图。"""
    anno = info["anno"]
    duration = anno["duration"]
    gt_segments = anno["annotations"]
    pred_segments = predictions.get(video_id, [])

    # 提取帧
    frames = extract_frames(info["video"], duration, NUM_FRAMES)
    if frames is None:
        print(f"  [WARN] 无法读取视频: {video_id}")
        return False

    # ── 创建 Figure ──
    fig = plt.figure(figsize=(18, 4 + 1.5))
    gs = GridSpec(2, 1, height_ratios=[1.6, 1], hspace=0.22)

    # ── 上排：视频帧 ──
    ax_frames = fig.add_subplot(gs[0])
    ax_frames.set_title(f"{video_id}  (duration: {duration:.1f}s)", fontsize=11, loc="left")
    ax_frames.axis("off")

    # 拼接帧为一条连续画面
    if frames:
        # 统一缩放到相同高度
        target_h = 160
        resized_frames = []
        for t, frm in frames:
            h, w = frm.shape[:2]
            new_w = int(w * target_h / h)
            resized = cv2.resize(frm, (new_w, target_h))
            # 在底部标注时间戳
            resized_frames.append(resized)

        # 水平拼接
        concat = np.concatenate(resized_frames, axis=1)
        ax_frames.imshow(concat)
        # 在每帧下方标注时间
        total_w = concat.shape[1]
        for i in range(NUM_FRAMES):
            t = duration * (i + 0.5) / NUM_FRAMES
            x_pos = (i + 0.5) * total_w / NUM_FRAMES
            ax_frames.text(
                x_pos, target_h + 6, f"{t:.1f}s",
                ha="center", va="top", fontsize=7, color="black"
            )

    # ── 下排：时间轴对比 ──
    ax_timeline = fig.add_subplot(gs[1])

    # 收集所有用到的类别，分配 Y 位置
    all_labels = set()
    for gt in gt_segments:
        all_labels.add(gt["label_id"])
    for pred in pred_segments:
        all_labels.add(pred[2])
    label_list = sorted(all_labels)
    label_y = {lid: i for i, lid in enumerate(label_list)}  # class ID -> row
    n_rows = len(label_list)

    ax_timeline.set_ylim(-0.5, n_rows - 0.5)
    ax_timeline.set_xlim(0, duration)
    ax_timeline.set_xlabel("Time (seconds)", fontsize=10)
    ax_timeline.set_yticks(range(n_rows))
    ax_timeline.set_yticklabels(
        [label_dict.get(lid, f"cls_{lid}") for lid in label_list], fontsize=7
    )
    ax_timeline.invert_yaxis()

    # GT 条
    bar_height = 0.35
    for gt in gt_segments:
        lid = gt["label_id"]
        y = label_y[lid]
        start, end = gt["segment"]
        ax_timeline.barh(
            y + bar_height / 2,
            end - start,
            bar_height,
            left=start,
            color=GT_COLOR,
            alpha=GT_ALPHA,
            edgecolor="#27ae60",
            linewidth=0.8,
            zorder=3,
        )

    # Prediction 条 (向下偏移)
    for pred in pred_segments:
        t_start, t_end, lid, score = pred
        y = label_y.get(lid, 0)
        alpha = PRED_ALPHA * min(1.0, score / 0.5)  # 分数越低越透明
        ax_timeline.barh(
            y - bar_height / 2,
            t_end - t_start,
            bar_height,
            left=t_start,
            color=PRED_COLOR,
            alpha=max(0.15, alpha),
            edgecolor="#c0392b",
            linewidth=0.6,
            zorder=3,
        )

    # 在预测条上标注分数
    for pred in pred_segments:
        t_start, t_end, lid, score = pred
        if score >= score_thresh + 0.1:
            y = label_y.get(lid, 0) - bar_height / 2
            cx = (t_start + t_end) / 2
            ax_timeline.text(
                cx, y, f"{score:.2f}", ha="center", va="center",
                fontsize=5, color="white", fontweight="bold", zorder=4,
            )

    # 图例
    legend_patches = [
        mpatches.Patch(color=GT_COLOR, alpha=GT_ALPHA, label="Ground Truth"),
        mpatches.Patch(color=PRED_COLOR, alpha=PRED_ALPHA, label=f"Prediction (score >= {score_thresh})"),
    ]
    ax_timeline.legend(handles=legend_patches, fontsize=8, loc="upper right")

    # 网格
    ax_timeline.grid(axis="x", alpha=0.3, linestyle="--")
    ax_timeline.set_axisbelow(True)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return True


def main():
    parser = argparse.ArgumentParser(description="ActivityNet 可视化：GT vs 模型预测")
    parser.add_argument("--pkl", type=str, required=True,
                        help="eval_results.pkl 路径 (由 eval.py --saveonly 生成)")
    parser.add_argument("--video-id", type=str, default=None,
                        help="指定视频 youtube_id (不带 v_ 前缀)")
    parser.add_argument("--num-samples", type=int, default=10,
                        help="随机采样视频数 (--video-id 指定时忽略)")
    parser.add_argument("--score-thresh", type=float, default=0.1,
                        help="置信度阈值，低于此值的预测不显示")
    parser.add_argument("--seed", type=int, default=42,
                        help="随机种子")
    parser.add_argument("--output-dir", type=str, default="vis_output",
                        help="输出目录")
    parser.add_argument("--list-overlap", action="store_true",
                        help="仅列出可用的重叠视频数量并退出")
    args = parser.parse_args()

    # ── 构建重叠集 ──
    print("构建视频-特征-标注重叠集 ...")
    overlap = build_triple_overlap()
    print(f"三者重叠视频数: {len(overlap)}")

    if args.list_overlap:
        return

    if len(overlap) == 0:
        print("错误: 没有任何视频同时具备原始视频 + 特征 + 标注。")
        sys.exit(1)

    # ── 加载标注 & 预测 ──
    label_dict = build_label_dict()
    print(f"类别数: {len(label_dict)}")

    if not os.path.isfile(args.pkl):
        print(f"错误: 预测文件不存在: {args.pkl}")
        print("请先运行: python eval.py configs/anet_tsp.yaml ckpt/anet_tsp_baseline/ --saveonly")
        sys.exit(1)

    print(f"加载预测: {args.pkl}")
    predictions = load_predictions(args.pkl, score_thresh=0.0)  # 不过滤，显示时用阈值
    print(f"有预测的视频数: {len(predictions)}")

    # ── 选择视频 ──
    if args.video_id:
        selected = [args.video_id] if args.video_id in overlap else []
        if not selected:
            print(f"错误: video_id '{args.video_id}' 不在重叠集中或不是 validation split")
            # 尝试查找
            if args.video_id in overlap:
                pass
            else:
                found_vid = find_video_path(args.video_id)
                if found_vid:
                    print(f"  视频存在但可能不在 validation split: {found_vid}")
                else:
                    print(f"  视频文件不存在，或不在 validation split")
            sys.exit(1)
    else:
        rng = random.Random(args.seed)
        candidates = list(overlap.keys())
        # 优先选有预测结果的视频
        has_pred = [v for v in candidates if v in predictions]
        no_pred = [v for v in candidates if v not in predictions]
        print(f"重叠集中有预测结果的: {len(has_pred)}, 无预测: {len(no_pred)}")
        if len(has_pred) >= args.num_samples:
            selected = rng.sample(has_pred, args.num_samples)
        else:
            selected = has_pred + rng.sample(no_pred, args.num_samples - len(has_pred))

    print(f"选中 {len(selected)} 个视频进行可视化")

    # ── 生成可视化 ──
    os.makedirs(args.output_dir, exist_ok=True)
    n_ok = 0
    for i, vid in enumerate(selected):
        out_path = os.path.join(args.output_dir, f"{vid}.png")
        print(f"  [{i+1}/{len(selected)}] {vid} ...", end=" ")
        ok = draw_visualization(vid, overlap[vid], predictions, label_dict,
                                args.score_thresh, out_path)
        if ok:
            n_ok += 1
            print(f"-> {out_path}")
        else:
            print("SKIP")

    print(f"\n完成: {n_ok}/{len(selected)} 个视频已输出到 {args.output_dir}/")


if __name__ == "__main__":
    main()

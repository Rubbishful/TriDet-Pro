"""
按动作时长分层分析：将 GT 实例按时长分组，分别计算 recall。

用法:
    python evaluate/duration_analysis.py --pred results.pkl --json data/thumos/annotations/thumos14.json
"""

import os, sys, argparse, json, pickle
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from libs.utils.metrics import segment_iou

# 时长分桶: 短(<2s), 中(2-10s), 长(>10s)
BINS = [('短 (<2s)', 0, 2), ('中 (2-10s)', 2, 10), ('长 (>10s)', 10, 1e6)]


def load_gt(json_path, split='test'):
    """加载 GT 标注。返回 {video_id: [(start, end, label), ...]}"""
    with open(json_path, 'r') as f:
        db = json.load(f)
    gts = {}
    for vid, info in db['database'].items():
        if info['subset'].lower() != split:
            continue
        instances = []
        for ann in info.get('annotations', []):
            s, e = ann['segment']
            instances.append((float(s), float(e), ann['label_id']))
        gts[vid] = instances
    return gts


def load_preds(pkl_path):
    """加载模型预测结果。"""
    with open(pkl_path, 'rb') as f:
        return pickle.load(f)


def compute_recall(gt_instances, pred_instances, tiou_thresh=0.5):
    """计算单组 GT 的 recall。"""
    if len(gt_instances) == 0:
        return 1.0  # 无 GT 视为 prefect
    matched = [False] * len(gt_instances)
    for pred in pred_instances:
        p_seg = np.array([pred['t-start'], pred['t-end']])
        for i, (s, e, _) in enumerate(gt_instances):
            if matched[i]:
                continue
            gt_seg = np.array([[s, e]])
            tiou = segment_iou(p_seg, gt_seg)[0]
            if tiou >= tiou_thresh:
                matched[i] = True
                break
    return sum(matched) / len(matched)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pred', required=True, help='评估 pickle 文件路径')
    parser.add_argument('--json', default=None,
                        help='GT annotation JSON path (required)')
    parser.add_argument('--split', default='test', help='数据集 split')
    parser.add_argument('--tiou', type=float, default=0.5, help='tIoU 阈值')
    parser.add_argument('--out', default=None, help='输出图表路径')
    args = parser.parse_args()

    gt_data = load_gt(args.json, args.split)
    preds = load_preds(args.pred)

    # 按视频 ID 索引预测
    pred_by_vid = {}
    for p in preds:
        pred_by_vid.setdefault(p['video-id'], []).append(p)

    # 按视频分桶统计
    bin_results = {name: {'gt': 0, 'matched': 0} for name, _, _ in BINS}
    per_bin_recalls = {name: [] for name, _, _ in BINS}

    for vid, gt_instances in gt_data.items():
        vid_preds = pred_by_vid.get(vid, [])
        for s, e, label in gt_instances:
            duration = e - s
            for bin_name, lo, hi in BINS:
                if lo <= duration < hi:
                    bin_results[bin_name]['gt'] += 1
                    break

        # 计算此视频在每个分桶中的 recall
        for bin_name, lo, hi in BINS:
            bin_gts = [(s, e, l) for s, e, l in gt_instances if lo <= (e - s) < hi]
            if bin_gts:
                r = compute_recall(bin_gts, vid_preds, args.tiou)
                per_bin_recalls[bin_name].append(r)

    # 输出统计
    print(f"按动作时长分层分析 (tIoU={args.tiou})")
    print(f"{'时长分组':<15} {'GT数':>6} {'平均Recall':>10}")
    print('-' * 35)
    avg_recalls = []
    labels = []
    for bin_name, lo, hi in BINS:
        n_gt = bin_results[bin_name]['gt']
        recalls = per_bin_recalls[bin_name]
        avg_r = np.mean(recalls) if recalls else 0.0
        avg_recalls.append(avg_r)
        labels.append(f'{bin_name}\n(n={n_gt})')
        print(f"{bin_name:<15} {n_gt:>6} {avg_r:>10.1%}")

    # 画图
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ['#2ecc71', '#3498db', '#e74c3c']
    bars = ax.bar(labels, [r * 100 for r in avg_recalls], color=colors, edgecolor='white')
    for bar, val in zip(bars, avg_recalls):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f'{val:.1%}', ha='center', fontsize=12)
    ax.set_ylabel('Recall', fontsize=13)
    ax.set_title(f'Recall by Action Duration (tIoU={args.tiou})', fontsize=14)
    ax.set_ylim(0, max(avg_recalls) * 1.2 * 100 + 5 if avg_recalls else 100)
    plt.tight_layout()

    out_path = args.out or os.path.join(os.path.dirname(__file__), 'duration_analysis.png')
    plt.savefig(out_path, dpi=150)
    print(f"\n图表已保存至: {out_path}")


if __name__ == '__main__':
    main()

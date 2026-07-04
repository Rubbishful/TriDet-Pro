"""
按视频动作密度分层分析：按每个视频的 GT 实例数分桶，分别计算 mAP。

用法:
    python evaluate/density_analysis.py --pred results.pkl --json data/thumos/annotations/thumos14.json
"""

import os, sys, argparse, json, pickle
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from libs.utils.metrics import segment_iou

# 密度分桶
DENSITY_BINS = [
    ('1-3', 1, 3),
    ('4-6', 4, 6),
    ('7-10', 7, 10),
    ('>10', 11, 999),
]


def load_gt(json_path, split='test'):
    with open(json_path, 'r') as f:
        db = json.load(f)
    gts = {}
    for vid, info in db['database'].items():
        if info['subset'].lower() != split:
            continue
        instances = [(float(a['segment'][0]), float(a['segment'][1]), a['label_id'])
                     for a in info.get('annotations', [])]
        gts[vid] = instances
    return gts


def load_preds(pkl_path):
    with open(pkl_path, 'rb') as f:
        return pickle.load(f)


def compute_ap(gt_instances, pred_instances, tiou_thresh):
    """简化版 AP 计算（跨所有类别，单 tIoU 阈值）。"""
    if not gt_instances:
        return 0.0  # 无 GT 时返回 0

    # 按置信度排序
    preds = sorted(pred_instances, key=lambda x: x['score'], reverse=True)

    npos = len(gt_instances)
    tp = np.zeros(len(preds))
    fp = np.zeros(len(preds))
    matched = np.ones(npos) * -1

    for i, pred in enumerate(preds):
        p_seg = np.array([pred['t-start'], pred['t-end']])
        best_tiou = 0
        best_j = -1
        for j, (s, e, _) in enumerate(gt_instances):
            if matched[j] >= 0:
                continue
            gt_seg = np.array([[s, e]])
            tiou = segment_iou(p_seg, gt_seg)[0]
            if tiou > best_tiou:
                best_tiou = tiou
                best_j = j

        if best_tiou >= tiou_thresh:
            tp[i] = 1
            matched[best_j] = i
        else:
            fp[i] = 1

    tp_cumsum = np.cumsum(tp)
    fp_cumsum = np.cumsum(fp)
    recalls = tp_cumsum / npos
    precisions = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-8)

    # 插值 AP
    mrec = np.hstack([[0], recalls, [1]])
    mpre = np.hstack([[0], precisions, [0]])
    for i in range(len(mpre) - 1, 0, -1):
        mpre[i - 1] = max(mpre[i - 1], mpre[i])
    idx = np.where(mrec[1:] != mrec[:-1])[0] + 1
    ap = np.sum((mrec[idx] - mrec[idx - 1]) * mpre[idx])
    return ap


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pred', required=True, help='评估 pickle 文件路径')
    parser.add_argument('--json', default=None, help='GT annotation JSON path (required)')
    parser.add_argument('--split', default='test')
    parser.add_argument('--tiou', type=float, default=0.5)
    parser.add_argument('--out', default=None)
    args = parser.parse_args()

    gt_data = load_gt(args.json, args.split)
    preds = load_preds(args.pred)

    pred_by_vid = {}
    for p in preds:
        pred_by_vid.setdefault(p['video-id'], []).append(p)

    # 按密度分桶
    bin_aps = {name: [] for name, _, _ in DENSITY_BINS}
    video_densities = []

    for vid, gt_instances in gt_data.items():
        n_gt = len(gt_instances)
        video_densities.append(n_gt)
        vid_preds = pred_by_vid.get(vid, [])

        for bin_name, lo, hi in DENSITY_BINS:
            if lo <= n_gt <= hi:
                ap = compute_ap(gt_instances, vid_preds, args.tiou)
                bin_aps[bin_name].append(ap)
                break

    # 输出
    print(f"按视频动作密度分层分析 (tIoU={args.tiou})")
    print(f"{'密度分组':<12} {'视频数':>6} {'平均mAP':>10}")
    print('-' * 32)
    values = []
    labels = []
    for bin_name, lo, hi in DENSITY_BINS:
        aps = bin_aps[bin_name]
        avg_ap = np.mean(aps) if aps else 0.0
        values.append(avg_ap * 100)
        labels.append(f'{bin_name}\n(n={len(aps)})')
        print(f"{bin_name:<12} {len(aps):>6} {avg_ap:>10.1%}")

    # 画图
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ['#2ecc71', '#3498db', '#f39c12', '#e74c3c']
    bars = ax.bar(labels, values, color=colors, edgecolor='white')
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f'{val:.1f}%', ha='center', fontsize=12)
    ax.set_ylabel('mAP (%)', fontsize=13)
    ax.set_title(f'mAP by Action Density per Video (tIoU={args.tiou})', fontsize=14)
    ax.set_ylim(0, max(values) * 1.2 + 2 if max(values) > 0 else 100)
    plt.tight_layout()

    out_path = args.out or os.path.join(os.path.dirname(__file__), 'density_analysis.png')
    plt.savefig(out_path, dpi=150)
    print(f"\n图表已保存至: {out_path}")

    # 额外：密度分布直方图
    fig2, ax2 = plt.subplots(figsize=(8, 4))
    ax2.hist(video_densities, bins=20, edgecolor='white', color='#3498db')
    ax2.set_xlabel('GT instances per video', fontsize=13)
    ax2.set_ylabel('Video count', fontsize=13)
    ax2.set_title('Distribution of Action Density', fontsize=14)
    plt.tight_layout()
    hist_path = out_path.replace('.png', '_hist.png') if out_path else \
        os.path.join(os.path.dirname(__file__), 'density_hist.png')
    plt.savefig(hist_path, dpi=150)
    print(f"分布直方图: {hist_path}")


if __name__ == '__main__':
    main()

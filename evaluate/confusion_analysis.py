"""
混淆矩阵分析：统计 THUMOS14 的类别间误分类模式。

用法:
    python evaluate/confusion_analysis.py --pred results.pkl --json data/thumos/annotations/thumos14.json
"""

import os, sys, argparse, json, pickle
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from libs.utils.metrics import segment_iou


def load_gt(json_path, split='test'):
    with open(json_path, 'r') as f:
        db = json.load(f)
    gts = {}
    label_names = {}
    for vid, info in db['database'].items():
        if info['subset'].lower() != split:
            continue
        instances = []
        for ann in info.get('annotations', []):
            label_names[ann['label_id']] = ann['label']
            instances.append((float(ann['segment'][0]), float(ann['segment'][1]), ann['label_id']))
        gts[vid] = instances
    return gts, label_names


def load_preds(pkl_path):
    with open(pkl_path, 'rb') as f:
        return pickle.load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pred', required=True, help='评估 pickle 文件路径')
    parser.add_argument('--json', default=None, help='GT annotation JSON path (required)')
    parser.add_argument('--split', default='test')
    parser.add_argument('--tiou', type=float, default=0.5)
    parser.add_argument('--out', default=None)
    args = parser.parse_args()

    gt_data, label_names = load_gt(args.json, args.split)
    preds = load_preds(args.pred)
    num_classes = len(label_names)

    pred_by_vid = {}
    for p in preds:
        pred_by_vid.setdefault(p['video-id'], []).append(p)

    # 混淆矩阵: cm[gt_label, pred_label] = count
    cm = np.zeros((num_classes, num_classes), dtype=np.int32)

    for vid, gt_instances in gt_data.items():
        vid_preds = pred_by_vid.get(vid, [])
        # 按置信度排序的预测
        vid_preds_sorted = sorted(vid_preds, key=lambda x: x['score'], reverse=True)
        matched_gt = set()

        for pred in vid_preds_sorted:
            p_seg = np.array([pred['t-start'], pred['t-end']])
            p_label = int(pred['label'])
            best_tiou = 0
            best_gt_idx = -1

            for idx, (s, e, gt_label) in enumerate(gt_instances):
                if idx in matched_gt:
                    continue
                gt_seg = np.array([[s, e]])
                tiou = segment_iou(p_seg, gt_seg)[0]
                if tiou > best_tiou:
                    best_tiou = tiou
                    best_gt_idx = idx

            if best_tiou >= args.tiou and best_gt_idx >= 0:
                gt_label = gt_instances[best_gt_idx][2]
                cm[gt_label, p_label] += 1
                matched_gt.add(best_gt_idx)

    # 归一化为行百分比（每个 GT 类别的预测分布）
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_norm = cm.astype(float) / row_sums.clip(min=1)

    # 输出 top-5 误分类对
    print(f"THUMOS14 混淆分析 (tIoU={args.tiou}, {num_classes} 类)")
    print()
    print("Top 误分类对 (GT → 预测):")

    errors = []
    for i in range(num_classes):
        for j in range(num_classes):
            if i != j and cm[i, j] > 0:
                errors.append((i, j, cm[i, j]))

    errors.sort(key=lambda x: -x[2])
    for gt_idx, pred_idx, count in errors[:10]:
        gt_name = label_names.get(gt_idx, f'cls_{gt_idx}')
        pred_name = label_names.get(pred_idx, f'cls_{pred_idx}')
        print(f"  {gt_name} → {pred_name}: {count} 次")

    # 画热力图
    labels = [label_names.get(i, f'{i}') for i in range(num_classes)]

    fig, ax = plt.subplots(figsize=(14, 12))
    im = ax.imshow(cm_norm, cmap='YlOrRd', aspect='auto')

    ax.set_xticks(range(num_classes))
    ax.set_yticks(range(num_classes))
    ax.set_xticklabels(labels, rotation=90, fontsize=7)
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_xlabel('Predicted Label', fontsize=13)
    ax.set_ylabel('Ground Truth Label', fontsize=13)
    ax.set_title('THUMOS14 Confusion Matrix', fontsize=14)

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Fraction of GT', fontsize=11)
    plt.tight_layout()

    out_path = args.out or os.path.join(os.path.dirname(__file__), 'confusion_matrix.png')
    plt.savefig(out_path, dpi=150)
    print(f"\n图表已保存至: {out_path}")


if __name__ == '__main__':
    main()

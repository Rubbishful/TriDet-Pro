"""
Error Analysis for Overlapping / Dense Action Detection

Quantifies TriDet's failure modes in multi-instance, temporally overlapping
scenarios. Designed to work with THUMOS14 validation-set predictions.

Usage:
    python analysis/error_analysis.py \
        --gt_json ./data/thumos14.json \
        --pred_pkl ./ckpt/predictions.pkl \
        --split validation

Outputs (saved to analysis/output/):
    - density_recall.csv       : recall stratified by per-video instance count
    - co_detection.csv         : same-class co-detection rate (gap < 3s)
    - overlap_bucket.csv       : recall bucketed by overlap degree
    - failure_cases/*.png      : timeline visualisations of worst failures
"""

import argparse
import json
import os
import pickle
import sys
from collections import defaultdict

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def segment_iou(a, b):
    """tIoU between two segments (start, end)."""
    inter = max(0.0, min(a[1], b[1]) - max(a[0], b[0]))
    union = (a[1] - a[0]) + (b[1] - b[0]) - inter
    return inter / union if union > 1e-8 else 0.0


def load_gt(json_file, split='validation'):
    with open(json_file, 'r') as f:
        db = json.load(f)['database']
    gt = {}
    for vid, info in db.items():
        if info.get('subset', '').lower() != split:
            continue
        segs = []
        labels = []
        for ann in info.get('annotations', []):
            segs.append(ann['segment'])
            labels.append(ann['label_id'])
        if segs:
            gt[vid] = {
                'segments': np.array(segs, dtype=np.float32),
                'labels': np.array(labels, dtype=np.int64),
                'duration': info.get('duration', 1e8),
            }
    return gt


def load_preds(pkl_file):
    with open(pkl_file, 'rb') as f:
        raw = pickle.load(f)
    # raw is a dict with 'video-id', 't-start', 't-end', 'label', 'score'
    preds = {}
    for vid, start, end, label, score in zip(
        raw['video-id'], raw['t-start'], raw['t-end'],
        raw['label'], raw['score'],
    ):
        if vid not in preds:
            preds[vid] = {'segments': [], 'labels': [], 'scores': []}
        preds[vid]['segments'].append([float(start), float(end)])
        preds[vid]['labels'].append(int(label))
        preds[vid]['scores'].append(float(score))
    for vid in preds:
        preds[vid]['segments'] = np.array(preds[vid]['segments'])
        preds[vid]['labels'] = np.array(preds[vid]['labels'])
        preds[vid]['scores'] = np.array(preds[vid]['scores'])
    return preds


def match_predictions(gt_segs, gt_labels, pred_segs, pred_labels, pred_scores,
                      tiou_thresh=0.5):
    """Greedy match: each GT matched to highest-scoring prediction with tIoU >= thresh."""
    matched = np.zeros(len(gt_segs), dtype=bool)
    if len(pred_segs) == 0:
        return matched

    order = np.argsort(pred_scores)[::-1]
    used = np.zeros(len(pred_segs), dtype=bool)
    for pi in order:
        best_gt, best_iou = -1, 0.0
        for gi in range(len(gt_segs)):
            if matched[gi] or gt_labels[gi] != pred_labels[pi]:
                continue
            iou = segment_iou(gt_segs[gi], pred_segs[pi])
            if iou > best_iou:
                best_iou = iou
                best_gt = gi
        if best_gt >= 0 and best_iou >= tiou_thresh:
            matched[best_gt] = True
            used[pi] = True
    return matched


# ---------------------------------------------------------------------------
# 1. Density-stratified recall
# ---------------------------------------------------------------------------

def compute_density_recall(gt, preds, tiou_thresh=0.5):
    rows = []
    for vid, vgt in gt.items():
        n_inst = len(vgt['segments'])
        vpred = preds.get(vid, {'segments': np.empty((0, 2)),
                                'labels': np.empty(0, dtype=int),
                                'scores': np.empty(0)})
        matched = match_predictions(vgt['segments'], vgt['labels'],
                                    vpred['segments'], vpred['labels'],
                                    vpred['scores'], tiou_thresh)
        recall = matched.mean() if len(matched) > 0 else 0.0
        rows.append({'video_id': vid, 'num_instances': n_inst,
                     'matched': int(matched.sum()), 'total': n_inst,
                     'recall': recall})
    df = pd.DataFrame(rows)

    # density tiers
    bins = [0, 5, 15, 50, 999]
    labels = ['1-5', '6-15', '16-50', '>50']
    df['density_tier'] = pd.cut(df['num_instances'], bins=bins,
                                labels=labels, right=True)
    summary = df.groupby('density_tier', observed=False).agg(
        total_videos=('video_id', 'count'),
        avg_recall=('recall', 'mean'),
        total_matched=('matched', 'sum'),
        total_instances=('total', 'sum'),
    ).reset_index()
    summary['macro_recall'] = summary['total_matched'] / summary['total_instances']
    return df, summary


# ---------------------------------------------------------------------------
# 2. Same-class co-detection rate
# ---------------------------------------------------------------------------

def compute_co_detection(gt, preds, gap_thresh=3.0, tiou_thresh=0.5):
    """For same-class instance pairs with temporal gap < gap_thresh,
    compute the rate at which BOTH are correctly detected."""
    rows = []
    for vid, vgt in gt.items():
        vpred = preds.get(vid, {'segments': np.empty((0, 2)),
                                'labels': np.empty(0, dtype=int),
                                'scores': np.empty(0)})
        segs, labs = vgt['segments'], vgt['labels']
        n = len(segs)
        for i in range(n):
            for j in range(i + 1, n):
                if labs[i] != labs[j]:
                    continue
                # temporal gap
                gap = max(segs[i][0] - segs[j][1], segs[j][0] - segs[i][1])
                if gap >= gap_thresh:
                    continue
                # check individual detection
                matched_i = match_predictions(
                    np.array([segs[i]]), np.array([labs[i]]),
                    vpred['segments'], vpred['labels'], vpred['scores'],
                    tiou_thresh)[0]
                matched_j = match_predictions(
                    np.array([segs[j]]), np.array([labs[j]]),
                    vpred['segments'], vpred['labels'], vpred['scores'],
                    tiou_thresh)[0]
                rows.append({
                    'video_id': vid,
                    'class': labs[i],
                    'gap_sec': gap,
                    'both_detected': int(matched_i and matched_j),
                    'only_first': int(matched_i and not matched_j),
                    'only_second': int(not matched_i and matched_j),
                    'neither': int(not matched_i and not matched_j),
                })
    if not rows:
        return pd.DataFrame(), {'co_detection_rate': 0.0, 'num_pairs': 0}
    df = pd.DataFrame(rows)
    summary = {
        'co_detection_rate': df['both_detected'].mean(),
        'num_pairs': len(df),
        'at_least_one_rate': (df['both_detected'].sum() +
                              df['only_first'].sum() +
                              df['only_second'].sum()) / len(df),
    }
    return df, summary


# ---------------------------------------------------------------------------
# 3. Overlap bucket recall
# ---------------------------------------------------------------------------

def compute_overlap_bucket_recall(gt, preds, tiou_thresh=0.5):
    """For each GT instance, compute the fraction of its duration that
    overlaps with other GT instances (of any class). Bucket by this
    fraction and report per-bucket recall."""
    rows = []
    for vid, vgt in gt.items():
        vpred = preds.get(vid, {'segments': np.empty((0, 2)),
                                'labels': np.empty(0, dtype=int),
                                'scores': np.empty(0)})
        segs = vgt['segments']
        labs = vgt['labels']
        n = len(segs)
        if n == 0:
            continue

        # per-instance matching
        matched = match_predictions(segs, labs,
                                    vpred['segments'], vpred['labels'],
                                    vpred['scores'], tiou_thresh)

        for i in range(n):
            duration = segs[i][1] - segs[i][0]
            if duration <= 0:
                overlap_frac = 0.0
            else:
                max_overlap = 0.0
                for j in range(n):
                    if i == j:
                        continue
                    inter = max(0.0, min(segs[i][1], segs[j][1]) -
                                max(segs[i][0], segs[j][0]))
                    overlap_frac_ij = inter / duration
                    max_overlap = max(max_overlap, overlap_frac_ij)
                overlap_frac = max_overlap

            rows.append({
                'video_id': vid,
                'instance_idx': i,
                'class': labs[i],
                'duration': duration,
                'max_overlap_frac': overlap_frac,
                'detected': int(matched[i]),
            })

    df = pd.DataFrame(rows)
    if df.empty:
        return df, pd.DataFrame()

    # bucket
    bins = [-0.01, 0.0, 0.3, 0.7, 1.01]
    labels = ['0%', '0-30%', '30-70%', '>70%']
    df['overlap_bucket'] = pd.cut(df['max_overlap_frac'], bins=bins,
                                  labels=labels, right=True)
    summary = df.groupby('overlap_bucket', observed=False).agg(
        num_instances=('detected', 'count'),
        num_detected=('detected', 'sum'),
    ).reset_index()
    summary['recall'] = summary['num_detected'] / summary['num_instances']
    return df, summary


# ---------------------------------------------------------------------------
# 4. Failure-case timeline visualisation
# ---------------------------------------------------------------------------

def find_worst_failures(gt, preds, tiou_thresh=0.5, top_n=10):
    """Return top-N videos sorted by number of missed detections in
    high-overlap scenarios."""
    scores = []
    for vid, vgt in gt.items():
        vpred = preds.get(vid, {'segments': np.empty((0, 2)),
                                'labels': np.empty(0, dtype=int),
                                'scores': np.empty(0)})
        segs, labs = vgt['segments'], vgt['labels']
        n = len(segs)
        if n < 3:
            continue

        matched = match_predictions(segs, labs,
                                    vpred['segments'], vpred['labels'],
                                    vpred['scores'], tiou_thresh)
        # count misses among overlapping instances
        miss_count = 0
        for i in range(n):
            if matched[i]:
                continue
            # check if this instance overlaps with another
            for j in range(n):
                if i == j:
                    continue
                iou = segment_iou(segs[i], segs[j])
                if iou > 0.1:
                    miss_count += 1
                    break
        scores.append((vid, miss_count, n, matched.sum()))

    scores.sort(key=lambda x: x[1], reverse=True)
    return scores[:top_n]


def plot_timeline(gt_vid, pred_vid, output_path):
    """Matplotlib timeline chart for one video."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print("[WARN] matplotlib not available — skipping timeline plots")
        return

    duration = gt_vid.get('duration', max(
        gt_vid['segments'][:, 1].max() if len(gt_vid['segments']) > 0 else 10,
        pred_vid['segments'][:, 1].max() if len(pred_vid['segments']) > 0 else 10,
    ))

    fig, (ax_gt, ax_pred) = plt.subplots(2, 1, figsize=(14, 6), sharex=True)

    colors = plt.cm.tab20(np.linspace(0, 1, 20))

    # Ground truth
    for i, (seg, lab) in enumerate(zip(gt_vid['segments'], gt_vid['labels'])):
        ax_gt.barh(0, seg[1] - seg[0], left=seg[0], height=0.6,
                   color=colors[lab % 20], alpha=0.7, edgecolor='black')
        ax_gt.text(seg[0] + (seg[1] - seg[0]) / 2, 0, str(lab),
                   ha='center', va='center', fontsize=8, fontweight='bold')
    ax_gt.set_ylabel('GT')
    ax_gt.set_yticks([])
    ax_gt.set_xlim(0, duration)
    ax_gt.set_title('Ground Truth Actions')

    # Predictions
    pred_segs = pred_vid.get('segments', np.empty((0, 2)))
    pred_labs = pred_vid.get('labels', np.empty(0, dtype=int))
    pred_sc = pred_vid.get('scores', np.empty(0))
    order = np.argsort(pred_sc)[::-1] if len(pred_sc) > 0 else []
    for rank, pi in enumerate(order[:50]):  # top 50
        seg = pred_segs[pi]
        lab = pred_labs[pi]
        sc = pred_sc[pi]
        ax_pred.barh(0, seg[1] - seg[0], left=seg[0], height=0.6,
                     color=colors[lab % 20], alpha=min(1.0, sc + 0.3),
                     edgecolor='black' if sc > 0.5 else 'gray')
        if sc > 0.3:
            ax_pred.text(seg[0] + (seg[1] - seg[0]) / 2, 0,
                         f'{lab}({sc:.1f})', ha='center', va='center',
                         fontsize=6)
    ax_pred.set_ylabel('Pred')
    ax_pred.set_yticks([])
    ax_pred.set_xlabel('Time (seconds)')
    ax_pred.set_xlim(0, duration)
    ax_pred.set_title('Predictions (top-50)')

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=100)
    plt.close()
    print(f"  Saved timeline: {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Overlap / dense action detection error analysis')
    parser.add_argument('--gt_json', required=True,
                        help='Path to THUMOS14 annotation JSON')
    parser.add_argument('--pred_pkl', required=True,
                        help='Path to prediction pickle file')
    parser.add_argument('--split', default='validation',
                        help='Dataset split to analyse')
    parser.add_argument('--tiou', type=float, default=0.5,
                        help='tIoU threshold for matching')
    parser.add_argument('--output_dir', default='analysis/output',
                        help='Directory for output files')
    parser.add_argument('--plot_timelines', action='store_true', default=True,
                        help='Generate timeline failure-case plots')
    parser.add_argument('--no_plot', action='store_false', dest='plot_timelines',
                        help='Skip timeline plots')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    failures_dir = os.path.join(args.output_dir, 'failure_cases')
    os.makedirs(failures_dir, exist_ok=True)

    print("=" * 60)
    print("  TriDet Overlap / Dense Action Error Analysis")
    print("=" * 60)

    # Load data
    print("\n[1/5] Loading ground truth ...")
    gt = load_gt(args.gt_json, args.split)
    print(f"  Loaded {len(gt)} videos from split '{args.split}'")

    print("[2/5] Loading predictions ...")
    preds = load_preds(args.pred_pkl)
    print(f"  Loaded predictions for {len(preds)} videos")
    common = set(gt) & set(preds)
    print(f"  {len(common)} videos in common")

    # 1. Density-stratified recall
    print("\n[3/5] Computing density-stratified recall ...")
    density_df, density_summary = compute_density_recall(gt, preds, args.tiou)
    density_summary.to_csv(os.path.join(args.output_dir, 'density_recall.csv'),
                           index=False)
    print(density_summary.to_string())

    # 2. Co-detection rate
    print("\n[4/5] Computing same-class co-detection rate ...")
    codet_df, codet_summary = compute_co_detection(gt, preds, args.tiou)
    if codet_summary['num_pairs'] > 0:
        pd.DataFrame([codet_summary]).to_csv(
            os.path.join(args.output_dir, 'co_detection.csv'), index=False)
        print(f"  Co-detection rate: {codet_summary['co_detection_rate']:.3f}")
        print(f"  At-least-one rate: {codet_summary['at_least_one_rate']:.3f}")
        print(f"  Total same-class close pairs: {codet_summary['num_pairs']}")
    else:
        print("  No same-class close pairs found.")

    # 3. Overlap bucket recall
    print("\n[5/5] Computing overlap-bucket recall ...")
    overlap_df, overlap_summary = compute_overlap_bucket_recall(gt, preds, args.tiou)
    overlap_summary.to_csv(os.path.join(args.output_dir, 'overlap_bucket.csv'),
                           index=False)
    print(overlap_summary.to_string())

    # 4. Failure case timelines
    if args.plot_timelines:
        print("\n[Extra] Generating failure-case timelines ...")
        worst = find_worst_failures(gt, preds, args.tiou, top_n=10)
        for rank, (vid, misses, total, matched) in enumerate(worst[:10]):
            print(f"  [{rank+1}] {vid}: missed {misses} overlapping instances "
                  f"({matched}/{total} detected)")
            plot_timeline(
                gt[vid],
                preds.get(vid, {'segments': np.empty((0, 2)),
                                'labels': np.empty(0, dtype=int),
                                'scores': np.empty(0)}),
                os.path.join(failures_dir, f'{rank+1:02d}_{vid}.png'),
            )

    print(f"\nDone. Results saved to {args.output_dir}/")
    return 0


if __name__ == '__main__':
    sys.exit(main())

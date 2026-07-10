"""
统一的预测结果分析脚本 — 整合 confusion / density / duration / ablation 四维分析。

用法:
    # 全部维度分析
    python evaluate/analyze_predictions.py --pred <predictions.pkl> --json <gt.json> --analysis all

    # 单项分析
    python evaluate/analyze_predictions.py --pred <pkl> --json <gt.json> --analysis confusion
    python evaluate/analyze_predictions.py --pred <pkl> --json <gt.json> --analysis density
    python evaluate/analyze_predictions.py --pred <pkl> --json <gt.json> --analysis duration

    # 消融对比 (需要多个 summary.json)
    python evaluate/analyze_predictions.py --ablation-dir work/results/ --json <gt.json> --analysis ablation
"""

import os, sys, argparse, json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# =====================================================
# Publication-style matplotlib configuration
# =====================================================
PALETTE = {
    "blue_main":      "#0F4D92",
    "blue_secondary": "#3775BA",
    "green_3":        "#8BCF8B",
    "red_strong":     "#B64342",
    "red_1":          "#F6CFCB",
    "teal":           "#42949E",
    "violet":         "#9A4D8E",
    "neutral":        "#CFCECE",
    "highlight":      "#FFD700",
    "orange":         "#E67E22",
    "dark":           "#2C3E50",
}

def _apply_style():
    """Configure matplotlib rcParams for publication-quality figures."""
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Microsoft YaHei", "Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "font.size": 14,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "axes.linewidth": 2.0,
        "axes.unicode_minus": False,
        "legend.frameon": False,
        "legend.fontsize": 11,
        "svg.fonttype": "none",
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
    })

_apply_style()

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from libs.utils.metrics import segment_iou
from evaluate.common import load_gt, load_preds, preds_by_video

RESULT_DIR = os.path.join(REPO, 'evaluate', 'results')
os.makedirs(RESULT_DIR, exist_ok=True)


# ============================================================
# 1. 混淆矩阵分析
# ============================================================

def run_confusion_analysis(gts, label_names, pred_by_vid, tiou_thresh=0.5, out_path=None):
    """统计类别间误分类模式, 画热力图."""
    num_classes = len(label_names)
    cm = np.zeros((num_classes, num_classes), dtype=np.int32)

    for vid, gt_instances in gts.items():
        if vid not in pred_by_vid:
            continue
        pred = pred_by_vid[vid]
        matched_gt = set()
        # 按置信度排序
        order = np.argsort(-pred['scores'])
        for p_idx in order:
            p_seg = np.array([pred['segments'][p_idx][0], pred['segments'][p_idx][1]])
            p_label = int(pred['labels'][p_idx])
            best_tiou, best_idx = 0, -1
            for g_idx, (s, e, gt_label) in enumerate(gt_instances):
                if g_idx in matched_gt:
                    continue
                gt_seg = np.array([[s, e]])
                tiou = segment_iou(p_seg, gt_seg)[0]
                if tiou > best_tiou:
                    best_tiou, best_idx = tiou, g_idx
            if best_tiou >= tiou_thresh and best_idx >= 0:
                gt_label = gt_instances[best_idx][2]
                cm[gt_label, p_label] += 1
                matched_gt.add(best_idx)

    # 输出 Top 误分类对
    errors = [(i, j, int(cm[i, j]))
              for i in range(num_classes) for j in range(num_classes)
              if i != j and cm[i, j] > 0]
    errors.sort(key=lambda x: -x[2])

    print(f"\n{'='*60}")
    print(f"  混淆矩阵分析 (tIoU={tiou_thresh}, {num_classes} 类)")
    print(f"{'='*60}")
    print("  Top 10 误分类对 (GT → Pred):")
    for gt_idx, pred_idx, count in errors[:10]:
        gt_name = label_names.get(gt_idx, f'cls_{gt_idx}')
        pred_name = label_names.get(pred_idx, f'cls_{pred_idx}')
        print(f"    {gt_name:<22} → {pred_name:<22}  {count:>3}x")

    # 热力图
    row_sums = cm.sum(axis=1, keepdims=True).clip(min=1)
    cm_norm = cm.astype(float) / row_sums
    labels = [label_names.get(i, f'{i}') for i in range(num_classes)]

    fig, ax = plt.subplots(figsize=(14, 12))
    im = ax.imshow(cm_norm, cmap='YlOrRd', aspect='auto')
    ax.set_xticks(range(num_classes))
    ax.set_yticks(range(num_classes))
    ax.set_xticklabels(labels, rotation=90, fontsize=7)
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_xlabel('预测类别', fontsize=13)
    ax.set_ylabel('真实类别', fontsize=13)
    ax.set_title('THUMOS14 混淆矩阵', fontsize=14)
    cbar = plt.colorbar(im, ax=ax, shrink=0.82, fraction=0.046)
    cbar.set_label('GT 占比', fontsize=11)
    plt.tight_layout(pad=2)

    out = out_path or os.path.join(RESULT_DIR, 'confusion_matrix.png')
    plt.savefig(out, dpi=300)
    plt.close()
    print(f"  图表: {out}")
    return cm


# ============================================================
# 2. 密度分层分析
# ============================================================

DENSITY_BINS = [
    ('1-3', 1, 3),
    ('4-6', 4, 6),
    ('7-10', 7, 10),
    ('>10', 11, 999),
]


def compute_ap_simple(gt_instances, pred_instances, tiou_thresh):
    """简化版 AP 计算 (跨所有类别, 单 tIoU 阈值)."""
    if not gt_instances:
        return 0.0
    preds = sorted(pred_instances, key=lambda x: x['score'], reverse=True)
    npos = len(gt_instances)
    tp = np.zeros(len(preds))
    matched = np.ones(npos) * -1

    for i, pred in enumerate(preds):
        p_seg = np.array([pred['t-start'], pred['t-end']])
        best_tiou, best_j = 0, -1
        for j, (s, e, _) in enumerate(gt_instances):
            if matched[j] >= 0:
                continue
            tiou = segment_iou(p_seg, np.array([[s, e]]))[0]
            if tiou > best_tiou:
                best_tiou, best_j = tiou, j
        if best_tiou >= tiou_thresh and best_j >= 0:
            tp[i] = 1
            matched[best_j] = i

    tp_cumsum = np.cumsum(tp)
    fp_cumsum = np.cumsum(1 - tp)
    recalls = tp_cumsum / npos
    precisions = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-8)

    mrec = np.hstack([[0], recalls, [1]])
    mpre = np.hstack([[0], precisions, [0]])
    for i in range(len(mpre) - 1, 0, -1):
        mpre[i - 1] = max(mpre[i - 1], mpre[i])
    idx = np.where(mrec[1:] != mrec[:-1])[0] + 1
    return np.sum((mrec[idx] - mrec[idx - 1]) * mpre[idx])


def run_density_analysis(gts, pred_by_vid, tiou_thresh=0.5, out_path=None):
    """按视频动作密度分桶, 分别计算 mAP."""
    # 转换为 density_analysis 需要的格式
    preds_flat = []
    for vid, pred in pred_by_vid.items():
        for seg, score, label in zip(pred['segments'], pred['scores'], pred['labels']):
            preds_flat.append({
                'video-id': vid,
                't-start': float(seg[0]),
                't-end': float(seg[1]),
                'label': int(label),
                'score': float(score),
            })
    pred_by_vid_flat = {}
    for p in preds_flat:
        pred_by_vid_flat.setdefault(p['video-id'], []).append(p)

    bin_aps = {name: [] for name, _, _ in DENSITY_BINS}

    for vid, gt_instances in gts.items():
        n_gt = len(gt_instances)
        vid_preds = pred_by_vid_flat.get(vid, [])
        for bin_name, lo, hi in DENSITY_BINS:
            if lo <= n_gt <= hi:
                ap = compute_ap_simple(gt_instances, vid_preds, tiou_thresh)
                bin_aps[bin_name].append(ap)
                break

    print(f"\n{'='*60}")
    print(f"  密度分层分析 (tIoU={tiou_thresh})")
    print(f"{'='*60}")
    print(f"  {'密度分组':<12} {'视频数':>6} {'平均mAP':>10}")
    print(f"  {'-'*32}")

    values, labels = [], []
    for bin_name, lo, hi in DENSITY_BINS:
        aps = bin_aps[bin_name]
        avg_ap = np.mean(aps) * 100 if aps else 0.0
        values.append(avg_ap)
        labels.append(f'{bin_name}\n(n={len(aps)})')
        print(f"  {bin_name:<12} {len(aps):>6} {avg_ap:>10.1f}%")

    # 柱状图
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = [PALETTE["green_3"], PALETTE["teal"], PALETTE["orange"], PALETTE["red_strong"]]
    bars = ax.bar(labels, values, color=colors, edgecolor='black', linewidth=1.0)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f'{val:.1f}%', ha='center', fontsize=14)
    ax.set_ylabel('mAP (%)', fontsize=13)
    ax.set_title(f'视频密度分层 mAP (tIoU={tiou_thresh})', fontsize=14)
    ax.set_ylim(0, max(values) * 1.15 + 2 if max(values) > 0 else 100)
    plt.tight_layout(pad=2)

    out = out_path or os.path.join(RESULT_DIR, 'density_analysis.png')
    plt.savefig(out, dpi=300)
    plt.close()
    print(f"  图表: {out}")

    # 密度分布直方图
    densities = [len(gt_list) for gt_list in gts.values()]
    fig2, ax2 = plt.subplots(figsize=(8, 4))
    ax2.hist(densities, bins=20, edgecolor='black', linewidth=0.8, color=PALETTE["teal"])
    ax2.axvline(np.mean(densities), color=PALETTE["red_strong"], linestyle='--', linewidth=2,
                label=f'均值={np.mean(densities):.1f}')
    ax2.set_xlabel('每视频 GT 实例数', fontsize=13)
    ax2.set_ylabel('视频数量', fontsize=13)
    ax2.set_title('动作密度分布', fontsize=14)
    ax2.legend()
    plt.tight_layout(pad=2)

    hist_out = out_path.replace('.png', '_hist.png') if out_path else \
        os.path.join(RESULT_DIR, 'density_hist.png')
    plt.savefig(hist_out, dpi=300)
    plt.close()
    print(f"  图表: {hist_out}")


# ============================================================
# 3. 时长分层分析
# ============================================================

DURATION_BINS = [
    ('短 (<2s)',  0, 2),
    ('中 (2-10s)', 2, 10),
    ('长 (>10s)', 10, 1e6),
]


def run_duration_analysis(gts, pred_by_vid, tiou_thresh=0.5, out_path=None):
    """按动作时长分桶, 分别计算 recall."""
    bin_results = {name: {'gt': 0, 'matched': 0} for name, _, _ in DURATION_BINS}
    per_bin_recalls = {name: [] for name, _, _ in DURATION_BINS}

    for vid, gt_instances in gts.items():
        vid_pred = pred_by_vid.get(vid)
        if vid_pred is None:
            continue

        for bin_name, lo, hi in DURATION_BINS:
            bin_gts = [(s, e, l) for s, e, l in gt_instances if lo <= (e - s) < hi]
            if not bin_gts:
                continue
            bin_results[bin_name]['gt'] += len(bin_gts)

            # 计算此视频在该时长桶的 recall
            matched = set()
            for p_idx in range(len(vid_pred['segments'])):
                p_seg = np.array([vid_pred['segments'][p_idx][0],
                                  vid_pred['segments'][p_idx][1]])
                for g_idx, (s, e, _) in enumerate(bin_gts):
                    if g_idx in matched:
                        continue
                    tiou = segment_iou(p_seg, np.array([[s, e]]))[0]
                    if tiou >= tiou_thresh:
                        matched.add(g_idx)
                        break
            recall = len(matched) / len(bin_gts) if bin_gts else 1.0
            per_bin_recalls[bin_name].append(recall)

    print(f"\n{'='*60}")
    print(f"  时长分层分析 (tIoU={tiou_thresh})")
    print(f"{'='*60}")
    print(f"  {'时长分组':<15} {'GT实例数':>8} {'平均Recall':>10}")
    print(f"  {'-'*37}")

    labels_dur, values_dur = [], []
    for bin_name, lo, hi in DURATION_BINS:
        n_gt = bin_results[bin_name]['gt']
        recalls = per_bin_recalls[bin_name]
        avg_r = np.mean(recalls) * 100 if recalls else 0.0
        values_dur.append(avg_r)
        labels_dur.append(f'{bin_name}\n(n={n_gt})')
        print(f"  {bin_name:<15} {n_gt:>8} {avg_r:>10.1f}%")

    # 柱状图
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = [PALETTE["green_3"], PALETTE["teal"], PALETTE["red_strong"]]
    bars = ax.bar(labels_dur, values_dur, color=colors, edgecolor='black', linewidth=1.0)
    for bar, val in zip(bars, values_dur):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f'{val:.1f}%', ha='center', fontsize=14)
    ax.set_ylabel('召回率 (%)', fontsize=13)
    ax.set_title(f'动作时长分层 Recall (tIoU={tiou_thresh})', fontsize=14)
    ax.set_ylim(0, max(values_dur) * 1.15 + 5 if max(values_dur) > 0 else 100)
    plt.tight_layout(pad=2)

    out = out_path or os.path.join(RESULT_DIR, 'duration_analysis.png')
    plt.savefig(out, dpi=300)
    plt.close()
    print(f"  图表: {out}")


# ============================================================
# 4. 消融对比分析
# ============================================================

def run_ablation_comparison(ablation_dir, out_dir=None):
    """从 work/results/<exp_id>/summary.json 读取数据, 画消融对比图."""
    import glob as _glob
    out_dir = out_dir or RESULT_DIR

    summaries = {}
    # 支持目录结构: ablation_dir/<exp_id>/summary.json
    for exp_id in ['baseline', 'A1', 'A2']:
        sp = os.path.join(ablation_dir, exp_id, 'summary.json')
        if os.path.exists(sp):
            with open(sp) as f:
                summaries[exp_id] = json.load(f)

    if not summaries:
        print("  [SKIP] 无消融摘要数据")
        return

    print(f"\n{'='*60}")
    print("  消融实验对比")
    print(f"{'='*60}")

    for exp_id, s in summaries.items():
        print(f"  {exp_id:<10} avg_mAP = {s.get('avg_mAP', 0):.2f}%")

    if len(summaries) < 2:
        return

    # 柱状图
    exp_order = ['baseline', 'A1', 'A2']
    names = ['基线\nSGP + Trident-head',
             'A1: 移除 Trident\n(仅 SGP)',
             'A2: 移除 SGP*\n(Conv + Trident)']
    vals = [summaries.get(eid, {}).get('avg_mAP', 0) for eid in exp_order]
    colors = [PALETTE["red_strong"], PALETTE["blue_main"], PALETTE["violet"]]

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(range(len(exp_order)), vals, color=colors, edgecolor='black', linewidth=1.0, width=0.5)
    ax.set_xticks(range(len(exp_order)))
    ax.set_xticklabels(names, fontsize=10)
    for i, (bar, val) in enumerate(zip(bars, vals)):
        diff = val - vals[0]
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                f'{val:.2f}%\n({diff:+.2f}%)', ha='center', fontsize=14, fontweight='bold')
    ax.set_ylabel('平均 mAP (%)', fontsize=13)
    ax.set_title('消融实验：TriDet 组件拆解', fontsize=14)
    ax.set_ylim(0, max(vals) * 1.15 + 2)
    plt.tight_layout(pad=2)

    ab_path = os.path.join(out_dir, 'ablation_comparison.png')
    plt.savefig(ab_path, dpi=300)
    plt.close()
    print(f"  图表: {ab_path}")

    # tIoU 曲线
    fig2, ax2 = plt.subplots(figsize=(8, 5))
    tious = [0.3, 0.4, 0.5, 0.6, 0.7]
    markers = [('baseline', PALETTE["red_strong"], 'D', '基线'),
               ('A1', PALETTE["blue_main"], 'o', 'A1: 移除 Trident'),
               ('A2', PALETTE["violet"], 's', 'A2: 移除 SGP*')]
    for eid, color, marker, label in markers:
        if eid in summaries:
            maps = [float(v) for v in summaries[eid]['mAP_per_tiou'].values()]
            ax2.plot(tious, maps, f'{marker}-', color=color, linewidth=2,
                    markersize=8, label=label)

    ax2.set_xlabel('tIoU 阈值', fontsize=13)
    ax2.set_ylabel('mAP (%)', fontsize=13)
    ax2.set_title('mAP vs tIoU — 组件消融', fontsize=14)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.15, linewidth=0.5)
    plt.tight_layout(pad=2)

    tiou_path = os.path.join(out_dir, 'map_vs_tiou.png')
    plt.savefig(tiou_path, dpi=300)
    plt.close()
    print(f"  图表: {tiou_path}")


# ============================================================
# 主入口
# ============================================================

def main():
    parser = argparse.ArgumentParser(description='TriDet 预测结果统一分析')
    parser.add_argument('--pred', default=None,
                        help='预测 pickle 文件路径 (confusion/density/duration 需要)')
    parser.add_argument('--json', default=None,
                        help='THUMOS14 GT 标注 JSON 路径')
    parser.add_argument('--split', default='test', help='数据集 split')
    parser.add_argument('--tiou', type=float, default=0.5, help='tIoU 阈值')
    parser.add_argument('--out', default=None, help='图表输出路径前缀')
    parser.add_argument('--analysis', default='all',
                        choices=['confusion', 'density', 'duration', 'ablation', 'all'],
                        help='分析维度 (默认: all)')
    parser.add_argument('--ablation-dir', default=None,
                        help='消融结果目录 (含 baseline/A1/A2/summary.json)')
    args = parser.parse_args()

    analyses = (['confusion', 'density', 'duration', 'ablation']
                if args.analysis == 'all' else [args.analysis])

    # 加载数据 (ablation 不需要 pred pickle)
    gts, label_names = None, None
    pred_by_vid = None

    if any(a in analyses for a in ['confusion', 'density', 'duration']):
        if not args.json:
            print("错误: 需要 --json 参数 (GT 标注 JSON 路径)")
            sys.exit(1)
        if not args.pred:
            print("错误: 需要 --pred 参数 (预测 pickle 路径)")
            sys.exit(1)

        gts, label_names = load_gt(args.json, args.split)
        preds = load_preds(args.pred)
        pred_by_vid = preds_by_video(preds)
        print(f"加载: {len(gts)} 视频 GT, {len(pred_by_vid)} 视频预测, "
              f"{len(label_names)} 类")

    for analysis in analyses:
        if analysis == 'confusion':
            run_confusion_analysis(gts, label_names, pred_by_vid, args.tiou, args.out)
        elif analysis == 'density':
            run_density_analysis(gts, pred_by_vid, args.tiou, args.out)
        elif analysis == 'duration':
            run_duration_analysis(gts, pred_by_vid, args.tiou, args.out)
        elif analysis == 'ablation':
            if not args.ablation_dir:
                print("  [SKIP] 消融对比需要 --ablation-dir")
                continue
            run_ablation_comparison(args.ablation_dir, args.out)

    print(f"\n{'='*60}")
    print(f"  分析完成! 输出目录: {RESULT_DIR}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()

"""
分析 IoU Head 迭代实验, 生成图表。

数据: evaluate/results/iou_head_iterations.csv
迭代: R1 (4层+TAL) → R2 (2层+BCE) → R3 (2层+QFL+warmup)

用法:
    python evaluate/analyze_iou_head.py
"""

import os, sys, csv
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
RESULT_DIR = os.path.join(REPO, 'evaluate', 'results')
CSV_PATH = os.path.join(RESULT_DIR, 'iou_head_iterations.csv')

ITER_COLORS = {
    'R1': PALETTE["red_1"],
    'R2': PALETTE["orange"],
    'R3': PALETTE["blue_main"],
}

METRICS = ['mAP_03', 'mAP_05', 'mAP_07', 'avg_mAP']
METRIC_LABELS = ['mAP@0.3', 'mAP@0.5', 'mAP@0.7', '平均 mAP']
TIOU_STEPS = [0.3, 0.5, 0.7]


def load_results(csv_path):
    results = {}
    if not os.path.exists(csv_path):
        print(f"  [WARN] CSV not found: {csv_path}")
        return results
    with open(csv_path, 'r', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            eid = row.get('实验编号', '').strip()
            if not eid:
                continue
            results[eid] = {
                'exp_id': eid,
                'group': row.get('分组', ''),
                'name': row.get('配置变更', ''),
                'mAP_03': float(row.get('mAP@0.3', 0)),
                'mAP_05': float(row.get('mAP@0.5', 0)),
                'mAP_07': float(row.get('mAP@0.7', 0)),
                'avg_mAP': float(row.get('avg_mAP', 0)),
            }
    return results


def create_plots(results):
    if len(results) < 2:
        print("  数据不足")
        return

    # Sort by iteration order: R1, R2, R3
    ids = sorted(results.keys())

    # ====================================================================
    # Plot 1: tIoU 递进曲线 — mAP@0.3→0.5→0.7 for each iteration
    # ====================================================================
    fig, ax = plt.subplots(figsize=(10, 6))
    for eid in ids:
        r = results[eid]
        vals = [r['mAP_03'], r['mAP_05'], r['mAP_07']]
        color = ITER_COLORS.get(eid, PALETTE["neutral"])
        label = f"{eid}: {r['name'][:30]}"
        ax.plot(TIOU_STEPS, vals, 'o-', color=color, linewidth=2.5,
                markersize=10, label=label)
        for x, y in zip(TIOU_STEPS, vals):
            ax.annotate(f'{y:.1f}%', (x, y), textcoords="offset points",
                       xytext=(0, 10), ha='center', fontsize=11,
                       color=color, fontweight='bold')

    ax.set_xlabel('tIoU 阈值', fontsize=14)
    ax.set_ylabel('mAP (%)', fontsize=14)
    ax.set_title('IoU Head 迭代 — tIoU 递进曲线', fontsize=15)
    ax.set_xticks(TIOU_STEPS)
    ax.legend(fontsize=10, loc='lower left')
    ax.grid(True, alpha=0.15, linewidth=0.5)
    ax.set_ylim(30, max(results[e]['mAP_03'] for e in ids) * 1.15)

    plt.tight_layout(pad=2)
    path = os.path.join(RESULT_DIR, 'iou_head_tiou_curve.png')
    plt.savefig(path)
    plt.close()
    print(f"  图表: {path}")

    # ====================================================================
    # Plot 2: 迭代对比柱状图 — 3 迭代 × 4 指标分组
    # ====================================================================
    fig, ax = plt.subplots(figsize=(14, 6))
    n_iter = len(ids)
    n_metrics = len(METRICS)
    bar_width = 0.18
    group_width = n_metrics * bar_width + 0.25

    for ii, eid in enumerate(ids):
        r = results[eid]
        vals = [r[m] for m in METRICS]
        color = ITER_COLORS.get(eid, PALETTE["neutral"])
        x_positions = [ii * group_width + mi * bar_width
                       - (n_metrics - 1) * bar_width / 2
                       for mi in range(n_metrics)]
        bars = ax.bar(x_positions, vals, bar_width, color=color,
                      edgecolor='black', linewidth=1.0,
                      label=f"{eid}: {r['name'][:25]}")
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.4,
                    f'{val:.1f}', ha='center', fontsize=9, color=color,
                    fontweight='bold')

    # Difference annotations between iterations
    for ii in range(len(ids) - 1):
        r_prev = results[ids[ii]]
        r_next = results[ids[ii + 1]]
        gain = r_next['avg_mAP'] - r_prev['avg_mAP']
        mid_x = (ii + 0.5) * group_width + 3 * bar_width
        ax.annotate(f'+{gain:.1f}%', xy=(mid_x, 72),
                    ha='center', fontsize=12, fontweight='bold',
                    color=PALETTE["green_3"],
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))

    group_centers = [i * group_width for i in range(n_iter)]
    ax.set_xticks(group_centers)
    ax.set_xticklabels([f"{eid}\n{results[eid]['name'][:20]}" for eid in ids],
                       fontsize=10)
    ax.set_ylabel('mAP (%)', fontsize=14)
    ax.set_title('IoU Head 迭代 — 多指标对比', fontsize=15)
    all_vals = [results[e][m] for e in ids for m in METRICS]
    ax.set_ylim(0, max(all_vals) * 1.15 + 2)

    # Metric legend label
    metric_note = '  '.join(METRIC_LABELS)
    ax.text(0.99, 0.94, f'指标 (左→右): {metric_note}',
            transform=ax.transAxes, ha='right', va='top',
            fontsize=9, color=PALETTE["dark"],
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

    ax.legend(fontsize=10, loc='upper left')

    plt.tight_layout(pad=2)
    path = os.path.join(RESULT_DIR, 'iou_head_comparison.png')
    plt.savefig(path)
    plt.close()
    print(f"  图表: {path}")

    # ====================================================================
    # Plot 3: 迭代改进步幅瀑布图
    # ====================================================================
    fig, ax = plt.subplots(figsize=(9, 6))
    base = results[ids[0]]
    cumsum = base['avg_mAP']
    x_labels = [f"{ids[0]}\n{base['name'][:18]}"]
    heights = [base['avg_mAP']]
    bar_colors = [ITER_COLORS[ids[0]]]
    bottom = [0]

    for ii in range(1, len(ids)):
        prev = results[ids[ii - 1]]
        curr = results[ids[ii]]
        gain = curr['avg_mAP'] - prev['avg_mAP']
        cumsum += gain

        # Show delta as a separate bar stacked on top
        x_labels.append(f"+{gain:+.1f}%")
        heights.append(gain)
        bar_colors.append(PALETTE["green_3"] if gain > 0 else PALETTE["red_strong"])
        bottom.append(prev['avg_mAP'])

    ax.bar(range(len(ids)), [base['avg_mAP']] + [0] * (len(ids) - 1),
           color=ITER_COLORS[ids[0]], edgecolor='black', linewidth=1.0,
           label=ids[0])

    for ii in range(1, len(ids)):
        prev = results[ids[ii - 1]]
        curr = results[ids[ii]]
        gain = curr['avg_mAP'] - prev['avg_mAP']
        c = PALETTE["green_3"] if gain > 0 else PALETTE["red_strong"]
        ax.bar(ii, gain, bottom=prev['avg_mAP'], color=c,
               edgecolor='black', linewidth=1.0,
               label=f'{ids[ii]}: {gain:+.1f}%')

    # Horizontal reference line at final value
    final_val = results[ids[-1]]['avg_mAP']
    ax.axhline(y=final_val, color=PALETTE["blue_main"], linestyle='--',
              linewidth=1.5, alpha=0.5,
              label=f'最终 avg_mAP={final_val:.2f}%')

    ax.set_xticks(range(len(ids)))
    ax.set_xticklabels(ids, fontsize=13)
    ax.set_ylabel('avg_mAP (%)', fontsize=14)
    ax.set_title('IoU Head 迭代 — 改进步幅', fontsize=15)
    ax.legend(fontsize=10, loc='upper left')
    ax.set_ylim(0, final_val * 1.15)

    plt.tight_layout(pad=2)
    path = os.path.join(RESULT_DIR, 'iou_head_waterfall.png')
    plt.savefig(path)
    plt.close()
    print(f"  图表: {path}")


def main():
    print("=" * 60)
    print("  IoU Head 迭代实验分析")
    print("=" * 60)

    results = load_results(CSV_PATH)
    if not results:
        print("  无结果数据")
        return
    print(f"  加载了 {len(results)} 项实验结果")

    try:
        create_plots(results)
    except Exception as e:
        print(f"  图表生成失败: {e}")
        import traceback
        traceback.print_exc()

    print(f"\n  分析完成! 输出目录: {RESULT_DIR}")


if __name__ == '__main__':
    main()

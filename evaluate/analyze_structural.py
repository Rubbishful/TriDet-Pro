"""
分析结构改进实验结果, 生成图表。

数据: evaluate/results/structural_improvements.csv
分组: BiFPN (2), RegLoss (3), SE (3)

用法:
    python evaluate/analyze_structural.py
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
CSV_PATH = os.path.join(RESULT_DIR, 'structural_improvements.csv')

GROUP_COLORS = {
    'BiFPN':    PALETTE["teal"],
    'RegLoss':  PALETTE["orange"],
    'SE':       PALETTE["blue_secondary"],
}

METRICS = ['mAP_03', 'mAP_05', 'mAP_07', 'avg_mAP']
METRIC_LABELS = ['mAP@0.3', 'mAP@0.5', 'mAP@0.7', '平均 mAP']


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

    ids = list(results.keys())
    groups = sorted(set(r['group'] for r in results.values()))

    # ====================================================================
    # Plot 1: 全量总览 — avg_mAP by group color
    # ====================================================================
    fig, ax = plt.subplots(figsize=(16, 6))
    avgs = [results[e]['avg_mAP'] for e in ids]
    colors = [GROUP_COLORS.get(results[e]['group'], PALETTE["neutral"]) for e in ids]
    names = [f"{e}\n{results[e]['name'][:18]}" for e in ids]

    bars = ax.bar(range(len(ids)), avgs, color=colors, edgecolor='black', linewidth=1.0)
    ax.set_xticks(range(len(ids)))
    ax.set_xticklabels(names, fontsize=9)
    ax.set_ylabel('平均 mAP (%)', fontsize=13)
    ax.set_title('结构改进实验 — 全量总览', fontsize=15)

    # Value annotations
    for bar, val in zip(bars, avgs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.4,
                f'{val:.2f}%', ha='center', fontsize=12, fontweight='bold',
                color=PALETTE["dark"])

    ax.set_ylim(0, max(avgs) * 1.12 + 2)

    # Group legend
    from matplotlib.patches import Patch
    legend_handles = [Patch(facecolor=c, label=g) for g, c in GROUP_COLORS.items()]
    ax.legend(handles=legend_handles, fontsize=11, loc='lower right')

    plt.tight_layout(pad=2)
    path = os.path.join(RESULT_DIR, 'structural_overview.png')
    plt.savefig(path)
    plt.close()
    print(f"  图表: {path}")

    # ====================================================================
    # Plot 2: 多指标分组柱状图 — 8 实验 × 4 指标
    # ====================================================================
    fig, ax = plt.subplots(figsize=(24, 7))
    n_exp = len(ids)
    n_metrics = len(METRICS)
    bar_width = 0.18
    group_width = n_metrics * bar_width + 0.25

    for ei, eid in enumerate(ids):
        r = results[eid]
        vals = [r[m] for m in METRICS]
        gcolor = GROUP_COLORS.get(r['group'], PALETTE["neutral"])
        x_positions = [ei * group_width + mi * bar_width
                       - (n_metrics - 1) * bar_width / 2
                       for mi in range(n_metrics)]
        bars = ax.bar(x_positions, vals, bar_width, color=gcolor,
                      edgecolor='black', linewidth=0.8,
                      label=f"{eid}: {r['name'][:20]}" if ei < 5 else None)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                    f'{val:.0f}', ha='center', fontsize=7, color=gcolor)

    group_centers = [i * group_width for i in range(n_exp)]
    ax.set_xticks(group_centers)
    ax.set_xticklabels([f"{eid}\n{results[eid]['name'][:16]}" for eid in ids],
                       fontsize=8)
    ax.set_ylabel('mAP (%)', fontsize=13)
    ax.set_title('结构改进 — 多指标分组对比', fontsize=15)
    all_vals = [results[e][m] for e in ids for m in METRICS]
    ax.set_ylim(0, max(all_vals) * 1.15 + 2)

    # Metric legend as text
    from matplotlib.patches import Patch
    metric_colors = [PALETTE["blue_main"], PALETTE["teal"],
                     PALETTE["orange"], PALETTE["red_strong"]]
    metric_patches = [Patch(facecolor=c, label=l)
                      for c, l in zip(metric_colors, METRIC_LABELS)]
    # Add a small note about metric colors via text
    ax.text(0.99, 0.95, '  '.join(METRIC_LABELS),
            transform=ax.transAxes, ha='right', va='top',
            fontsize=9, color=PALETTE["dark"],
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

    plt.tight_layout(pad=2)
    path = os.path.join(RESULT_DIR, 'structural_multimetric.png')
    plt.savefig(path)
    plt.close()
    print(f"  图表: {path}")

    # ====================================================================
    # Plot 3: 组内最佳 vs 全量对比 (avg_mAP highlight)
    # ====================================================================
    fig, ax = plt.subplots(figsize=(12, 6))
    group_best = {}
    for g in groups:
        group_exps = [e for e in ids if results[e]['group'] == g]
        best = max(group_exps, key=lambda e: results[e]['avg_mAP'])
        group_best[g] = best

    best_ids = list(group_best.values())
    best_avgs = [results[e]['avg_mAP'] for e in best_ids]
    best_names = [f"{results[e]['exp_id']}\n{results[e]['name'][:20]}" for e in best_ids]
    best_colors = [GROUP_COLORS[results[e]['group']] for e in best_ids]

    bars = ax.bar(range(len(best_ids)), best_avgs, color=best_colors,
                  edgecolor='black', linewidth=1.5, width=0.45)
    ax.set_xticks(range(len(best_ids)))
    ax.set_xticklabels(best_names, fontsize=11)
    ax.set_ylabel('平均 mAP (%)', fontsize=13)
    ax.set_title('各组最优方案对比', fontsize=15)

    for bar, val in zip(bars, best_avgs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f'{val:.2f}%', ha='center', fontsize=14, fontweight='bold',
                color=PALETTE["dark"])

    ax.set_ylim(0, max(best_avgs) * 1.15 + 3)
    plt.tight_layout(pad=2)
    path = os.path.join(RESULT_DIR, 'structural_best.png')
    plt.savefig(path)
    plt.close()
    print(f"  图表: {path}")


def main():
    print("=" * 60)
    print("  结构改进实验分析")
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

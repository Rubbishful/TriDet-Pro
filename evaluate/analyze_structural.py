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

BASELINE_MAP = 68.51
UNIFIED_COLORS = [PALETTE["blue_main"], PALETTE["blue_secondary"], PALETTE["teal"]]

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


def _plot_group_bar(results, group, filename, colors, title):
    """Grouped bar chart: baseline + experiments in *group*, 4 metrics side-by-side."""
    exps = sorted([e for e, r in results.items() if r['group'] == group])
    if not exps:
        return

    # Baseline + experiments
    all_bars = ['基线'] + exps
    n_all = len(all_bars)
    n_metrics = len(METRICS)
    bar_width = 0.16
    group_width = n_all * bar_width + 0.22
    fig_width = max(10, n_metrics * 2.2 + 3)
    fig, ax = plt.subplots(figsize=(fig_width, 6))

    baseline_vals = [BASELINE_MAP] * n_metrics
    baseline_color = PALETTE["red_strong"]

    for bi, bar_label in enumerate(all_bars):
        if bar_label == '基线':
            vals = baseline_vals
            color = baseline_color
            label = f'基线 ({BASELINE_MAP})'
        else:
            r = results[bar_label]
            vals = [r[m] for m in METRICS]
            ei = exps.index(bar_label)
            color = colors[ei % len(colors)]
            label = f"{bar_label}: {r['name']}"

        x_positions = [mi * group_width + bi * bar_width
                       - (n_all - 1) * bar_width / 2
                       for mi in range(n_metrics)]
        bars = ax.bar(x_positions, vals, bar_width, color=color,
                      edgecolor='black', linewidth=1.0, label=label)
        for bar, val in zip(bars, vals):
            clr = PALETTE["dark"] if bar_label == '基线' else color
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.8,
                    f'{val:.1f}', ha='center', fontsize=10, color=clr,
                    fontweight='bold')

    group_centers = [i * group_width for i in range(n_metrics)]
    ax.set_xticks(group_centers)
    ax.set_xticklabels(METRIC_LABELS, fontsize=13)
    ax.set_ylabel('mAP (%)', fontsize=14)
    ax.set_title(title, fontsize=16)
    all_vals = [r[m] for r in [results[e] for e in exps] for m in METRICS] + baseline_vals
    ax.set_ylim(0, max(all_vals) * 1.18 + 2)
    ax.legend(fontsize=8, ncol=1, loc='upper left',
              bbox_to_anchor=(1.01, 1))

    plt.tight_layout(pad=2)
    path = os.path.join(RESULT_DIR, filename)
    plt.savefig(path)
    plt.close()
    print(f"  图表: {path}")


def create_plots(results):
    if len(results) < 2:
        print("  数据不足")
        return

    ids = list(results.keys())
    groups = sorted(set(r['group'] for r in results.values()))

    # ====================================================================
    # Plot 1: BiFPN 分组对比
    # ====================================================================
    _plot_group_bar(results, 'BiFPN', 'structural_bifpn.png',
                    UNIFIED_COLORS,
                    'BiFPN 结构对比')

    # ====================================================================
    # Plot 2: RegLoss 分组对比
    # ====================================================================
    _plot_group_bar(results, 'RegLoss', 'structural_regloss.png',
                    UNIFIED_COLORS,
                    '回归损失函数对比')

    # ====================================================================
    # Plot 3: SE 分组对比
    # ====================================================================
    _plot_group_bar(results, 'SE', 'structural_se.png',
                    UNIFIED_COLORS,
                    'SE 通道注意力对比')

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
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.8,
                    f'{val:.1f}', ha='center', fontsize=7, color=gcolor)

    group_centers = [i * group_width for i in range(n_exp)]
    ax.set_xticks(group_centers)
    ax.set_xticklabels([f"{eid}\n{results[eid]['name'][:40]}" for eid in ids],
                       fontsize=8)
    ax.set_ylabel('mAP (%)', fontsize=13)
    ax.set_title('结构改进 — 多指标分组对比', fontsize=15)
    all_vals = [results[e][m] for e in ids for m in METRICS] + [BASELINE_MAP] * n_metrics
    ax.set_ylim(0, max(all_vals) * 1.15 + 2)

    # Baseline horizontal reference
    ax.axhline(y=BASELINE_MAP, color=PALETTE["red_strong"], linestyle='--',
              linewidth=1, alpha=0.6, label=f'基线 ({BASELINE_MAP})')
    ax.legend(fontsize=8, loc='lower right')

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
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.8,
                f'{val:.2f}%', ha='center', fontsize=14, fontweight='bold',
                color=PALETTE["dark"])

    ax.set_ylim(0, max(best_avgs + [BASELINE_MAP]) * 1.15 + 3)

    # Baseline reference line
    ax.axhline(y=BASELINE_MAP, color=PALETTE["red_strong"], linestyle='--',
              linewidth=1.5, alpha=0.6, label=f'基线 ({BASELINE_MAP})')
    ax.legend(fontsize=10, loc='lower right')
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

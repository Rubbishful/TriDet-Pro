"""
分析消融实验结果, 生成综合报告和图表。

用法:
    python work_1/analyze.py
"""

import os, sys, csv, json
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
WORK_DIR = os.path.join(REPO, 'evaluate', 'results')
CSV_PATH = os.path.join(WORK_DIR, 'ablation_results.csv')
REPORT_PATH = os.path.join(WORK_DIR, 'ablation_report.md')

# =====================================================
# A1-A8 实验结构定义
# =====================================================
ABLATION_SPEC = {
    'A1': {
        'title': 'A1: Trident-head → 普通回归头',
        'question': '离散化边界回归是否优于直接连续回归？',
        'baseline': 'baseline',
        'experiments': ['A1'],
        'param_name': 'use_trident_head',
        'param_values': {'baseline': True, 'A1': False},
        'expected': 'avg_mAP -2~4%, 高tIoU下降更明显',
    },
    'A2': {
        'title': 'A2: SGP Backbone → Conv Backbone',
        'question': 'SGP 的 5 分支多尺度感知对性能贡献有多大？',
        'baseline': 'baseline',
        'experiments': ['A2'],
        'param_name': 'backbone_type',
        'param_values': {'baseline': 'SGP', 'A2': 'conv'},
        'expected': 'avg_mAP -5~7%, 影响最大',
    },
    'A3': {
        'title': 'A3: SGP 窗口尺寸消融',
        'question': 'convkw 卷积核尺寸如何影响多尺度特征提取？',
        'baseline': 'A3_w1',
        'experiments': ['A3_w1','A3_w3','A3_w5','A3_w7','A3_w9','A3_w11','A3_per_layer'],
        'param_name': 'n_sgp_win_size',
        'param_values': {'A3_w1': 1, 'A3_w3': 3, 'A3_w5': 5, 'A3_w7': 7, 'A3_w9': 9, 'A3_w11': 11, 'A3_per_layer': '逐层'},
        'expected': '存在最优窗口值',
    },
    'A4': {
        'title': 'A4: SGP 的 k 参数消融',
        'question': '大窗口分支 convkw 的扩展比例对多尺度融合的敏感度？',
        'baseline': 'A4_k5.0',
        'experiments': ['A4_k1.0','A4_k1.5','A4_k3.0','A4_k5.0','A4_k7.0'],
        'param_name': 'k',
        'param_values': {'A4_k1.0': 1.0, 'A4_k1.5': 1.5, 'A4_k3.0': 3.0, 'A4_k5.0': 5.0, 'A4_k7.0': 7.0},
        'expected': 'k=1.5~5.0 optimal',
    },
    'A5': {
        'title': 'A5: DIoU → GIoU 损失',
        'question': '1D 时序检测中 GIoU 退化为 IoU, DIoU 的优势多大？',
        'baseline': 'baseline',
        'experiments': ['A5'],
        'param_name': 'loss_type',
        'param_values': {'baseline': 'diou', 'A5': 'giou'},
        'expected': 'DIoU > GIoU, 高tIoU更明显',
    },
    'A6': {
        'title': 'A6: FPN vs Identity Neck',
        'question': 'SGP 门控已提供跨尺度信息, FPN 融合是否冗余？',
        'baseline': 'A6_identity',
        'experiments': ['A6_identity','A6_fpn'],
        'param_name': 'fpn_type',
        'param_values': {'A6_identity': 'identity', 'A6_fpn': 'fpn'},
        'expected': 'Identity ≈ FPN (±1~3%)',
    },
    'A7': {
        'title': 'A7: 中心采样半径消融',
        'question': '正样本分配策略中, 中心采样半径多大最优？',
        'baseline': 'A7_r1.5',
        'experiments': ['A7_r0.0','A7_r0.5','A7_r1.0','A7_r1.5','A7_r2.0'],
        'param_name': 'center_sample_radius',
        'param_values': {'A7_r0.0': 0.0, 'A7_r0.5': 0.5, 'A7_r1.0': 1.0, 'A7_r1.5': 1.5, 'A7_r2.0': 2.0},
        'expected': 'r=1.0~1.5 optimal',
    },
    'A8': {
        'title': 'A8: 分类损失权重消融',
        'question': '分类与回归损失的最优权重比？',
        'baseline': 'A8_lw1.0',
        'experiments': ['A8_lw0.5','A8_lw1.0','A8_lw2.0','A8_lw5.0'],
        'param_name': 'loss_weight',
        'param_values': {'A8_lw0.5': 0.5, 'A8_lw1.0': 1.0, 'A8_lw2.0': 2.0, 'A8_lw5.0': 5.0},
        'expected': '过高→分类主导, 过低→回归主导',
    },
}


def load_results(csv_path):
    """加载CSV结果, 返回 {exp_id: {field: value}}."""
    results = {}
    if not os.path.exists(csv_path):
        print(f"  [WARN] CSV not found: {csv_path}")
        return results
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            exp_id = row.get('实验编号', '').strip()
            if not exp_id:
                continue
            results[exp_id] = {
                'exp_id': exp_id,
                'group': row.get('分组', ''),
                'name': row.get('配置变更', ''),
                'mAP_03': _parse_num(row.get('mAP@0.3')),
                'mAP_05': _parse_num(row.get('mAP@0.5')),
                'mAP_07': _parse_num(row.get('mAP@0.7')),
                'avg_mAP': _parse_num(row.get('avg_mAP')),
                'train_time': row.get('训练时间(min)', ''),
                'status': row.get('状态', ''),
            }
    return results


def _parse_num(s):
    if s is None or str(s).strip() == '':
        return None
    try:
        return float(s)
    except (ValueError, TypeError):
        return None


def generate_report(results):
    """生成Markdown报告。"""
    bl = results.get('baseline', {})

    lines = []
    lines.append("# TriDet 消融实验报告 (A1-A8)")
    lines.append("")
    lines.append(f"**生成日期**: 2026-07-04 | **数据集**: THUMOS14 | **GPU**: RTX 4060 Laptop (8GB)")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## 实验结果汇总")
    lines.append("")
    lines.append("| 实验 | 描述 | mAP@0.3 | mAP@0.5 | mAP@0.7 | avg_mAP | 训练时间 | 状态 |")
    lines.append("|------|------|---------|---------|---------|---------|----------|------|")

    for exp_id in sorted(results.keys()):
        r = results[exp_id]
        m03 = f'{r["mAP_03"]:.2f}%' if r['mAP_03'] is not None else '—'
        m05 = f'{r["mAP_05"]:.2f}%' if r['mAP_05'] is not None else '—'
        m07 = f'{r["mAP_07"]:.2f}%' if r['mAP_07'] is not None else '—'
        avg = f'{r["avg_mAP"]:.2f}%' if r['avg_mAP'] is not None else '—'
        lines.append(f"| {exp_id} | {r['name']} | {m03} | {m05} | {m07} | {avg} | {r['train_time']} | {r['status']} |")

    lines.append("")
    lines.append("---")
    lines.append("")

    # ============ 每组消融的详细分析 ============
    for group_id, spec in ABLATION_SPEC.items():
        lines.append(f"## {spec['title']}")
        lines.append("")
        lines.append(f"**科学问题**: {spec['question']}")
        lines.append("")

        # 收集本组结果
        group_results = {}
        for eid in spec['experiments']:
            if eid in results:
                group_results[eid] = results[eid]
        if 'baseline' in spec['experiments'] or spec['baseline'] == 'baseline':
            if 'baseline' in results and 'baseline' not in group_results:
                group_results['baseline'] = results['baseline']

        if len(group_results) < 2:
            lines.append(f"*结果不完整 ({len(group_results)}/{len(spec['experiments'])} 项), 待补充*")
            lines.append("")
            continue

        # 对比表
        lines.append("| 实验 | 参数值 | mAP@0.3 | mAP@0.5 | mAP@0.7 | avg_mAP |")
        lines.append("|------|--------|---------|---------|---------|---------|")

        baseline_id = spec['baseline']
        bl_avg = None
        for eid in spec['experiments']:
            if eid not in group_results:
                continue
            r = group_results[eid]
            param_val = spec['param_values'].get(eid, '—')
            m03 = f'{r["mAP_03"]:.2f}%' if r['mAP_03'] is not None else '—'
            m05 = f'{r["mAP_05"]:.2f}%' if r['mAP_05'] is not None else '—'
            m07 = f'{r["mAP_07"]:.2f}%' if r['mAP_07'] is not None else '—'
            avg = f'{r["avg_mAP"]:.2f}%' if r['avg_mAP'] is not None else '—'
            lines.append(f"| {eid} | {param_val} | {m03} | {m05} | {m07} | {avg} |")
            if eid == baseline_id:
                bl_avg = r['avg_mAP']

        lines.append("")

        # 相对于基线的差异
        if bl_avg is not None:
            lines.append("**相对基线差异**:")
            lines.append("")
            lines.append("| 实验 | avg_mAP 差异 | 结论 |")
            lines.append("|------|-------------|------|")
            for eid in spec['experiments']:
                if eid not in group_results or eid == baseline_id:
                    continue
                r = group_results[eid]
                if r['avg_mAP'] is not None:
                    diff = r['avg_mAP'] - bl_avg
                    conclusion = _interpret_diff(diff, spec['expected'])
                    lines.append(f"| {eid} | {diff:+.2f}% | {conclusion} |")
            lines.append("")

        lines.append(f"**预期**: {spec['expected']}")
        lines.append("")
        lines.append("---")
        lines.append("")

    # ============ 综合发现 ============
    lines.append("## 综合发现与结论")
    lines.append("")

    # 各组件贡献排序
    contributions = []
    for group_id in ['A1', 'A2', 'A5', 'A6']:
        spec = ABLATION_SPEC[group_id]
        bl = results.get(spec['baseline'], {})
        for eid in spec['experiments']:
            if eid == spec['baseline']:
                continue
            r = results.get(eid, {})
            if bl.get('avg_mAP') and r.get('avg_mAP'):
                diff = bl['avg_mAP'] - r['avg_mAP']
                contributions.append((spec['title'], diff))

    if contributions:
        contributions.sort(key=lambda x: -x[1])
        lines.append("### 组件重要性排序")
        lines.append("")
        lines.append("| 排名 | 消融项 | avg_mAP 损失 |")
        lines.append("|------|--------|-------------|")
        for i, (name, diff) in enumerate(contributions, 1):
            lines.append(f"| {i} | {name} | {diff:.2f}% |")
        lines.append("")

    lines.append("### 最优超参数推荐")
    lines.append("")
    for group_id in ['A3', 'A4', 'A7', 'A8']:
        spec = ABLATION_SPEC[group_id]
        best_exp, best_avg = None, -1
        for eid in spec['experiments']:
            r = results.get(eid, {})
            if r.get('avg_mAP') and r['avg_mAP'] > best_avg:
                best_avg = r['avg_mAP']
                best_exp = eid
        if best_exp:
            param_val = spec['param_values'].get(best_exp, '?')
            lines.append(f"- **{spec['title']}**: 最优 `{spec['param_name']}={param_val}` (avg_mAP={best_avg:.2f}%)")
        else:
            lines.append(f"- **{spec['title']}**: 结果待补充")
    lines.append("")

    # 写入
    with open(REPORT_PATH, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    print(f"  报告已生成: {REPORT_PATH}")


def _interpret_diff(diff, expected):
    """根据差异给出定性结论。"""
    if abs(diff) < 0.5:
        return "无明显差异"
    if diff < 0:
        return f"负面: 性能退化 ({abs(diff):.1f}%)"
    else:
        return f"正面: 性能提升 ({diff:.1f}%)"


def create_plots(results):
    """生成消融对比图表。"""
    # 1. 总览柱状图 (颜色按分组)
    exp_order = ['baseline', 'A1', 'A2',
                 'A3_w1','A3_w3','A3_w5','A3_w7','A3_w9','A3_w11','A3_per_layer',
                 'A4_k1.0','A4_k1.5','A4_k3.0','A4_k5.0','A4_k7.0',
                 'A5',
                 'A6_identity','A6_fpn',
                 'A7_r0.0','A7_r0.5','A7_r1.0','A7_r1.5','A7_r2.0',
                 'A8_lw0.5','A8_lw1.0','A8_lw2.0','A8_lw5.0']
    exp_order = [e for e in exp_order if e in results]

    if len(exp_order) < 2:
        print("  数据不足, 无法生成图表")
        return

    group_colors = {
        'baseline': PALETTE["red_strong"], 'A1': PALETTE["blue_main"], 'A2': PALETTE["violet"],
        'A3': PALETTE["teal"], 'A4': PALETTE["orange"], 'A5': PALETTE["green_3"],
        'A6': PALETTE["blue_secondary"], 'A7': PALETTE["blue_secondary"], 'A8': PALETTE["violet"],
    }

    # --- Plot 1: 总览 bar chart ---
    fig, ax = plt.subplots(figsize=(28, 7))
    ids = exp_order
    avgs = [results[e].get('avg_mAP', 0) or 0 for e in ids]

    # Build per-bar colors with gradients for parameter-sweep groups
    _build_overview_colors(results, ids, group_colors)

    bars = ax.bar(range(len(ids)), avgs, color=colors, edgecolor='black', linewidth=1.0)
    ax.set_xticks(range(len(ids)))
    ax.set_xticklabels(ids, rotation=45, ha='right', fontsize=8)
    ax.set_ylabel('平均 mAP (%)', fontsize=13)
    ax.set_title('TriDet 消融实验总览', fontsize=14)

    # 基线 + 分组图例（单个合并 legend）
    bl_idx = exp_order.index('baseline') if 'baseline' in exp_order else -1
    legend_handles, legend_labels = _build_group_legend_handles()
    if bl_idx >= 0:
        baseline_line = plt.Line2D([0], [0], color=PALETTE["red_strong"],
                                   linestyle='--', linewidth=1, alpha=0.7,
                                   label=f'基线 ({avgs[bl_idx]:.2f}%)')
        ax.axhline(y=avgs[bl_idx], color=PALETTE["red_strong"],
                   linestyle='--', linewidth=1, alpha=0.7)
        legend_handles.insert(0, baseline_line)
        legend_labels.insert(0, baseline_line.get_label())
    ax.legend(handles=legend_handles, labels=legend_labels,
              fontsize=7.5, ncol=3, loc='upper center',
              bbox_to_anchor=(0.5, -0.14))

    plt.tight_layout(pad=2)
    path = os.path.join(WORK_DIR, 'ablation_overview.png')
    plt.savefig(path)
    plt.close()
    print(f"  图表: {path}")

    # --- Plot 2-5: 参数扫描折线图 (A3, A4, A7, A8) ---
    for group_id, spec in [('A3', ABLATION_SPEC['A3']),
                            ('A4', ABLATION_SPEC['A4']),
                            ('A7', ABLATION_SPEC['A7']),
                            ('A8', ABLATION_SPEC['A8'])]:
        x_vals, y_vals, labels = [], [], []
        for eid in spec['experiments']:
            if eid in results and results[eid].get('avg_mAP') is not None:
                # Skip 'per_layer' for line chart
                if eid == 'A3_per_layer':
                    continue
                pv = spec['param_values'].get(eid)
                if isinstance(pv, (int, float)):
                    x_vals.append(pv)
                    y_vals.append(results[eid]['avg_mAP'])
                    labels.append(eid)

        if len(x_vals) < 2:
            continue

        fig, ax = plt.subplots(figsize=(9, 5))
        ax.plot(x_vals, y_vals, 'o-', color=PALETTE["blue_main"], linewidth=2, markersize=8)
        for x, y, lbl in zip(x_vals, y_vals, labels):
            ax.annotate(f'{y:.2f}%', (x, y), textcoords="offset points",
                       xytext=(0, 8), ha='center', fontsize=12)

        # Expand y-axis for annotation headroom
        y_pad = (max(y_vals) - min(y_vals)) * 0.25
        ax.set_ylim(min(y_vals) - y_pad, max(y_vals) + y_pad * 2)

        # Mark baseline
        bl_id = spec['baseline']
        if bl_id in results and results[bl_id].get('avg_mAP') is not None:
            bl_val = spec['param_values'].get(bl_id)
            if isinstance(bl_val, (int, float)):
                ax.axvline(x=bl_val, color=PALETTE["red_strong"], linestyle='--', alpha=0.5, label=f'基线 ({bl_val})')
                ax.legend(fontsize=10)

        ax.set_xlabel(spec['param_name'], fontsize=13)
        ax.set_ylabel('平均 mAP (%)', fontsize=13)
        ax.set_title(spec['title'], fontsize=14)
        ax.grid(True, alpha=0.15, linewidth=0.5)
        plt.tight_layout(pad=2)
        path = os.path.join(WORK_DIR, f'ablation_{group_id.lower()}.png')
        plt.savefig(path)
        plt.close()
        print(f"  图表: {path}")

    # --- Plot 3: Core comparison (A1, A2, A5, A6 vs baseline) ---
    core_groups = ['A1', 'A2', 'A5', 'A6']
    fig, ax = plt.subplots(figsize=(13, 6))
    core_ids = ['baseline']
    core_names = ['基线']
    for gid in core_groups:
        spec = ABLATION_SPEC[gid]
        for eid in spec['experiments']:
            if eid in results and eid != spec.get('baseline') and eid != 'baseline':
                core_ids.append(eid)
                core_names.append(f"{eid}\n{results[eid].get('name', '')[:20]}")

    core_avgs = [results[e].get('avg_mAP', 0) or 0 for e in core_ids]
    core_colors = [PALETTE["red_strong"]] + [PALETTE["red_1"]] * (len(core_ids) - 1)

    bars = ax.bar(range(len(core_ids)), core_avgs, color=core_colors, edgecolor='black', linewidth=1.0, width=0.5)
    ax.set_xticks(range(len(core_ids)))
    ax.set_xticklabels(core_names, fontsize=11)
    ax.set_ylabel('平均 mAP (%)', fontsize=13)
    ax.set_title('核心消融：组件移除影响', fontsize=14)

    # Expand y-axis for annotation headroom
    ax.set_ylim(0, max(core_avgs) * 1.30 + 2)

    # 标注差值 — use bar-matched color with ample vertical offset
    bl_avg = core_avgs[0] if core_avgs else 0
    for i, (bar, val) in enumerate(zip(bars, core_avgs)):
        diff = val - bl_avg
        # Use the bar's own color darkened for text readability
        bar_color = core_colors[i]
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + (max(core_avgs) - min(core_avgs)) * 0.06 + 0.8,
                f'{val:.2f}%\n({diff:+.2f}%)', ha='center', fontsize=13,
                color=bar_color, fontweight='bold')

    plt.tight_layout(pad=2)
    path = os.path.join(WORK_DIR, 'ablation_core.png')
    plt.savefig(path)
    plt.close()
    print(f"  图表: {path}")

    # --- Plot 6: Multi-metric grouped bar (core ablations × 4 metrics) ---
    _plot_multimetric_core(results)

    # --- Plot 7: 参数扫描合并面板 (A3/A4/A7/A8 2×2) ---
    _plot_param_grid(results)


def _plot_multimetric_core(results):
    """Grouped bar chart: baseline vs each ablated variant across mAP thresholds."""
    metrics = ['mAP_03', 'mAP_05', 'mAP_07', 'avg_mAP']
    metric_labels = ['mAP@0.3', 'mAP@0.5', 'mAP@0.7', '平均 mAP']
    variant_colors = [PALETTE["red_strong"], PALETTE["blue_main"],
                      PALETTE["violet"], PALETTE["green_3"], PALETTE["teal"]]
    variant_labels = ['基线', 'A1: 移除 Trident', 'A2: 移除 SGP*',
                      'A5: GIoU', 'A6: FPN']
    variant_ids = ['baseline', 'A1', 'A2', 'A5', 'A6_fpn']

    n_metrics = len(metrics)
    n_variants = len(variant_ids)
    bar_width = 0.14
    group_width = n_variants * bar_width + 0.18

    fig, ax = plt.subplots(figsize=(20, 7))

    for vi, (vid, vlabel, vcolor) in enumerate(zip(variant_ids, variant_labels, variant_colors)):
        vals = [results.get(vid, {}).get(m, 0) or 0 for m in metrics]
        x_positions = [i * group_width + vi * bar_width
                       - (n_variants - 1) * bar_width / 2
                       for i in range(n_metrics)]
        bars = ax.bar(x_positions, vals, bar_width, color=vcolor,
                      edgecolor='black', linewidth=0.8, label=vlabel)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                    f'{val:.1f}', ha='center', fontsize=9, color=vcolor)

    group_centers = [i * group_width for i in range(n_metrics)]
    ax.set_xticks(group_centers)
    ax.set_xticklabels(metric_labels, fontsize=14)
    ax.set_ylabel('mAP (%)', fontsize=14)
    ax.set_title('核心消融 — 多指标对比', fontsize=16)
    all_vals = [(results.get(vid, {}).get(m, 0) or 0) for vid in variant_ids for m in metrics]
    ax.set_ylim(0, max(all_vals) * 1.18 + 2)
    ax.legend(fontsize=11, loc='upper right')

    plt.tight_layout(pad=2)
    path = os.path.join(WORK_DIR, 'ablation_multimetric.png')
    plt.savefig(path)
    plt.close()
    print(f"  图表: {path}")


def _plot_param_grid(results):
    """2×2 subplot grid: A3, A4, A7, A8 parameter sweeps."""
    groups = [('A3', ABLATION_SPEC['A3']),
              ('A4', ABLATION_SPEC['A4']),
              ('A7', ABLATION_SPEC['A7']),
              ('A8', ABLATION_SPEC['A8'])]

    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    axes = axes.flatten()

    for ax, (group_id, spec) in zip(axes, groups):
        x_vals, y_vals = [], []
        for eid in spec['experiments']:
            if eid in results and results[eid].get('avg_mAP') is not None:
                if eid == 'A3_per_layer':
                    continue
                pv = spec['param_values'].get(eid)
                if isinstance(pv, (int, float)):
                    x_vals.append(pv)
                    y_vals.append(results[eid]['avg_mAP'])

        if len(x_vals) < 2:
            ax.text(0.5, 0.5, '数据不足', ha='center', va='center',
                    transform=ax.transAxes, fontsize=14, color=PALETTE["neutral"])
            ax.set_title(spec['title'], fontsize=14)
            continue

        ax.plot(x_vals, y_vals, 'o-', color=PALETTE["blue_main"], linewidth=2, markersize=9)
        for x, y in zip(x_vals, y_vals):
            ax.annotate(f'{y:.1f}', (x, y), textcoords="offset points",
                       xytext=(0, 8), ha='center', fontsize=11)

        # Baseline marker
        bl_id = spec['baseline']
        if bl_id in results and results[bl_id].get('avg_mAP') is not None:
            bl_val = spec['param_values'].get(bl_id)
            if isinstance(bl_val, (int, float)):
                ax.axvline(x=bl_val, color=PALETTE["red_strong"], linestyle='--',
                          alpha=0.5, linewidth=1.5, label=f'基线 ({bl_val})')
                ax.legend(fontsize=10)

        y_pad = (max(y_vals) - min(y_vals)) * 0.25
        ax.set_ylim(min(y_vals) - y_pad, max(y_vals) + y_pad * 2)
        ax.set_xlabel(spec['param_name'], fontsize=13)
        ax.set_ylabel('平均 mAP (%)', fontsize=13)
        ax.set_title(spec['title'], fontsize=15)
        ax.grid(True, alpha=0.15, linewidth=0.5)

    fig.suptitle('参数敏感性 — 网格总览', fontsize=18, fontweight='bold', y=1.01)
    plt.tight_layout(pad=2)
    path = os.path.join(WORK_DIR, 'ablation_param_grid.png')
    plt.savefig(path)
    plt.close()
    print(f"  图表: {path}")


# =====================================================
# Overview chart color helpers
# =====================================================

def _interpolate_color(hex_color, alpha):
    """Blend a hex color towards white by given alpha (1.0 = full, 0.0 = white)."""
    r = int(hex_color[1:3], 16)
    g = int(hex_color[3:5], 16)
    b = int(hex_color[5:7], 16)
    r = int(r * alpha + 255 * (1 - alpha))
    g = int(g * alpha + 255 * (1 - alpha))
    b = int(b * alpha + 255 * (1 - alpha))
    return f'#{r:02x}{g:02x}{b:02x}'


def _build_overview_colors(results, ids, group_colors):
    """Populate global *colors* list with per-bar colors.

    Parameter-sweep groups (A3, A4, A7, A8) get alpha gradients so
    parameter-progression is visible within each group.
    """
    global colors
    # Param groups: group_id -> (alpha_min, alpha_max)
    param_gradients = {
        'A3': (0.45, 1.0),   # teal gradient
        'A4': (0.42, 1.0),   # orange gradient
        'A7': (0.40, 1.0),   # blue gradient
        'A8': (0.48, 1.0),   # violet gradient
    }
    # Count bars per group for gradient step calculation
    group_counts = {}
    for eid in ids:
        grp = results[eid].get('group', 'A8')
        group_counts[grp] = group_counts.get(grp, 0) + 1

    colors = []
    group_idx = {}
    for eid in ids:
        grp = results[eid].get('group', 'A8')
        base = group_colors.get(grp, PALETTE["neutral"])
        if grp in param_gradients:
            n = group_counts[grp]
            idx = group_idx.get(grp, 0)
            alpha_min, alpha_max = param_gradients[grp]
            alpha = alpha_min + (alpha_max - alpha_min) * idx / max(n - 1, 1)
            colors.append(_interpolate_color(base, alpha))
            group_idx[grp] = idx + 1
        elif grp == 'A5':
            # Single bar, use full color
            colors.append(PALETTE["green_3"])
        else:
            colors.append(base)


def _build_group_legend_handles():
    """Return (handles, labels) for the overview group-color legend."""
    from matplotlib.patches import Patch
    handles = [
        Patch(facecolor=PALETTE["red_strong"],     label='基线'),
        Patch(facecolor=PALETTE["blue_main"],       label='A1: 移除 Trident-head'),
        Patch(facecolor=PALETTE["violet"],          label='A2: 移除 SGP'),
        Patch(facecolor=PALETTE["teal"],            label='A3: SGP 窗口尺寸'),
        Patch(facecolor=PALETTE["orange"],          label='A4: k 参数'),
        Patch(facecolor=PALETTE["green_3"],         label='A5: DIoU→GIoU'),
        Patch(facecolor=PALETTE["blue_secondary"],  label='A6: FPN / A7: 采样半径'),
        Patch(facecolor=PALETTE["red_1"],           label='A8: 损失权重'),
    ]
    labels = [h.get_label() for h in handles]
    return handles, labels


def main():
    print("=" * 60)
    print("  TriDet 消融实验分析")
    print("=" * 60)

    results = load_results(CSV_PATH)
    if not results:
        print("  无结果数据, 请先运行 train_all.py 收集实验结果")
        return

    print(f"  加载了 {len(results)} 项实验结果")

    # 生成报告
    generate_report(results)

    # 生成图表
    try:
        create_plots(results)
    except Exception as e:
        print(f"  图表生成失败: {e}")

    print("\n  分析完成!")
    print(f"  报告: {REPORT_PATH}")
    print(f"  图表: {WORK_DIR}/ablation_*.png")


if __name__ == '__main__':
    main()

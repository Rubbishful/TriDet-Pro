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
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WORK_DIR = os.path.join(REPO, 'evaluate')
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
        'baseline': '#2ecc71', 'A1': '#3498db', 'A2': '#e74c3c',
        'A3': '#f39c12', 'A4': '#9b59b6', 'A5': '#1abc9c',
        'A6': '#e67e22', 'A7': '#2c3e50', 'A8': '#95a5a6',
    }

    # --- Plot 1: 总览 bar chart ---
    fig, ax = plt.subplots(figsize=(16, 6))
    ids = exp_order
    avgs = [results[e].get('avg_mAP', 0) or 0 for e in ids]
    colors = [group_colors.get(results[e].get('group', 'A8'), '#95a5a6') for e in ids]

    bars = ax.bar(range(len(ids)), avgs, color=colors, edgecolor='white')
    ax.set_xticks(range(len(ids)))
    ax.set_xticklabels(ids, rotation=45, ha='right', fontsize=8)
    ax.set_ylabel('Average mAP (%)', fontsize=13)
    ax.set_title('TriDet Ablation Study — All Experiments', fontsize=14)

    # 标记基线
    bl_idx = exp_order.index('baseline') if 'baseline' in exp_order else -1
    if bl_idx >= 0:
        ax.axhline(y=avgs[bl_idx], color='#2ecc71', linestyle='--', linewidth=1, alpha=0.7, label=f'Baseline ({avgs[bl_idx]:.2f}%)')
        ax.legend(fontsize=10)

    plt.tight_layout()
    path = os.path.join(WORK_DIR, 'ablation_overview.png')
    plt.savefig(path, dpi=150)
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

        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(x_vals, y_vals, 'o-', color='#3498db', linewidth=2, markersize=8)
        for x, y, lbl in zip(x_vals, y_vals, labels):
            ax.annotate(f'{y:.2f}%', (x, y), textcoords="offset points",
                       xytext=(0, 10), ha='center', fontsize=9)

        # Mark baseline
        bl_id = spec['baseline']
        if bl_id in results and results[bl_id].get('avg_mAP') is not None:
            bl_val = spec['param_values'].get(bl_id)
            if isinstance(bl_val, (int, float)):
                ax.axvline(x=bl_val, color='#e74c3c', linestyle='--', alpha=0.5, label=f'Baseline ({bl_val})')
                ax.legend(fontsize=10)

        ax.set_xlabel(spec['param_name'], fontsize=13)
        ax.set_ylabel('Average mAP (%)', fontsize=13)
        ax.set_title(spec['title'], fontsize=14)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        path = os.path.join(WORK_DIR, f'ablation_{group_id.lower()}.png')
        plt.savefig(path, dpi=150)
        plt.close()
        print(f"  图表: {path}")

    # --- Plot 3: Core comparison (A1, A2, A5, A6 vs baseline) ---
    core_groups = ['A1', 'A2', 'A5', 'A6']
    fig, ax = plt.subplots(figsize=(10, 5))
    core_ids = ['baseline']
    core_names = ['Baseline']
    for gid in core_groups:
        spec = ABLATION_SPEC[gid]
        for eid in spec['experiments']:
            if eid in results and eid != spec.get('baseline') and eid != 'baseline':
                core_ids.append(eid)
                core_names.append(f"{eid}\n{results[eid].get('name', '')[:20]}")

    core_avgs = [results[e].get('avg_mAP', 0) or 0 for e in core_ids]
    core_colors = ['#2ecc71'] + ['#e74c3c'] * (len(core_ids) - 1)

    bars = ax.bar(range(len(core_ids)), core_avgs, color=core_colors, edgecolor='white', width=0.5)
    ax.set_xticks(range(len(core_ids)))
    ax.set_xticklabels(core_names, fontsize=9)
    ax.set_ylabel('Average mAP (%)', fontsize=13)
    ax.set_title('Core Ablation: Component Removal Impact', fontsize=14)

    # 标注差值
    bl_avg = core_avgs[0] if core_avgs else 0
    for i, (bar, val) in enumerate(zip(bars, core_avgs)):
        diff = val - bl_avg
        color = '#e74c3c' if diff < 0 else '#2ecc71'
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
                f'{val:.2f}%\n({diff:+.2f}%)', ha='center', fontsize=9, color=color)

    plt.tight_layout()
    path = os.path.join(WORK_DIR, 'ablation_core.png')
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  图表: {path}")


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

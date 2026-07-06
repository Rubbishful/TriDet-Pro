"""
TriDet 增强可视化引擎 — 从统一数据生成多维度分析图表。

所有函数接受 load_all_results() 的返回值, 返回 matplotlib Figure 对象,
同时保存到磁盘。可用于 MD 报告嵌入和 HTML 仪表板。

用法:
    from evaluate.visualize import plot_overview, plot_grid_heatmaps, ...
    fig = plot_overview(all_data, out_dir='evaluate/results/')
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.patches import FancyBboxPatch

matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

# ── 全局调色板 ──
PALETTE = {
    'baseline':    '#2ecc71',
    'ablation':    '#3498db',
    'structural':  '#e74c3c',
    'iou_head':    '#f39c12',
    'iou_grid':    '#9b59b6',
    'global':      '#1abc9c',
    'SE':          '#e67e22',
    'BiFPN':       '#2c3e50',
    'RegLoss':     '#8e44ad',
    'IoU Head':    '#d35400',
    'IoU Grid':    '#7f8c8d',
    '基线':         '#2ecc71',
}

GROUP_COLORS = {
    'ablation':   '#3498db',
    'structural': '#e74c3c',
    'iou_head':   '#f39c12',
    'iou_grid':   '#9b59b6',
}


def _get_group_color(exp):
    """根据实验来源返回颜色."""
    st = exp.get('source_type', 'other')
    return GROUP_COLORS.get(st, '#95a5a6')


def _fmt_val(v, default='—'):
    """安全格式化数值."""
    if v is None:
        return default
    return f'{v:.2f}'


def _save_and_close(fig, path, dpi=150):
    """保存并关闭 Figure."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    return path


# ============================================================
# 1. 总览柱状图
# ============================================================

def plot_overview(all_data, out_dir, top_n=40, figsize=(18, 7)):
    """所有有效实验按 avg_mAP 降序柱状图."""
    from evaluate.common import get_valid_experiments
    exps = get_valid_experiments(all_data)
    if not exps:
        return None

    exps = exps[:top_n]
    ids = [e['exp_id'] for e in exps]
    avgs = [e.get('avg_mAP', 0) or 0 for e in exps]
    colors = [_get_group_color(e) for e in exps]

    fig, ax = plt.subplots(figsize=figsize)
    bars = ax.bar(range(len(ids)), avgs, color=colors, edgecolor='white', linewidth=0.5)

    # 基线参考线
    bl_exp = next((e for e in exps if e['exp_id'] == 'Baseline' or e['exp_id'] == 'baseline'), None)
    bl_val = bl_exp.get('avg_mAP') if bl_exp else None
    if bl_val is not None:
        ax.axhline(y=bl_val, color='#2ecc71', linestyle='--', linewidth=1,
                   alpha=0.7, label=f'Baseline ({bl_val:.2f}%)')
        ax.legend(fontsize=10, loc='upper right')

    ax.set_xticks(range(len(ids)))
    ax.set_xticklabels(ids, rotation=70, ha='right', fontsize=7)
    ax.set_ylabel('Average mAP (%)', fontsize=13)
    ax.set_title(f'TriDet All Experiments — Top {len(ids)} by avg mAP', fontsize=14)
    ax.set_ylim(min(avgs) * 0.92, max(avgs) * 1.05)

    # 图例: 按来源
    from matplotlib.patches import Patch
    legend_patches = [Patch(color=c, label=l) for l, c in GROUP_COLORS.items()]
    ax.legend(handles=legend_patches + (
        [Patch(color='#2ecc71', label='Baseline')] if bl_val else []),
              fontsize=8, loc='lower left', ncol=2)

    plt.tight_layout()
    path = os.path.join(out_dir, 'comprehensive_overview.png')
    _save_and_close(fig, path)
    return fig


# ============================================================
# 2. 分组对比面板 (4 面板)
# ============================================================

def plot_group_comparison(all_data, out_dir, figsize=(16, 12)):
    """四面板并列: Ablation 核心 / Structural / IoU Head / IoU Grid Top-K."""
    from evaluate.common import classify_experiments, get_valid_experiments
    groups = classify_experiments(all_data)
    bl_val = None
    for eid, exp in all_data.get('experiments', {}).items():
        if eid in ('baseline', 'Baseline') and exp.get('avg_mAP') is not None:
            bl_val = exp['avg_mAP']
            break

    fig, axes = plt.subplots(2, 2, figsize=figsize)
    axes = axes.flatten()

    # ── Panel 1: Ablation 核心 (A1, A2, A5, A6 vs baseline) ──
    ax = axes[0]
    core_ids = ['baseline', 'Baseline', 'A1', 'A2', 'A5', 'A6_fpn']
    ab_exps = [e for e in groups.get('ablation', []) if e['exp_id'] in core_ids]
    # 补充 baseline 如果不在 ablation 组
    baseline_exp = all_data['experiments'].get('baseline') or all_data['experiments'].get('Baseline')
    if baseline_exp and baseline_exp['exp_id'] not in {e['exp_id'] for e in ab_exps}:
        ab_exps.insert(0, baseline_exp)

    if ab_exps:
        ids = [e['exp_id'] for e in ab_exps]
        avgs = [e.get('avg_mAP', 0) or 0 for e in ab_exps]
        colors_ab = ['#2ecc71' if eid in ('baseline', 'Baseline') else '#e74c3c' for eid in ids]
        bars = ax.bar(range(len(ids)), avgs, color=colors_ab, edgecolor='white', width=0.5)
        bl = avgs[0] if avgs else 0
        for bar, val in zip(bars, avgs):
            diff = val - bl
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                    f'{val:.2f}%\n({diff:+.2f}%)', ha='center', fontsize=9)
        ax.set_xticks(range(len(ids)))
        ax.set_xticklabels(ids, fontsize=9)
        ax.set_title('Ablation: Core Component Removal', fontsize=13)
        ax.set_ylabel('avg mAP (%)')
    else:
        ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Ablation: Core Component Removal', fontsize=13)

    # ── Panel 2: Structural (SE / BiFPN / RegLoss) ──
    ax = axes[1]
    struct_exps = groups.get('structural', [])
    if struct_exps:
        struct_exps.sort(key=lambda x: -(x.get('avg_mAP') or -999))
        ids = [e['exp_id'] for e in struct_exps]
        avgs = [e.get('avg_mAP', 0) or 0 for e in struct_exps]
        grp_colors = {'SE': '#e67e22', 'BiFPN': '#2c3e50', 'RegLoss': '#8e44ad'}
        colors_st = [grp_colors.get(e.get('group', ''), '#95a5a6') for e in struct_exps]
        bars = ax.bar(range(len(ids)), avgs, color=colors_st, edgecolor='white', width=0.5)
        if bl_val:
            ax.axhline(y=bl_val, color='#2ecc71', linestyle='--', linewidth=1, alpha=0.7)
        for bar, val in zip(bars, avgs):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.2,
                    f'{val:.2f}%', ha='center', fontsize=9)
        ax.set_xticks(range(len(ids)))
        ax.set_xticklabels(ids, fontsize=9)
        ax.set_title('Structural Improvements (SE / BiFPN / RegLoss)', fontsize=13)
        ax.set_ylabel('avg mAP (%)')
        from matplotlib.patches import Patch as Pt
        ax.legend(handles=[Pt(color=c, label=l) for l, c in grp_colors.items()], fontsize=8)
    else:
        ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Structural Improvements', fontsize=13)

    # ── Panel 3: IoU Head 迭代 ──
    ax = axes[2]
    iou_exps = groups.get('iou_head', [])
    if not iou_exps:
        iou_exps = [e for e in all_data.get('experiments', {}).values()
                    if e.get('source_type') == 'iou_head']
    if iou_exps:
        iou_exps.sort(key=lambda x: x['exp_id'])
        ids = [e['exp_id'] for e in iou_exps]
        avgs = [e.get('avg_mAP', 0) or 0 for e in iou_exps]
        bars = ax.bar(range(len(ids)), avgs, color='#f39c12', edgecolor='white', width=0.4)
        if bl_val:
            ax.axhline(y=bl_val, color='#2ecc71', linestyle='--', linewidth=1, alpha=0.7)
        for bar, val in zip(bars, avgs):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.2,
                    f'{val:.2f}%', ha='center', fontsize=10)
        ax.set_xticks(range(len(ids)))
        ax.set_xticklabels(ids, fontsize=10)
        ax.set_title('IoU Head Iterations', fontsize=13)
        ax.set_ylabel('avg mAP (%)')
    else:
        ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
        ax.set_title('IoU Head Iterations', fontsize=13)

    # ── Panel 4: IoU Grid Top-K ──
    ax = axes[3]
    grid_exps = groups.get('iou_grid', [])
    if grid_exps:
        grid_exps.sort(key=lambda x: -(x.get('avg_mAP') or -999))
        top_grid = grid_exps[:12]
        ids = [e['exp_id'] for e in top_grid]
        avgs = [e.get('avg_mAP', 0) or 0 for e in top_grid]
        colors_gr = [PALETTE['iou_grid'] if i < 3 else '#bdc3c7' for i in range(len(ids))]
        bars = ax.bar(range(len(ids)), avgs, color=colors_gr, edgecolor='white', width=0.5)
        for bar, val in zip(bars, avgs):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.15,
                    f'{val:.2f}%', ha='center', fontsize=8)
        ax.set_xticks(range(len(ids)))
        ax.set_xticklabels(ids, rotation=45, ha='right', fontsize=8)
        ax.set_title('IoU Grid Search — Top 12', fontsize=13)
        ax.set_ylabel('avg mAP (%)')
    else:
        ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
        ax.set_title('IoU Grid Search', fontsize=13)

    fig.suptitle('TriDet Experiment Groups — Comparative Dashboard', fontsize=15, fontweight='bold', y=1.01)
    plt.tight_layout()
    path = os.path.join(out_dir, 'comprehensive_groups.png')
    _save_and_close(fig, path)
    return fig


# ============================================================
# 3. 网格搜索热力图
# ============================================================

def plot_grid_heatmaps(all_data, out_dir, figsize=(14, 10)):
    """IoU 网格搜索 2×2 热力图面板."""
    from evaluate.common import classify_experiments
    groups = classify_experiments(all_data)
    grid_exps = groups.get('iou_grid', [])
    if len(grid_exps) < 4:
        return None

    valid = [e for e in grid_exps if e.get('avg_mAP') is not None]
    if not valid:
        return None

    # 提取参数值集合
    lws = sorted(set(e['params'].get('loss_weight') for e in valid if e['params'].get('loss_weight') is not None))
    layers_vals = sorted(set(e['params'].get('layers') for e in valid if e['params'].get('layers') is not None))
    per_levels = sorted(set(e['params'].get('per_level') for e in valid if e['params'].get('per_level') is not None))
    residuals = sorted(set(e['params'].get('residual') for e in valid if e['params'].get('residual') is not None))

    def _build_heatmap(x_key, x_vals, y_key, y_vals, aggregate=True):
        """构建热力图矩阵, 对其余维度取平均."""
        matrix = np.full((len(y_vals), len(x_vals)), np.nan)
        for yi, yv in enumerate(y_vals):
            for xi, xv in enumerate(x_vals):
                matches = [e['avg_mAP'] for e in valid
                          if e['params'].get(x_key) == xv
                          and e['params'].get(y_key) == yv]
                if matches:
                    matrix[yi, xi] = np.mean(matches)
        return matrix

    fig, axes = plt.subplots(2, 2, figsize=figsize)
    heatmap_configs = [
        (axes[0, 0], 'loss_weight', lws, 'layers', layers_vals,
         'loss_weight × layers', 'loss_weight', 'layers'),
        (axes[0, 1], 'loss_weight', lws, 'per_level', per_levels,
         'loss_weight × per_level', 'loss_weight', 'per_level'),
        (axes[1, 0], 'loss_weight', lws, 'residual', residuals,
         'loss_weight × residual', 'loss_weight', 'residual'),
        (axes[1, 1], 'layers', layers_vals, 'residual', residuals,
         'layers × residual', 'layers', 'residual'),
    ]

    vmin = min(e['avg_mAP'] for e in valid)
    vmax = max(e['avg_mAP'] for e in valid)

    for ax, xk, xv, yk, yv, title, xlabel, ylabel in heatmap_configs:
        matrix = _build_heatmap(xk, xv, yk, yv)
        im = ax.imshow(matrix, cmap='YlOrRd', aspect='auto', vmin=vmin, vmax=vmax,
                       origin='lower')
        ax.set_xticks(range(len(xv)))
        ax.set_xticklabels([str(v) for v in xv], fontsize=10)
        ax.set_yticks(range(len(yv)))
        ax.set_yticklabels([str(v) for v in yv], fontsize=10)
        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(title, fontsize=13)

        # 标注数值
        for yi in range(len(yv)):
            for xi in range(len(xv)):
                val = matrix[yi, xi]
                if not np.isnan(val):
                    text_color = 'white' if val > vmin + (vmax - vmin) * 0.6 else 'black'
                    ax.text(xi, yi, f'{val:.1f}', ha='center', va='center',
                           fontsize=9, color=text_color, fontweight='bold')

    # 统一 colorbar
    cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.7])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label('avg mAP (%)', fontsize=12)

    fig.suptitle('IoU Head Hyperparameter Grid Search — Heatmaps', fontsize=15, fontweight='bold')
    plt.subplots_adjust(right=0.90)
    path = os.path.join(out_dir, 'comprehensive_grid_heatmap.png')
    _save_and_close(fig, path)
    return fig


# ============================================================
# 4. 多组 tIoU-mAP 曲线
# ============================================================

def plot_tiou_curves(all_data, out_dir, figsize=(10, 6)):
    """从 CSV 中 mAP@0.3/0.5/0.7 构建 tIoU 曲线。每组的代表实验对比."""
    from evaluate.common import classify_experiments
    groups = classify_experiments(all_data)
    tious = [0.3, 0.5, 0.7]

    # 选择代表实验
    representatives = []

    # 基线
    bl = all_data['experiments'].get('baseline') or all_data['experiments'].get('Baseline')
    if bl and all(bl.get(f'mAP_0{t}') is not None for t in [3, 5, 7]):
        representatives.append(('Baseline', '#2ecc71', 'D-', bl))

    # Ablation 代表
    for eid, color, marker in [('A1', '#3498db', 'o-'), ('A2', '#e74c3c', 's-')]:
        exp = all_data['experiments'].get(eid)
        if exp and all(exp.get(f'mAP_0{t}') is not None for t in [3, 5, 7]):
            representatives.append((eid, color, marker, exp))

    # Structural 最优
    struct = groups.get('structural', [])
    struct_valid = sorted(
        [e for e in struct if all(e.get(f'mAP_0{t}') is not None for t in [3, 5, 7])],
        key=lambda x: -(x.get('avg_mAP') or -999))
    for e in struct_valid[:2]:
        representatives.append((e['exp_id'], '#e67e22', '^--', e))

    # IoU Grid 最优
    grid = groups.get('iou_grid', [])
    grid_valid = sorted(
        [e for e in grid if all(e.get(f'mAP_0{t}') is not None for t in [3, 5, 7])],
        key=lambda x: -(x.get('avg_mAP') or -999))
    for e in grid_valid[:2]:
        representatives.append((e['exp_id'], '#9b59b6', 'v--', e))

    if len(representatives) < 2:
        return None

    fig, ax = plt.subplots(figsize=figsize)

    for label, color, marker, exp in representatives:
        maps = [exp.get(f'mAP_03'), exp.get(f'mAP_05'), exp.get(f'mAP_07')]
        if all(m is not None for m in maps):
            ax.plot(tious, maps, marker, color=color, linewidth=2,
                   markersize=8, label=f'{label} ({exp.get("avg_mAP", 0):.1f}%)')

    ax.set_xlabel('tIoU Threshold', fontsize=13)
    ax.set_ylabel('mAP (%)', fontsize=13)
    ax.set_title('mAP vs tIoU — Multi-Group Comparison', fontsize=14)
    ax.legend(fontsize=9, loc='lower left')
    ax.grid(True, alpha=0.3)
    ax.set_xticks(tious)

    plt.tight_layout()
    path = os.path.join(out_dir, 'comprehensive_tiou_curves.png')
    _save_and_close(fig, path)
    return fig


# ============================================================
# 5. 训练效率散点图
# ============================================================

def plot_efficiency_scatter(all_data, out_dir, figsize=(10, 7)):
    """x=训练时间, y=avg_mAP, 颜色=分组, 气泡大小自适应."""
    from evaluate.common import get_valid_experiments
    exps = [e for e in get_valid_experiments(all_data) if e.get('train_time')]

    # 解析训练时间
    def _parse_time(tt):
        try:
            return float(str(tt).strip())
        except (ValueError, TypeError):
            return None

    pts = []
    for e in exps:
        t = _parse_time(e.get('train_time', ''))
        if t is not None and t > 0:
            pts.append((e, t))

    if len(pts) < 3:
        return None

    fig, ax = plt.subplots(figsize=figsize)

    for e, t in pts:
        color = _get_group_color(e)
        avg = e.get('avg_mAP', 0) or 0
        ax.scatter(t, avg, c=color, s=80, alpha=0.7, edgecolors='white', linewidth=0.5,
                  label=e.get('source_type', 'other'))

    # 标注 Top-3 和最慢的
    pts_sorted_avg = sorted(pts, key=lambda x: -(x[0].get('avg_mAP') or 0))
    for e, t in pts_sorted_avg[:3]:
        ax.annotate(e['exp_id'], (t, e['avg_mAP']),
                   textcoords="offset points", xytext=(5, 5), fontsize=8)

    ax.set_xlabel('Training Time (min)', fontsize=13)
    ax.set_ylabel('Average mAP (%)', fontsize=13)
    ax.set_title('Training Efficiency: mAP vs Time', fontsize=14)

    # 去重图例
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), fontsize=9)

    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(out_dir, 'comprehensive_efficiency.png')
    _save_and_close(fig, path)
    return fig


# ============================================================
# 6. 雷达图
# ============================================================

def plot_radar(all_data, out_dir, exp_ids=None, figsize=(10, 8)):
    """多实验在 mAP@0.3/0.5/0.7/avg 四维度雷达图对比."""
    from evaluate.common import get_valid_experiments

    if exp_ids is None:
        # 自动选择: baseline + 各组最优
        from evaluate.common import classify_experiments
        exp_ids = []
        bl = all_data['experiments'].get('baseline') or all_data['experiments'].get('Baseline')
        if bl:
            exp_ids.append(bl['exp_id'])
        groups = classify_experiments(all_data)
        for gname in ['ablation', 'structural', 'iou_grid']:
            gexps = sorted(groups.get(gname, []), key=lambda x: -(x.get('avg_mAP') or -999))
            if gexps:
                exp_ids.append(gexps[0]['exp_id'])

    if len(exp_ids) < 2:
        return None

    dimensions = ['mAP@0.3', 'mAP@0.5', 'mAP@0.7', 'avg mAP']
    dim_keys = ['mAP_03', 'mAP_05', 'mAP_07', 'avg_mAP']
    N = len(dimensions)

    # 提取数据
    selected = []
    for eid in exp_ids:
        exp = all_data['experiments'].get(eid)
        if exp is None:
            continue
        vals = [exp.get(k) for k in dim_keys]
        if all(v is not None for v in vals):
            selected.append((eid, vals))

    if len(selected) < 2:
        return None

    # 归一化到 [0, 1] (相对于观测范围)
    all_vals = [v for _, vals in selected for v in vals]
    vmin, vmax = min(all_vals), max(all_vals)

    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]  # 闭合

    fig, ax = plt.subplots(figsize=figsize, subplot_kw=dict(polar=True))
    colors = plt.cm.tab10(np.linspace(0, 1, len(selected)))

    for (eid, vals), color in zip(selected, colors):
        norm_vals = [(v - vmin) / (vmax - vmin + 1e-8) for v in vals]
        norm_vals += norm_vals[:1]
        ax.fill(angles, norm_vals, alpha=0.1, color=color)
        ax.plot(angles, norm_vals, 'o-', linewidth=2, color=color, label=f'{eid} (avg={vals[3]:.1f}%)')

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(dimensions, fontsize=12)
    ax.set_ylim(0, 1.1)
    ax.set_title('Multi-Dimensional mAP Comparison', fontsize=14, pad=20)
    ax.legend(fontsize=9, loc='lower right', bbox_to_anchor=(1.3, 0))

    plt.tight_layout()
    path = os.path.join(out_dir, 'comprehensive_radar.png')
    _save_and_close(fig, path)
    return fig


# ============================================================
# 7. 参数扫描折线图 (数据驱动)
# ============================================================

def plot_param_scans(all_data, out_dir, figsize=(14, 10)):
    """自动检测所有包含参数扫的实验组, 画折线图."""
    from evaluate.common import classify_experiments
    groups = classify_experiments(all_data)

    # 收集可扫描的组
    scannable = []

    # Ablation 中的 A3 (n_sgp_win_size)
    a3_exps = [e for e in groups.get('ablation', [])
               if e['exp_id'].startswith('A3_w') and e.get('avg_mAP') is not None]
    if a3_exps:
        pts = []
        for e in a3_exps:
            try:
                w = int(e['exp_id'].split('_w')[1])
                pts.append((w, e['avg_mAP'], e['exp_id']))
            except (ValueError, IndexError):
                continue
        if len(pts) >= 2:
            pts.sort()
            scannable.append(('A3: SGP Window Size', 'n_sgp_win_size', pts))

    # Ablation 中的 A4 (k)
    a4_exps = [e for e in groups.get('ablation', [])
               if e['exp_id'].startswith('A4_k') and e.get('avg_mAP') is not None]
    if a4_exps:
        pts = []
        for e in a4_exps:
            try:
                k = float(e['exp_id'].split('_k')[1])
                pts.append((k, e['avg_mAP'], e['exp_id']))
            except (ValueError, IndexError):
                continue
        if len(pts) >= 2:
            pts.sort()
            scannable.append(('A4: SGP k Factor', 'k', pts))

    # Ablation 中的 A7 (center_sample_radius)
    a7_exps = [e for e in groups.get('ablation', [])
               if e['exp_id'].startswith('A7_r') and e.get('avg_mAP') is not None]
    if a7_exps:
        pts = []
        for e in a7_exps:
            try:
                r = float(e['exp_id'].split('_r')[1])
                pts.append((r, e['avg_mAP'], e['exp_id']))
            except (ValueError, IndexError):
                continue
        if len(pts) >= 2:
            pts.sort()
            scannable.append(('A7: Center Sample Radius', 'center_sample_radius', pts))

    # Ablation 中的 A8 (loss_weight)
    a8_exps = [e for e in groups.get('ablation', [])
               if e['exp_id'].startswith('A8_lw') and e.get('avg_mAP') is not None]
    if a8_exps:
        pts = []
        for e in a8_exps:
            try:
                lw = float(e['exp_id'].split('_lw')[1])
                pts.append((lw, e['avg_mAP'], e['exp_id']))
            except (ValueError, IndexError):
                continue
        if len(pts) >= 2:
            pts.sort()
            scannable.append(('A8: Classification Loss Weight', 'loss_weight', pts))

    # IoU Grid: loss_weight 扫描 (固定 layers=2, 其余平均)
    grid_exps = groups.get('iou_grid', [])
    if grid_exps:
        grid_valid = [e for e in grid_exps if e.get('avg_mAP') is not None]
        lws = sorted(set(e['params'].get('loss_weight') for e in grid_valid
                         if e['params'].get('loss_weight') is not None))
        if len(lws) >= 2:
            pts = []
            for lw in lws:
                matches = [e['avg_mAP'] for e in grid_valid
                          if e['params'].get('loss_weight') == lw]
                if matches:
                    pts.append((lw, np.mean(matches), f'loss_weight={lw}'))
            pts.sort()
            scannable.append(('IoU Grid: loss_weight (averaged)', 'iou_loss_weight', pts))

    if not scannable:
        return None

    n = len(scannable)
    cols = min(2, n)
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    if n == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    for idx, (title, xlabel, pts) in enumerate(scannable):
        ax = axes[idx]
        xv = [p[0] for p in pts]
        yv = [p[1] for p in pts]
        labels = [p[2] for p in pts]

        ax.plot(xv, yv, 'o-', color='#3498db', linewidth=2, markersize=8)
        for x, y, lbl in zip(xv, yv, labels):
            ax.annotate(f'{y:.2f}%', (x, y), textcoords="offset points",
                       xytext=(0, 10), ha='center', fontsize=9)

        # 标注最优
        best_idx = np.argmax(yv)
        ax.axvline(x=xv[best_idx], color='#e74c3c', linestyle='--', alpha=0.4)
        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel('avg mAP (%)', fontsize=12)
        ax.set_title(title, fontsize=13)
        ax.grid(True, alpha=0.3)

    # 隐藏多余子图
    for idx in range(n, len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle('Parameter Sensitivity Analysis', fontsize=15, fontweight='bold')
    plt.tight_layout()
    path = os.path.join(out_dir, 'comprehensive_param_scans.png')
    _save_and_close(fig, path)
    return fig


# ============================================================
# 8. 综合报告生成
# ============================================================

def generate_markdown_report(all_data, plot_paths, out_path):
    """生成 Markdown 综合报告."""
    from evaluate.common import get_valid_experiments, classify_experiments
    from datetime import datetime

    exps = get_valid_experiments(all_data)
    groups = classify_experiments(all_data)
    bl = all_data['experiments'].get('baseline') or all_data['experiments'].get('Baseline')
    bl_val = bl.get('avg_mAP') if bl else None

    lines = []
    lines.append("# TriDet 综合实验分析报告")
    lines.append("")
    lines.append(f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    lines.append(f"**数据集**: THUMOS14  |  **总实验数**: {len(exps)}  |  **CSV 来源**: {len(all_data['csv_types'])}")
    lines.append("")

    if bl_val:
        lines.append(f"**基线**: {bl.get('name', 'TriDet')} — avg mAP = {bl_val:.2f}%")
        lines.append("")

    # ── 完整排名 Top-30 ──
    lines.append("---")
    lines.append("")
    lines.append("## 完整排名（Top 30）")
    lines.append("")
    lines.append("| 排名 | 实验 | 分组 | 来源 | mAP@0.3 | mAP@0.5 | mAP@0.7 | avg mAP | vs 基线 |")
    lines.append("|------|------|------|------|---------|---------|---------|---------|---------|")

    for rank, e in enumerate(exps[:30], 1):
        avg = e.get('avg_mAP')
        vs = ''
        if avg is not None and bl_val is not None:
            vs = f'{avg - bl_val:+.2f}%'
        lines.append(
            f"| {rank} | {e['exp_id']} | {e.get('group', '')} | {e.get('source_type', '')} | "
            f"{_fmt_val(e.get('mAP_03'))} | {_fmt_val(e.get('mAP_05'))} | "
            f"{_fmt_val(e.get('mAP_07'))} | {_fmt_val(avg)} | {vs} |")

    lines.append("")

    # ── 分组汇总 ──
    lines.append("---")
    lines.append("")
    lines.append("## 分组汇总")
    lines.append("")
    lines.append("| 分组 | 实验数 | 最优 | avg mAP | 最差 | avg mAP |")
    lines.append("|------|--------|------|---------|------|---------|")

    for gname in ['ablation', 'structural', 'iou_head', 'iou_grid']:
        gexps = [e for e in groups.get(gname, []) if e.get('avg_mAP') is not None]
        if not gexps:
            continue
        best = max(gexps, key=lambda x: x.get('avg_mAP') or -999)
        worst = min(gexps, key=lambda x: x.get('avg_mAP') or 999)
        lines.append(
            f"| {gname} | {len(gexps)} | {best['exp_id']} | {best.get('avg_mAP', 0):.2f}% | "
            f"{worst['exp_id']} | {worst.get('avg_mAP', 0):.2f}% |")

    lines.append("")

    # ── 超过基线的实验 ──
    if bl_val:
        above = [e for e in exps if e.get('avg_mAP') and e['avg_mAP'] > bl_val]
        lines.append("---")
        lines.append("")
        lines.append(f"## 超过基线的实验 ({len(above)} 项)")
        lines.append("")
        if above:
            lines.append("| 实验 | avg mAP | Δ vs 基线 | 分组 |")
            lines.append("|------|---------|-----------|------|")
            for e in sorted(above, key=lambda x: -(x.get('avg_mAP') or 0)):
                lines.append(
                    f"| {e['exp_id']} | {e['avg_mAP']:.2f}% | "
                    f"{e['avg_mAP'] - bl_val:+.2f}% | {e.get('group', '')} |")
        else:
            lines.append("*无实验超过基线。*")
        lines.append("")

    # ── IoU Grid 最优配置 ──
    grid_exps = groups.get('iou_grid', [])
    if grid_exps:
        grid_valid = sorted(
            [e for e in grid_exps if e.get('avg_mAP') is not None],
            key=lambda x: -(x.get('avg_mAP') or -999))
        if grid_valid:
            lines.append("---")
            lines.append("")
            lines.append("## IoU Head 网格搜索 — 最优配置")
            lines.append("")
            best = grid_valid[0]
            params = best.get('params', {})
            lines.append(f"- **最优实验**: {best['exp_id']} — avg mAP = {best['avg_mAP']:.2f}%")
            lines.append(f"- **参数**: loss_weight={params.get('loss_weight')}, "
                        f"per_level={params.get('per_level')}, "
                        f"residual={params.get('residual')}, "
                        f"layers={params.get('layers')}")
            lines.append("")
            lines.append("**Top 5 配置**:")
            lines.append("")
            lines.append("| 实验 | loss_weight | per_level | residual | layers | avg mAP |")
            lines.append("|------|-------------|-----------|----------|--------|---------|")
            for e in grid_valid[:5]:
                p = e.get('params', {})
                lines.append(
                    f"| {e['exp_id']} | {p.get('loss_weight')} | {p.get('per_level')} | "
                    f"{p.get('residual')} | {p.get('layers')} | {e['avg_mAP']:.2f}% |")
            lines.append("")

    # ── 图表索引 ──
    lines.append("---")
    lines.append("")
    lines.append("## 生成图表")
    lines.append("")
    for name, path in plot_paths.items():
        if path and os.path.exists(path):
            fname = os.path.basename(path)
            lines.append(f"- **{name}**: `{fname}`")
    lines.append("")

    # ── 结论 ──
    lines.append("---")
    lines.append("")
    lines.append("## 结论与建议")
    lines.append("")
    if bl_val:
        best_all = exps[0] if exps else None
        if best_all and best_all.get('avg_mAP'):
            lines.append(f"- **全局最优**: {best_all['exp_id']} — avg mAP = {best_all['avg_mAP']:.2f}% "
                        f"({best_all['avg_mAP'] - bl_val:+.2f}% vs 基线)")
        if above:
            lines.append(f"- **正向改进**: {len(above)} 个实验超过基线，最佳提升 {above[0]['avg_mAP'] - bl_val:+.2f}%")
        else:
            lines.append("- **无正向改进**: 所有实验均未超过基线性能")

    lines.append("")

    report = '\n'.join(lines)
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(report)
    return report


def generate_html_report(all_data, plot_paths, out_path):
    """生成单文件 HTML 仪表板 (图表内嵌 base64)."""
    import base64
    from datetime import datetime

    # 内嵌图片
    embedded = {}
    for name, path in plot_paths.items():
        if path and os.path.exists(path):
            with open(path, 'rb') as f:
                b64 = base64.b64encode(f.read()).decode('utf-8')
                embedded[name] = f'data:image/png;base64,{b64}'

    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>TriDet 综合实验分析</title>
<style>
  body {{ font-family: 'Segoe UI', 'Microsoft YaHei', sans-serif; max-width: 1200px; margin: 0 auto; padding: 20px; background: #f5f6fa; }}
  h1 {{ color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px; }}
  h2 {{ color: #34495e; margin-top: 30px; }}
  .meta {{ color: #7f8c8d; font-size: 14px; }}
  .chart {{ background: white; border-radius: 8px; padding: 15px; margin: 20px 0; box-shadow: 0 2px 8px rgba(0,0,0,0.1); }}
  .chart h3 {{ margin-top: 0; color: #2c3e50; }}
  .chart img {{ width: 100%; border-radius: 4px; }}
  .grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }}
  @media (max-width: 800px) {{ .grid {{ grid-template-columns: 1fr; }} }}
</style>
</head>
<body>
<h1>TriDet 综合实验分析仪表板</h1>
<p class="meta">生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M')}  |  数据集: THUMOS14  |  实验数: {len(all_data['experiments'])}</p>
"""

    chart_order = [
        ('总览排名', 'comprehensive_overview.png'),
        ('分组对比面板', 'comprehensive_groups.png'),
        ('网格搜索热力图', 'comprehensive_grid_heatmap.png'),
        ('tIoU-mAP 曲线', 'comprehensive_tiou_curves.png'),
        ('参数扫描', 'comprehensive_param_scans.png'),
        ('训练效率', 'comprehensive_efficiency.png'),
        ('雷达图对比', 'comprehensive_radar.png'),
    ]

    for title, fname in chart_order:
        if title in embedded:
            html += f"""
<div class="chart">
  <h3>{title}</h3>
  <img src="{embedded[title]}" alt="{title}">
</div>"""

    html += """
</body>
</html>"""

    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(html)
    return html


# ============================================================
# 9. 运行所有可视化
# ============================================================

def run_all_visualizations(all_data, out_dir):
    """运行所有可视化, 返回 {chart_name: saved_path}."""
    os.makedirs(out_dir, exist_ok=True)
    paths = {}

    funcs = [
        ('总览排名', plot_overview),
        ('分组对比面板', plot_group_comparison),
        ('网格搜索热力图', plot_grid_heatmaps),
        ('tIoU-mAP 曲线', plot_tiou_curves),
        ('参数扫描', plot_param_scans),
        ('训练效率', plot_efficiency_scatter),
        ('雷达图对比', plot_radar),
    ]

    for name, func in funcs:
        try:
            fig = func(all_data, out_dir)
            if fig is not None:
                # 用对应的文件名
                fname_map = {
                    '总览排名': 'comprehensive_overview.png',
                    '分组对比面板': 'comprehensive_groups.png',
                    '网格搜索热力图': 'comprehensive_grid_heatmap.png',
                    'tIoU-mAP 曲线': 'comprehensive_tiou_curves.png',
                    '参数扫描': 'comprehensive_param_scans.png',
                    '训练效率': 'comprehensive_efficiency.png',
                    '雷达图对比': 'comprehensive_radar.png',
                }
                fname = fname_map.get(name, f'comprehensive_{name}.png')
                paths[name] = os.path.join(out_dir, fname)
            else:
                print(f"  [SKIP] {name}: 数据不足")
        except Exception as e:
            print(f"  [FAIL] {name}: {e}")

    return paths

"""
TriDet 全局改进实验 — 汇总脚本

读取三个分组实验 CSV + 基线结果, 合并为全局排名表与 Markdown 报告。

用法:
    python evaluate/aggregate_global_results.py

输入:
    evaluate/results/structural_improvements.csv
    evaluate/results/iou_head_iterations.csv
    evaluate/results/iou_grid_search.csv

输出:
    evaluate/results/global_improvements.csv
    evaluate/results/global_improvements_report.md
"""

import os, csv, sys
from pathlib import Path
from datetime import datetime

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

RESULT_DIR = REPO / 'evaluate' / 'results'

# 基线数据 (来自 overall_exp_config.md)
BASELINE = {
    'id': 'Baseline',
    'group': '基线',
    'name': 'TriDet (SGP+Trident+DIoU)',
    'avg_mAP': 68.59,
    'status': 'BASELINE',
}

# 输入 CSV 文件
INPUT_CSVS = [
    RESULT_DIR / 'structural_improvements.csv',
    RESULT_DIR / 'iou_head_iterations.csv',
    RESULT_DIR / 'iou_grid_search.csv',
]

# 输出
OUTPUT_CSV  = RESULT_DIR / 'global_improvements.csv'
OUTPUT_MD   = RESULT_DIR / 'global_improvements_report.md'


def read_csv(path):
    """读取 CSV 返回 list[dict]."""
    if not path.exists():
        print(f"  [WARN] 文件不存在, 跳过: {path}")
        return []
    with open(path, 'r', encoding='utf-8') as f:
        return list(csv.DictReader(f))


def parse_float(v):
    """安全解析浮点数."""
    if v is None or v == '':
        return None
    try:
        return float(v)
    except (ValueError, TypeError):
        return None


def main():
    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    # ── 1. 读取所有 CSV ──
    all_rows = []
    print("读取输入:")
    for csv_path in INPUT_CSVS:
        rows = read_csv(csv_path)
        print(f"  {csv_path.name}: {len(rows)} 行")
        for r in rows:
            # 标准化字段名
            exp_id = r.get('实验编号', r.get('实验编号', ''))
            if not exp_id:
                continue
            all_rows.append({
                'id': exp_id,
                'group': r.get('分组', r.get('group', '')),
                'name': r.get('配置变更', r.get('name', '')),
                'mAP@0.3': parse_float(r.get('mAP@0.3')),
                'mAP@0.5': parse_float(r.get('mAP@0.5')),
                'mAP@0.7': parse_float(r.get('mAP@0.7')),
                'avg_mAP': parse_float(r.get('avg_mAP')),
                'train_time': r.get('训练时间(min)', r.get('train_time', '')),
                'status': r.get('状态', r.get('status', '')),
                'timestamp': r.get('时间戳', r.get('timestamp', '')),
            })

    # ── 2. 加入基线 ──
    all_rows.append({
        'id': BASELINE['id'],
        'group': BASELINE['group'],
        'name': BASELINE['name'],
        'avg_mAP': BASELINE['avg_mAP'],
        'status': BASELINE['status'],
    })
    print(f"  基线: 1 行 (avg_mAP={BASELINE['avg_mAP']}%)")

    # ── 3. 按 avg_mAP 降序排列 ──
    def sort_key(row):
        v = row.get('avg_mAP')
        return v if v is not None else -999.0

    all_rows.sort(key=sort_key, reverse=True)

    # ── 4. 写入全局 CSV ──
    def _fmt(v):
        if v is None:
            return ''
        if isinstance(v, float):
            return f'{v:.2f}'
        return str(v)

    with open(OUTPUT_CSV, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['排名', '实验编号', '分组', '配置变更',
                    'mAP@0.3', 'mAP@0.5', 'mAP@0.7', 'avg_mAP',
                    'vs基线', '训练时间(min)', '状态', '时间戳'])
        for rank, r in enumerate(all_rows, 1):
            avg = r.get('avg_mAP')
            vs_baseline = ''
            if avg is not None:
                delta = avg - BASELINE['avg_mAP']
                vs_baseline = f'{delta:+.2f}'
            w.writerow([
                rank, r['id'], r.get('group', ''), r.get('name', ''),
                _fmt(r.get('mAP@0.3')), _fmt(r.get('mAP@0.5')),
                _fmt(r.get('mAP@0.7')), _fmt(avg),
                vs_baseline,
                r.get('train_time', ''), r.get('status', ''),
                r.get('timestamp', ''),
            ])

    print(f"\n  全局 CSV: {OUTPUT_CSV}  ({len(all_rows)} 行)")

    # ── 5. 生成 Markdown 报告 ──
    lines = []
    lines.append("# TriDet 全局改进实验 — 汇总报告\n")
    lines.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
    lines.append(f"基线: {BASELINE['name']}  (avg mAP = {BASELINE['avg_mAP']}%)\n")
    lines.append(f"总实验数: {len(all_rows)}（含基线）\n")

    # ── 5a. 分组汇总 ──
    lines.append("## 分组汇总\n")
    groups = {}
    for r in all_rows:
        g = r.get('group', '其他')
        groups.setdefault(g, []).append(r)

    group_order = ['基线', 'SE', 'BiFPN', 'RegLoss', 'IoU Head', 'IoU Grid', '其他']
    for g in group_order:
        if g in groups:
            exps = groups.pop(g)
            best = max(exps, key=lambda x: x.get('avg_mAP') or -999)
            worst = min(exps, key=lambda x: x.get('avg_mAP') or -999)
            avgs = [e.get('avg_mAP') for e in exps if e.get('avg_mAP') is not None]
            avg_val = sum(avgs) / len(avgs) if avgs else None
            b = best.get('avg_mAP')
            w = worst.get('avg_mAP')
            lines.append(f"| **{g}** | {len(exps)} | "
                         f"{b:.2f}% | {w:.2f}% | "
                         f"{avg_val:.2f}% |" if avg_val else f"| **{g}** | {len(exps)} | - | - | - |")
    # remaining groups
    for g, exps in groups.items():
        best = max(exps, key=lambda x: x.get('avg_mAP') or -999)
        worst = min(exps, key=lambda x: x.get('avg_mAP') or -999)
        avgs = [e.get('avg_mAP') for e in exps if e.get('avg_mAP') is not None]
        avg_val = sum(avgs) / len(avgs) if avgs else None
        b = best.get('avg_mAP')
        w = worst.get('avg_mAP')
        if avg_val is not None:
            lines.append(f"| **{g}** | {len(exps)} | {b:.2f}% | {w:.2f}% | {avg_val:.2f}% |")
        else:
            lines.append(f"| **{g}** | {len(exps)} | - | - | - |")

    # ── 5b. 完整排名表 ──
    lines.append("\n## 完整排名（按 avg mAP 降序）\n")
    lines.append("| 排名 | 实验 | 分组 | avg mAP | vs 基线 | 状态 |")
    lines.append("|------|------|------|---------|---------|------|")

    for rank, r in enumerate(all_rows, 1):
        avg = r.get('avg_mAP')
        avg_str = f'{avg:.2f}%' if avg is not None else '-'
        vs = ''
        if avg is not None:
            delta = avg - BASELINE['avg_mAP']
            vs = f'{delta:+.2f}%'
        status = r.get('status', '')
        name = r.get('name', r['id'])
        # truncate long names
        if len(name) > 60:
            name = name[:57] + '...'
        lines.append(f"| {rank} | {name} | {r.get('group', '')} | {avg_str} | {vs} | {status} |")

    # ── 5c. 结论 ──
    lines.append("\n## 结论\n")
    completed = [r for r in all_rows if r.get('avg_mAP') is not None]
    if completed:
        best = completed[0]
        lines.append(f"- **最优**: {best.get('name', best['id'])}  "
                     f"(avg mAP = {best['avg_mAP']:.2f}%, "
                     f"vs 基线 {best['avg_mAP'] - BASELINE['avg_mAP']:+.2f}%)")
        above_baseline = [r for r in completed
                         if r.get('avg_mAP') is not None
                         and r.get('avg_mAP') > BASELINE['avg_mAP']]
        lines.append(f"- 超过基线的实验: {len(above_baseline)} / {len(completed)}")
    else:
        lines.append("- 暂无有效实验结果。请先运行实验脚本。")

    lines.append(f"\n- 基线: {BASELINE['name']} = {BASELINE['avg_mAP']}% avg mAP\n")

    report = '\n'.join(lines)
    with open(OUTPUT_MD, 'w', encoding='utf-8') as f:
        f.write(report)

    print(f"  报告: {OUTPUT_MD}")
    print("  完成!")


if __name__ == '__main__':
    main()

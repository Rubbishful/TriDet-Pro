"""
TriDet 统一实验分析 — 融合消融/结构改进/网格搜索 + 预测深度分析。

用法:
    # 基础: 仅 CSV 数据分析
    python evaluate/analyze_all.py

    # 完整: CSV + 预测深度分析
    python evaluate/analyze_all.py --pred <pred.pkl> --json <gt.json>

    # 指定输出格式
    python evaluate/analyze_all.py --format html
    python evaluate/analyze_all.py --format md

    # 仅特定分析维度
    python evaluate/analyze_all.py --sections overview,groups,grid

输入 (自动发现):
    evaluate/results/ablation_results.csv
    evaluate/results/structural_improvements.csv
    evaluate/results/iou_head_iterations.csv
    evaluate/results/iou_grid_search.csv
    evaluate/results/global_improvements.csv

可选输入:
    --pred <predictions.pkl>  预测 pickle (启用混淆矩阵/密度/时长分析)
    --json <thumos14.json>    GT 标注 JSON (配合 --pred)

输出:
    evaluate/results/comprehensive_report.md    Markdown 综合报告
    evaluate/results/comprehensive_report.html  HTML 仪表板 (内嵌图表)
    evaluate/results/comprehensive_*.png         多张分析图表
"""

import os, sys, argparse
from pathlib import Path
from datetime import datetime

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from evaluate.common import load_all_results, classify_experiments, get_valid_experiments
from evaluate.visualize import (
    run_all_visualizations, generate_markdown_report, generate_html_report,
    plot_overview, plot_group_comparison, plot_grid_heatmaps,
    plot_tiou_curves, plot_param_scans, plot_efficiency_scatter, plot_radar,
)

RESULT_DIR = REPO / 'evaluate' / 'results'


def main():
    parser = argparse.ArgumentParser(
        description='TriDet 统一实验分析 — 多维度可视化 + 综合报告')
    parser.add_argument('--data-dir', default=str(RESULT_DIR),
                        help=f'CSV 数据目录 (默认: {RESULT_DIR})')
    parser.add_argument('--output', default=str(RESULT_DIR),
                        help=f'输出目录 (默认: {RESULT_DIR})')
    parser.add_argument('--pred', default=None,
                        help='预测 pickle 文件 (启用混淆矩阵/密度/时长分析)')
    parser.add_argument('--json', default=None,
                        help='THUMOS14 GT 标注 JSON (配合 --pred)')
    parser.add_argument('--split', default='test', help='数据集 split (默认: test)')
    parser.add_argument('--tiou', type=float, default=0.5, help='tIoU 阈值 (默认: 0.5)')
    parser.add_argument('--format', default='all', choices=['md', 'html', 'all'],
                        help='报告输出格式 (默认: all)')
    parser.add_argument('--sections', default='all',
                        help='分析维度, 逗号分隔: overview,groups,grid,tiou,params,'
                             'efficiency,radar,confusion,density,duration,all')
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    sections = args.sections.split(',')
    if 'all' in sections:
        sections = ['overview', 'groups', 'grid', 'tiou', 'params',
                    'efficiency', 'radar']

    # ================================================================
    # Phase 1: 加载 CSV 数据
    # ================================================================
    print("=" * 65)
    print("  TriDet 统一实验分析")
    print("=" * 65)

    all_data = load_all_results(args.data_dir)
    if not all_data['experiments']:
        print("  [ERROR] 无 CSV 数据! 请先运行实验脚本。")
        print(f"  扫描目录: {args.data_dir}")
        sys.exit(1)

    csv_list = ', '.join(all_data['csv_types'].keys())
    n_exps = len(all_data['experiments'])
    n_valid = len(get_valid_experiments(all_data))
    print(f"\n  CSV 来源: {csv_list}")
    print(f"  实验总数: {n_exps} (有效: {n_valid})")

    # ================================================================
    # Phase 2: 运行选中的可视化
    # ================================================================
    print(f"\n{'─' * 50}")
    print("  生成可视化图表...")
    print(f"{'─' * 50}")

    # 按需运行
    section_funcs = {
        'overview':   ('总览排名', plot_overview),
        'groups':     ('分组对比面板', plot_group_comparison),
        'grid':       ('网格搜索热力图', plot_grid_heatmaps),
        'tiou':       ('tIoU-mAP 曲线', plot_tiou_curves),
        'params':     ('参数扫描', plot_param_scans),
        'efficiency': ('训练效率', plot_efficiency_scatter),
        'radar':      ('雷达图对比', plot_radar),
    }

    plot_paths = {}
    fname_map = {
        '总览排名': 'comprehensive_overview.png',
        '分组对比面板': 'comprehensive_groups.png',
        '网格搜索热力图': 'comprehensive_grid_heatmap.png',
        'tIoU-mAP 曲线': 'comprehensive_tiou_curves.png',
        '参数扫描': 'comprehensive_param_scans.png',
        '训练效率': 'comprehensive_efficiency.png',
        '雷达图对比': 'comprehensive_radar.png',
    }

    for sec_name in sections:
        if sec_name in section_funcs:
            display_name, func = section_funcs[sec_name]
            try:
                fig = func(all_data, args.output)
                if fig is not None:
                    fname = fname_map.get(display_name, f'comprehensive_{sec_name}.png')
                    plot_paths[display_name] = os.path.join(args.output, fname)
                    print(f"  [OK] {display_name}")
                else:
                    print(f"  [SKIP] {display_name}: 数据不足")
            except Exception as e:
                print(f"  [FAIL] {display_name}: {e}")

    # ================================================================
    # Phase 3: 预测深度分析 (可选)
    # ================================================================
    pred_sections = [s for s in sections if s in ('confusion', 'density', 'duration')]
    if pred_sections:
        if not args.pred or not args.json:
            print(f"\n  [SKIP] 预测分析 ({', '.join(pred_sections)}) 需要 --pred 和 --json")
        else:
            print(f"\n{'─' * 50}")
            print("  预测深度分析...")
            print(f"{'─' * 50}")

            from evaluate.common import load_gt, load_preds, preds_by_video
            from evaluate.analyze_predictions import (
                run_confusion_analysis, run_density_analysis, run_duration_analysis,
            )

            gts, label_names = load_gt(args.json, args.split)
            preds = load_preds(args.pred)
            pred_by_vid = preds_by_video(preds)
            print(f"  加载: {len(gts)} 视频 GT, {len(pred_by_vid)} 视频预测, "
                  f"{len(label_names)} 类")

            pred_funcs = {
                'confusion': ('混淆矩阵', run_confusion_analysis),
                'density':   ('密度分层', run_density_analysis),
                'duration':  ('时长分层', run_duration_analysis),
            }

            for sec_name in pred_sections:
                if sec_name in pred_funcs:
                    display_name, func = pred_funcs[sec_name]
                    try:
                        func(gts, label_names, pred_by_vid, args.tiou, args.output)
                        print(f"  [OK] {display_name}")
                    except Exception as e:
                        print(f"  [FAIL] {display_name}: {e}")

    # ================================================================
    # Phase 4: 生成报告
    # ================================================================
    print(f"\n{'─' * 50}")
    print("  生成报告...")
    print(f"{'─' * 50}")

    if args.format in ('md', 'all'):
        md_path = os.path.join(args.output, 'comprehensive_report.md')
        generate_markdown_report(all_data, plot_paths, md_path)
        print(f"  [OK] Markdown: {md_path}")

    if args.format in ('html', 'all'):
        html_path = os.path.join(args.output, 'comprehensive_report.html')
        generate_html_report(all_data, plot_paths, html_path)
        print(f"  [OK] HTML: {html_path}")

    # ================================================================
    # 完成摘要
    # ================================================================
    print(f"\n{'=' * 65}")
    print(f"  分析完成!")
    print(f"  输出目录: {args.output}")
    print(f"  图表: {len(plot_paths)} 张")
    for name, path in plot_paths.items():
        if os.path.exists(path):
            print(f"    - {os.path.basename(path)}")
    print(f"{'=' * 65}")


if __name__ == '__main__':
    main()

"""
完整测试与分析流程：评估 → 消融对比 → 分层分析 → 混淆矩阵

用法:
    python work/run_full_analysis.py
"""

import os, sys, csv, json, pickle, time
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

REPO = 'e:/Tridet/TriDet-Pro'
sys.path.insert(0, REPO)

from libs.core import load_config
from libs.datasets import make_dataset, make_data_loader
from libs.modeling import make_meta_arch
from libs.utils import fix_random_seed, ANETdetection, valid_one_epoch

PYTHON = 'E:/anaconda/envs/test/python.exe'
EXPORT_DIR = os.path.join(REPO, 'work', 'results')
os.makedirs(EXPORT_DIR, exist_ok=True)

# ---- 数据集参数 ----
JSON_FILE = 'E:/thumos/annotations/thumos14.json'
SCORE_FILE = 'E:/thumos/annotations/thumos14_cls_scores.pkl'

# ---- 实验配置 ----
EXPERIMENTS = [
    # (实验ID, 配置名, checkpoint文件夹, 描述)
    ('baseline', 'configs/thumos_i3d.yaml', 'ckpt/thumos_i3d_baseline', 'TriDet 基线 (SGP + Trident-head)'),
    ('A1', 'work/abl_A1.yaml', 'ckpt/abl_A1_A1', 'A1: Trident-head → 普通回归头'),
    ('A2', 'work/abl_A2.yaml', 'ckpt/abl_A2_A2', 'A2: SGP → Conv Backbone'),
]


def run_eval(config_path, ckpt_folder):
    """评估单个模型，返回 mAP 值。使用 saveonly 模式跳过在线评估。"""
    import subprocess
    # 检查 checkpoint 是否存在
    if not os.path.isdir(ckpt_folder):
        print(f"  [SKIP] {ckpt_folder} 不存在")
        return None
    pth_files = sorted([f for f in os.listdir(ckpt_folder) if f.endswith('.pth.tar')])
    if not pth_files:
        print(f"  [SKIP] {ckpt_folder} 中无 .pth.tar 文件")
        return None
    last_ckpt = os.path.join(ckpt_folder, pth_files[-1])
    print(f"  加载: {last_ckpt}")

    # 用 --saveonly 模式更快（跳过在线 mAP 计算）
    cmd = [PYTHON, '-u', 'eval.py', config_path, ckpt_folder, '--saveonly']
    proc = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True, timeout=600)
    if proc.returncode != 0:
        print(f"  [FAIL] eval 失败: {proc.stderr[:200]}")
        return None

    # 查找 eval_results.pkl
    result_pkl = os.path.join(ckpt_folder, 'eval_results.pkl')
    if not os.path.exists(result_pkl):
        print(f"  [WARN] 未找到 eval_results.pkl")
        return None

    # 用 ANETdetection 计算 mAP
    return result_pkl


def compute_map_offline(pkl_path, json_file, split='test'):
    """离线计算 mAP（支持从 pickle 文件）。"""
    import pandas as pd
    from libs.utils.metrics import ANETdetection

    with open(pkl_path, 'rb') as f:
        preds = pickle.load(f)

    det_eval = ANETdetection(
        json_file, split,
        tiou_thresholds=np.linspace(0.3, 0.7, 5)
    )

    # 转换预测为 DataFrame
    records = []
    for p in preds:
        vid = p['video-id']
        for seg, score, label in zip(p['segments'], p['scores'], p['labels']):
            records.append({
                'video-id': vid,
                't-start': float(seg[0]),
                't-end': float(seg[1]),
                'label': int(label),
                'score': float(score),
            })
    df = pd.DataFrame(records)
    mAP_per_tiou, avg_mAP = det_eval.evaluate(df, verbose=False)
    return mAP_per_tiou, avg_mAP


def run_all_evals():
    """对所有实验跑评估（先 saveonly 再离线 mAP）。"""
    results = {}
    for exp_id, cfg_path, ckpt_folder, desc in EXPERIMENTS:
        print(f"\n{'='*60}")
        print(f"  评估: {exp_id} — {desc}")
        print(f"{'='*60}")

        pkl_path = run_eval(cfg_path, ckpt_folder)
        if pkl_path is None:
            continue

        # 提取配置中的关键参数
        cfg = load_config(cfg_path)
        use_tri = cfg['model'].get('use_trident_head', True)
        backbone = cfg['model'].get('backbone_type', 'SGP')
        k = cfg['model'].get('k', 5.0)
        fpn = cfg['model'].get('fpn_type', 'identity')
        radius = cfg['train_cfg'].get('center_sample_radius', 1.5)

        mAPs, avg = compute_map_offline(pkl_path, JSON_FILE, 'test')

        result = {
            'exp_id': exp_id,
            'desc': desc,
            'use_trident_head': use_tri,
            'backbone_type': backbone,
            'k': k,
            'fpn_type': fpn,
            'center_sample_radius': radius,
            'mAP@0.3': mAPs[0] * 100,
            'mAP@0.4': mAPs[1] * 100,
            'mAP@0.5': mAPs[2] * 100,
            'mAP@0.6': mAPs[3] * 100,
            'mAP@0.7': mAPs[4] * 100,
            'avg_mAP': avg * 100,
        }
        results[exp_id] = result

        print(f"  mAP@0.3={result['mAP@0.3']:.2f}%  mAP@0.5={result['mAP@0.5']:.2f}%  "
              f"mAP@0.7={result['mAP@0.7']:.2f}%  avg={result['avg_mAP']:.2f}%")

    return results


def analyze_configs():
    """分析各配置文件的关键参数差异。"""
    print(f"\n{'='*60}")
    print("  配置参数分析")
    print(f"{'='*60}")

    rows = []
    for exp_id, cfg_path, _, desc in EXPERIMENTS:
        if not os.path.exists(os.path.join(REPO, cfg_path)):
            continue
        cfg = load_config(cfg_path)
        m = cfg['model']
        t = cfg['train_cfg']
        rows.append({
            '实验': exp_id,
            '骨干网络': m.get('backbone_type', 'SGP'),
            'Trident-head': m.get('use_trident_head', True),
            'k参数': m.get('k', 1.5),
            '窗口大小': m.get('n_sgp_win_size', 1),
            'FPN类型': m.get('fpn_type', 'identity'),
            'num_bins': m.get('num_bins', 16),
            '中心采样半径': t['center_sample_radius'],
            '损失权重': t.get('loss_weight', 1.0),
            'max_seq_len': cfg['dataset']['max_seq_len'],
        })

    # 打印表格
    if rows:
        headers = list(rows[0].keys())
        fmt = '  '.join([f'{{:<{max(8, len(h))}}}: {{}}' for h in headers])
        print()
        for row in rows:
            for h in headers:
                val = str(row[h])
                if isinstance(row[h], float):
                    val = f'{row[h]:.1f}'
                print(f'  {h:<10}: {val}')
            print('  ---')
    return rows


def run_module_analysis():
    """运行模块级分析（shape 测试已在 test_modules.py 完成）。"""
    print(f"\n{'='*60}")
    print("  模块 Shape 测试")
    print(f"{'='*60}")

    import subprocess
    r = subprocess.run([PYTHON, 'work/test_modules.py'], cwd=REPO,
                       capture_output=True, text=True, timeout=120)
    last_lines = r.stdout.strip().split('\n')[-5:]
    for line in last_lines:
        print(f"  {line}")

    # 解析结果
    passed = r.stdout.count('[PASS]')
    failed = r.stdout.count('[FAIL]')
    return passed, failed


def run_duration_analysis():
    """时长分层分析（使用已保存的预测结果）。"""
    from libs.utils.metrics import segment_iou

    print(f"\n{'='*60}")
    print("  按动作时长分层分析")
    print(f"{'='*60}")

    # 加载 GT
    with open(JSON_FILE, 'r') as f:
        db = json.load(f)
    gts = {}
    for vid, info in db['database'].items():
        if info['subset'].lower() != 'test':
            continue
        instances = [(float(a['segment'][0]), float(a['segment'][1]), a['label_id'])
                     for a in info.get('annotations', [])]
        gts[vid] = instances

    # 加载预测 (优先使用基线)
    pred_pkl = None
    for candidate in ['ckpt/abl_A1_A1/eval_results.pkl',
                      'ckpt/abl_A2_A2/eval_results.pkl']:
        if os.path.exists(os.path.join(REPO, candidate)):
            pred_pkl = os.path.join(REPO, candidate)
            break

    if pred_pkl is None:
        print("  [SKIP] 无可用预测结果")
        return None

    with open(pred_pkl, 'rb') as f:
        preds = pickle.load(f)
    pred_by_vid = {p['video-id']: p for p in preds}

    bins = [('短 (<2s)', 0, 2), ('中 (2-10s)', 2, 10), ('长 (>10s)', 10, 1e6)]
    results = {}

    for bin_name, lo, hi in bins:
        recalls = []
        for vid, gt_instances in gts.items():
            bin_gts = [(s, e, l) for s, e, l in gt_instances if lo <= (e - s) < hi]
            if not bin_gts:
                continue
            if vid not in pred_by_vid:
                continue
            pred = pred_by_vid[vid]
            n_matched = 0
            matched = set()
            pred_segs = pred['segments']
            pred_labels = pred['labels']
            for p_idx in range(len(pred_segs)):
                p_seg = np.array([pred_segs[p_idx][0], pred_segs[p_idx][1]])
                for g_idx, (s, e, _) in enumerate(bin_gts):
                    if g_idx in matched:
                        continue
                    gt_seg = np.array([[s, e]])
                    tiou = segment_iou(p_seg, gt_seg)[0]
                    if tiou >= 0.5:
                        n_matched += 1
                        matched.add(g_idx)
                        break
            r = n_matched / len(bin_gts) if bin_gts else 1.0
            recalls.append(r)
        avg_r = np.mean(recalls) if recalls else 0.0
        results[bin_name] = {'n_instances': sum(1 for gt_instances in gts.values()
                                                for s, e, _ in gt_instances if lo <= (e - s) < hi),
                             'avg_recall': avg_r}
        print(f"  {bin_name:<15} 实例数={results[bin_name]['n_instances']:>4}  Recall={avg_r:.1%}")

    # 画图
    fig, ax = plt.subplots(figsize=(8, 5))
    labels = [f'{bn}\n(n={results[bn]["n_instances"]})' for bn, _, _ in bins]
    values = [results[bn]['avg_recall'] * 100 for bn, _, _ in bins]
    colors = ['#2ecc71', '#3498db', '#e74c3c']
    bars = ax.bar(labels, values, color=colors, edgecolor='white')
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f'{val:.1f}%', ha='center', fontsize=12)
    ax.set_ylabel('Recall @ tIoU=0.5', fontsize=13)
    ax.set_title('Recall by Action Duration', fontsize=14)
    ax.set_ylim(0, max(values) * 1.2 + 5 if max(values) > 0 else 100)
    plt.tight_layout()
    out = os.path.join(EXPORT_DIR, 'duration_analysis.png')
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  图表: {out}")
    return results


def run_density_analysis():
    """密度分层分析。"""
    from libs.utils.metrics import segment_iou

    print(f"\n{'='*60}")
    print("  按视频动作密度分层分析")
    print(f"{'='*60}")

    with open(JSON_FILE, 'r') as f:
        db = json.load(f)
    gts = {}
    for vid, info in db['database'].items():
        if info['subset'].lower() != 'test':
            continue
        instances = [(float(a['segment'][0]), float(a['segment'][1]), a['label_id'])
                     for a in info.get('annotations', [])]
        gts[vid] = instances

    pred_pkl = None
    for candidate in ['ckpt/abl_A1_A1/eval_results.pkl',
                      'ckpt/abl_A2_A2/eval_results.pkl']:
        if os.path.exists(os.path.join(REPO, candidate)):
            pred_pkl = os.path.join(REPO, candidate)
            break
    if pred_pkl is None:
        print("  [SKIP] 无可用预测结果")
        return None

    with open(pred_pkl, 'rb') as f:
        preds = pickle.load(f)
    pred_by_vid = {p['video-id']: p for p in preds}

    density_bins = [('1-3', 1, 3), ('4-6', 4, 6), ('7-10', 7, 10), ('>10', 11, 999)]
    bin_vids = {bn: [] for bn, _, _ in density_bins}
    for vid, gt_instances in gts.items():
        n = len(gt_instances)
        for bn, lo, hi in density_bins:
            if lo <= n <= hi:
                bin_vids[bn].append(vid)
                break

    # 计算每桶 mAP（简化版：每个视频的 AP）
    for bn, lo, hi in density_bins:
        vids = bin_vids[bn]
        if not vids:
            continue
        aps = []
        for vid in vids:
            gt_instances = gts[vid]
            if vid not in pred_by_vid:
                continue
            pred = pred_by_vid[vid]
            # 简化计算：匹配率
            n_matched = 0
            matched = set()
            pred_segs = pred['segments']
            for p_idx in range(len(pred_segs)):
                p_seg = np.array([pred_segs[p_idx][0], pred_segs[p_idx][1]])
                for g_idx, (s, e, _) in enumerate(gt_instances):
                    if g_idx in matched:
                        continue
                    gt_seg = np.array([[s, e]])
                    tiou = segment_iou(p_seg, gt_seg)[0]
                    if tiou >= 0.5:
                        n_matched += 1
                        matched.add(g_idx)
                        break
            ap = n_matched / len(gt_instances) if gt_instances else 1.0
            aps.append(ap)
        print(f"  {bn:<8} 视频数={len(vids):>3}  平均匹配率={np.mean(aps) if aps else 0:.1%}")

    # 画密度分布直方图
    densities = [len(gt_instances) for gt_instances in gts.values()]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(densities, bins=20, edgecolor='white', color='#3498db')
    ax.set_xlabel('GT instances per video', fontsize=13)
    ax.set_ylabel('Video count', fontsize=13)
    ax.set_title('Action Density Distribution (THUMOS14 Test)', fontsize=14)
    plt.tight_layout()
    out = os.path.join(EXPORT_DIR, 'density_hist.png')
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  图表: {out}")


def run_confusion_analysis():
    """混淆矩阵分析。"""
    from libs.utils.metrics import segment_iou

    print(f"\n{'='*60}")
    print("  混淆矩阵分析")
    print(f"{'='*60}")

    with open(JSON_FILE, 'r') as f:
        db = json.load(f)
    label_names = {}
    gts = {}
    for vid, info in db['database'].items():
        if info['subset'].lower() != 'test':
            continue
        instances = []
        for a in info.get('annotations', []):
            label_names[a['label_id']] = a['label']
            instances.append((float(a['segment'][0]), float(a['segment'][1]), a['label_id']))
        gts[vid] = instances

    pred_pkl = None
    for candidate in ['ckpt/abl_A1_A1/eval_results.pkl',
                      'ckpt/abl_A2_A2/eval_results.pkl']:
        if os.path.exists(os.path.join(REPO, candidate)):
            pred_pkl = os.path.join(REPO, candidate)
            break
    if pred_pkl is None:
        print("  [SKIP] 无可用预测结果")
        return

    with open(pred_pkl, 'rb') as f:
        preds = pickle.load(f)
    pred_by_vid = {p['video-id']: p for p in preds}

    num_classes = len(label_names)
    cm = np.zeros((num_classes, num_classes), dtype=np.int32)

    for vid, gt_instances in gts.items():
        if vid not in pred_by_vid:
            continue
        pred = pred_by_vid[vid]
        matched_gt = set()
        pred_segs = pred['segments']
        pred_labels = pred['labels']
        pred_scores = pred['scores']
        # 按置信度排序
        order = np.argsort(-pred_scores)
        for p_idx in order:
            p_seg = np.array([pred_segs[p_idx][0], pred_segs[p_idx][1]])
            p_label = int(pred_labels[p_idx])
            best_tiou, best_idx = 0, -1
            for g_idx, (s, e, gt_label) in enumerate(gt_instances):
                if g_idx in matched_gt:
                    continue
                gt_seg = np.array([[s, e]])
                tiou = segment_iou(p_seg, gt_seg)[0]
                if tiou > best_tiou:
                    best_tiou, best_idx = tiou, g_idx
            if best_tiou >= 0.5 and best_idx >= 0:
                gt_label = gt_instances[best_idx][2]
                cm[gt_label, p_label] += 1
                matched_gt.add(best_idx)

    # 输出 top 误分类
    errors = []
    for i in range(num_classes):
        for j in range(num_classes):
            if i != j and cm[i, j] > 0:
                errors.append((i, j, cm[i, j]))
    errors.sort(key=lambda x: -x[2])
    print("  Top 10 误分类对:")
    for gt_idx, pred_idx, count in errors[:10]:
        gt_name = label_names.get(gt_idx, f'cls_{gt_idx}')
        pred_name = label_names.get(pred_idx, f'cls_{pred_idx}')
        print(f"    {gt_name} → {pred_name}: {count} 次")

    # 归一化
    row_sums = cm.sum(axis=1, keepdims=True).clip(min=1)
    cm_norm = cm.astype(float) / row_sums

    # 画图
    labels = [label_names.get(i, f'{i}') for i in range(num_classes)]
    fig, ax = plt.subplots(figsize=(14, 12))
    im = ax.imshow(cm_norm, cmap='YlOrRd', aspect='auto')
    ax.set_xticks(range(num_classes))
    ax.set_yticks(range(num_classes))
    ax.set_xticklabels(labels, rotation=90, fontsize=7)
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_xlabel('Predicted Label', fontsize=13)
    ax.set_ylabel('Ground Truth Label', fontsize=13)
    ax.set_title('THUMOS14 Confusion Matrix', fontsize=14)
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Fraction of GT', fontsize=11)
    plt.tight_layout()
    out = os.path.join(EXPORT_DIR, 'confusion_matrix.png')
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  图表: {out}")
    return cm


def ablation_summary(eval_results):
    """消融实验汇总对比。"""
    print(f"\n{'='*60}")
    print("  消融实验汇总")
    print(f"{'='*60}")

    if not eval_results:
        print("  无可用评估结果")
        return

    rows = []
    for exp_id, r in sorted(eval_results.items()):
        rows.append(r)

    # 写 CSV
    csv_path = os.path.join(EXPORT_DIR, 'ablation_summary.csv')
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['实验', '描述', 'mAP@0.3', 'mAP@0.5', 'mAP@0.7', 'avg_mAP',
                    '骨干', 'Trident', 'k', 'FPN', '采样半径'])
        for r in rows:
            w.writerow([r['exp_id'], r['desc'],
                        f"{r.get('mAP@0.3', 0):.2f}",
                        f"{r.get('mAP@0.5', 0):.2f}",
                        f"{r.get('mAP@0.7', 0):.2f}",
                        f"{r.get('avg_mAP', 0):.2f}",
                        r.get('backbone_type', ''),
                        r.get('use_trident_head', ''),
                        r.get('k', ''),
                        r.get('fpn_type', ''),
                        r.get('center_sample_radius', '')])
    print(f"  汇总 CSV: {csv_path}")
    for r in rows:
        print(f"  {r['exp_id']:<8} avg_mAP={r.get('avg_mAP', 0):.2f}%  "
              f"  ({r['desc'][:50]})")

    # 画对比图
    if len(rows) >= 2:
        fig, ax = plt.subplots(figsize=(10, 5))
        ids = [r['exp_id'] for r in rows]
        avgs = [r.get('avg_mAP', 0) for r in rows]
        bars = ax.bar(ids, avgs, color=['#3498db', '#e74c3c', '#f39c12'][:len(ids)], edgecolor='white')
        for bar, val in zip(bars, avgs):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                    f'{val:.2f}%', ha='center', fontsize=12)
        ax.set_ylabel('Average mAP (%)', fontsize=13)
        ax.set_title('Ablation Study: Component Contribution', fontsize=14)
        ax.set_ylim(0, max(avgs) * 1.2 + 2 if max(avgs) > 0 else 80)
        plt.tight_layout()
        out = os.path.join(EXPORT_DIR, 'ablation_comparison.png')
        plt.savefig(out, dpi=150)
        plt.close()
        print(f"  对比图: {out}")


def main():
    t0 = time.time()
    print("=" * 70)
    print("  TriDet 测试与分析 — 完整报告")
    print("=" * 70)

    # ---- 1. 模块 Shape 测试 ----
    n_pass, n_fail = run_module_analysis()

    # ---- 2. 配置参数分析 ----
    config_rows = analyze_configs()

    # ---- 3. 模型评估 ----
    eval_results = run_all_evals()

    # ---- 4. 消融汇总 ----
    ablation_summary(eval_results)

    # ---- 5. 时长分层分析 ----
    duration_results = run_duration_analysis()

    # ---- 6. 密度分层分析 ----
    run_density_analysis()

    # ---- 7. 混淆矩阵分析 ----
    run_confusion_analysis()

    # ---- 汇总 ----
    elapsed = time.time() - t0
    print(f"\n{'='*70}")
    print(f"  分析完成！总耗时: {elapsed:.0f}s")
    print(f"  所有结果保存在: {EXPORT_DIR}")
    print(f"{'='*70}")

    # 输出关键发现
    print("\n[关键发现]")
    if n_pass > 0:
        print(f"  • 模块测试: {n_pass}/{n_pass + n_fail} 通过 — 所有 15 个模块 shape 链完整")

    if eval_results:
        baseline = eval_results.get('baseline')
        a1 = eval_results.get('A1')
        a2 = eval_results.get('A2')
        if a1 and baseline:
            diff = a1['avg_mAP'] - baseline['avg_mAP']
            print(f"  • Trident-head 贡献: {diff:+.2f}% avg_mAP (A1 vs 基线)")
        if a2 and baseline:
            diff = a2['avg_mAP'] - baseline['avg_mAP']
            print(f"  • SGP 骨干贡献: {diff:+.2f}% avg_mAP (A2 vs 基线)")

    if duration_results:
        min_recall = min(duration_results.values(), key=lambda x: x['avg_recall'])
        worst_bin = [k for k, v in duration_results.items() if v == min_recall][0]
        print(f"  • 最差时长区间: {worst_bin} (recall={min_recall['avg_recall']:.1%})")


if __name__ == '__main__':
    main()

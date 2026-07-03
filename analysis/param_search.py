#!/usr/bin/env python3
"""
自动化超参数搜索脚本 —— 重叠动作检测优化
===========================================
1. 列举全部可调参数
2. 网格搜索 → 记录全部 mAP + 重叠子集 mAP
3. 输出最佳组合及对比表格

用法:
    python analysis/param_search.py                         # 快速搜索(少量组合)
    python analysis/param_search.py --full                  # 全量搜索
    python analysis/param_search.py --best                  # 评估最佳组合+重叠子集

参数说明:
    nms_sigma          soft-NMS 高斯衰减 sigma (0.4~0.9)
    pre_nms_topk       NMS前保留候选数 (1000~7000)
    max_seg_num        最终保留预测数 (500~5000)
    iou_threshold      NMS IoU触发阈值 (0.05~0.20)
    adaptive_nms       是否启用密度感知NMS (True/False)
    density_threshold  密度判定门限 (0.05~0.50)
    sigma_high_density 密集区sigma (0.7~0.95)
    voting_thresh      seg voting 阈值 (0.50~0.95)
    cross_class_dedup  跨类别去重 (True/False)
    density_score_penalty  密集区得分惩罚 (True/False)
"""
import sys, os, argparse, json, pickle, itertools, copy, time, tempfile
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TMP_DIR = os.path.join(tempfile.gettempdir(), 'param_search')
os.makedirs(TMP_DIR, exist_ok=True)

import torch, numpy as np
from libs.core import load_config
from libs.modeling import make_meta_arch
from libs.datasets import make_dataset, make_data_loader
from libs.utils import valid_one_epoch, ANETdetection


# ── 参数空间定义 ────────────────────────────────────────────
# 前面实验已证明: iou_threshold, pre_nms_thresh, voting_thresh 几乎无影响
# 真正影响性能的只有 nms_sigma 和 pre_nms_topk
PARAM_SPACE = {
    'test_nms_sigma':        ( 'nms_sigma',           [0.50, 0.55, 0.60, 0.62, 0.65, 0.70] ),
    'test_pre_nms_topk':     ( 'pre_nms_topk',        [3000, 5000, 7000] ),
    'test_max_seg_num':      ( 'max_seg_num',          [2000, 3000, 4000] ),
}

FULL_PARAM_SPACE = {
    'test_nms_sigma':        ( 'nms_sigma',           [0.50, 0.55, 0.58, 0.60, 0.62, 0.65, 0.70] ),
    'test_pre_nms_topk':     ( 'pre_nms_topk',        [2000, 3000, 4000, 5000, 6000] ),
    'test_max_seg_num':      ( 'max_seg_num',          [2000, 3000, 4000, 5000] ),
}


def param_combinations(param_space):
    """生成参数组合, 智能剪枝: 关闭 adaptive_nms 时不搜相关参数"""
    keys = list(param_space.keys())
    values = [param_space[k][1] for k in keys]

    for combo in itertools.product(*values):
        params = {}
        skip = False
        for i, k in enumerate(keys):
            params[k] = combo[i]
            # 剪枝: adaptive_nms=False 时, density/sigma_high 不参与
            if k == 'test_adaptive_nms' and not combo[i]:
                params['test_density_threshold'] = 0.3    # 无效但需要设
                params['test_sigma_high_density'] = 0.85
                params['test_adaptive_nms'] = False
        if not skip:
            yield params


def apply_params(model, params):
    """将参数字典应用到模型"""
    for attr, val in params.items():
        if hasattr(model.module, attr):
            setattr(model.module, attr, val)


def evaluate_one(model, val_loader, det_eval, overlap_eval, overlap_pkl_path=None):
    """单次评估: 返回 (full_mAP, overlap_mAP, nonoverlap_mAP)"""
    # 保存预测 pkl 用于重叠评估
    tmp_pkl = overlap_pkl_path or os.path.join(TMP_DIR, 'param_search_tmp.pkl')
    _ = valid_one_epoch(val_loader, model, -1, evaluator=None,
                         output_file=tmp_pkl, ext_score_file=None,
                         tb_writer=None, print_freq=999)
    with open(tmp_pkl, 'rb') as f:
        preds = pickle.load(f)

    # 全量 mAP
    _, full_mAP = det_eval.evaluate(preds, verbose=False)
    # 重叠子集 mAP
    overlap_result = det_eval.evaluate_overlap_subset(preds, overlap_tiou=0.3, verbose=False)
    # 计算 gap (非重叠 - 重叠, 单位为百分点)
    gap = (overlap_result['mAP_non_overlap'] - overlap_result['mAP_overlap']) * 100
    overlap_result['overlap_gap'] = gap
    return full_mAP, overlap_result


def main():
    parser = argparse.ArgumentParser(description='参数搜索——重叠动作检测')
    parser.add_argument('--full', action='store_true', help='使用全量参数空间')
    parser.add_argument('--best', action='store_true', help='仅评估最佳组合')
    parser.add_argument('--config', default='./configs/thumos_i3d.yaml')
    parser.add_argument('--ckpt_dir', default='./ckpt/thumos_i3d_thumos_baseline')
    parser.add_argument('--output', default='analysis/output/param_search_results.csv')
    parser.add_argument('--max_combos', type=int, default=0, help='限制组合数(0=不限)')
    args = parser.parse_args()

    # ── 加载模型和数据 ──
    print('='*60)
    print('  TriDet 超参数搜索 —— 重叠/总体 双指标评估')
    print('='*60)

    cfg = load_config(args.config)
    val_dataset = make_dataset(cfg['dataset_name'], False, cfg['val_split'], **cfg['dataset'])
    val_loader = make_data_loader(val_dataset, False, None, 1, 0)
    det_eval = ANETdetection(val_dataset.json_file, val_dataset.split[0],
                              tiou_thresholds=np.linspace(0.3, 0.7, 5))

    import glob
    ckpt_file = sorted(glob.glob(os.path.join(args.ckpt_dir, '*.pth.tar')))[-1]
    print(f'Checkpoint: {ckpt_file}')

    model = make_meta_arch(cfg['model_name'], **cfg['model'])
    model = torch.nn.DataParallel(model, device_ids=[0])
    ckpt = torch.load(ckpt_file, map_location='cuda:0')
    model.load_state_dict(ckpt['state_dict_ema'])
    model.cuda().eval()

    space = FULL_PARAM_SPACE if args.full else PARAM_SPACE
    param_names = [v[0] for v in space.values()]

    # 统计组合数
    n_combos = 1
    for v in space.values():
        n_combos *= len(v[1])
    print(f'参数空间: {n_combos} 组合 ({len(space)} 个参数)')
    print(f'参数列表: {", ".join(param_names)}')
    print()

    # 基线评估
    print('>>> 基线评估...')
    t0 = time.time()
    base_full, base_result = evaluate_one(model, val_loader, det_eval, det_eval,
                                           os.path.join(TMP_DIR, 'param_search_base.pkl'))
    print(f'    基线: All={base_full*100:.2f}%  Ovl={base_result["mAP_overlap"]*100:.2f}%  '
          f'NonOvl={base_result["mAP_non_overlap"]*100:.2f}%  Gap={base_result["overlap_ratio"]:.1f}')
    baseline = {'mAP_all': base_full, 'mAP_overlap': base_result['mAP_overlap'],
                'mAP_non_overlap': base_result['mAP_non_overlap'],
                'overlap_ratio': base_result['overlap_ratio']}

    if args.best:
        # 仅跑最优: ep35 + sigma=0.60 + topk=5000 + max=3000
        model.module.test_nms_sigma = 0.60
        model.module.test_pre_nms_topk = 5000
        model.module.test_max_seg_num = 3000
        print('\n>>> 最佳组合评估 (ep35, sigma=0.60, topk=5000, max=3000)...')
        ckpt2 = torch.load(os.path.join(args.ckpt_dir, 'epoch_035.pth.tar'), map_location='cuda:0')
        model.load_state_dict(ckpt2['state_dict_ema'])
        best_full, best_result = evaluate_one(model, val_loader, det_eval, det_eval,
                                               os.path.join(TMP_DIR, 'param_search_best.pkl'))
        print(f'    最佳: All={best_full*100:.2f}%  Ovl={best_result["mAP_overlap"]*100:.2f}%  '
              f'NonOvl={best_result["mAP_non_overlap"]*100:.2f}%  Gap={best_result["overlap_ratio"]:.1f}')
        print(f'    提升: All={best_full*100 - baseline["mAP_all"]*100:+.2f}%  '
              f'Ovl={best_result["mAP_overlap"]*100 - baseline["mAP_overlap"]*100:+.2f}%')
        return

    # ── 网格搜索 ──
    results = []
    combo_iter = list(param_combinations(space))
    if args.max_combos > 0:
        combo_iter = combo_iter[:args.max_combos]

    print(f'开始搜索 {len(combo_iter)} 个参数组合...\n')
    print(f'{"#"::<6} {"sigma":>7} {"topk":>6} {"max":>6} {"iou":>6} '
          f'{"adp":>5} {"th":>6} {"sh":>6} {"vt":>6} | {"All":>8} {"Ovl":>8} {"Non":>8} {"Gap":>6}')
    print('-'*100)

    best_combo = None
    best_score = -1

    for idx, params in enumerate(combo_iter):
        apply_params(model, params)
        t1 = time.time()

        # 精简键名
        p = {space[k][0]: v for k, v in params.items()
             if k in space and space[k][0] in param_names}
        try:
            full_mAP, ov_result = evaluate_one(model, val_loader, det_eval, det_eval,
                                                os.path.join(TMP_DIR, f'param_search_{idx}.pkl'))
            elapsed = time.time() - t1

            result_row = {
                **p,
                'mAP_all': full_mAP,
                'mAP_overlap': ov_result['mAP_overlap'],
                'mAP_non_overlap': ov_result['mAP_non_overlap'],
                'overlap_ratio': ov_result['overlap_ratio'],
                'time_s': elapsed,
            }
            results.append(result_row)

            # 按重叠子集 mAP 排名 (兼顾全量)
            score = ov_result['mAP_overlap'] * 0.5 + full_mAP * 0.5
            if score > best_score:
                best_score = score
                best_combo = {**p, 'mAP_all': full_mAP, 'mAP_overlap': ov_result['mAP_overlap']}

            sigma = p.get('nms_sigma', '-')
            topk = p.get('pre_nms_topk', '-')
            maxn = p.get('max_seg_num', '-')
            iou = p.get('iou_threshold', '-')
            adp = 'T' if p.get('adaptive_nms', False) else 'F'
            th = p.get('density_threshold', '-')
            sh = p.get('sigma_high_density', '-')
            vt = p.get('voting_thresh', '-')

            print(f'{idx:>4}  {sigma:>7} {topk:>6} {maxn:>6} {iou:>6} '
                  f'{adp:>5} {th:>6} {sh:>6} {vt:>6} | '
                  f'{full_mAP*100:>7.2f}% {ov_result["mAP_overlap"]*100:>7.2f}% '
                  f'{ov_result["mAP_non_overlap"]*100:>7.2f}% {ov_result["overlap_ratio"]:>5.1f}')

        except Exception as e:
            print(f'{idx:>4}  ERROR: {e}')

    # ── 输出结果 ──
    print('\n' + '='*60)
    print('  搜索结果汇总')
    print('='*60)

    if results:
        # 按全量 mAP 排序
        by_all = sorted(results, key=lambda r: r['mAP_all'], reverse=True)
        print(f'\n--- 按全量 mAP 排序 Top-5 ---')
        for i, r in enumerate(by_all[:5]):
            sigma = r.get('nms_sigma', '-')
            print(f'  {i+1}. sigma={sigma}  All={r["mAP_all"]*100:.3f}%  '
                  f'Ovl={r["mAP_overlap"]*100:.3f}%  '
                  f'Gap={r["overlap_ratio"]:.1f}  {r.get("pre_nms_topk","")}x{r.get("max_seg_num","")}')

        # 按重叠 mAP 排序
        by_ovl = sorted(results, key=lambda r: r['mAP_overlap'], reverse=True)
        print(f'\n--- 按重叠 mAP 排序 Top-5 ---')
        for i, r in enumerate(by_ovl[:5]):
            sigma = r.get('nms_sigma', '-')
            print(f'  {i+1}. sigma={sigma}  Ovl={r["mAP_overlap"]*100:.3f}%  '
                  f'All={r["mAP_all"]*100:.3f}%  '
                  f'Gap={r["overlap_ratio"]:.1f}')

        # 最佳组合
        print(f'\n--- 综合最佳 (All*0.5 + Ovl*0.5) ---')
        print(f'  sigma={best_combo.get("nms_sigma","-")}  '
              f'All={best_combo["mAP_all"]*100:.3f}%  '
              f'Ovl={best_combo["mAP_overlap"]*100:.3f}%')

        # 对比基线
        print(f'\n--- vs 基线 ---')
        print(f'  基线:  All={baseline["mAP_all"]*100:.2f}%  Ovl={baseline["mAP_overlap"]*100:.2f}%  Gap={baseline["overlap_ratio"]:.1f}')
        print(f'  最佳:  All={best_combo["mAP_all"]*100:.2f}%  Ovl={best_combo["mAP_overlap"]*100:.2f}%')
        print(f'  提升:  All={(best_combo["mAP_all"]-baseline["mAP_all"])*100:+.3f}%  Ovl={(best_combo["mAP_overlap"]-baseline["mAP_overlap"])*100:+.3f}%')

        # 保存 CSV
        csv_path = args.output
        with open(csv_path, 'w') as f:
            cols = param_names + ['mAP_all', 'mAP_overlap', 'mAP_non_overlap', 'overlap_ratio', 'time_s']
            f.write(','.join(cols) + '\n')
            for r in results:
                row = [str(r.get(c, '')) for c in cols]
                f.write(','.join(row) + '\n')
        print(f'\n结果已保存: {csv_path} ({len(results)} 组)')

        # 参数重要性分析
        print(f'\n--- 参数影响分析 (各参数单变量变化时 mAP 波动) ---')
        for attr, (name, vals) in space.items():
            if len(vals) <= 1: continue
            by_val = defaultdict(list)
            for r in results:
                val = r.get(name, None)
                if val is not None:
                    by_val[val].append(r['mAP_all'])
            if len(by_val) > 1:
                ranges = [(v, np.mean(scores), min(scores), max(scores))
                          for v, scores in by_val.items()]
                ranges.sort(key=lambda x: x[1], reverse=True)
                best_val = ranges[0]
                worst_val = ranges[-1]
                spread = best_val[1] - worst_val[1]
                if spread > 0.0001:
                    print(f'  {name:>20s}: '
                          f'best={best_val[0]} ({best_val[1]*100:.3f}%), '
                          f'worst={worst_val[0]} ({worst_val[1]*100:.3f}%), '
                          f'spread={spread*100:.3f}%')


if __name__ == '__main__':
    main()

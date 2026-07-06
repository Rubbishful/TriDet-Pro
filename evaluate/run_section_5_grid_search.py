"""
TriDet 全局改进实验 — 第五部分 IoU Head 超参数网格搜索

对应 overall_exp_config.md 第五节 5.2 超参数网格搜索（36 组合）:
  搜索空间: loss_weight ∈ {0.1, 0.25, 0.5} × per_level ∈ {F, T} × residual ∈ {F, T} × layers ∈ {2, 3, 4}
  固定条件: THUMOS14 I3D, BCE pos-only (关闭 TAL), 40 epochs

用法:
    python evaluate/run_section_5_grid_search.py                     # 全部 36 组合
    python evaluate/run_section_5_grid_search.py --limit 10           # 仅前 10 个
    python evaluate/run_section_5_grid_search.py --start GS-05        # 从 GS-05 续跑
    python evaluate/run_section_5_grid_search.py --task train         # 仅训练(跳过评估)

路径约定:
    数据:  ./data/thumos/i3d_features/
    模型:  ckpt/<exp_dir>/
    输出:  evaluate/results/iou_grid_search.csv
"""

import os, sys, subprocess, time, argparse, tempfile, itertools
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from evaluate.common import deep_set, dump_yaml, parse_maps

# ---- 路径配置 ----
PYTHON      = sys.executable
RESULT_DIR  = REPO / 'evaluate' / 'results'
CKPT_DIR    = REPO / 'ckpt'
CSV_PATH    = RESULT_DIR / 'iou_grid_search.csv'
BASE_CONFIG = REPO / 'configs' / 'thumos_i3d.yaml'

# 网格搜索 epochs（与文档 40 epochs 一致）
GRID_EPOCHS = 40


# ============================================================
# 搜索空间定义
# ============================================================

SEARCH_SPACE = {
    'iou_loss_weight': [0.1, 0.25, 0.5],
    'iou_per_level':   [False, True],
    'iou_head_residual': [False, True],
    'iou_head_layers': [2, 3, 4],
}

FIXED_OVERRIDES = {
    'model.use_iou_head': True,
    'model.iou_head_dim': 512,
    'model.iou_loss_type': 'bce',       # BCE pos-only (关闭 TAL)
    'model.iou_warmup_epochs': 0,
    'model.tal_topk': 0,                # 关闭 TAL
    # 40 epochs
    'opt.epochs': GRID_EPOCHS,
    'opt.warmup_epochs': 20,            # keep warmup as baseline
}


def generate_grid_id(lw, pl, res, layers):
    """生成简短实验 ID: GS-01 ~ GS-36"""
    # 按 loss_weight 排序, 编稳定索引
    pass  # 在下面统一编号


def build_experiment_list():
    """生成全部 36 个实验定义, 按 mAP 潜力排序: 低 loss_weight 优先."""
    exps = []
    idx = 1
    # 按 loss_weight asc → per_level asc → residual asc → layers asc
    for lw in SEARCH_SPACE['iou_loss_weight']:
        for pl in SEARCH_SPACE['iou_per_level']:
            for res in SEARCH_SPACE['iou_head_residual']:
                for layers in SEARCH_SPACE['iou_head_layers']:
                    exp_id = f'GS-{idx:02d}'
                    idx += 1

                    desc_parts = [f'lw={lw}']
                    if pl:
                        desc_parts.append('per_lvl')
                    if res:
                        desc_parts.append('res')
                    desc_parts.append(f'L{layers}')

                    exps.append({
                        'id': exp_id,
                        'group': 'IoU Grid',
                        'desc': ', '.join(desc_parts),
                        'config_desc': f'IoUHead({", ".join(desc_parts)})',
                        'overrides': {
                            **FIXED_OVERRIDES,
                            'model.iou_loss_weight': lw,
                            'model.iou_per_level': pl,
                            'model.iou_head_residual': res,
                            'model.iou_head_layers': layers,
                        },
                    })
    return exps


EXPERIMENTS = build_experiment_list()


# ============================================================
# 工具函数
# ============================================================

def run_cmd(cmd, desc, timeout_s=5400):
    """运行命令, 输出重定向到临时文件.
    返回 (ok: bool, stdout: str, elapsed_min: float)."""
    print(f"\n  [{desc}]")
    sys.stdout.flush()
    t0 = time.time()

    out_f = tempfile.NamedTemporaryFile(
        mode='w+', suffix='.log', delete=False,
        dir=str(RESULT_DIR), prefix='run_')
    out_path = out_f.name
    try:
        r = subprocess.run(
            cmd, cwd=str(REPO),
            stdout=out_f, stderr=subprocess.STDOUT,
            text=True, timeout=timeout_s)
    except subprocess.TimeoutExpired:
        out_f.close()
        el = (time.time() - t0) / 60
        print(f"  [TIMEOUT] {el:.0f}min")
        try:
            with open(out_path, 'r', encoding='utf-8', errors='replace') as f:
                partial = f.read()
        except Exception:
            partial = ''
        try:
            os.unlink(out_path)
        except Exception:
            pass
        return False, partial, el
    finally:
        out_f.close()

    el = (time.time() - t0) / 60
    ok = r.returncode == 0
    tag = 'OK' if ok else f'FAIL(rc={r.returncode})'
    print(f"  [{desc}] {tag} [{el:.0f}min]")

    with open(out_path, 'r', encoding='utf-8', errors='replace') as f:
        out_text = f.read()

    lines = [x for x in out_text.strip().split('\n') if x.strip()]
    for l in lines[-5:]:
        print(f"    {l[:150]}")

    try:
        os.unlink(out_path)
    except Exception:
        pass

    return ok, out_text, el


def save_csv(all_results):
    """写入 CSV 到 evaluate/results/iou_grid_search.csv."""
    import csv
    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    def _fmt(v):
        if v is None:
            return ''
        if isinstance(v, float):
            return f'{v:.2f}'
        return str(v)

    with open(CSV_PATH, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['实验编号', 'loss_weight', 'per_level', 'residual', 'layers',
                    'mAP@0.3', 'mAP@0.5', 'mAP@0.7', 'avg_mAP',
                    '训练时间(min)', '状态', '时间戳'])
        for exp_id in sorted(all_results.keys()):
            r = all_results[exp_id]
            w.writerow([
                exp_id,
                _fmt(r.get('loss_weight')),
                str(r.get('per_level', '')),
                str(r.get('residual', '')),
                str(r.get('layers', '')),
                _fmt(r.get('mAP@0.3')), _fmt(r.get('mAP@0.5')),
                _fmt(r.get('mAP@0.7')), _fmt(r.get('avg_mAP')),
                r.get('train_time', ''), r.get('status', ''),
                r.get('timestamp', ''),
            ])


# ============================================================
# 核心逻辑
# ============================================================

def train_and_evaluate(exp, all_results, ts, task='all'):
    """训练 → 评估单个网格组合."""
    exp_id = exp['id']
    group = exp['group']
    name = exp['config_desc']
    overrides = exp['overrides']

    print(f"\n{'='*55}")
    print(f"  {exp_id}: {exp['desc']}  [{group}]")
    print(f"{'='*55}")

    # —— 生成覆写配置 ——
    from libs.core.config import load_config as _load_cfg
    cfg = _load_cfg(str(BASE_CONFIG))
    for key_path, val in overrides.items():
        deep_set(cfg, key_path, val)

    exp_cfg_path = RESULT_DIR / f'grid_{exp_id}.yaml'
    dump_yaml(cfg, str(exp_cfg_path))
    print(f"  配置: {exp_cfg_path}")

    train_min = ''

    if task in ('all', 'train'):
        # —— 训练（40 epochs, 超时放宽到 4h） ——
        ok, _, train_min = run_cmd(
            [PYTHON, '-u', 'train.py', str(exp_cfg_path), '--output', exp_id],
            f'train {exp_id}', timeout_s=14400)

        if not ok:
            all_results[exp_id] = {
                'group': group, 'name': name,
                'loss_weight': overrides.get('model.iou_loss_weight'),
                'per_level': overrides.get('model.iou_per_level'),
                'residual': overrides.get('model.iou_head_residual'),
                'layers': overrides.get('model.iou_head_layers'),
                'train_time': f'{train_min:.0f}', 'status': 'TRAIN_FAILED',
                'timestamp': ts,
            }
            save_csv(all_results)
            return

    if task in ('all', 'eval'):
        # —— 评估 ——
        ckpt_path = CKPT_DIR / f'grid_{exp_id}_{exp_id}'
        ok2, out_text, _ = run_cmd(
            [PYTHON, '-u', 'eval.py', str(exp_cfg_path), str(ckpt_path), '-p', '10'],
            f'eval {exp_id}', timeout_s=7200)

        maps = parse_maps(out_text) if ok2 else {}
        all_results[exp_id] = {
            'group': group, 'name': name,
            'loss_weight': overrides.get('model.iou_loss_weight'),
            'per_level': overrides.get('model.iou_per_level'),
            'residual': overrides.get('model.iou_head_residual'),
            'layers': overrides.get('model.iou_head_layers'),
            'mAP@0.3': maps.get(0.3), 'mAP@0.5': maps.get(0.5),
            'mAP@0.7': maps.get(0.7), 'avg_mAP': maps.get('avg'),
            'train_time': f'{train_min:.0f}' if isinstance(train_min, float) else train_min,
            'status': 'OK' if maps else 'EVAL_FAILED',
            'timestamp': ts,
        }
        save_csv(all_results)

        if maps:
            print(f"  ★ [{exp_id}] avg_mAP={maps.get('avg', '?'):.2f}%")
    else:
        all_results[exp_id] = {
            'group': group, 'name': name,
            'loss_weight': overrides.get('model.iou_loss_weight'),
            'per_level': overrides.get('model.iou_per_level'),
            'residual': overrides.get('model.iou_head_residual'),
            'layers': overrides.get('model.iou_head_layers'),
            'train_time': f'{train_min:.0f}' if isinstance(train_min, float) else '',
            'status': 'TRAINED',
            'timestamp': ts,
        }
        save_csv(all_results)


# ============================================================
# 主入口
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description='TriDet IoU Head 超参数网格搜索 (36 组合, 40 epochs/trial)')
    parser.add_argument('--limit', type=int, default=0,
                        help='限制执行前 N 个实验 (0=全部)')
    parser.add_argument('--start', type=str, default='',
                        help='从指定实验 ID 开始 (断点续跑, 如 GS-05)')
    parser.add_argument('--task', type=str, default='all',
                        choices=['all', 'train', 'eval'],
                        help='执行阶段: all(训+评), train(仅训练), eval(仅评估)')
    args = parser.parse_args()

    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    all_results = {}
    ts = datetime.now().strftime('%Y-%m-%d %H:%M')

    queue = list(EXPERIMENTS)

    # —— 断点续跑 ——
    if args.start:
        skip = True
        filtered = []
        for e in queue:
            if skip and e['id'] == args.start:
                skip = False
            if not skip:
                filtered.append(e)
        if skip:
            print(f"  [WARN] 未找到起始实验: {args.start}")
        queue = filtered

    # —— 数量限制 ——
    if args.limit > 0:
        queue = queue[:args.limit]

    total = len(queue)
    if total == 0:
        print("  没有需要执行的实验!")
        return

    print(f"\n{'='*70}")
    print(f"  Section 5 网格搜索 — {total} 项 × 40 epochs")
    print(f"  搜索空间: loss_weight×3 × per_level×2 × residual×2 × layers×3 = 36")
    print(f"  输出: {CSV_PATH}")
    print(f"{'='*70}")

    for i, exp in enumerate(queue):
        print(f"\n  >>> [{i+1}/{total}] {exp['id']}: {exp['desc']} <<<")
        train_and_evaluate(exp, all_results, ts, task=args.task)

    print(f"\n{'='*70}")
    print(f"  全部完成! {len(all_results)} results → {CSV_PATH}")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()

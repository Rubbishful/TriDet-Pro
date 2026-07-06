"""
TriDet 全局改进实验 — 第二部分(通道注意力) + 第三部分(BiFPN) + 第四部分(回归损失)

对应 overall_exp_config.md:
  二、通道注意力（SE 模块）— 3 实验
  三、BiFPN 特征金字塔增强 — 2 实验
  四、回归损失函数增强 — 3 实验

用法:
    python evaluate/run_section_2_4_structural.py                     # 全部执行
    python evaluate/run_section_2_4_structural.py --group se           # 仅 SE
    python evaluate/run_section_2_4_structural.py --group bifpn        # 仅 BiFPN
    python evaluate/run_section_2_4_structural.py --group reg_loss     # 仅回归损失
    python evaluate/run_section_2_4_structural.py --exp SE-1,RL-2      # 指定实验
    python evaluate/run_section_2_4_structural.py --start SE-2         # 断点续跑
    python evaluate/run_section_2_4_structural.py --task train         # 仅训练
    python evaluate/run_section_2_4_structural.py --task eval --exp SE-1  # 仅评估(已有ckpt)

路径约定:
    数据:  ./data/thumos/i3d_features/
    模型:  ckpt/<exp_dir>/
    输出:  evaluate/results/structural_improvements.csv
"""

import os, sys, subprocess, time, argparse, tempfile
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from evaluate.common import deep_set, dump_yaml, parse_maps

# ---- 路径配置 ----
PYTHON      = sys.executable
RESULT_DIR  = REPO / 'evaluate' / 'results'
CKPT_DIR    = REPO / 'ckpt'
CSV_PATH    = RESULT_DIR / 'structural_improvements.csv'
BASE_CONFIG = REPO / 'configs' / 'thumos_i3d.yaml'


# ============================================================
# 实验定义 — 8 个实验, 按分组排列
# ============================================================

EXPERIMENTS = [
    # ── 二、通道注意力 (SE) ──
    {
        'id': 'SE-1', 'group': 'SE', 'desc': 'pre_fusion, r=16',
        'config_desc': 'SE(pre_fusion, r=16)',
        'overrides': {
            'model.use_att': True,
            'model.att_type': 'SE',
            'model.att_position': 'pre_fusion',
            'model.att_reduction': 16,
        },
    },
    {
        'id': 'SE-2', 'group': 'SE', 'desc': 'fusion, r=16',
        'config_desc': 'SE(fusion, r=16)',
        'overrides': {
            'model.use_att': True,
            'model.att_type': 'SE',
            'model.att_position': 'fusion',
            'model.att_reduction': 16,
        },
    },
    {
        'id': 'SE-3', 'group': 'SE', 'desc': 'pre_fusion, r=4',
        'config_desc': 'SE(pre_fusion, r=4)',
        'overrides': {
            'model.use_att': True,
            'model.att_type': 'SE',
            'model.att_position': 'pre_fusion',
            'model.att_reduction': 4,
        },
    },
    # ── 三、BiFPN 特征金字塔增强 ──
    {
        'id': 'BiFPN-1', 'group': 'BiFPN', 'desc': 'sum fusion',
        'config_desc': 'BiFPN(sum)',
        'overrides': {
            'model.fpn_type': 'bifpn',
            'model.bifpn_num_repeats': 1,
            'model.bifpn_fusion_method': 'sum',
            'model.bifpn_drop_path': 0.1,
        },
    },
    {
        'id': 'BiFPN-2', 'group': 'BiFPN', 'desc': 'fast_norm fusion',
        'config_desc': 'BiFPN(fast_norm)',
        'overrides': {
            'model.fpn_type': 'bifpn',
            'model.bifpn_num_repeats': 1,
            'model.bifpn_fusion_method': 'fast_norm',
            'model.bifpn_drop_path': 0.1,
        },
    },
    # ── 四、回归损失函数增强 ──
    {
        'id': 'RL-1', 'group': 'RegLoss', 'desc': 'Focaler-DIoU (d=0, u=0.95)',
        'config_desc': 'Focaler-DIoU(d=0,u=0.95)',
        'overrides': {
            'model.reg_loss_type': 'focaler_diou',
            'model.reg_loss_kwargs': {'d': 0.0, 'u': 0.95},
        },
    },
    {
        'id': 'RL-2', 'group': 'RegLoss', 'desc': 'EIoU',
        'config_desc': 'EIoU',
        'overrides': {
            'model.reg_loss_type': 'eiou',
            'model.reg_loss_kwargs': {},
        },
    },
    {
        'id': 'RL-3', 'group': 'RegLoss', 'desc': 'Alpha-DIoU (α=3.0)',
        'config_desc': 'Alpha-DIoU(α=3.0)',
        'overrides': {
            'model.reg_loss_type': 'alpha_diou',
            'model.reg_loss_kwargs': {'alpha': 3.0},
        },
    },
]


# ============================================================
# 工具函数
# ============================================================

def run_cmd(cmd, desc, timeout_s=3600):
    """运行命令, 输出重定向到临时文件避免 PIPE 死锁.
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
    """写入 CSV 到 evaluate/results/structural_improvements.csv."""
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
        w.writerow(['实验编号', '分组', '配置变更', 'mAP@0.3', 'mAP@0.5',
                    'mAP@0.7', 'avg_mAP', '训练时间(min)', '状态', '时间戳'])
        for exp_id in sorted(all_results.keys(), key=_sort_key):
            r = all_results[exp_id]
            w.writerow([
                exp_id, r.get('group', ''), r.get('name', ''),
                _fmt(r.get('mAP@0.3')), _fmt(r.get('mAP@0.5')),
                _fmt(r.get('mAP@0.7')), _fmt(r.get('avg_mAP')),
                r.get('train_time', ''), r.get('status', ''),
                r.get('timestamp', ''),
            ])


def _sort_key(exp_id):
    """按实验编号排序: SE-1 < SE-2 < ... < BiFPN-1 < ... < RL-1 < ..."""
    parts = exp_id.split('-')
    return (parts[0], int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 0)


# ============================================================
# 核心逻辑
# ============================================================

def train_and_evaluate(exp, all_results, ts, task='all'):
    """训练 → 评估单个实验."""
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

    exp_cfg_path = RESULT_DIR / f'struct_{exp_id}.yaml'
    dump_yaml(cfg, str(exp_cfg_path))
    print(f"  配置: {exp_cfg_path}")

    train_min = ''

    if task in ('all', 'train'):
        # —— 训练 ——
        ok, _, train_min = run_cmd(
            [PYTHON, '-u', 'train.py', str(exp_cfg_path), '--output', exp_id],
            f'train {exp_id}', timeout_s=14400)

        if not ok:
            all_results[exp_id] = {
                'group': group, 'name': name,
                'train_time': f'{train_min:.0f}', 'status': 'TRAIN_FAILED',
                'timestamp': ts,
            }
            save_csv(all_results)
            return

    if task in ('all', 'eval'):
        # —— 评估 ——
        # train.py 的 ckpt 目录 = cfg['output_folder'] / (config_name + '_' + output)
        # 这里 config_name = struct_<exp_id>, output = exp_id → ckpt/struct_<exp_id>_<exp_id>/
        ckpt_path = CKPT_DIR / f'struct_{exp_id}_{exp_id}'
        ok2, out_text, _ = run_cmd(
            [PYTHON, '-u', 'eval.py', str(exp_cfg_path), str(ckpt_path), '-p', '10'],
            f'eval {exp_id}', timeout_s=7200)

        maps = parse_maps(out_text) if ok2 else {}
        all_results[exp_id] = {
            'group': group, 'name': name,
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
        # task == 'train' only
        all_results[exp_id] = {
            'group': group, 'name': name,
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
        description='TriDet 全局改进实验 — SE / BiFPN / 回归损失')
    parser.add_argument('--group', type=str, default='',
                        help='按分组筛选: se, bifpn, reg_loss')
    parser.add_argument('--exp', type=str, default='',
                        help='指定实验编号, 逗号分隔 (如 SE-1,RL-2)')
    parser.add_argument('--start', type=str, default='',
                        help='从指定实验开始训练 (断点续跑)')
    parser.add_argument('--task', type=str, default='all',
                        choices=['all', 'train', 'eval'],
                        help='执行阶段: all(训+评), train(仅训练), eval(仅评估,需已有ckpt)')
    args = parser.parse_args()

    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    all_results = {}  # 默认不加载历史, 每次全量重跑
    ts = datetime.now().strftime('%Y-%m-%d %H:%M')

    # —— 筛选实验 ——
    if args.exp:
        exp_ids = [e.strip() for e in args.exp.split(',')]
        queue = [e for e in EXPERIMENTS if e['id'] in exp_ids]
        missing = [eid for eid in exp_ids if eid not in {e['id'] for e in EXPERIMENTS}]
        if missing:
            print(f"  [WARN] 未知实验: {', '.join(missing)}")
    elif args.group:
        grp = args.group.upper()
        if grp == 'REG_LOSS':
            grp = 'RegLoss'
        queue = [e for e in EXPERIMENTS if e['group'].upper() == grp.upper()]
        if not queue:
            print(f"  [ERROR] 未知分组: {args.group}, 可选: SE, BiFPN, RegLoss")
            return
    else:
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

    total = len(queue)
    if total == 0:
        print("  没有需要执行的实验!")
        return

    print(f"\n{'='*70}")
    print(f"  Section 2-4 结构改进实验 — {total} 项")
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

"""
TriDet 消融实验 — 统一运行器 (合并 run_ablation/run_all/run_training/train_all/do_all)

功能:
    Phase 1: 评估已有 checkpoint (baseline, A1, A2)
    Phase 2: 训练新实验并评估 → CSV

用法:
    python evaluate/run_ablation_experiments.py                    # 全部执行
    python evaluate/run_ablation_experiments.py --eval-only        # 仅评估已有 ckpt
    python evaluate/run_ablation_experiments.py --train-only       # 仅训练新实验
    python evaluate/run_ablation_experiments.py --exp A1,A2,A3_w3  # 仅指定实验
    python evaluate/run_ablation_experiments.py --all              # 全部 A1-A8
    python evaluate/run_ablation_experiments.py --baseline         # 仅训练基线
    python evaluate/run_ablation_experiments.py --start A5         # 从指定实验开始
"""

import os, sys, subprocess, time, argparse, tempfile
from datetime import datetime
from pathlib import Path

from evaluate.common import deep_set, dump_yaml, parse_maps, load_ablation_csv, save_ablation_csv

# ---- 路径配置 ----
REPO = Path(__file__).resolve().parent.parent
PYTHON = sys.executable
WORK_DIR = REPO / 'evaluate'
CKPT_DIR = REPO / 'ckpt'
CSV_PATH = WORK_DIR / 'ablation_results.csv'
BASE_CONFIG = REPO / 'configs/thumos_i3d.yaml'


# ============================================================
# 实验定义
# ============================================================

# 已有 checkpoint (仅评估, 不训练)
EXISTING_CKPTS = {
    'baseline': {
        'name': 'Baseline (SGP+Trident+DIoU)',
        'group': 'baseline',
        'config': 'configs/thumos_i3d.yaml',
        'ckpt': 'ckpt/thumos_i3d_baseline',
        'epochs': 40,
    },
    'A1': {
        'name': 'Trident-head→普通回归头',
        'group': 'A1',
        'config': 'evaluate/abl_A1.yaml',
        'ckpt': 'ckpt/abl_A1_A1',
        'epochs': 40,
    },
    'A2': {
        'name': 'SGP→Conv (20ep 下界估计)',
        'group': 'A2',
        'config': 'evaluate/abl_A2.yaml',
        'ckpt': 'ckpt/abl_A2_A2',
        'epochs': 20,
    },
}

# 基线等效实验 — 直接复用 baseline 结果
BASELINE_EQUIVALENTS = {
    'A3_w1':       ('A3', 'SGP 窗口 w=1 (基线)'),
    'A4_k5.0':     ('A4', 'k=5.0 (基线)'),
    'A6_identity': ('A6', 'FPN Identity (基线)'),
    'A7_r1.5':     ('A7', '中心采样 r=1.5 (基线)'),
    'A8_lw1.0':    ('A8', 'loss_weight=1.0 (基线)'),
}

# 训练队列 (按优先级排序)
TRAIN_QUEUE = [
    # --- A3: 窗口消融 ---
    ('A3_w3',  'SGP窗口 w=3',                 'A3', {'model.n_sgp_win_size': 3}),
    ('A3_w5',  'SGP窗口 w=5',                 'A3', {'model.n_sgp_win_size': 5}),
    ('A3_w7',  'SGP窗口 w=7',                 'A3', {'model.n_sgp_win_size': 7}),
    ('A3_w9',  'SGP窗口 w=9',                 'A3', {'model.n_sgp_win_size': 9}),
    ('A3_w11', 'SGP窗口 w=11',                'A3', {'model.n_sgp_win_size': 11}),
    ('A3_per_layer', 'SGP窗口 [1,3,5,7,9,11]','A3', {'model.n_sgp_win_size': [1,3,5,7,9,11]}),
    # --- A4: k 参数 ---
    ('A4_k1.0', 'k=1.0',  'A4', {'model.k': 1.0}),
    ('A4_k1.5', 'k=1.5',  'A4', {'model.k': 1.5}),
    ('A4_k3.0', 'k=3.0',  'A4', {'model.k': 3.0}),
    ('A4_k7.0', 'k=7.0',  'A4', {'model.k': 7.0}),
    # --- A5: DIoU→GIoU ---
    ('A5', 'DIoU→GIoU',    'A5', {'train_cfg.loss_type': 'giou'}),
    # --- A6: FPN ---
    ('A6_fpn', 'FPN 融合', 'A6', {'model.fpn_type': 'fpn'}),
    # --- A7: 中心采样 ---
    ('A7_r0.0', '采样半径 r=0.0', 'A7', {'train_cfg.center_sample_radius': 0.0}),
    ('A7_r0.5', '采样半径 r=0.5', 'A7', {'train_cfg.center_sample_radius': 0.5}),
    ('A7_r1.0', '采样半径 r=1.0', 'A7', {'train_cfg.center_sample_radius': 1.0}),
    ('A7_r2.0', '采样半径 r=2.0', 'A7', {'train_cfg.center_sample_radius': 2.0}),
    # --- A8: 损失权重 ---
    ('A8_lw0.5', '损失权重 lw=0.5', 'A8', {'train_cfg.loss_weight': 0.5}),
    ('A8_lw2.0', '损失权重 lw=2.0', 'A8', {'train_cfg.loss_weight': 2.0}),
    ('A8_lw5.0', '损失权重 lw=5.0', 'A8', {'train_cfg.loss_weight': 5.0}),
]

# 已知结果 (硬编码, 避免重复跑)
KNOWN_RESULTS = {
    'baseline': {'mAP@0.3': 75.1, 'mAP@0.5': 62.6, 'mAP@0.7': 38.3, 'avg_mAP': 59.31},
    'A1':       {'mAP@0.3': 74.8, 'mAP@0.5': 62.1, 'mAP@0.7': 36.4, 'avg_mAP': 58.38},
    'A2':       {'mAP@0.3': 61.7, 'mAP@0.5': 47.8, 'mAP@0.7': 21.7, 'avg_mAP': 44.46},
}


# ============================================================
# 工具函数
# ============================================================

def run_cmd(cmd, desc, timeout_s=3600):
    """运行命令, 输出重定向到临时文件避免 PIPE 缓冲区死锁.
    返回 (ok: bool, stdout: str, elapsed_min: float)."""
    print(f"\n  [{desc}]")
    sys.stdout.flush()
    t0 = time.time()

    out_f = tempfile.NamedTemporaryFile(
        mode='w+', suffix='.log', delete=False,
        dir=str(WORK_DIR), prefix=f'run_{desc.replace(" ","_")[:30]}_')
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
        # 读取已输出的内容
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

    # 打印最后几行
    lines = [x for x in out_text.strip().split('\n') if x.strip()]
    for l in lines[-5:]:
        print(f"    {l[:150]}")

    # 训练成功保留日志, eval 日志可清理
    if not ok or 'eval' in desc.lower():
        try:
            os.unlink(out_path)
        except Exception:
            pass
    else:
        print(f"    (log: {out_path})")

    return ok, out_text, el


# ============================================================
# CSV 管理
# ============================================================

def save_csv(all_results):
    """写入所有结果到 CSV."""
    save_ablation_csv(all_results, str(CSV_PATH))


def load_csv():
    """加载已有 CSV 结果, 返回 {exp_id: row_dict}."""
    return load_ablation_csv(str(CSV_PATH))


# ============================================================
# 核心逻辑
# ============================================================

def evaluate_existing(exp_id, info, all_results, ts):
    """评估已有 checkpoint."""
    name = info['name']
    group = info['group']
    cfg_path = REPO / info['config']
    ckpt_path = REPO / info['ckpt']
    epochs = info['epochs']

    # 检查是否已有结果
    if exp_id in all_results and all_results[exp_id].get('avg_mAP'):
        print(f"\n  [{exp_id}] 已有结果, 跳过")
        return

    print(f"\n  [{exp_id}] {name}  ({epochs}ep 权重)")

    if not ckpt_path.is_dir():
        print(f"  [SKIP] checkpoint 目录不存在: {ckpt_path}")
        all_results[exp_id] = {
            'group': group, 'name': name,
            'train_time': '', 'status': 'CKPT_MISSING', 'timestamp': ts,
        }
        save_csv(all_results)
        return

    ok, out_text, el = run_cmd(
        [PYTHON, '-u', 'eval.py', str(cfg_path), str(ckpt_path), '-p', '10'],
        f'eval {exp_id}', timeout_s=7200)

    maps = parse_maps(out_text) if ok else {}
    if maps:
        all_results[exp_id] = {
            'group': group, 'name': name,
            'mAP@0.3': maps.get(0.3), 'mAP@0.5': maps.get(0.5),
            'mAP@0.7': maps.get(0.7), 'avg_mAP': maps.get('avg'),
            'train_time': f'{epochs}ep (已有)', 'status': 'OK', 'timestamp': ts,
        }
        print(f"  ★ mAP@0.3={maps.get(0.3, '?'):.1f}%  mAP@0.5={maps.get(0.5, '?'):.1f}%  "
              f"mAP@0.7={maps.get(0.7, '?'):.1f}%  avg={maps.get('avg', '?'):.2f}%")
    else:
        all_results[exp_id] = {
            'group': group, 'name': name,
            'train_time': '', 'status': 'EVAL_FAILED', 'timestamp': ts,
        }
    save_csv(all_results)


def train_and_evaluate(exp_id, name, group, overrides, all_results, ts):
    """训练 + 评估单个实验."""
    print(f"\n{'='*55}")
    print(f"  {exp_id}: {name}  [{group}]")
    print(f"{'='*55}")

    # 生成配置
    sys.path.insert(0, str(REPO))
    from libs.core.config import load_config as _load_cfg
    cfg = _load_cfg(str(BASE_CONFIG))
    for key_path, val in overrides.items():
        deep_set(cfg, key_path, val)
    exp_config = str(WORK_DIR / f'abl_{exp_id}.yaml')
    dump_yaml(cfg, exp_config)
    print(f"  配置: {exp_config}")

    # 训练
    ok, _, train_min = run_cmd(
        [PYTHON, '-u', 'train.py', exp_config, '--output', exp_id],
        f'train {exp_id}', timeout_s=14400)  # 4h max

    if not ok:
        all_results[exp_id] = {
            'group': group, 'name': name,
            'train_time': f'{train_min:.0f}', 'status': 'TRAIN_FAILED',
            'timestamp': ts,
        }
        save_csv(all_results)
        return

    # 评估
    ckpt_path = str(CKPT_DIR / f'abl_{exp_id}_{exp_id}')
    ok2, out_text, _ = run_cmd(
        [PYTHON, '-u', 'eval.py', exp_config, ckpt_path, '-p', '10'],
        f'eval {exp_id}', timeout_s=7200)

    maps = parse_maps(out_text) if ok2 else {}
    all_results[exp_id] = {
        'group': group, 'name': name,
        'mAP@0.3': maps.get(0.3), 'mAP@0.5': maps.get(0.5),
        'mAP@0.7': maps.get(0.7), 'avg_mAP': maps.get('avg'),
        'train_time': f'{train_min:.0f}', 'status': 'OK' if maps else 'EVAL_FAILED',
        'timestamp': ts,
    }
    save_csv(all_results)

    if maps:
        print(f"  ★ [{exp_id}] avg_mAP={maps.get('avg', '?'):.2f}%")


def apply_known_results(all_results, ts):
    """将硬编码的已知结果写入 all_results (如果尚不存在)."""
    for exp_id, known in KNOWN_RESULTS.items():
        if exp_id not in all_results or not all_results[exp_id].get('avg_mAP'):
            info = EXISTING_CKPTS.get(exp_id, {})
            all_results[exp_id] = {
                'group': info.get('group', exp_id),
                'name': info.get('name', exp_id),
                'mAP@0.3': known['mAP@0.3'],
                'mAP@0.5': known['mAP@0.5'],
                'mAP@0.7': known['mAP@0.7'],
                'avg_mAP': known['avg_mAP'],
                'train_time': f'{info.get("epochs", "?")}ep (已知)',
                'status': 'KNOWN',
                'timestamp': ts,
            }
    save_csv(all_results)


def apply_baseline_equivalents(all_results, ts):
    """将基线结果复制到等效实验."""
    bl = all_results.get('baseline', {})
    if not bl or bl.get('status') not in ('OK', 'KNOWN'):
        return
    for exp_id, (group, name) in BASELINE_EQUIVALENTS.items():
        if exp_id not in all_results:
            all_results[exp_id] = {
                'group': group, 'name': name,
                'mAP@0.3': bl.get('mAP@0.3'),
                'mAP@0.5': bl.get('mAP@0.5'),
                'mAP@0.7': bl.get('mAP@0.7'),
                'avg_mAP': bl.get('avg_mAP'),
                'train_time': '(复用基线)',
                'status': 'BASELINE_EQ',
                'timestamp': ts,
            }
    save_csv(all_results)


# ============================================================
# 主入口
# ============================================================

def main():
    parser = argparse.ArgumentParser(description='TriDet 消融实验统一运行器')
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--eval-only', action='store_true', help='仅评估已有 checkpoint')
    mode.add_argument('--train-only', action='store_true', help='仅训练新实验 (跳过已有 ckpt 评估)')
    mode.add_argument('--all', action='store_true', help='运行全部 A1-A8 消融实验')
    mode.add_argument('--baseline', action='store_true', help='仅评估基线模型')
    parser.add_argument('--exp', type=str, default='', help='指定实验编号, 逗号分隔 (如 A1,A2,A3_w3)')
    parser.add_argument('--start', type=str, default='', help='从指定实验开始训练 (断点续跑)')
    parser.add_argument('--skip-known', action='store_true', help='跳过硬编码已知结果, 强制重新评估')
    args = parser.parse_args()

    WORK_DIR.mkdir(exist_ok=True)
    all_results = load_csv()
    ts = datetime.now().strftime('%Y-%m-%d %H:%M')

    # ── 模式: 仅基线 ──
    if args.baseline:
        evaluate_existing('baseline', EXISTING_CKPTS['baseline'], all_results, ts)
        print(f"\n  结果: {CSV_PATH}")
        return

    # ── 模式: 指定实验 ──
    if args.exp:
        exp_ids = [e.strip() for e in args.exp.split(',')]
        for eid in exp_ids:
            if eid in EXISTING_CKPTS:
                evaluate_existing(eid, EXISTING_CKPTS[eid], all_results, ts)
            else:
                # 在训练队列中查找
                found = False
                for tid, tname, tgroup, toverrides in TRAIN_QUEUE:
                    if tid == eid:
                        found = True
                        # 检查是否已有结果
                        if eid in all_results and all_results[eid].get('status') == 'OK':
                            print(f"\n  [{eid}] 已有完成结果, 跳过 (用 --skip-known 强制重新运行)")
                            continue
                        train_and_evaluate(tid, tname, tgroup, toverrides, all_results, ts)
                        break
                if not found:
                    print(f"  [WARN] 未知实验: {eid}")
        print(f"\n  结果: {CSV_PATH}")
        return

    # ── 默认 / --all / --eval-only / --train-only 模式 ──

    # Phase 1: 评估已有 checkpoint
    if not args.train_only:
        print("=" * 70)
        print("  Phase 1: 评估已有 checkpoint (baseline, A1, A2)")
        print("=" * 70)

        if not args.skip_known:
            apply_known_results(all_results, ts)

        for exp_id, info in EXISTING_CKPTS.items():
            evaluate_existing(exp_id, info, all_results, ts)

        apply_baseline_equivalents(all_results, ts)
        print(f"\n  Phase 1 完成。共 {len(all_results)} 项结果。")

    if args.eval_only:
        print(f"\n  结果 → {CSV_PATH}")
        print(f"  运行分析: python evaluate/analyze.py")
        return

    # Phase 2: 训练队列
    need_train = []
    skip_flag = bool(args.start)
    for tid, tname, tgroup, toverrides in TRAIN_QUEUE:
        if skip_flag and tid != args.start:
            continue
        skip_flag = False
        if tid in all_results and all_results[tid].get('status') in ('OK', 'KNOWN'):
            print(f"  [SKIP] {tid} 已有完成结果")
            continue
        need_train.append((tid, tname, tgroup, toverrides))

    total = len(need_train)
    if total == 0:
        print("\n  所有实验已完成!")
        print(f"  结果: {CSV_PATH}")
        return

    print(f"\n{'='*70}")
    print(f"  Phase 2: 训练队列 — {total} 项 (约 {total}h)")
    print(f"{'='*70}")

    for i, (tid, tname, tgroup, toverrides) in enumerate(need_train):
        print(f"\n  >>> [{i+1}/{total}] {tid}: {tname} <<<")
        train_and_evaluate(tid, tname, tgroup, toverrides, all_results, ts)

    print(f"\n{'='*70}")
    print(f"  全部完成! {len(all_results)} results → {CSV_PATH}")
    print(f"  运行分析: python evaluate/analyze.py")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()

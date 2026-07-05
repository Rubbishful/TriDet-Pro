"""
TriDet 消融实验 — 统一运行器

功能:
    Phase 1: 评估基线 checkpoint (已有)
    Phase 2: 训练全部 A1-A8 消融实验 → 评估 → CSV

用法:
    python evaluate/run_ablation_experiments.py                     # 全部执行
    python evaluate/run_ablation_experiments.py --baseline          # 仅评估基线
    python evaluate/run_ablation_experiments.py --train-only        # 仅训练 (跳过基线)
    python evaluate/run_ablation_experiments.py --exp A1,A2,A5      # 仅指定实验
    python evaluate/run_ablation_experiments.py --start A5          # 断点续跑

路径约定:
    数据:  data/thumos/i3d_features/
    模型:  ckpt/<exp_dir>/
    输出:  evaluate/results/
"""

import os, sys, subprocess, time, argparse, tempfile
from datetime import datetime
from pathlib import Path

from evaluate.common import deep_set, dump_yaml, parse_maps, load_ablation_csv, save_ablation_csv

# ---- 路径配置 ----
REPO        = Path(__file__).resolve().parent.parent
PYTHON      = sys.executable
RESULT_DIR  = REPO / 'evaluate' / 'results'
CKPT_DIR    = REPO / 'ckpt'
CSV_PATH    = RESULT_DIR / 'ablation_results.csv'
BASE_CONFIG = REPO / 'configs' / 'thumos_i3d.yaml'


# ============================================================
# 实验定义
# ============================================================

# 基线 (已有 checkpoint, 仅评估)
BASELINE = {
    'name': 'Baseline (SGP+Trident+DIoU)',
    'group': 'baseline',
    'config': 'configs/thumos_i3d.yaml',
    'ckpt': 'ckpt/thumos_i3d_baseline',
}

# 基线等效实验 — 复用在 Phase 1 中测出的 baseline 结果
BASELINE_EQUIVALENTS = {
    'A3_w1':       ('A3', 'SGP 窗口 w=1 (基线)'),
    'A4_k5.0':     ('A4', 'k=5.0 (基线)'),
    'A6_identity': ('A6', 'FPN Identity (基线)'),
    'A7_r1.5':     ('A7', '中心采样 r=1.5 (基线)'),
    'A8_lw1.0':    ('A8', 'loss_weight=1.0 (基线)'),
}

# 训练队列 — 全部消融实验, 按优先级排序
TRAIN_QUEUE = [
    # --- A1: Trident-head → 普通回归头 ---
    ('A1', 'Trident-head→普通回归头', 'A1', {'model.use_trident_head': False}),
    # --- A2: SGP → Conv Backbone ---
    ('A2', 'SGP→Conv Backbone', 'A2', {'model.backbone_type': 'conv'}),
    # --- A3: SGP 窗口消融 ---
    ('A3_w3',  'SGP窗口 w=3',                 'A3', {'model.n_sgp_win_size': 3}),
    ('A3_w5',  'SGP窗口 w=5',                 'A3', {'model.n_sgp_win_size': 5}),
    ('A3_w7',  'SGP窗口 w=7',                 'A3', {'model.n_sgp_win_size': 7}),
    ('A3_w9',  'SGP窗口 w=9',                 'A3', {'model.n_sgp_win_size': 9}),
    ('A3_w11', 'SGP窗口 w=11',                'A3', {'model.n_sgp_win_size': 11}),
    ('A3_per_layer', 'SGP窗口 [1,3,5,7,9,11]','A3', {'model.n_sgp_win_size': [1,3,5,7,9,11]}),
    # --- A4: k 参数消融 ---
    ('A4_k1.0', 'k=1.0',  'A4', {'model.k': 1.0}),
    ('A4_k1.5', 'k=1.5',  'A4', {'model.k': 1.5}),
    ('A4_k3.0', 'k=3.0',  'A4', {'model.k': 3.0}),
    ('A4_k7.0', 'k=7.0',  'A4', {'model.k': 7.0}),
    # --- A5: DIoU → GIoU ---
    ('A5', 'DIoU→GIoU 损失',  'A5', {'train_cfg.loss_type': 'giou'}),
    # --- A6: FPN vs Identity ---
    ('A6_fpn', 'FPN 融合',   'A6', {'model.fpn_type': 'fpn'}),
    # --- A7: 中心采样半径 ---
    ('A7_r0.0', '采样半径 r=0.0', 'A7', {'train_cfg.center_sample_radius': 0.0}),
    ('A7_r0.5', '采样半径 r=0.5', 'A7', {'train_cfg.center_sample_radius': 0.5}),
    ('A7_r1.0', '采样半径 r=1.0', 'A7', {'train_cfg.center_sample_radius': 1.0}),
    ('A7_r2.0', '采样半径 r=2.0', 'A7', {'train_cfg.center_sample_radius': 2.0}),
    # --- A8: 分类损失权重 ---
    ('A8_lw0.5', '损失权重 lw=0.5', 'A8', {'train_cfg.loss_weight': 0.5}),
    ('A8_lw2.0', '损失权重 lw=2.0', 'A8', {'train_cfg.loss_weight': 2.0}),
    ('A8_lw5.0', '损失权重 lw=5.0', 'A8', {'train_cfg.loss_weight': 5.0}),
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
        dir=str(RESULT_DIR), prefix=f'run_')
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
    """写入 CSV 到 evaluate/results/ablation_results.csv."""
    save_ablation_csv(all_results, str(CSV_PATH))


def load_csv():
    """加载已有 CSV 结果."""
    return load_ablation_csv(str(CSV_PATH))


# ============================================================
# 核心逻辑
# ============================================================

def evaluate_baseline(all_results, ts):
    """评估 baseline checkpoint (已有, 仅评估)."""
    exp_id = 'baseline'
    name = BASELINE['name']
    group = BASELINE['group']
    cfg_path = REPO / BASELINE['config']
    ckpt_path = REPO / BASELINE['ckpt']

    if exp_id in all_results and all_results[exp_id].get('status') == 'OK':
        print(f"\n  [baseline] 已有完成结果, 跳过")
        return True

    print(f"\n  [baseline] {name}")

    if not ckpt_path.is_dir():
        print(f"  [ERROR] checkpoint 目录不存在: {ckpt_path}")
        return False

    ok, out_text, el = run_cmd(
        [PYTHON, '-u', 'eval.py', str(cfg_path), str(ckpt_path), '-p', '10'],
        'eval baseline', timeout_s=7200)

    maps = parse_maps(out_text) if ok else {}
    if maps:
        all_results[exp_id] = {
            'group': group, 'name': name,
            'mAP@0.3': maps.get(0.3), 'mAP@0.5': maps.get(0.5),
            'mAP@0.7': maps.get(0.7), 'avg_mAP': maps.get('avg'),
            'train_time': '(已有 ckpt)', 'status': 'OK', 'timestamp': ts,
        }
        print(f"  ★ mAP@0.3={maps.get(0.3, '?'):.1f}%  "
              f"mAP@0.5={maps.get(0.5, '?'):.1f}%  "
              f"mAP@0.7={maps.get(0.7, '?'):.1f}%  "
              f"avg={maps.get('avg', '?'):.2f}%")
        return True
    else:
        all_results[exp_id] = {
            'group': group, 'name': name,
            'train_time': '', 'status': 'EVAL_FAILED', 'timestamp': ts,
        }
        return False


def apply_baseline_equivalents(all_results, ts):
    """将基线结果复制到参数等效的实验."""
    bl = all_results.get('baseline', {})
    if not bl or bl.get('status') != 'OK':
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


def train_and_evaluate(exp_id, name, group, overrides, all_results, ts):
    """训练 → 评估单个消融实验."""
    print(f"\n{'='*55}")
    print(f"  {exp_id}: {name}  [{group}]")
    print(f"{'='*55}")

    # 跳过已完成项
    if exp_id in all_results and all_results[exp_id].get('status') == 'OK':
        print(f"  [SKIP] 已有完成结果")
        return

    # 生成配置
    sys.path.insert(0, str(REPO))
    from libs.core.config import load_config as _load_cfg
    cfg = _load_cfg(str(BASE_CONFIG))
    for key_path, val in overrides.items():
        deep_set(cfg, key_path, val)

    exp_cfg_path = RESULT_DIR / f'abl_{exp_id}.yaml'
    dump_yaml(cfg, str(exp_cfg_path))
    print(f"  配置: {exp_cfg_path}")

    # 训练
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

    # train.py 的 ckpt 目录 = cfg['output_folder'] / (config_name + '_' + output)
    # 即 ckpt/abl_<exp_id>_<exp_id>/
    ckpt_path = CKPT_DIR / f'abl_{exp_id}_{exp_id}'
    ok2, out_text, _ = run_cmd(
        [PYTHON, '-u', 'eval.py', str(exp_cfg_path), str(ckpt_path), '-p', '10'],
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


# ============================================================
# 主入口
# ============================================================

def main():
    parser = argparse.ArgumentParser(description='TriDet 消融实验统一运行器')
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--baseline', action='store_true', help='仅评估基线模型')
    mode.add_argument('--train-only', action='store_true', help='仅训练消融实验 (跳过基线评估)')
    parser.add_argument('--exp', type=str, default='',
                        help='指定实验编号, 逗号分隔 (如 A1,A2,A5)')
    parser.add_argument('--start', type=str, default='',
                        help='从指定实验开始训练 (断点续跑)')
    parser.add_argument('--force', action='store_true',
                        help='强制重新训练已在 CSV 中有结果的项目')
    args = parser.parse_args()

    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    all_results = load_csv()
    ts = datetime.now().strftime('%Y-%m-%d %H:%M')

    # ── 模式: 仅基线 ──
    if args.baseline:
        evaluate_baseline(all_results, ts)
        save_csv(all_results)
        apply_baseline_equivalents(all_results, ts)
        print(f"\n  结果: {CSV_PATH}")
        return

    # ── 模式: 指定实验 ──
    if args.exp:
        exp_ids = [e.strip() for e in args.exp.split(',')]
        for eid in exp_ids:
            if eid == 'baseline':
                evaluate_baseline(all_results, ts)
                save_csv(all_results)
                apply_baseline_equivalents(all_results, ts)
                continue

            found = False
            for tid, tname, tgroup, toverrides in TRAIN_QUEUE:
                if tid == eid:
                    found = True
                    if not args.force and tid in all_results and all_results[tid].get('status') == 'OK':
                        print(f"\n  [{tid}] 已有完成结果, 跳过 (用 --force 强制重跑)")
                        continue
                    train_and_evaluate(tid, tname, tgroup, toverrides, all_results, ts)
                    break
            if not found:
                print(f"  [WARN] 未知实验: {eid}")
        print(f"\n  结果: {CSV_PATH}")
        return

    # ── 默认: Phase 1 (基线) + Phase 2 (全部消融) ──

    # Phase 1: 评估基线
    if not args.train_only:
        print("=" * 70)
        print("  Phase 1: 评估基线 (已有 checkpoint)")
        print("  ckpt: ckpt/thumos_i3d_baseline/")
        print("=" * 70)

        evaluate_baseline(all_results, ts)
        save_csv(all_results)
        apply_baseline_equivalents(all_results, ts)
        print(f"\n  Phase 1 完成。共 {len(all_results)} 项结果。")

    # Phase 2: 训练消融实验
    need_train = []
    skip_flag = bool(args.start)
    for tid, tname, tgroup, toverrides in TRAIN_QUEUE:
        if skip_flag and tid != args.start:
            continue
        skip_flag = False
        if not args.force and tid in all_results and all_results[tid].get('status') == 'OK':
            print(f"  [SKIP] {tid} 已有完成结果")
            continue
        need_train.append((tid, tname, tgroup, toverrides))

    total = len(need_train)
    if total == 0:
        print("\n  所有实验已完成!")
        print(f"  结果: {CSV_PATH}")
        return

    print(f"\n{'='*70}")
    print(f"  Phase 2: 消融训练 — {total} 项 (~{total}h)")
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

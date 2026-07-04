"""
TriDet 消融实验 — 完整自动化 (A1-A8)。

1. 评估已有 checkpoint (baseline, A1, A2)
2. 训练新实验并评估
3. 生成 CSV 结果

用法:
    python work_1/run_all.py                      # 全部执行
    python work_1/run_all.py --eval-only          # 仅评估已有ckpt
    python work_1/run_all.py --train-only         # 仅训练新实验(跳过已有ckpt)
    python work_1/run_all.py --start A5           # 从指定实验开始训练
"""

import os, sys, subprocess, csv, yaml, re, time, argparse
from datetime import datetime

REPO = 'e:/Tridet/TriDet-Pro'
PYTHON = 'E:/anaconda/envs/test/python.exe'
WORK_DIR = os.path.join(REPO, 'work_1')
CKPT_DIR = os.path.join(REPO, 'ckpt')
CSV_PATH = os.path.join(WORK_DIR, 'ablation_results.csv')
BASE_CONFIG = os.path.join(REPO, 'configs/thumos_i3d.yaml')

# =====================================================
# 已有 checkpoint (仅评估, 不训练)
# =====================================================
EXISTING_CKPTS = {
    'baseline': ('configs/thumos_i3d.yaml', 'ckpt/thumos_i3d_baseline', 'Baseline (SGP+Trident+DIoU)', 'baseline', 40),
    'A1':       ('work/abl_A1.yaml',        'ckpt/abl_A1_A1',             'Trident-head→普通回归头',       'A1', 40),
    'A2':       ('work/abl_A2.yaml',        'ckpt/abl_A2_A2',             'SGP→Conv (20ep 下界估计)',        'A2', 20),
}

# 基线等效实验 (复用 baseline 结果)
BASELINE_EQ = {
    'A3_w1':       ('A3', 'SGP 窗口 w=1 (基线)'),
    'A4_k5.0':     ('A4', 'k=5.0 (基线)'),
    'A6_identity': ('A6', 'FPN Identity (基线)'),
    'A7_r1.5':     ('A7', '中心采样 r=1.5 (基线)'),
    'A8_lw1.0':    ('A8', 'loss_weight=1.0 (基线)'),
}

# =====================================================
# 训练队列 (按优先级)
# =====================================================
TRAIN_QUEUE = [
    # (exp_id, name, group, overrides_dict)
    # --- A3: 窗口消融 ---
    ('A3_w3',  '窗口 w=3',                'A3', {'model.n_sgp_win_size': 3}),
    ('A3_w5',  '窗口 w=5',                'A3', {'model.n_sgp_win_size': 5}),
    ('A3_w7',  '窗口 w=7',                'A3', {'model.n_sgp_win_size': 7}),
    ('A3_w9',  '窗口 w=9',                'A3', {'model.n_sgp_win_size': 9}),
    ('A3_w11', '窗口 w=11',               'A3', {'model.n_sgp_win_size': 11}),
    ('A3_per_layer', '窗口逐层[1,3,5,7,9,11]','A3', {'model.n_sgp_win_size': [1,3,5,7,9,11]}),
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
    ('A7_r0.0', 'r=0.0',   'A7', {'train_cfg.center_sample_radius': 0.0}),
    ('A7_r0.5', 'r=0.5',   'A7', {'train_cfg.center_sample_radius': 0.5}),
    ('A7_r1.0', 'r=1.0',   'A7', {'train_cfg.center_sample_radius': 1.0}),
    ('A7_r2.0', 'r=2.0',   'A7', {'train_cfg.center_sample_radius': 2.0}),
    # --- A8: 损失权重 ---
    ('A8_lw0.5', 'lw=0.5', 'A8', {'train_cfg.loss_weight': 0.5}),
    ('A8_lw2.0', 'lw=2.0', 'A8', {'train_cfg.loss_weight': 2.0}),
    ('A8_lw5.0', 'lw=5.0', 'A8', {'train_cfg.loss_weight': 5.0}),
]


def deep_set(d, key_path, value):
    keys = key_path.split('.')
    for k in keys[:-1]:
        d = d[k]
    d[keys[-1]] = value


def dump_yaml(cfg, path):
    class Dumper(yaml.Dumper): pass
    def _list_repr(dumper, data):
        if len(data) <= 6:
            return dumper.represent_sequence('tag:yaml.org,2002:seq', data, flow_style=True)
        return dumper.represent_sequence('tag:yaml.org,2002:seq', data, flow_style=False)
    Dumper.add_representer(list, _list_repr)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        yaml.dump(cfg, f, Dumper=Dumper, default_flow_style=False, sort_keys=False, allow_unicode=True)


def parse_map(output_text):
    """从 eval.py 输出解析 mAP 值。"""
    maps = {}
    for line in output_text.split('\n'):
        line = line.strip()
        # 格式: "tIoU = 0.30:   mAP = 75.10 (%)"
        if 'tIoU =' in line and 'mAP' in line:
            try:
                tiou_str = re.search(r'tIoU\s*=\s*([\d.]+)', line)
                map_str = re.search(r'mAP\s*=\s*([\d.]+)', line)
                if tiou_str and map_str:
                    tiou = float(tiou_str.group(1))
                    mAP = float(map_str.group(1))
                    maps[tiou] = mAP
            except (ValueError, AttributeError):
                pass
        # 格式: "Average mAP: 59.31 (%)"
        if 'Average mAP' in line or 'Avearge mAP' in line:
            try:
                avg = re.search(r'[\d.]+', line.split(':')[1])
                if avg:
                    maps['avg'] = float(avg.group())
            except (ValueError, IndexError):
                pass
    return maps


def run_eval(config_path, ckpt_folder):
    """运行 eval.py, 解析并返回 mAP。"""
    print(f"  [EVAL] {ckpt_folder}")
    t0 = time.time()
    try:
        proc = subprocess.run(
            [PYTHON, '-u', 'eval.py', config_path, ckpt_folder],
            cwd=REPO, capture_output=True, text=True, timeout=1800
        )
    except subprocess.TimeoutExpired:
        print(f"  [TIMEOUT] eval 超时(30min)")
        return None

    elapsed = time.time() - t0
    print(f"  eval 耗时: {elapsed:.0f}s, rc={proc.returncode}")

    if proc.returncode != 0:
        print(f"  [FAIL] eval stderr: {proc.stderr[-300:]}")
        return None

    maps = parse_map(proc.stdout)
    if maps:
        print(f"  ★ mAP@0.3={maps.get(0.3, '?'):.2f}%  mAP@0.5={maps.get(0.5, '?'):.2f}%  mAP@0.7={maps.get(0.7, '?'):.2f}%  avg={maps.get('avg', '?'):.2f}%")
    else:
        print(f"  [WARN] 未能解析 mAP, 输出最后10行:")
        for line in proc.stdout.strip().split('\n')[-10:]:
            print(f"    {line}")
    return maps


def train_experiment(exp_id, config_path):
    """训练实验。"""
    print(f"  [TRAIN] {exp_id}")
    t0 = time.time()
    try:
        proc = subprocess.run(
            [PYTHON, '-u', 'train.py', config_path, '--output', exp_id],
            cwd=REPO, capture_output=True, text=True, timeout=14400  # 4h max
        )
    except subprocess.TimeoutExpired:
        elapsed = (time.time() - t0) / 60
        print(f"  [TIMEOUT] 训练超时(4h), {elapsed:.0f}min")
        return False, elapsed

    elapsed = (time.time() - t0) / 60
    ok = proc.returncode == 0
    print(f"  训练耗时: {elapsed:.0f}min, {'OK' if ok else 'FAIL'}")
    if not ok:
        print(f"  stderr: {proc.stderr[-500:]}")
    return ok, elapsed


# =====================================================
# CSV 管理
# =====================================================
def fmt(v):
    """安全格式化数值"""
    if v is None:
        return ''
    return f'{v:.2f}' if isinstance(v, (int, float)) else str(v)


def load_csv():
    if not os.path.exists(CSV_PATH):
        return {}
    with open(CSV_PATH, 'r', encoding='utf-8') as f:
        return {row['实验编号'].strip(): row for row in csv.DictReader(f)}


def save_csv(all_results):
    os.makedirs(WORK_DIR, exist_ok=True)
    with open(CSV_PATH, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['实验编号','分组','配置变更','mAP@0.3','mAP@0.5',
                    'mAP@0.7','avg_mAP','训练时间(min)','状态','时间戳'])
        for exp_id in sorted(all_results.keys()):
            r = all_results[exp_id]
            w.writerow([exp_id, r.get('group',''), r.get('name',''),
                       fmt(r.get('mAP@0.3')), fmt(r.get('mAP@0.5')),
                       fmt(r.get('mAP@0.7')), fmt(r.get('avg_mAP')),
                       r.get('train_time',''), r.get('status',''),
                       r.get('timestamp','')])


# =====================================================
# 主流程
# =====================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--eval-only', action='store_true')
    parser.add_argument('--train-only', action='store_true')
    parser.add_argument('--start', type=str, default='')
    args = parser.parse_args()

    os.makedirs(WORK_DIR, exist_ok=True)
    all_results = load_csv()
    ts = datetime.now().strftime('%Y-%m-%d %H:%M')

    # ============================================
    # Phase 1: 评估已有 checkpoint
    # ============================================
    if not args.train_only:
        print("\n" + "="*70)
        print("  Phase 1: 评估已有 checkpoint")
        print("="*70)

        for exp_id, (cfg_path, ckpt, name, group, trained_ep) in EXISTING_CKPTS.items():
            if exp_id in all_results and all_results[exp_id].get('avg_mAP'):
                print(f"\n  [{exp_id}] 已有结果, 跳过")
                continue

            print(f"\n  [{exp_id}] {name}  ({trained_ep}ep 权重)")
            maps = run_eval(os.path.join(REPO, cfg_path), os.path.join(REPO, ckpt))
            if maps:
                all_results[exp_id] = {
                    'exp_id': exp_id, 'group': group, 'name': name,
                    'mAP@0.3': maps.get(0.3), 'mAP@0.5': maps.get(0.5),
                    'mAP@0.7': maps.get(0.7), 'avg_mAP': maps.get('avg'),
                    'train_time': f'{trained_ep}ep (已有)', 'status': 'OK',
                    'timestamp': ts,
                }
            else:
                all_results[exp_id] = {
                    'exp_id': exp_id, 'group': group, 'name': name,
                    'train_time': '', 'status': 'EVAL_FAILED', 'timestamp': ts,
                }
            save_csv(all_results)

        # 复制 baseline 结果到等效实验
        bl = all_results.get('baseline', {})
        if bl and bl.get('status') == 'OK':
            for exp_id, (group, name) in BASELINE_EQ.items():
                all_results[exp_id] = {
                    'exp_id': exp_id, 'group': group, 'name': name,
                    'mAP@0.3': bl.get('mAP@0.3'), 'mAP@0.5': bl.get('mAP@0.5'),
                    'mAP@0.7': bl.get('mAP@0.7'), 'avg_mAP': bl.get('avg_mAP'),
                    'train_time': '(复用基线)', 'status': 'BASELINE_EQ',
                    'timestamp': ts,
                }
            save_csv(all_results)

        print(f"\n  Phase 1 完成。共 {len(all_results)} 项结果。")

    if args.eval_only:
        print(f"\n  [DONE] 仅评估模式。结果: {CSV_PATH}")
        return

    # ============================================
    # Phase 2: 训练队列
    # ============================================
    need_train = []
    for exp_id, name, group, overrides in TRAIN_QUEUE:
        if exp_id in all_results and all_results[exp_id].get('status') == 'OK':
            print(f"  [SKIP] {exp_id} 已有完成结果")
            continue
        need_train.append((exp_id, name, group, overrides))

    total = len(need_train)
    print(f"\n{'='*70}")
    print(f"  Phase 2: 训练队列 — {total} 项 (约 {total}h)")
    print(f"{'='*70}")

    skip_flag = bool(args.start)
    for i, (exp_id, name, group, overrides) in enumerate(need_train):
        if skip_flag:
            if exp_id == args.start:
                skip_flag = False
            else:
                continue

        print(f"\n{'='*60}")
        print(f"  [{i+1}/{total}] {exp_id}: {name}  [{group}]")
        print(f"{'='*60}")

        # 1. 生成配置
        from libs.core import load_config as _load
        cfg = _load(BASE_CONFIG)
        for key_path, val in overrides.items():
            deep_set(cfg, key_path, val)
        exp_config = os.path.join(WORK_DIR, f'abl_{exp_id}.yaml')
        dump_yaml(cfg, exp_config)
        print(f"  配置: {exp_config}")

        # 2. 训练
        ok, train_time = train_experiment(exp_id, exp_config)
        if not ok:
            all_results[exp_id] = {
                'exp_id': exp_id, 'group': group, 'name': name,
                'train_time': f'{train_time:.0f}', 'status': 'TRAIN_FAILED',
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M'),
            }
            save_csv(all_results)
            print(f"  [FAIL] {exp_id} 训练失败, 继续下一个")
            continue

        # 3. 评估
        ckpt_folder = os.path.join(CKPT_DIR, f'abl_{exp_id}_{exp_id}')
        maps = run_eval(exp_config, ckpt_folder)

        result = {
            'exp_id': exp_id, 'group': group, 'name': name,
            'train_time': f'{train_time:.0f}',
            'status': 'OK' if maps else 'EVAL_FAILED',
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M'),
        }
        if maps:
            result.update({
                'mAP@0.3': maps.get(0.3), 'mAP@0.5': maps.get(0.5),
                'mAP@0.7': maps.get(0.7), 'avg_mAP': maps.get('avg'),
            })
        all_results[exp_id] = result
        save_csv(all_results)

    # ============================================
    # Done
    # ============================================
    print(f"\n{'='*70}")
    print(f"  全部完成! {len(all_results)} results → {CSV_PATH}")
    print(f"  运行分析: python work_1/analyze.py")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()

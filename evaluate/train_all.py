"""
批量训练所有消融实验的脚本。按优先级顺序运行。

用法:
    python work_1/train_all.py
    python work_1/train_all.py --start A3_w3    # 从特定实验开始
"""

import os, sys, subprocess, csv, time, argparse
from datetime import datetime

REPO = 'e:/Tridet/TriDet-Pro'
PYTHON = 'E:/anaconda/envs/test/python.exe'
WORK_DIR = os.path.join(REPO, 'work_1')

# =====================================================
# 实验队列 (按优先级排序, 跳过与基线相同的实验)
# =====================================================
# A3_w1, A4_k5.0, A6_identity, A7_r1.5, A8_lw1.0 与基线相同, 直接复用基线结果
# A1 已有完整 checkpoint, 直接 eval (单独处理)

EXPERIMENTS = {
    # --- 核心对比 ---
    'A2':  {'name': 'SGP→Conv',                  'group': 'A2', 'overrides': {'model.backbone_type': 'conv'}},
    # --- A3: 窗口消融 (跳过 w=1 因为它=基线) ---
    'A3_w3':  {'name': '窗口 w=3',              'group': 'A3', 'overrides': {'model.n_sgp_win_size': 3}},
    'A3_w5':  {'name': '窗口 w=5',              'group': 'A3', 'overrides': {'model.n_sgp_win_size': 5}},
    'A3_w7':  {'name': '窗口 w=7',              'group': 'A3', 'overrides': {'model.n_sgp_win_size': 7}},
    'A3_w9':  {'name': '窗口 w=9',              'group': 'A3', 'overrides': {'model.n_sgp_win_size': 9}},
    'A3_w11': {'name': '窗口 w=11',             'group': 'A3', 'overrides': {'model.n_sgp_win_size': 11}},
    'A3_per_layer': {'name': '窗口[1,3,5,7,9,11]', 'group': 'A3', 'overrides': {'model.n_sgp_win_size': [1,3,5,7,9,11]}},
    # --- A4: k参数消融 (跳过 k=5.0 因为它=基线) ---
    'A4_k1.0': {'name': 'k=1.0',               'group': 'A4', 'overrides': {'model.k': 1.0}},
    'A4_k1.5': {'name': 'k=1.5',               'group': 'A4', 'overrides': {'model.k': 1.5}},
    'A4_k3.0': {'name': 'k=3.0',               'group': 'A4', 'overrides': {'model.k': 3.0}},
    'A4_k7.0': {'name': 'k=7.0',               'group': 'A4', 'overrides': {'model.k': 7.0}},
    # --- A5: DIoU→GIoU ---
    'A5':  {'name': 'DIoU→GIoU',               'group': 'A5', 'overrides': {'train_cfg.loss_type': 'giou'}},
    # --- A6: FPN对比 (跳过 identity 因为它=基线) ---
    'A6_fpn': {'name': 'FPN 融合',              'group': 'A6', 'overrides': {'model.fpn_type': 'fpn'}},
    # --- A7: 中心采样半径 (跳过 r=1.5 因为它=基线) ---
    'A7_r0.0': {'name': '采样半径 r=0.0',       'group': 'A7', 'overrides': {'train_cfg.center_sample_radius': 0.0}},
    'A7_r0.5': {'name': '采样半径 r=0.5',       'group': 'A7', 'overrides': {'train_cfg.center_sample_radius': 0.5}},
    'A7_r1.0': {'name': '采样半径 r=1.0',       'group': 'A7', 'overrides': {'train_cfg.center_sample_radius': 1.0}},
    'A7_r2.0': {'name': '采样半径 r=2.0',       'group': 'A7', 'overrides': {'train_cfg.center_sample_radius': 2.0}},
    # --- A8: 损失权重 (跳过 lw=1.0 因为它=基线) ---
    'A8_lw0.5': {'name': 'loss_weight=0.5',     'group': 'A8', 'overrides': {'train_cfg.loss_weight': 0.5}},
    'A8_lw2.0': {'name': 'loss_weight=2.0',     'group': 'A8', 'overrides': {'train_cfg.loss_weight': 2.0}},
    'A8_lw5.0': {'name': 'loss_weight=5.0',     'group': 'A8', 'overrides': {'train_cfg.loss_weight': 5.0}},
}

# 基线等效实验: 直接复用基线结果
BASELINE_EQUIVALENTS = {
    'A3_w1':       {'name': '窗口 w=1 (基线)',          'group': 'A3'},
    'A4_k5.0':     {'name': 'k=5.0 (基线)',             'group': 'A4'},
    'A6_identity': {'name': 'FPN Identity (基线)',      'group': 'A6'},
    'A7_r1.5':     {'name': '采样半径 r=1.5 (基线)',    'group': 'A7'},
    'A8_lw1.0':    {'name': 'loss_weight=1.0 (基线)',   'group': 'A8'},
}

CKPT_DIR = os.path.join(REPO, 'ckpt')
BASE_CONFIG = os.path.join(REPO, 'configs/thumos_i3d.yaml')


def load_yaml(path):
    import yaml
    with open(path, 'r') as f:
        return yaml.load(f, Loader=yaml.FullLoader)


def dump_yaml(cfg, path):
    import yaml
    class Dumper(yaml.Dumper): pass
    def _list_repr(dumper, data):
        if len(data) <= 6:
            return dumper.represent_sequence('tag:yaml.org,2002:seq', data, flow_style=True)
        return dumper.represent_sequence('tag:yaml.org,2002:seq', data, flow_style=False)
    Dumper.add_representer(list, _list_repr)
    with open(path, 'w') as f:
        yaml.dump(cfg, f, Dumper=Dumper, default_flow_style=False, sort_keys=False, allow_unicode=True)


def deep_set(d, key_path, value):
    keys = key_path.split('.')
    for k in keys[:-1]:
        d = d[k]
    d[keys[-1]] = value


def run_cmd(cmd, desc, timeout=7200):
    """运行命令, 实时打印输出。返回 (success, output_lines)"""
    print(f"\n  [{desc}] 开始...")
    start = time.time()
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, cwd=REPO, bufsize=1)
    out_lines = []
    try:
        for line in proc.stdout:
            line = line.rstrip()
            print(f"    {line}")
            out_lines.append(line)
        proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()
        elapsed = (time.time() - start) / 60
        print(f"  [{desc}] TIMEOUT after {elapsed:.0f}min")
        return False, out_lines

    elapsed = (time.time() - start) / 60
    ok = proc.returncode == 0
    print(f"  [{desc}] {'OK' if ok else 'FAIL(rc='+str(proc.returncode)+')'} [{elapsed:.0f}min]")
    return ok, out_lines


def parse_map(output_lines):
    maps = {}
    for line in output_lines:
        line = line.strip()
        if 'tIoU =' in line and 'mAP' in line:
            try:
                parts = line.split('=')
                tiou = float(parts[1].split(':')[0].strip())
                map_val = float(parts[2].strip().replace('%', '').replace('(%)', '').strip())
                maps[tiou] = map_val
            except (ValueError, IndexError):
                pass
        if 'Avearge mAP' in line or 'Average mAP' in line:
            try:
                maps['avg'] = float(line.split(':')[1].strip().replace('%', '').replace('(%)', '').strip())
            except (ValueError, IndexError):
                pass
    return maps


def run_single_experiment(exp_id, name, group, overrides):
    """训练+评估单个实验。"""
    print(f"\n{'='*60}")
    print(f"  实验 {exp_id}: {name} [{group}]")
    print(f"{'='*60}")

    # 1. 生成配置
    cfg = load_yaml(BASE_CONFIG)
    for key_path, val in overrides.items():
        deep_set(cfg, key_path, val)
    exp_config = os.path.join(WORK_DIR, f'abl_{exp_id}.yaml')
    dump_yaml(cfg, exp_config)

    # 2. 训练
    start = time.time()
    ok, train_out = run_cmd(
        [PYTHON, 'train.py', exp_config, '--output', exp_id],
        f'Train {exp_id}'
    )
    train_time = (time.time() - start) / 60
    if not ok:
        return {'exp_id': exp_id, 'name': name, 'group': group,
                'status': 'TRAIN_FAILED', 'train_time_min': round(train_time, 1)}

    # 3. 评估
    ckpt_folder = os.path.join(CKPT_DIR, f'abl_{exp_id}_{exp_id}')
    ok, eval_out = run_cmd(
        [PYTHON, 'eval.py', exp_config, ckpt_folder],
        f'Eval {exp_id}'
    )
    maps = parse_map(eval_out) if ok else {}
    return {
        'exp_id': exp_id,
        'name': name,
        'group': group,
        'mAP@0.3': maps.get(0.3),
        'mAP@0.5': maps.get(0.5),
        'mAP@0.7': maps.get(0.7),
        'avg_mAP': maps.get('avg'),
        'train_time_min': round(train_time, 1),
        'status': 'OK' if ok else 'EVAL_FAILED',
    }


def evaluate_existing(exp_id, config_path, ckpt_folder, name, group):
    """仅评估已有checkpoint。"""
    print(f"\n{'='*60}")
    print(f"  评估已有 {exp_id}: {name} [{group}]")
    print(f"{'='*60}")

    ok, eval_out = run_cmd(
        [PYTHON, 'eval.py', config_path, ckpt_folder],
        f'Eval {exp_id}'
    )
    maps = parse_map(eval_out) if ok else {}
    return {
        'exp_id': exp_id,
        'name': name,
        'group': group,
        'mAP@0.3': maps.get(0.3),
        'mAP@0.5': maps.get(0.5),
        'mAP@0.7': maps.get(0.7),
        'avg_mAP': maps.get('avg'),
        'train_time_min': 0,
        'status': 'OK' if ok else 'EVAL_FAILED',
    }


def write_csv(results, csv_path):
    """写入所有结果到CSV。"""
    import csv
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['实验编号', '分组', '配置变更', 'mAP@0.3', 'mAP@0.5',
                    'mAP@0.7', 'avg_mAP', '训练时间(min)', '状态', '时间戳'])
        for r in results:
            def fmt(v):
                if v is None: return ''
                return f'{v:.2f}' if isinstance(v, (int, float)) else str(v)

            w.writerow([
                r.get('exp_id', ''), r.get('group', ''), r.get('name', ''),
                fmt(r.get('mAP@0.3')), fmt(r.get('mAP@0.5')),
                fmt(r.get('mAP@0.7')), fmt(r.get('avg_mAP')),
                r.get('train_time_min', ''), r.get('status', ''),
                datetime.now().strftime('%Y-%m-%d %H:%M'),
            ])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--start', type=str, default='', help='从指定实验开始')
    parser.add_argument('--eval-only', action='store_true', help='仅评估已有checkpoint, 不训练')
    args = parser.parse_args()

    os.makedirs(WORK_DIR, exist_ok=True)
    csv_path = os.path.join(WORK_DIR, 'ablation_results.csv')
    all_results = []
    start_processing = args.start == ''

    # ============================================================
    # 第1步: 基线 (eval existing checkpoint)
    # ============================================================
    print("\n" + "="*70)
    print("  [1] 基线模型评估")
    print("="*70)
    bl_result = evaluate_existing(
        'baseline', 'configs/thumos_i3d.yaml',
        os.path.join(CKPT_DIR, 'thumos_i3d_baseline'),
        'Baseline (SGP + Trident + DIoU)', 'baseline'
    )
    all_results.append(bl_result)
    bl_maps = {k: v for k, v in bl_result.items() if k.startswith('mAP') or k == 'avg_mAP'}
    print(f"\n  ★ 基线: mAP@0.3={bl_maps.get('mAP@0.3'):.2f}  mAP@0.5={bl_maps.get('mAP@0.5'):.2f}  mAP@0.7={bl_maps.get('mAP@0.7'):.2f}  avg={bl_maps.get('avg_mAP'):.2f}")

    # 将基线结果复制给基线等效实验
    for exp_id, info in BASELINE_EQUIVALENTS.items():
        r = bl_result.copy()
        r['exp_id'] = exp_id
        r['name'] = info['name']
        r['group'] = info['group']
        r['train_time_min'] = 0
        r['status'] = 'BASELINE_EQ'
        all_results.append(r)

    # ============================================================
    # 第2步: A1 (eval existing checkpoint)
    # ============================================================
    print("\n" + "="*70)
    print("  [2] A1: Trident-head → 普通回归头")
    print("="*70)
    a1_result = evaluate_existing(
        'A1', os.path.join(REPO, 'work/abl_A1.yaml'),
        os.path.join(CKPT_DIR, 'abl_A1_A1'),
        'Trident-head→普通回归头 (use_trident_head=False)', 'A1'
    )
    all_results.append(a1_result)

    # ============================================================
    # 第3步: 训练队列 A2-A8
    # ============================================================
    train_queue = list(EXPERIMENTS.items())
    n_total = len(train_queue)
    print(f"\n{'='*70}")
    print(f"  训练队列: {n_total} 项 ({n_total} × ~1h = ~{n_total}h)")
    print(f"{'='*70}")

    for i, (exp_id, info) in enumerate(train_queue):
        if not start_processing and exp_id != args.start:
            continue
        start_processing = True

        if args.eval_only:
            # 检查是否有已有checkpoint
            ckpt_folder = os.path.join(CKPT_DIR, f'abl_{exp_id}_{exp_id}')
            if not os.path.isdir(ckpt_folder):
                print(f"  [{i+1}/{n_total}] {exp_id}: 无checkpoint, 跳过")
                continue
            result = evaluate_existing(
                exp_id, os.path.join(WORK_DIR, f'abl_{exp_id}.yaml'),
                ckpt_folder, info['name'], info['group']
            )
        else:
            print(f"\n  >>> [{i+1}/{n_total}] {exp_id}: {info['name']} <<<")
            result = run_single_experiment(
                exp_id, info['name'], info['group'], info['overrides']
            )

        all_results.append(result)
        # 每个实验后保存 CSV (防止中断丢失)
        write_csv(all_results, csv_path)

    # ============================================================
    # 第4步: 最终保存
    # ============================================================
    write_csv(all_results, csv_path)
    print(f"\n{'='*70}")
    print(f"  全部完成! {len(all_results)} 项结果已保存到:")
    print(f"  {csv_path}")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()

"""
TriDet 消融实验 — 单文件完整自动化

直接运行: python work_1/do_all.py

Step 1: 评估已有 checkpoint (baseline, A1, A2) — 用 subprocess 调用 eval.py
Step 2: 训练 18 个新实验 (A3-A8) — 用 subprocess 调用 train.py + eval.py
Step 3: 生成 CSV

每步即时打印进度, 结果即时写入 CSV 防止丢失。
"""

import os, sys, subprocess, csv, yaml, re, time, argparse
from datetime import datetime
from pathlib import Path

# ---- 路径配置 ----
REPO = Path('e:/Tridet/TriDet-Pro')
PYTHON = 'E:/anaconda/envs/test/python.exe'
WORK = REPO / 'work_1'
CKPT = REPO / 'ckpt'
CSV_FILE = WORK / 'ablation_results.csv'
BASE_CFG = REPO / 'configs/thumos_i3d.yaml'

# ---- 已有 checkpoint ----
EXISTING = [
    ('baseline', 'Baseline (SGP+Trident+DIoU)', 'baseline',
     str(REPO / 'configs/thumos_i3d.yaml'), str(CKPT / 'thumos_i3d_baseline'), 40),
    ('A1', 'Trident-head→普通回归头', 'A1',
     str(REPO / 'work/abl_A1.yaml'), str(CKPT / 'abl_A1_A1'), 40),
    ('A2', 'SGP→Conv (20ep)', 'A2',
     str(REPO / 'work/abl_A2.yaml'), str(CKPT / 'abl_A2_A2'), 20),
]

# ---- 基线等效 (直接复用 baseline) ----
BL_EQ = {
    'A3_w1': ('A3', '窗口 w=1 (基线)'),
    'A4_k5.0': ('A4', 'k=5.0 (基线)'),
    'A6_identity': ('A6', 'FPN Identity (基线)'),
    'A7_r1.5': ('A7', 'r=1.5 (基线)'),
    'A8_lw1.0': ('A8', 'loss_weight=1.0 (基线)'),
}

# ---- 需要训练的实验 ----
TO_TRAIN = [
    # A3: SGP 窗口
    ('A3_w3',  '窗口 w=3',          'A3', {'model.n_sgp_win_size': 3}),
    ('A3_w5',  '窗口 w=5',          'A3', {'model.n_sgp_win_size': 5}),
    ('A3_w7',  '窗口 w=7',          'A3', {'model.n_sgp_win_size': 7}),
    ('A3_w9',  '窗口 w=9',          'A3', {'model.n_sgp_win_size': 9}),
    ('A3_w11', '窗口 w=11',         'A3', {'model.n_sgp_win_size': 11}),
    ('A3_pl',  '窗口 [1,3,5,7,9,11]', 'A3', {'model.n_sgp_win_size': [1,3,5,7,9,11]}),
    # A4: k 参数
    ('A4_k1.0', 'k=1.0', 'A4', {'model.k': 1.0}),
    ('A4_k1.5', 'k=1.5', 'A4', {'model.k': 1.5}),
    ('A4_k3.0', 'k=3.0', 'A4', {'model.k': 3.0}),
    ('A4_k7.0', 'k=7.0', 'A4', {'model.k': 7.0}),
    # A5: GIoU
    ('A5', 'DIoU→GIoU', 'A5', {'train_cfg.loss_type': 'giou'}),
    # A6: FPN
    ('A6_fpn', 'FPN 融合', 'A6', {'model.fpn_type': 'fpn'}),
    # A7: 中心采样
    ('A7_r0.0', 'r=0.0', 'A7', {'train_cfg.center_sample_radius': 0.0}),
    ('A7_r0.5', 'r=0.5', 'A7', {'train_cfg.center_sample_radius': 0.5}),
    ('A7_r1.0', 'r=1.0', 'A7', {'train_cfg.center_sample_radius': 1.0}),
    ('A7_r2.0', 'r=2.0', 'A7', {'train_cfg.center_sample_radius': 2.0}),
    # A8: 损失权重
    ('A8_lw0.5', 'lw=0.5', 'A8', {'train_cfg.loss_weight': 0.5}),
    ('A8_lw2.0', 'lw=2.0', 'A8', {'train_cfg.loss_weight': 2.0}),
    ('A8_lw5.0', 'lw=5.0', 'A8', {'train_cfg.loss_weight': 5.0}),
]


def deep_set(d, key_path, value):
    keys = key_path.split('.')
    for k in keys[:-1]:
        d = d[k]
    d[keys[-1]] = value


def dump_yaml(cfg_dict, path):
    class Dumper(yaml.Dumper): pass
    def _lr(dumper, data):
        return dumper.represent_sequence(
            'tag:yaml.org,2002:seq', data,
            flow_style=len(data) <= 6)
    Dumper.add_representer(list, _lr)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        yaml.dump(cfg_dict, f, Dumper=Dumper, default_flow_style=False,
                  sort_keys=False, allow_unicode=True)


def parse_maps(text):
    """从 eval.py stdout 解析 mAP 值."""
    maps = {}
    for line in text.split('\n'):
        m = re.search(r'tIoU\s*=\s*([\d.]+)\s*:\s*mAP\s*=\s*([\d.]+)', line)
        if m:
            maps[float(m.group(1))] = float(m.group(2))
        m2 = re.search(r'(?:Avearge|Average)\s+mAP\s*:\s*([\d.]+)', line)
        if m2:
            maps['avg'] = float(m2.group(1))
    return maps


def run_cmd(args_list, desc, timeout_s=3600):
    """运行命令, 返回 (success, stdout, stderr, elapsed_min)."""
    print(f"\n  [{desc}]")
    sys.stdout.flush()
    t0 = time.time()
    try:
        r = subprocess.run(
            args_list, cwd=str(REPO),
            capture_output=True, text=True, timeout=timeout_s)
    except subprocess.TimeoutExpired:
        el = (time.time() - t0) / 60
        print(f"  [TIMEOUT] after {el:.0f}min")
        return False, '', '', el
    el = (time.time() - t0) / 60
    ok = r.returncode == 0
    tag = 'OK' if ok else f'FAIL(rc={r.returncode})'
    print(f"  [{desc}] {tag} [{el:.0f}min]")
    if r.stdout:
        # 打印最后 3 行非空
        lines = [l for l in r.stdout.strip().split('\n') if l.strip()]
        for l in lines[-3:]:
            print(f"    {l[:120]}")
    if r.stderr and not ok:
        for l in r.stderr.strip().split('\n')[-3:]:
            print(f"    stderr: {l[:200]}")
    return ok, r.stdout, r.stderr, el


def load_csv():
    if not CSV_FILE.exists():
        return {}
    with open(CSV_FILE, 'r', encoding='utf-8') as f:
        return {r['实验编号'].strip(): r for r in csv.DictReader(f)}


def f(val):
    if val is None: return ''
    return f'{val:.2f}' if isinstance(val, (int, float)) else str(val)


def save_csv(all_res):
    os.makedirs(WORK, exist_ok=True)
    with open(CSV_FILE, 'w', newline='', encoding='utf-8') as fh:
        w = csv.writer(fh)
        w.writerow(['实验编号','分组','配置变更','mAP@0.3','mAP@0.5',
                    'mAP@0.7','avg_mAP','训练时间(min)','状态','时间戳'])
        for eid in sorted(all_res.keys()):
            r = all_res[eid]
            w.writerow([eid, r.get('group',''), r.get('name',''),
                       f(r.get('mAP@0.3')), f(r.get('mAP@0.5')),
                       f(r.get('mAP@0.7')), f(r.get('avg_mAP')),
                       r.get('train_time',''), r.get('status',''),
                       r.get('ts','')])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--eval-only', action='store_true')
    parser.add_argument('--train-only', action='store_true')
    parser.add_argument('--start', type=str, default='')
    args = parser.parse_args()

    WORK.mkdir(exist_ok=True)
    all_r = load_csv()
    now = lambda: datetime.now().strftime('%Y-%m-%d %H:%M')

    # ========================================
    # Phase 1: 评估已有 checkpoint
    # ========================================
    if not args.train_only:
        print("=" * 60)
        print("  Phase 1/2: 评估已有 checkpoint (baseline, A1, A2)")
        print("=" * 60)

        for eid, name, grp, cfg_p, ckpt_p, ep in EXISTING:
            if eid in all_r and all_r[eid].get('avg_mAP'):
                print(f"  [{eid}] 已有结果, skip")
                continue

            print(f"\n  [{eid}] {name}  ({ep}ep)")
            ok, out, err, el = run_cmd(
                [PYTHON, '-u', 'eval.py', cfg_p, ckpt_p, '-p', '10'],
                f'eval {eid}', timeout_s=3600)

            maps = parse_maps(out) if ok else {}
            all_r[eid] = {
                'exp_id': eid, 'group': grp, 'name': name,
                'mAP@0.3': maps.get(0.3), 'mAP@0.5': maps.get(0.5),
                'mAP@0.7': maps.get(0.7), 'avg_mAP': maps.get('avg'),
                'train_time': f'{ep}ep(已有)', 'status': 'OK' if maps else 'EVAL_FAIL',
                'ts': now(),
            }
            save_csv(all_r)

        # 复制基线→等效实验
        bl = all_r.get('baseline', {})
        if bl and bl.get('status') == 'OK':
            for eid, (grp, name) in BL_EQ.items():
                if eid not in all_r:
                    all_r[eid] = {
                        'exp_id': eid, 'group': grp, 'name': name,
                        'mAP@0.3': bl.get('mAP@0.3'), 'mAP@0.5': bl.get('mAP@0.5'),
                        'mAP@0.7': bl.get('mAP@0.7'), 'avg_mAP': bl.get('avg_mAP'),
                        'train_time': '(基线复用)', 'status': 'BASELINE_EQ',
                        'ts': now(),
                    }
            save_csv(all_r)

        print(f"\n  Phase 1 DONE. {len(all_r)} results so far.")

    if args.eval_only:
        print(f"  Results → {CSV_FILE}")
        return

    # ========================================
    # Phase 2: 训练新实验
    # ========================================
    todo = [(e, n, g, o) for e, n, g, o in TO_TRAIN
            if e not in all_r or not all_r[e].get('avg_mAP')]

    todo = todo[:]
    if args.start:
        todo = [(e,n,g,o) for e,n,g,o in todo
                if todo.index((e,n,g,o)) >=
                   next((i for i,(x,_,_,_) in enumerate(todo) if x==args.start), 0)]

    if not todo:
        print("  Everything done!")
        return

    print(f"\n{'='*60}")
    print(f"  Phase 2/2: 训练 {len(todo)} 项 (~{len(todo)}h total)")
    print(f"{'='*60}")

    for i, (eid, name, grp, overrides) in enumerate(todo):
        n_tot = len(todo)
        hdr = f"[{i+1}/{n_tot}] {eid}: {name}"
        print(f"\n{'='*55}\n  {hdr}\n{'='*55}")

        # 1. 生成配置
        sys.path.insert(0, str(REPO))
        from libs.core.config import load_config
        cfg = load_config(str(BASE_CFG))
        for kp, v in overrides.items():
            deep_set(cfg, kp, v)
        exp_cfg = str(WORK / f'abl_{eid}.yaml')
        dump_yaml(cfg, exp_cfg)

        # 2. 训练
        ok, _, _, train_min = run_cmd(
            [PYTHON, '-u', 'train.py', exp_cfg, '--output', eid],
            f'train {eid}', timeout_s=14400)  # 4h max

        if not ok:
            all_r[eid] = {
                'exp_id': eid, 'group': grp, 'name': name,
                'train_time': f'{train_min:.0f}', 'status': 'TRAIN_FAIL', 'ts': now(),
            }
            save_csv(all_r)
            continue

        # 3. 评估
        ckpt_p = str(CKPT / f'abl_{eid}_{eid}')
        ok2, out2, _, _ = run_cmd(
            [PYTHON, '-u', 'eval.py', exp_cfg, ckpt_p, '-p', '10'],
            f'eval {eid}', timeout_s=3600)

        maps2 = parse_maps(out2) if ok2 else {}
        all_r[eid] = {
            'exp_id': eid, 'group': grp, 'name': name,
            'mAP@0.3': maps2.get(0.3), 'mAP@0.5': maps2.get(0.5),
            'mAP@0.7': maps2.get(0.7), 'avg_mAP': maps2.get('avg'),
            'train_time': f'{train_min:.0f}', 'status': 'OK' if maps2 else 'EVAL_FAIL',
            'ts': now(),
        }
        save_csv(all_r)

    print(f"\n{'='*60}")
    print(f"  ALL DONE! {len(all_r)} results → {CSV_FILE}")
    print(f"  Run: python work_1/analyze.py")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()

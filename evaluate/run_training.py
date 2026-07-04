"""
TriDet 消融实验 — 训练 + 评估 (A3-A8)。

已知结果直接写入, 新实验逐项训练+评估。

用法: python work_1/run_training.py
"""

import os, sys, subprocess, csv, yaml, re, time
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
PYTHON = sys.executable
WORK = REPO / 'evaluate'
CKPT = REPO / 'ckpt'
CSV_FILE = WORK / 'ablation_results.csv'

# ===== 已知结果 (来自 evaluate/results/analysis_report.md) =====
KNOWN = {
    'baseline': {
        'exp_id': 'baseline', 'group': 'baseline',
        'name': 'Baseline (SGP+Trident+DIoU)',
        'mAP@0.3': 75.1, 'mAP@0.5': 62.6, 'mAP@0.7': 38.3, 'avg_mAP': 59.31,
        'train_time': '40ep', 'status': 'KNOWN',
    },
    'A1': {
        'exp_id': 'A1', 'group': 'A1',
        'name': 'Trident-head→普通回归头',
        'mAP@0.3': 74.8, 'mAP@0.5': 62.1, 'mAP@0.7': 36.4, 'avg_mAP': 58.38,
        'train_time': '40ep', 'status': 'KNOWN',
    },
    'A2': {
        'exp_id': 'A2', 'group': 'A2',
        'name': 'SGP→Conv (20ep, 下界估计)',
        'mAP@0.3': 61.7, 'mAP@0.5': 47.8, 'mAP@0.7': 21.7, 'avg_mAP': 44.46,
        'train_time': '20ep', 'status': 'KNOWN_PARTIAL',
    },
}

# 基线等效
BL_EQ = {
    'A3_w1':       ('A3', '窗口 w=1 (基线)'),
    'A4_k5.0':     ('A4', 'k=5.0 (基线)'),
    'A6_identity': ('A6', 'FPN Identity (基线)'),
    'A7_r1.5':     ('A7', 'r=1.5 (基线)'),
    'A8_lw1.0':    ('A8', 'loss_weight=1.0 (基线)'),
}

# ===== 待训练实验 (按优先级) =====
TO_TRAIN = [
    ('A3_w3',  'SGP窗口 w=3',           'A3', {'model.n_sgp_win_size': 3}),
    ('A3_w5',  'SGP窗口 w=5',           'A3', {'model.n_sgp_win_size': 5}),
    ('A3_w7',  'SGP窗口 w=7',           'A3', {'model.n_sgp_win_size': 7}),
    ('A3_w9',  'SGP窗口 w=9',           'A3', {'model.n_sgp_win_size': 9}),
    ('A3_w11', 'SGP窗口 w=11',          'A3', {'model.n_sgp_win_size': 11}),
    ('A3_per_layer',  'SGP窗口 [1,3,5,7,9,11]','A3', {'model.n_sgp_win_size': [1,3,5,7,9,11]}),
    ('A4_k1.0','k=1.0',  'A4', {'model.k': 1.0}),
    ('A4_k1.5','k=1.5',  'A4', {'model.k': 1.5}),
    ('A4_k3.0','k=3.0',  'A4', {'model.k': 3.0}),
    ('A4_k7.0','k=7.0',  'A4', {'model.k': 7.0}),
    ('A5',     'DIoU→GIoU','A5', {'train_cfg.loss_type': 'giou'}),
    ('A6_fpn', 'FPN 融合', 'A6', {'model.fpn_type': 'fpn'}),
    ('A7_r0.0','采样 r=0.0','A7', {'train_cfg.center_sample_radius': 0.0}),
    ('A7_r0.5','采样 r=0.5','A7', {'train_cfg.center_sample_radius': 0.5}),
    ('A7_r1.0','采样 r=1.0','A7', {'train_cfg.center_sample_radius': 1.0}),
    ('A7_r2.0','采样 r=2.0','A7', {'train_cfg.center_sample_radius': 2.0}),
    ('A8_lw0.5','lw=0.5','A8', {'train_cfg.loss_weight': 0.5}),
    ('A8_lw2.0','lw=2.0','A8', {'train_cfg.loss_weight': 2.0}),
    ('A8_lw5.0','lw=5.0','A8', {'train_cfg.loss_weight': 5.0}),
]


def deep_set(d, key_path, value):
    keys = key_path.split('.')
    for k in keys[:-1]:
        d = d[k]
    d[keys[-1]] = value


def dump_yaml(cfg, path):
    class Dumper(yaml.Dumper): pass
    def _lr(dumper, data):
        return dumper.represent_sequence('tag:yaml.org,2002:seq', data,
                                         flow_style=len(data) <= 6)
    Dumper.add_representer(list, _lr)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        yaml.dump(cfg, f, Dumper=Dumper, default_flow_style=False,
                  sort_keys=False, allow_unicode=True)


def parse_maps(text):
    maps = {}
    for line in text.split('\n'):
        m = re.search(r'tIoU\s*=\s*([\d.]+)\s*:\s*mAP\s*=\s*([\d.]+)', line)
        if m:
            maps[round(float(m.group(1)), 1)] = float(m.group(2))
        m2 = re.search(r'(?:Avearge|Average)\s+mAP\s*:\s*([\d.]+)', line)
        if m2:
            maps['avg'] = float(m2.group(1))
    return maps


def run_cmd(cmd, desc, timeout_s=3600):
    """运行命令, stdout/stderr 重定向到文件避免 PIPE 缓冲区死锁。"""
    import tempfile
    print(f"\n  [{desc}]")
    sys.stdout.flush()
    t0 = time.time()

    # 输出到临时文件, 避免 subprocess.PIPE 缓冲区满导致死锁
    out_f = tempfile.NamedTemporaryFile(mode='w+', suffix='.log', delete=False,
                                         dir=str(WORK), prefix=f'run_{desc.replace(" ","_")[:20]}_')
    out_path = out_f.name
    try:
        r = subprocess.run(cmd, cwd=str(REPO),
                           stdout=out_f, stderr=subprocess.STDOUT,
                           text=True, timeout=timeout_s)
    except subprocess.TimeoutExpired:
        out_f.close()
        el = (time.time() - t0) / 60
        print(f"  [TIMEOUT] {el:.0f}min")
        return False, '', '', el
    finally:
        out_f.close()

    el = (time.time() - t0) / 60
    ok = r.returncode == 0
    print(f"  [{'OK' if ok else f'FAIL({r.returncode})'}] {el:.0f}min")

    # 从文件读取输出
    with open(out_path, 'r', encoding='utf-8', errors='replace') as f:
        out_text = f.read()

    # 打印最后几行
    lines = [x for x in out_text.strip().split('\n') if x.strip()]
    for l in lines[-3:]:
        print(f"    {l[:150]}")
    if not ok:
        print(f"    (full log: {out_path})")

    # 如果是训练成功, 保留日志; eval 日志可删除
    if ok and 'eval' not in desc.lower():
        # 保留训练日志
        pass
    else:
        try:
            os.unlink(out_path)
        except:
            pass

    return ok, out_text, '', el


def save_csv(all_r):
    WORK.mkdir(exist_ok=True)
    with open(CSV_FILE, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['实验编号','分组','配置变更','mAP@0.3','mAP@0.5',
                    'mAP@0.7','avg_mAP','训练时间(min)','状态','时间戳'])
        for eid in sorted(all_r.keys()):
            r = all_r[eid]
            def ff(v): return f'{v:.1f}' if isinstance(v, float) else f'{v:.2f}' if isinstance(v, float) else str(v) if v is not None else ''
            w.writerow([eid, r.get('group',''), r.get('name',''),
                       ff(r.get('mAP@0.3')), ff(r.get('mAP@0.5')),
                       ff(r.get('mAP@0.7')), ff(r.get('avg_mAP')),
                       r.get('train_time',''), r.get('status',''),
                       r.get('ts','')])


def main():
    WORK.mkdir(exist_ok=True)
    now = lambda: datetime.now().strftime('%Y-%m-%d %H:%M')
    all_r = {}

    # 1. 写入已知结果
    for eid, r in KNOWN.items():
        r['ts'] = now()
        all_r[eid] = r.copy()

    # 2. 基线等效
    bl = all_r['baseline']
    for eid, (grp, name) in BL_EQ.items():
        all_r[eid] = {
            'exp_id': eid, 'group': grp, 'name': name,
            'mAP@0.3': bl['mAP@0.3'], 'mAP@0.5': bl['mAP@0.5'],
            'mAP@0.7': bl['mAP@0.7'], 'avg_mAP': bl['avg_mAP'],
            'train_time': '(复用基线)', 'status': 'BASELINE_EQ', 'ts': now(),
        }
    save_csv(all_r)

    # 3. 训练新实验
    sys.path.insert(0, str(REPO))
    from libs.core.config import load_config
    BASE_CFG = str(REPO / 'configs/thumos_i3d.yaml')

    total = len(TO_TRAIN)
    print(f"\n{'='*60}")
    print(f"  Training {total} experiments (~{total}h)")
    print(f"{'='*60}")

    for i, (eid, name, grp, overrides) in enumerate(TO_TRAIN):
        print(f"\n{'='*55}")
        print(f"  [{i+1}/{total}] {eid}: {name} [{grp}]")
        print(f"{'='*55}")

        # Generate config
        cfg = load_config(BASE_CFG)
        for kp, v in overrides.items():
            deep_set(cfg, kp, v)
        exp_cfg = str(WORK / f'abl_{eid}.yaml')
        dump_yaml(cfg, exp_cfg)
        print(f"  Config: {exp_cfg}")

        # Train
        ok, _, _, train_min = run_cmd(
            [PYTHON, '-u', 'train.py', exp_cfg, '--output', eid],
            f'train {eid}', timeout_s=14400)

        if not ok:
            all_r[eid] = {
                'exp_id': eid, 'group': grp, 'name': name,
                'train_time': f'{train_min:.0f}', 'status': 'TRAIN_FAIL', 'ts': now(),
            }
            save_csv(all_r)
            continue

        # Eval (with long timeout)
        ckpt_p = str(CKPT / f'abl_{eid}_{eid}')
        ok2, out2, _, eval_min = run_cmd(
            [PYTHON, '-u', 'eval.py', exp_cfg, ckpt_p, '-p', '10'],
            f'eval {eid}', timeout_s=7200)  # 2h for eval

        maps2 = parse_maps(out2) if ok2 else {}
        all_r[eid] = {
            'exp_id': eid, 'group': grp, 'name': name,
            'mAP@0.3': maps2.get(0.3), 'mAP@0.5': maps2.get(0.5),
            'mAP@0.7': maps2.get(0.7), 'avg_mAP': maps2.get('avg'),
            'train_time': f'{train_min:.0f}+{eval_min:.0f}',
            'status': 'OK' if maps2 else 'EVAL_FAIL', 'ts': now(),
        }
        save_csv(all_r)
        print(f"  [{eid}] Done! mAP_avg={maps2.get('avg', '?')}")

    print(f"\n{'='*60}")
    print(f"  ALL DONE! {len(all_r)} results → {CSV_FILE}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()

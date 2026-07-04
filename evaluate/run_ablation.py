"""
消融实验自动化脚本 (完整版 A1-A8)。

对每一项消融:
    1. 加载基线配置，修改指定参数，保存为临时 yaml
    2. 训练 (train.py)
    3. 评估 (eval.py)
    4. 记录结果到 ablation_results.csv

用法:
    python work_1/run_ablation.py --exp A1              # 单实验
    python work_1/run_ablation.py --exp A1,A2,A3_w3     # 多项
    python work_1/run_ablation.py --all                 # 全部 A1-A8
    python work_1/run_ablation.py --baseline            # 仅训练基线
"""

import os, sys, subprocess, csv, argparse, time, re
from datetime import datetime

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PYTHON = sys.executable
BASE_CONFIG = os.path.join(REPO, 'configs/thumos_i3d.yaml')
WORK_DIR = os.path.join(REPO, 'evaluate')
CKPT_DIR = os.path.join(REPO, 'ckpt')
CSV_FILE = os.path.join(WORK_DIR, 'ablation_results.csv')

# ---- 消融定义 (完整版) ----
ABLATIONS = {
    # === A1: Trident-head → 普通回归头 ===
    'A1': {
        'name': 'Trident-head → 普通回归头',
        'group': 'A1',
        'overrides': {'model.use_trident_head': False},
    },
    # === A2: SGP Backbone → Conv Backbone ===
    'A2': {
        'name': 'SGP Backbone → Conv Backbone',
        'group': 'A2',
        'overrides': {'model.backbone_type': 'conv'},
    },
    # === A3: SGP 窗口尺寸消融 ===
    'A3_w1': {
        'name': 'SGP 窗口 w=1 (基线)',
        'group': 'A3',
        'overrides': {'model.n_sgp_win_size': 1},
    },
    'A3_w3': {
        'name': 'SGP 窗口 w=3',
        'group': 'A3',
        'overrides': {'model.n_sgp_win_size': 3},
    },
    'A3_w5': {
        'name': 'SGP 窗口 w=5',
        'group': 'A3',
        'overrides': {'model.n_sgp_win_size': 5},
    },
    'A3_w7': {
        'name': 'SGP 窗口 w=7',
        'group': 'A3',
        'overrides': {'model.n_sgp_win_size': 7},
    },
    'A3_w9': {
        'name': 'SGP 窗口 w=9',
        'group': 'A3',
        'overrides': {'model.n_sgp_win_size': 9},
    },
    'A3_w11': {
        'name': 'SGP 窗口 w=11',
        'group': 'A3',
        'overrides': {'model.n_sgp_win_size': 11},
    },
    'A3_per_layer': {
        'name': 'SGP 窗口逐层 [1,3,5,7,9,11]',
        'group': 'A3',
        'overrides': {'model.n_sgp_win_size': [1, 3, 5, 7, 9, 11]},
    },
    # === A4: SGP 的 k 参数消融 ===
    'A4_k1.0': {
        'name': 'k=1.0',
        'group': 'A4',
        'overrides': {'model.k': 1.0},
    },
    'A4_k1.5': {
        'name': 'k=1.5',
        'group': 'A4',
        'overrides': {'model.k': 1.5},
    },
    'A4_k3.0': {
        'name': 'k=3.0',
        'group': 'A4',
        'overrides': {'model.k': 3.0},
    },
    'A4_k5.0': {
        'name': 'k=5.0 (基线)',
        'group': 'A4',
        'overrides': {'model.k': 5.0},
    },
    'A4_k7.0': {
        'name': 'k=7.0',
        'group': 'A4',
        'overrides': {'model.k': 7.0},
    },
    # === A5: DIoU → GIoU 损失 (通过 loss_type 配置, 不再 manual) ===
    'A5': {
        'name': 'DIoU → GIoU 损失',
        'group': 'A5',
        'overrides': {'train_cfg.loss_type': 'giou'},
    },
    # === A6: FPN vs Identity Neck ===
    'A6_identity': {
        'name': 'FPN Identity (基线)',
        'group': 'A6',
        'overrides': {'model.fpn_type': 'identity'},
    },
    'A6_fpn': {
        'name': 'FPN → 标准 FPN 融合',
        'group': 'A6',
        'overrides': {'model.fpn_type': 'fpn'},
    },
    # === A7: 中心采样半径消融 ===
    'A7_r0.0': {
        'name': '中心采样 radius=0.0',
        'group': 'A7',
        'overrides': {'train_cfg.center_sample_radius': 0.0},
    },
    'A7_r0.5': {
        'name': '中心采样 radius=0.5',
        'group': 'A7',
        'overrides': {'train_cfg.center_sample_radius': 0.5},
    },
    'A7_r1.0': {
        'name': '中心采样 radius=1.0',
        'group': 'A7',
        'overrides': {'train_cfg.center_sample_radius': 1.0},
    },
    'A7_r1.5': {
        'name': '中心采样 radius=1.5 (基线)',
        'group': 'A7',
        'overrides': {'train_cfg.center_sample_radius': 1.5},
    },
    'A7_r2.0': {
        'name': '中心采样 radius=2.0',
        'group': 'A7',
        'overrides': {'train_cfg.center_sample_radius': 2.0},
    },
    # === A8: 分类损失权重消融 ===
    'A8_lw0.5': {
        'name': '损失权重 loss_weight=0.5',
        'group': 'A8',
        'overrides': {'train_cfg.loss_weight': 0.5},
    },
    'A8_lw1.0': {
        'name': '损失权重 loss_weight=1.0 (基线)',
        'group': 'A8',
        'overrides': {'train_cfg.loss_weight': 1.0},
    },
    'A8_lw2.0': {
        'name': '损失权重 loss_weight=2.0',
        'group': 'A8',
        'overrides': {'train_cfg.loss_weight': 2.0},
    },
    'A8_lw5.0': {
        'name': '损失权重 loss_weight=5.0',
        'group': 'A8',
        'overrides': {'train_cfg.loss_weight': 5.0},
    },
}


def deep_set(d, key_path, value):
    """设置嵌套 dict 的值. key_path 形如 'model.k' 或 'train_cfg.loss_weight'"""
    keys = key_path.split('.')
    for k in keys[:-1]:
        d = d[k]
    d[keys[-1]] = value


def load_and_override_config(base_path, overrides):
    """加载基线配置并应用覆写。"""
    import yaml
    with open(base_path, 'r') as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)

    for key_path, val in overrides.items():
        deep_set(cfg, key_path, val)
    return cfg


def dump_config(cfg, out_path):
    """将配置写入 yaml 文件。"""
    import yaml

    class ConfigDumper(yaml.Dumper):
        pass

    def _list_representer(dumper, data):
        if len(data) <= 6:
            return dumper.represent_sequence(
                'tag:yaml.org,2002:seq', data, flow_style=True)
        return dumper.represent_sequence(
            'tag:yaml.org,2002:seq', data, flow_style=False)

    ConfigDumper.add_representer(list, _list_representer)

    with open(out_path, 'w') as f:
        yaml.dump(cfg, f, Dumper=ConfigDumper, default_flow_style=False,
                  sort_keys=False, allow_unicode=True)
    print(f"  配置已写入: {out_path}")


def run_cmd(cmd, desc):
    """运行命令并实时打印输出。"""
    print(f"\n  [{desc}]")
    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, cwd=REPO, bufsize=1)
    out_lines = []
    for line in proc.stdout:
        line = line.rstrip()
        print(f"    {line}")
        out_lines.append(line)
    proc.wait()
    ok = proc.returncode == 0
    print(f"  [{desc}] {'OK' if ok else 'FAIL(rc=' + str(proc.returncode) + ')'}")
    return ok, out_lines


def parse_map(output_lines):
    """从 eval 输出解析 mAP 值。"""
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


def run_ablation(exp_id, info):
    """执行单次消融实验。"""
    name = info['name']
    group = info.get('group', exp_id)
    overrides = info.get('overrides', {})

    print(f"\n{'='*60}")
    print(f"  消融实验 {exp_id}: {name}  [{group}]")
    print(f"{'='*60}")

    # 检查是否已运行
    if os.path.exists(CSV_FILE):
        with open(CSV_FILE, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            for row in reader:
                if row and row[0] == exp_id:
                    print(f"  [SKIP] {exp_id} 已有结果记录, 跳过")
                    return None

    # 1. 生成配置文件
    cfg = load_and_override_config(BASE_CONFIG, overrides)
    exp_config = os.path.join(WORK_DIR, f'abl_{exp_id}.yaml')
    dump_config(cfg, exp_config)

    # 2. 训练
    start = time.time()
    ok, train_out = run_cmd(
        [PYTHON, 'train.py', exp_config, '--output', exp_id],
        f'训练 {exp_id}'
    )
    train_time = (time.time() - start) / 60
    if not ok:
        return {'exp_id': exp_id, 'name': name, 'group': group,
                'status': 'TRAIN_FAILED',
                'train_time_min': round(train_time, 1)}

    # 3. 评估
    ckpt_folder = os.path.join(CKPT_DIR, f'abl_{exp_id}_{exp_id}')
    ok, eval_out = run_cmd(
        [PYTHON, 'eval.py', exp_config, ckpt_folder],
        f'评估 {exp_id}'
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


def ensure_csv_header():
    if not os.path.exists(CSV_FILE):
        with open(CSV_FILE, 'w', newline='', encoding='utf-8') as f:
            w = csv.writer(f)
            w.writerow(['实验编号', '分组', '配置变更', 'mAP@0.3', 'mAP@0.5',
                        'mAP@0.7', 'avg_mAP', '训练时间(min)', '状态', '时间戳'])


def log_result(result):
    ensure_csv_header()
    with open(CSV_FILE, 'a', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow([
            result.get('exp_id', ''),
            result.get('group', ''),
            result.get('name', ''),
            _fmt(result.get('mAP@0.3')),
            _fmt(result.get('mAP@0.5')),
            _fmt(result.get('mAP@0.7')),
            _fmt(result.get('avg_mAP')),
            result.get('train_time_min', ''),
            result.get('status', ''),
            datetime.now().strftime('%Y-%m-%d %H:%M'),
        ])
    print(f"  结果已记录到: {CSV_FILE}")


def _fmt(val):
    if val is None:
        return ''
    return f'{val:.2f}' if isinstance(val, (int, float)) else str(val)


def get_all_experiment_ids():
    """按优先级排序的实验 ID: 基线和核心对比先跑"""
    priority_order = ['A1', 'A2', 'A5',  # 核心对比 (单实验, 1h each)
                      'A6_identity', 'A6_fpn',  # FPN 对比
                      'A3_w1', 'A3_w3', 'A3_w5', 'A3_w7', 'A3_w9', 'A3_w11', 'A3_per_layer',  # 窗口消融
                      'A4_k1.0', 'A4_k1.5', 'A4_k3.0', 'A4_k5.0', 'A4_k7.0',  # k 参数消融
                      'A7_r0.0', 'A7_r0.5', 'A7_r1.0', 'A7_r1.5', 'A7_r2.0',  # 中心采样
                      'A8_lw0.5', 'A8_lw1.0', 'A8_lw2.0', 'A8_lw5.0',  # 损失权重
                      ]
    return [e for e in priority_order if e in ABLATIONS]


def train_baseline():
    """训练基线模型 (完整配置)。"""
    print("\n" + "="*60)
    print("  训练基线模型 (SGP + Trident-head + DIoU)")
    print("="*60)

    # 基线模型使用原始配置, 不覆写任何参数
    ok, out = run_cmd(
        [PYTHON, 'train.py', BASE_CONFIG, '--output', 'baseline'],
        '基线训练'
    )
    if not ok:
        print("  基线训练失败!")
        return None

    result = run_cmd(
        [PYTHON, 'eval.py', BASE_CONFIG,
         os.path.join(CKPT_DIR, 'thumos_i3d_baseline')],
        '基线评估'
    )
    if result[0]:
        maps = parse_map(result[1])
        return {
            'exp_id': 'baseline',
            'name': 'Baseline (SGP + Trident + DIoU)',
            'group': 'baseline',
            'mAP@0.3': maps.get(0.3),
            'mAP@0.5': maps.get(0.5),
            'mAP@0.7': maps.get(0.7),
            'avg_mAP': maps.get('avg'),
            'train_time_min': 0,
            'status': 'OK',
        }
    return None


def main():
    parser = argparse.ArgumentParser(description='TriDet 消融实验自动化')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--exp', type=str, help='实验编号, 逗号分隔, 如 A1,A2,A3_w3')
    group.add_argument('--all', action='store_true', help='运行所有 A1-A8 消融')
    group.add_argument('--baseline', action='store_true', help='仅训练基线模型')
    args = parser.parse_args()

    os.makedirs(WORK_DIR, exist_ok=True)

    if args.baseline:
        result = train_baseline()
        if result:
            log_result(result)
        return

    if args.all:
        exps = get_all_experiment_ids()
        print(f"\n  共 {len(exps)} 项消融实验待执行")
        print(f"  预计总时间: ~{len(exps)}h (每项约 1h)")
    else:
        exps = [e.strip() for e in args.exp.split(',')]

    for i, exp_id in enumerate(exps):
        if exp_id not in ABLATIONS:
            print(f"  [WARN] 未知实验: {exp_id}, 跳过")
            continue
        print(f"\n  [{i+1}/{len(exps)}] 执行: {exp_id}")
        result = run_ablation(exp_id, ABLATIONS[exp_id])
        if result:
            log_result(result)


if __name__ == '__main__':
    main()

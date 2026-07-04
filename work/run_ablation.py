"""
消融实验自动化脚本。

对每一项消融:
    1. 加载基线配置，修改指定参数，保存为临时 yaml
    2. 训练 (train.py)
    3. 评估 (eval.py)
    4. 记录结果到 ablation_results.csv

用法:
    python work/run_ablation.py --exp A1          # 单实验
    python work/run_ablation.py --exp A1,A2,A3    # 多项
    python work/run_ablation.py --all             # 全部
"""

import os, sys, subprocess, csv, argparse, time
from datetime import datetime

REPO = 'e:/Tridet/TriDet-Pro'
PYTHON = 'E:/anaconda/envs/test/python.exe'
BASE_CONFIG = os.path.join(REPO, 'configs/thumos_i3d.yaml')
WORK_DIR = os.path.join(REPO, 'work')
CKPT_DIR = os.path.join(REPO, 'ckpt')
CSV_FILE = os.path.join(WORK_DIR, 'ablation_results.csv')


# ---- 消融定义 ----
ABLATIONS = {
    'A1': {
        'name': 'Trident-head → 普通回归头',
        'overrides': {'model.use_trident_head': False},
    },
    'A2': {
        'name': 'SGP Backbone → Conv Backbone',
        'overrides': {'model.backbone_type': 'conv'},
    },
    'A3_w3': {
        'name': 'SGP 窗口 [1,3,5,7,9,11]',
        'overrides': {'model.n_sgp_win_size': [1, 3, 5, 7, 9, 11]},
    },
    'A4_k1': {
        'name': 'k=1.0',
        'overrides': {'model.k': 1.0},
    },
    'A4_k3': {
        'name': 'k=3.0',
        'overrides': {'model.k': 3.0},
    },
    'A4_k7': {
        'name': 'k=7.0',
        'overrides': {'model.k': 7.0},
    },
    'A5': {
        'name': 'DIoU → GIoU 损失',
        'overrides': {},  # 需修改 meta_archs.py 源码
        'manual': True,
    },
    'A6': {
        'name': 'FPN → Identity Neck (对比)',
        'overrides': {'model.fpn_type': 'fpn'},
    },
    'A7_r0': {
        'name': '中心采样 radius=0.0',
        'overrides': {'train_cfg.center_sample_radius': 0.0},
    },
    'A7_r1': {
        'name': '中心采样 radius=1.0',
        'overrides': {'train_cfg.center_sample_radius': 1.0},
    },
    'A8_lw05': {
        'name': '分类损失权重 loss_weight=0.5',
        'overrides': {'train_cfg.loss_weight': 0.5},
    },
    'A8_lw2': {
        'name': '分类损失权重 loss_weight=2.0',
        'overrides': {'train_cfg.loss_weight': 2.0},
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

    # 应用覆写
    for key_path, val in overrides.items():
        deep_set(cfg, key_path, val)
    return cfg


def dump_config(cfg, out_path):
    """将配置写入 yaml 文件（保持可读性）。"""
    import yaml
    # 自定义 Dumper，用 flow style 写列表以减少行数
    class ConfigDumper(yaml.Dumper):
        pass

    def _list_representer(dumper, data):
        if len(data) <= 6:
            return dumper.represent_sequence('tag:yaml.org,2002:seq', data, flow_style=True)
        return dumper.represent_sequence('tag:yaml.org,2002:seq', data, flow_style=False)

    ConfigDumper.add_representer(list, _list_representer)

    with open(out_path, 'w') as f:
        yaml.dump(cfg, f, Dumper=ConfigDumper, default_flow_style=False, sort_keys=False, allow_unicode=True)
    print(f"  配置已写入: {out_path}")


def run_cmd(cmd, desc):
    """运行命令，打印输出。"""
    print(f"\n  [{desc}]")
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, cwd=REPO)
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
            except:
                pass
        if 'Avearge mAP' in line or 'Average mAP' in line:
            try:
                maps['avg'] = float(line.split(':')[1].strip().replace('%', '').replace('(%)', '').strip())
            except:
                pass
    return maps


def run_ablation(exp_id, info):
    """执行单次消融实验。"""
    name = info['name']
    overrides = info.get('overrides', {})

    print(f"\n{'='*60}")
    print(f"  消融实验 {exp_id}: {name}")
    print(f"{'='*60}")

    if info.get('manual'):
        print(f"  [SKIP] 此项需手动修改源码后再运行")
        return None

    # 1. 生成覆盖后的配置文件
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
        return {'exp_id': exp_id, 'name': name, 'status': 'TRAIN_FAILED',
                'train_time_min': round(train_time, 1)}

    # 3. 评估
    # train.py names ckpt as: cfg_filename + '_' + output_name
    # config = work/abl_A1.yaml → 'abl_A1' + '_' + 'A1' = 'abl_A1_A1'
    ckpt_folder = os.path.join(CKPT_DIR, f'abl_{exp_id}_{exp_id}')
    ok, eval_out = run_cmd(
        [PYTHON, 'eval.py', exp_config, ckpt_folder],
        f'评估 {exp_id}'
    )
    eval_time = (time.time() - start) / 60

    maps = parse_map(eval_out) if ok else {}
    return {
        'exp_id': exp_id,
        'name': name,
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
            w.writerow(['实验编号', '配置变更', 'mAP@0.3', 'mAP@0.5', 'mAP@0.7',
                        'avg_mAP', '训练时间(min)', '时间戳'])


def log_result(result):
    ensure_csv_header()
    with open(CSV_FILE, 'a', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow([
            result.get('exp_id', ''),
            result.get('name', ''),
            result.get('mAP@0.3', '') if result.get('mAP@0.3') is not None else '',
            result.get('mAP@0.5', '') if result.get('mAP@0.5') is not None else '',
            result.get('mAP@0.7', '') if result.get('mAP@0.7') is not None else '',
            result.get('avg_mAP', '') if result.get('avg_mAP') is not None else '',
            result.get('train_time_min', ''),
            datetime.now().strftime('%Y-%m-%d %H:%M'),
        ])
    print(f"  结果已记录到: {CSV_FILE}")


def main():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--exp', type=str, help='实验编号, 逗号分隔, 如 A1,A2')
    group.add_argument('--all', action='store_true', help='运行所有非手动消融')
    group.add_argument('--baseline', action='store_true', help='仅训练基线模型')
    args = parser.parse_args()

    if args.baseline:
        print("训练基线模型...")
        ok, out = run_cmd(
            [PYTHON, 'train.py', BASE_CONFIG, '--output', 'baseline'],
            '基线训练'
        )
        if ok:
            run_cmd(
                [PYTHON, 'eval.py', BASE_CONFIG,
                 os.path.join(CKPT_DIR, 'thumos_i3d_baseline')],
                '基线评估'
            )
        return

    if args.all:
        exps = sorted(ABLATIONS.keys())
    else:
        exps = [e.strip() for e in args.exp.split(',')]

    for exp_id in exps:
        if exp_id not in ABLATIONS:
            print(f"  [WARN] 未知实验: {exp_id}, 跳过")
            continue
        result = run_ablation(exp_id, ABLATIONS[exp_id])
        if result:
            log_result(result)


if __name__ == '__main__':
    main()

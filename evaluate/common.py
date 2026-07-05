"""
evaluate/ 公共工具库 — GT 加载、预测加载、配置覆写、mAP 解析、CSV 管理等。

用法:
    from evaluate.common import load_gt, load_preds, parse_maps, deep_set, dump_yaml
"""

import os, csv, json, pickle, yaml
import numpy as np


# ============================================================
# 数据加载
# ============================================================

def load_gt(json_path, split='test'):
    """加载 THUMOS14 GT 标注.
    返回:
        gts: {video_id: [(start, end, label_id), ...]}
        label_names: {label_id: label_name}
    """
    with open(json_path, 'r') as f:
        db = json.load(f)
    gts = {}
    label_names = {}
    for vid, info in db['database'].items():
        if info['subset'].lower() != split:
            continue
        instances = []
        for ann in info.get('annotations', []):
            label_names[ann['label_id']] = ann['label']
            instances.append((float(ann['segment'][0]), float(ann['segment'][1]), ann['label_id']))
        gts[vid] = instances
    return gts, label_names


def load_preds(pkl_path):
    """加载预测 pickle 文件.
    返回: list of dict, 每个 dict 含 'video-id', 'segments', 'scores', 'labels'.
    """
    with open(pkl_path, 'rb') as f:
        return pickle.load(f)


def preds_by_video(preds):
    """将预测列表按 video-id 索引.
    返回: {video_id: pred_dict} (单视频预测) 或 {video_id: [pred_dict, ...]}.
    """
    # 检测是否为每个视频包含多个预测记录
    if preds and 'segments' in preds[0]:
        return {p['video-id']: p for p in preds}
    result = {}
    for p in preds:
        result.setdefault(p['video-id'], []).append(p)
    return result


# ============================================================
# 配置覆写
# ============================================================

def deep_set(d, key_path, value):
    """设置嵌套 dict 的值, 如 'model.k' → d['model']['k'] = value."""
    keys = key_path.split('.')
    for k in keys[:-1]:
        d = d[k]
    d[keys[-1]] = value


def dump_yaml(cfg, path):
    """将配置字典写入 YAML 文件 (含短列表 flow_style 优化)."""
    class Dumper(yaml.Dumper):
        pass

    def _list_repr(dumper, data):
        if len(data) <= 6:
            return dumper.represent_sequence(
                'tag:yaml.org,2002:seq', data, flow_style=True)
        return dumper.represent_sequence(
            'tag:yaml.org,2002:seq', data, flow_style=False)

    Dumper.add_representer(list, _list_repr)
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        yaml.dump(cfg, f, Dumper=Dumper, default_flow_style=False,
                  sort_keys=False, allow_unicode=True)


# ============================================================
# eval.py 输出解析
# ============================================================

def parse_maps(output_text):
    """从 eval.py stdout 解析 mAP 值.
    返回: {tiou: mAP_value, 'avg': avg_mAP}.
    """
    import re
    maps = {}
    for line in output_text.split('\n'):
        line = line.strip()
        m = re.search(r'tIoU\s*=\s*([\d.]+)\s*:\s*mAP\s*=\s*([\d.]+)', line)
        if m:
            maps[float(m.group(1))] = float(m.group(2))
        m2 = re.search(r'(?:Avearge|Average)\s+mAP\s*:\s*([\d.]+)', line)
        if m2:
            maps['avg'] = float(m2.group(1))
    return maps


# ============================================================
# CSV 结果管理
# ============================================================

def _fmt(v):
    """安全格式化数值."""
    if v is None:
        return ''
    if isinstance(v, float):
        return f'{v:.2f}'
    return str(v)


def load_ablation_csv(csv_path):
    """加载消融实验 CSV, 返回 {exp_id: row_dict}."""
    if not os.path.exists(csv_path):
        return {}
    with open(csv_path, 'r', encoding='utf-8') as f:
        return {row['实验编号'].strip(): row for row in csv.DictReader(f)}


def save_ablation_csv(all_results, csv_path):
    """写入消融实验结果到 CSV."""
    os.makedirs(os.path.dirname(csv_path) or '.', exist_ok=True)
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['实验编号', '分组', '配置变更', 'mAP@0.3', 'mAP@0.5',
                    'mAP@0.7', 'avg_mAP', '训练时间(min)', '状态', '时间戳'])
        for exp_id in sorted(all_results.keys()):
            r = all_results[exp_id]
            w.writerow([
                exp_id, r.get('group', ''), r.get('name', ''),
                _fmt(r.get('mAP@0.3')), _fmt(r.get('mAP@0.5')),
                _fmt(r.get('mAP@0.7')), _fmt(r.get('avg_mAP')),
                r.get('train_time', ''), r.get('status', ''),
                r.get('timestamp', ''),
            ])


# ============================================================
# mAP 评估辅助
# ============================================================

def compute_mAP_offline(preds_list, json_file, split='test',
                        tiou_thresholds=None):
    """离线计算 mAP (从 pickle 数据, 不依赖 eval.py).
    preds_list: list of {'video-id', 'segments', 'scores', 'labels'}
    返回: (mAP_per_tiou_array, avg_mAP).
    """
    import pandas as pd
    from libs.utils.metrics import ANETdetection

    if tiou_thresholds is None:
        tiou_thresholds = np.linspace(0.3, 0.7, 5)

    det_eval = ANETdetection(json_file, split, tiou_thresholds=tiou_thresholds)

    records = []
    for p in preds_list:
        vid = p['video-id']
        for seg, score, label in zip(p['segments'], p['scores'], p['labels']):
            records.append({
                'video-id': vid,
                't-start': float(seg[0]),
                't-end': float(seg[1]),
                'label': int(label),
                'score': float(score),
            })
    df = pd.DataFrame(records)
    mAP, avg_mAP = det_eval.evaluate(df, verbose=False)
    return mAP, avg_mAP

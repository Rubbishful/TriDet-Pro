"""
快速评估脚本：加载模型 → 推理 → 保存结果 → 离线计算 mAP

用法:
    python work/eval_fast.py --config work/abl_A1.yaml --ckpt ckpt/abl_A1_A1 --output work/results/
"""

import os, sys, argparse, pickle, json, time
import numpy as np
import torch
import torch.nn as nn
import pandas as pd

REPO = 'e:/Tridet/TriDet-Pro'
sys.path.insert(0, REPO)

from libs.core import load_config
from libs.datasets import make_dataset, make_data_loader
from libs.modeling import make_meta_arch
from libs.utils import fix_random_seed
from libs.utils.metrics import ANETdetection

PYTHON = 'E:/anaconda/envs/test/python.exe'
JSON_FILE = 'E:/thumos/annotations/thumos14.json'
SCORE_FILE = 'E:/thumos/annotations/thumos14_cls_scores.pkl'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--ckpt', required=True)
    parser.add_argument('--output', default='work/results/')
    parser.add_argument('--split', default='test')
    parser.add_argument('--max-videos', type=int, default=0, help='限制视频数(调试用), 0=全部')
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    # 1. 加载配置
    cfg = load_config(args.config)

    # 启用 score fusion + hard NMS 加速
    cfg['test_cfg']['ext_score_file'] = SCORE_FILE
    cfg['test_cfg']['multiclass_nms'] = False
    cfg['test_cfg']['nms_method'] = 'hard'       # hard NMS 比 soft 快 ~10x
    cfg['test_cfg']['voting_thresh'] = 0          # 关闭 voting
    cfg['test_cfg']['pre_nms_topk'] = 500
    cfg['test_cfg']['max_seg_num'] = 200

    print(f"Config: {args.config}")
    print(f"Checkpoint: {args.ckpt}")
    print(f"Score fusion: enabled")
    print(f"multiclass_nms: False")
    print()

    # 2. 创建数据集
    _ = fix_random_seed(0, include_cuda=True)
    val_dataset = make_dataset(cfg['dataset_name'], False, cfg['val_split'], **cfg['dataset'])

    if args.max_videos > 0:
        val_dataset.data_list = val_dataset.data_list[:args.max_videos]

    print(f"验证集视频数: {len(val_dataset.data_list)}")

    # 3. 创建模型
    model = make_meta_arch(cfg['model_name'], **cfg['model'])
    model = nn.DataParallel(model, device_ids=[0])
    model.eval()

    # 4. 加载权重
    ckpt_files = sorted([f for f in os.listdir(args.ckpt) if f.endswith('.pth.tar')])
    if not ckpt_files:
        print("错误: 无 checkpoint 文件")
        return
    ckpt_path = os.path.join(args.ckpt, ckpt_files[-1])
    print(f"加载: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location='cuda:0')
    model.load_state_dict(ckpt['state_dict_ema'])
    del ckpt

    # 5. 推理
    print(f"\n开始推理...")
    all_results = []
    t_start = time.time()

    for idx in range(len(val_dataset)):
        sample = val_dataset[idx]
        vid = sample['video_id']
        duration = sample['duration']
        fps = sample['fps']

        with torch.no_grad():
            results = model([sample])

        r = results[0]
        segs = r['segments'].cpu().numpy()
        scores = r['scores'].cpu().numpy()
        labels = r['labels'].cpu().numpy()

        all_results.append({
            'video-id': vid,
            'segments': segs,
            'scores': scores,
            'labels': labels,
        })

        if (idx + 1) % 20 == 0:
            elapsed = time.time() - t_start
            per_vid = elapsed / (idx + 1)
            eta = per_vid * (len(val_dataset) - idx - 1)
            print(f"  [{idx+1}/{len(val_dataset)}] {per_vid:.1f}s/vid, ETA: {eta:.0f}s")

    elapsed = time.time() - t_start
    print(f"推理完成: {elapsed:.0f}s ({elapsed/len(val_dataset):.1f}s/vid)")

    # 6. 保存结果
    pkl_path = os.path.join(args.output, 'predictions.pkl')
    with open(pkl_path, 'wb') as f:
        pickle.dump(all_results, f)
    print(f"预测保存: {pkl_path}")

    # 7. 离线计算 mAP
    print("\n计算 mAP...")
    val_db_vars = val_dataset.get_attributes()
    det_eval = ANETdetection(
        JSON_FILE, args.split,
        tiou_thresholds=val_db_vars['tiou_thresholds']
    )

    records = []
    for p in all_results:
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
    mAP, avg_mAP = det_eval.evaluate(df, verbose=True)

    # 8. 保存结果摘要
    summary = {
        'config': args.config,
        'checkpoint': ckpt_path,
        'mAP_per_tiou': {f'{val_db_vars["tiou_thresholds"][i]:.2f}': float(mAP[i]) * 100
                         for i in range(len(mAP))},
        'avg_mAP': float(avg_mAP) * 100,
        'n_videos': len(val_dataset),
        'n_predictions': len(records),
        'inference_time_s': elapsed,
    }

    summary_path = os.path.join(args.output, 'summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\n摘要保存: {summary_path}")

    return summary


if __name__ == '__main__':
    main()

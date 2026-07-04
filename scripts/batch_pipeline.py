"""
TriDet 批处理流水线 —— 自动处理整个文件夹的视频。

功能:
  1. 扫描输入文件夹中所有视频
  2. 批量提取 I3D 2048-dim (RGB+Flow) 特征
  3. 自动生成标注 JSON 和配置文件
  4. 运行 TriDet 推理 (GPU)
  5. 输出每个视频的检测结果

用法:
  python scripts/batch_pipeline.py \
      --video_dir ./input_videos \
      --output_dir ./pipeline_output \
      --checkpoint ./epoch_039.pth.tar \
      --device cuda:0

输出:
  output_dir/
  ├── features/          # 特征 .npy 文件
  ├── meta.json          # 特征提取元信息
  ├── config.yaml        # 自动生成的 TriDet 配置
  ├── annotations.json   # 自动生成的标注文件
  ├── predictions.pkl    # 所有视频的原始预测
  └── results.csv        # 每个视频的 Top-K 预测表格
"""

import argparse
import csv
import json
import os
import sys
import time
import pickle
import subprocess

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
sys.path.insert(0, _PROJECT_ROOT)
sys.path.insert(0, _SCRIPT_DIR)

from extract_features import (
    load_i3d_model, load_frames_from_video, build_windows,
    extract_features_for_video, compute_optical_flow_farneback,
)
from libs.core import load_config
from libs.datasets import make_dataset, make_data_loader
from libs.modeling import make_meta_arch
from libs.utils import fix_random_seed

# ---------------------------------------------------------------------------
# THUMOS14 20-class label map
# ---------------------------------------------------------------------------

LABEL_MAP = {
    0:  "BaseballPitch",
    1:  "BasketballDunk",
    2:  "Billiards",
    3:  "CleanAndJerk",
    4:  "CliffDiving",
    5:  "CricketBowling",
    6:  "CricketShot",
    7:  "Diving",
    8:  "FrisbeeCatch",
    9:  "GolfSwing",
    10: "HammerThrow",
    11: "HighJump",
    12: "JavelinThrow",
    13: "LongJump",
    14: "PoleVault",
    15: "Shotput",
    16: "SoccerPenalty",
    17: "TennisSwing",
    18: "ThrowDiscus",
    19: "VolleyballSpiking",
}


# ---------------------------------------------------------------------------
# Step 1: Feature Extraction
# ---------------------------------------------------------------------------

def extract_features_for_videos(video_dir, output_dir, args):
    """
    扫描 video_dir, 为每个视频提取 I3D 特征 (2048-dim).
    返回 video_id → {fps, duration, total_frames, num_windows} 的元信息.
    """
    import cv2
    from PIL import Image

    VIDEO_EXTS = ('.mp4', '.avi', '.mkv', '.mov', '.webm', '.MP4', '.AVI', '.MKV', '.MOV')

    video_files = []
    for f in sorted(os.listdir(video_dir)):
        if any(f.endswith(ext) for ext in VIDEO_EXTS):
            video_files.append(f)

    if not video_files:
        raise FileNotFoundError(f"No video files found in {video_dir}")

    os.makedirs(output_dir, exist_ok=True)
    device = torch.device(args.device)

    # Load models
    print(f"\n{'='*60}")
    print(f"  Step 1: Feature Extraction ({len(video_files)} videos)")
    print(f"{'='*60}")

    rgb_model_path = args.rgb_model
    flow_model_path = args.flow_model
    if not os.path.isabs(rgb_model_path):
        rgb_model_path = os.path.join(_PROJECT_ROOT, rgb_model_path)
    if not os.path.isabs(flow_model_path):
        flow_model_path = os.path.join(_PROJECT_ROOT, flow_model_path)

    print(f"[INFO] Loading RGB model: {rgb_model_path}")
    model_rgb = load_i3d_model(rgb_model_path, 'rgb', device)
    print(f"[INFO] Loading Flow model: {flow_model_path}")
    model_flow = load_i3d_model(flow_model_path, 'flow', device)

    target_size = (args.frame_width, args.frame_height)
    video_meta = {}

    for idx, video_name in enumerate(video_files):
        video_path = os.path.join(video_dir, video_name)
        video_id = os.path.splitext(video_name)[0]
        npy_path = os.path.join(output_dir, f"{video_id}.npy")

        if os.path.exists(npy_path) and not args.overwrite:
            print(f"\n[{idx+1}/{len(video_files)}] {video_id}: SKIP (features exist)")
            # Load info from existing npy
            existing = np.load(npy_path)
            video_meta[video_id] = {
                'fps': args.video_fps,
                'duration': existing.shape[0] * args.feat_stride / args.video_fps,
                'total_frames': existing.shape[0] * args.feat_stride,
                'num_windows': existing.shape[0],
            }
            continue

        print(f"\n[{idx+1}/{len(video_files)}] {video_id}: Extracting...")

        try:
            # Load frames
            frames, actual_fps = load_frames_from_video(
                video_path, target_fps=args.video_fps, target_size=target_size
            )
            total_frames = frames.shape[0]

            if total_frames < args.num_frames:
                print(f"  [WARN] Too short: {total_frames} frames < {args.num_frames}, skip")
                continue

            # Build windows
            window_indices = build_windows(total_frames, args.num_frames, args.feat_stride)

            # RGB features
            feats_rgb = extract_features_for_video(
                frames, model_rgb, window_indices,
                sample_mode=args.sample_mode, crop_size=args.crop_size,
                batch_size=args.batch_size, device=device,
            )

            # Flow features
            print(f"  Computing optical flow ({total_frames} frames)...")
            flow_frames = compute_optical_flow_farneback(frames)
            flow_total = flow_frames.shape[0]
            flow_window_indices = build_windows(flow_total, args.num_frames, args.feat_stride)
            feats_flow = extract_features_for_video(
                flow_frames, model_flow, flow_window_indices,
                sample_mode=args.sample_mode, crop_size=args.crop_size,
                batch_size=args.batch_size, device=device,
            )

            # Concatenate
            feats = np.concatenate([feats_rgb, feats_flow], axis=1).astype(np.float32)
            np.save(npy_path, feats)

            duration = total_frames / actual_fps if actual_fps > 0 else 0
            video_meta[video_id] = {
                'fps': actual_fps,
                'duration': duration,
                'total_frames': total_frames,
                'num_windows': feats.shape[0],
            }
            print(f"  [OK] frames={total_frames} windows={feats.shape[0]} dim={feats.shape[1]} [{feats.nbytes/1024/1024:.1f}MB]")

        except Exception as e:
            print(f"  [ERROR] {e}")
            continue

    # Save meta
    meta = {
        'config': {
            'mode': 'rgb+flow',
            'feat_stride': args.feat_stride,
            'num_frames': args.num_frames,
            'crop_size': args.crop_size,
            'sample_mode': args.sample_mode,
            'video_fps': args.video_fps,
        },
        'videos': video_meta,
    }
    with open(os.path.join(output_dir, 'meta.json'), 'w') as f:
        json.dump(meta, f, indent=2, ensure_ascii=False, default=str)

    return video_meta


# ---------------------------------------------------------------------------
# Step 2: Generate annotation JSON & config
# ---------------------------------------------------------------------------

def generate_annotations(video_meta, output_path):
    """生成 THUMOS14 格式的标注 JSON."""
    database = {}

    # Add dummy entry for label_dict population
    database["_dummy_"] = {
        "subset": "test",
        "fps": 25,
        "duration": 0.1,
        "annotations": [
            {"segment": [0, 0.01], "label": name, "label_id": lid}
            for lid, name in LABEL_MAP.items()
        ]
    }

    for video_id, info in video_meta.items():
        database[video_id] = {
            "subset": "test",
            "fps": info['fps'],
            "duration": info['duration'],
            "annotations": []
        }

    with open(output_path, 'w') as f:
        json.dump({"database": database}, f, indent=2, ensure_ascii=False)
    print(f"[INFO] Annotations saved: {output_path}")


def generate_config(output_dir, feat_dir, annotations_path):
    """生成 TriDet 配置文件 (与训练 config 对齐)."""
    import yaml
    try:
        import yaml
    except ImportError:
        # PyYAML should already be installed
        import yaml

    config = {
        'dataset_name': 'thumos',
        'train_split': ['validation'],
        'val_split': ['test'],
        'dataset': {
            'json_file': annotations_path,
            'feat_folder': feat_dir,
            'file_prefix': None,
            'file_ext': '.npy',
            'num_classes': 20,
            'input_dim': 2048,
            'feat_stride': 4,
            'num_frames': 16,
            'default_fps': 25,
            'downsample_rate': 1,
            'trunc_thresh': 0.5,
            'crop_ratio': [0.9, 1.0],
            'max_seq_len': 2304,
        },
        'model': {
            'fpn_type': 'identity',
            'backbone_type': 'SGP',
            'downsample_type': 'max',
            'scale_factor': 2,
            'max_buffer_len_factor': 6.0,
            'backbone_arch': [2, 2, 5],
            'n_sgp_win_size': 1,
            'embd_dim': 512,
            'embd_kernel_size': 3,
            'embd_with_ln': True,
            'fpn_dim': 512,
            'fpn_with_ln': True,
            'head_dim': 512,
            'head_kernel_size': 3,
            'head_num_layers': 3,
            'head_with_ln': True,
            'use_abs_pe': False,
            'init_conv_vars': 0,
            'regression_range': [[0, 4], [4, 8], [8, 16], [16, 32], [32, 64], [64, 10000]],
            'num_bins': 16,
            'k': 5,
            'iou_weight_power': 0.2,
            'use_trident_head': True,
            'sgp_mlp_dim': 768,
            'input_noise': 0.0005,
        },
        'opt': {
            'learning_rate': 0.0001,
            'warmup_epochs': 20,
            'epochs': 20,
            'weight_decay': 0.025,
        },
        'loader': {'batch_size': 1},
        'train_cfg': {
            'init_loss_norm': 100,
            'clip_grad_l2norm': 1.0,
            'cls_prior_prob': 0.01,
            'center_sample': 'radius',
            'center_sample_radius': 1.5,
            'droppath': 0.1,
        },
        'test_cfg': {
            'voting_thresh': 0.7,
            'pre_nms_topk': 2000,
            'max_seg_num': 2000,
            'min_score': 0.001,
            'multiclass_nms': True,
            'nms_sigma': 0.5,
            'nms_method': 'soft',
            'duration_thresh': 0.05,
            'iou_threshold': 0.1,
        },
        'output_folder': './ckpt/',
    }

    config_path = os.path.join(output_dir, 'config.yaml')
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
    print(f"[INFO] Config saved: {config_path}")
    return config_path


# ---------------------------------------------------------------------------
# Step 3: TriDet Inference
# ---------------------------------------------------------------------------

def run_tridet_inference(config_path, checkpoint_path, output_dir, device):
    """
    加载 TriDet 模型, 对数据集中所有视频进行推理.
    返回原始预测 dict.
    """
    print(f"\n{'='*60}")
    print(f"  Step 3: TriDet Inference")
    print(f"{'='*60}")

    cfg = load_config(config_path)
    cfg['devices'] = [device]

    rng = fix_random_seed(cfg.get('init_rand_seed', 1234567891), include_cuda=True)

    # Dataset
    val_dataset = make_dataset(
        cfg['dataset_name'], False, cfg['val_split'], **cfg['dataset']
    )
    val_loader = make_data_loader(
        val_dataset, False, None, 1, cfg['loader'].get('num_workers', 4)
    )

    # Model
    model = make_meta_arch(cfg['model_name'], **cfg['model'])
    model = torch.nn.DataParallel(model, device_ids=[torch.device(d).index for d in cfg['devices']])

    # Load checkpoint
    print(f"[INFO] Loading checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=cfg['devices'][0])
    if 'state_dict_ema' in ckpt:
        model.load_state_dict(ckpt['state_dict_ema'])
        print("[INFO] Using EMA weights")
    else:
        model.load_state_dict(ckpt['state_dict'])
    del ckpt

    model.eval()

    # Inference loop
    all_results = {
        'video-id': [],
        't-start': [],
        't-end': [],
        'label': [],
        'score': [],
    }

    n_videos = 0
    t_start = time.time()

    for video_list in val_loader:
        with torch.no_grad():
            output = model(video_list)  # output is list of dicts

        for vid_idx in range(len(output)):
            if output[vid_idx]['segments'].shape[0] > 0:
                all_results['video-id'].extend(
                    [output[vid_idx]['video_id']] *
                    output[vid_idx]['segments'].shape[0]
                )
                all_results['t-start'].append(output[vid_idx]['segments'][:, 0].cpu().numpy())
                all_results['t-end'].append(output[vid_idx]['segments'][:, 1].cpu().numpy())
                all_results['label'].append(output[vid_idx]['labels'].cpu().numpy())
                all_results['score'].append(output[vid_idx]['scores'].cpu().numpy())

            n_videos += 1

    t_elapsed = time.time() - t_start

    # Concatenate arrays
    all_results['t-start'] = np.concatenate(all_results['t-start']) if all_results['t-start'] else np.array([])
    all_results['t-end'] = np.concatenate(all_results['t-end']) if all_results['t-end'] else np.array([])
    all_results['label'] = np.concatenate(all_results['label']) if all_results['label'] else np.array([])
    all_results['score'] = np.concatenate(all_results['score']) if all_results['score'] else np.array([])

    print(f"\n[Profile] Processed {n_videos} videos in {t_elapsed:.1f}s ({t_elapsed/n_videos:.2f}s/video)" if n_videos else "")
    print(f"[Profile] Total detections: {len(all_results['video-id'])}")

    return all_results


# ---------------------------------------------------------------------------
# Step 4: Export Results
# ---------------------------------------------------------------------------

def export_results(predictions, video_meta, output_dir, top_k=10, min_score=0.01):
    """导出 CSV 结果和汇总表."""
    print(f"\n{'='*60}")
    print(f"  Step 4: Export Results")
    print(f"{'='*60}")

    # Save raw predictions
    pkl_path = os.path.join(output_dir, 'predictions.pkl')
    with open(pkl_path, 'wb') as f:
        pickle.dump(predictions, f)
    print(f"[INFO] Raw predictions: {pkl_path}")

    # Save CSV per-video
    csv_path = os.path.join(output_dir, 'results.csv')
    idx = np.argsort(predictions['score'])[::-1]  # sort by score desc

    with open(csv_path, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        writer.writerow(['video_id', 't_start', 't_end', 'action', 'confidence', 'duration_s'])

        written = 0
        for j in idx:
            score = predictions['score'][j]
            if score < min_score:
                continue
            video_id = predictions['video-id'][j]
            t_start = predictions['t-start'][j]
            t_end = predictions['t-end'][j]
            label_id = int(predictions['label'][j])
            action = LABEL_MAP.get(label_id, f'class_{label_id}')

            writer.writerow([
                video_id, f'{t_start:.2f}', f'{t_end:.2f}',
                action, f'{score:.4f}', f'{t_end - t_start:.2f}'
            ])
            written += 1

    print(f"[INFO] Results CSV: {csv_path} ({written} detections with score >= {min_score})")

    # Print per-video summary
    print(f"\n{'='*60}")
    print(f"  Per-Video Summary")
    print(f"{'='*60}")

    all_video_ids = set(predictions['video-id'])
    for vid in sorted(all_video_ids):
        mask = [p == vid for p in predictions['video-id']]
        video_scores = predictions['score'][np.array(mask)]
        video_labels = predictions['label'][np.array(mask)]
        video_starts = predictions['t-start'][np.array(mask)]
        video_ends = predictions['t-end'][np.array(mask)]

        # Top-k for this video
        local_idx = np.argsort(video_scores)[::-1][:top_k]
        print(f"\n  [{vid}]  ({len(video_scores)} detections total)")

        duration_str = ""
        if vid in video_meta:
            duration_str = f"  duration={video_meta[vid]['duration']:.1f}s"
        print(f"  {'':-<50}")
        print(f"  {'Rank':<5} {'Time':<16} {'Action':<20} {'Score':>8}")

        for rank, li in enumerate(local_idx):
            s = video_scores[li]
            if s < min_score and rank > 0:
                continue
            t_s = video_starts[li]
            t_e = video_ends[li]
            lbl = LABEL_MAP.get(int(video_labels[li]), f'class_{int(video_labels[li])}')
            print(f"  {rank+1:<5} [{t_s:5.1f}s - {t_e:5.1f}s]  {lbl:<20} {s:>8.4f}")

    return csv_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="TriDet Batch Pipeline — Extract I3D features + Run inference on a folder of videos",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic: process all videos in folder
  python scripts/batch_pipeline.py \\
      --video_dir ./my_videos \\
      --output_dir ./pipeline_output \\
      --checkpoint ./epoch_039.pth.tar

  # With custom model weights
  python scripts/batch_pipeline.py \\
      --video_dir D:/videos \\
      --output_dir D:/results \\
      --checkpoint ./epoch_039.pth.tar \\
      --rgb_model pytorch-i3d-feature-extraction-master/models/rgb_imagenet.pt \\
      --flow_model pytorch-i3d-feature-extraction-master/models/flow_imagenet.pt

Input format:
  python scripts/batch_pipeline.py --video_dir <输入视频文件夹> --output_dir <输出文件夹> --checkpoint <权重文件>
        """
    )

    # Required
    parser.add_argument('--video_dir', type=str, required=True,
                        help='输入视频文件夹路径')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='输出文件夹路径')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='TriDet 预训练权重 .pth.tar 文件路径')

    # Feature extraction options
    feat_group = parser.add_argument_group('Feature Extraction')
    feat_group.add_argument('--rgb_model', type=str,
                            default='pytorch-i3d-feature-extraction-master/models/rgb_imagenet.pt',
                            help='RGB I3D 权重路径')
    feat_group.add_argument('--flow_model', type=str,
                            default='pytorch-i3d-feature-extraction-master/models/flow_imagenet.pt',
                            help='Flow I3D 权重路径')
    feat_group.add_argument('--feat_stride', type=int, default=4)
    feat_group.add_argument('--num_frames', type=int, default=16)
    feat_group.add_argument('--crop_size', type=int, default=224)
    feat_group.add_argument('--sample_mode', type=str, default='center_crop',
                            choices=['center_crop', 'resize'])
    feat_group.add_argument('--video_fps', type=int, default=25)
    feat_group.add_argument('--frame_width', type=int, default=340)
    feat_group.add_argument('--frame_height', type=int, default=256)

    # Inference options
    infer_group = parser.add_argument_group('Inference')
    infer_group.add_argument('--device', type=str, default='cuda:0',
                             help='计算设备 (cuda:0 或 cpu)')
    infer_group.add_argument('--batch_size', type=int, default=16,
                             help='I3D 推理 batch size')
    infer_group.add_argument('--min_score', type=float, default=0.01,
                             help='导出 CSV 的最低置信度阈值')

    # Misc
    parser.add_argument('--overwrite', action='store_true', default=False,
                        help='覆盖已有的特征文件')
    parser.add_argument('--top_k', type=int, default=5,
                        help='每个视频显示 Top-K 预测')
    parser.add_argument('--skip_feature_extraction', action='store_true', default=False,
                        help='跳过特征提取 (使用已有的 .npy 文件)')

    args = parser.parse_args()

    # Resolve paths
    video_dir = os.path.abspath(args.video_dir)
    output_dir = os.path.abspath(args.output_dir)
    feat_dir = os.path.join(output_dir, 'features')
    annotations_path = os.path.join(output_dir, 'annotations.json')

    print("=" * 60)
    print("  TriDet Batch Pipeline")
    print("=" * 60)
    print(f"  Video dir:   {video_dir}")
    print(f"  Output dir:  {output_dir}")
    print(f"  Checkpoint:  {args.checkpoint}")
    print(f"  Device:      {args.device}")

    # Step 1: Feature Extraction
    t0 = time.time()
    video_meta = {}
    if not args.skip_feature_extraction:
        video_meta = extract_features_for_videos(video_dir, feat_dir, args)
    else:
        # Try to load existing meta
        meta_path = os.path.join(feat_dir, 'meta.json')
        if os.path.exists(meta_path):
            with open(meta_path, 'r') as f:
                meta = json.load(f)
            video_meta = meta.get('videos', {})
            print(f"[INFO] Loaded existing meta: {len(video_meta)} videos")
        else:
            print("[WARN] --skip_feature_extraction but no meta.json found")

    if not video_meta:
        print("[ERROR] No videos processed. Check input directory.")
        sys.exit(1)

    t1 = time.time()

    # Step 2: Generate annotations & config
    generate_annotations(video_meta, annotations_path)
    config_path = generate_config(output_dir, feat_dir, annotations_path)
    t2 = time.time()

    # Step 3: TriDet Inference
    predictions = run_tridet_inference(config_path, args.checkpoint, output_dir, args.device)
    t3 = time.time()

    # Step 4: Export
    csv_path = export_results(predictions, video_meta, output_dir,
                              top_k=args.top_k, min_score=args.min_score)
    t4 = time.time()

    # Summary
    print(f"\n{'='*60}")
    print(f"  Pipeline Complete")
    print(f"{'='*60}")
    print(f"  Videos processed:   {len(video_meta)}")
    print(f"  Total detections:   {len(predictions['video-id'])}")
    print(f"  Feature extraction: {t1 - t0:.0f}s")
    print(f"  Config generation:  {t2 - t1:.0f}s")
    print(f"  TriDet inference:   {t3 - t2:.0f}s")
    print(f"  Result export:      {t4 - t3:.0f}s")
    print(f"  Total:              {t4 - t0:.0f}s")
    print(f"\n  Output files:")
    print(f"    Features:       {feat_dir}/")
    print(f"    Raw predictions: {output_dir}/predictions.pkl")
    print(f"    Results CSV:    {csv_path}")
    print(f"    Config:         {config_path}")
    print(f"    Annotations:    {annotations_path}")


if __name__ == '__main__':
    main()

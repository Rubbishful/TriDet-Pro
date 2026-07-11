"""
TriDet Batch Pipeline — process all videos in a folder end-to-end.

Workflow:
  1. Scan input folder for video files
  2. Extract I3D 2048-dim (RGB+Flow) features in batch
  3. Auto-generate annotation JSON and config YAML
  4. Run TriDet inference (GPU)
  5. Export per-video detection results

Usage:
  python E2E/batch_pipeline.py \
      --video_dir ./data/videos/ \
      --output_dir ./result/ \
      --ckpt ckpt/thumos_i3d_baseline/epoch_039.pth.tar \
      --device cuda:0

Output:
  output_dir/
  ├── features/          # .npy feature files
  ├── meta.json          # feature extraction metadata
  ├── config.yaml        # auto-generated TriDet config
  ├── annotations.json   # auto-generated annotation file
  ├── predictions.pkl    # raw predictions for all videos
  └── results.csv        # Top-K prediction table per video
"""

import argparse
import csv
import json
import os
import pickle
import sys
import time
from datetime import timedelta

import numpy as np
import torch
import yaml

# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
sys.path.insert(0, _PROJECT_ROOT)
sys.path.insert(0, _SCRIPT_DIR)

from E2E.module.features import (
    build_windows,
    extract_features_for_video,
    load_i3d_model,
)
from E2E.module.flow import compute_optical_flow
from E2E.module.loader import load_frames_from_video
from E2E.module.visualizer import (
    create_annotated_video,
    extract_keyframes,
    filter_actions_by_video,
)
from libs.core import load_config
from libs.datasets import make_data_loader, make_dataset
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
    """Extract I3D features (2048-dim RGB+Flow) for all videos in a folder.

    Scans *video_dir* for video files, loads the I3D RGB and Flow models,
    then processes each video through frame loading, window building,
    RGB feature extraction, optical-flow computation, and Flow feature
    extraction.  Features are concatenated and saved as ``.npy`` files.

    Args:
        video_dir: Path to the folder containing input video files.
        output_dir: Path where ``.npy`` feature files and ``meta.json``
            will be written.
        args: Parsed command-line arguments (argparse.Namespace). Expected
            attributes: device, rgb_model, flow_model, frame_width,
            frame_height, target_fps, num_frames, feat_stride, sample_mode,
            crop_size, batch_size, overwrite.

    Returns:
        dict[str, dict]: Mapping from ``video_id`` to metadata dict with
            keys ``fps``, ``duration``, ``total_frames``, ``num_windows``.

    Raises:
        FileNotFoundError: If no video files are found in *video_dir*.
    """

    VIDEO_EXTS = (".mp4", ".avi", ".mkv", ".mov", ".webm", ".MP4", ".AVI", ".MKV", ".MOV")

    if not os.path.isdir(video_dir):
        raise FileNotFoundError(f"Video directory not found: {video_dir}")

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
    model_rgb = load_i3d_model(rgb_model_path, "rgb", device)
    print(f"[INFO] Loading Flow model: {flow_model_path}")
    model_flow = load_i3d_model(flow_model_path, "flow", device)

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
                "fps": args.target_fps,
                "duration": existing.shape[0] * args.feat_stride / args.target_fps,
                "total_frames": existing.shape[0] * args.feat_stride,
                "num_windows": existing.shape[0],
                "feature_extraction_time": 0,
            }
            continue

        print(f"\n[{idx+1}/{len(video_files)}] {video_id}: Extracting...")
        t_vid = time.time()

        try:
            # Load frames
            frames, actual_fps = load_frames_from_video(
                video_path, target_fps=args.target_fps, target_size=target_size
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
            flow_frames = compute_optical_flow(frames)
            del frames  # free RGB frames; only flow_frames needed going forward
            flow_total = flow_frames.shape[0]
            flow_window_indices = build_windows(flow_total, args.num_frames, args.feat_stride)
            feats_flow = extract_features_for_video(
                flow_frames, model_flow, flow_window_indices,
                sample_mode=args.sample_mode, crop_size=args.crop_size,
                batch_size=args.batch_size, device=device,
            )

            # Align RGB and Flow window counts (flow has T-1 frames; may differ by 1)
            min_wins = min(feats_rgb.shape[0], feats_flow.shape[0])
            feats_rgb = feats_rgb[:min_wins]
            feats_flow = feats_flow[:min_wins]

            # Concatenate
            feats = np.concatenate([feats_rgb, feats_flow], axis=1).astype(np.float32)
            np.save(npy_path, feats)

            duration = total_frames / actual_fps if actual_fps > 0 else 0
            video_meta[video_id] = {
                "fps": actual_fps,
                "duration": duration,
                "total_frames": total_frames,
                "num_windows": feats.shape[0],
                "feature_extraction_time": round(time.time() - t_vid, 2),
            }
            print(f"  [OK] frames={total_frames} windows={feats.shape[0]} dim={feats.shape[1]} [{feats.nbytes/1024/1024:.1f}MB]")

        except Exception as e:
            print(f"  [ERROR] {e}")
            continue

    # Save meta
    meta = {
        "config": {
            "mode": "rgb+flow",
            "feat_stride": args.feat_stride,
            "num_frames": args.num_frames,
            "crop_size": args.crop_size,
            "sample_mode": args.sample_mode,
            "video_fps": args.target_fps,
        },
        "videos": video_meta,
    }
    with open(os.path.join(output_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False, default=str)

    return video_meta


# ---------------------------------------------------------------------------
# Step 2: Generate annotation JSON & config
# ---------------------------------------------------------------------------

def generate_annotations(video_meta, output_path):
    """Generate a THUMOS14-format annotation JSON file.

    Args:
        video_meta: Mapping from video_id to metadata dict (as returned by
            :func:`extract_features_for_videos`).
        output_path: File path for the generated ``annotations.json``.
    """
    database = {}

    # Add dummy entry for label_dict population
    database["_dummy_"] = {
        "subset": "test",
        "fps": 25,
        "duration": 0.1,
        "annotations": [
            {"segment": [0, 0.01], "label": name, "label_id": lid}
            for lid, name in LABEL_MAP.items()
        ],
    }

    for video_id, info in video_meta.items():
        database[video_id] = {
            "subset": "test",
            "fps": info["fps"],
            "duration": info["duration"],
            "annotations": [],
        }

    with open(output_path, "w") as f:
        json.dump({"database": database}, f, indent=2, ensure_ascii=False)
    print(f"[INFO] Annotations saved: {output_path}")


def generate_config(output_dir, feat_dir, annotations_path):
    """Generate a TriDet inference config YAML aligned with training config.

    Args:
        output_dir: Root output directory for the pipeline run.
        feat_dir: Directory containing ``.npy`` feature files.
        annotations_path: Path to the annotations JSON file.

    Returns:
        str: Path to the generated ``config.yaml``.
    """
    config = {
        "dataset_name": "thumos",
        "train_split": ["validation"],
        "val_split": ["test"],
        "dataset": {
            "json_file": annotations_path,
            "feat_folder": feat_dir,
            "file_prefix": None,
            "file_ext": ".npy",
            "num_classes": 20,
            "input_dim": 2048,
            "feat_stride": 4,
            "num_frames": 16,
            "default_fps": 25,
            "downsample_rate": 1,
            "trunc_thresh": 0.5,
            "crop_ratio": [0.9, 1.0],
            "max_seq_len": 2304,
        },
        "model": {
            "fpn_type": "identity",
            "backbone_type": "SGP",
            "downsample_type": "max",
            "scale_factor": 2,
            "max_buffer_len_factor": 6.0,
            "backbone_arch": [2, 2, 5],
            "n_sgp_win_size": 1,
            "embd_dim": 512,
            "embd_kernel_size": 3,
            "embd_with_ln": True,
            "fpn_dim": 512,
            "fpn_with_ln": True,
            "head_dim": 512,
            "head_kernel_size": 3,
            "head_num_layers": 3,
            "head_with_ln": True,
            "use_abs_pe": False,
            "init_conv_vars": 0,
            "regression_range": [[0, 4], [4, 8], [8, 16], [16, 32], [32, 64], [64, 10000]],
            "num_bins": 16,
            "k": 5,
            "iou_weight_power": 0.2,
            "use_trident_head": True,
            "sgp_mlp_dim": 768,
            "input_noise": 0.0005,
        },
        "opt": {
            "learning_rate": 0.0001,
            "warmup_epochs": 20,
            "epochs": 20,
            "weight_decay": 0.025,
        },
        "loader": {"batch_size": 1},
        "train_cfg": {
            "init_loss_norm": 100,
            "clip_grad_l2norm": 1.0,
            "cls_prior_prob": 0.01,
            "center_sample": "radius",
            "center_sample_radius": 1.5,
            "droppath": 0.1,
        },
        "test_cfg": {
            "voting_thresh": 0.7,
            "pre_nms_topk": 2000,
            "max_seg_num": 2000,
            "min_score": 0.001,
            "multiclass_nms": True,
            "nms_sigma": 0.5,
            "nms_method": "soft",
            "duration_thresh": 0.05,
            "iou_threshold": 0.1,
        },
        "output_folder": "./ckpt/",
    }

    config_path = os.path.join(output_dir, "config.yaml")
    with open(config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
    print(f"[INFO] Config saved: {config_path}")
    return config_path


# ---------------------------------------------------------------------------
# Step 3: TriDet Inference
# ---------------------------------------------------------------------------

def run_tridet_inference(config_path, checkpoint_path, output_dir, device):
    """Load the TriDet model and run inference on all videos in the dataset.

    Args:
        config_path: Path to the TriDet config YAML file.
        checkpoint_path: Path to the ``.pth.tar`` checkpoint.
        output_dir: Output directory (unused; reserved for future use).
        device: Torch device string (e.g. ``"cuda:0"``).

    Returns:
        dict: Raw predictions with keys ``video-id``, ``t-start``,
            ``t-end``, ``label``, ``score``.
    """
    print(f"\n{'='*60}")
    print(f"  Step 3: TriDet Inference")
    print(f"{'='*60}")

    cfg = load_config(config_path)
    cfg["devices"] = [device]

    rng = fix_random_seed(cfg.get("init_rand_seed", 1234567891), include_cuda=True)

    # Dataset
    val_dataset = make_dataset(
        cfg["dataset_name"], False, cfg["val_split"], **cfg["dataset"]
    )
    val_loader = make_data_loader(
        val_dataset, False, None, 1, cfg["loader"].get("num_workers", 4)
    )

    # Model
    model = make_meta_arch(cfg["model_name"], **cfg["model"])
    model = torch.nn.DataParallel(model, device_ids=[torch.device(d).index for d in cfg["devices"]])

    # Load checkpoint
    print(f"[INFO] Loading checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=cfg["devices"][0])
    if "state_dict_ema" in ckpt:
        model.load_state_dict(ckpt["state_dict_ema"])
        print("[INFO] Using EMA weights")
    else:
        model.load_state_dict(ckpt["state_dict"])
    del ckpt

    model.eval()

    # Inference loop
    all_results = {
        "video-id": [],
        "t-start": [],
        "t-end": [],
        "label": [],
        "score": [],
    }

    n_videos = 0
    t_start = time.time()

    for video_list in val_loader:
        with torch.no_grad():
            output = model(video_list)  # output is list of dicts

        for vid_idx in range(len(output)):
            if output[vid_idx]["segments"].shape[0] > 0:
                all_results["video-id"].extend(
                    [output[vid_idx]["video_id"]] *
                    output[vid_idx]["segments"].shape[0]
                )
                all_results["t-start"].append(output[vid_idx]["segments"][:, 0].cpu().numpy())
                all_results["t-end"].append(output[vid_idx]["segments"][:, 1].cpu().numpy())
                all_results["label"].append(output[vid_idx]["labels"].cpu().numpy())
                all_results["score"].append(output[vid_idx]["scores"].cpu().numpy())

            n_videos += 1

    t_elapsed = time.time() - t_start

    # Concatenate arrays
    all_results["t-start"] = np.concatenate(all_results["t-start"]) if all_results["t-start"] else np.array([])
    all_results["t-end"] = np.concatenate(all_results["t-end"]) if all_results["t-end"] else np.array([])
    all_results["label"] = np.concatenate(all_results["label"]) if all_results["label"] else np.array([])
    all_results["score"] = np.concatenate(all_results["score"]) if all_results["score"] else np.array([])

    if n_videos:
        print(f"\n[INFO] Processed {n_videos} videos in {t_elapsed:.1f}s ({t_elapsed/n_videos:.2f}s/video)")
    print(f"[INFO] Total detections: {len(all_results['video-id'])}")

    return all_results


# ---------------------------------------------------------------------------
# Step 4: Export Results
# ---------------------------------------------------------------------------

def export_results(predictions, video_meta, output_dir, top_k=10, min_score=0.01):
    """Export detection results to CSV and print per-video summaries.

    Args:
        predictions: Raw predictions dict from :func:`run_tridet_inference`.
        video_meta: Mapping from video_id to metadata dict.
        output_dir: Directory for ``predictions.pkl`` and ``results.csv``.
        top_k: Number of top predictions to display per video.
        min_score: Minimum confidence threshold for CSV output.

    Returns:
        str: Path to the generated ``results.csv``.
    """
    print(f"\n{'='*60}")
    print(f"  Step 4: Export Results")
    print(f"{'='*60}")

    # Save raw predictions
    pkl_path = os.path.join(output_dir, "predictions.pkl")
    with open(pkl_path, "wb") as f:
        pickle.dump(predictions, f)
    print(f"[INFO] Raw predictions: {pkl_path}")

    # Save CSV per-video
    csv_path = os.path.join(output_dir, "results.csv")
    idx = np.argsort(predictions["score"])[::-1]  # sort by score desc

    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["video_id", "t_start", "t_end", "action", "confidence", "duration_s"])

        written = 0
        for j in idx:
            score = predictions["score"][j]
            if score < min_score:
                continue
            video_id = predictions["video-id"][j]
            t_start = predictions["t-start"][j]
            t_end = predictions["t-end"][j]
            label_id = int(predictions["label"][j])
            action = LABEL_MAP.get(label_id, f"class_{label_id}")

            writer.writerow([
                video_id, f"{t_start:.2f}", f"{t_end:.2f}",
                action, f"{score:.4f}", f"{t_end - t_start:.2f}"
            ])
            written += 1

    print(f"[INFO] Results CSV: {csv_path} ({written} detections with score >= {min_score})")

    # Print per-video summary
    print(f"\n{'='*60}")
    print(f"  Per-Video Summary")
    print(f"{'='*60}")

    all_video_ids = set(predictions["video-id"])
    for vid in sorted(all_video_ids):
        mask = [p == vid for p in predictions["video-id"]]
        video_scores = predictions["score"][np.array(mask)]
        video_labels = predictions["label"][np.array(mask)]
        video_starts = predictions["t-start"][np.array(mask)]
        video_ends = predictions["t-end"][np.array(mask)]

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
            lbl = LABEL_MAP.get(int(video_labels[li]), f"class_{int(video_labels[li])}")
            print(f"  {rank+1:<5} [{t_s:5.1f}s - {t_e:5.1f}s]  {lbl:<20} {s:>8.4f}")

    return csv_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    """Parse command-line arguments and run the full batch pipeline."""
    parser = argparse.ArgumentParser(
        description="TriDet Batch Pipeline — Extract I3D features + Run inference on a folder of videos",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic: process all videos in folder
  python E2E/batch_pipeline.py \\
      --video_dir ./my_videos \\
      --output_dir ./pipeline_output \\
      --ckpt ckpt/thumos_i3d_baseline/epoch_039.pth.tar

  # With custom model weights
  python E2E/batch_pipeline.py \\
      --video_dir ./videos \\
      --output_dir ./results \\
      --ckpt ckpt/thumos_i3d_baseline/epoch_039.pth.tar \\
      --rgb_model E2E/model/rgb_imagenet.pt \\
      --flow_model E2E/model/flow_imagenet.pt

Input format:
  python E2E/batch_pipeline.py --video_dir <input_folder> --output_dir <output_folder> --ckpt <weights>
        """
    )

    # Required
    parser.add_argument("--video_dir", type=str, required=True,
                        help="Path to the input video folder")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Path to the output folder")
    parser.add_argument("--ckpt", type=str, default="ckpt/thumos_i3d_baseline/epoch_039.pth.tar",
                        help="Path to the TriDet checkpoint .pth.tar file")

    # Feature extraction options
    feat_group = parser.add_argument_group("Feature Extraction")
    feat_group.add_argument("--rgb_model", type=str,
                            default="E2E/model/rgb_imagenet.pt",
                            help="Path to the RGB I3D weights")
    feat_group.add_argument("--flow_model", type=str,
                            default="E2E/model/flow_imagenet.pt",
                            help="Path to the Flow I3D weights")
    feat_group.add_argument("--feat_stride", type=int, default=4)
    feat_group.add_argument("--num_frames", type=int, default=16)
    feat_group.add_argument("--crop_size", type=int, default=224)
    feat_group.add_argument("--sample_mode", type=str, default="center_crop",
                            choices=["center_crop", "resize"])
    feat_group.add_argument("--target_fps", type=int, default=25)
    feat_group.add_argument("--frame_width", type=int, default=340)
    feat_group.add_argument("--frame_height", type=int, default=256)

    # Inference options
    infer_group = parser.add_argument_group("Inference")
    infer_group.add_argument("--device", type=str, default="cuda:0",
                             help="Compute device (cuda:0 or cpu)")
    infer_group.add_argument("--batch_size", type=int, default=16,
                             help="I3D inference batch size")
    infer_group.add_argument("--min_score", type=float, default=0.01,
                             help="Minimum confidence threshold for CSV export")

    # Misc
    parser.add_argument("--overwrite", action="store_true", default=False,
                        help="Overwrite existing feature files")
    parser.add_argument("--top_k", type=int, default=5,
                        help="Number of top-k predictions to display per video")
    parser.add_argument("--skip_feature_extraction", action="store_true", default=False,
                        help="Skip feature extraction (use existing .npy files)")

    # Visualization / YOLO options
    viz_group = parser.add_argument_group("Visualization (optional)")
    viz_group.add_argument("--visualize", action="store_true", default=False,
                        help="Generate annotated videos with YOLO boxes + action labels")
    viz_group.add_argument("--yolo_model", type=str, default="E2E/model/yolov8n.pt",
                        help="YOLO model name or path (default: E2E/model/yolov8n.pt)")
    viz_group.add_argument("--yolo_conf", type=float, default=0.3,
                        help="YOLO confidence threshold (default: 0.3)")
    viz_group.add_argument("--no_timeline", action="store_true", default=False,
                        help="Disable the bottom timeline strip in output video")
    viz_group.add_argument("--save_keyframes", action="store_true", default=False,
                        help="Save top-K keyframe images (requires --visualize)")
    viz_group.add_argument("--keyframe_top_k", type=int, default=5,
                        help="Max keyframes to save per video (default: 5)")

    args = parser.parse_args()

    # Resolve paths
    video_dir = os.path.abspath(args.video_dir)
    output_dir = os.path.abspath(args.output_dir)
    feat_dir = os.path.join(output_dir, "features")
    annotations_path = os.path.join(output_dir, "annotations.json")

    # Validate inputs
    if not os.path.isdir(video_dir):
        print(f"[ERROR] Video directory not found: {video_dir}")
        sys.exit(1)
    if not args.skip_feature_extraction and not os.path.isfile(args.ckpt):
        print(f"[ERROR] Checkpoint not found: {args.ckpt}")
        sys.exit(1)

    print("=" * 60)
    print("  TriDet Batch Pipeline")
    print("=" * 60)
    print(f"  Video dir:   {video_dir}")
    print(f"  Output dir:  {output_dir}")
    print(f"  Checkpoint:  {args.ckpt}")
    print(f"  Device:      {args.device}")

    # Step 1: Feature Extraction
    t0 = time.time()
    video_meta = {}
    if not args.skip_feature_extraction:
        video_meta = extract_features_for_videos(video_dir, feat_dir, args)
    else:
        # Try to load existing meta
        meta_path = os.path.join(feat_dir, "meta.json")
        if os.path.exists(meta_path):
            with open(meta_path, "r") as f:
                meta = json.load(f)
            video_meta = meta.get("videos", {})
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
    predictions = run_tridet_inference(config_path, args.ckpt, output_dir, args.device)
    t3 = time.time()

    # Step 4: Export
    csv_path = export_results(predictions, video_meta, output_dir,
                              top_k=args.top_k, min_score=args.min_score)
    t4 = time.time()

    # ========================================================================
    # Step 5: YOLO + Annotated Videos (optional)
    # ========================================================================
    t5 = t4
    if args.visualize:
        print(f"\n{'='*60}")
        print(f"  Step 5: YOLO Person Detection + Annotated Videos")
        print(f"{'='*60}")
        t5_start = time.time()

        from libs.subject.detector import SubjectDetector

        print(f"[YOLO] Loading model: {args.yolo_model}")
        detector = SubjectDetector(model_name=args.yolo_model, device=args.device)

        n_done = 0
        for video_id in sorted(video_meta.keys()):
            # Find original video file
            video_path = None
            VIDEO_EXTS = (".mp4", ".avi", ".mkv", ".mov", ".webm", ".MP4", ".AVI", ".MKV", ".MOV")
            for ext in VIDEO_EXTS:
                candidate = os.path.join(video_dir, f"{video_id}{ext}")
                if os.path.isfile(candidate):
                    video_path = candidate
                    break
            if video_path is None:
                print(f"  [SKIP] {video_id}: video file not found in {video_dir}")
                continue

            # Get actions for this video
            actions = filter_actions_by_video(
                [], predictions, video_id, LABEL_MAP
            )
            if not actions:
                print(f"  [SKIP] {video_id}: no action detections")
                continue

            output_video_path = os.path.join(output_dir, f"{video_id}_annotated.mp4")
            print(f"  [{n_done+1}/{len(video_meta)}] {video_id}: {len(actions)} actions -> {os.path.basename(output_video_path)}")

            try:
                create_annotated_video(
                    video_path=video_path,
                    output_path=output_video_path,
                    action_results=actions,
                    detector=detector,
                    conf_threshold=args.yolo_conf,
                    show_timeline=not args.no_timeline,
                    progress=False,  # too noisy in batch mode
                )

                # Keyframe extraction
                if args.save_keyframes:
                    kf_paths = extract_keyframes(
                        video_path=video_path,
                        action_results=actions,
                        output_dir=output_dir,
                        video_name=video_id,
                        detector=detector,
                        conf_threshold=args.yolo_conf,
                        top_k=args.keyframe_top_k,
                    )
                    if kf_paths:
                        print(f"    [KF] {len(kf_paths)} keyframes saved")

                n_done += 1
            except Exception as e:
                print(f"  [ERROR] {video_id}: {e}")
                continue

        t5 = time.time()
        print(f"  Done: {n_done} annotated videos in {t5 - t5_start:.0f}s")

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
    if args.visualize:
        print(f"  Video visualization:{t5 - t4:.0f}s")
    print(f"  Total:              {t5 - t0:.0f}s")
    print(f"\n  Output files:")
    print(f"    Features:       {feat_dir}/")
    print(f"    Raw predictions: {output_dir}/predictions.pkl")
    print(f"    Results CSV:    {csv_path}")
    print(f"    Config:         {config_path}")
    print(f"    Annotations:    {annotations_path}")

    # --- Write profiling JSON ---
    per_video = {}
    for vid, info in video_meta.items():
        per_video[vid] = {
            "duration_hms": str(timedelta(seconds=int(info.get("duration", 0)))),
            "duration_seconds": round(info.get("duration", 0), 2),
            "total_frames": info.get("total_frames", 0),
            "fps": round(info.get("fps", 0), 2) if info.get("fps", 0) else 0,
            "feature_extraction_time": round(info.get("feature_extraction_time", 0), 2),
            "num_windows": info.get("num_windows", 0),
        }

    profiling = {
        "video_dir": video_dir,
        "total_videos": len(video_meta),
        "total_detections": len(predictions["video-id"]),
        "aggregate_stages": {
            "feature_extraction": round(t1 - t0, 2),
            "config_generation": round(t2 - t1, 2),
            "tridet_inference": round(t3 - t2, 2),
            "result_export": round(t4 - t3, 2),
        },
        "per_video": per_video,
        "total_time_seconds": round(t5 - t0, 2),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    if args.visualize:
        profiling["aggregate_stages"]["visualization"] = round(t5 - t4, 2)

    os.makedirs(output_dir, exist_ok=True)
    profile_path = os.path.join(output_dir, "pipeline_profile.json")
    with open(profile_path, "w", encoding="utf-8") as f:
        json.dump(profiling, f, indent=2, ensure_ascii=False)
    print(f"[INFO] Profile saved: {profile_path}")


if __name__ == "__main__":
    main()

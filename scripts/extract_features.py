"""
Feature extraction script — extract I3D features from video frames in TriDet-compatible format.

Model: InceptionI3D (Inception-v1 inflated to 3D)
Pretrained weights: Kinetics-400 (rgb_imagenet.pt / flow_imagenet.pt)

Input:
  - Raw video files (--video_dir): Real-time decoding via OpenCV
  - Pre-extracted frame images (--frames_dir): Follows pytorch-i3d img_*.jpg convention

Output:
  - .npy files, shape (T, C), dtype float32, directly loadable by TriDet DataLoader
  - meta.json: Records extraction config and per-video statistics

Usage:
  # THUMOS14: Extract RGB features from pre-extracted frames (1024-dim)
  python scripts/extract_features.py \
      --frames_dir ./thumos_frames \
      --output_dir ./thumos_i3d_features \
      --mode rgb \
      --rgb_model pytorch-i3d-feature-extraction-master/models/rgb_imagenet.pt \
      --feat_stride 4

  # THUMOS14: Extract RGB+Flow features from raw videos (2048-dim)
  python scripts/extract_features.py \
      --video_dir D:/Code/THUMOS14/thumos/videos \
      --output_dir D:/Code/THUMOS14/thumos/i3d_features \
      --mode rgb+flow \
      --rgb_model pytorch-i3d-feature-extraction-master/models/rgb_imagenet.pt \
      --flow_model pytorch-i3d-feature-extraction-master/models/flow_imagenet.pt \
      --feat_stride 4 \
      --video_fps 25

  # Single video: Specify video path, automatic video_id from filename
  python scripts/extract_features.py \
      --video_path /path/to/video.mp4 \
      --output_dir ./single_feat \
      --mode rgb
"""

import argparse
import json
import os
import sys

import numpy as np
import torch

# Ensure project root is on sys.path so E2E module can be found
_PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJ_ROOT not in sys.path:
    sys.path.insert(0, _PROJ_ROOT)

from E2E.loader import (
    load_frames_from_dir,
    load_frames_from_video,
    find_videos_from_dir,
    find_frames_from_dir,
)
from E2E.flow import compute_optical_flow
from E2E.features import build_windows, load_i3d_model, extract_features_for_video


# ============================================================================
# Main
# ============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Extract I3D features from video frames for TriDet",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # RGB only from frame images
  python extract_features.py --frames_dir ./frames --output_dir ./feats --mode rgb

  # RGB+Flow (2048-dim) from raw videos
  python extract_features.py --video_dir ./videos --output_dir ./feats --mode rgb+flow

  # Single video
  python extract_features.py --video_path test.mp4 --output_dir ./feats --mode rgb
        """,
    )

    # ---- Input source (choose one) ----
    input_group = parser.add_argument_group("Input source (choose one)")
    input_group.add_argument("--video_dir", type=str, default=None,
                             help="Directory of raw video files")
    input_group.add_argument("--frames_dir", type=str, default=None,
                             help="Directory of per-video frame subdirectories")
    input_group.add_argument("--video_path", type=str, default=None,
                             help="Path to a single video file")

    # ---- Output ----
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for .npy feature files")

    # ---- Model ----
    model_group = parser.add_argument_group("Model weights")
    model_group.add_argument("--rgb_model", type=str,
                             default="pytorch-i3d-feature-extraction-master/models/rgb_imagenet.pt",
                             help="Path to RGB I3D weights")
    model_group.add_argument("--flow_model", type=str,
                             default="pytorch-i3d-feature-extraction-master/models/flow_imagenet.pt",
                             help="Path to Flow I3D weights")
    model_group.add_argument("--mode", type=str, default="rgb",
                             choices=["rgb", "flow", "rgb+flow"],
                             help="Feature mode: rgb (1024-dim), flow (1024-dim), "
                                  "rgb+flow (2048-dim)")

    # ---- Feature extraction parameters ----
    feat_group = parser.add_argument_group("Feature extraction parameters")
    feat_group.add_argument("--feat_stride", type=int, default=4,
                            help="Frame stride between consecutive windows (default: 4)")
    feat_group.add_argument("--num_frames", type=int, default=16,
                            help="Frames per window (default: 16)")
    feat_group.add_argument("--crop_size", type=int, default=224,
                            help="Spatial crop size in pixels (default: 224)")
    feat_group.add_argument("--sample_mode", type=str, default="center_crop",
                            choices=["center_crop", "resize"],
                            help="Spatial sampling mode (default: center_crop)")

    # ---- Performance ----
    perf_group = parser.add_argument_group("Performance")
    perf_group.add_argument("--batch_size", type=int, default=16,
                            help="Batch size for GPU processing (default: 16)")
    perf_group.add_argument("--device", type=str, default="cuda:0",
                            help="Computation device (default: cuda:0)")

    # ---- Video/Frame reading ----
    io_group = parser.add_argument_group("Video/Frame reading")
    io_group.add_argument("--video_fps", type=int, default=25,
                          help="Target FPS when reading videos (default: 25)")
    io_group.add_argument("--video_ext", type=str, nargs="+",
                          default=[".mp4", ".avi", ".mkv", ".webm"],
                          help="Video file extensions (default: .mp4 .avi .mkv .webm)")
    io_group.add_argument("--frame_pattern", type=str, default="img_{:06d}.jpg",
                          help="Frame file naming pattern (default: img_{:06d}.jpg)")
    io_group.add_argument("--frame_width", type=int, default=340,
                          help="Input frame width, before crop (default: 340)")
    io_group.add_argument("--frame_height", type=int, default=256,
                          help="Input frame height, before crop (default: 256)")

    # ---- Miscellaneous ----
    misc_group = parser.add_argument_group("Miscellaneous")
    misc_group.add_argument("--overwrite", action="store_true", default=False,
                            help="Overwrite existing .npy files")
    misc_group.add_argument("--dataset_json", type=str, default=None,
                            help="Path to THUMOS14-style annotation JSON for metadata")

    args = parser.parse_args()

    # ---- Validate: input sources are mutually exclusive ----
    sources = [args.video_dir, args.frames_dir, args.video_path]
    num_sources = sum(1 for s in sources if s is not None)
    if num_sources != 1:
        parser.error("Exactly one of --video_dir, --frames_dir, --video_path required")

    # ---- Validate model weights ----
    if args.mode in ("rgb", "rgb+flow"):
        if not os.path.exists(args.rgb_model):
            parser.error(f"RGB model not found: {args.rgb_model}")
    if args.mode in ("flow", "rgb+flow"):
        if not os.path.exists(args.flow_model):
            parser.error(f"Flow model not found: {args.flow_model}")

    # ---- Device ----
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device: {device}")

    # ---- Load models ----
    models = {}
    if args.mode in ("rgb", "rgb+flow"):
        print(f"[INFO] Loading RGB model: {args.rgb_model}")
        models["rgb"] = load_i3d_model(args.rgb_model, "rgb", device)
    if args.mode in ("flow", "rgb+flow"):
        print(f"[INFO] Loading Flow model: {args.flow_model}")
        models["flow"] = load_i3d_model(args.flow_model, "flow", device)

    # ---- Discover videos ----
    if args.video_path:
        vid = os.path.splitext(os.path.basename(args.video_path))[0]
        video_list = [(vid, args.video_path)]
        use_video = True
    elif args.video_dir:
        video_list = find_videos_from_dir(args.video_dir, tuple(args.video_ext))
        use_video = True
        print(f"[INFO] Found {len(video_list)} videos in {args.video_dir}")
    else:
        # frames_dir: each subdirectory = one video's frame sequence
        prefix = None  # No forced 'v' prefix, compatible with more naming schemes
        video_list = find_frames_from_dir(args.frames_dir, prefix_filter=prefix)
        use_video = False
        print(f"[INFO] Found {len(video_list)} frame directories in {args.frames_dir}")

    if len(video_list) == 0:
        print("[ERROR] No videos/frames found!")
        sys.exit(1)

    target_size = (args.frame_width, args.frame_height)

    # ---- Create output directory ----
    os.makedirs(args.output_dir, exist_ok=True)

    # ---- Load annotations (optional) ----
    annotation_db = {}
    if args.dataset_json and os.path.exists(args.dataset_json):
        with open(args.dataset_json, "r") as f:
            ann_data = json.load(f)
        annotation_db = ann_data.get("database", ann_data)
        print(f"[INFO] Loaded annotation data for {len(annotation_db)} videos")

    # ---- Process each video ----
    meta = {
        "config": {
            "mode": args.mode,
            "feat_stride": args.feat_stride,
            "num_frames": args.num_frames,
            "crop_size": args.crop_size,
            "sample_mode": args.sample_mode,
            "frame_width": args.frame_width,
            "frame_height": args.frame_height,
            "video_fps": args.video_fps,
        },
        "videos": {},
    }

    success_count = 0
    skip_count = 0
    error_count = 0

    for vid, vpath in video_list:
        output_path = os.path.join(args.output_dir, f"{vid}.npy")

        if os.path.exists(output_path) and not args.overwrite:
            print(f"[SKIP] {vid}: output already exists")
            skip_count += 1
            continue

        try:
            # --- Load frames ---
            if use_video:
                frames, actual_fps = load_frames_from_video(
                    vpath, target_fps=args.video_fps, target_size=target_size
                )
            else:
                frames = load_frames_from_dir(
                    vpath, frame_pattern=args.frame_pattern, target_size=target_size
                )
                actual_fps = float(args.video_fps)

            total_frames = frames.shape[0]

            if total_frames < args.num_frames:
                print(f"[WARN] {vid}: {total_frames} frames < window size {args.num_frames}, skipping")
                error_count += 1
                continue

            # --- Compute optical flow (if Flow model is needed) ---
            flow_frames = None
            if "flow" in models:
                print(f"[INFO] Computing optical flow for {vid} ({total_frames} frames)...")
                flow_frames = compute_optical_flow(frames)
                flow_total = flow_frames.shape[0]  # T-1
                print(f"[INFO] Optical flow done: {flow_total} flow frames")

            # --- Generate sliding windows ---
            window_indices_rgb = build_windows(total_frames, args.num_frames, args.feat_stride)

            # --- Extract features ---
            feat_list = []
            for mode_key in models.keys():
                if mode_key == "flow":
                    # Flow uses flow frames (T-1 frames)
                    if flow_total < args.num_frames:
                        print(f"[WARN] {vid}: {flow_total} flow frames < window {args.num_frames}, skipping flow")
                        continue
                    flow_window_indices = build_windows(flow_total, args.num_frames, args.feat_stride)
                    feats = extract_features_for_video(
                        flow_frames, models[mode_key], flow_window_indices,
                        sample_mode=args.sample_mode,
                        crop_size=args.crop_size,
                        batch_size=args.batch_size,
                        device=device,
                    )
                else:
                    feats = extract_features_for_video(
                        frames, models[mode_key], window_indices_rgb,
                        sample_mode=args.sample_mode,
                        crop_size=args.crop_size,
                        batch_size=args.batch_size,
                        device=device,
                    )
                feat_list.append(feats)

            # Use RGB window count for metadata
            num_windows = window_indices_rgb.shape[0]

            # --- Concatenate RGB+Flow ---
            if len(feat_list) == 2:
                final_feats = np.concatenate(feat_list, axis=1)  # (T, 2048)
            else:
                final_feats = feat_list[0]  # (T, 1024)

            # --- Save .npy (T, C) float32 ---
            np.save(output_path, final_feats.astype(np.float32))

            # --- Record metadata ---
            video_meta = {
                "total_frames": total_frames,
                "num_windows": num_windows,
                "feature_dim": int(final_feats.shape[1]),
                "fps": actual_fps,
                "duration": total_frames / actual_fps if actual_fps > 0 else None,
            }
            # Supplement from annotation JSON
            if vid in annotation_db:
                ann_info = annotation_db[vid]
                video_meta["subset"] = ann_info.get("subset", "unknown")
                if "duration" in ann_info:
                    video_meta["annotated_duration"] = ann_info["duration"]

            meta["videos"][vid] = video_meta

            print(f"[OK] {vid}: frames={total_frames} windows={num_windows} dim={final_feats.shape[1]}")
            success_count += 1

        except Exception as e:
            print(f"[ERROR] {vid}: {e}")
            error_count += 1
            continue

    # ---- Save meta.json ----
    meta_path = os.path.join(args.output_dir, "meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False, default=str)

    # ---- Summary ----
    print(f"\n[INFO] === Done ===")
    print(f"  Success: {success_count}")
    print(f"  Skipped: {skip_count}")
    print(f"  Errors:  {error_count}")
    print(f"  Meta:    {meta_path}")

    # Key hints for user
    if args.mode == "rgb+flow":
        print(f"\n[HINT] Features are 2048-dim (RGB 1024 + Flow 1024).")
        print(f"  Set input_dim: 2048, feat_stride: {args.feat_stride}, "
              f"num_frames: {args.num_frames} in your config YAML.")
    elif args.mode == "rgb":
        print(f"\n[HINT] Features are 1024-dim (RGB only).")
        print(f"  The standard THUMOS14 config expects 2048-dim.")
        print(f"  Either re-run with --mode rgb+flow, or set input_dim: 1024 in config.")


if __name__ == "__main__":
    main()

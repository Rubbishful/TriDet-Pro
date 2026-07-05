"""
End-to-end video action detection pipeline
==========================================
Video -> OpenCV frame extraction -> Optical flow -> I3D features (2048-dim)
      -> TriDet model (GPU) -> C NMS -> Detection results TXT
      -> [optional] YOLO person detection + annotated video output

Usage:
    python scripts/full_pipeline.py \
        --video id3shuju/shipin/01.mp4 \
        --config configs/id3_i3d.yaml \
        --ckpt ckpt/thumos_i3d_baseline/epoch_039.pth.tar \
        --output_dir result

    # With annotated video output:
    python scripts/full_pipeline.py \
        --video id3shuju/shipin/01.mp4 \
        --visualize
"""

import argparse
import os
import sys
import time

import numpy as np
import torch

# Project imports
_PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJ_ROOT not in sys.path:
    sys.path.insert(0, _PROJ_ROOT)

from libs.core import load_config
from libs.modeling import make_meta_arch

# E2E module imports
from E2E.loader import load_frames_from_video
from E2E.flow import compute_optical_flow
from E2E.features import build_windows, load_i3d_model, extract_features_for_video
from E2E.inference import run_tridet_inference, save_results_txt, THUMOS14_LABEL_NAMES
from E2E.visualizer import (
    create_annotated_video,
    predictions_to_action_list,
)


def _resolve_path(path, base_dir=_PROJ_ROOT):
    """Resolve a possibly-relative path against project root.

    Args:
        path: Path string (may be relative or absolute).
        base_dir: Base directory for resolving relative paths.

    Returns:
        Absolute path string.
    """
    if os.path.isabs(path):
        return path
    return os.path.normpath(os.path.join(base_dir, path))


# ============================================================================
# Main Pipeline
# ============================================================================


def main():

    print("Running")

    parser = argparse.ArgumentParser(
        description="End-to-end video action detection: frames -> flow -> I3D -> TriDet -> NMS -> TXT")

    # Required arguments
    parser.add_argument("--video", type=str, required=True,
                        help="Input video path")
    parser.add_argument("--config", type=str, default="configs/id3_i3d.yaml",
                        help="TriDet config file")
    parser.add_argument("--ckpt", type=str, default="ckpt/thumos_i3d_baseline/epoch_039.pth.tar",
                        help="TriDet model checkpoint")
    parser.add_argument("--output_dir", type=str, default="result",
                        help="Output directory for results")

    # Optional arguments
    parser.add_argument("--rgb_model", type=str,
                        default="E2E/model/rgb_imagenet.pt",
                        help="RGB I3D model weights")
    parser.add_argument("--flow_model", type=str,
                        default="E2E/model/flow_imagenet.pt",
                        help="Flow I3D model weights")
    parser.add_argument("--device", type=str, default="cuda:0",
                        help="Computation device")
    parser.add_argument("--target_fps", type=int, default=25,
                        help="Target frame rate")
    parser.add_argument("--feat_stride", type=int, default=4,
                        help="Feature window stride")
    parser.add_argument("--num_frames", type=int, default=16,
                        help="Frames per sliding window")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="I3D feature extraction batch size")
    parser.add_argument("--save_npy", action="store_true", default=False,
                        help="Also save intermediate .npy feature file")

    # Visualization / YOLO options
    viz_group = parser.add_argument_group("Visualization (optional)")
    viz_group.add_argument("--visualize", action="store_true", default=False,
                        help="Generate annotated video with YOLO boxes + action labels")
    viz_group.add_argument("--yolo_model", type=str, default="E2E/model/yolov8n.pt",
                        help="YOLO model name or path (default: E2E/model/yolov8n.pt)")
    viz_group.add_argument("--yolo_conf", type=float, default=0.3,
                        help="YOLO confidence threshold (default: 0.3)")
    viz_group.add_argument("--no_timeline", action="store_true", default=False,
                        help="Disable the bottom timeline strip in output video")

    args = parser.parse_args()

    # Resolve all paths against project root
    video_path = _resolve_path(args.video)
    config_path = _resolve_path(args.config)
    ckpt_path = _resolve_path(args.ckpt)
    output_dir = _resolve_path(args.output_dir)
    rgb_model_path = _resolve_path(args.rgb_model)
    flow_model_path = _resolve_path(args.flow_model)

    # --- Validate inputs ---
    if not os.path.exists(video_path):
        print(f"[ERROR] Video file not found: {video_path}")
        sys.exit(1)
    if not os.path.exists(config_path):
        print(f"[ERROR] Config file not found: {config_path}")
        sys.exit(1)
    if not os.path.exists(ckpt_path):
        print(f"[ERROR] Model checkpoint not found: {ckpt_path}")
        sys.exit(1)

    # --- Device ---
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device: {device}")
    if device.type == "cuda":
        print(f"[INFO] GPU: {torch.cuda.get_device_name(device)}")
        print(f"[INFO] VRAM: {torch.cuda.get_device_properties(device).total_memory / 1024**3:.1f} GB")

    overall_start = time.time()

    # ========================================================================
    # Step 1: Frame extraction (OpenCV)
    # ========================================================================
    total_steps = 7 if args.visualize else 5
    step_str = lambda s: f"[Step {s}/{total_steps}]"

    print("\n" + "=" * 60)
    print(f"{step_str(1)} OpenCV Frame Extraction")
    print("=" * 60)
    t1 = time.time()

    # Read native FPS before extraction for logging
    import cv2
    cap_temp = cv2.VideoCapture(video_path)
    native_fps = cap_temp.get(cv2.CAP_PROP_FPS)
    cap_temp.release()
    if native_fps <= 0:
        native_fps = 30.0

    frames, actual_fps = load_frames_from_video(
        video_path,
        target_fps=args.target_fps,
        target_size=(340, 256)
    )
    total_frames = frames.shape[0]
    duration = total_frames / actual_fps

    print(f"         Original FPS={native_fps:.2f}, Target FPS={actual_fps}, "
          f"Sampling interval={native_fps / actual_fps:.2f}")
    print(f"         Done in {time.time() - t1:.1f}s")
    print(f"         Total frames={total_frames}, FPS={actual_fps}, Duration={duration:.1f}s")

    # ========================================================================
    # Step 2: Optical flow (Farneback)
    # ========================================================================
    print("\n" + "=" * 60)
    print(f"{step_str(2)} Optical Flow Computation (Farneback)")
    print("=" * 60)
    t2 = time.time()

    flow_frames = compute_optical_flow(frames)
    flow_total = flow_frames.shape[0]

    print(f"         Input {total_frames} frames -> {flow_total} flow frames")
    print(f"         Done in {time.time() - t2:.1f}s")

    # ========================================================================
    # Step 3: I3D feature extraction (2048-dim = RGB 1024 + Flow 1024)
    # ========================================================================
    print("\n" + "=" * 60)
    print(f"{step_str(3)} I3D Feature Extraction (RGB + Flow -> 2048-dim)")
    print("=" * 60)
    t3 = time.time()

    # Load I3D models
    print(f"[I3D] Loading RGB model: {rgb_model_path}")
    rgb_model = load_i3d_model(rgb_model_path, "rgb", device)
    print(f"[I3D] Loading Flow model: {flow_model_path}")
    flow_model = load_i3d_model(flow_model_path, "flow", device)

    # RGB sliding windows
    window_indices_rgb = build_windows(total_frames, args.num_frames, args.feat_stride)
    print(f"[I3D] RGB windows: {window_indices_rgb.shape[0]}")

    # RGB features (1024-dim)
    print("[I3D] Extracting RGB features...")
    rgb_feats = extract_features_for_video(
        frames, rgb_model, window_indices_rgb,
        sample_mode="center_crop", crop_size=224,
        batch_size=args.batch_size, device=device
    )
    print(f"      RGB feature shape: {rgb_feats.shape}")

    # Flow sliding windows
    window_indices_flow = build_windows(flow_total, args.num_frames, args.feat_stride)
    print(f"[I3D] Flow windows: {window_indices_flow.shape[0]}")

    # Flow features (1024-dim)
    print("[I3D] Extracting Flow features...")
    flow_feats = extract_features_for_video(
        flow_frames, flow_model, window_indices_flow,
        sample_mode="center_crop", crop_size=224,
        batch_size=args.batch_size, device=device
    )
    print(f"      Flow feature shape: {flow_feats.shape}")

    # Align RGB and Flow features (take minimum window count)
    min_windows = min(rgb_feats.shape[0], flow_feats.shape[0])
    rgb_feats = rgb_feats[:min_windows]
    flow_feats = flow_feats[:min_windows]

    # Concatenate -> 2048-dim
    combined_feats = np.concatenate([rgb_feats, flow_feats], axis=1)
    print(f"[I3D] Combined feature shape: {combined_feats.shape}  (2048-dim)")
    print(f"         Done in {time.time() - t3:.1f}s")

    # Optional: save intermediate features
    if args.save_npy:
        npy_dir = os.path.join(output_dir, "features")
        os.makedirs(npy_dir, exist_ok=True)
        vid_name = os.path.splitext(os.path.basename(video_path))[0]
        np.save(os.path.join(npy_dir, f"{vid_name}.npy"),
                combined_feats.astype(np.float32))
        print(f"[INFO] Intermediate features saved to {npy_dir}/{vid_name}.npy")

    # Release I3D model VRAM
    del rgb_model, flow_model
    torch.cuda.empty_cache()

    # ========================================================================
    # Step 4: TriDet model inference + C NMS
    # ========================================================================
    print("\n" + "=" * 60)
    print(f"{step_str(4)} TriDet Model Inference + C NMS")
    print("=" * 60)
    t4 = time.time()

    # Load config
    cfg = load_config(config_path)

    # ---- Dynamically adjust max_seq_len to fit actual feature length ----
    # Original max_seq_len=128 is for training; inference on long videos needs more
    # Must be divisible by max_div_factor (backbone_arch[-1]=5 -> 6 FPN levels -> 2^5=32)
    max_div_factor = cfg["model"]["scale_factor"] ** cfg["model"]["backbone_arch"][-1]
    raw_feat_len = combined_feats.shape[0]
    required_min = ((raw_feat_len + max_div_factor - 1) // max_div_factor) * max_div_factor
    original_max_seq_len = cfg["dataset"]["max_seq_len"]
    if required_min > original_max_seq_len:
        print(f"[TriDet] max_seq_len insufficient: original={original_max_seq_len}, "
              f"required >= {required_min} (feature length={raw_feat_len})")
        cfg["dataset"]["max_seq_len"] = required_min
        print(f"[TriDet] Auto-adjusted to {required_min}")

    # Ensure input dimensions match
    cfg["model"]["input_dim"] = cfg["dataset"]["input_dim"]
    cfg["model"]["num_classes"] = cfg["dataset"]["num_classes"]
    cfg["model"]["max_seq_len"] = cfg["dataset"]["max_seq_len"]
    cfg["model"]["train_cfg"] = cfg["train_cfg"]
    cfg["model"]["test_cfg"] = cfg["test_cfg"]

    print(f"[TriDet] Config: input_dim={cfg['dataset']['input_dim']}, "
          f"num_classes={cfg['dataset']['num_classes']}, "
          f"max_seq_len={cfg['dataset']['max_seq_len']}")

    # Build model
    model = make_meta_arch(cfg["model_name"], **cfg["model"])
    total_params = sum(p.numel() for p in model.parameters())
    print(f"[TriDet] Parameters: {total_params:,}")

    # Load EMA weights (checkpoint saved with DataParallel, has module. prefix)
    print(f"[TriDet] Loading checkpoint: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)

    if "state_dict_ema" in checkpoint:
        state_dict = checkpoint["state_dict_ema"]
        print("[TriDet] Using EMA weights")
    else:
        state_dict = checkpoint["state_dict"]
        print("[TriDet] Using standard weights (EMA not found)")

    # Handle DataParallel module. prefix
    # When loading into non-DataParallel model, strip the prefix
    if any(k.startswith("module.") for k in state_dict.keys()):
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        print("[TriDet] Removed DataParallel 'module.' prefix")

    model.load_state_dict(state_dict)
    del checkpoint

    model = model.to(device)
    model.eval()
    print("[TriDet] Model loaded successfully")

    # Inference (C NMS applied internally)
    vid_name = os.path.splitext(os.path.basename(video_path))[0]
    results = run_tridet_inference(
        combined_feats, model, cfg, device,
        video_id=vid_name,
        fps=actual_fps,
        duration=duration
    )

    print(f"         Done in {time.time() - t4:.1f}s")

    # ========================================================================
    # Step 5: Save results TXT
    # ========================================================================
    print("\n" + "=" * 60)
    print(f"{step_str(5)} Save Detection Results")
    print("=" * 60)

    save_results_txt(results, output_dir)

    # ========================================================================
    # Step 6-7: YOLO + Annotated Video (optional)
    # ========================================================================
    if args.visualize:
        # --- Step 6: YOLO person detection ---
        print("\n" + "=" * 60)
        print(f"{step_str(6)} YOLO Person Detection + Annotated Video")
        print("=" * 60)
        t6 = time.time()

        from libs.subject.detector import SubjectDetector

        print(f"[YOLO] Loading model: {args.yolo_model}")
        detector = SubjectDetector(model_name=args.yolo_model, device=str(device))

        # Convert TriDet results to flat action list
        action_list = predictions_to_action_list(results, THUMOS14_LABEL_NAMES)
        print(f"[YOLO] Action segments to visualize: {len(action_list)}")

        # Output video path
        output_video_path = os.path.join(output_dir, f"{vid_name}_annotated.mp4")

        print(f"[YOLO] Source: {video_path}")
        print(f"[YOLO] Output: {output_video_path}")
        print(f"[YOLO] Native FPS={native_fps:.2f}, "
              f"conf_threshold={args.yolo_conf}")

        create_annotated_video(
            video_path=video_path,
            output_path=output_video_path,
            action_results=action_list,
            detector=detector,
            conf_threshold=args.yolo_conf,
            show_timeline=not args.no_timeline,
            progress=True,
        )

        print(f"         Done in {time.time() - t6:.1f}s")

    overall_end = time.time()
    print("\n" + "=" * 60)
    print("  Pipeline Complete!")
    print("=" * 60)
    print(f"  Total time: {overall_end - overall_start:.1f}s")
    print(f"  Output directory: {os.path.abspath(output_dir)}")

    if args.visualize:
        output_video_path = os.path.join(output_dir, f"{vid_name}_annotated.mp4")
        if os.path.exists(output_video_path):
            size_mb = os.path.getsize(output_video_path) / (1024 * 1024)
            print(f"  Annotated video:  {os.path.abspath(output_video_path)}  ({size_mb:.1f} MB)")
    for r in results:
        segs = r["segments"].cpu().numpy()
        scores = r["scores"].cpu().numpy()
        labels = r["labels"].cpu().numpy()
        print(f"\n  Video: {r['video_id']}")
        print(f"  Detected {len(segs)} action segments")
        if len(segs) > 0:
            top_k = min(5, len(segs))
            sort_idx = np.argsort(-scores)
            print(f"  Top-{top_k} detections:")
            for i in range(top_k):
                idx = sort_idx[i]
                label_name = THUMOS14_LABEL_NAMES.get(int(labels[idx]), f"class_{int(labels[idx])}")
                print(f"    [{float(segs[idx, 0]):.1f}s - {float(segs[idx, 1]):.1f}s] "
                      f"{label_name} "
                      f"(conf={float(scores[idx]):.4f})")


if __name__ == "__main__":
    main()

"""TriDet model inference and result saving.

Provides the final stage of the end-to-end pipeline:
  - run_tridet_inference: Run TriDet model on extracted I3D features.
  - save_results_txt: Save detection results as human-readable TXT files.
"""

import os

import numpy as np
import torch


# ============================================================================
# THUMOS14 Label Names
# ============================================================================

THUMOS14_LABEL_NAMES = {
    0: "BaseballPitch", 1: "BasketballDunk", 2: "Billiards",
    3: "CleanAndJerk", 4: "CliffDiving", 5: "CricketBowling",
    6: "CricketShot", 7: "Diving", 8: "FrisbeeCatch",
    9: "GolfSwing", 10: "HammerThrow", 11: "HighJump",
    12: "JavelinThrow", 13: "LongJump", 14: "PoleVault",
    15: "Shotput", 16: "SoccerPenalty", 17: "TennisSwing",
    18: "ThrowDiscus", 19: "VolleyballSpiking",
}


# ============================================================================
# TriDet Inference
# ============================================================================


def run_tridet_inference(feats_npy, model, cfg, device,
                         video_id="01", fps=25.0, duration=None):
    """Run TriDet model inference with built-in C-NMS postprocessing.

    Args:
        feats_npy: np.ndarray (T, 2048), I3D features.
        model: TriDet model in eval mode.
        cfg: Configuration dict (from load_config).
        device: torch.device.
        video_id: Video identifier string.
        fps: Frame rate.
        duration: Video duration in seconds (None = auto-compute).

    Returns:
        results: List of dicts, each containing:
            video_id, segments (N, 2), scores (N,), labels (N,).
    """
    feat_stride = cfg["dataset"]["feat_stride"]
    num_frames = cfg["dataset"]["num_frames"]

    # Estimate duration from feature length
    if duration is None:
        duration = feats_npy.shape[0] * feat_stride / fps

    # Build C x T tensor (model expected format)
    feats_tensor = torch.from_numpy(
        np.ascontiguousarray(feats_npy.transpose())
    ).float()

    # Construct inference input (single sample)
    video_data = [{
        "video_id": video_id,
        "feats": feats_tensor,          # C x T
        "fps": fps,
        "duration": duration,
        "feat_stride": feat_stride,
        "feat_num_frames": num_frames,
        "segments": None,               # no GT for inference
        "labels": None,
    }]

    model.eval()
    with torch.no_grad():
        results = model(video_data)

    return results


# ============================================================================
# Result Saving
# ============================================================================


def save_results_txt(results, output_dir, label_names=None):
    """Save detection results as readable TXT files.

    Output format (one detection per line):
        rank  start_time  end_time  score  label_id  label_name

    Produces two files per video:
      - {video_id}.txt: Full detection table sorted by confidence.
      - {video_id}_summary.txt: Per-class counts and top-10 detections.

    Args:
        results: Model inference output (list of dict).
        output_dir: Output directory for TXT files.
        label_names: dict {label_id: label_name}. Defaults to THUMOS14 labels.
    """
    if label_names is None:
        label_names = THUMOS14_LABEL_NAMES

    os.makedirs(output_dir, exist_ok=True)

    for r in results:
        vid = r["video_id"]
        segs = r["segments"].cpu().numpy()
        scores = r["scores"].cpu().numpy()
        labels = r["labels"].cpu().numpy()

        txt_path = os.path.join(output_dir, f"{vid}.txt")

        with open(txt_path, "w", encoding="utf-8") as f:
            f.write("# TriDet Action Detection Results\n")
            f.write(f"# Video: {vid}\n")
            f.write(f"# Detections: {len(segs)}\n")
            f.write("# " + "-" * 60 + "\n")
            f.write(f"{'# Rank':>5s}  "
                    f"{'Start(s)':>10s}  "
                    f"{'End(s)':>10s}  "
                    f"{'Score':>8s}  "
                    f"{'LabelID':>6s}  "
                    f"{'Label'}\n")

            # Sort by confidence (descending)
            sort_idx = np.argsort(-scores)
            for rank, idx in enumerate(sort_idx, 1):
                start_s = float(segs[idx, 0])
                end_s = float(segs[idx, 1])
                score = float(scores[idx])
                label_id = int(labels[idx])
                label_str = label_names.get(label_id, f"class_{label_id}")

                f.write(f"{rank:5d}  "
                        f"{start_s:10.2f}  "
                        f"{end_s:10.2f}  "
                        f"{score:8.4f}  "
                        f"{label_id:6d}  "
                        f"{label_str}\n")

        print(f"[INFO] Detection results saved: {txt_path}  ({len(segs)} detections)")

        # Also save summary
        summary_path = os.path.join(output_dir, f"{vid}_summary.txt")
        with open(summary_path, "w", encoding="utf-8") as f:
            f.write(f"Video: {vid}\n")
            f.write(f"Total detections: {len(segs)}\n")
            f.write(f"Detection time range: {float(segs[:, 0].min()):.1f}s - "
                    f"{float(segs[:, 1].max()):.1f}s\n\n")

            # Per-class counts
            unique_labels, counts = np.unique(labels, return_counts=True)
            f.write("Detections per class:\n")
            for lid, cnt in zip(unique_labels, counts):
                label_str = label_names.get(int(lid), f"class_{int(lid)}")
                f.write(f"  {label_str}: {cnt}\n")

            # Top-10 high confidence
            top_k = min(10, len(segs))
            top_idx = sort_idx[:top_k]
            f.write(f"\nTop-{top_k} detections by confidence:\n")
            for rank, idx in enumerate(top_idx, 1):
                start_s = float(segs[idx, 0])
                end_s = float(segs[idx, 1])
                score = float(scores[idx])
                label_id = int(labels[idx])
                label_str = label_names.get(label_id, f"class_{label_id}")
                f.write(f"  #{rank}: [{start_s:.2f}s - {end_s:.2f}s] "
                        f"{label_str} (conf={score:.4f})\n")

        print(f"[INFO] Summary saved: {summary_path}")

    return txt_path

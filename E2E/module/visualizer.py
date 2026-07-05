"""Video visualization for E2E action detection pipeline.

Draws YOLO person detection bounding boxes and TriDet action detection
labels + confidence scores on each frame, then writes an annotated video.

Usage (standalone)::

    from E2E.module.visualizer import create_annotated_video

    create_annotated_video(
        video_path="data/01.mp4",
        output_path="result/01_annotated.mp4",
        action_results=action_results,      # list of action dicts
        label_names=THUMOS14_LABEL_NAMES,
        detector=detector,                   # SubjectDetector instance
    )
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


# ============================================================================
# Drawing Helpers
# ============================================================================

# Color palette for action labels (20-class friendly, BGR)
_ACTION_COLORS = [
    (0, 255, 0),      # green
    (255, 0, 0),      # blue
    (0, 0, 255),      # red
    (255, 255, 0),    # cyan
    (255, 0, 255),    # magenta
    (0, 255, 255),    # yellow
    (0, 255, 128),    # lime
    (0, 128, 255),    # orange
    (255, 128, 0),    # sky
    (128, 0, 255),    # purple
    (128, 128, 255),  # salmon
    (128, 255, 128),  # mint
    (255, 128, 128),  # lavender
    (255, 255, 128),  # cream
    (192, 192, 0),    # teal
    (192, 0, 192),    # plum
    (0, 192, 192),    # olive
    (0, 192, 0),      # forest
    (0, 0, 192),      # maroon
    (192, 0, 0),      # navy
]


def _pick_color(label_str: str) -> Tuple[int, int, int]:
    """Deterministic colour per action label string."""
    idx = hash(label_str) % len(_ACTION_COLORS)
    return _ACTION_COLORS[idx]


def draw_yolo_boxes(frame_bgr: np.ndarray,
                    persons: List[Dict],
                    box_color: Tuple[int, int, int] = (0, 255, 0),
                    thickness: int = 2,
                    font_scale: float = 0.6) -> None:
    """Draw YOLO person-detection bounding boxes onto *frame_bgr* (in-place).

    Args:
        frame_bgr: BGR image (H, W, 3), modified in-place.
        persons: List of detection dicts, each with ``bbox`` (x1,y1,x2,y2)
            and ``confidence``.
        box_color: BGR tuple for the rectangle.
        thickness: Line thickness.
        font_scale: OpenCV font scale for the confidence label.
    """
    for p in persons:
        x1, y1, x2, y2 = map(int, p["bbox"])
        conf = p["confidence"]
        cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), box_color, thickness)
        cv2.putText(frame_bgr, f"person {conf:.2f}",
                    (x1, max(y1 - 8, 16)),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, box_color, thickness)


def draw_action_overlay(frame_bgr: np.ndarray,
                        active_actions: List[Dict],
                        font_scale: float = 0.7,
                        thickness: int = 2,
                        top_margin: int = 10) -> None:
    """Draw active action labels in the top-left corner of *frame_bgr*.

    Args:
        frame_bgr: BGR image (H, W, 3), modified in-place.
        active_actions: List of dicts with ``label``, ``score``, ``start``,
            ``end``.  Sorted by score descending; at most ~5 displayed.
        font_scale: OpenCV font scale.
        thickness: Line thickness.
        top_margin: Pixels from the top edge for the first label.
    """
    if not active_actions:
        return

    y = top_margin + 25
    for act in active_actions:
        color = _pick_color(act["label"])
        text = (f"{act['label']}  {act['score']:.2f}  "
                f"[{act['start']:.1f}s-{act['end']:.1f}s]")

        # Draw semi-transparent background bar
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX,
                                       font_scale, thickness)
        overlay = frame_bgr.copy()
        cv2.rectangle(overlay, (5, y - th - 6), (tw + 15, y + 6),
                      (40, 40, 40), -1)
        cv2.addWeighted(overlay, 0.55, frame_bgr, 0.45, 0, frame_bgr)

        cv2.putText(frame_bgr, text, (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)
        y += th + 14


def draw_timeline_strip(frame_bgr: np.ndarray,
                        all_actions: List[Dict],
                        current_time: float,
                        video_duration: float,
                        strip_height: int = 60) -> None:
    """Draw a timeline bar at the bottom showing all action segments.

    Args:
        frame_bgr: BGR image (H, W, 3), modified in-place.
        all_actions: All action dicts (``start``, ``end``, ``label``, ``score``).
        current_time: Current frame timestamp in seconds.
        video_duration: Total video duration in seconds.
        strip_height: Height of the timeline strip in pixels.
    """
    H, W = frame_bgr.shape[:2]
    y0 = H - strip_height

    # Background
    cv2.rectangle(frame_bgr, (0, y0), (W, H), (30, 30, 30), -1)

    if video_duration <= 0:
        return

    # Scale: pixels per second
    scale = W / video_duration

    # Draw each action segment
    for act in all_actions:
        x1 = int(act["start"] * scale)
        x2 = int(act["end"] * scale)
        x1 = max(0, min(W - 1, x1))
        x2 = max(x1 + 2, min(W, x2))
        color = _pick_color(act["label"])
        cv2.rectangle(frame_bgr, (x1, y0 + 4), (x2, y0 + strip_height - 4),
                      color, -1)

    # Current position cursor
    cx = int(current_time * scale)
    cx = max(0, min(W - 1, cx))
    cv2.line(frame_bgr, (cx, y0), (cx, H), (255, 255, 255), 2)

    # Time markers
    cv2.putText(frame_bgr, "0s", (4, H - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
    dur_text = f"{video_duration:.0f}s"
    (tw, _), _ = cv2.getTextSize(dur_text, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
    cv2.putText(frame_bgr, dur_text, (W - tw - 4, H - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)


# ============================================================================
# Frame-level Helpers
# ============================================================================

def get_active_actions(timestamp: float,
                       action_results: List[Dict],
                       max_display: int = 5) -> List[Dict]:
    """Return actions whose segment covers *timestamp*.

    Args:
        timestamp: Frame time in seconds.
        action_results: List of dicts with ``start``, ``end``, ``label``,
            ``score``.
        max_display: Max number of actions to return.

    Returns:
        List of active action dicts, sorted by score descending.
    """
    active = []
    for act in action_results:
        if act["start"] <= timestamp <= act["end"]:
            active.append(act)
    active.sort(key=lambda a: a["score"], reverse=True)
    return active[:max_display]


def predictions_to_action_list(predictions,
                               label_names: Dict[int, str],
                               score_threshold: float = 0.0) -> List[Dict]:
    """Convert TriDet predictions to a flat list of action dicts.

    Supports two input formats:

    1. **full_pipeline format** — list of per-video dicts::

         [{"video_id": str, "segments": Tensor(N,2), "scores": Tensor(N,),
           "labels": Tensor(N,)}]

    2. **batch_pipeline format** — flat aggregated dict::

         {"video-id": list[str], "t-start": array, "t-end": array,
          "label": array, "score": array}

    Args:
        predictions: Predictions in either format.
        label_names: Label-id → name mapping.
        score_threshold: Minimum score to include.

    Returns:
        List of action dicts with keys ``start``, ``end``, ``label``, ``score``.
    """
    results = []

    # Format 1: list of per-video dicts (full_pipeline)
    if isinstance(predictions, list):
        for r in predictions:
            segs = r["segments"]
            scores = r["scores"]
            labels = r["labels"]
            # Handle both tensor and numpy
            if hasattr(segs, "cpu"):
                segs = segs.cpu().numpy()
            if hasattr(scores, "cpu"):
                scores = scores.cpu().numpy()
            if hasattr(labels, "cpu"):
                labels = labels.cpu().numpy()
            for i in range(len(segs)):
                s = float(scores[i])
                if s < score_threshold:
                    continue
                lid = int(labels[i])
                results.append({
                    "start": float(segs[i][0]),
                    "end": float(segs[i][1]),
                    "label": label_names.get(lid, f"class_{lid}"),
                    "score": s,
                })
        return results

    # Format 2: flat aggregated dict (batch_pipeline)
    if isinstance(predictions, dict) and "video-id" in predictions:
        vids = predictions["video-id"]
        t_starts = predictions["t-start"]
        t_ends = predictions["t-end"]
        p_labels = predictions["label"]
        p_scores = predictions["score"]
        for i in range(len(vids)):
            s = float(p_scores[i])
            if s < score_threshold:
                continue
            lid = int(p_labels[i])
            results.append({
                "start": float(t_starts[i]),
                "end": float(t_ends[i]),
                "label": label_names.get(lid, f"class_{lid}"),
                "score": s,
            })
        return results

    raise TypeError(f"Unsupported predictions type: {type(predictions)}")


def filter_actions_by_video(action_list: List[Dict],
                            predictions,
                            video_id: str,
                            label_names: Dict[int, str]) -> List[Dict]:
    """Filter predictions to a single video_id, returning a flat action list.

    Accepts both full_pipeline and batch_pipeline formats.

    Args:
        action_list: Optional pre-built list from full_pipeline format
            (already filtered to one video).  Pass ``[]`` to build from
            *predictions* instead.
        predictions: Raw batch_pipeline format predictions (flat dict), or
            full_pipeline list.  Used when *action_list* is empty.
        video_id: Video identifier to keep.
        label_names: Label-id → name mapping.

    Returns:
        Filtered list of action dicts.
    """
    if action_list:
        return action_list  # already built from full_pipeline format

    # Build from raw predictions
    results = []

    if isinstance(predictions, list):
        # full_pipeline format: already per-video, just convert
        for r in predictions:
            if r["video_id"] != video_id:
                continue
            segs = r["segments"]
            scores = r["scores"]
            labels = r["labels"]
            if hasattr(segs, "cpu"):
                segs = segs.cpu().numpy()
            if hasattr(scores, "cpu"):
                scores = scores.cpu().numpy()
            if hasattr(labels, "cpu"):
                labels = labels.cpu().numpy()
            for i in range(len(segs)):
                lid = int(labels[i])
                results.append({
                    "start": float(segs[i][0]),
                    "end": float(segs[i][1]),
                    "label": label_names.get(lid, f"class_{lid}"),
                    "score": float(scores[i]),
                })
        return results

    # batch_pipeline format: flat dict, filter by video-id
    if isinstance(predictions, dict) and "video-id" in predictions:
        vids = predictions["video-id"]
        t_starts = predictions["t-start"]
        t_ends = predictions["t-end"]
        p_labels = predictions["label"]
        p_scores = predictions["score"]
        for i in range(len(vids)):
            if vids[i] != video_id:
                continue
            lid = int(p_labels[i])
            results.append({
                "start": float(t_starts[i]),
                "end": float(t_ends[i]),
                "label": label_names.get(lid, f"class_{lid}"),
                "score": float(p_scores[i]),
            })
        return results

    return results


# ============================================================================
# Main Video Writer
# ============================================================================

def create_annotated_video(
    video_path: str,
    output_path: str,
    action_results: List[Dict],
    *,
    detector=None,
    conf_threshold: float = 0.5,
    max_display_actions: int = 5,
    target_fps: Optional[float] = None,
    target_size: Optional[Tuple[int, int]] = None,
    show_timeline: bool = True,
    progress: bool = True,
    max_frames: Optional[int] = None,
) -> str:
    """Create an annotated video with YOLO boxes + action labels.

    Reads the source video frame-by-frame, optionally runs YOLO person
    detection, overlays active action-detection labels, and writes an
    annotated MP4.

    Args:
        video_path: Path to the source video file.
        output_path: Output path for annotated video (``.mp4``).
        action_results: Flat list of action dicts from
            :func:`predictions_to_action_list`, each with ``start``,
            ``end``, ``label``, ``score``.
        detector: :class:`SubjectDetector` instance, or ``None`` to skip
            YOLO person detection.
        conf_threshold: YOLO confidence threshold.
        max_display_actions: Max number of concurrent action labels to
            show per frame.
        target_fps: Optional output frame rate (resample).  ``None``
            uses the source video's native FPS.
        target_size: Optional output (width, height).  ``None`` uses the
            source video's native resolution.
        show_timeline: Whether to draw the bottom timeline strip.
        progress: Whether to print a progress bar via tqdm.
        max_frames: Cap on number of frames to process (for debugging).

    Returns:
        The *output_path* as a string.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    src_fps = cap.get(cv2.CAP_PROP_FPS)
    src_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    src_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if src_fps <= 0:
        src_fps = 30.0
    if src_total <= 0:
        src_total = 1

    # Determine output parameters
    out_fps = target_fps if target_fps else src_fps
    out_size = target_size if target_size else (src_width, src_height)
    video_duration = src_total / src_fps

    # Compute sampling for target_fps
    if target_fps and target_fps != src_fps:
        sample_interval = max(1, src_fps / target_fps)
    else:
        sample_interval = 1.0

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, out_fps, out_size)
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Cannot create video writer for: {output_path}")

    # Progress bar
    pbar = None
    if progress:
        try:
            from tqdm import tqdm
            pbar = tqdm(total=src_total, desc="Visualizing", unit="frame")
        except ImportError:
            pass

    frame_idx = 0
    next_sample = 0.0
    written = 0

    while True:
        ret, frame_bgr = cap.read()
        if not ret:
            break

        timestamp = frame_idx / src_fps

        # Resample for target FPS
        if frame_idx < int(next_sample):
            frame_idx += 1
            if pbar:
                pbar.update(1)
            continue

        next_sample += sample_interval

        # Resize if needed
        if out_size != (src_width, src_height):
            frame_bgr = cv2.resize(frame_bgr, out_size,
                                   interpolation=cv2.INTER_LINEAR)

        # --- YOLO person detection ---
        persons = []
        if detector is not None:
            persons = detector.detect(frame_bgr, conf_threshold=conf_threshold)

        # --- Active action labels ---
        active = get_active_actions(timestamp, action_results,
                                    max_display=max_display_actions)

        # --- Draw overlays ---
        draw_yolo_boxes(frame_bgr, persons)
        draw_action_overlay(frame_bgr, active)

        if show_timeline:
            draw_timeline_strip(frame_bgr, action_results,
                                timestamp, video_duration)

        writer.write(frame_bgr)
        written += 1
        frame_idx += 1

        if pbar:
            pbar.update(1)

        if max_frames and written >= max_frames:
            break

    cap.release()
    writer.release()
    if pbar:
        pbar.close()

    if written == 0:
        raise RuntimeError("No frames written to output video")

    return output_path

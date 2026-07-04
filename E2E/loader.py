"""Frame loading utilities for I3D feature extraction.

Supports loading frames from:
  - Individual image files (PIL)
  - Frame directories (sorted by filename pattern)
  - Video files (OpenCV, with FPS resampling)

Normalization follows the reference pytorch-i3d convention: (data * 2/255) - 1.
"""

import glob
import os
import re

import cv2
import numpy as np
from PIL import Image


# ============================================================================
# Single Frame Loading
# ============================================================================


def load_single_frame(frame_path, resize_dims=None):
    """Load a single frame image, normalize to [-1, 1].

    Args:
        frame_path: Path to the image file.
        resize_dims: (width, height) tuple, or None.

    Returns:
        np.ndarray (H, W, 3), dtype float, range [-1, 1].
    """
    img = Image.open(frame_path).convert("RGB")

    if resize_dims is not None:
        img = img.resize(resize_dims, Image.LANCZOS)

    data = np.array(img).astype(np.float32)
    data = (data * 2.0 / 255.0) - 1.0
    return data


# ============================================================================
# Directory-based Frame Loading
# ============================================================================


def load_frames_from_dir(frames_dir, frame_pattern="img_{:06d}.jpg",
                         target_size=None):
    """Load sorted frame images from a directory.

    Args:
        frames_dir: Directory containing frame images.
        frame_pattern: Filename pattern, e.g. "img_{:06d}.jpg" or "frame_*.jpg".
        target_size: (width, height) tuple, or None.

    Returns:
        np.ndarray (T, H, W, 3), normalized to [-1, 1].
    """
    # Support both Python format patterns and glob wildcards
    if "{" in frame_pattern:
        glob_pattern = re.sub(r"\{[^}]*\}", "*", frame_pattern)
    else:
        glob_pattern = frame_pattern

    frame_files = sorted(glob.glob(os.path.join(frames_dir, glob_pattern)))
    if len(frame_files) == 0:
        # Fallback: match all common image extensions
        exts = (".jpg", ".jpeg", ".png", ".bmp")
        frame_files = sorted([
            os.path.join(frames_dir, f)
            for f in os.listdir(frames_dir)
            if f.lower().endswith(exts)
        ])

    if len(frame_files) == 0:
        raise FileNotFoundError(f"No frame images found in {frames_dir}")

    frames = []
    for fp in frame_files:
        frames.append(load_single_frame(fp, resize_dims=target_size))

    return np.stack(frames, axis=0)  # (T, H, W, 3)


# ============================================================================
# Video-based Frame Loading (OpenCV)
# ============================================================================


def load_frames_from_video(video_path, target_fps=None,
                           target_size=None):
    """Extract frames from a video file via OpenCV.

    Args:
        video_path: Path to the video file.
        target_fps: Target frame rate (None = use native FPS).
        target_size: (width, height) tuple, or None.

    Returns:
        frames: np.ndarray (T, H, W, 3), normalized to [-1, 1].
        actual_fps: Actual frame rate used for sampling.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    native_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if native_fps <= 0:
        native_fps = 30.0
    if total_frames <= 0:
        total_frames = 1

    # Compute sampling interval
    if target_fps is not None and target_fps > 0:
        sample_interval = max(1, native_fps / target_fps)
        actual_fps = target_fps
    else:
        sample_interval = 1.0
        actual_fps = native_fps

    frames = []
    read_idx = 0
    next_sample = 0.0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if read_idx >= int(next_sample):
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if target_size is not None:
                frame = cv2.resize(frame, target_size, interpolation=cv2.INTER_LINEAR)
            data = frame.astype(np.float32)
            data = (data * 2.0 / 255.0) - 1.0
            frames.append(data)
            next_sample += sample_interval

        read_idx += 1

    cap.release()

    if len(frames) == 0:
        raise RuntimeError(f"No valid frames extracted from {video_path}")

    return np.stack(frames, axis=0), actual_fps


# ============================================================================
# Video / Frame Discovery
# ============================================================================


def find_videos_from_dir(video_dir, video_exts=(".mp4", ".avi", ".mkv", ".webm")):
    """Scan a directory for video files.

    Args:
        video_dir: Directory to scan.
        video_exts: Tuple of recognized video file extensions.

    Returns:
        List of (video_id, video_path) tuples sorted by path.
    """
    video_files = []
    for ext in video_exts:
        video_files.extend(glob.glob(os.path.join(video_dir, f"*{ext}")))
        video_files.extend(glob.glob(os.path.join(video_dir, f"*{ext.upper()}")))
    video_files = sorted(set(video_files))
    video_list = []
    for vp in video_files:
        vid = os.path.splitext(os.path.basename(vp))[0]
        video_list.append((vid, vp))
    return video_list


def find_frames_from_dir(frames_dir, prefix_filter=None):
    """Scan a directory for per-video frame subdirectories.

    Following the pytorch-i3d convention, each subdirectory corresponds
    to one video's frame sequence.

    Args:
        frames_dir: Root directory containing frame subdirectories.
        prefix_filter: Optional prefix to filter subdirectory names
            (e.g. "v_" for "v_" prefix convention).

    Returns:
        List of (video_id, frames_path) tuples sorted by name.
    """
    subdirs = [
        d for d in os.listdir(frames_dir)
        if os.path.isdir(os.path.join(frames_dir, d))
    ]
    if prefix_filter is not None:
        subdirs = [d for d in subdirs if d.startswith(prefix_filter)]
    subdirs = sorted(subdirs)
    video_list = []
    for sd in subdirs:
        video_list.append((sd, os.path.join(frames_dir, sd)))
    return video_list

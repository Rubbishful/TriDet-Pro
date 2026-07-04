"""E2E: End-to-end video action detection module.

Provides:
  - I3D model definition (InceptionI3D)
  - Frame loading utilities (video, directory, single frame)
  - Optical flow computation (Farneback)
  - I3D feature extraction (sliding windows, batched GPU inference)
  - TriDet inference and result saving

Usage:
    from E2E import (
        InceptionI3d,
        load_frames_from_video,
        compute_optical_flow,
        build_windows,
        load_i3d_model,
        extract_features_for_video,
        run_tridet_inference,
        save_results_txt,
        THUMOS14_LABEL_NAMES,
    )
"""

from .i3d import InceptionI3d, Unit3D, MaxPool3dSamePadding, InceptionModule
from .loader import (
    load_single_frame,
    load_frames_from_dir,
    load_frames_from_video,
    find_videos_from_dir,
    find_frames_from_dir,
)
from .flow import compute_optical_flow
from .features import build_windows, load_i3d_model, extract_features_for_video
from .inference import run_tridet_inference, save_results_txt, THUMOS14_LABEL_NAMES

__all__ = [
    # I3D model
    "InceptionI3d",
    "Unit3D",
    "MaxPool3dSamePadding",
    "InceptionModule",
    # Frame loading
    "load_single_frame",
    "load_frames_from_dir",
    "load_frames_from_video",
    "find_videos_from_dir",
    "find_frames_from_dir",
    # Optical flow
    "compute_optical_flow",
    # Feature extraction
    "build_windows",
    "load_i3d_model",
    "extract_features_for_video",
    # Inference
    "run_tridet_inference",
    "save_results_txt",
    "THUMOS14_LABEL_NAMES",
]

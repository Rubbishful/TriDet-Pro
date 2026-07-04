"""I3D feature extraction core.

Provides:
  - build_windows: Generate sliding window frame indices
  - load_i3d_model: Load InceptionI3D with pretrained weights
  - extract_features_for_video: Batch-extract I3D features for a single video
"""

import numpy as np
import torch
from PIL import Image

from E2E.i3d import InceptionI3d


# ============================================================================
# Sliding Window Index
# ============================================================================


def build_windows(total_frames, window_size=16, stride=4):
    """Generate frame indices for sliding window extraction.

    num_windows = floor((total_frames - window_size) / stride) + 1

    For THUMOS14 (feat_stride=4, num_frames=16):
      - stride=4 means adjacent windows overlap by 12 frames
      - 200 frame video -> 47 windows: [0..15], [4..19], ..., [184..199]

    Args:
        total_frames: Total number of frames in the video.
        window_size: Number of frames per window.
        stride: Frame interval between consecutive window starts.

    Returns:
        np.ndarray (num_windows, window_size), dtype int.
    """
    if total_frames < window_size:
        raise ValueError(
            f"Video too short: {total_frames} frames < {window_size} window size"
        )

    last_start = total_frames - window_size
    num_windows = last_start // stride + 1
    starts = np.arange(0, num_windows * stride, stride)
    indices = starts[:, None] + np.arange(window_size)  # broadcasting
    return indices


# ============================================================================
# Model Loading
# ============================================================================


def load_i3d_model(weights_path, mode, device):
    """Load InceptionI3D model with pretrained Kinetics-400 weights.

    Args:
        weights_path: Path to .pt weights file.
        mode: "rgb" (in_channels=3) or "flow" (in_channels=2).
        device: torch.device.

    Returns:
        InceptionI3d model in eval mode, on the target device.
    """
    in_channels = 3 if mode == "rgb" else 2
    model = InceptionI3d(num_classes=400, in_channels=in_channels)

    state_dict = torch.load(weights_path, map_location=device, weights_only=False)
    model.load_state_dict(state_dict)
    model.eval()
    return model.to(device)


# ============================================================================
# Core Feature Extraction
# ============================================================================


@torch.no_grad()
def extract_features_for_video(frames, model, window_indices,
                                sample_mode="center_crop", crop_size=224,
                                batch_size=16, device="cuda:0"):
    """Extract I3D features for a single video.

    Pipeline (matching reference pytorch-i3d):
      1. Index frames by window_indices -> (B, window_size, H, W, C)
      2. Spatial crop (center_crop or resize) -> (B, window_size, crop, crop, C)
      3. Transpose -> (B, C, window_size, crop, crop)  [I3D NCHW format]
      4. model.extract_features() -> (B, 1024, 1, 1, 1)
      5. Squeeze -> (B, 1024)

    Args:
        frames: np.ndarray (T_total, H, W, C), normalized to [-1, 1].
        model: InceptionI3d in eval mode.
        window_indices: np.ndarray (num_windows, window_size).
        sample_mode: Spatial sampling mode: "center_crop" or "resize".
        crop_size: Spatial crop size in pixels (square).
        batch_size: Number of windows per GPU batch.
        device: Computation device.

    Returns:
        features: np.ndarray (num_windows, 1024).
    """
    num_windows = window_indices.shape[0]
    window_size = window_indices.shape[1]
    in_h, in_w = frames.shape[1], frames.shape[2]

    all_features = []

    for batch_start in range(0, num_windows, batch_size):
        batch_end = min(batch_start + batch_size, num_windows)
        batch_indices = window_indices[batch_start:batch_end]  # (B, window_size)
        B = batch_indices.shape[0]

        # (1) Gather frames -> (B, window_size, H, W, C)
        batch_data = frames[batch_indices]  # advanced indexing -> (B, window_size, H, W, C)

        # (2) Spatial processing
        if sample_mode == "center_crop":
            # Reference: data[:,:,16:240,58:282,:]  (340x256 -> 224x224 center crop)
            h_start = (in_h - crop_size) // 2
            w_start = (in_w - crop_size) // 2
            batch_data = batch_data[:, :, h_start:h_start + crop_size,
                                    w_start:w_start + crop_size, :]
        elif sample_mode == "resize":
            # Per-frame PIL resize; may be slow for large batches but preserves compatibility
            resized = np.zeros((B, window_size, crop_size, crop_size, frames.shape[3]),
                               dtype=batch_data.dtype)
            for bi in range(B):
                for wi in range(window_size):
                    img = Image.fromarray(
                        ((batch_data[bi, wi] + 1.0) * 127.5).astype(np.uint8)
                    )
                    img = img.resize((crop_size, crop_size), Image.LANCZOS)
                    resized[bi, wi] = (np.array(img).astype(np.float64) * 2.0 / 255.0) - 1.0
            batch_data = resized
        # else: keep original spatial dimensions

        # (3) Transpose: (B, window_size, H, W, C) -> (B, C, window_size, H, W)
        batch_data = batch_data.transpose(0, 4, 1, 2, 3)
        batch_data = torch.from_numpy(batch_data).float().to(device)

        # (4) I3D forward pass
        b_features = model.extract_features(batch_data)  # (B, 1024, 1, 1, 1)

        # (5) Squeeze spatial + temporal dims -> (B, 1024)
        b_features = b_features.data.cpu().numpy()[:, :, 0, 0, 0]
        all_features.append(b_features)

    return np.concatenate(all_features, axis=0)

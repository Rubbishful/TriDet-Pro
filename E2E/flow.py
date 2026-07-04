"""Farneback optical flow computation for I3D Flow model input.

I3D flow convention (following DeepMind / pytorch-i3d):
  - Raw flow (dx, dy) is clipped to [-flow_clip, flow_clip] pixels
  - Normalized to [-1, 1] via: flow_out = flow_clipped / flow_clip

This skips the intermediate uint8 image step and directly outputs
normalized 2-channel flow frames.
"""

import cv2
import numpy as np


def compute_optical_flow(frames, flow_clip=20.0):
    """Compute Farneback optical flow on consecutive RGB frames.

    Produces 2-channel output suitable for the I3D Flow model.

    Process:
      1. Convert frames from [-1, 1] to uint8 [0, 255]
      2. Convert to grayscale
      3. Calculate Farneback optical flow between consecutive frames
      4. Clip to [-flow_clip, flow_clip] and normalize to [-1, 1]

    Args:
        frames: np.ndarray (T, H, W, 3), range [-1, 1] (RGB frames).
        flow_clip: Optical flow clipping threshold in pixels (default 20).

    Returns:
        flow_frames: np.ndarray (T-1, H, W, 2), range [-1, 1].
    """
    T = frames.shape[0]
    if T < 2:
        raise ValueError(f"Need at least 2 frames for optical flow, got {T}")

    # Convert frames from [-1, 1] to uint8 [0, 255] for Farneback
    frames_uint8 = ((frames + 1.0) * 127.5).clip(0, 255).astype(np.uint8)

    # Convert to grayscale
    gray_frames = [cv2.cvtColor(frames_uint8[0], cv2.COLOR_RGB2GRAY)]
    for i in range(1, T):
        gray_frames.append(cv2.cvtColor(frames_uint8[i], cv2.COLOR_RGB2GRAY))

    flow_list = []
    for i in range(1, T):
        flow = cv2.calcOpticalFlowFarneback(
            gray_frames[i - 1], gray_frames[i],
            None, 0.5, 3, 15, 3, 5, 1.2, 0
        )  # (H, W, 2)
        # Clip and normalize to [-1, 1]
        flow = np.clip(flow, -flow_clip, flow_clip) / flow_clip
        flow_list.append(flow)

    return np.stack(flow_list, axis=0)  # (T-1, H, W, 2)

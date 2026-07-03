"""
特征提取脚本 — 基于社区标准 InceptionI3D 从视频帧提取特征, 输出 TriDet 兼容格式.

模型: InceptionI3D (pytorch_i3d.py — Inception-v1 膨胀为 3D)
预训练权重: Kinetics-400 (rgb_imagenet.pt / flow_imagenet.pt)

输入:
  - 原始视频文件 (--video_dir): 使用 OpenCV 实时解码
  - 预提取帧图像 (--frames_dir): 参考 pytorch-i3d 的 img_*.jpg 约定

输出:
  - .npy 文件, 形状 (T, C), dtype float32, 可直接被 TriDet DataLoader 加载
  - meta.json: 记录特征提取配置和每个视频的统计信息

用法示例:
  # THUMOS14: 从预提取帧提取 RGB 特征 (1024-dim)
  python scripts/extract_features.py \
      --frames_dir ./thumos_frames \
      --output_dir ./thumos_i3d_features \
      --mode rgb \
      --rgb_model pytorch-i3d-feature-extraction-master/models/rgb_imagenet.pt \
      --feat_stride 4

  # THUMOS14: 从原始视频提取 RGB+Flow 特征 (2048-dim)
  python scripts/extract_features.py \
      --video_dir D:/Code/THUMOS14/thumos/videos \
      --output_dir D:/Code/THUMOS14/thumos/i3d_features \
      --mode rgb+flow \
      --rgb_model pytorch-i3d-feature-extraction-master/models/rgb_imagenet.pt \
      --flow_model pytorch-i3d-feature-extraction-master/models/flow_imagenet.pt \
      --feat_stride 4 \
      --video_fps 25

  # 单视频: 指定视频路径, 自动以文件名为 video_id
  python scripts/extract_features.py \
      --video_path /path/to/video.mp4 \
      --output_dir ./single_feat \
      --mode rgb
"""

import argparse
import glob
import json
import os
import sys

import numpy as np
import torch
from PIL import Image

# 添加 scripts 目录到 path, 确保可以导入 pytorch_i3d
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

from pytorch_i3d import InceptionI3d  # noqa: E402


# ============================================================================
# Model Loading
# ============================================================================


def load_i3d_model(weights_path, mode, device):
    """
    加载 InceptionI3D 模型及预训练权重.

    Args:
        weights_path: .pt 权重文件路径
        mode:  'rgb'  (in_channels=3) 或 'flow' (in_channels=2)
        device: torch.device
    Returns:
        model (eval mode)
    """
    in_channels = 3 if mode == 'rgb' else 2
    model = InceptionI3d(num_classes=400, in_channels=in_channels)

    state_dict = torch.load(weights_path, map_location=device, weights_only=False)
    model.load_state_dict(state_dict)
    model.eval()
    return model.to(device)


# ============================================================================
# Frame Loading  (归一化公式与参考 pytorch-i3d 完全一致: (data * 2/255) - 1)
# ============================================================================


def load_single_frame(frame_path, resize_dims=None):
    """
    加载单张帧图像, 归一化到 [-1, 1].

    Args:
        frame_path: 图像文件路径
        resize_dims: (width, height) 或 None
    Returns:
        np.ndarray (H, W, 3), dtype float, 值域 [-1, 1]
    """
    img = Image.open(frame_path).convert('RGB')

    if resize_dims is not None:
        img = img.resize(resize_dims, Image.LANCZOS)

    data = np.array(img).astype(np.float64)
    data = (data * 2.0 / 255.0) - 1.0
    return data


def load_frames_from_dir(frames_dir, frame_pattern='img_{:06d}.jpg',
                         target_size=None):
    """
    从目录加载已排序的帧图像.

    Args:
        frames_dir:   帧图像目录
        frame_pattern: 文件名模式, e.g. 'img_{:06d}.jpg' 或 'frame_*.jpg'
        target_size:  (width, height) 或 None
    Returns:
        frames: np.ndarray (T, H, W, 3), 归一化到 [-1, 1]
    """
    # 支持两种模式: {:06d} 格式 或 通配符 *
    if '{' in frame_pattern:
        # 将 python format 转为 glob 模式: img_{:06d}.jpg -> img_*.jpg
        import re
        glob_pattern = re.sub(r'\{[^}]*\}', '*', frame_pattern)
    else:
        glob_pattern = frame_pattern

    frame_files = sorted(glob.glob(os.path.join(frames_dir, glob_pattern)))
    if len(frame_files) == 0:
        # 回退: 不按模式匹配, 直接取所有图像文件
        exts = ('.jpg', '.jpeg', '.png', '.bmp')
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


def load_frames_from_video(video_path, target_fps=None,
                           target_size=None):
    """
    从视频文件读取帧 (OpenCV).

    Args:
        video_path: 视频文件路径
        target_fps: 目标 FPS (None=使用原始 FPS)
        target_size: (width, height) 或 None
    Returns:
        frames: np.ndarray (T, H, W, 3), 归一化到 [-1, 1]
        actual_fps: 实际采样帧率
    """
    import cv2

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    native_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if native_fps <= 0:
        native_fps = 30.0
    if total_frames <= 0:
        total_frames = 1

    # 计算采样步长
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
            data = frame.astype(np.float64)
            data = (data * 2.0 / 255.0) - 1.0
            frames.append(data)
            next_sample += sample_interval

        read_idx += 1

    cap.release()

    if len(frames) == 0:
        raise RuntimeError(f"No valid frames extracted from {video_path}")

    return np.stack(frames, axis=0), actual_fps


# ============================================================================
# Optical Flow Computation
# ============================================================================


def compute_optical_flow_farneback(frames, flow_clip=20.0):
    """
    对连续 RGB 帧计算 Farneback 光流, 输出 I3D Flow 模型所需的 2-通道格式.

    I3D 光流约定 (参考 DeepMind / pytorch-i3d):
      - flow_x, flow_y 值域映射到 [0, 255] 后存储为图像
      - 加载时做 (data * 2/255) - 1 归一化
      - 等效于: flow_net = flow_raw / flow_clip

    我们跳过图像中间态, 直接输出归一化到 [-1, 1] 的光流.
      1. Farneback 光流 → (dx, dy) 像素位移
      2. clip 到 [-flow_clip, flow_clip]
      3. 缩放到 [-1, 1]: flow_out = flow_clipped / flow_clip

    Args:
        frames:    np.ndarray (T, H, W, 3), 值域 [-1, 1] (RGB 帧)
        flow_clip: 光流截断阈值 (pixel), 默认 20

    Returns:
        flow_frames: np.ndarray (T-1, H, W, 2), 值域 [-1, 1]
    """
    import cv2

    T = frames.shape[0]
    if T < 2:
        raise ValueError(f"Need at least 2 frames for optical flow, got {T}")

    # 将帧从 [-1, 1] 转到 uint8 [0, 255] 供 Farneback 使用
    frames_uint8 = ((frames + 1.0) * 127.5).clip(0, 255).astype(np.uint8)

    # 转灰度
    gray_frames = [cv2.cvtColor(frames_uint8[0], cv2.COLOR_RGB2GRAY)]
    for i in range(1, T):
        gray_frames.append(cv2.cvtColor(frames_uint8[i], cv2.COLOR_RGB2GRAY))

    flow_list = []
    for i in range(1, T):
        flow = cv2.calcOpticalFlowFarneback(
            gray_frames[i - 1], gray_frames[i],
            None, 0.5, 3, 15, 3, 5, 1.2, 0
        )  # (H, W, 2)
        # clip & normalize to [-1, 1]
        flow = np.clip(flow, -flow_clip, flow_clip) / flow_clip
        flow_list.append(flow)

    return np.stack(flow_list, axis=0)  # (T-1, H, W, 2)


# ============================================================================
# Sliding Window Index
# ============================================================================


def build_windows(total_frames, window_size=16, stride=4):
    """
    生成滑动窗口的帧索引.

    num_windows = floor((total_frames - window_size) / stride) + 1

    对于 THUMOS14 (feat_stride=4, num_frames=16):
      - stride=4 时相邻窗口重叠 12 帧
      - 200 帧视频 → 47 个窗口: [0..15], [4..19], ..., [184..199]

    Args:
        total_frames: 视频总帧数
        window_size:  每个窗口包含的帧数
        stride:       相邻窗口起始帧的间隔
    Returns:
        np.ndarray (num_windows, window_size), dtype int
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
# Core Feature Extraction
# ============================================================================


@torch.no_grad()
def extract_features_for_video(frames, model, window_indices,
                                sample_mode='center_crop', crop_size=224,
                                batch_size=16, device='cuda:0'):
    """
    对单个视频提取 I3D 特征.

    处理流程 (与参考 pytorch-i3d 一致):
      1. 按 window_indices 取出帧 → (B, window_size, H, W, C)
      2. 空间裁剪 (center_crop 或 resize) → (B, window_size, crop_size, crop_size, C)
      3. transpose → (B, C, window_size, crop_size, crop_size)  [I3D 期望的 NCHW 格式]
      4. model.extract_features() → (B, 1024, 1, 1, 1)
      5. squeeze → (B, 1024)

    Args:
        frames:         np.ndarray (T_total, H, W, C), 已归一化到 [-1, 1]
        model:          InceptionI3d (eval mode)
        window_indices: np.ndarray (num_windows, window_size)
        sample_mode:    'center_crop' | 'resize'
        crop_size:      空间裁剪尺寸 (正方形)
        batch_size:     每批处理的窗口数
        device:         计算设备
    Returns:
        features: np.ndarray (num_windows, 1024)
    """
    num_windows = window_indices.shape[0]
    window_size = window_indices.shape[1]
    in_h, in_w = frames.shape[1], frames.shape[2]

    all_features = []

    for batch_start in range(0, num_windows, batch_size):
        batch_end = min(batch_start + batch_size, num_windows)
        batch_indices = window_indices[batch_start:batch_end]  # (B, window_size)
        B = batch_indices.shape[0]

        # (1) 取出帧 → (B, window_size, H, W, C)
        batch_data = frames[batch_indices]  # advanced indexing → (B, window_size, H, W, C)

        # (2) 空间处理
        if sample_mode == 'center_crop':
            # 参考代码: data[:,:,16:240,58:282,:]  (340x256 → 224x224 center crop)
            h_start = (in_h - crop_size) // 2
            w_start = (in_w - crop_size) // 2
            batch_data = batch_data[:, :, h_start:h_start + crop_size,
                                    w_start:w_start + crop_size, :]
        elif sample_mode == 'resize':
            # 使用 PIL 逐帧 resize; 对于大批量可能有性能瓶颈, 但保留兼容性
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
        # else: 不做空间变换, 使用原始尺寸

        # (3) transpose: (B, window_size, H, W, C) → (B, C, window_size, H, W)
        batch_data = batch_data.transpose(0, 4, 1, 2, 3)
        batch_data = torch.from_numpy(batch_data).float().to(device)

        # (4) I3D 前向
        b_features = model.extract_features(batch_data)  # (B, 1024, 1, 1, 1)

        # (5) squeeze 时空维度 → (B, 1024)
        b_features = b_features.data.cpu().numpy()[:, :, 0, 0, 0]
        all_features.append(b_features)

    return np.concatenate(all_features, axis=0)


# ============================================================================
# Video Discovery
# ============================================================================


def find_videos_from_dir(video_dir, video_exts=('.mp4', '.avi', '.mkv', '.webm')):
    """扫描目录获取视频列表."""
    video_files = []
    for ext in video_exts:
        video_files.extend(glob.glob(os.path.join(video_dir, f'*{ext}')))
        video_files.extend(glob.glob(os.path.join(video_dir, f'*{ext.upper()}')))
    video_files = sorted(set(video_files))
    video_list = []
    for vp in video_files:
        vid = os.path.splitext(os.path.basename(vp))[0]
        video_list.append((vid, vp))
    return video_list


def find_frames_from_dir(frames_dir, prefix_filter=None):
    """
    扫描 frames_dir 下的子目录, 每个子目录对应一个视频的帧序列.
    参考 pytorch-i3d 约定: 子目录名以 'v' 开头.
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


# ============================================================================
# Main
# ============================================================================


def main():
    parser = argparse.ArgumentParser(
        description='Extract I3D features from video frames for TriDet',
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

    # ---- 输入源 (三选一) ----
    input_group = parser.add_argument_group('Input source (choose one)')
    input_group.add_argument('--video_dir', type=str, default=None,
                             help='Directory of raw video files')
    input_group.add_argument('--frames_dir', type=str, default=None,
                             help='Directory of per-video frame subdirectories')
    input_group.add_argument('--video_path', type=str, default=None,
                             help='Path to a single video file')

    # ---- 输出 ----
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory for .npy feature files')

    # ---- 模型 ----
    model_group = parser.add_argument_group('Model weights')
    model_group.add_argument('--rgb_model', type=str,
                             default='pytorch-i3d-feature-extraction-master/models/rgb_imagenet.pt',
                             help='Path to RGB I3D weights')
    model_group.add_argument('--flow_model', type=str,
                             default='pytorch-i3d-feature-extraction-master/models/flow_imagenet.pt',
                             help='Path to Flow I3D weights')
    model_group.add_argument('--mode', type=str, default='rgb',
                             choices=['rgb', 'flow', 'rgb+flow'],
                             help='Feature mode: rgb (1024-dim), flow (1024-dim), '
                                  'rgb+flow (2048-dim)')

    # ---- 特征提取参数 ----
    feat_group = parser.add_argument_group('Feature extraction parameters')
    feat_group.add_argument('--feat_stride', type=int, default=4,
                            help='Frame stride between consecutive windows (default: 4)')
    feat_group.add_argument('--num_frames', type=int, default=16,
                            help='Frames per window (default: 16)')
    feat_group.add_argument('--crop_size', type=int, default=224,
                            help='Spatial crop size in pixels (default: 224)')
    feat_group.add_argument('--sample_mode', type=str, default='center_crop',
                            choices=['center_crop', 'resize'],
                            help='Spatial sampling mode (default: center_crop)')

    # ---- 性能 ----
    perf_group = parser.add_argument_group('Performance')
    perf_group.add_argument('--batch_size', type=int, default=16,
                            help='Batch size for GPU processing (default: 16)')
    perf_group.add_argument('--device', type=str, default='cuda:0',
                            help='Computation device (default: cuda:0)')

    # ---- 视频/帧读取 ----
    io_group = parser.add_argument_group('Video/Frame reading')
    io_group.add_argument('--video_fps', type=int, default=25,
                          help='Target FPS when reading videos (default: 25)')
    io_group.add_argument('--video_ext', type=str, nargs='+',
                          default=['.mp4', '.avi', '.mkv', '.webm'],
                          help='Video file extensions (default: .mp4 .avi .mkv .webm)')
    io_group.add_argument('--frame_pattern', type=str, default='img_{:06d}.jpg',
                          help='Frame file naming pattern (default: img_{:06d}.jpg)')
    io_group.add_argument('--frame_width', type=int, default=340,
                          help='Input frame width, before crop (default: 340)')
    io_group.add_argument('--frame_height', type=int, default=256,
                          help='Input frame height, before crop (default: 256)')

    # ---- 杂项 ----
    misc_group = parser.add_argument_group('Miscellaneous')
    misc_group.add_argument('--overwrite', action='store_true', default=False,
                            help='Overwrite existing .npy files')
    misc_group.add_argument('--dataset_json', type=str, default=None,
                            help='Path to THUMOS14-style annotation JSON for metadata')

    args = parser.parse_args()

    # ---- 验证: 输入源互斥 ----
    sources = [args.video_dir, args.frames_dir, args.video_path]
    num_sources = sum(1 for s in sources if s is not None)
    if num_sources != 1:
        parser.error("Exactly one of --video_dir, --frames_dir, --video_path required")

    # ---- 验证模型权重 ----
    if args.mode in ('rgb', 'rgb+flow'):
        if not os.path.exists(args.rgb_model):
            parser.error(f"RGB model not found: {args.rgb_model}")
    if args.mode in ('flow', 'rgb+flow'):
        if not os.path.exists(args.flow_model):
            parser.error(f"Flow model not found: {args.flow_model}")

    # ---- 设备 ----
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"[INFO] Device: {device}")

    # ---- 加载模型 ----
    models = {}
    if args.mode in ('rgb', 'rgb+flow'):
        print(f"[INFO] Loading RGB model: {args.rgb_model}")
        models['rgb'] = load_i3d_model(args.rgb_model, 'rgb', device)
    if args.mode in ('flow', 'rgb+flow'):
        print(f"[INFO] Loading Flow model: {args.flow_model}")
        models['flow'] = load_i3d_model(args.flow_model, 'flow', device)

    # ---- 发现视频 ----
    if args.video_path:
        vid = os.path.splitext(os.path.basename(args.video_path))[0]
        video_list = [(vid, args.video_path)]
        use_video = True
    elif args.video_dir:
        video_list = find_videos_from_dir(args.video_dir, tuple(args.video_ext))
        use_video = True
        print(f"[INFO] Found {len(video_list)} videos in {args.video_dir}")
    else:
        # frames_dir: 每个子目录 = 一个视频的帧序列
        prefix = None  # 不强制 'v' 前缀, 兼容更多命名方式
        video_list = find_frames_from_dir(args.frames_dir, prefix_filter=prefix)
        use_video = False
        print(f"[INFO] Found {len(video_list)} frame directories in {args.frames_dir}")

    if len(video_list) == 0:
        print("[ERROR] No videos/frames found!")
        sys.exit(1)

    target_size = (args.frame_width, args.frame_height)

    # ---- 创建输出目录 ----
    os.makedirs(args.output_dir, exist_ok=True)

    # ---- 加载标注 (可选) ----
    annotation_db = {}
    if args.dataset_json and os.path.exists(args.dataset_json):
        with open(args.dataset_json, 'r') as f:
            ann_data = json.load(f)
        annotation_db = ann_data.get('database', ann_data)
        print(f"[INFO] Loaded annotation data for {len(annotation_db)} videos")

    # ---- 处理每个视频 ----
    meta = {
        'config': {
            'mode': args.mode,
            'feat_stride': args.feat_stride,
            'num_frames': args.num_frames,
            'crop_size': args.crop_size,
            'sample_mode': args.sample_mode,
            'frame_width': args.frame_width,
            'frame_height': args.frame_height,
            'video_fps': args.video_fps,
        },
        'videos': {},
    }

    success_count = 0
    skip_count = 0
    error_count = 0

    for vid, vpath in video_list:
        output_path = os.path.join(args.output_dir, f'{vid}.npy')

        if os.path.exists(output_path) and not args.overwrite:
            print(f"[SKIP] {vid}: output already exists")
            skip_count += 1
            continue

        try:
            # --- 加载帧 ---
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

            # --- 计算光流 (若需要 Flow 模型) ---
            flow_frames = None
            if 'flow' in models:
                print(f"[INFO] Computing optical flow for {vid} ({total_frames} frames)...")
                flow_frames = compute_optical_flow_farneback(frames)
                flow_total = flow_frames.shape[0]  # T-1
                print(f"[INFO] Optical flow done: {flow_total} flow frames")

            # --- 生成滑窗 ---
            window_indices_rgb = build_windows(total_frames, args.num_frames, args.feat_stride)

            # --- 提取特征 ---
            feat_list = []
            for mode_key in models.keys():
                if mode_key == 'flow':
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

            # --- 拼接 RGB+Flow ---
            if len(feat_list) == 2:
                final_feats = np.concatenate(feat_list, axis=1)  # (T, 2048)
            else:
                final_feats = feat_list[0]  # (T, 1024)

            # --- 保存 .npy (T, C) float32 ---
            np.save(output_path, final_feats.astype(np.float32))

            # --- 记录元信息 ---
            video_meta = {
                'total_frames': total_frames,
                'num_windows': num_windows,
                'feature_dim': int(final_feats.shape[1]),
                'fps': actual_fps,
                'duration': total_frames / actual_fps if actual_fps > 0 else None,
            }
            # 从标注 JSON 补充信息
            if vid in annotation_db:
                ann_info = annotation_db[vid]
                video_meta['subset'] = ann_info.get('subset', 'unknown')
                if 'duration' in ann_info:
                    video_meta['annotated_duration'] = ann_info['duration']

            meta['videos'][vid] = video_meta

            print(f"[OK] {vid}: frames={total_frames} windows={num_windows} dim={final_feats.shape[1]}")
            success_count += 1

        except Exception as e:
            print(f"[ERROR] {vid}: {e}")
            error_count += 1
            continue

    # ---- 保存 meta.json ----
    meta_path = os.path.join(args.output_dir, 'meta.json')
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2, ensure_ascii=False, default=str)

    # ---- 总结 ----
    print(f"\n[INFO] === Done ===")
    print(f"  Success: {success_count}")
    print(f"  Skipped: {skip_count}")
    print(f"  Errors:  {error_count}")
    print(f"  Meta:    {meta_path}")

    # 输出给用户的关键提示
    if args.mode == 'rgb+flow':
        print(f"\n[HINT] Features are 2048-dim (RGB 1024 + Flow 1024).")
        print(f"  Set input_dim: 2048, feat_stride: {args.feat_stride}, "
              f"num_frames: {args.num_frames} in your config YAML.")
    elif args.mode == 'rgb':
        print(f"\n[HINT] Features are 1024-dim (RGB only).")
        print(f"  The standard THUMOS14 config expects 2048-dim.")
        print(f"  Either re-run with --mode rgb+flow, or set input_dim: 1024 in config.")


if __name__ == '__main__':
    main()

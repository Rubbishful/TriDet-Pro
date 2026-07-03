"""
端到端视频动作检测流水线
========================
视频 → OpenCV 帧提取 → 光流计算 → I3D 特征 (2048-dim)
     → TriDet 模型 (RTX 4060) → C NMS → 检测结果 TXT

用法:
    python scripts/full_pipeline.py \
        --video id3shuju/shipin/01.mp4 \
        --config configs/id3_i3d.yaml \
        --ckpt epoch_039.pth.tar \
        --output_dir result
"""

import argparse
import json
import os
import sys
import time

import cv2
import numpy as np
import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# 将项目根目录和 scripts 目录加入 sys.path
# ---------------------------------------------------------------------------
_PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SCRIPT_DIR = os.path.join(_PROJ_ROOT, 'scripts')
for _p in (_PROJ_ROOT, _SCRIPT_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from libs.core import load_config
from libs.modeling import make_meta_arch

# I3D 模型 (两个 pytorch_i3d.py 内容一致, 使用 scripts 下的)
from pytorch_i3d import InceptionI3d


# ============================================================================
# 1. 视频帧提取 (OpenCV)
# ============================================================================

def extract_frames_opencv(video_path, target_fps=25, target_size=(340, 256)):
    """
    用 OpenCV 从视频中提取帧.

    Args:
        video_path: 视频文件路径
        target_fps: 目标帧率
        target_size: (width, height) resize 尺寸
    Returns:
        frames:      np.ndarray (T, H, W, 3), float64, [-1, 1]
        actual_fps:  float
        native_fps:  float
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频: {video_path}")

    native_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if native_fps <= 0:
        native_fps = 30.0
    if total_frames <= 0:
        total_frames = 1

    if target_fps is not None and target_fps > 0:
        sample_interval = max(1, native_fps / target_fps)
        actual_fps = float(target_fps)
    else:
        sample_interval = 1.0
        actual_fps = float(native_fps)

    frames = []
    read_idx = 0
    next_sample = 0.0

    print(f"[帧提取] 视频: {video_path}")
    print(f"         原始 FPS={native_fps:.2f}, 目标 FPS={actual_fps}, "
          f"采样间隔={sample_interval:.2f}")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if read_idx >= int(next_sample):
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if target_size is not None:
                frame = cv2.resize(frame, target_size, interpolation=cv2.INTER_LINEAR)
            data = frame.astype(np.float64)
            data = (data * 2.0 / 255.0) - 1.0   # 归一化到 [-1, 1]
            frames.append(data)
            next_sample += sample_interval

        read_idx += 1

    cap.release()

    if len(frames) == 0:
        raise RuntimeError(f"未能从视频中提取到任何帧: {video_path}")

    frames = np.stack(frames, axis=0)
    print(f"         提取 {frames.shape[0]} 帧, 尺寸={frames.shape[1]}x{frames.shape[2]}")
    return frames, actual_fps, native_fps


# ============================================================================
# 2. 光流计算 (Farneback)
# ============================================================================

def compute_optical_flow(frames, flow_clip=20.0):
    """
    对连续 RGB 帧计算 Farneback 光流, 输出 I3D Flow 模型所需的 2-通道格式.

    Args:
        frames:    np.ndarray (T, H, W, 3), float64, [-1, 1]
        flow_clip: 光流截断阈值 (像素)
    Returns:
        flow_frames: np.ndarray (T-1, H, W, 2), float64, [-1, 1]
    """
    T = frames.shape[0]
    if T < 2:
        raise ValueError(f"光流计算需要至少 2 帧, 当前只有 {T} 帧")

    # [-1, 1] → uint8 [0, 255]
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
        flow = np.clip(flow, -flow_clip, flow_clip) / flow_clip
        flow_list.append(flow)

    print(f"[光流计算] 输入 {T} 帧 → 输出 {len(flow_list)} 个光流帧")
    return np.stack(flow_list, axis=0)


# ============================================================================
# 3. I3D 特征提取 (RGB + Flow = 2048-dim)
# ============================================================================

def build_windows(total_frames, window_size=16, stride=4):
    """生成滑窗索引."""
    if total_frames < window_size:
        raise ValueError(f"视频太短: {total_frames} 帧 < {window_size} 窗口大小")
    last_start = total_frames - window_size
    num_windows = last_start // stride + 1
    starts = np.arange(0, num_windows * stride, stride)
    indices = starts[:, None] + np.arange(window_size)
    return indices


def load_i3d_model(weights_path, mode, device):
    """加载 InceptionI3D 模型及预训练权重."""
    in_channels = 3 if mode == 'rgb' else 2
    model = InceptionI3d(num_classes=400, in_channels=in_channels)
    state_dict = torch.load(weights_path, map_location=device, weights_only=False)
    model.load_state_dict(state_dict)
    model.eval()
    return model.to(device)


@torch.no_grad()
def extract_i3d_features(frames, model, window_indices,
                         sample_mode='center_crop', crop_size=224,
                         batch_size=16, device='cuda:0'):
    """
    对单个视频提取 I3D 特征.

    Args:
        frames:         np.ndarray (T_total, H, W, C), [-1, 1]
        model:          InceptionI3d (eval mode)
        window_indices: np.ndarray (num_windows, window_size)
        crop_size:      224
        batch_size:     GPU 批大小
        device:         torch device
    Returns:
        features: np.ndarray (num_windows, 1024)
    """
    num_windows = window_indices.shape[0]
    window_size = window_indices.shape[1]
    in_h, in_w = frames.shape[1], frames.shape[2]

    all_features = []

    for batch_start in range(0, num_windows, batch_size):
        batch_end = min(batch_start + batch_size, num_windows)
        batch_indices = window_indices[batch_start:batch_end]
        B = batch_indices.shape[0]

        batch_data = frames[batch_indices]  # (B, window_size, H, W, C)

        # Center crop
        if sample_mode == 'center_crop':
            h_start = (in_h - crop_size) // 2
            w_start = (in_w - crop_size) // 2
            batch_data = batch_data[:, :,
                                    h_start:h_start + crop_size,
                                    w_start:w_start + crop_size, :]

        # Transpose: (B, window_size, H, W, C) → (B, C, window_size, H, W)
        batch_data = batch_data.transpose(0, 4, 1, 2, 3)
        batch_data = torch.from_numpy(batch_data).float().to(device)

        b_features = model.extract_features(batch_data)  # (B, 1024, 1, 1, 1)
        b_features = b_features.data.cpu().numpy()[:, :, 0, 0, 0]
        all_features.append(b_features)

    return np.concatenate(all_features, axis=0)


# ============================================================================
# 4. TriDet 模型推理 + C NMS
# ============================================================================

def run_tridet_inference(feats_npy, model, cfg, device,
                         video_id='01', fps=25.0, duration=None):
    """
    用 TriDet 模型进行推理 (内部包含 C NMS 后处理).

    Args:
        feats_npy: np.ndarray (T, 2048), I3D 特征
        model:     TriDet 模型 (eval mode)
        cfg:       配置 dict
        device:    torch device
        video_id:  视频标识
        fps:       帧率
        duration:  视频时长 (秒), None 则自动推算
    Returns:
        results: list of dict, 每个 dict 包含:
                 video_id, segments (N,2), scores (N,), labels (N,)
    """
    feat_stride = cfg['dataset']['feat_stride']
    num_frames = cfg['dataset']['num_frames']

    # 推算时长
    if duration is None:
        duration = feats_npy.shape[0] * feat_stride / fps

    # C x T (模型期望格式)
    feats_tensor = torch.from_numpy(
        np.ascontiguousarray(feats_npy.transpose())
    ).float()

    # 构造输入 (inference 只需要一个样本)
    video_data = [{
        'video_id': video_id,
        'feats': feats_tensor,          # C x T
        'fps': fps,
        'duration': duration,
        'feat_stride': feat_stride,
        'feat_num_frames': num_frames,
        'segments': None,               # inference 无需 GT
        'labels': None,
    }]

    model.eval()
    with torch.no_grad():
        results = model(video_data)

    return results


# ============================================================================
# 5. 结果保存为 TXT
# ============================================================================

def save_results_txt(results, output_dir, label_names=None):
    """
    将检测结果保存为可读的 TXT 文件.

    输出格式 (每行一个检测):
        video_id  start_time  end_time  score  label_name

    Args:
        results:     model 推理输出 (list of dict)
        output_dir:  输出目录
        label_names: dict {label_id: label_name}, None 则用默认 THUMOS14 标签
    """
    # THUMOS14 默认标签
    if label_names is None:
        label_names = {
            0: "BaseballPitch", 1: "BasketballDunk", 2: "Billiards",
            3: "CleanAndJerk", 4: "CliffDiving", 5: "CricketBowling",
            6: "CricketShot", 7: "Diving", 8: "FrisbeeCatch",
            9: "GolfSwing", 10: "HammerThrow", 11: "HighJump",
            12: "JavelinThrow", 13: "LongJump", 14: "PoleVault",
            15: "Shotput", 16: "SoccerPenalty", 17: "TennisSwing",
            18: "ThrowDiscus", 19: "VolleyballSpiking",
        }

    os.makedirs(output_dir, exist_ok=True)

    for r in results:
        vid = r['video_id']
        segs = r['segments'].cpu().numpy()
        scores = r['scores'].cpu().numpy()
        labels = r['labels'].cpu().numpy()

        txt_path = os.path.join(output_dir, f"{vid}.txt")

        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write("# TriDet 动作检测结果\n")
            f.write(f"# 视频: {vid}\n")
            f.write(f"# 检测数: {len(segs)}\n")
            f.write("# " + "-" * 60 + "\n")
            f.write(f"{'# 序号':>5s}  "
                    f"{'起始(s)':>10s}  "
                    f"{'结束(s)':>10s}  "
                    f"{'置信度':>8s}  "
                    f"{'标签ID':>6s}  "
                    f"{'标签名'}\n")

            # 按置信度从高到低排序
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

        print(f"[结果保存] {txt_path}  ({len(segs)} 个检测)")

        # 同时保存摘要
        summary_path = os.path.join(output_dir, f"{vid}_summary.txt")
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write(f"视频: {vid}\n")
            f.write(f"总检测数: {len(segs)}\n")
            f.write(f"检测时长范围: {float(segs[:, 0].min()):.1f}s - "
                    f"{float(segs[:, 1].max()):.1f}s\n\n")

            # 按类别统计
            unique_labels, counts = np.unique(labels, return_counts=True)
            f.write("各类别检测数量:\n")
            for lid, cnt in zip(unique_labels, counts):
                label_str = label_names.get(int(lid), f"class_{int(lid)}")
                f.write(f"  {label_str}: {cnt}\n")

            # Top-10 高置信度
            top_k = min(10, len(segs))
            top_idx = sort_idx[:top_k]
            f.write(f"\nTop-{top_k} 高置信度检测:\n")
            for rank, idx in enumerate(top_idx, 1):
                start_s = float(segs[idx, 0])
                end_s = float(segs[idx, 1])
                score = float(scores[idx])
                label_id = int(labels[idx])
                label_str = label_names.get(label_id, f"class_{label_id}")
                f.write(f"  #{rank}: [{start_s:.2f}s - {end_s:.2f}s] "
                        f"{label_str} (conf={score:.4f})\n")

        print(f"[结果保存] {summary_path}")

    return txt_path


# ============================================================================
# 主流水线
# ============================================================================

def main():

    print("Running")

    parser = argparse.ArgumentParser(
        description='端到端视频动作检测: 帧提取→光流→I3D→TriDet→NMS→TXT')

    # 必须参数
    parser.add_argument('--video', type=str, required=True,
                        help='输入视频路径')
    parser.add_argument('--config', type=str, default='configs/id3_i3d.yaml',
                        help='TriDet 配置文件')
    parser.add_argument('--ckpt', type=str, default='epoch_039.pth.tar',
                        help='TriDet 模型权重')
    parser.add_argument('--output_dir', type=str, default='result',
                        help='结果输出目录')

    # 可选参数
    parser.add_argument('--rgb_model', type=str,
                        default='pytorch-i3d-feature-extraction-master/models/rgb_imagenet.pt',
                        help='RGB I3D 模型权重')
    parser.add_argument('--flow_model', type=str,
                        default='pytorch-i3d-feature-extraction-master/models/flow_imagenet.pt',
                        help='Flow I3D 模型权重')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='计算设备')
    parser.add_argument('--target_fps', type=int, default=25,
                        help='目标帧率')
    parser.add_argument('--feat_stride', type=int, default=4,
                        help='特征滑窗步长')
    parser.add_argument('--num_frames', type=int, default=16,
                        help='每个滑窗的帧数')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='I3D 特征提取批大小')
    parser.add_argument('--save_npy', action='store_true', default=False,
                        help='同时保存中间 .npy 特征文件')

    args = parser.parse_args()

    # --- 验证输入 ---
    if not os.path.exists(args.video):
        print(f"[ERROR] 视频文件不存在: {args.video}")
        sys.exit(1)
    if not os.path.exists(args.config):
        print(f"[ERROR] 配置文件不存在: {args.config}")
        sys.exit(1)
    if not os.path.exists(args.ckpt):
        print(f"[ERROR] 模型权重不存在: {args.ckpt}")
        sys.exit(1)

    # --- 切换到项目根目录 (确保相对路径正确) ---
    os.chdir(_PROJ_ROOT)

    # --- 设备 ---
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"[设备] {device}")
    if device.type == 'cuda':
        print(f"[GPU] {torch.cuda.get_device_name(device)}")
        print(f"[显存] {torch.cuda.get_device_properties(device).total_memory / 1024**3:.1f} GB")

    overall_start = time.time()

    # ========================================================================
    # Step 1: 帧提取 (OpenCV)
    # ========================================================================
    print("\n" + "=" * 60)
    print("[Step 1/5] OpenCV 帧提取")
    print("=" * 60)
    t1 = time.time()

    frames, actual_fps, native_fps = extract_frames_opencv(
        args.video,
        target_fps=args.target_fps,
        target_size=(340, 256)
    )
    total_frames = frames.shape[0]
    duration = total_frames / actual_fps

    print(f"         完成, 耗时 {time.time() - t1:.1f}s")
    print(f"         总帧数={total_frames}, FPS={actual_fps}, 时长={duration:.1f}s")

    # ========================================================================
    # Step 2: 光流计算 (Farneback)
    # ========================================================================
    print("\n" + "=" * 60)
    print("[Step 2/5] 光流计算 (Farneback)")
    print("=" * 60)
    t2 = time.time()

    flow_frames = compute_optical_flow(frames)
    flow_total = flow_frames.shape[0]

    print(f"         完成, 耗时 {time.time() - t2:.1f}s")
    print(f"         光流帧数={flow_total}")

    # ========================================================================
    # Step 3: I3D 特征提取 (2048-dim = RGB 1024 + Flow 1024)
    # ========================================================================
    print("\n" + "=" * 60)
    print("[Step 3/5] I3D 特征提取 (RGB + Flow → 2048-dim)")
    print("=" * 60)
    t3 = time.time()

    # 加载 I3D 模型
    print(f"[I3D] 加载 RGB 模型: {args.rgb_model}")
    rgb_model = load_i3d_model(args.rgb_model, 'rgb', device)
    print(f"[I3D] 加载 Flow 模型: {args.flow_model}")
    flow_model = load_i3d_model(args.flow_model, 'flow', device)

    # RGB 滑窗
    window_indices_rgb = build_windows(total_frames, args.num_frames, args.feat_stride)
    print(f"[I3D] RGB 滑窗: {window_indices_rgb.shape[0]} 个窗口")

    # RGB 特征 (1024-dim)
    print("[I3D] 提取 RGB 特征...")
    rgb_feats = extract_i3d_features(
        frames, rgb_model, window_indices_rgb,
        sample_mode='center_crop', crop_size=224,
        batch_size=args.batch_size, device=device
    )
    print(f"      RGB 特征形状: {rgb_feats.shape}")

    # Flow 滑窗
    window_indices_flow = build_windows(flow_total, args.num_frames, args.feat_stride)
    print(f"[I3D] Flow 滑窗: {window_indices_flow.shape[0]} 个窗口")

    # Flow 特征 (1024-dim)
    print("[I3D] 提取 Flow 特征...")
    flow_feats = extract_i3d_features(
        flow_frames, flow_model, window_indices_flow,
        sample_mode='center_crop', crop_size=224,
        batch_size=args.batch_size, device=device
    )
    print(f"      Flow 特征形状: {flow_feats.shape}")

    # 对齐 RGB 和 Flow 特征 (取最小窗口数)
    min_windows = min(rgb_feats.shape[0], flow_feats.shape[0])
    rgb_feats = rgb_feats[:min_windows]
    flow_feats = flow_feats[:min_windows]

    # 拼接 → 2048-dim
    combined_feats = np.concatenate([rgb_feats, flow_feats], axis=1)
    print(f"[I3D] 拼接特征形状: {combined_feats.shape}  (2048-dim)")
    print(f"         完成, 耗时 {time.time() - t3:.1f}s")

    # 可选: 保存中间特征
    if args.save_npy:
        npy_dir = os.path.join(args.output_dir, 'features')
        os.makedirs(npy_dir, exist_ok=True)
        vid_name = os.path.splitext(os.path.basename(args.video))[0]
        np.save(os.path.join(npy_dir, f'{vid_name}.npy'),
                combined_feats.astype(np.float32))
        print(f"[中间特征] 已保存到 {npy_dir}/{vid_name}.npy")

    # 释放 I3D 模型显存
    del rgb_model, flow_model
    torch.cuda.empty_cache()

    # ========================================================================
    # Step 4: TriDet 模型推理 + C NMS
    # ========================================================================
    print("\n" + "=" * 60)
    print("[Step 4/5] TriDet 模型推理 + C NMS")
    print("=" * 60)
    t4 = time.time()

    # 加载配置
    cfg = load_config(args.config)

    # ---- 动态调整 max_seq_len 以适配实际特征长度 ----
    # 原始 max_seq_len=128 是针对训练设置的小值, 推理长视频需要增大
    # 必须能被 max_div_factor 整除 (backbone_arch[-1]=5 → 6 FPN levels → 2^5=32)
    max_div_factor = cfg['model']['scale_factor'] ** cfg['model']['backbone_arch'][-1]
    raw_feat_len = combined_feats.shape[0]
    required_min = ((raw_feat_len + max_div_factor - 1) // max_div_factor) * max_div_factor
    original_max_seq_len = cfg['dataset']['max_seq_len']
    if required_min > original_max_seq_len:
        print(f"[TriDet] max_seq_len 不足: 原始={original_max_seq_len}, "
              f"需要 >={required_min} (特征长度={raw_feat_len})")
        cfg['dataset']['max_seq_len'] = required_min
        print(f"[TriDet] 已自动调整为 {required_min}")

    # 确保输入维度匹配
    cfg['model']['input_dim'] = cfg['dataset']['input_dim']
    cfg['model']['num_classes'] = cfg['dataset']['num_classes']
    cfg['model']['max_seq_len'] = cfg['dataset']['max_seq_len']
    cfg['model']['train_cfg'] = cfg['train_cfg']
    cfg['model']['test_cfg'] = cfg['test_cfg']

    print(f"[TriDet] 配置: input_dim={cfg['dataset']['input_dim']}, "
          f"num_classes={cfg['dataset']['num_classes']}, "
          f"max_seq_len={cfg['dataset']['max_seq_len']}")

    # 构建模型
    model = make_meta_arch(cfg['model_name'], **cfg['model'])
    total_params = sum(p.numel() for p in model.parameters())
    print(f"[TriDet] 参数量: {total_params:,}")

    # 加载 EMA 权重 (checkpoint 使用 DataParallel 保存, 带 module. 前缀)
    print(f"[TriDet] 加载权重: {args.ckpt}")
    checkpoint = torch.load(args.ckpt, map_location=device, weights_only=False)

    if 'state_dict_ema' in checkpoint:
        state_dict = checkpoint['state_dict_ema']
        print("[TriDet] 使用 EMA 权重")
    else:
        state_dict = checkpoint['state_dict']
        print("[TriDet] 使用普通权重 (未找到 EMA)")

    # 处理 DataParallel 的 module. 前缀
    # 直接加载到非 DataParallel 模型需要去掉前缀
    if any(k.startswith('module.') for k in state_dict.keys()):
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        print("[TriDet] 已移除 DataParallel 的 'module.' 前缀")

    model.load_state_dict(state_dict)
    del checkpoint

    model = model.to(device)
    model.eval()
    print("[TriDet] 模型加载完成")

    # 推理 (内部自动执行 C NMS)
    vid_name = os.path.splitext(os.path.basename(args.video))[0]
    results = run_tridet_inference(
        combined_feats, model, cfg, device,
        video_id=vid_name,
        fps=actual_fps,
        duration=duration
    )

    print(f"         完成, 耗时 {time.time() - t4:.1f}s")

    # ========================================================================
    # Step 5: 保存结果 TXT
    # ========================================================================
    print("\n" + "=" * 60)
    print("[Step 5/5] 保存检测结果 TXT")
    print("=" * 60)

    save_results_txt(results, args.output_dir)

    overall_end = time.time()
    print("\n" + "=" * 60)
    print("  全部完成!")
    print("=" * 60)
    print(f"  总耗时: {overall_end - overall_start:.1f}s")
    print(f"  输出目录: {os.path.abspath(args.output_dir)}")

    # 打印检测摘要
    for r in results:
        segs = r['segments'].cpu().numpy()
        scores = r['scores'].cpu().numpy()
        labels = r['labels'].cpu().numpy()
        print(f"\n  视频: {r['video_id']}")
        print(f"  检测到 {len(segs)} 个动作段")
        if len(segs) > 0:
            top_k = min(5, len(segs))
            sort_idx = np.argsort(-scores)
            print(f"  Top-{top_k} 检测:")
            label_names = {
                0: "BaseballPitch", 1: "BasketballDunk", 2: "Billiards",
                3: "CleanAndJerk", 4: "CliffDiving", 5: "CricketBowling",
                6: "CricketShot", 7: "Diving", 8: "FrisbeeCatch",
                9: "GolfSwing", 10: "HammerThrow", 11: "HighJump",
                12: "JavelinThrow", 13: "LongJump", 14: "PoleVault",
                15: "Shotput", 16: "SoccerPenalty", 17: "TennisSwing",
                18: "ThrowDiscus", 19: "VolleyballSpiking",
            }
            for i in range(top_k):
                idx = sort_idx[i]
                print(f"    [{float(segs[idx, 0]):.1f}s - {float(segs[idx, 1]):.1f}s] "
                      f"{label_names.get(int(labels[idx]), f'class_{int(labels[idx])}')} "
                      f"(conf={float(scores[idx]):.4f})")


if __name__ == '__main__':
    main()

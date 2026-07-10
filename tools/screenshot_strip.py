#!/usr/bin/env python
"""
视频截图横向拼接工具

从 MP4 视频中按时间区间均匀截取指定数量的帧，横向拼接为一张图片。

用法:
    python tools/screenshot_strip.py <video.mp4> <start> <end> <count> [--output result/]

参数:
    video.mp4   输入视频路径
    start       开始时间，格式 mm:ss（如 1:30 表示 1 分 30 秒）
    end         结束时间，格式 mm:ss
    count       截图数量（≥2）
    --output    输出目录，默认 result/

示例:
    python tools/screenshot_strip.py video.mp4 0:30 2:15 5
    python tools/screenshot_strip.py video.mp4 1:00 3:00 8 --output my_strips/

输出:
    横向拼接的 PNG 图片，保存在输出目录下，文件名含视频名和时间戳。
"""

import argparse
import os
import sys
import time as time_module

# Windows 上避免 MSMF 后端问题（与项目其他脚本保持一致）
os.environ["OPENCV_VIDEOIO_PRIORITY_MSMF"] = "0"

import cv2
import numpy as np
from PIL import Image


def parse_time(t: str) -> float:
    """
    解析 mm:ss 或 m:ss 或 ss 格式的时间字符串为秒数（浮点）。

    >>> parse_time("1:30")
    90.0
    >>> parse_time("0:05")
    5.0
    >>> parse_time("120")
    120.0
    >>> parse_time("2:00.5")
    120.5
    """
    t = t.strip()
    if ":" in t:
        parts = t.split(":")
        if len(parts) != 2:
            raise ValueError(f"无法解析时间 '{t}'：应为 mm:ss 格式")
        minutes = float(parts[0])
        seconds = float(parts[1])
        return minutes * 60.0 + seconds
    else:
        return float(t)


def extract_frame_at_time(cap: cv2.VideoCapture, time_sec: float) -> np.ndarray | None:
    """
    在指定时间点提取一帧。

    使用 CAP_PROP_POS_MSEC 进行毫秒级定位，然后读取一帧。
    返回 BGR 格式的 numpy 数组（H×W×3），失败返回 None。
    """
    cap.set(cv2.CAP_PROP_POS_MSEC, time_sec * 1000.0)
    ret, frame = cap.read()
    if not ret or frame is None:
        return None
    return frame


def concatenate_horizontal(frames: list[np.ndarray]) -> np.ndarray:
    """
    将多帧 BGR 图像横向拼接。

    所有帧高度统一为最高帧的高度（保持宽高比缩放），
    左右之间加 2px 分隔线。
    """
    if len(frames) == 1:
        return frames[0]

    # 统一高度为最高帧
    max_h = max(f.shape[0] for f in frames)
    resized = []
    for f in frames:
        h, w = f.shape[:2]
        if h != max_h:
            new_w = int(w * max_h / h)
            f = cv2.resize(f, (new_w, max_h), interpolation=cv2.INTER_LANCZOS4)
        resized.append(f)

    # 添加分隔线（白色）
    separator = np.full((max_h, 2, 3), 255, dtype=np.uint8)

    # 拼接
    parts = []
    for i, f in enumerate(resized):
        if i > 0:
            parts.append(separator)
        parts.append(f)

    return np.hstack(parts)


def main():
    parser = argparse.ArgumentParser(
        description="从视频中截取多帧并横向拼接为一张图片",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python tools/screenshot_strip.py video.mp4 0:30 2:15 5
  python tools/screenshot_strip.py video.mp4 1:00 3:00 8 --output my_strips/
        """,
    )
    parser.add_argument("video", help="输入 MP4 视频路径")
    parser.add_argument("start", help="开始时间 (mm:ss)")
    parser.add_argument("end", help="结束时间 (mm:ss)")
    parser.add_argument("count", type=int, help="截图数量 (≥2)")
    parser.add_argument(
        "--output", "-o", default="result/", help="输出目录 (默认 result/)"
    )
    args = parser.parse_args()

    # ── 参数校验 ──────────────────────────────────────────────
    if args.count < 2:
        print("错误: 截图数量必须 ≥ 2", file=sys.stderr)
        sys.exit(1)

    if not os.path.isfile(args.video):
        print(f"错误: 视频文件不存在: {args.video}", file=sys.stderr)
        sys.exit(1)

    try:
        start_sec = parse_time(args.start)
        end_sec = parse_time(args.end)
    except ValueError as e:
        print(f"错误: {e}", file=sys.stderr)
        sys.exit(1)

    if start_sec < 0:
        print("错误: 开始时间不能为负数", file=sys.stderr)
        sys.exit(1)
    if end_sec <= start_sec:
        print("错误: 结束时间必须大于开始时间", file=sys.stderr)
        sys.exit(1)

    # ── 打开视频 ──────────────────────────────────────────────
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"错误: 无法打开视频: {args.video}", file=sys.stderr)
        sys.exit(1)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    duration = total_frames / fps if fps > 0 else 0

    print(f"视频信息: {total_frames} 帧, {fps:.2f} fps, 时长 {duration:.1f}s")
    print(f"截取区间: {args.start} → {args.end} (共 {end_sec - start_sec:.1f}s)")
    print(f"截图数量: {args.count}")

    if end_sec > duration + 0.5:  # 容忍 0.5s 误差
        print(
            f"警告: 结束时间 ({end_sec:.1f}s) 超出视频时长 ({duration:.1f}s)，将截取到视频末尾"
        )
        end_sec = min(end_sec, duration)
        if end_sec <= start_sec:
            print("错误: 调整后结束时间不早于开始时间", file=sys.stderr)
            cap.release()
            sys.exit(1)

    # ── 计算截取时间点 ────────────────────────────────────────
    if args.count == 1:
        time_points = [(start_sec + end_sec) / 2]  # 单张取中点
    else:
        time_points = [
            start_sec + i * (end_sec - start_sec) / (args.count - 1)
            for i in range(args.count)
        ]

    # ── 逐帧截取 ──────────────────────────────────────────────
    frames = []
    for i, t in enumerate(time_points):
        mins = int(t // 60)
        secs = t % 60
        ts_label = f"{mins}:{secs:05.2f}"
        print(f"  [{i+1}/{args.count}] 截取 {ts_label} ...", end=" ")
        frame = extract_frame_at_time(cap, t)
        if frame is None:
            print("失败!")
            continue
        print(f"成功 ({frame.shape[1]}×{frame.shape[0]})")
        frames.append(frame)

    cap.release()

    if len(frames) < 2:
        print("错误: 有效截图不足 2 张，无法拼接", file=sys.stderr)
        sys.exit(1)
    if len(frames) < args.count:
        print(f"警告: 仅成功截取 {len(frames)}/{args.count} 帧")

    # ── 横向拼接 ──────────────────────────────────────────────
    print(f"拼接 {len(frames)} 张截图 ...")
    strip = concatenate_horizontal(frames)

    # ── 保存 ──────────────────────────────────────────────────
    os.makedirs(args.output, exist_ok=True)

    video_stem = os.path.splitext(os.path.basename(args.video))[0]
    timestamp = time_module.strftime("%Y%m%d_%H%M%S")
    out_name = f"{video_stem}_strip_{timestamp}.png"
    out_path = os.path.join(args.output, out_name)

    # 转换为 RGB 后用 Pillow 保存高质量 PNG
    strip_rgb = cv2.cvtColor(strip, cv2.COLOR_BGR2RGB)
    Image.fromarray(strip_rgb).save(
        out_path, format="PNG", compress_level=0  # 0 = 无压缩，最快保存
    )
    print(f"\n✓ 已保存: {out_path}")
    print(f"  尺寸: {strip.shape[1]}×{strip.shape[0]} px")


if __name__ == "__main__":
    main()

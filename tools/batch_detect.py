#!/usr/bin/env python
"""
批量视频动作检测 + 精细化耗时分析

逐视频运行 E2E full_pipeline 的所有阶段，并在每个子步骤记录耗时，
输出单视频 JSON 与汇总 CSV，支持 YOLO 主体检测耗时采集。

用法:
    # 基础模式（无 YOLO）
    python tools/batch_detect.py

    # 含 YOLO 可视化 + 主体检测耗时
    python tools/batch_detect.py --visualize

    # 指定输入输出
    python tools/batch_detect.py --video_dir data/videos --output_dir result/

    # 断点续跑（跳过已有 profile 的视频）
    python tools/batch_detect.py --resume
"""

import argparse
import csv
import json
import os
import sys
import time as time_module
from datetime import timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

# ── 项目路径 ──────────────────────────────────────────────────
_PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJ_ROOT not in sys.path:
    sys.path.insert(0, _PROJ_ROOT)

# Windows MSMF 兼容
os.environ["OPENCV_VIDEOIO_PRIORITY_MSMF"] = "0"

import cv2

from libs.core import load_config
from libs.modeling import make_meta_arch
from E2E.module.loader import load_frames_from_video
from E2E.module.flow import compute_optical_flow
from E2E.module.features import build_windows, load_i3d_model, extract_features_for_video
from E2E.module.inference import run_tridet_inference, save_results_txt, THUMOS14_LABEL_NAMES


# ============================================================================
# 工具函数
# ============================================================================


def _resolve_path(path: str, base_dir: str = _PROJ_ROOT) -> str:
    """解析路径（相对 → 绝对）。"""
    if os.path.isabs(path):
        return path
    return os.path.normpath(os.path.join(base_dir, path))


def _format_duration(seconds: float) -> str:
    """秒 → H:MM:SS 字符串。"""
    return str(timedelta(seconds=int(seconds)))


def _video_metadata(video_path: str) -> Dict[str, Any]:
    """读取视频元数据（不解码全部帧）。"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频: {video_path}")

    meta = {
        "file_size_mb": round(os.path.getsize(video_path) / (1024 * 1024), 2),
        "native_fps": round(cap.get(cv2.CAP_PROP_FPS), 2),
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "total_frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
    }
    if meta["native_fps"] <= 0:
        meta["native_fps"] = 30.0
    if meta["total_frames"] > 0:
        meta["duration_seconds"] = round(meta["total_frames"] / meta["native_fps"], 2)
    else:
        meta["duration_seconds"] = 0
    cap.release()
    return meta


# ============================================================================
# TimedDetector — 包装 SubjectDetector 以采集 YOLO 逐帧耗时
# ============================================================================


class TimedDetector:
    """包装 SubjectDetector，记录每次 detect() 调用的耗时。"""

    def __init__(self, detector, device: str = "cuda:0"):
        self._detector = detector
        self._device = device
        self.call_times: List[float] = []  # 每次 detect 耗时（秒）
        self.call_count: int = 0

    def detect(self, frame: np.ndarray, conf_threshold: float = 0.5) -> list:
        t0 = time_module.time()
        result = self._detector.detect(frame, conf_threshold=conf_threshold)
        elapsed = time_module.time() - t0
        self.call_times.append(elapsed)
        self.call_count += 1
        return result

    @property
    def total_detect_time(self) -> float:
        return sum(self.call_times)

    @property
    def stats(self) -> Dict[str, Any]:
        if not self.call_times:
            return {"count": 0, "total_s": 0.0, "avg_ms": 0.0,
                    "min_ms": 0.0, "max_ms": 0.0}
        arr = np.array(self.call_times)
        return {
            "count": int(len(arr)),
            "total_s": round(float(arr.sum()), 3),
            "avg_ms": round(float(arr.mean()) * 1000, 2),
            "min_ms": round(float(arr.min()) * 1000, 2),
            "max_ms": round(float(arr.max()) * 1000, 2),
        }


# ============================================================================
# 单视频处理
# ============================================================================


def process_one_video(
    video_path: str,
    video_name: str,
    args: argparse.Namespace,
    device: torch.device,
) -> Dict[str, Any]:
    """处理单个视频，返回完整耗时 profile 字典。"""

    profile: Dict[str, Any] = {
        "video_name": video_name,
        "video_path": video_path,
        "timestamp": time_module.strftime("%Y-%m-%d %H:%M:%S"),
    }

    t_overall_start = time_module.time()

    # ── 视频元数据 ──────────────────────────────────────────
    try:
        meta = _video_metadata(video_path)
        profile.update(meta)
    except Exception as e:
        profile["error"] = f"元数据读取失败: {e}"
        profile["total_time_seconds"] = round(time_module.time() - t_overall_start, 2)
        return profile

    profile["duration_hms"] = _format_duration(profile["duration_seconds"])

    print(f"\n{'=' * 70}")
    print(f"  视频: {video_name}  |  {profile['width']}×{profile['height']}  "
          f"|  {profile['native_fps']}fps  |  {profile['duration_hms']}  "
          f"|  {profile['file_size_mb']}MB")
    print(f"{'=' * 70}")

    stage_times: Dict[str, float] = {}

    # ====================================================================
    # Step 1: 抽帧
    # ====================================================================
    print("[Step 1/5] 抽帧 ...", end=" ", flush=True)
    t1 = time_module.time()
    try:
        frames, actual_fps = load_frames_from_video(
            video_path,
            target_fps=args.target_fps,
            target_size=(340, 256),
        )
    except Exception as e:
        profile["error"] = f"抽帧失败: {e}"
        profile["total_time_seconds"] = round(time_module.time() - t_overall_start, 2)
        return profile

    total_frames = frames.shape[0]
    duration_s = total_frames / actual_fps if actual_fps > 0 else 0
    stage_times["01_frame_extraction"] = round(time_module.time() - t1, 3)
    profile["extracted_frames"] = total_frames
    profile["actual_fps"] = round(actual_fps, 2)
    print(f"✓ {total_frames} 帧, {stage_times['01_frame_extraction']:.1f}s")

    # ====================================================================
    # Step 2: 光流
    # ====================================================================
    print("[Step 2/5] 光流计算 ...", end=" ", flush=True)
    t2 = time_module.time()
    try:
        flow_frames = compute_optical_flow(frames)
    except Exception as e:
        profile["error"] = f"光流计算失败: {e}"
        profile["total_time_seconds"] = round(time_module.time() - t_overall_start, 2)
        profile["stages"] = stage_times
        return profile

    flow_total = flow_frames.shape[0]
    stage_times["02_optical_flow"] = round(time_module.time() - t2, 3)
    print(f"✓ {flow_total} 帧, {stage_times['02_optical_flow']:.1f}s")

    # ====================================================================
    # Step 3: I3D 特征提取（精细拆分）
    # ====================================================================
    print("[Step 3/5] I3D 特征提取 ...")
    t3 = time_module.time()

    # 3a. 加载 RGB I3D 模型
    print("          [3a] 加载 RGB I3D 模型 ...", end=" ", flush=True)
    t3a = time_module.time()
    rgb_model = load_i3d_model(args.rgb_model, "rgb", device)
    stage_times["03a_i3d_rgb_model_load"] = round(time_module.time() - t3a, 3)
    rgb_params = sum(p.numel() for p in rgb_model.parameters())
    print(f"✓ {stage_times['03a_i3d_rgb_model_load']:.1f}s  ({rgb_params:,} params)")

    # 3b. RGB 滑动窗口
    print("          [3b] RGB 滑动窗口 ...", end=" ", flush=True)
    t3b = time_module.time()
    window_indices_rgb = build_windows(total_frames, args.num_frames, args.feat_stride)
    stage_times["03b_rgb_windows"] = round(time_module.time() - t3b, 3)
    print(f"✓ {window_indices_rgb.shape[0]} 窗口, {stage_times['03b_rgb_windows']:.2f}s")

    # 3c. RGB 特征提取
    print("          [3c] RGB 特征提取 ...", end=" ", flush=True)
    t3c = time_module.time()
    rgb_feats = extract_features_for_video(
        frames, rgb_model, window_indices_rgb,
        sample_mode="center_crop", crop_size=224,
        batch_size=args.batch_size, device=device,
    )
    stage_times["03c_rgb_features"] = round(time_module.time() - t3c, 3)
    print(f"✓ {rgb_feats.shape}, {stage_times['03c_rgb_features']:.1f}s")

    # 3d. 加载 Flow I3D 模型
    print("          [3d] 加载 Flow I3D 模型 ...", end=" ", flush=True)
    t3d = time_module.time()
    flow_model = load_i3d_model(args.flow_model, "flow", device)
    stage_times["03d_i3d_flow_model_load"] = round(time_module.time() - t3d, 3)
    flow_params = sum(p.numel() for p in flow_model.parameters())
    print(f"✓ {stage_times['03d_i3d_flow_model_load']:.1f}s  ({flow_params:,} params)")

    # 3e. Flow 滑动窗口
    print("          [3e] Flow 滑动窗口 ...", end=" ", flush=True)
    t3e = time_module.time()
    window_indices_flow = build_windows(flow_total, args.num_frames, args.feat_stride)
    stage_times["03e_flow_windows"] = round(time_module.time() - t3e, 3)
    print(f"✓ {window_indices_flow.shape[0]} 窗口, {stage_times['03e_flow_windows']:.2f}s")

    # 3f. Flow 特征提取
    print("          [3f] Flow 特征提取 ...", end=" ", flush=True)
    t3f = time_module.time()
    flow_feats = extract_features_for_video(
        flow_frames, flow_model, window_indices_flow,
        sample_mode="center_crop", crop_size=224,
        batch_size=args.batch_size, device=device,
    )
    stage_times["03f_flow_features"] = round(time_module.time() - t3f, 3)
    print(f"✓ {flow_feats.shape}, {stage_times['03f_flow_features']:.1f}s")

    # 3g. 对齐 & 拼接
    print("          [3g] 特征拼接 ...", end=" ", flush=True)
    t3g = time_module.time()
    min_windows = min(rgb_feats.shape[0], flow_feats.shape[0])
    rgb_feats = rgb_feats[:min_windows]
    flow_feats = flow_feats[:min_windows]
    combined_feats = np.concatenate([rgb_feats, flow_feats], axis=1)
    stage_times["03g_feature_concat"] = round(time_module.time() - t3g, 3)
    print(f"✓ {combined_feats.shape} (2048-dim), {stage_times['03g_feature_concat']:.2f}s")

    stage_times["03_i3d_total"] = round(time_module.time() - t3, 3)

    # 释放 I3D 模型显存
    del rgb_model, flow_model
    torch.cuda.empty_cache()
    if device.type == "cuda":
        profile["gpu_after_i3d_mb"] = round(
            torch.cuda.max_memory_allocated(device) / (1024 * 1024), 1
        )

    # ====================================================================
    # Step 4: TriDet 推理
    # ====================================================================
    print("[Step 4/5] TriDet 推理 ...")

    # 4a. 加载配置
    print("          [4a] 加载配置 ...", end=" ", flush=True)
    t4a = time_module.time()
    cfg = load_config(args.config)
    stage_times["04a_config_load"] = round(time_module.time() - t4a, 3)
    print(f"✓ {stage_times['04a_config_load']:.2f}s")

    # 4b. 调整 max_seq_len
    max_div_factor = cfg["model"]["scale_factor"] ** cfg["model"]["backbone_arch"][-1]
    raw_feat_len = combined_feats.shape[0]
    required_min = ((raw_feat_len + max_div_factor - 1) // max_div_factor) * max_div_factor
    original_max_seq_len = cfg["dataset"]["max_seq_len"]
    if required_min > original_max_seq_len:
        cfg["dataset"]["max_seq_len"] = required_min

    cfg["model"]["input_dim"] = cfg["dataset"]["input_dim"]
    cfg["model"]["num_classes"] = cfg["dataset"]["num_classes"]
    cfg["model"]["max_seq_len"] = cfg["dataset"]["max_seq_len"]
    cfg["model"]["train_cfg"] = cfg["train_cfg"]
    cfg["model"]["test_cfg"] = cfg["test_cfg"]

    profile["tridet_config"] = {
        "input_dim": cfg["dataset"]["input_dim"],
        "num_classes": cfg["dataset"]["num_classes"],
        "max_seq_len": cfg["dataset"]["max_seq_len"],
        "original_max_seq_len": original_max_seq_len,
    }

    # 4b. 构建模型
    print("          [4b] 构建模型 ...", end=" ", flush=True)
    t4b = time_module.time()
    model = make_meta_arch(cfg["model_name"], **cfg["model"])
    total_params = sum(p.numel() for p in model.parameters())
    stage_times["04b_model_build"] = round(time_module.time() - t4b, 3)
    profile["tridet_params"] = total_params
    print(f"✓ {total_params:,} params, {stage_times['04b_model_build']:.1f}s")

    # 4c. 加载 checkpoint
    print("          [4c] 加载 checkpoint ...", end=" ", flush=True)
    t4c = time_module.time()
    checkpoint = torch.load(args.ckpt, map_location=device, weights_only=False)
    if "state_dict_ema" in checkpoint:
        state_dict = checkpoint["state_dict_ema"]
    else:
        state_dict = checkpoint["state_dict"]
    if any(k.startswith("module.") for k in state_dict.keys()):
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    del checkpoint
    model = model.to(device)
    model.eval()
    stage_times["04c_checkpoint_load"] = round(time_module.time() - t4c, 3)
    print(f"✓ {stage_times['04c_checkpoint_load']:.1f}s")

    # 4d. 推理
    print("          [4d] 前向推理 ...", end=" ", flush=True)
    t4d = time_module.time()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        mem_before = torch.cuda.memory_allocated(device) / (1024 * 1024)

    results = run_tridet_inference(
        combined_feats, model, cfg, device,
        video_id=video_name,
        fps=actual_fps,
        duration=duration_s,
    )

    if device.type == "cuda":
        mem_after = torch.cuda.memory_allocated(device) / (1024 * 1024)
        mem_peak = torch.cuda.max_memory_allocated(device) / (1024 * 1024)
        profile["gpu_tridet_infer_mem_before_mb"] = round(mem_before, 1)
        profile["gpu_tridet_infer_mem_after_mb"] = round(mem_after, 1)
        profile["gpu_tridet_infer_mem_peak_mb"] = round(mem_peak, 1)

    stage_times["04d_tridet_inference"] = round(time_module.time() - t4d, 3)
    num_detections = sum(r["segments"].shape[0] for r in results)
    profile["num_detections"] = num_detections
    print(f"✓ {num_detections} 个检测, {stage_times['04d_tridet_inference']:.1f}s")

    stage_times["04_tridet_total"] = round(
        stage_times["04a_config_load"] + stage_times["04b_model_build"]
        + stage_times["04c_checkpoint_load"] + stage_times["04d_tridet_inference"], 3
    )

    # 释放 TriDet 模型
    del model
    torch.cuda.empty_cache()

    # ====================================================================
    # Step 5: 保存结果
    # ====================================================================
    print("[Step 5/5] 保存结果 ...", end=" ", flush=True)
    t5 = time_module.time()
    save_results_txt(results, args.output_dir)
    stage_times["05_save_results"] = round(time_module.time() - t5, 3)
    print(f"✓ {stage_times['05_save_results']:.2f}s")

    # ====================================================================
    # Step 6-7: YOLO + 可视化（可选）
    # ====================================================================
    if args.visualize:
        print("[Step 6-7] YOLO 主体检测 + 可视化 ...")
        from libs.subject.detector import SubjectDetector
        from E2E.module.visualizer import (
            create_annotated_video,
            predictions_to_action_list,
        )

        # 6a. 加载 YOLO 模型
        print("          [6a] 加载 YOLO 模型 ...", end=" ", flush=True)
        t6a = time_module.time()
        raw_detector = SubjectDetector(model_name=args.yolo_model, device=str(device))
        timed_detector = TimedDetector(raw_detector, device=str(device))
        stage_times["06a_yolo_model_load"] = round(time_module.time() - t6a, 3)
        print(f"✓ {stage_times['06a_yolo_model_load']:.1f}s")

        # 转换检测结果
        action_list = predictions_to_action_list(results, THUMOS14_LABEL_NAMES)
        output_video_path = os.path.join(args.output_dir, f"{video_name}_annotated.mp4")

        # 6b. 可视化（YOLO 检测耗时在 TimedDetector 内部记录）
        print("          [6b] 生成标注视频 ...", end=" ", flush=True)
        t6b = time_module.time()
        create_annotated_video(
            video_path=video_path,
            output_path=output_video_path,
            action_results=action_list,
            detector=timed_detector,
            conf_threshold=args.yolo_conf,
            show_timeline=not args.no_timeline,
            progress=False,  # 批次模式关闭进度条
        )
        vis_total = time_module.time() - t6b
        stage_times["06b_visualization_total"] = round(vis_total, 3)

        # 拆分 YOLO 检测耗时 vs 视频编码耗时
        yolo_stats = timed_detector.stats
        stage_times["06b_yolo_detect_total"] = round(yolo_stats["total_s"], 3)
        stage_times["06b_video_encode_etc"] = round(
            vis_total - yolo_stats["total_s"], 3
        )
        profile["yolo_detection"] = yolo_stats
        profile["annotated_video"] = output_video_path

        print(f"          ✓ 标注视频完成 ({vis_total:.1f}s)")
        print(f"            YOLO 检测: {yolo_stats['count']} 帧, "
              f"总 {yolo_stats['total_s']:.1f}s, "
              f"平均 {yolo_stats['avg_ms']:.1f}ms/帧")
        print(f"            编码/绘图: {stage_times['06b_video_encode_etc']:.1f}s")

        stage_times["06_visualization_total"] = round(
            stage_times["06a_yolo_model_load"] + vis_total, 3
        )

    # ── 汇总 ──────────────────────────────────────────────────
    profile["stages"] = stage_times
    profile["total_time_seconds"] = round(time_module.time() - t_overall_start, 2)
    profile["total_time_hms"] = _format_duration(profile["total_time_seconds"])

    if device.type == "cuda":
        profile["gpu_peak_mb"] = round(
            torch.cuda.max_memory_allocated(device) / (1024 * 1024), 1
        )

    # 保存单视频 profile JSON
    os.makedirs(args.output_dir, exist_ok=True)
    profile_path = os.path.join(args.output_dir, f"{video_name}_profile.json")
    with open(profile_path, "w", encoding="utf-8") as f:
        json.dump(profile, f, indent=2, ensure_ascii=False)

    print(f"\n  ✓ 总耗时: {profile['total_time_hms']}  |  profile: {profile_path}")
    return profile


# ============================================================================
# 汇总报告
# ============================================================================


def _all_stage_keys(profiles: List[Dict]) -> List[str]:
    """从所有 profile 中提取一致的 stages 键名列表。"""
    keys = []
    for p in profiles:
        for k in p.get("stages", {}):
            if k not in keys:
                keys.append(k)
    # 按编号排序
    keys.sort()
    return keys


def print_summary_table(profiles: List[Dict], stage_keys: List[str]) -> None:
    """打印终端汇总表格（纯 ASCII，无第三方依赖）。"""
    if not profiles:
        return

    # 列定义
    col_video = 28
    col_dur = 10
    col_time = 9
    cols = [("Video", col_video), ("Duration", col_dur),
            ("Total", col_time)]
    for k in stage_keys:
        short = k.split("_", 1)[-1][:14]  # 去掉数字前缀，截断
        cols.append((short, 8))

    # 表头
    header = "  ".join(f"{name:{w}}" for name, w in cols)
    sep = "  ".join("-" * w for _, w in cols)
    print(f"\n{'=' * len(header)}")
    print(header)
    print(sep)

    # 数据行
    totals: Dict[str, float] = {}
    for p in profiles:
        vid = p.get("video_name", "?")[:col_video]
        dur = p.get("duration_hms", "?")
        total = p.get("total_time_seconds", 0)
        row = f"{vid:{col_video}}  {dur:{col_dur}}  {total:{col_time}.1f}"
        stages = p.get("stages", {})
        for k in stage_keys:
            v = stages.get(k, 0)
            row += f"  {v:7.1f}" if isinstance(v, (int, float)) else f"  {'?':>7}"
            if k not in totals:
                totals[k] = 0.0
            totals[k] += v
        print(row)
        if "total_time_seconds" not in totals:
            totals["total_time_seconds"] = 0.0
        totals["total_time_seconds"] += total

    # 平均值行
    n = len(profiles)
    print(sep)
    avg_total = totals.get("total_time_seconds", 0) / n
    row = f"{'AVERAGE (n=' + str(n) + ')':{col_video}}  {'':{col_dur}}  {avg_total:{col_time}.1f}"
    for k in stage_keys:
        row += f"  {totals.get(k, 0) / n:7.1f}"
    print(row)

    # 合计行
    sum_total = totals.get("total_time_seconds", 0)
    row = f"{'SUM':{col_video}}  {'':{col_dur}}  {sum_total:{col_time}.1f}"
    for k in stage_keys:
        row += f"  {totals.get(k, 0):7.1f}"
    print(row)
    print(f"{'=' * len(header)}")


def save_summary_csv(profiles: List[Dict], stage_keys: List[str],
                     output_dir: str) -> str:
    """保存汇总 CSV。"""
    csv_path = os.path.join(output_dir, "batch_summary.csv")

    # CSV 列：视频名 + 元数据 + 所有 stage + total + error
    fieldnames = [
        "video_name", "duration_seconds", "duration_hms",
        "native_fps", "width", "height", "file_size_mb",
        "extracted_frames", "num_detections",
    ] + stage_keys + ["total_time_seconds", "error"]

    # 如果有 YOLO 数据，追加
    yolo_fields = ["yolo_frames", "yolo_total_s", "yolo_avg_ms",
                   "yolo_min_ms", "yolo_max_ms"]

    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames + yolo_fields,
                                extrasaction="ignore")
        writer.writerow({fn: fn for fn in writer.fieldnames})  # 写表头

        for p in profiles:
            row = {
                "video_name": p.get("video_name", "?"),
                "duration_seconds": p.get("duration_seconds", 0),
                "duration_hms": p.get("duration_hms", ""),
                "native_fps": p.get("native_fps", 0),
                "width": p.get("width", 0),
                "height": p.get("height", 0),
                "file_size_mb": p.get("file_size_mb", 0),
                "extracted_frames": p.get("extracted_frames", 0),
                "num_detections": p.get("num_detections", 0),
                "total_time_seconds": p.get("total_time_seconds", 0),
                "error": p.get("error", ""),
            }
            stages = p.get("stages", {})
            for k in stage_keys:
                row[k] = stages.get(k, "")
            yolo = p.get("yolo_detection", {})
            row["yolo_frames"] = yolo.get("count", "")
            row["yolo_total_s"] = yolo.get("total_s", "")
            row["yolo_avg_ms"] = yolo.get("avg_ms", "")
            row["yolo_min_ms"] = yolo.get("min_ms", "")
            row["yolo_max_ms"] = yolo.get("max_ms", "")
            writer.writerow(row)

        # 平均值行
        n = len(profiles)
        if n > 0:
            avg_row = {"video_name": f"AVERAGE (n={n})"}
            for fn in fieldnames:
                if fn == "video_name":
                    continue
                vals = [p.get(fn, 0) for p in profiles
                        if isinstance(p.get(fn), (int, float))]
                if vals:
                    avg_row[fn] = round(sum(vals) / len(vals), 3)
            writer.writerow(avg_row)

    return csv_path


# ============================================================================
# 入口
# ============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="批量视频动作检测 + 精细化耗时分析",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python tools/batch_detect.py
  python tools/batch_detect.py --visualize
  python tools/batch_detect.py --video_dir data/videos --resume
        """,
    )

    # 路径参数
    parser.add_argument("--video_dir", type=str, default="data/videos",
                        help="视频目录 (默认 data/videos)")
    parser.add_argument("--output_dir", type=str, default="result",
                        help="输出目录 (默认 result/)")
    parser.add_argument("--config", type=str, default="configs/id3_i3d.yaml",
                        help="TriDet 配置文件")
    parser.add_argument("--ckpt", type=str,
                        default="ckpt/thumos_i3d_baseline/epoch_039.pth.tar",
                        help="TriDet checkpoint")
    parser.add_argument("--rgb_model", type=str,
                        default="E2E/model/rgb_imagenet.pt",
                        help="RGB I3D 权重")
    parser.add_argument("--flow_model", type=str,
                        default="E2E/model/flow_imagenet.pt",
                        help="Flow I3D 权重")

    # 处理参数
    parser.add_argument("--device", type=str, default="cuda:0",
                        help="计算设备 (默认 cuda:0)")
    parser.add_argument("--target_fps", type=int, default=25,
                        help="目标帧率 (默认 25)")
    parser.add_argument("--feat_stride", type=int, default=4,
                        help="特征窗口步长 (默认 4)")
    parser.add_argument("--num_frames", type=int, default=16,
                        help="每窗口帧数 (默认 16)")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="I3D 批次大小 (默认 16)")

    # 可视化 / YOLO
    viz_group = parser.add_argument_group("可视化 (可选)")
    viz_group.add_argument("--visualize", action="store_true", default=False,
                           help="启用 YOLO + 标注视频输出")
    viz_group.add_argument("--yolo_model", type=str,
                           default="E2E/model/yolov8n.pt",
                           help="YOLO 模型路径")
    viz_group.add_argument("--yolo_conf", type=float, default=0.3,
                           help="YOLO 置信度阈值 (默认 0.3)")
    viz_group.add_argument("--no_timeline", action="store_true", default=False,
                           help="禁用底部时间轴")

    # 运行控制
    parser.add_argument("--resume", action="store_true", default=False,
                        help="断点续跑：跳过已有 profile 的视频")

    args = parser.parse_args()

    # ── 路径解析 ──────────────────────────────────────────────
    video_dir = _resolve_path(args.video_dir)
    output_dir = _resolve_path(args.output_dir)
    args.config = _resolve_path(args.config)
    args.ckpt = _resolve_path(args.ckpt)
    args.rgb_model = _resolve_path(args.rgb_model)
    args.flow_model = _resolve_path(args.flow_model)
    args.yolo_model = _resolve_path(args.yolo_model)

    # ── 校验 ──────────────────────────────────────────────────
    if not os.path.isdir(video_dir):
        print(f"[ERROR] 视频目录不存在: {video_dir}", file=sys.stderr)
        sys.exit(1)
    for fp, label in [(args.config, "config"), (args.ckpt, "checkpoint"),
                       (args.rgb_model, "RGB model"), (args.flow_model, "Flow model")]:
        if not os.path.exists(fp):
            print(f"[ERROR] {label} 不存在: {fp}", file=sys.stderr)
            sys.exit(1)

    # ── 扫描视频 ──────────────────────────────────────────────
    video_exts = (".mp4", ".avi", ".mkv", ".webm", ".MP4", ".AVI", ".MKV", ".WEBM")
    video_files = sorted(
        f for f in os.listdir(video_dir)
        if os.path.splitext(f)[1] in video_exts
    )
    if not video_files:
        print(f"[ERROR] {video_dir} 下未找到视频文件", file=sys.stderr)
        sys.exit(1)

    os.makedirs(output_dir, exist_ok=True)

    print(f"{'=' * 70}")
    print(f"  批量视频动作检测 + 耗时分析")
    print(f"{'=' * 70}")
    print(f"  视频目录:   {video_dir}")
    print(f"  输出目录:   {output_dir}")
    print(f"  视频数量:   {len(video_files)}")
    print(f"  可视化:     {'是 (YOLO)' if args.visualize else '否'}")
    print(f"  断点续跑:   {'是' if args.resume else '否'}")
    for i, vf in enumerate(video_files, 1):
        size_mb = os.path.getsize(os.path.join(video_dir, vf)) / (1024 * 1024)
        print(f"    [{i}] {vf}  ({size_mb:.1f} MB)")
    print(f"{'=' * 70}")

    # ── 设备 ──────────────────────────────────────────────────
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"\n[INFO] 设备: {device}")
    if device.type == "cuda":
        print(f"[INFO] GPU: {torch.cuda.get_device_name(device)}")
        print(f"[INFO] VRAM: {torch.cuda.get_device_properties(device).total_memory / 1024**3:.1f} GB")

    # ── 逐视频处理 ────────────────────────────────────────────
    profiles: List[Dict] = []
    success_count = 0
    skip_count = 0
    fail_count = 0

    batch_start = time_module.time()

    for idx, vf in enumerate(video_files, 1):
        video_path = os.path.join(video_dir, vf)
        video_name = os.path.splitext(vf)[0]

        # 断点续跑检查
        profile_path = os.path.join(output_dir, f"{video_name}_profile.json")
        if args.resume and os.path.exists(profile_path):
            print(f"\n[{idx}/{len(video_files)}] {video_name} — 跳过 (已有 profile)")
            try:
                with open(profile_path, "r", encoding="utf-8") as f:
                    existing = json.load(f)
                profiles.append(existing)
                skip_count += 1
            except Exception:
                pass  # 损坏的 profile，重新处理
            else:
                continue

        print(f"\n{'#' * 70}")
        print(f"  [{idx}/{len(video_files)}] 处理: {video_name}")
        print(f"{'#' * 70}")

        try:
            profile = process_one_video(video_path, video_name, args, device)
            profiles.append(profile)
            if "error" in profile:
                fail_count += 1
                print(f"\n  ⚠ 失败: {profile['error']}")
            else:
                success_count += 1
        except Exception as e:
            fail_count += 1
            error_profile = {
                "video_name": video_name,
                "video_path": video_path,
                "error": str(e),
                "timestamp": time_module.strftime("%Y-%m-%d %H:%M:%S"),
                "total_time_seconds": 0,
                "stages": {},
            }
            profiles.append(error_profile)
            # 仍然保存错误的 profile
            with open(profile_path, "w", encoding="utf-8") as f:
                json.dump(error_profile, f, indent=2, ensure_ascii=False)
            print(f"\n  ⚠ 异常: {e}")

    # ── 汇总 ──────────────────────────────────────────────────
    batch_total = time_module.time() - batch_start
    print(f"\n{'=' * 70}")
    print(f"  批次完成!")
    print(f"{'=' * 70}")
    print(f"  成功: {success_count}  |  跳过: {skip_count}  |  失败: {fail_count}")
    print(f"  批次总耗时: {_format_duration(batch_total)}  ({batch_total:.1f}s)")

    # 终端表格
    valid_profiles = [p for p in profiles if "error" not in p]
    if valid_profiles:
        stage_keys = _all_stage_keys(valid_profiles)
        print_summary_table(valid_profiles, stage_keys)

        csv_path = save_summary_csv(valid_profiles, stage_keys, output_dir)
        print(f"\n  汇总 CSV: {csv_path}")

    print(f"\n  输出目录: {os.path.abspath(output_dir)}")


if __name__ == "__main__":
    main()

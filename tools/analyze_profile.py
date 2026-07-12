#!/usr/bin/env python
"""
Profiling 数据分析脚本（JSON 版）

读取 batch_detect.py 产出的 video_profile_json/*.json，输出三张独立表格：
  1. 概述表：特征提取 / TriDet推理 / 可视化 / 总耗时
  2. 可视化分析表：YOLO检测 / 视频编码 / 单帧YOLO / 可视化总耗时
  3. 特征提取分析表：抽帧 / 光流 / RGB特征 / 光流特征 / 特征提取总耗时

每张表末行输出平均数。

用法:
    python tools/analyze_profile.py [--dir result/video_profile_json/] [--output result/]
"""

import argparse
import json
import os
import sys


# ============================================================================
# 工具函数
# ============================================================================


def load_profiles(profile_dir: str) -> list[dict]:
    """加载目录下所有 *_profile.json，按视频名排序。"""
    profiles = []
    for fname in sorted(os.listdir(profile_dir)):
        if not fname.endswith("_profile.json"):
            continue
        path = os.path.join(profile_dir, fname)
        with open(path, "r", encoding="utf-8") as f:
            profiles.append(json.load(f))
    return profiles


def fmt_time(s: float) -> str:
    """时间格式化：≥1s 显示 X.Xs，<1s 显示 XXX ms。"""
    if s < 0.0005:
        return "0"
    if s < 1.0:
        return f"{s * 1000:.0f} ms"
    return f"{s:.1f}s"


def fmt_duration(s: float) -> str:
    """时长格式化：m:ss。"""
    m = int(s // 60)
    sec = int(s % 60)
    return f"{m}:{sec:02d}"


def avg(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


# ============================================================================
# 表定义
# ============================================================================


def build_table1(profiles: list[dict]) -> tuple[list[str], list[list[str]], list[str]]:
    """表1 — 概述表"""
    header = ["视频名称", "时长", "帧数", "特征提取", "TriDet推理", "可视化", "总耗时"]
    rows = []

    fe_vals, tri_vals, vis_vals, total_vals = [], [], [], []

    for p in profiles:
        s = p["stages"]
        fe = s["01_frame_extraction"] + s["02_optical_flow"] + s["03_i3d_total"]
        tri = s["04_tridet_total"]
        vis = s["05_save_results"] + s["06_visualization_total"]
        total = p["total_time_seconds"]

        fe_vals.append(fe)
        tri_vals.append(tri)
        vis_vals.append(vis)
        total_vals.append(total)

        rows.append([
            p["video_name"],
            fmt_duration(p["duration_seconds"]),
            str(p["extracted_frames"]),
            fmt_time(fe),
            fmt_time(tri),
            fmt_time(vis),
            fmt_time(total),
        ])

    # 平均数行
    avg_row = [
        "平均",
        fmt_duration(avg([p["duration_seconds"] for p in profiles])),
        f"{avg([p['extracted_frames'] for p in profiles]):.0f}",
        fmt_time(avg(fe_vals)),
        fmt_time(avg(tri_vals)),
        fmt_time(avg(vis_vals)),
        fmt_time(avg(total_vals)),
    ]

    return header, rows, avg_row


def build_table2(profiles: list[dict]) -> tuple[list[str], list[list[str]], list[str]]:
    """表2 — 可视化分析表"""
    header = ["视频名称", "时长", "帧数", "YOLO主体检测", "视频编码", "平均每帧YOLO(ms)", "可视化总耗时"]
    rows = []

    yolo_vals, enc_vals, vis_vals = [], [], []

    for p in profiles:
        s = p["stages"]
        yolo = s["06a_yolo_model_load"] + s["06b_yolo_detect_total"]
        enc = s["06b_video_encode_etc"]
        vis = s["05_save_results"] + s["06_visualization_total"]
        yolo_avg = p.get("yolo_detection", {}).get("avg_ms", 0)

        yolo_vals.append(yolo)
        enc_vals.append(enc)
        vis_vals.append(vis)

        rows.append([
            p["video_name"],
            fmt_duration(p["duration_seconds"]),
            str(p["extracted_frames"]),
            fmt_time(yolo),
            fmt_time(enc),
            f"{yolo_avg:.1f}",
            fmt_time(vis),
        ])

    avg_row = [
        "平均",
        fmt_duration(avg([p["duration_seconds"] for p in profiles])),
        f"{avg([p['extracted_frames'] for p in profiles]):.0f}",
        fmt_time(avg(yolo_vals)),
        fmt_time(avg(enc_vals)),
        f"{avg([p.get('yolo_detection', {}).get('avg_ms', 0) for p in profiles]):.1f}",
        fmt_time(avg(vis_vals)),
    ]

    return header, rows, avg_row


def build_table3(profiles: list[dict]) -> tuple[list[str], list[list[str]], list[str]]:
    """表3 — 特征提取分析表"""
    header = ["视频名称", "时长", "帧数", "抽帧", "光流提取", "RGB特征计算", "光流特征计算", "特征提取总耗时"]
    rows = []

    ext_vals, of_vals, rgb_vals, flow_vals, fe_vals = [], [], [], [], []

    for p in profiles:
        s = p["stages"]
        ext = s["01_frame_extraction"]
        of = s["02_optical_flow"]
        rgb = s["03c_rgb_features"]
        flow = s["03f_flow_features"]
        fe = s["01_frame_extraction"] + s["02_optical_flow"] + s["03_i3d_total"]

        ext_vals.append(ext)
        of_vals.append(of)
        rgb_vals.append(rgb)
        flow_vals.append(flow)
        fe_vals.append(fe)

        rows.append([
            p["video_name"],
            fmt_duration(p["duration_seconds"]),
            str(p["extracted_frames"]),
            fmt_time(ext),
            fmt_time(of),
            fmt_time(rgb),
            fmt_time(flow),
            fmt_time(fe),
        ])

    avg_row = [
        "平均",
        fmt_duration(avg([p["duration_seconds"] for p in profiles])),
        f"{avg([p['extracted_frames'] for p in profiles]):.0f}",
        fmt_time(avg(ext_vals)),
        fmt_time(avg(of_vals)),
        fmt_time(avg(rgb_vals)),
        fmt_time(avg(flow_vals)),
        fmt_time(avg(fe_vals)),
    ]

    return header, rows, avg_row


# ============================================================================
# 输出
# ============================================================================


def compute_col_widths(header: list[str], rows: list[list[str]], avg_row: list[str]) -> list[int]:
    """计算每列最大宽度（考虑中文双字节）。"""
    all_lines = [header] + rows + [avg_row]
    widths = [0] * len(header)
    for line in all_lines:
        for i, cell in enumerate(line):
            w = 0
            for ch in cell:
                w += 2 if '\u4e00' <= ch <= '\u9fff' or '\u3000' <= ch <= '\u303f' else 1
            widths[i] = max(widths[i], w)
    return widths


def pad_cell(text: str, width: int) -> str:
    """按显示宽度填充单元格。"""
    cur = 0
    for ch in text:
        cur += 2 if '\u4e00' <= ch <= '\u9fff' or '\u3000' <= ch <= '\u303f' else 1
    padding = width - cur
    return text + " " * max(padding, 0)


def print_separator(widths: list[int]):
    print("+" + "+".join("-" * (w + 2) for w in widths) + "+")


def print_row(cells: list[str], widths: list[int]):
    print("| " + " | ".join(pad_cell(c, w) for c, w in zip(cells, widths)) + " |")


def print_table(title: str, header: list[str], rows: list[list[str]], avg_row: list[str]):
    """打印表格（含分隔线、表头、数据行、平均数行）。"""
    widths = compute_col_widths(header, rows, avg_row)

    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print(f"{'=' * 70}")

    print_separator(widths)
    print_row(header, widths)
    print_separator(widths)
    for row in rows:
        print_row(row, widths)
    print_separator(widths)
    print_row(avg_row, widths)
    print_separator(widths)


def export_csv_tables(output_path: str,
                      tables: list[tuple[str, list[str], list[list[str]], list[str]]]):
    """导出三张表到 CSV。"""
    import csv
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    try:
        with open(output_path, "w", encoding="utf-8-sig", newline="") as f:
            writer = csv.writer(f)
            for idx, (title, header, rows, avg_row) in enumerate(tables):
                if idx > 0:
                    writer.writerow([])
                writer.writerow([f"=== {title} ==="])
                writer.writerow(header)
                for row in rows:
                    writer.writerow(row)
                writer.writerow(avg_row)
        print(f"\n报告已导出: {output_path}")
    except PermissionError:
        print(f"\n[警告] 无法写入 {output_path}（文件可能已被 Excel 打开），请关闭后重试")


# ============================================================================
# 主入口
# ============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="分析 batch_detect.py 产出的 video_profile_json 数据"
    )
    parser.add_argument(
        "--dir", default="result/video_profile_json/",
        help="profile JSON 目录 (默认 result/video_profile_json/)"
    )
    parser.add_argument(
        "--output", "-o", default="result/",
        help="输出目录 (默认 result/)"
    )
    args = parser.parse_args()

    if not os.path.isdir(args.dir):
        print(f"[错误] 目录不存在: {args.dir}", file=sys.stderr)
        sys.exit(1)

    profiles = load_profiles(args.dir)
    if not profiles:
        print(f"[错误] 未在 {args.dir} 中找到 *_profile.json 文件", file=sys.stderr)
        sys.exit(1)

    print(f"加载 {len(profiles)} 个视频的 profiling 数据")

    # 构建三张表
    table1 = ("表1 — 概述表",) + build_table1(profiles)
    table2 = ("表2 — 可视化分析表",) + build_table2(profiles)
    table3 = ("表3 — 特征提取分析表",) + build_table3(profiles)
    tables = [table1, table2, table3]

    # 终端输出
    for title, header, rows, avg_row in tables:
        print_table(title, header, rows, avg_row)

    # CSV 导出
    report_path = os.path.join(args.output, "analysis_report.csv")
    export_csv_tables(report_path, tables)


if __name__ == "__main__":
    main()

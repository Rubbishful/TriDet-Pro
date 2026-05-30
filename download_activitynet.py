"""
ActivityNet 200 全量视频下载脚本

逐类下载所有 200 个类别的原始视频，用于后续主体检测关联、端到端特征等开发。
视频存储在 D:\Code\ActivityNet\anet_video，由 FiftyOne/yt-dlp 从 YouTube 拉取。

用法:
    conda activate PatternRecognition
    python download_activitynet.py --test          # 测试：每类 1 个视频
    python download_activitynet.py                  # 正式：每类 5 个（默认）
    python download_activitynet.py --per-class 10   # 每类 10 个
    python download_activitynet.py --per-class 10 --max-gb 200
    python download_activitynet.py --test --start-class 50  # 从第 50 类继续

断点续传:
    中断后重新运行相同命令即可从断点继续（依赖 download_checkpoint.json）

要求:
    - Anaconda 虚拟环境 PatternRecognition，已安装 fiftyone
"""

import os
import sys
import time
import json
import logging
import argparse
import urllib.request
from pathlib import Path

import fiftyone as fo
import fiftyone.zoo as foz

# ---------------------------------------------------------------------------
# 全量 200 类别表（来自 ActivityNet 1.3 annotations）
# ---------------------------------------------------------------------------
ALL_CLASSES = [
    "Applying sunscreen", "Archery", "Arm wrestling", "Assembling bicycle",
    "BMX", "Baking cookies", "Ballet", "Bathing dog", "Baton twirling",
    "Beach soccer", "Beer pong", "Belly dance", "Blow-drying hair",
    "Blowing leaves", "Braiding hair", "Breakdancing", "Brushing hair",
    "Brushing teeth", "Building sandcastles", "Bullfighting", "Bungee jumping",
    "Calf roping", "Camel ride", "Canoeing", "Capoeira",
    "Carving jack-o-lanterns", "Changing car wheel", "Cheerleading",
    "Chopping wood", "Clean and jerk", "Cleaning shoes", "Cleaning sink",
    "Cleaning windows", "Clipping cat claws", "Cricket", "Croquet", "Cumbia",
    "Curling", "Cutting the grass", "Decorating the Christmas tree",
    "Disc dog", "Discus throw", "Dodgeball", "Doing a powerbomb",
    "Doing crunches", "Doing fencing", "Doing karate", "Doing kickboxing",
    "Doing motocross", "Doing nails", "Doing step aerobics", "Drinking beer",
    "Drinking coffee", "Drum corps", "Elliptical trainer", "Fixing bicycle",
    "Fixing the roof", "Fun sliding down", "Futsal", "Gargling mouthwash",
    "Getting a haircut", "Getting a piercing", "Getting a tattoo",
    "Grooming dog", "Grooming horse", "Hammer throw", "Hand car wash",
    "Hand washing clothes", "Hanging wallpaper", "Having an ice cream",
    "High jump", "Hitting a pinata", "Hopscotch", "Horseback riding",
    "Hula hoop", "Hurling", "Ice fishing", "Installing carpet",
    "Ironing clothes", "Javelin throw", "Kayaking", "Kite flying", "Kneeling",
    "Knitting", "Laying tile", "Layup drill in basketball", "Long jump",
    "Longboarding", "Making a cake", "Making a lemonade", "Making a sandwich",
    "Making an omelette", "Mixing drinks", "Mooping floor", "Mowing the lawn",
    "Paintball", "Painting", "Painting fence", "Painting furniture",
    "Peeling potatoes", "Ping-pong", "Plastering", "Plataform diving",
    "Playing accordion", "Playing badminton", "Playing bagpipes",
    "Playing beach volleyball", "Playing blackjack", "Playing congas",
    "Playing drums", "Playing field hockey", "Playing flauta",
    "Playing guitarra", "Playing harmonica", "Playing ice hockey",
    "Playing kickball", "Playing lacrosse", "Playing piano", "Playing polo",
    "Playing pool", "Playing racquetball", "Playing rubik cube",
    "Playing saxophone", "Playing squash", "Playing ten pins",
    "Playing violin", "Playing water polo", "Pole vault", "Polishing forniture",
    "Polishing shoes", "Powerbocking", "Preparing pasta", "Preparing salad",
    "Putting in contact lenses", "Putting on makeup", "Putting on shoes",
    "Rafting", "Raking leaves", "Removing curlers", "Removing ice from car",
    "Riding bumper cars", "River tubing", "Rock climbing",
    "Rock-paper-scissors", "Rollerblading", "Roof shingle removal",
    "Rope skipping", "Running a marathon", "Sailing", "Scuba diving",
    "Sharpening knives", "Shaving", "Shaving legs", "Shot put",
    "Shoveling snow", "Shuffleboard", "Skateboarding", "Skiing", "Slacklining",
    "Smoking a cigarette", "Smoking hookah", "Snatch", "Snow tubing",
    "Snowboarding", "Spinning", "Spread mulch", "Springboard diving",
    "Starting a campfire", "Sumo", "Surfing", "Swimming",
    "Swinging at the playground", "Table soccer", "Tai chi", "Tango",
    "Tennis serve with ball bouncing", "Throwing darts",
    "Trimming branches or hedges", "Triple jump", "Tug of war", "Tumbling",
    "Using parallel bars", "Using the balance beam", "Using the monkey bar",
    "Using the pommel horse", "Using the rowing machine", "Using uneven bars",
    "Vacuuming floor", "Volleyball", "Wakeboarding", "Walking the dog",
    "Washing dishes", "Washing face", "Washing hands", "Waterskiing",
    "Waxing skis", "Welding", "Windsurfing", "Wrapping presents", "Zumba",
]

# ---------------------------------------------------------------------------
# 默认配置
# ---------------------------------------------------------------------------
DATASET_NAME = "activitynet-200"
ZOO_DIR = r"D:\Code\ActivityNet\anet_video"
CHECKPOINT_FILE = os.path.join(ZOO_DIR, "download_checkpoint.json")
DEFAULT_SPLIT = "validation"
DEFAULT_PER_CLASS = 5
DEFAULT_MAX_GB = 200
DEFAULT_SEED = 42

# 单类下载重试
MAX_RETRIES_PER_CLASS = 3
RETRY_BASE_DELAY = 30

# FiftyOne 内置下载用 etaw.download_file，在某些网络环境下可能失败（空文件）。
# 这里直接用 urllib 预下载 labels 文件，避免 FiftyOne 内部解析空文件报错。
_LABELS_URL = (
    "https://github.com/activitynet/ActivityNet/raw/refs/heads/"
    "master/Evaluation/data/activity_net.v1-3.min.json"
)


def _ensure_labels():
    """确保 activitynet-100 和 activitynet-200 的 labels.json 存在且有效。"""
    for version in ("100", "200"):
        labels_dir = os.path.join(ZOO_DIR, f"activitynet-{version}")
        labels_path = os.path.join(labels_dir, "labels.json")
        # 检查是否有效 JSON（非空、可解析）
        valid = False
        if os.path.isfile(labels_path) and os.path.getsize(labels_path) > 0:
            try:
                with open(labels_path, "r") as f:
                    json.load(f)
                valid = True
            except (json.JSONDecodeError, ValueError):
                pass
        if not valid:
            os.makedirs(labels_dir, exist_ok=True)
            logger = logging.getLogger(__name__)
            logger.info(f"下载 labels: {labels_path}")
            urllib.request.urlretrieve(_LABELS_URL, labels_path)


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    return logging.getLogger(__name__)


def load_checkpoint():
    """加载断点续传记录，返回已完成的类别索引集合。"""
    if not os.path.exists(CHECKPOINT_FILE):
        return set()
    try:
        with open(CHECKPOINT_FILE, "r") as f:
            data = json.load(f)
        return set(data.get("completed_indices", []))
    except (json.JSONDecodeError, KeyError):
        return set()


def save_checkpoint(completed_indices, stats):
    """保存当前进度。"""
    os.makedirs(ZOO_DIR, exist_ok=True)
    with open(CHECKPOINT_FILE, "w") as f:
        json.dump({
            "completed_indices": sorted(completed_indices),
            "total_classes": len(ALL_CLASSES),
            "stats": stats,
        }, f, indent=2, ensure_ascii=False)


def _scan_video_files():
    """扫描 ZOO_DIR 下所有已下载的视频文件 (FiftyOne 可能去重到 activitynet-100)。"""
    result = []
    for root, dirs, files in os.walk(ZOO_DIR):
        for f in files:
            if f.endswith(".mp4"):
                result.append(os.path.join(root, f))
    return result


def compute_total_size_gb():
    """统计已下载视频的总大小 (GB)。"""
    total = 0
    for fpath in _scan_video_files():
        total += os.path.getsize(fpath)
    return total / (1024 ** 3)


def count_videos():
    """统计已下载视频数量。"""
    return len(_scan_video_files())


def try_download_one_class(class_name, per_class, args, logger):
    """下载单个类别的视频，返回 (success, video_count_added)。"""
    logger.info(f"  请求 {per_class} 个样本（shuffle=True）...")

    try:
        dataset = foz.load_zoo_dataset(
            DATASET_NAME,
            split=args.split,
            classes=[class_name],
            max_duration=args.max_duration,
            max_samples=per_class,
            shuffle=True,
            seed=args.seed,
            overwrite=True,  # 逐类强制重建，避免复用旧数据集跳过下载
        )
    except Exception as e:
        logger.error(f"  FiftyOne 下载异常: {e}")
        return False, 0

    if dataset is None:
        logger.warning("  返回 None dataset")
        return True, 0

    # 统计该类实际落盘的视频数
    count = 0
    for sample in dataset:
        if os.path.exists(sample.filepath):
            count += 1
    logger.info(f"  实际落盘 {count} 个视频")
    return True, count


def main():
    parser = argparse.ArgumentParser(
        description="下载 ActivityNet 200 全量视频到 D:\\Code\\ActivityNet\\anet_video",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
    python download_activitynet.py --test           # 测试：每类 1 个
    python download_activitynet.py                   # 正式：每类 5 个, 上限 200GB
    python download_activitynet.py --per-class 10    # 每类 10 个
    python download_activitynet.py --test --start-class 50  # 断点续传
        """,
    )
    parser.add_argument("--test", action="store_true",
                        help="测试模式：每类只下载 1 个视频")
    parser.add_argument("--per-class", type=int, default=None,
                        help=f"每类下载视频数 (默认: 测试=1, 正式={DEFAULT_PER_CLASS})")
    parser.add_argument("--split", default=DEFAULT_SPLIT,
                        choices=["training", "validation", "test"],
                        help="数据集 split (default: %(default)s)")
    parser.add_argument("--max-duration", type=int, default=None,
                        help="视频最大时长/秒 (default: 不限制)")
    parser.add_argument("--max-gb", type=int, default=DEFAULT_MAX_GB,
                        help="总大小上限 GB (default: %(default)s)")
    parser.add_argument("--start-class", type=int, default=0,
                        help="从第 N 个类别开始 (default: 0)")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED,
                        help="随机种子 (default: %(default)s)")
    args = parser.parse_args()

    # 确定 per_class
    if args.per_class is not None:
        per_class = args.per_class
    elif args.test:
        per_class = 1
    else:
        per_class = DEFAULT_PER_CLASS

    logger = setup_logging()

    # ---- 路径配置 ----
    os.makedirs(ZOO_DIR, exist_ok=True)
    _ensure_labels()
    fo.config.dataset_zoo_dir = ZOO_DIR
    fo.config.default_dataset_dir = ZOO_DIR

    # ---- 断点续传 ----
    completed = load_checkpoint()
    stats = {"total_size_gb": 0.0, "total_videos": 0, "per_class": {}}

    logger.info("=" * 60)
    logger.info(f"ActivityNet 200 全量下载")
    logger.info(f"  Zoo 目录: {ZOO_DIR}")
    logger.info(f"  Split: {args.split}")
    logger.info(f"  模式: {'测试 (1/类)' if args.test else f'正式 ({per_class}/类)'}")
    logger.info(f"  总大小上限: {args.max_gb} GB")
    logger.info(f"  总类别数: {len(ALL_CLASSES)}")
    logger.info(f"  已完成: {len(completed)} 类")
    logger.info("=" * 60)

    total_downloaded = 0
    total_skipped = 0
    total_failed = 0

    for idx, class_name in enumerate(ALL_CLASSES):
        # 跳过已完成的
        if idx in completed:
            total_skipped += 1
            continue

        # 支持从指定位置开始
        if idx < args.start_class and idx not in completed:
            continue

        current_size_gb = compute_total_size_gb()
        current_videos = count_videos()

        # 检查容量上限
        if not args.test and current_size_gb >= args.max_gb:
            logger.info(f"已达容量上限 {args.max_gb} GB (当前 {current_size_gb:.1f} GB)，停止下载")
            break

        logger.info(f"[{idx + 1}/{len(ALL_CLASSES)}] {class_name} "
                     f"| 已用空间: {current_size_gb:.1f} GB | 视频数: {current_videos}")

        # 重试下载
        success = False
        count = 0
        for attempt in range(1, MAX_RETRIES_PER_CLASS + 1):
            try:
                success, count = try_download_one_class(class_name, per_class, args, logger)
                if success:
                    break
            except Exception as e:
                logger.error(f"  异常 (第 {attempt}/{MAX_RETRIES_PER_CLASS} 次): {e}")
            if attempt < MAX_RETRIES_PER_CLASS:
                delay = RETRY_BASE_DELAY * attempt
                logger.info(f"  等待 {delay}s 后重试...")
                time.sleep(delay)

        if success:
            total_downloaded += 1
            completed.add(idx)
        else:
            total_failed += 1
            logger.warning(f"  {class_name} 下载失败，跳过")

        # 保存断点
        stats["total_size_gb"] = round(compute_total_size_gb(), 2)
        stats["total_videos"] = count_videos()
        stats["per_class"][class_name] = count
        save_checkpoint(completed, stats)

        # 下载间隔，避免被限流
        time.sleep(2)

    # ---- 汇总 ----
    final_size = compute_total_size_gb()
    final_videos = count_videos()
    logger.info("=" * 60)
    logger.info(f"下载完成")
    logger.info(f"  成功: {total_downloaded} 类")
    logger.info(f"  跳过(已完成): {total_skipped} 类")
    logger.info(f"  失败: {total_failed} 类")
    logger.info(f"  视频总数: {final_videos}")
    logger.info(f"  总大小: {final_size:.2f} GB")
    logger.info(f"  存储路径: {ZOO_DIR}")
    logger.info("=" * 60)

    if total_failed > 0:
        logger.warning("有部分类别下载失败，可重新运行脚本重试")
        sys.exit(1)


if __name__ == "__main__":
    main()

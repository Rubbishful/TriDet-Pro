#!/usr/bin/env python
"""TriDet 测试运行工具 — 自动发现并执行所有测试脚本。

Usage:
    python run_tests.py              # 运行全部测试
    python run_tests.py --verbose    # 显示详细输出
    python run_tests.py --list       # 只列出测试文件
    python run_tests.py -k smoke     # 只运行名称匹配 "smoke" 的测试
"""
import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent
TESTS_DIR = ROOT / "tests"


def discover_tests():
    """发现 tests/test_*.py 并按名称排序。"""
    if not TESTS_DIR.is_dir():
        return []
    return sorted(TESTS_DIR.glob("test_*.py"))


def run_one(test_path, verbose=False, timeout=300):
    """执行单个测试文件，隔离到子进程中。

    Returns:
        (name, passed, stdout, stderr, elapsed)
    """
    name = test_path.stem
    t0 = time.perf_counter()
    try:
        result = subprocess.run(
            [sys.executable, str(test_path)],
            capture_output=True, text=True,
            cwd=str(ROOT),
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        elapsed = time.perf_counter() - t0
        return name, False, "", f"TIMEOUT after {timeout}s", elapsed

    elapsed = time.perf_counter() - t0
    passed = result.returncode == 0
    stdout = result.stdout.strip()
    stderr = result.stderr.strip()

    return name, passed, stdout, stderr, elapsed


def main():
    parser = argparse.ArgumentParser(description="TriDet 测试运行工具")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="显示每个测试的完整输出")
    parser.add_argument("--list", "-l", action="store_true",
                        help="只列出测试文件，不执行")
    parser.add_argument("-k", type=str, default=None,
                        help="只运行名称包含指定字符串的测试 (如 -k smoke)")
    parser.add_argument("--timeout", type=int, default=300,
                        help="每个测试的超时秒数 (默认 300)")
    args = parser.parse_args()

    test_files = discover_tests()

    if not test_files:
        print("[ERROR] tests/ 目录下未找到 test_*.py 文件")
        sys.exit(1)

    # 名称过滤
    if args.k:
        test_files = [f for f in test_files if args.k.lower() in f.stem.lower()]
        if not test_files:
            print(f"[ERROR] 未找到名称匹配 '{args.k}' 的测试文件")
            sys.exit(1)

    print(f"Found {len(test_files)} test file(s)\n")

    if args.list:
        for tf in test_files:
            print(f"  {tf.relative_to(ROOT)}")
        return

    passed, failed, total_elapsed = 0, [], 0.0

    for tf in test_files:
        name = tf.stem
        header = f"{'=' * 60}\n  RUN  {name}\n{'=' * 60}"
        print(header)

        name, ok, stdout, stderr, elapsed = run_one(tf, verbose=args.verbose,
                                                     timeout=args.timeout)
        total_elapsed += elapsed

        if stdout and (args.verbose or not ok):
            print(stdout)
        if stderr and not ok:
            print(f"\n  STDERR:\n{stderr}")

        if ok:
            print(f"\n  PASS  {name}  ({elapsed:.1f}s)")
            passed += 1
        else:
            print(f"\n  FAIL  {name}  ({elapsed:.1f}s)")
            failed.append(name)

    print(f"\n{'=' * 60}")
    print(f"  RESULT: {passed}/{len(test_files)} passed  ({total_elapsed:.1f}s total)")
    if failed:
        print(f"  FAILED: {', '.join(failed)}")
    print(f"{'=' * 60}")

    sys.exit(0 if not failed else 1)


if __name__ == "__main__":
    main()

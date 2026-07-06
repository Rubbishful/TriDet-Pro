"""
临时串联脚本 — 连续运行 Section 2-4 和 Section 5 迭代实验
(网格搜索已运行完毕)

用法:
    python evaluate/run_sections_2_to_5.py
"""

import subprocess, sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
PYTHON = sys.executable

SCRIPTS = [
    'evaluate/run_section_2_4_structural.py',
    'evaluate/run_section_5_iou_head.py',
]

for script in SCRIPTS:
    print(f"\n{'='*70}")
    print(f"  运行: {script}")
    print(f"{'='*70}")
    r = subprocess.run([PYTHON, '-u', str(REPO / script)], cwd=str(REPO))
    if r.returncode != 0:
        print(f"\n[FATAL] {script} 失败 (rc={r.returncode}), 终止。")
        sys.exit(r.returncode)

print("\n全部完成!")

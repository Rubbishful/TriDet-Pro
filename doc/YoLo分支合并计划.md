Now I have all the evidence needed. Let me present the detailed merge plan.

---

# 合并整理与验证计划 — `YoLo_premerge` 分支

## 前置状态

`YoLo_premerge` 已存在且与 `origin/dev-DengWenshuo` 完全相同（commit `3d1f101`）。当前工作区干净。代码已合并，核心工作是**整理、修复、去重、验证**。

---

## 1. 死代码清理（YoLo/ 目录及冗余文件）

- **删除 `YoLo/libs/` 整个子目录**（`__init__.py`、`subject/__init__.py`、`subject/detector.py`）——这是旧版 `SubjectDetector`，永远不会被导入（`test_detect.py` 从 `libs.subject.detector` 导入，而非从 `YoLo.libs`），且功能上 `libs/subject/detector.py` 是更完善的版本。
- **移动 `YoLo/test_detect.py`** 到项目根目录或 `tools/` 目录，因为它是独立的演示脚本，放在 `YoLo/` 下会让人误以为它是 `YoLo` 包的一部分。
- **删除 `YoLo/` 空目录**（移动 `test_detect.py` 后它将是空的）。
- **删除空文件 `test2 .md`**（文件名含空格，大小为 0 字节）。
- **评估 `test.md`** 是否还有用（103 字节），若为临时笔记则删除。
- **删除 `node_modules/`、`package.json`、`package-lock.json`**——Python 项目中不应有 Node.js 依赖残留，或将其加入 `.gitignore`（当前已在 untracked 中）。

---

## 2. 严重 Bug 修复（会导致运行时崩溃）

- **`libs/core/config.py:148-154` — `_merge` 函数无法覆盖已存在的标量值**：当 YAML 中的 key 在 DEFAULTS 中已存在且值不是 dict 时，`_merge` 只递归处理子 dict，却**静默忽略**所有标量/列表覆盖。这意味着 YAML 配置文件中的标量参数（如 `batch_size`、`learning_rate`、`num_classes` 等）覆盖完全无效。修复：添加 `else: dst[k] = v` 分支处理非 dict 覆盖。
- **`libs/utils/train_utils.py:189` — `schedule_config` 未定义**：在 warmup + multistep 调度器分支中，`schedule_config["gamma"]` 应为 `optimizer_config["schedule_gamma"]`。当前会抛出 `NameError`。
- **`libs/datasets/hacs.py:149` — `self.file_prefix` 未定义**：`HacsDataset.__init__` 从未设置 `self.file_prefix`，但在 `backbone_type == 'i3d'` 时使用它。会导致 `AttributeError`。修复：在 `__init__` 中添加 `self.file_prefix = ''` 或从参数传入。
- **`requirements.txt` — `opencv-python==4.13.0` 版本不存在**：截至 2025 年，最新 opencv-python 版本约 4.10.x，4.13.0 不存在。应改为实际存在的版本（如 `>=4.10.0`）。
- **`requirements.txt` — 缺少 `torchvision` 和 `fiftyone`**：README 指示单独安装，但应为可选依赖或明确标注。

---

## 3. 硬编码绝对路径修复（Windows 路径，跨平台不可用）

- **`evaluate/analyze.py`、`do_all.py`、`run_ablation.py`、`run_all.py`、`run_training.py`、`train_all.py`**：将所有 `REPO = 'e:/Tridet/TriDet-Pro'` 替换为相对于脚本位置的路径计算（如 `Path(__file__).resolve().parent.parent`）。
- **`evaluate/analyze.py`、`do_all.py`、`run_ablation.py`、`run_all.py`、`run_training.py`、`train_all.py`**：将 `PYTHON = 'E:/anaconda/envs/test/python.exe'` 替换为 `sys.executable`。
- **`evaluate/confusion_analysis.py`、`duration_analysis.py`、`density_analysis.py`、`eval_fast.py`、`run_full_analysis.py`**：将 `JSON_FILE = 'E:/thumos/annotations/thumos14.json'` 和 `SCORE_FILE` 默认值改为 `None`，并将在缺失时提示用户通过 CLI 参数提供。
- **`visualize.py:36-38`**：将 `ANNOTATION_FILE`、`VIDEO_DIR`、`FEATURE_DIR` 从硬编码路径改为 CLI 参数（使用 `argparse`，保留当前硬编码值作为 default）。
- **`download_activitynet.py:97`**：将 `ZOO_DIR` 从 `r"D:\Code\ActivityNet\anet_video"` 改为 CLI 参数。
- **`tools/*.sh`**：将硬编码的 epoch 编号（`epoch_014.pth.tar` 等）替换为 `$(ls ckpt/*/epoch_*.pth.tar | sort -V | tail -1)` 以自动选取最新检查点；添加 `set -e` 和 `$1` 参数验证。

---

## 4. evaluate/ 目录大规模去重整合

当前有 **6 个脚本**独立重复维护同一套消融实验定义（A1-A8），3 个脚本各自实现完全相同的 duration/density/confusion 分析逻辑，5 个脚本各自定义相同的 `deep_set`/`dump_config`/`parse_map`/`run_cmd` 辅助函数。

- **创建 `evaluate/experiments.py`**：抽取公共的消融实验定义（`ABLATION_SPEC`、`EXPERIMENTS`）和共享辅助函数（`deep_set`、`dump_config`、`parse_map`、`run_cmd`），从此集中维护。
- **重构 `do_all.py`、`run_all.py`、`run_ablation.py`、`run_training.py`、`train_all.py`**：全部改为从 `evaluate.experiments` 导入实验定义和辅助函数，消除 5 份重复代码。
- **保留 `confusion_analysis.py`、`duration_analysis.py`、`density_analysis.py`** 作为标准独立分析脚本（它们有正确的 `argparse` CLI 和 `if __name__` 防护）。
- **废弃 `analyze_results.py`**（无 `if __name__` 防护，导入即运行）和 **`run_full_analysis.py`**（复制了独立分析脚本的逻辑），将它们的功能标记为已由独立脚本替代，或改为调用独立脚本的编排器。
- **废弃或合并 `eval_fast.py`**：它似乎是已废弃的原型（定义了 `PYTHON` 却从未使用，未被任何编排脚本调用）。若仍有用则整合到 `eval.py`。
- **修复实验 ID 命名不一致**：`do_all.py` 和 `run_training.py` 使用 `'A3_pl'`，而 `analyze.py` 期望 `'A3_per_layer'`——统一为后者。

---

## 5. 代码风格与质量问题修复

- **无用 import 清理**：
  - `libs/modeling/models.py` — 删除 `import os`
  - `libs/datasets/datasets.py` — 删除 `import os`
  - `libs/utils/postprocessing.py` — 删除 `import shutil, time, torch` 和 `from .metrics import ANETdetection`
  - `libs/datasets/data_utils.py` — 删除重复的 `import random`（第 1 行和第 5 行）
  - `test.py` — 删除 `import numpy`
- **多 import 同行拆分**（PEP8 E401）：`train_utils.py`、`postprocessing.py`、`metrics.py`、`nms.py`、`lr_schedulers.py`、`data_utils.py` 中将 `import os, sys` 等拆为每行一个。
- **`crop_ratio == None` → `crop_ratio is None`**：修复 `data_utils.py`、`anet.py`、`thumos14.py`、`epic_kitchens.py`、`hacs.py` 中的 PEP8 违规。
- **拼写错误 `boudary_kernel_size` → `boundary_kernel_size`**：在 `config.py:68` 和 `meta_archs.py:213` 两处修复（需同步更新引用此 key 的 YAML 配置文件）。
- **调试残留清理**：
  - `metrics.py:334` — 删除 `print(1)`
  - `nms.py:134` — 删除未使用的 `valid_inds` 变量
- **`train.py`**：`os.mkdir` → `os.makedirs(..., exist_ok=True)`；修复重复的 `"""4."""` 注释编号。
- **`eval.py:106`**：argparse description 从 "Train" 改为 "Evaluate"。
- **`test.py:91-94`**：AMP argparse 逻辑修正（`default=True` + `store_true` 无效组合）。
- **`test.py:119-122`**：删除重复的 `cfg["model"]` 字段同步（`_update_config` 已处理）。
- **`hacs.py:77`**：删除注释掉的死代码 `tiou_thresholds`。
- **`weight_init.py`**：评估是否可用 `torch.nn.init.trunc_normal_`（PyTorch 1.10+ 内置）替换自实现的版本。

---

## 6. 数据集文件性能修复

- **`anet.py:129`、`thumos14.py:131`、`epic_kitchens.py:136`、`hacs.py:129-134`**：将 `dict_db += ({...}, )` 的 O(n²) 元组拼接改为 `list.append` + 最后转换。
- **`hacs.py:150,154,159`**：`np.load` 改为使用 context manager（`with np.load(...) as data:`）以正确关闭文件句柄。

---

## 7. 配置与文档整理

- **`requirements.txt`**：补全 `torchvision`、`fiftyone`（标记为可选）；修正 `opencv-python` 版本号；将 `numpy==1.21.2` 放宽为 `numpy>=1.21,<2.0`。
- **`.gitignore`**：添加 `node_modules/`、`package.json`、`package-lock.json`、`.DS_Store`。
- **`README.md`**：修复 Windows 路径引用；统一文档中 `分工与开发规划.md` 的路径引用。
- **`test.md`**：评估内容是否有效，无效则删除。

---

## 8. 最终验证

- 运行 `python test.py`（8GB VRAM 冒烟测试）确认训练+推理管道可用。
- 运行 `python -c "from libs.core.config import load_config; cfg = load_config('configs/thumos_i3d.yaml'); print(cfg['dataset']['num_classes'])"` 验证 YAML 覆盖在修复后生效。
- 运行 `python -c "from libs.datasets.hacs import HacsDataset"` 验证 hacs 导入不报错。
- 运行 `python -m pytest` 或手动执行 `evaluate/test_modules.py` 确认模块测试通过。
- 用 `git diff --stat` 确认变更范围合理。

---

> **总结**：当前代码库已合并 `dev-DengWenshuo` 的工作，但存在 **3 个运行时 Bug**（config 合并失效、scheduler NameError、hacs AttributeError）、**大量硬编码 Windows 路径**（跨平台不可用）、**evaluate/ 目录中共 ~2000 行重复代码**、以及多处代码风格违规。上述计划按严重性降序排列，每层独立可验证。

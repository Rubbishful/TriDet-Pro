# TriDet — 时序动作检测课程设计

基于 CVPR 2023 [TriDet](https://arxiv.org/abs/2303.07347) 论文的时序动作检测（Temporal Action Detection）复现与改进项目。

原始论文仓库: [dingfengshi/TriDet](https://github.com/dingfengshi/TriDet) | 上游 README: [README_upstream.md](README_upstream.md)

## 项目结构

```
Tridet/
├── train.py / eval.py          # 训练 / 评估入口
├── download_activitynet.py     # ActivityNet 原始视频下载（FiftyOne）
├── configs/                    # 各数据集配置文件
│   ├── thumos_i3d.yaml         # THUMOS14（I3D, 20 类）
│   └── anet_tsp.yaml           # ActivityNet（TSP, 200 类）
├── libs/
│   ├── modeling/               # TriDet 模型实现
│   │   ├── backbones.py        # SGP Backbone
│   │   ├── blocks.py           # 基础模块
│   │   ├── meta_archs.py       # 主模型（训练/推理/后处理）
│   │   ├── necks.py            # FPN
│   │   └── losses.py           # 分类/回归损失
│   ├── datasets/               # 数据集加载
│   └── utils/                  # NMS、评估指标、训练工具
├── tools/                      # 一键训练+评估脚本
├── doc/                        # 项目文档
│   └── 开发规划V2.1.md          # 当前开发规划
├── log/                        # 训练/评估日志
├── ckpt/                       # 模型权重（不纳入 Git）
└── analysis/                   # 错误分析模块（规划中）
```

## 当前进度

### Phase 1: 基线复现 ✅

| 数据集 | mAP（复现） | 论文 mAP | 状态 |
|--------|------------|----------|------|
| THUMOS14 | **68.59%** | 69.27% | ✅ 成功复现 |
| ActivityNet | **36.54%** | ~36.5% | ✅ 成功复现 |

- THUMOS14: I3D 特征 (2048-dim), 20 类, tIoU=0.3:0.1:0.7
- ActivityNet: TSP 特征 (512-dim), 200 类, tIoU=0.5:0.05:0.95

### Phase 2: 改进与分析（进行中）

详见 [doc/开发规划V2.1.md](doc/开发规划V2.1.md)

- 多实例 / 重叠动作分析
- 主体检测关联
- 端到端特征提取

## 环境配置

### 1. 安装 Anaconda

Anaconda 是 Python 虚拟环境管理工具，可以为每个项目创建独立的 Python 环境，避免依赖冲突。

- 下载: [Anaconda 官网](https://www.anaconda.com/download) 或 [清华镜像](https://mirrors.tuna.tsinghua.edu.cn/anaconda/archive/)
- 安装后验证: 终端输入 `conda --version`
- 常用命令:
  ```bash
  conda create -n 环境名 python=3.9    # 创建环境
  conda activate 环境名                 # 激活环境
  conda deactivate                     # 退出环境
  conda info --envs                    # 查看所有环境
  conda install 包名                   # 在当前环境安装包
  ```

### 2. 创建项目环境

```bash
# 创建环境（Python 3.9）
conda create -n PatternRecognition python=3.9 -y

# 激活环境
conda activate PatternRecognition

# 安装 PyTorch（CUDA 11.8 版本，可根据 GPU 调整）
pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu118

# 安装项目依赖
cd d:/Code/Tridet
pip install -r requirements.txt

# 安装 FiftyOne（用于 ActivityNet 视频下载）
pip install fiftyone

# 编译 C 扩展 NMS（需要 Visual Studio Build Tools）
cd libs/utils
python setup.py build_ext --inplace
cd ../..
```

### 3. C 扩展编译要求

NMS C 扩展 (`nms_1d_cpu`) 编译需要:
- Visual Studio 2015+ (MSVC 编译器)
- 编译命令: `cd libs/utils && python setup.py build_ext --inplace`

如果缺少运行库和编译工具很可能会编译失败。
如果编译失败，系统会自动回退到纯 Python 实现，但速度会慢上百倍，正常情况下评估脚本运行时间短则几十秒，长则也就几分钟，而纯python运行几个小时都跑不完。

到这个网址下载Visual Studio Installer
https://visualstudio.microsoft.com/zh-hans/vs/older-downloads/
下载安装后打开选择安装Visual Studio生成工具，注意安装工作负荷，选择使用C++的桌面开发，除默认选项外注意勾选倒数第三个MSVC v143，这是核心编译工具。

### 4. 数据集准备

- **特征文件** (`.npy`): 从 [ActionFormer 仓库](https://github.com/happyharrycn/actionformer_release) 下载，放到 `./data/` 目录
- **原始视频**: 运行 `python download_activitynet.py --test` 下载到 `D:\Code\ActivityNet\anet_video`

## Git 协作规范

### 基本流程

```bash
# 1. 拉取最新代码
git pull origin master

# 2. 查看状态
git status

# 3. 添加改动
git add <文件>

# 4. 提交
git commit -m "类型: 简述"

# 5. 推送
git push origin master
```

### Commit 消息规范

采用简洁的中文/英文格式:

```
类型: 简述改动内容
```

**类型**:
- `feat` — 新功能
- `fix` — Bug 修复
- `docs` — 文档变更
- `refactor` — 代码重构
- `chore` — 杂项（依赖更新、配置调整等）

**示例**:
```
feat: 添加重叠动作检测分析模块
fix: C扩展nms_1d_cpu导入失败
docs: 更新 README 项目结构说明
```

### 注意事项

- **Push 前先 pull**，避免冲突
- **不要提交大文件**: `ckpt/`、`*.pth.tar`、`*.pkl`、`__pycache__/` 已在 `.gitignore`
- 发现有冲突时，与组员沟通协调解决
- 当前采用 **单分支 master 协作**，不强制 feature branch 流程

## AI Agent 开发环境配置

本项目推荐使用 **VSCode + Claude Code 插件 + DeepSeek API** 搭建 AI 辅助开发环境。

### 为什么用 Claude Code？

Claude Code 是 Anthropic 推出的 AI 编程助手，可以:
- 理解和修改项目代码
- 执行终端命令（训练、评估等）
- 搜索和分析代码库
- 生成文档和报告

本项目的开发（包括基线复现、代码修改、文档编写）全程在 Claude Code 协助下完成。

### 配置步骤

**1. 安装 VSCode 扩展**

在 VSCode 扩展商店搜索 `Claude Code` 并安装。

**2. 获取 DeepSeek API Key**

- 注册 [DeepSeek 开放平台](https://platform.deepseek.com/)
- 在 API Keys 页面创建 API Key
- 充值（DeepSeek 价格较低，个人开发通常每月几十元）

**3. 配置 Claude Code**

打开 VSCode 设置 (`settings.json`)，添加:

```json
{
  "claudeCode.anthropicBaseUrl": "https://api.deepseek.com/anthropic",
  "claudeCode.primaryApiKey": "你的DeepSeek-API-Key",
  "claudeCode.model": "deepseek-v4-pro"
}
```

**4. 重启 VSCode**

配置完成后重启 VSCode，侧边栏会出现 Claude Code 图标。在输入框中即可与 AI Agent 对话。

### 使用示例

在 Claude Code 对话框中:
- `跑一下 THUMOS14 的评估` — AI 会自动执行评估命令
- `这段代码有什么问题` (选中代码) — AI 会分析并给出修改建议
- `帮我在 analysis/ 下新建错误分析脚本` — AI 会创建完整的分析脚本

### 配置视频教程

B站详细配置演示: [BV1ia9UBPESQ](https://www.bilibili.com/video/BV1ia9UBPESQ/)

## 快速开始

### 训练

```bash
conda activate PatternRecognition

# THUMOS14
python train.py ./configs/thumos_i3d.yaml --output thumos_baseline

# ActivityNet (需要先准备好 TSP 特征)
python train.py ./configs/anet_tsp.yaml --output anet_baseline
```

### 评估

```bash
# THUMOS14
python eval.py ./configs/thumos_i3d.yaml ./ckpt/thumos_baseline/

# ActivityNet
python eval.py ./configs/anet_tsp.yaml ./ckpt/anet_baseline/
```

### 下载原始视频

```bash
# 测试模式: 每类下载 1 个视频
python download_activitynet.py --test

# 正式下载: 每类 5 个
python download_activitynet.py --per-class 5
```

### 可视化：GT vs 预测对比

`visualize.py` 根据 YouTube 视频 ID 自动匹配原始视频、预提取特征和标注，生成时间轴对比图。

```bash
# 1. 先生成预测结果 (pkl)
python eval.py ./configs/anet_tsp.yaml ./ckpt/anet_tsp_baseline/ --saveonly

# 2. 随机采样 10 个视频生成对比图
python visualize.py --pkl ./ckpt/anet_tsp_baseline/eval_results.pkl --num-samples 10

# 3. 指定单个视频
python visualize.py --pkl ./ckpt/anet_tsp_baseline/eval_results.pkl --video-id sJFgo9H6zNo
```

输出 PNG 保存在 `./vis_output/` 下，上排为视频截图（等距抽取 5 帧），下排为时间轴：
- **绿色条** — Ground Truth 标注段
- **红色条** — 模型预测段（透明度随置信度变化，标注分数）

三者映射关系: 标注 key (youtube_id) → 特征 `v_{id}.npy` → 视频 `v_{id}.mp4`

## 参考

- TriDet 论文: [arXiv 2303.07347](https://arxiv.org/abs/2303.07347)
- 原始代码: [dingfengshi/TriDet](https://github.com/dingfengshi/TriDet)
- ActionFormer: [happyharrycn/actionformer_release](https://github.com/happyharrycn/actionformer_release)

# 全局改进实验配置与数据总览

> 提取自论文第五章「全局改进探索」。所有实验基于 TriDet 基线，使用预提取 I3D 特征（THUMOS14），训练 20 epochs（论文原始为 40 epochs），GPU 为 RTX 4060 Laptop (8GB)。

## 一、基线复现

| 数据集 | 特征类型 | 本文 avg mAP | 论文 avg mAP | 差异 | 训练轮次 |
|--------|----------|-------------|-------------|------|----------|
| THUMOS14 | I3D | 68.59% | 69.27% | −0.68% | 20 vs 40 |
| ActivityNet | TSP | 36.54% | ∼36.5% | — | 20 |

## 二、通道注意力（SE 模块）

| 实验 | 插入位置 | 压缩比 r | avg mAP | vs 基线 | 训练轮次 |
|------|----------|----------|---------|---------|----------|
| 基线 | — | — | 68.59% | — | 20 |
| SE-1 | pre_fusion（LayerNorm 后） | 16 | 68.34% | −0.25% | 20 |
| SE-2 | fusion（残差后） | 16 | 67.26% | −1.33% | 20 |
| SE-3 | pre_fusion（LayerNorm 后） | 4 | 66.77% | −1.82% | 20 |

**固定条件**：THUMOS14 I3D，Backbone 各 stage 后插入 SE，SGP Block 五分支结构不变。

## 三、BiFPN 特征金字塔增强

| 实验 | Neck 类型 | 融合方式 | avg mAP | vs 基线 | 训练轮次 |
|------|----------|----------|---------|---------|----------|
| 基线 | Identity Neck | — | 68.59% | — | 20 |
| BiFPN-1 | BiFPN | sum | 62.49% | −6.10% | 20 |
| BiFPN-2 | BiFPN | fast_norm | 62.69% | −5.90% | 20 |

**固定条件**：THUMOS14 I3D，fpn_type='bifpn'，SGP Backbone 多 stage 下采样（scale_factor=2, 3 stage）。

## 四、回归损失函数增强

| 实验 | 损失函数 | 关键参数 | avg mAP | vs 基线 | 训练轮次 |
|------|----------|----------|---------|---------|----------|
| 基线 | DIoU | — | 68.59% | — | 20 |
| RL-1 | Focaler-DIoU | d=0, u=0.95 | 67.40% | −1.19% | 20 |
| RL-2 | EIoU | — | 67.04% | −1.55% | 20 |
| RL-3 | Alpha-DIoU | α=3.0 | 66.09% | −2.50% | 20 |

**固定条件**：THUMOS14 I3D，一维 (l,r) 偏移量参数化，iou_weight_power=0.2。

## 五、置信度-IoU 联合增强（IoU Head）

### 5.1 迭代过程

| Round | 方案 | 关键配置 | avg mAP | vs 基线 | 训练轮次 |
|-------|------|----------|---------|---------|----------|
| 基线 | — | — | 68.59% | — | 20 |
| R1 | BCE all-pos + TAL | IoU Head 4层, TAL(α=1.0,β=4.0) | 52.78% | −15.81% | 20 |
| R2 | BCE pos-only | 关闭TAL, prior_prob | 66.90% | −1.69% | 20 |
| R3 | QFL + warmup | β=2.0, warmup 5 epochs | 60.89% | −7.70% | 20 |

**固定条件**：THUMOS14 I3D，MaskedConv1D IoU Head，推理时 confidence = σ(cls)·σ(iou)。

### 5.2 超参数网格搜索（Round 4, 36 组合）

| 排名 | loss_weight | per_level | residual | layers | mAP |
|------|-------------|-----------|----------|--------|-----|
| 1 | 0.10 | False | True | 2 | 56.34% |
| 2 | 0.10 | False | False | 4 | 55.84% |
| 3 | 0.10 | False | True | 3 | 55.29% |
| 4 | 0.10 | False | False | 2 | 55.28% |
| 5 | 0.10 | True | True | 2 | 54.87% |
| 6 | 0.10 | False | True | 4 | 54.33% |
| 7 | 0.10 | False | False | 3 | 54.28% |
| 8 | 0.25 | False | False | 4 | 54.28% |
| 9 | 0.10 | True | True | 4 | 54.26% |
| 10 | 0.25 | False | True | 4 | 54.09% |
| 11 | 0.25 | False | True | 3 | 53.89% |
| 12 | 0.10 | True | True | 3 | 53.46% |
| 13 | 0.25 | True | True | 4 | 53.44% |
| 14 | 0.10 | True | False | 2 | 53.36% |
| 15 | 0.50 | False | True | 3 | 53.14% |
| 16 | 0.25 | False | True | 2 | 52.90% |
| 17 | 0.25 | False | False | 3 | 52.85% |
| 18 | 0.10 | True | False | 3 | 52.56% |
| 19 | 0.25 | True | True | 2 | 52.48% |
| 20 | 0.50 | False | False | 4 | 52.23% |
| 21 | 0.10 | True | False | 4 | 52.15% |
| 22 | 0.50 | False | False | 3 | 51.54% |
| 23 | 0.25 | True | True | 3 | 51.51% |
| 24 | 0.25 | True | False | 4 | 51.45% |
| 25 | 0.50 | True | True | 3 | 51.45% |
| 26 | 0.50 | True | True | 2 | 51.38% |
| 27 | 0.25 | False | False | 2 | 51.24% |
| 28 | 0.50 | False | False | 2 | 51.11% |
| 29 | 0.25 | True | False | 2 | 51.06% |
| 30 | 0.50 | True | False | 3 | 50.96% |
| 31 | 0.25 | True | False | 3 | 50.96% |
| 32 | 0.50 | False | True | 2 | 50.65% |
| 33 | 0.50 | False | True | 4 | 50.55% |
| 34 | 0.50 | True | False | 2 | 50.51% |
| 35 | 0.50 | True | False | 4 | 49.73% |
| 36 | 0.50 | True | True | 4 | 49.12% |

**固定条件**：THUMOS14 I3D，BCE pos-only，每个 trial 训练 80 epochs，90%/10% train/val split，以代理 mAP 评判。搜索空间：loss_weight∈{0.1,0.25,0.5} × per_level∈{F,T} × residual∈{F,T} × layers∈{2,3,4}。

## 六、全局汇总（按 mAP 降序）

| 排名 | 实验 | avg mAP | vs 基线 | 结论 |
|------|------|---------|---------|------|
| — | **基线** | **68.59%** | — | 复现成功 |
| 1 | SE（pre_fusion, r=16） | 68.34% | −0.25% | 放弃 |
| 2 | SE（fusion, r=16） | 67.26% | −1.33% | 放弃 |
| 3 | Focaler-DIoU | 67.40% | −1.19% | 不推荐 |
| 4 | EIoU | 67.04% | −1.55% | 不推荐 |
| 5 | IoU Head + BCE（Round 2） | 66.90% | −1.69% | 不推荐 |
| 6 | SE（pre_fusion, r=4） | 66.77% | −1.82% | 放弃 |
| 7 | Alpha-DIoU（α=3） | 66.09% | −2.50% | 不推荐 |
| 8 | BiFPN（fast_norm） | 62.69% | −5.90% | 放弃 |
| 9 | BiFPN（sum） | 62.49% | −6.10% | 放弃 |
| 10 | IoU Head + QFL（Round 3） | 60.89% | −7.70% | 放弃 |
| 11 | IoU Head（Round 1） | 52.78% | −15.81% | 放弃 |

## 通用实验条件

| 条件 | 取值 |
|------|------|
| 模型基线 | TriDet (CVPR 2023) |
| 数据集 | THUMOS14（全部改进实验） |
| 特征类型 | I3D 预提取特征（双流 RGB+Flow，拼接后 2048 维） |
| 训练轮次 | 20 epochs（除 Round 4 为 80 epochs） |
| GPU | NVIDIA GeForce RTX 4060 Laptop (8GB) |
| 优化器 | AdamW |
| 评估指标 | avg mAP @ tIoU ∈ {0.3, 0.4, 0.5, 0.6, 0.7} |
| 说明 | 论文原始训练 40 epochs，本文因算力限制统一使用 20 epochs，基线差距 0.68% 在合理范围内 |

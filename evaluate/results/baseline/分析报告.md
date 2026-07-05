# TriDet 测试与分析报告

**生成日期**: 2026-07-04 | **实验分支**: Baseline

---

## 一、实验设置

| 项目 | 值 |
|------|----|
| 数据集 | THUMOS14 (validation 训练, test 评估) |
| GPU | NVIDIA GeForce RTX 4060 Laptop (8GB) |
| 特征 | I3D 预提取 (C=2048, feat_stride=4) |
| 评估策略 | Score Fusion + Hard NMS (统一) |

## 二、模块 Shape 测试

**95/95 项全部通过**

验证了从输入 `(B, 2048, T)` 到输出 `(segments, scores, labels)` 的完整 shape 链，覆盖 15 个模块：
MaskedConv1D, LayerNorm, ConvBlock, SGPBlock, SGPBackbone, ConvBackbone,
FPN1D, FPNIdentity, PointGenerator, ClsHead, RegHead (Trident & Plain)

## 三、消融实验

| 实验 | 描述 | mAP@0.3 | mAP@0.5 | mAP@0.7 | avg_mAP |
|------|------|---------|---------|---------|---------|
| 基线 | SGP骨干 + Trident-head (完整版) | 75.1% | 62.6% | 38.3% | 59.31% |
| A1 | SGP骨干 + 普通回归头 (无 Trident) | 74.8% | 62.1% | 36.4% | 58.38% |
| A2 | Conv骨干 + Trident-head (无 SGP)* | 61.7% | 47.8% | 21.7% | 44.46% |

\* A2 = ConvBackbone + Trident-head, 仅训练 20/40 epochs, 结果为下界估计

### 组件贡献量化

| 组件 | 计算公式 | avg_mAP 贡献 | 结论 |
|------|---------|-------------|------|
| **Trident-head** | 基线 - A1 | **+0.93%** | 边界精度小幅提升，尤其在 tIoU>=0.7 时 |
| **SGP backbone** | 基线 - A2 | **+14.85%** | TriDet 性能的绝对核心支撑 |

### 关键发现

1. **SGP 是最核心的组件**: 移除 SGP 导致 mAP 下降 14.9%。5分支多尺度全局感知 (psi+fc+convw+convkw+global_fc) 是 TriDet 区别于普通 TAD 方法的根基。

2. **Trident-head 提供边界精度增益**: 移除 Trident-head 后 avg_mAP 下降 0.9%，主要体现在高 tIoU 阈值（mAP@0.7: 38.3% → 36.4%），说明离散化边界回归改善了精确定位能力。

3. **两组件协同**: Trident-head 的连续回归离散化 + SGP 的多尺度特征提取 + DIoU 损失的中心距离惩罚，三者协同构成 TriDet 的完整检测能力。

## 四、时长分层分析

| 时长分组 | GT 实例数 | Recall @ tIoU=0.5 |
|----------|-----------|-------------------|
| Short (<2s) | 1037 | 67.1% |
| Medium (2-10s) | 2052 | 80.3% |
| Long (>10s) | 269 | 84.0% |

中等时长动作 recall 最高 (80.3%)，短动作因边界模糊难以精确定位 (67.1%)。

## 五、混淆矩阵

Top 误分类对主要集中在视觉/运动模式相似的类别之间：
- Diving -> CliffDiving (35x) — 两类高度重叠，THUMOS 中 CliffDiving 是 Diving 的子集
- CricketShot <-> CricketBowling (18x, 13x) — 板球类动作相似
- CricketShot -> FrisbeeCatch (16x) — 手臂投掷模式相似

## 六、分析图表

| 图表 | 路径 |
|------|------|
| 时长分层 | evaluate/results/duration_analysis.png |
| 密度分布 | evaluate/results/density_hist.png |
| 混淆矩阵 | evaluate/results/confusion_matrix.png |
| 消融对比 | evaluate/results/ablation_comparison.png |
| mAP vs tIoU | evaluate/results/map_vs_tiou.png |

## 七、模块接口文档

15 个模块的 `forward()` 均已添加完整的 shape docstring（见 meta_archs.py, backbones.py, blocks.py 等），
包含输入/输出张量形状、物理含义和内部数据流。

## 八、设计决策分析

详见 [evaluate/design_analysis.md](design_analysis.md)，涵盖 5 个核心 WHY 问题：
1. center_sample='radius' 的选择依据
2. num_bins=16 的精度-效率权衡
3. SGPBlock 5 分支的设计哲学
4. max_seq_len 必须被 32 整除的原因
5. DIoU vs GIoU 的 1D 场景分析

---
*报告由 evaluate/analyze_results.py 自动生成*

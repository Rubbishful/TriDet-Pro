# SGP Block 与通道注意力机制分析

## 1. SGP Block 结构详解

SGP (Scale-expanded Global Perception) Block 是 TriDet 主干网络的核心构建块，负责**多尺度时间特征提取**。每个 SGP Block 同时从 5 条路径捕获不同时间范围的信息：

```
输入 x: (B, C, T), mask: (B, 1, T)
  │
  ├─ 1. 下采样 (downsample)
  │     stride > 1 时: MaxPool1d 或 AvgPool1d
  │     stride = 1 时: Identity (stem 层)
  │
  ├─ 2. LayerNorm → out = self.ln(x)
  │
  ├─ 3. [SE 注入点 — pre_fusion 位置] → 见第 3 节
  │     if att_position == 'pre_fusion': out = self.att(out)
  │     在 5 分支 split 之前对通道做重标定，影响所有下游分支
  │
  ├─ 4. 五分支并行计算:
  │     ┌──────────────────────────────────────────────────────┐
  │     │ psi     = DepthwiseConv1d(out, kernel=K)              │  ← 局部时间模式
  │     │ fc      = DepthwiseConv1d(out, kernel=1)              │  ← 逐帧即时特征
  │     │ convw   = DepthwiseConv1d(out, kernel=K)              │  ← 局部窗口
  │     │ convkw  = DepthwiseConv1d(out, kernel=up_K)           │  ← 扩展窗口 (尺度扩展)
  │     │                                                          up_K = round((K+1) × k), 取奇数
  │     │ phi     = relu(DepthwiseConv1d(GAP(out), kernel=1))   │  ← 全局上下文向量
  │     └──────────────────────────────────────────────────────┘
  │
  ├─ 5. 融合: out = fc × phi + (convw + convkw) × psi + out
  │      ┌─────────────┬──────────────────────┬──────────┐
  │      │ fc × phi    │ 逐帧特征受全局上下文调制  │ "全局感知" │
  │      │ convw × psi │ 局部窗口通过 psi 门控     │ 精细时间  │
  │      │ convkw × psi│ 扩展窗口通过 psi 门控     │ 粗糙时间  │
  │      │ + out       │ 残差连接                 │ 恒等通路   │
  │      └─────────────┴──────────────────────┴──────────┘
  │
  ├─ 6. DropPath + 残差: out = x × mask + drop_path(out)
  │
  ├─ 7. FFN: out = out + drop_path(MLP(GroupNorm(out)))
  │     MLP = Conv1d(n_embd → 4×n_embd) → GELU → Conv1d(4×n_embd → n_embd)
  │
  └─ 输出: out (B, C, T'), mask (B, 1, T')
```

### 关键设计意图

| 设计元素 | 机制 | 目的 |
|---|---|---|
| **尺度扩展** | `convkw` 使用比 `convw` 大 `k` 倍的 kernel | 同时捕获细粒度和粗粒度时间模式 |
| **全局感知** | `phi` 来自全局平均池化，调制逐帧分支 `fc` | 用整个序列的上下文信息调节逐帧特征 |
| **psi 门控** | `convw` 和 `convkw` 都乘以 `psi` | 让局部和扩展窗口共享同一个门控信号 |
| **残差叠加** | 所有分支结果与原始 `out` 相加 | 保证信息流动，防止梯度消失 |

### 参数: kernel_size 与 up_size

- stem 层: `kernel_size=1` → `up_size = round(2×1.5) = 3` (奇数)
- branch 层: `kernel_size` 由配置 `sgp_win_size` 逐层指定，典型值 [3, 5, 7, 9, 11]

---

## 2. 主干网络结构 (SGPBackbone)

```
输入特征 (B, C_in, T)
  │
  ├─ Embedding: arch[0] 个 MaskedConv1D → ReLU
  │     C_in → n_embd (2048→512 for I3D)
  │     可选正弦位置编码
  │
  ├─ Stem: arch[1] 个 SGPBlock (kernel=1, stride=1)
  │     无下采样，保持全时间分辨率
  │     输出: (B, 512, T) → FPN 级别 0
  │
  └─ Branch: arch[2] 个 SGPBlock (kernel=K_i, stride=scale_factor)
        每个 block 将时间维度缩小 scale_factor 倍 (通常为 2)
        输出: (B, 512, T/2^i) → FPN 级别 1..5
```

默认配置 `arch=(2,2,5)`, `scale_factor=2`:
- 2 个 embedding conv
- 2 个 stem SGPBlock (全分辨率)
- 5 个 branch SGPBlock (逐级下采样至 T/32)
- 共 6 级 FPN 输出

---

## 3. SE 注意力注入位置分析

### 3.1 原始实现: Fusion 位置 (已弃用)

原实现将 SE 置于 5 分支融合 + DropPath 残差之后、FFN 之前：

```
out = self.ln(x)
  → 五分支计算 → 融合
  → out = x * out_mask + drop_path(融合)
  → out = self.att(out)     ← SE 原位置
  → FFN
```

**问题诊断** (基于 epoch 69 检查点):

| 指标 | 观测值 | 期望值 |
|---|---|---|
| SE 权重均值 | ~0 (静态), ~0.498 (Runtime Sigmoid 输出) | 偏离 0，分布 [0,1] |
| 通道被抑制 (<0.1) | 0.1% | 10-30% |
| 通道被增强 (>0.9) | 0.1% | 10-30% |
| 方差 (std) | 0.020-0.188 | >0.3 |

结论: **SE 层完全退化**。Sigmoid 输出全部 ≈0.5，等价于恒等变换。根源是融合后 `out = x + drop_path(融合)` 被 identity 路径主导（x 项远大于融合项），SE 的输入缺乏通道间差异性，梯度信号无法驱动 SE 学习。

### 3.2 当前实现: Pre-fusion 位置

将 SE 前移到 LayerNorm 之后、5 分支 split 之前：

```
out = self.ln(x)
  → out = self.att(out)     ← SE 新位置 (pre_fusion)
  → 五分支计算 → 融合 → DropPath → FFN
```

LayerNorm 输出是归一化后的"纯净"特征，通道间保留了原始的差异化信息。在分流处理之前先做通道筛选，SE 的通道权重会通过 `psi`、`fc`、`convw`、`convkw` 四条路径传播，影响力覆盖整个 SGP 融合过程。

### 3.3 训练结果对比

| 配置 | mAP | 说明 |
|---|---|---|
| 无 SE (baseline) | 68.59% | 基准 |
| fusion SE (r=16, epoch 69) | ~66.5% | SE 有害，-2.0% |
| **pre_fusion SE (r=16, epoch 55)** | **68.34%** | 基本消除负面影响，-0.25% |

pre_fusion 成功解决了 fusion 位置的退化问题，mAP 接近无 SE 的 baseline 水平。

### 3.4 Pre-fusion Runtime 权重分析 (epoch 55)

```
Layer          Mean    Std    <0.1    >0.9
branch.0.att   0.514   0.107  0.0%    0.0%
branch.1.att   0.507   0.098  0.0%    0.0%
branch.2.att   0.504   0.118  0.0%    0.1%
branch.3.att   0.516   0.122  0.0%    0.0%
branch.4.att   0.499   0.094  0.0%    0.0%
stem.0.att     0.491   0.086  0.0%    0.0%
stem.1.att     0.488   0.209  2.1%    2.0%
```

与 fusion 位置对比：

| 指标 | fusion (epoch 69) | pre_fusion (epoch 55) |
|---|---|---|
| Branch std | 0.020–0.059 | **0.094–0.122** (~2×) |
| stem.1 std | 0.188 | **0.209** |
| 通道抑制 <0.1 | ~0% | stem.1 有 2.1% |
| 通道增强 >0.9 | ~0% | stem.1 有 2.0% |

关键发现:
- pre_fusion 让 branch 层方差翻倍，确认了"输入信号质量"是 SE 学习的关键瓶颈
- stem.1（最后 stem block，全时间分辨率）最活跃，约 4% 通道 (~20/512) 产生了非平凡注意力
- 但 96%+ 的通道权重仍集中在 0.5 附近，距离有效注意力（10-30% 极端值）还差一个数量级

### 3.5 三种位置对比

```
                    SGPBlock forward 流程
                    ═══════════════════════

  x ──→ downsample ──→ LayerNorm ──┬──→ [A: pre_fusion]    ← 当前实现
                                    │         ↓
                                    ├──→ psi, fc, convw, convkw, global_fc
                                    │         ↓
                                    ├──→ fc×phi + (convw+convkw)×psi + out
                                    │         ↓
                                    ├──→ x + drop_path(out)
                                    │         ↓
                                    ├──→ [B: fusion]        ← 已弃用
                                    │         ↓
                                    ├──→ GroupNorm → MLP
                                    │         ↓
                                    └──→ [C: mlp]           ← 备选
```

| 位置 | 插入时机 | 输入特征特点 | 状态 |
|---|---|---|---|
| **A. pre_fusion** | LN 后，分支 split 前 | 归一化后的干净特征，通道间差异大 | **当前使用** |
| **B. fusion** | 5 分支融合 + 残差后 | 被 `x + drop_path(融合)` 的 identity 路径主导 | 已证实退化 |
| **C. mlp** | FFN 残差后 | 经过 MLP 非线性变换 | 未测试 |

### 3.6 选择 Pre-fusion 的理由

1. **信号质量最高**: LayerNorm 输出的通道间保留了差异化信息，SE 有足够的信号去学习通道重要性，不会被残差连接的 identity 项淹没
2. **影响力最大**: 一次通道重标定通过所有分支 (`psi`, `fc`, `convw`, `convkw`) 传播，影响整个 SGP 融合过程
3. **最小改动**: 仅移动 `self.att()` 的调用位置，不增加参数量，不修改 SELayer 本身
4. **直觉合理**: 在"分流处理"之前决定哪些通道更重要 — 先筛选信息，再分发到不同时间尺度处理

### 3.7 潜在风险

- Pre-fusion 的 SE 权重如果趋向极端（饱和到 0 或 1），可能永久性阻断某些通道进入分支处理
- 但由于融合阶段有残差连接 `out = fc*phi + (convw+convkw)*psi + out`，被抑制的通道仍可通过恒等路径传递，不会完全丢失

---

## 4. SELayer 内部机制

### 4.1 结构

SELayer（Squeeze-and-Excitation，1D 版本）通过两个 1×1 Conv1d 实现通道间的压缩-重建：

```python
class SELayer(nn.Module):
    def __init__(self, channels, reduction=16):
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),                          # 时间维压缩
            nn.Conv1d(channels, channels // reduction, 1),    # Squeeze: C → C/r
            nn.ReLU(inplace=True),
            nn.Conv1d(channels // reduction, channels, 1),    # Excitation: C/r → C
            nn.Sigmoid()                                       # 权重归一化到 (0,1)
        )

    def forward(self, x):
        return x * self.fc(x)   # 逐通道乘法
```

### 4.2 reduction 参数

`reduction` 控制 Squeeze 阶段的压缩比，决定 bottleneck 的宽度：

```
reduction = C / bottleneck_dim
```

以当前 C=512 为例：

| reduction | bottleneck | Squeeze 参数 | Excitation 参数 | 总参数量 | 压缩率 |
|---|---|---|---|---|---|
| 16 | 32 | 512×32+32=16,416 | 32×512+512=16,896 | **33,312** | 93.75% |
| 8 | 64 | 512×64+64=32,832 | 64×512+512=33,280 | **66,112** | 87.50% |
| 4 | 128 | 512×128+128=65,664 | 128×512+512=66,048 | **131,712** | 75.00% |

### 4.3 压缩实现原理

SE 的压缩不是通过降采样或 stride，而是通过 **kernel_size=1 的 Conv1d 在两个方向上的矩阵乘法**。

**Step 1 — 时间池化**: `AdaptiveAvgPool1d(1)` 将每个通道的时间序列压缩为单个标量

```
(B, 512, T) → (B, 512, 1)
每个通道在时间上取全局平均，得到 512 个标量
```

**Step 2 — Squeeze (fc.1)**: `Conv1d(512, C/r, kernel=1)` 即线性变换

```
对每个时间位置（只剩 1 个），做 512 → C/r 的线性变换:

  out[j] = bias[j] + Σ_{i=0}^{511} weight[j, i] × input[i]
  j = 0, 1, ..., C/r - 1

(B, 512, 1) → (B, C/r, 1)
```

**Step 3 — ReLU**: 非线性激活，丢弃负值

**Step 4 — Excitation (fc.3)**: `Conv1d(C/r, 512, kernel=1)` 逆线性变换

```
从 C/r 维重建 512 维:

  out[i] = bias[i] + Σ_{j=0}^{C/r-1} weight[i, j] × input[j]
  i = 0, 1, ..., 511

(B, C/r, 1) → (B, 512, 1)
```

**Step 5 — Sigmoid**: 映射到 (0, 1)，得到 512 个通道注意力权重

**Step 6 — Scale**: `x * weight`，每个通道乘以对应权重

### 4.4 信息瓶颈分析

```
512 维通道信息
    ↓ 压缩到 C/r 维
    ↓ ReLU 丢弃负值（额外信息损失）
    ↓ 从 C/r 维重建 512 维
    → 512 个注意力权重
```

当 `reduction=16` 时，bottleneck 仅 32 维，需编码 512 个通道间的所有相互关系。这个 93.75% 的信息压缩率极其激进——如果 32 维表示不足以捕捉通道间有意义的差异，网络的最优策略就是输出近似常数权重（~0.5），最小化错误重标定带来的 loss 风险。

这正是两个训练实验中观察到的现象：无论 fusion 还是 pre_fusion 位置，SE 权重都趋近 0.5。

### 4.5 优化方向: 减小 reduction

pre_fusion 解决了信号质量问题（位置），但 SE 仍然受限于 bottleneck 过窄。下一步将 `reduction` 从 16 降至 4：

- bottleneck 从 32 → 128，容量提升 4×
- 每个 SE 层参数量从 ~33K → ~132K，7 个 SE 层总计 ~0.92M
- 相对模型总参数量（约 30M+），增加约 3%，可以接受
- 更宽的 bottleneck 允许 SE 编码更丰富的通道间关系，有望打破"所有权重趋近 0.5"的退化状态

---

## 5. 实现细节

### 5.1 修改文件清单

| 文件 | 改动内容 |
|---|---|
| `libs/modeling/blocks.py` | `att_position='pre_fusion'` 支持：LN 后调用 `self.att(out)`，调整 `att_channels` 为 `n_embd` |
| `libs/modeling/backbones.py` | 无逻辑改动（已透传 `att_position`），仅更新注释 |
| `libs/modeling/meta_archs.py` | 无逻辑改动（已透传 `att_position`），仅更新注释 |
| `libs/core/config.py` | 默认值注释更新为 `'pre_fusion' \| 'fusion' \| 'mlp'` |
| `configs/thumos_i3d_se.yaml` | `att_position: pre_fusion`，`att_reduction: 4`，`batch_size: 8`，`learning_rate: 0.0002` |

### 5.2 SGPBlock 核心代码

```python
# __init__ — 根据位置选择 SE 输入通道数
if att_position == 'pre_fusion':
    att_channels = n_embd        # LN 输出始终是 n_embd
else:
    att_channels = n_out if n_out is not None else n_embd

# forward — 三个互斥的 SE 调用位置
out = self.ln(x)

if self.att_position == 'pre_fusion':
    out = self.att(out)          # ① LN 后、分支前 (当前使用)

psi = self.psi(out)              # 重标定后的特征流入所有分支
fc = self.fc(out)
convw = self.convw(out)
convkw = self.convkw(out)
phi = torch.relu(self.global_fc(out.mean(dim=-1, keepdim=True)))
out = fc * phi + (convw + convkw) * psi + out

out = x * out_mask + self.drop_path_out(out)

if self.att_position == 'fusion':
    out = self.att(out)          # ② 融合后 (已弃用)

out = out + self.drop_path_mlp(self.mlp(self.gn(out)))

if self.att_position == 'mlp':
    out = self.att(out)          # ③ MLP 后 (备选)
```

### 5.3 当前训练配置

```yaml
# configs/thumos_i3d_se.yaml (当前)
model:
  use_att: True
  att_type: SE
  att_position: pre_fusion     # LayerNorm 后注入
  att_reduction: 4               # bottleneck = 512/4 = 128 (从 16→32 降下来)
loader:
  batch_size: 8                  # 从 4 翻倍，显存利用率 ~80%
  num_workers: 4
opt:
  learning_rate: 0.0002          # 线性缩放: 0.0001 × (8/4)
  warmup_epochs: 20
  epochs: 50
  weight_decay: 0.025
```

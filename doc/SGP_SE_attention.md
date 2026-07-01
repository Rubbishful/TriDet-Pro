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

### 3.3 三种位置对比

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
                                    ├──→ [B: fusion]        ← 原位置 (已弃用)
                                    │         ↓
                                    ├──→ GroupNorm → MLP
                                    │         ↓
                                    └──→ [C: mlp]           ← 备选
```

| 位置 | 插入时机 | 输入特征特点 | 状态 |
|---|---|---|---|
| **A. pre_fusion** | LN 后，分支 split 前 | 归一化后的干净特征，通道间差异大 | **当前使用** |
| **B. fusion** | 5 分支融合 + 残差后 | 被 `x + drop_path(融合)` 的 identity 路径主导 | 已证实退化 |
| **C. mlp** | FFN 残差后 | 经过 MLP 非线性变换 | 未测试，预计与 fusion 类似 |

### 3.4 选择 Pre-fusion 的理由

1. **信号质量最高**: LayerNorm 输出的通道间保留了差异化信息，SE 有足够的信号去学习通道重要性，不会被残差连接的 identity 项淹没
2. **影响力最大**: 一次通道重标定通过所有分支 (`psi`, `fc`, `convw`, `convkw`) 传播，影响整个 SGP 融合过程
3. **最小改动**: 仅移动 `self.att()` 的调用位置，不增加参数量，不修改 SELayer 本身
4. **直觉合理**: 在"分流处理"之前决定哪些通道更重要 — 先筛选信息，再分发到不同时间尺度处理

### 3.5 潜在风险

- Pre-fusion 的 SE 权重如果趋向极端（饱和到 0 或 1），可能永久性阻断某些通道进入分支处理
- 但由于融合阶段有残差连接 `out = fc*phi + (convw+convkw)*psi + out`，被抑制的通道仍可通过恒等路径传递，不会完全丢失

---

## 4. 实现细节

### 4.1 修改文件清单

| 文件 | 改动内容 |
|---|---|
| `libs/modeling/blocks.py` | `att_position='pre_fusion'` 支持：LN 后调用 `self.att(out)`，调整 `att_channels` 为 `n_embd` |
| `libs/modeling/backbones.py` | 无逻辑改动（已透传 `att_position`），仅更新注释 |
| `libs/modeling/meta_archs.py` | 无逻辑改动（已透传 `att_position`），仅更新注释 |
| `libs/core/config.py` | 默认值注释更新为 `'pre_fusion' \| 'fusion' \| 'mlp'` |
| `configs/thumos_i3d_se.yaml` | `att_position: pre_fusion`，`batch_size: 8`，`learning_rate: 0.0002` |

### 4.2 SGPBlock 核心代码

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
    out = self.att(out)          # ② 融合后 (原行为，已弃用)

out = out + self.drop_path_mlp(self.mlp(self.gn(out)))

if self.att_position == 'mlp':
    out = self.att(out)          # ③ MLP 后 (备选)
```

### 4.3 训练配置

```yaml
# configs/thumos_i3d_se.yaml (当前)
model:
  use_att: True
  att_type: SE
  att_position: pre_fusion     # LayerNorm 后注入
  att_reduction: 16
loader:
  batch_size: 8                # 显存利用率 ~80% (基于 40% @ batch_size=4 估算)
  num_workers: 4
opt:
  learning_rate: 0.0002        # 线性缩放: 0.0001 × (8/4)
  warmup_epochs: 20
  epochs: 50
  weight_decay: 0.025
```

### 4.4 验证结果

使用 `tools/inspect_se_weights.py` 验证 pre_fusion 实现正确性：

```
Runtime Weight Distribution (pre_fusion 位置, epoch_069 权重)
Layer          Mean    Std    <0.1    >0.9
branch.0.att   0.497   0.056  0.0%    0.0%
branch.1.att   0.498   0.082  0.0%    0.0%
branch.2.att   0.497   0.107  0.0%    0.0%
branch.3.att   0.495   0.064  0.0%    0.0%
branch.4.att   0.497   0.076  0.0%    0.0%
stem.0.att     0.499   0.048  0.0%    0.0%
stem.1.att     0.497   0.174  0.8%    0.7%
```

注意: 以上使用的 checkpoint 是 fusion 位置训练的旧权重，SE 本身是退化的（mean≈0.5）。pre_fusion 仅改变 SE 的调用位置，不改变权重结构。**需要重新训练**才能评估 pre_fusion 是否能让 SE 学到有意义的通道注意力。

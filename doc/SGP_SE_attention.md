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
  ├─ 3. 五分支并行计算:
  │     ┌──────────────────────────────────────────────────────┐
  │     │ psi     = DepthwiseConv1d(out, kernel=K)              │  ← 局部时间模式
  │     │ fc      = DepthwiseConv1d(out, kernel=1)              │  ← 逐帧即时特征
  │     │ convw   = DepthwiseConv1d(out, kernel=K)              │  ← 局部窗口
  │     │ convkw  = DepthwiseConv1d(out, kernel=up_K)           │  ← 扩展窗口 (尺度扩展)
  │     │                                                          up_K = round((K+1) × k), 取奇数
  │     │ phi     = relu(DepthwiseConv1d(GAP(out), kernel=1))   │  ← 全局上下文向量
  │     └──────────────────────────────────────────────────────┘
  │
  ├─ 4. 融合: out = fc × phi + (convw + convkw) × psi + out
  │      ┌─────────────┬──────────────────────┬──────────┐
  │      │ fc × phi    │ 逐帧特征受全局上下文调制  │ "全局感知" │
  │      │ convw × psi │ 局部窗口通过 psi 门控     │ 精细时间  │
  │      │ convkw × psi│ 扩展窗口通过 psi 门控     │ 粗糙时间  │
  │      │ + out       │ 残差连接                 │ 恒等通路   │
  │      └─────────────┴──────────────────────┴──────────┘
  │
  ├─ 5. DropPath + 残差: out = x × mask + drop_path(out)
  │
  ├─ 6. [SE 注入点] → 见第 3 节
  │
  ├─ 7. FFN: out = out + drop_path(MLP(GroupNorm(out)))
  │      MLP = Conv1d(n_embd → 4×n_embd) → GELU → Conv1d(4×n_embd → n_embd)
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

### 3.1 当前实现: Fusion 位置

```
out = self.ln(x)
  → 五分支计算 → 融合
  → out = x + drop_path(融合)
  → out = self.att(out)     ← SE 在这里
  → FFN
```

**问题诊断** (基于 epoch 69 检查点):

| 指标 | 观测值 | 期望值 |
|---|---|---|
| SE 权重均值 | ~0 (静态), ~0.498 (Runtime Sigmoid 输出) | 偏离 0，分布 [0,1] |
| 通道被抑制 (<0.1) | 0.1% | 10-30% |
| 通道被增强 (>0.9) | 0.1% | 10-30% |
| 方差 (std) | 0.020-0.188 | >0.3 |

结论: **SE 层完全退化**。Sigmoid 输出全部 ≈0.5，等价于恒等变换。

根因推测: 融合后的 `out` 已经过残差连接 `x + drop_path(融合)`，信号被 identity 主导（x 项远大于融合项），SE 的输入缺乏通道间差异性，梯度信号无法驱动 SE 学习有意义的注意力权重。

### 3.2 替代位置对比

```
                    SGPBlock forward 流程
                    ═══════════════════════

  x ──→ downsample ──→ LayerNorm ──┬──→ [A: Pre-fusion]    ← 新增
                                    │         ↓
                                    ├──→ psi, fc, convw, convkw, global_fc
                                    │         ↓
                                    ├──→ fc×phi + (convw+convkw)×psi + out
                                    │         ↓
                                    ├──→ x + drop_path(out)
                                    │         ↓
                                    ├──→ [B: Fusion]        ← 当前位置
                                    │         ↓
                                    ├──→ GroupNorm → MLP
                                    │         ↓
                                    └──→ [C: MLP 后]        ← 已有选项
```

| 位置 | 插入时机 | 输入特征特点 | 参数量 | 预期效果 |
|---|---|---|---|---|
| **A. Pre-fusion** | LN 后，分支 split 前 | 归一化后的干净特征，未经混合和残差叠加 | ~33K/block | 在信息分流前做通道筛选，影响所有 5 条分支。通道间差异大，SE 有充足的"原材料"学习 |
| **B. Fusion** (当前) | 5 分支融合 + 残差后 | 已通过 `x + drop_path(融合)` 叠加，identity 主导 | ~33K/block | 已证实退化 — SE 学到接近恒等的权重 |
| **C. MLP 后** (已有) | FFN 残差后 | 经过 MLP 非线性变换后 | ~33K/block | 与 fusion 类似，额外应用一次 SE，同样面对 identity 主导问题 |

### 3.3 选择 Pre-fusion 的理由

1. **信号质量最高**: LayerNorm 输出是归一化后的"纯净"特征，通道间保留了原始的差异化信息，SE 有足够的信号去学习通道重要性
2. **影响力最大**: 一次通道重标定影响后续所有 5 条分支，以及整个 SGP 融合过程
3. **最小改动**: 只需移动 `self.att()` 的调用位置，不增加参数量，不需要修改 SELayer 本身
4. **直觉合理**: 在"分流处理"之前决定哪些通道更重要，符合注意力机制的直觉 — 先筛选信息，再分发处理

### 3.4 潜在风险

- Pre-fusion 的 SE 权重如果趋向极端（饱和到 0 或 1），可能永久性阻断某些通道
- 但由于有残差连接 `out = fc*phi + (convw+convkw)*psi + out`，被抑制的通道仍可通过恒等路径传递，不会完全丢失

---

## 4. 实现要点

### 修改文件

| 文件 | 改动 |
|---|---|
| `libs/modeling/blocks.py` | SGPBlock: `att_position='pre_fusion'` 支持 |
| `libs/modeling/backbones.py` | 无改动（已透传 `att_position`） |
| `libs/modeling/meta_archs.py` | 无改动（已透传 `att_position`） |
| `libs/core/config.py` | 注释更新 |
| 配置文件 `configs/*.yaml` | 设置 `att_position: pre_fusion` |

### 关键代码逻辑

```python
# SGPBlock.__init__
if att_position == 'pre_fusion':
    att_channels = n_embd   # LN 输出维度
else:
    att_channels = n_out if n_out is not None else n_embd

# SGPBlock.forward
out = self.ln(x)
if self.att_position == 'pre_fusion':
    out = self.att(out)     # 在分支执行前重标定通道
# ... 五分支计算和融合 ...
if self.att_position == 'fusion':
    out = self.att(out)     # 融合后重标定 (原行为)
# ... FFN ...
if self.att_position == 'mlp':
    out = self.att(out)     # MLP 后重标定 (原行为)
```

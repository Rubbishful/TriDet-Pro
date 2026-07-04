# TriDet 设计决策分析

## 1. 为什么 `center_sample='radius'`？

**相关代码**: [meta_archs.py](../libs/modeling/meta_archs.py#L595-L614)

**机制**: 中心采样将正样本严格限制在 GT 段的中心区域（而非整个段）。每个时间点的中心半径为 `stride * center_sample_radius`。

**WHY**:

- **减少边界模糊**: 动作段的边界天然模糊——"起跳"和"滞空"的精确分界点很难界定。在段边界的 anchor 点容易学习到相互矛盾的回归目标。
- **提高标签质量**: 中心区域的 anchor 点特征更纯净（离非本动作区域更远），类别标签信噪比更高。
- **对比 center_sample='none'**: 所有 GT 段内的点都成为正样本（含边界模糊点），会导致回归训练不稳定。

**验证方式** (消融 A7): 设置 `center_sample_radius=0.0` 并对比基线。

---

## 2. 为什么 `num_bins=16`？

**相关代码**: [meta_archs.py](../libs/modeling/meta_archs.py#L382-L424)

**机制**: Trident-head 不直接回归连续的偏移量，而是将问题拆解为：
1. 中心偏移头输出 `2×(num_bins+1)=34` 个值（左右各 17 个 bin）
2. 左右边界头各输出 17 个分类得分
3. 三路输出相加后 softmax，用期望值公式还原为连续偏移量

**WHY 16**:

- **离散化消噪**: 连续回归在边界模糊时容易受噪声干扰。离散化相当于给回归加了"分辨率限制"，梯度更稳定。
- **精度-效率 trade-off**: 17 个 bin（含 0）覆盖了 FPN 各层的回归范围（最粗层到 10000 个特征单位）。16 → 32 倍增开销但精度收益递减。
- **与 FPN 层级联动**: 每层只负责回归范围内的 bin，第 i 层的 max_bin 对应 `regression_range[i][1] / stride`。

---

## 3. 为什么 SGPBlock 有 5 个分支？

**相关代码**: [blocks.py](../libs/modeling/blocks.py#L277-L293)

数学表达: `out = fc * phi + (convw + convkw) * psi + x`

| 分支 | 实现 | 建模含义 |
|------|------|----------|
| `psi` | depthwise_conv k3 | **即时局部特征**: 该时间点周围的局部时序上下文 |
| `fc` | pointwise_conv 1×1 | **逐通道变换**: 线性投映，与 psi 逐元素相乘 |
| `convw` | depthwise_conv k3 | **局部窗口上下文**: 标准感受野的时序模式 |
| `convkw` | depthwise_conv `round((k+1)*k)` | **大窗口上下文**: k 参数控制的扩展感受野 |
| `global_fc` | pointwise 1×1 → avgpool → relu → 1×1 | **全局通道门控 phi**: 基于全序列统计量调控 fc 的贡献 |

**WHY**:
- **多尺度融合**: `convw`（标准窗口）和 `convkw`（大窗口）并行工作，同时捕捉短时和长时时序依赖。
- **门控机制**: `global_fc` 生成全局通道注意力 `phi`，让网络学会"哪些通道的特征对当前输入更重要"。
- **深度可分离卷积**: 所有 5 个分支都用 `groups=n_embd`（深度可分离），参数量仅 O(C×K) 而非 O(C²×K)。15.9M 总参数中 SGP 部分不到 3M。
- **残差连接**: `+ x` 确保即使 SGP 学习失败，也不比普通卷积差（训练稳定性）。

---

## 4. 为什么 `max_seq_len` 必须被 32 整除？

**相关代码**: [loc_generators.py](../libs/modeling/loc_generators.py#L48)

```python
assert max_seq_len % scale_factor ** (fpn_levels - 1) == 0
```

FPN 有 6 层 (arch[2]=5 → 1 stem + 5 branch = 6 层)，每层 stride=2。

最低层（第 5 层）= 原始分辨率 → 最高层 = 原始分辨率 / 2^5 = 原始分辨率 / 32。

如果 `max_seq_len` 不能被 32 整除，最高 FPN 层的 `T` 将是小数——这无意义。

**实例**: max_seq_len=1500 → 自动对齐到 1472 (1472/32=46 → 整除)
**实例**: max_seq_len=2304 → 2304/32=72 → 原生整除

该约束保证 PointGenerator 能在所有 FPN 层上均匀地切割时间网格。

---

## 5. 为什么使用 DIoU 而非 GIoU？

**相关代码**: [losses.py](../libs/modeling/losses.py#L111-L170)

**关键差异**:

DIoU loss = `1 - IoU + rho² / c²`
- `rho`: 预测中心和 GT 中心的距离
- `c`: 最小外接段的长度

GIoU loss = `1 - IoU + (C - U) / C` 
- 在 1D 中退化为 `1 - IoU`（因为最小外接段就是并集本身）

**WHY DIoU**:

- **中心距离惩罚**: 即使 IoU 相同，偏移方向错误的预测框得分更低。例如：GT=(0,10)，Pred_A=(0,8) 和 Pred_B=(2,10) 的 IoU 相同，但 B 的中心偏差更大 → DIoU 给 B 更高的损失。
- **GIoU 退化为 IoU**: 在 1D 事件检测中，GIoU 的最小外接段 = 并集，所以 C-U=0。GIoU 完全等同 IoU，丢失了距离信息。
- **收敛速度**: DIoU 通过中心距离直接引导预测框向 GT 中心靠拢，收敛更快。

此分析解释了为什么原始论文选择 DIoU 作为回归损失。

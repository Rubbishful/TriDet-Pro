# 置信度预测 IoU 增强 — 规划提示词

---

## Part A: 原理与数学推导

### A.1 问题定义

TriDet 当前置信度 `score = sigmoid(cls_logits)` 仅依赖分类概率。对于**分类正确但边界模糊**的预测，score 依然很高，导致 NMS 后残留低质量框。改进思路：让置信度同时反映分类确定性和边界清晰度（IoU）。

### A.2 1D IoU 公式（TriDet 参数化）

TriDet 使用共享中心点 + 左右偏移参数化 `(t_start, t_end) = (c - o_l, c + o_r)`。

给定点中心 `c`、stride `s`、预测偏移 `(pl, pr)`、GT 偏移 `(gl, gr)`：

```
pred_left  = c - pl*s,    pred_right = c + pr*s
gt_left    = c - gl*s,    gt_right   = c + gr*s

inter = max(0, min(pred_right, gt_right) - max(pred_left, gt_left))
union = (pred_right - pred_left) + (gt_right - gt_left) - inter
IoU_1d = inter / (union + ε)
```

### A.3 Task-Aligned Assigner (TAL)

参照 YOLOX SimOTA。用 alignment 分数替代纯几何规则分配正样本：

```
alignment = cls_score^α × IoU^β    (论文推荐 α=1, β=4~6)
```

对每个 GT 段，选 alignment 最高的 top-K 个锚点作为正样本。

**与现有 IoU-Aware 加权的关系**：现有代码（meta_archs.py L755-763）用 IoU 对 cls_loss 加权——IoU 越高 loss 越低，迫使模型关注低 IoU 困难样本。TAL 则偏好高 IoU 样本作正样本。**方向相反，可能冲突**，消融实验中需设 `iou_weight_power=0` 测试交互效果。

### A.4 IoU 预测分支

独立预测头输出单通道 IoU logits，训练时预测与 GT 的 1D IoU。

- **训练**：正样本 target = 真实 1D IoU，负样本 target = 0。BCE Loss。
- **推理**：`confidence = sigmoid(cls_logits) × sigmoid(iou_logits)`

**梯度隔离**：计算 target IoU 时，pred_offsets 必须 `.detach()`，否则 IoU 分支的梯度会干扰回归头。

**类别广播**：IoU 预测是类别无关的 `[T]`，融合时需 `unsqueeze(-1)` 广播为 `[T, 1]` 再与 cls_prob `[T, num_classes]` 相乘。

### A.5 Tavily 核查清单

实现前搜索以下主题，补全最新公式细节和超参建议：

1. `"Task-Aligned Assigner YOLOX alignment metric top-k dynamic k formula"`
2. `"IoU prediction head BCE loss vs Smooth L1 ablation"`
3. `"IoU-aware classification IoU branch complementary redundant interaction"`

---

## Part B: 工程实现计划

### B.1 涉及文件

| 文件 | 改动 |
|------|------|
| [libs/modeling/meta_archs.py](libs/modeling/meta_archs.py) | 新增 IoUHead 类、TAL 逻辑、IoU 分支 loss、推理融合 |
| [libs/core/config.py](libs/core/config.py) | 新增 8 个配置项默认值 |
| [libs/utils/train_utils.py](libs/utils/train_utils.py) | IoUHead 加入 weight decay 白名单 |
| [configs/thumos_i3d.yaml](configs/thumos_i3d.yaml) | 实验配置覆盖 |

### B.2 IoUHead 架构

```python
class IoUHead(nn.Module):
    """4 层 MaskedConv1D，输入 FPN feat [B, C, T]，输出 [B, 1, T]"""
    # 层数: 4 (较 ClsHead 的 3 层多一层，IoU 是更复杂的回归任务)
    # 多 FPN 层级共享参数（与 ClsHead / RegHead 一致）
```

### B.3 实现步骤

**Step 1 — IoUHead 类** (meta_archs.py)

在 `ClsHead`/`RegHead` 之后新增。骨架复用 `MaskedConv1D` 堆叠，最后一层输出通道=1。

**Step 2 — TAL 标签分配** (meta_archs.py `losses()`)

在现有 `label_points` 产出候选正样本后，插入 TAL 重筛选：

```
for each GT:
    candidates = geometric_assignment 结果
    cls_score = sigmoid(out_cls[候选点, gt_class])
    iou = compute_1d_iou(decoded_offsets[候选点], gt_offsets[候选点])
    alignment = cls_score^α × iou^β
    选 top-K 作为最终正样本，其余标记为负
```

引入 `tal_start_epoch` 控制启动时机。当前配置 `warmup_epochs=20, epochs=20` 会导致整个训练在 warmup，**需增设独立的 `tal_start_epoch`（默认 5）。**

**Step 3 — IoU Branch 训练目标** (meta_archs.py `losses()`)

```
pred_offsets_detached = decoded_offsets[pos_mask].detach()
iou_target = compute_1d_iou(pred_offsets_detached, gt_offsets)  # 正样本
iou_label = zeros(B, total_T)  # 负样本填 0
iou_label[pos_mask] = iou_target
iou_loss = BCE(iou_logits, iou_label) * iou_loss_weight
```

**Step 4 — 推理融合** (meta_archs.py `inference_single_video()` L862)

```python
# 原始:
pred_prob = (cls_i.sigmoid() * mask_i.unsqueeze(-1)).flatten()
# 修改为:
pred_prob = (cls_i.sigmoid() * iou_i.sigmoid().unsqueeze(-1) * mask_i.unsqueeze(-1)).flatten()
```

**Step 5 — 配置注册** (config.py + __init__ 签名)

新增配置项：`use_iou_head`, `iou_head_dim`, `iou_head_layers`, `iou_loss_weight`, `tal_topk`, `tal_alpha`, `tal_beta`, `tal_start_epoch`。同步 `TriDet.__init__` 参数签名。

### B.4 TridentHead 适配

TridentHead 下 `decoded_offsets` 形状为 `[B, FT, num_classes, 2]`（非 Trident: `[B, FT, 2]`）。正样本需要取对应类别的偏移：`pred_offsets = decoded_offsets[gt_cls[pos_mask].bool()]`（已有代码 L725-727）。

### B.5 消融实验（5 组）

| # | 实验 | 配置 diff | 目的 |
|---|------|-----------|------|
| 1 | Baseline | 不改 | THUMOS14 68.59% |
| 2 | TAL only | `tal_start_epoch=5, tal_topk=12` | 验证动态分配 |
| 3 | IoU Head only | `use_iou_head=True` | 验证 IoU 预测 |
| 4 | TAL + IoU Head | 两者启用 | 完整方案 |
| 5 | TAL + IoU Head + 关闭 IoU-Aware | 以上 + `iou_weight_power=0` | 排除现有机制干扰 |

### B.6 注意事项

- IoU 分支 loss 不共享 `loss_normalizer`，使用独立的 `iou_loss_weight`
- BCE 正负样本不均衡可加 `pos_weight` 参数
- `tal_start_epoch` 必须 ≤ epochs 且 ≤ warmup_epochs 否则永不触发
- 新增 `nn.Module` 子类（IoUHead）后必须更新 `make_optimizer()` 的 weight decay 分类循环

---

## 附录：关键代码位置速查

```
meta_archs.py:582-664   label_points_single_video  (当前几何标签分配)
meta_archs.py:685-785   losses                      (损失计算)
meta_archs.py:720-733   decoded_offsets / pos_mask  (正样本解码)
meta_archs.py:755-763   现有 IoU-Aware 分类加权
meta_archs.py:843-936   inference_single_video      (L862 置信度计算)
meta_archs.py:394-436   decode_offset               (TridentHead 解码)
train_utils.py:29-119   make_optimizer              (weight decay 分类)
config.py:80-105        模型配置项区域
```

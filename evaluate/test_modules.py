"""
TriDet 模块单元测试：逐层验证 shape 正确性 + 损失函数数值测试。

用法:
    cd e:/Tridet/TriDet-Pro
    python work/test_modules.py
"""

import sys
sys.path.insert(0, '.')

import torch
import torch.nn as nn
import numpy as np

from libs.modeling.blocks import MaskedConv1D, ConvBlock, SGPBlock, LayerNorm
from libs.modeling.backbones import SGPBackbone, ConvBackbone
from libs.modeling.necks import FPN1D, FPNIdentity
from libs.modeling.loc_generators import PointGenerator
from libs.modeling.losses import sigmoid_focal_loss, ctr_giou_loss_1d, ctr_diou_loss_1d
from libs.modeling.meta_archs import ClsHead, RegHead

# ---- 测试配置（与 thumos_i3d.yaml 对齐）----
B = 2
INPUT_DIM = 2048   # I3D 特征维度
EMBD_DIM = 512      # 嵌入维度
SGP_MLP_DIM = 768
T = 128             # 测试序列长度
FPN_DIM = 512
NUM_CLASSES = 20
NUM_BINS = 16
F = 6               # FPN 层数 (arch[-1] + 1)

pass_count = 0
fail_count = 0


def check(condition, msg):
    global pass_count, fail_count
    if condition:
        pass_count += 1
        print(f"  [PASS] {msg}")
    else:
        fail_count += 1
        print(f"  [FAIL] {msg}")


def make_dummy_input():
    x = torch.randn(B, INPUT_DIM, T)
    mask = torch.ones(B, 1, T, dtype=torch.bool)
    return x, mask


# ============================================================
# 1. 基础组件测试
# ============================================================
print("=" * 60)
print("1. 基础组件测试 (MaskedConv1D, LayerNorm, ConvBlock)")
print("=" * 60)

# 1.1 MaskedConv1D
x, mask = make_dummy_input()
conv = MaskedConv1D(INPUT_DIM, EMBD_DIM, 3, stride=1, padding=1)
out, out_mask = conv(x, mask)
check(out.shape == (B, EMBD_DIM, T), f"MaskedConv1D stride=1: {out.shape} == ({B},{EMBD_DIM},{T})")
check(out_mask.shape == (B, 1, T), f"MaskedConv1D mask: {out_mask.shape}")

# 1.2 MaskedConv1D stride=2
conv_s2 = MaskedConv1D(EMBD_DIM, EMBD_DIM, 3, stride=2, padding=1)
x2, m2 = make_dummy_input()
x2 = torch.randn(B, EMBD_DIM, T)
out_s2, mask_s2 = conv_s2(x2, m2)
check(out_s2.shape == (B, EMBD_DIM, T // 2), f"MaskedConv1D stride=2: {out_s2.shape} == ({B},{EMBD_DIM},{T//2})")

# 1.3 LayerNorm
ln = LayerNorm(EMBD_DIM)
x_ln = torch.randn(B, EMBD_DIM, T)
out_ln = ln(x_ln)
check(out_ln.shape == (B, EMBD_DIM, T), f"LayerNorm: {out_ln.shape}")

# 1.4 ConvBlock stride=1
cb1 = ConvBlock(EMBD_DIM, kernel_size=3, n_ds_stride=1)
x_cb = torch.randn(B, EMBD_DIM, T)
out_cb, mask_cb = cb1(x_cb, mask)
check(out_cb.shape == (B, EMBD_DIM, T), f"ConvBlock stride=1: {out_cb.shape} == ({B},{EMBD_DIM},{T})")

# 1.5 ConvBlock stride=2
cb2 = ConvBlock(EMBD_DIM, kernel_size=3, n_ds_stride=2)
out_cb2, mask_cb2 = cb2(x_cb, mask)
check(out_cb2.shape == (B, EMBD_DIM, T // 2), f"ConvBlock stride=2: {out_cb2.shape} == ({B},{EMBD_DIM},{T//2})")


# ============================================================
# 2. SGPBlock 测试
# ============================================================
print("\n" + "=" * 60)
print("2. SGPBlock 测试")
print("=" * 60)

# 2.1 SGPBlock stride=1
sgp1 = SGPBlock(EMBD_DIM, kernel_size=3, n_ds_stride=1, k=5.0, n_hidden=SGP_MLP_DIM)
x_sgp = torch.randn(B, EMBD_DIM, T)
out_sgp, mask_sgp = sgp1(x_sgp, mask)
check(out_sgp.shape == (B, EMBD_DIM, T), f"SGPBlock stride=1: {out_sgp.shape} == ({B},{EMBD_DIM},{T})")
check(mask_sgp.shape == (B, 1, T), f"SGPBlock mask stride=1: {mask_sgp.shape}")

# 2.2 SGPBlock stride=2
sgp2 = SGPBlock(EMBD_DIM, kernel_size=3, n_ds_stride=2, k=5.0, n_hidden=SGP_MLP_DIM)
out_sgp2, mask_sgp2 = sgp2(x_sgp, mask)
check(out_sgp2.shape == (B, EMBD_DIM, T // 2), f"SGPBlock stride=2: {out_sgp2.shape} == ({B},{EMBD_DIM},{T//2})")

# 2.3 SGPBlock with drop path
sgp_dp = SGPBlock(EMBD_DIM, kernel_size=3, n_ds_stride=1, k=5.0, n_hidden=SGP_MLP_DIM, path_pdrop=0.1)
sgp_dp.train()
out_dp, mask_dp = sgp_dp(x_sgp, mask)
check(out_dp.shape == (B, EMBD_DIM, T), f"SGPBlock drop_path: {out_dp.shape}")


# ============================================================
# 3. Backbone 测试
# ============================================================
print("\n" + "=" * 60)
print("3. Backbone 测试")
print("=" * 60)

# 3.1 SGPBackbone (完整 arch=(2,2,5))
x_raw, mask_raw = make_dummy_input()
sgp_bb = SGPBackbone(
    n_in=INPUT_DIM, n_embd=EMBD_DIM, sgp_mlp_dim=SGP_MLP_DIM,
    n_embd_ks=3, max_len=T, arch=(2, 2, 5), scale_factor=2,
    sgp_win_size=[1] * F, k=5.0, init_conv_vars=1.0
)
feats, masks = sgp_bb(x_raw, mask_raw)
check(len(feats) == F, f"SGPBackbone: #feats={len(feats)}, expected {F}")
check(len(masks) == F, f"SGPBackbone: #masks={len(masks)}, expected {F}")
for i in range(F):
    expected_T = T // (2 ** i)
    check(feats[i].shape == (B, EMBD_DIM, expected_T),
          f"SGPBackbone feat[{i}]: {feats[i].shape} == ({B},{EMBD_DIM},{expected_T})")
    check(masks[i].shape == (B, 1, expected_T),
          f"SGPBackbone mask[{i}]: {masks[i].shape}")

# 3.2 ConvBackbone
conv_bb = ConvBackbone(
    n_in=INPUT_DIM, n_embd=EMBD_DIM, n_embd_ks=3, arch=(2, 2, 5), scale_factor=2
)
feats_c, masks_c = conv_bb(x_raw, mask_raw)
check(len(feats_c) == F, f"ConvBackbone: #feats={len(feats_c)}, expected {F}")
for i in range(F):
    expected_T = T // (2 ** i)
    check(feats_c[i].shape == (B, EMBD_DIM, expected_T),
          f"ConvBackbone feat[{i}]: {feats_c[i].shape} == ({B},{EMBD_DIM},{expected_T})")


# ============================================================
# 4. Neck 测试
# ============================================================
print("\n" + "=" * 60)
print("4. Neck 测试")
print("=" * 60)

in_channels_list = [EMBD_DIM] * F

# 4.1 FPN1D
fpn = FPN1D(in_channels=in_channels_list, out_channel=FPN_DIM, scale_factor=2, with_ln=True)
fpn_feats, fpn_masks = fpn(feats, masks)
check(len(fpn_feats) == F, f"FPN1D: #out={len(fpn_feats)}, expected {F}")
for i in range(F):
    expected_T = T // (2 ** i)
    check(fpn_feats[i].shape == (B, FPN_DIM, expected_T),
          f"FPN1D feat[{i}]: {fpn_feats[i].shape} == ({B},{FPN_DIM},{expected_T})")

# 4.2 FPNIdentity
fpni = FPNIdentity(in_channels=in_channels_list, out_channel=FPN_DIM, scale_factor=2, with_ln=True)
fpni_feats, fpni_masks = fpni(feats, masks)
check(len(fpni_feats) == F, f"FPNIdentity: #out={len(fpni_feats)}, expected {F}")
for i in range(F):
    expected_T = T // (2 ** i)
    check(fpni_feats[i].shape == (B, EMBD_DIM, expected_T),
          f"FPNIdentity feat[{i}]: {fpni_feats[i].shape} == ({B},{EMBD_DIM},{expected_T})")


# ============================================================
# 5. PointGenerator 测试
# ============================================================
print("\n" + "=" * 60)
print("5. PointGenerator 测试")
print("=" * 60)

reg_ranges = [(0, 4), (4, 8), (8, 16), (16, 32), (32, 64), (64, 10000)]
strides = [1, 2, 4, 8, 16, 32]
pg = PointGenerator(max_seq_len=T, fpn_levels=F, scale_factor=2,
                    regression_range=reg_ranges, strides=strides)
pts = pg(feats)
check(len(pts) == F, f"PointGenerator: #levels={len(pts)}, expected {F}")
for i in range(F):
    expected_T = T // (2 ** i)
    check(pts[i].shape == (expected_T, 4),
          f"PointGenerator pts[{i}]: {pts[i].shape} == ({expected_T}, 4)")
    # 验证列意义
    t_col = pts[i][:, 0]  # 时间坐标
    stride_col = pts[i][:, 3]  # 步长
    check(torch.all(stride_col == strides[i]),
          f"PointGenerator stride[{i}]: all == {strides[i]}")
    # 时间坐标应该是 0, stride, 2*stride, ...
    expected_ts = torch.arange(0, expected_T * strides[i], strides[i], dtype=torch.float)
    check(torch.allclose(t_col, expected_ts),
          f"PointGenerator coords[{i}]: correct grid")


# ============================================================
# 6. ClsHead & RegHead 测试
# ============================================================
print("\n" + "=" * 60)
print("6. Head 测试")
print("=" * 60)

cl = ClsHead(FPN_DIM, FPN_DIM, NUM_CLASSES, kernel_size=3, with_ln=True, num_layers=3)
cls_out = cl(fpni_feats, fpni_masks)
check(len(cls_out) == F, f"ClsHead: #outs={len(cls_out)}, expected {F}")
for i in range(F):
    expected_T = T // (2 ** i)
    check(cls_out[i].shape == (B, NUM_CLASSES, expected_T),
          f"ClsHead logits[{i}]: {cls_out[i].shape} == ({B},{NUM_CLASSES},{expected_T})")

# 6.2 RegHead with Trident-head (num_bins=16)
reg_t = RegHead(FPN_DIM, FPN_DIM, F, kernel_size=3, with_ln=True, num_layers=3, num_bins=NUM_BINS)
reg_out_t = reg_t(fpni_feats, fpni_masks)
check(len(reg_out_t) == F, f"RegHead(Trident): #outs={len(reg_out_t)}")
for i in range(F):
    expected_T = T // (2 ** i)
    expected_C = 2 * (NUM_BINS + 1)
    check(reg_out_t[i].shape == (B, expected_C, expected_T),
          f"RegHead(Trident) offset[{i}]: {reg_out_t[i].shape} == ({B},{expected_C},{expected_T})")

# 6.3 RegHead without Trident-head (num_bins=0)
reg_plain = RegHead(FPN_DIM, FPN_DIM, F, kernel_size=3, with_ln=True, num_layers=3, num_bins=0)
reg_out_p = reg_plain(fpni_feats, fpni_masks)
check(len(reg_out_p) == F, f"RegHead(plain): #outs={len(reg_out_p)}")
for i in range(F):
    expected_T = T // (2 ** i)
    check(reg_out_p[i].shape == (B, 2, expected_T),
          f"RegHead(plain) offset[{i}]: {reg_out_p[i].shape} == ({B},2,{expected_T})")


# ============================================================
# 7. 损失函数数值测试
# ============================================================
print("\n" + "=" * 60)
print("7. 损失函数数值测试")
print("=" * 60)

# 7.1 DIoU loss: 完美匹配 → 应为 0
pred = torch.tensor([[1.0, 2.0]], dtype=torch.float32)  # 左偏移1, 右偏移2
gt = torch.tensor([[1.0, 2.0]], dtype=torch.float32)
diou = ctr_diou_loss_1d(pred, gt, reduction='none')
check(torch.allclose(diou, torch.tensor(0.0), atol=1e-5),
      f"DIoU perfect match: {diou.item():.6f} == 0.0")

# 7.2 DIoU loss: 偏移值不等 → 应 > 0
pred2 = torch.tensor([[0.5, 1.5]], dtype=torch.float32)
diou2 = ctr_diou_loss_1d(pred2, gt, reduction='none')
check(diou2.item() > 0, f"DIoU mismatch: {diou2.item():.4f} > 0")

# 7.3 GIoU loss: 完美匹配 → 应为 0
giou = ctr_giou_loss_1d(pred, gt, reduction='none')
check(torch.allclose(giou, torch.tensor(0.0), atol=1e-5),
      f"GIoU perfect match: {giou.item():.6f} == 0.0")

# 7.4 GIoU loss: 偏移值不等 → 应 > 0
giou2 = ctr_giou_loss_1d(pred2, gt, reduction='none')
check(giou2.item() > 0, f"GIoU mismatch: {giou2.item():.4f} > 0")

# 7.5 Focal loss: 高置信度正确预测 → 损失小
# 输入 logits 高正值 → sigmoid ≈ 1, target = 1 → loss 应很小
fl_easy = sigmoid_focal_loss(
    torch.tensor([[10.0]]), torch.tensor([[1.0]]), alpha=-1, reduction='none')
check(fl_easy.item() < 0.01,
      f"Focal loss easy positive: {fl_easy.item():.6f} < 0.01")

# 7.6 Focal loss: 低置信度错误预测 → 损失大
fl_hard = sigmoid_focal_loss(
    torch.tensor([[-10.0]]), torch.tensor([[1.0]]), alpha=-1, reduction='none')
check(fl_hard.item() > 0.5,
      f"Focal loss hard positive: {fl_hard.item():.4f} > 0.5")

# 7.7 Focal loss: 对称性测试 (pos vs neg)
fl_pos = sigmoid_focal_loss(
    torch.tensor([[0.0]]), torch.tensor([[1.0]]), alpha=-1, reduction='none')  # p=0.5, target=1
fl_neg = sigmoid_focal_loss(
    torch.tensor([[0.0]]), torch.tensor([[0.0]]), alpha=-1, reduction='none')  # p=0.5, target=0
check(torch.allclose(fl_pos, fl_neg, atol=1e-5),
      f"Focal loss symmetry: pos={fl_pos.item():.4f}, neg={fl_neg.item():.4f}")

# 7.8 Focal loss: reduction 模式
inputs = torch.randn(10, 5)
targets = (torch.rand(10, 5) > 0.7).float()
fl_none = sigmoid_focal_loss(inputs, targets, reduction='none')
fl_mean = sigmoid_focal_loss(inputs, targets, reduction='mean')
fl_sum = sigmoid_focal_loss(inputs, targets, reduction='sum')
check(fl_none.shape == inputs.shape, f"FL none shape: {fl_none.shape}")
check(fl_mean.numel() == 1, f"FL mean is scalar: {fl_mean.numel()}")
check(fl_sum.numel() == 1, f"FL sum is scalar: {fl_sum.numel()}")


# ---- 汇总 ----
print("\n" + "=" * 60)
print(f"测试结果: {pass_count} PASS, {fail_count} FAIL")
if fail_count == 0:
    print("全部通过！")
    sys.exit(0)
else:
    print("存在失败项，请检查！")
    sys.exit(1)

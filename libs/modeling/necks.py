import torch
from torch import nn
from torch.nn import functional as F

from .models import register_neck
from .blocks import MaskedConv1D, LayerNorm, AffineDropPath

@register_neck("fpn")
class FPN1D(nn.Module):
    """
        Feature pyramid network
    """
    def __init__(
        self,
        in_channels,      # input feature channels, len(in_channels) = # levels
        out_channel,      # output feature channel
        scale_factor=2.0, # downsampling rate between two fpn levels
        start_level=0,    # start fpn level
        end_level=-1,     # end fpn level
        with_ln=True      # if to apply layer norm at the end
    ):
        super().__init__()
        assert isinstance(in_channels, list) or isinstance(in_channels, tuple)

        self.in_channels = in_channels
        self.out_channel = out_channel
        self.scale_factor = scale_factor

        self.start_level = start_level
        if end_level == -1:
            self.end_level = len(in_channels)
        else:
            self.end_level = end_level
        assert self.end_level <= len(in_channels)
        assert (self.start_level >= 0) and (self.start_level < self.end_level)

        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()
        self.fpn_norms = nn.ModuleList()
        for i in range(self.start_level, self.end_level):
            # disable bias if using layer norm
            l_conv = MaskedConv1D(
                in_channels[i], out_channel, 1, bias=(not with_ln))
            # use depthwise conv here for efficiency
            fpn_conv = MaskedConv1D(
                out_channel, out_channel, 3,
                padding=1, bias=(not with_ln), groups=out_channel
            )
            # layer norm for order (B C T)
            if with_ln:
                fpn_norm = LayerNorm(out_channel)
            else:
                fpn_norm = nn.Identity()

            self.lateral_convs.append(l_conv)
            self.fpn_convs.append(fpn_conv)
            self.fpn_norms.append(fpn_norm)

    def forward(self, inputs, fpn_masks):
        """FPN 自上而下多尺度融合。

        流程:
            1. lateral_convs: 各层 1x1 conv 对齐通道到 out_channel
            2. top-down: 从高层向低层逐层最近邻上采样相加
            3. fpn_convs: 融合后过 depthwise conv 3x3 + LayerNorm

        Args:
            inputs: Tuple[L] of (B, C_in[i], T_i) 骨干输出金字塔
            fpn_masks: Tuple[L] of (B, 1, T_i) 对应 mask
        Returns:
            fpn_feats: Tuple[used_levels] of (B, out_channel, T_i) 融合后特征
            fpn_masks: 原样返回
        """
        assert len(inputs) == len(self.in_channels)
        assert len(fpn_masks) ==  len(self.in_channels)

        # build laterals, fpn_masks will remain the same with 1x1 convs
        laterals = []
        for i in range(len(self.lateral_convs)):
            x, _ = self.lateral_convs[i](
                inputs[i + self.start_level], fpn_masks[i + self.start_level]
            )
            laterals.append(x)

        # build top-down path
        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):
            laterals[i-1] += F.interpolate(
                laterals[i],
                scale_factor=self.scale_factor,
                mode='nearest'
            )

        # fpn conv / norm -> outputs
        # mask will remain the same
        fpn_feats = tuple()
        for i in range(used_backbone_levels):
            x, _ = self.fpn_convs[i](
                laterals[i], fpn_masks[i + self.start_level])
            x = self.fpn_norms[i](x)
            fpn_feats += (x, )

        return fpn_feats, fpn_masks

@register_neck('identity')
class FPNIdentity(nn.Module):
    def __init__(
        self,
        in_channels,      # input feature channels, len(in_channels) = # levels
        out_channel,      # output feature channel
        scale_factor=2.0, # downsampling rate between two fpn levels
        start_level=0,    # start fpn level
        end_level=-1,     # end fpn level
        with_ln=True      # if to apply layer norm at the end
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channel = out_channel
        self.scale_factor = scale_factor

        self.start_level = start_level
        if end_level == -1:
            self.end_level = len(in_channels)
        else:
            self.end_level = end_level
        assert self.end_level <= len(in_channels)
        assert (self.start_level >= 0) and (self.start_level < self.end_level)

        self.fpn_norms = nn.ModuleList()
        for i in range(self.start_level, self.end_level):
            # check feat dims
            assert self.in_channels[i + self.start_level] == self.out_channel
            # layer norm for order (B C T)
            if with_ln:
                fpn_norm = LayerNorm(out_channel)
            else:
                fpn_norm = nn.Identity()
            self.fpn_norms.append(fpn_norm)

    def forward(self, inputs, fpn_masks):
        """恒等映射 Neck（仅 LayerNorm，不做跨层融合）。

        THUMOS14 默认配置使用此 neck，因为 SGP 骨干的多尺度特征已足够判别。

        Args:
            inputs: Tuple[L] of (B, C_in[i], T_i), 要求 C_in[i] == out_channel
            fpn_masks: Tuple[L] of (B, 1, T_i)
        Returns:
            fpn_feats: Tuple[used_levels] of (B, out_channel, T_i) 仅过 LayerNorm
            fpn_masks: 原样返回
        """
        assert len(inputs) == len(self.in_channels)
        assert len(fpn_masks) ==  len(self.in_channels)

        # apply norms, fpn_masks will remain the same with 1x1 convs
        fpn_feats = tuple()
        for i in range(len(self.fpn_norms)):
            x = self.fpn_norms[i](inputs[i + self.start_level])
            fpn_feats += (x, )

        return fpn_feats, fpn_masks


class BiFPNFusion(nn.Module):
    """
    Fast normalized fusion from EfficientDet BiFPN (Tan et al., CVPR 2020).
    Supports 2 or 3 input feature maps with learnable per-input scalar weights.
    """
    def __init__(self, num_inputs=2, fusion_method='fast_norm'):
        super().__init__()
        assert num_inputs in (2, 3)
        assert fusion_method in ('fast_norm', 'sum')
        self.num_inputs = num_inputs
        self.fusion_method = fusion_method
        self.eps = 1e-4
        if fusion_method == 'fast_norm':
            self.weights = nn.Parameter(torch.ones(num_inputs, dtype=torch.float32))

    def forward(self, *inputs):
        assert len(inputs) == self.num_inputs
        if self.fusion_method == 'sum':
            out = inputs[0]
            for x in inputs[1:]:
                out = out + x
            return out
        # fast_norm
        w = torch.relu(self.weights)
        w_sum = w.sum() + self.eps
        out = w[0] * inputs[0]
        for i in range(1, self.num_inputs):
            out = out + w[i] * inputs[i]
        return out / w_sum


class BiFPNBlock(nn.Module):
    """
    A single BiFPN block with top-down and bottom-up pathways.
    """
    def __init__(self, num_levels, out_channel, scale_factor=2.0, with_ln=True, fusion_method='fast_norm', drop_path=0.0):
        super().__init__()
        self.num_levels = num_levels
        self.scale_factor = scale_factor

        # Top-down pathway
        self.td_convs = nn.ModuleList()
        self.td_norms = nn.ModuleList()
        self.td_fusions = nn.ModuleList()
        for i in range(num_levels):
            self.td_convs.append(MaskedConv1D(
                out_channel, out_channel, 3, padding=1,
                bias=(not with_ln), groups=out_channel
            ))
            self.td_norms.append(LayerNorm(out_channel) if with_ln else nn.Identity())
            if i < num_levels - 1:
                self.td_fusions.append(BiFPNFusion(2, fusion_method))

        # Bottom-up pathway
        self.bu_convs = nn.ModuleList()
        self.bu_norms = nn.ModuleList()
        self.bu_fusions = nn.ModuleList()
        for i in range(num_levels):
            self.bu_convs.append(MaskedConv1D(
                out_channel, out_channel, 3, padding=1,
                bias=(not with_ln), groups=out_channel
            ))
            self.bu_norms.append(LayerNorm(out_channel) if with_ln else nn.Identity())
            n_inputs = 2 if i == 0 else 3
            self.bu_fusions.append(BiFPNFusion(n_inputs, fusion_method))

        # Downsample convs for bottom-up path
        self.ds_convs = nn.ModuleList()
        for i in range(num_levels - 1):
            self.ds_convs.append(MaskedConv1D(
                out_channel, out_channel, 3, stride=2, padding=1,
                bias=(not with_ln), groups=out_channel
            ))

        # Stochastic depth
        if drop_path > 0.0:
            self.drop_path = AffineDropPath(out_channel, drop_path)
        else:
            self.drop_path = nn.Identity()

    def forward(self, feats, masks):
        N = self.num_levels

        # Top-down pathway
        td_feats = [None] * N
        td_feats[N - 1], _ = self.td_convs[N - 1](feats[N - 1], masks[N - 1])
        td_feats[N - 1] = self.td_norms[N - 1](td_feats[N - 1])

        for i in range(N - 2, -1, -1):
            up_feat = F.interpolate(td_feats[i + 1], scale_factor=self.scale_factor, mode='nearest')
            fused = self.td_fusions[i](feats[i], up_feat)
            td_feats[i], _ = self.td_convs[i](fused, masks[i])
            td_feats[i] = self.td_norms[i](td_feats[i])

        # Bottom-up pathway
        out_feats = [None] * N
        fused = self.bu_fusions[0](feats[0], td_feats[0])
        out_feats[0], _ = self.bu_convs[0](fused, masks[0])
        out_feats[0] = self.bu_norms[0](out_feats[0])

        for i in range(1, N):
            down_feat, _ = self.ds_convs[i - 1](out_feats[i - 1], masks[i - 1])
            fused = self.bu_fusions[i](feats[i], td_feats[i], down_feat)
            out_feats[i], _ = self.bu_convs[i](fused, masks[i])
            out_feats[i] = self.bu_norms[i](out_feats[i])

        # Apply stochastic depth
        out_feats = [self.drop_path(f) for f in out_feats]

        return out_feats


@register_neck("bifpn")
class BiFPN1D(nn.Module):
    """
    Bidirectional Feature Pyramid Network for 1D temporal features.
    Based on EfficientDet BiFPN (Tan et al., CVPR 2020).
    """
    def __init__(
        self,
        in_channels,
        out_channel,
        scale_factor=2.0,
        start_level=0,
        end_level=-1,
        with_ln=True,
        num_repeats=1,
        fusion_method='sum',
        drop_path=0.0
    ):
        super().__init__()
        assert isinstance(in_channels, (list, tuple))

        self.in_channels = in_channels
        self.out_channel = out_channel
        self.scale_factor = scale_factor

        self.start_level = start_level
        if end_level == -1:
            self.end_level = len(in_channels)
        else:
            self.end_level = end_level
        assert self.end_level <= len(in_channels)
        assert (self.start_level >= 0) and (self.start_level < self.end_level)

        num_levels = self.end_level - self.start_level

        # Lateral 1x1 convs
        self.lateral_convs = nn.ModuleList()
        for i in range(self.start_level, self.end_level):
            l_conv = MaskedConv1D(
                in_channels[i], out_channel, 1, bias=(not with_ln))
            self.lateral_convs.append(l_conv)

        # Repeated BiFPN blocks
        self.bifpn_blocks = nn.ModuleList()
        for _ in range(num_repeats):
            self.bifpn_blocks.append(
                BiFPNBlock(num_levels, out_channel, scale_factor, with_ln, fusion_method, drop_path)
            )

    def forward(self, inputs, fpn_masks):
        assert len(inputs) == len(self.in_channels)
        assert len(fpn_masks) == len(self.in_channels)

        # Lateral projection
        feats = []
        for i in range(len(self.lateral_convs)):
            x, _ = self.lateral_convs[i](
                inputs[i + self.start_level], fpn_masks[i + self.start_level])
            feats.append(x)

        masks = [fpn_masks[i + self.start_level] for i in range(len(self.lateral_convs))]

        # Apply repeated BiFPN blocks (with residual connections)
        for block in self.bifpn_blocks:
            new_feats = block(feats, masks)
            feats = [new_feats[i] + feats[i] for i in range(len(feats))]

        return tuple(feats), fpn_masks

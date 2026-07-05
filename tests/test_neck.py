"""Smoke tests for FPN and BiFPN neck modules."""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
from libs.modeling.necks import FPN1D, FPNIdentity, BiFPNFusion, BiFPNBlock, BiFPN1D


def _make_feats_and_masks(n_levels=6, base_len=128):
    """Generate feats and corresponding fpn_masks for neck tests.
    
    Uses power-of-2 base_len to avoid interpolation size mismatch
    when FPN upsamples between unevenly-divided levels.
    """
    feats = [torch.randn(1, 512, base_len // (2 ** i)) for i in range(n_levels)]
    masks = tuple(torch.ones(1, 1, base_len // (2 ** i), dtype=torch.bool) for i in range(n_levels))
    return feats, masks


def test_fpn1d():
    """FPN1D: basic forward pass with 6 levels."""
    neck = FPN1D(in_channels=[512]*6, out_channel=512)
    feats, masks = _make_feats_and_masks()
    out_feats, _ = neck(feats, masks)
    assert len(out_feats) == 6
    for i, o in enumerate(out_feats):
        assert o.shape[1] == 512, f'Level {i}: expected 512 channels, got {o.shape[1]}'
    print(f'FPN1D: {[f.shape for f in feats]} -> {[f.shape for f in out_feats]} OK')


def test_fpn_identity():
    """FPNIdentity: should passthrough."""
    neck = FPNIdentity(in_channels=[512]*6, out_channel=512)
    feats, masks = _make_feats_and_masks()
    out_feats, _ = neck(feats, masks)
    for i, (fi, fo) in enumerate(zip(feats, out_feats)):
        assert fi.shape == fo.shape, f'FPNIdentity level {i}: shape mismatch'
    print(f'FPNIdentity: {len(out_feats)} levels OK')


def test_bifpn_fusion():
    """BiFPNFusion: weighted feature fusion (channel-agnostic)."""
    fusion = BiFPNFusion(num_inputs=2)
    x1 = torch.randn(1, 512, 50)
    x2 = torch.randn(1, 512, 50)
    out = fusion(x1, x2)
    assert out.shape == x1.shape
    print(f'BiFPNFusion: {x1.shape} + {x2.shape} -> {out.shape} OK')


def test_bifpn_block():
    """BiFPNBlock: single block forward pass."""
    block = BiFPNBlock(num_levels=6, out_channel=512)
    feats, masks_tuple = _make_feats_and_masks()
    keys = list(range(6))
    feat_dict = dict(zip(keys, feats))
    mask_dict = dict(zip(keys, masks_tuple))
    out = block(feat_dict, mask_dict)
    assert len(out) == 6
    for i, o in enumerate(out):
        assert o.shape[1] == 512, f'p{i}: wrong channels'
    print(f'BiFPNBlock: {[f.shape for f in feats]} -> {[o.shape for o in out]} OK')


def test_bifpn1d():
    """BiFPN1D: full BiFPN neck with multiple blocks."""
    neck = BiFPN1D(in_channels=[512]*6, out_channel=512, num_repeats=2)
    feats, masks = _make_feats_and_masks()
    out_feats, _ = neck(feats, masks)
    assert len(out_feats) == 6
    for i, o in enumerate(out_feats):
        assert o.shape[1] == 512
    print(f'BiFPN1D (2 repeats): {len(out_feats)} levels OK')
    params = sum(p.numel() for p in neck.parameters())
    print(f'BiFPN1D params: {params:,}')


if __name__ == '__main__':
    test_fpn1d()
    test_fpn_identity()
    test_bifpn_fusion()
    test_bifpn_block()
    test_bifpn1d()
    print('\n=== All neck tests passed! ===')

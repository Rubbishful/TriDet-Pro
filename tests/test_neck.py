"""Smoke tests for FPN and BiFPN neck modules."""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
from libs.modeling.necks import FPN1D, FPNIdentity, BiFPNFusion, BiFPNBlock, BiFPN1D


def test_fpn1d():
    """FPN1D: basic forward pass with 6 levels."""
    neck = FPN1D(in_channels=[512]*6, out_channel=512)
    feats = [torch.randn(1, 512, 100 // (2**i)) for i in range(6)]
    out = neck(feats)
    assert len(out) == 6
    for i, o in enumerate(out):
        assert o.shape[1] == 512, f'Level {i}: expected 512 channels, got {o.shape[1]}'
    print(f'FPN1D: {[f.shape for f in feats]} -> {[f.shape for f in out]} OK')


def test_fpn_identity():
    """FPNIdentity: should passthrough."""
    neck = FPNIdentity(in_channels=[512]*6, out_channel=512)
    feats = [torch.randn(1, 512, 100 // (2**i)) for i in range(6)]
    out = neck(feats)
    for i, (fi, fo) in enumerate(zip(feats, out)):
        assert fi.shape == fo.shape, f'FPNIdentity level {i}: shape mismatch'
    print(f'FPNIdentity: {len(out)} levels OK')


def test_bifpn_fusion():
    """BiFPNFusion: weighted feature fusion."""
    fusion = BiFPNFusion(channels=512)
    x1 = torch.randn(1, 512, 50)
    x2 = torch.randn(1, 512, 50)
    out = fusion(x1, x2)
    assert out.shape == x1.shape
    print(f'BiFPNFusion: {x1.shape} + {x2.shape} -> {out.shape} OK')


def test_bifpn_block():
    """BiFPNBlock: single block forward pass."""
    block = BiFPNBlock(in_channels=[512]*6, out_channel=512)
    feats = [torch.randn(1, 512, 100 // (2**i)) for i in range(6)]
    keys = [f'p{i}' for i in range(6)]
    feat_dict = dict(zip(keys, feats))
    out = block(feat_dict)
    assert len(out) == 6
    for k, v in out.items():
        assert v.shape[1] == 512, f'{k}: wrong channels'
    print(f'BiFPNBlock: {[f.shape for f in feats]} -> {[(k, v.shape) for k, v in out.items()]} OK')


def test_bifpn1d():
    """BiFPN1D: full BiFPN neck with multiple blocks."""
    neck = BiFPN1D(in_channels=[512]*6, out_channel=512, num_repeats=2)
    feats = [torch.randn(1, 512, 100 // (2**i)) for i in range(6)]
    out = neck(feats)
    assert len(out) == 6
    for i, o in enumerate(out):
        assert o.shape[1] == 512
    print(f'BiFPN1D (2 repeats): {len(out)} levels OK')
    params = sum(p.numel() for p in neck.parameters())
    print(f'BiFPN1D params: {params:,}')


if __name__ == '__main__':
    test_fpn1d()
    test_fpn_identity()
    test_bifpn_fusion()
    test_bifpn_block()
    test_bifpn1d()
    print('\n=== All neck tests passed! ===')

"""Smoke tests for channel attention modules and SGPBlock integration."""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
from libs.modeling.blocks import SELayer, ECALayer, SGPBlock


def test_selayer():
    """SELayer: shape preservation, param count sanity."""
    x = torch.randn(2, 512, 100)
    se = SELayer(512, reduction=16)
    out = se(x)
    assert out.shape == x.shape, f'SELayer shape mismatch: {out.shape} vs {x.shape}'
    params = sum(p.numel() for p in se.parameters())
    assert params > 0, 'SELayer should have parameters'
    print(f'SELayer(512, r=16): {out.shape} OK, params={params:,}')


def test_ecalayer():
    """ECALayer: shape preservation, lightweight."""
    x = torch.randn(2, 512, 100)
    eca = ECALayer(512, kernel_size=3)
    out = eca(x)
    assert out.shape == x.shape, f'ECALayer shape mismatch: {out.shape} vs {x.shape}'
    params = sum(p.numel() for p in eca.parameters())
    print(f'ECALayer(512, k=3): {out.shape} OK, params={params:,}')


def test_sgpblock_no_att():
    """SGPBlock(use_att=False) backward compatibility."""
    x = torch.randn(2, 512, 100)
    mask = torch.ones(2, 1, 100, dtype=torch.bool)
    block = SGPBlock(512, kernel_size=3, n_ds_stride=1, use_att=False)
    out, out_mask = block(x, mask)
    assert out.shape == x.shape
    assert out_mask.shape == mask.shape
    print(f'SGPBlock(att=off): {out.shape} OK')


def test_sgpblock_se_fusion():
    """SGPBlock with SE at fusion position."""
    x = torch.randn(2, 512, 100)
    mask = torch.ones(2, 1, 100, dtype=torch.bool)
    block = SGPBlock(512, kernel_size=3, n_ds_stride=1,
                     use_att=True, att_type='SE', att_position='fusion',
                     att_reduction=16)
    out, _ = block(x, mask)
    assert out.shape == x.shape
    att_params = sum(p.numel() for p in block.att.parameters())
    assert att_params > 0, 'Attention should have parameters'
    print(f'SGPBlock(SE, fusion): {out.shape} OK, att_params={att_params:,}')


def test_sgpblock_eca_fusion():
    """SGPBlock with ECA at fusion position."""
    x = torch.randn(2, 512, 100)
    mask = torch.ones(2, 1, 100, dtype=torch.bool)
    block = SGPBlock(512, kernel_size=3, n_ds_stride=1,
                     use_att=True, att_type='ECA', att_position='fusion',
                     att_kernel_size=3)
    out, _ = block(x, mask)
    assert out.shape == x.shape
    print(f'SGPBlock(ECA, fusion): {out.shape} OK')


def test_sgpblock_se_mlp():
    """SGPBlock with SE at mlp position."""
    x = torch.randn(2, 512, 100)
    mask = torch.ones(2, 1, 100, dtype=torch.bool)
    block = SGPBlock(512, kernel_size=3, n_ds_stride=1,
                     use_att=True, att_type='SE', att_position='mlp',
                     att_reduction=16)
    out, _ = block(x, mask)
    assert out.shape == x.shape
    print(f'SGPBlock(SE, mlp): {out.shape} OK')


def test_sgpblock_stride():
    """SGPBlock with n_ds_stride=2 downsampling."""
    x = torch.randn(2, 512, 100)
    mask = torch.ones(2, 1, 100, dtype=torch.bool)
    block = SGPBlock(512, kernel_size=3, n_ds_stride=2, use_att=False)
    out, out_mask = block(x, mask)
    expected_len = 100 // 2
    assert out.shape[2] == expected_len, f'Expected len={expected_len}, got {out.shape[2]}'
    print(f'SGPBlock(stride=2): {x.shape} -> {out.shape} OK')


if __name__ == '__main__':
    test_selayer()
    test_ecalayer()
    test_sgpblock_no_att()
    test_sgpblock_se_fusion()
    test_sgpblock_eca_fusion()
    test_sgpblock_se_mlp()
    test_sgpblock_stride()
    print('\n=== All attention tests passed! ===')

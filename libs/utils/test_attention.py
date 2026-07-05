"""Test script for channel attention modules (SE/ECA) in SGPBlock."""
import torch
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from libs.modeling.blocks import SELayer, ECALayer, SGPBlock
from libs.modeling.backbones import SGPBackbone
from libs.modeling.meta_archs import TriDet


def test_attention_layers():
    """Test 1: SELayer & ECALayer instantiation and shape."""
    print('=== Test 1: SELayer & ECALayer instantiation ===')
    x = torch.randn(2, 512, 100)

    se = SELayer(512, reduction=16)
    out = se(x)
    assert out.shape == x.shape, f'SELayer shape mismatch: {out.shape} vs {x.shape}'
    print(f'SELayer: input {x.shape} -> output {out.shape} OK, '
          f'params: {sum(p.numel() for p in se.parameters())}')

    eca = ECALayer(512, kernel_size=3)
    out = eca(x)
    assert out.shape == x.shape, f'ECALayer shape mismatch: {out.shape} vs {x.shape}'
    print(f'ECALayer: input {x.shape} -> output {out.shape} OK, '
          f'params: {sum(p.numel() for p in eca.parameters())}')


def test_sgpblock_default():
    """Test 2: SGPBlock backward compatibility (use_att=False)."""
    print('\n=== Test 2: SGPBlock backward compatibility (use_att=False) ===')
    x = torch.randn(2, 512, 100)
    mask = torch.ones(2, 1, 100, dtype=torch.bool)
    block = SGPBlock(512, kernel_size=3, n_ds_stride=1, use_att=False)
    out, out_mask = block(x, mask)
    assert out.shape == x.shape, f'shape mismatch: {out.shape} vs {x.shape}'
    print(f'SGPBlock (att=off): input {x.shape} -> output {out.shape} OK')


def test_sgpblock_se():
    """Test 3: SGPBlock with SE attention (fusion position)."""
    print('\n=== Test 3: SGPBlock with SE attention (fusion) ===')
    x = torch.randn(2, 512, 100)
    mask = torch.ones(2, 1, 100, dtype=torch.bool)
    block = SGPBlock(512, kernel_size=3, n_ds_stride=1,
                     use_att=True, att_type='SE', att_position='fusion', att_reduction=16)
    out, _ = block(x, mask)
    assert out.shape == x.shape, f'shape mismatch: {out.shape}'
    total = sum(p.numel() for p in block.parameters())
    att = sum(p.numel() for p in block.att.parameters())
    print(f'SGPBlock (SE, fusion): output {out.shape} OK')
    print(f'  Total params: {total:,}, Attention params: {att:,} ({att/total*100:.2f}%)')


def test_sgpblock_eca():
    """Test 4: SGPBlock with ECA attention (fusion position)."""
    print('\n=== Test 4: SGPBlock with ECA attention (fusion) ===')
    x = torch.randn(2, 512, 100)
    mask = torch.ones(2, 1, 100, dtype=torch.bool)
    block = SGPBlock(512, kernel_size=3, n_ds_stride=1,
                     use_att=True, att_type='ECA', att_position='fusion', att_kernel_size=3)
    out, _ = block(x, mask)
    assert out.shape == x.shape, f'shape mismatch: {out.shape}'
    att = sum(p.numel() for p in block.att.parameters())
    print(f'SGPBlock (ECA, fusion): output {out.shape} OK, Attention params: {att:,}')


def test_sgpblock_mlp_position():
    """Test 5: SGPBlock with SE attention at mlp position."""
    print('\n=== Test 5: SGPBlock with SE attention (mlp position) ===')
    x = torch.randn(2, 512, 100)
    mask = torch.ones(2, 1, 100, dtype=torch.bool)
    block = SGPBlock(512, kernel_size=3, n_ds_stride=1,
                     use_att=True, att_type='SE', att_position='mlp', att_reduction=16)
    out, _ = block(x, mask)
    assert out.shape == x.shape, f'shape mismatch: {out.shape}'
    print(f'SGPBlock (SE, mlp): output {out.shape} OK')


def test_deterministic_without_att():
    """Test 6: Two same SGPBlock(att=off) produce identical output."""
    print('\n=== Test 6: Deterministic output (use_att=False) ===')
    torch.manual_seed(42)
    x = torch.randn(1, 512, 64)
    mask = torch.ones(1, 1, 64, dtype=torch.bool)

    torch.manual_seed(123)
    block1 = SGPBlock(512, kernel_size=3, n_ds_stride=2, use_att=False)
    torch.manual_seed(123)
    block2 = SGPBlock(512, kernel_size=3, n_ds_stride=2, use_att=False)
    out1, _ = block1(x, mask)
    out2, _ = block2(x, mask)
    diff = (out1 - out2).abs().max().item()
    print(f'Max diff between two identical blocks: {diff:.10f}')
    assert diff == 0.0, 'Outputs should be identical!'


def test_sgpbackbone():
    """Test 7: SGPBackbone with and without attention."""
    print('\n=== Test 7: SGPBackbone integration ===')
    # Without attention
    bb_off = SGPBackbone(n_in=2304, n_embd=512, sgp_mlp_dim=512, n_embd_ks=3,
                         max_len=2304, sgp_win_size=[3]*6, use_att=False)
    print(f'SGPBackbone(att=off) params: {sum(p.numel() for p in bb_off.parameters()):,}')

    # With SE attention
    bb_se = SGPBackbone(n_in=2304, n_embd=512, sgp_mlp_dim=512, n_embd_ks=3,
                        max_len=2304, sgp_win_size=[3]*6, use_att=True, att_type='SE',
                        att_position='fusion', att_reduction=16)
    print(f'SGPBackbone(att=SE)  params: {sum(p.numel() for p in bb_se.parameters()):,}')

    # With ECA attention
    bb_eca = SGPBackbone(n_in=2304, n_embd=512, sgp_mlp_dim=512, n_embd_ks=3,
                         max_len=2304, sgp_win_size=[3]*6, use_att=True, att_type='ECA',
                         att_position='fusion', att_kernel_size=3)
    print(f'SGPBackbone(att=ECA) params: {sum(p.numel() for p in bb_eca.parameters()):,}')

    # Forward test
    x = torch.randn(1, 2304, 128)
    mask = torch.ones(1, 1, 128, dtype=torch.bool)
    feats, masks = bb_se(x, mask)
    print(f'Forward: input {x.shape} -> {len(feats)} levels, shapes: '
          f'{[f.shape for f in feats]}')
    assert len(feats) == 6, f'Expected 6 levels, got {len(feats)}'
    print('SGPBackbone forward OK')


def test_full_tridet():
    """Test 8: Full TriDet model with config-driven attention."""
    print('\n=== Test 8: Full TriDet model forward pass ===')
    from libs.core.config import load_default_config

    cfg = load_default_config()
    cfg = cfg.copy()

    # Override for quick test
    cfg['dataset']['input_dim'] = 2048
    cfg['dataset']['num_classes'] = 20
    cfg['dataset']['max_seq_len'] = 256
    cfg['model']['input_dim'] = 2048
    cfg['model']['num_classes'] = 20
    cfg['model']['max_seq_len'] = 256
    cfg['model']['sgp_mlp_dim'] = 512
    cfg['model']['n_sgp_win_size'] = 3
    cfg['model']['train_cfg'] = cfg['train_cfg']
    cfg['model']['test_cfg'] = cfg['test_cfg']

    # Test without attention
    print('--- Without attention ---')
    cfg['model']['use_att'] = False
    model_off = TriDet(**cfg['model'])
    params_off = sum(p.numel() for p in model_off.parameters())
    print(f'TriDet(att=off) params: {params_off:,}')

    # Test with SE attention
    print('--- With SE attention (fusion) ---')
    cfg['model']['use_att'] = True
    cfg['model']['att_type'] = 'SE'
    cfg['model']['att_position'] = 'fusion'
    cfg['model']['att_reduction'] = 16
    model_se = TriDet(**cfg['model'])
    params_se = sum(p.numel() for p in model_se.parameters())
    print(f'TriDet(att=SE)  params: {params_se:,}')
    print(f'Param increase: {params_se - params_off:,} ({(params_se - params_off) / params_off * 100:.3f}%)')

    # Test with ECA attention
    print('--- With ECA attention (fusion) ---')
    cfg['model']['att_type'] = 'ECA'
    cfg['model']['att_kernel_size'] = 3
    model_eca = TriDet(**cfg['model'])
    params_eca = sum(p.numel() for p in model_eca.parameters())
    print(f'TriDet(att=ECA) params: {params_eca:,}')
    print(f'Param increase: {params_eca - params_off:,} ({(params_eca - params_off) / params_off * 100:.3f}%)')

    # Forward pass test
    print('--- Forward pass test ---')
    model_se.eval()
    video_list = [{
        'video_id': 'test_0',
        'feats': torch.randn(2048, 200),
        'segments': None,
        'labels': None,
        'fps': 25.0,
        'duration': 200 * 16 / 25.0,
        'feat_stride': 16,
        'feat_num_frames': 32,
    }]
    with torch.no_grad():
        results = model_se(video_list)
    print(f'Inference results: {len(results)} videos, '
          f'segs={results[0]["segments"].shape}, scores={results[0]["scores"].shape}')


if __name__ == '__main__':
    test_attention_layers()
    test_sgpblock_default()
    test_sgpblock_se()
    test_sgpblock_eca()
    test_sgpblock_mlp_position()
    test_deterministic_without_att()
    test_sgpbackbone()
    test_full_tridet()
    print('\n=== All tests passed! ===')

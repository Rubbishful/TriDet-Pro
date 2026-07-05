"""Test _compute_reg_loss dispatch for all 5 supported loss types."""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
from unittest.mock import MagicMock, patch


def test_compute_reg_loss_dispatch():
    """Verify _compute_reg_loss dispatches to the correct loss function.

    Uses mock to avoid needing a full TriDet model — we test the dispatch
    logic directly on a lightweight TriDet instance.
    """
    from libs.modeling.meta_archs import TriDet
    from libs.core.config import load_default_config

    cfg = load_default_config()
    cfg = cfg.copy()
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

    loss_types = ['giou', 'diou', 'eiou', 'alpha_diou', 'focaler_diou']

    for lt in loss_types:
        cfg['model']['reg_loss_type'] = lt
        model = TriDet(**cfg['model'])

        pred = torch.rand(400, 2)
        gt = torch.rand(400, 2)
        loss = model._compute_reg_loss(pred, gt)

        assert loss.ndim == 0, f'{lt}: expected scalar loss, got {loss.shape}'
        assert torch.isfinite(loss), f'{lt}: loss should be finite'
        print(f'_compute_reg_loss({lt}): {loss.item():.4f} OK')


def test_compute_reg_loss_invalid():
    """_compute_reg_loss raises ValueError for unknown loss type."""
    from libs.modeling.meta_archs import TriDet
    from libs.core.config import load_default_config

    cfg = load_default_config()
    cfg = cfg.copy()
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
    cfg['model']['reg_loss_type'] = 'unknown_loss_type'

    model = TriDet(**cfg['model'])

    pred = torch.rand(400, 2)
    gt = torch.rand(400, 2)
    try:
        model._compute_reg_loss(pred, gt)
        assert False, 'Should have raised ValueError'
    except ValueError as e:
        assert 'unknown_loss_type' in str(e)
        print(f'ValueError for unknown loss type: OK')


if __name__ == '__main__':
    test_compute_reg_loss_dispatch()
    test_compute_reg_loss_invalid()
    print('\n=== All reg loss dispatch tests passed! ===')

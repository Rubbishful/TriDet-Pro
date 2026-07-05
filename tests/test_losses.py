"""Smoke tests for all loss functions in libs.modeling.losses."""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
from libs.modeling.losses import (
    sigmoid_focal_loss,
    ctr_giou_loss_1d,
    ctr_diou_loss_1d,
    ctr_eiou_loss_1d,
    ctr_alpha_diou_loss_1d,
    ctr_focaler_diou_loss_1d,
    quality_focal_loss,
)


def test_sigmoid_focal_loss():
    """sigmoid_focal_loss: shape & no NaN."""
    pred = torch.randn(4, 100, 20)
    target = torch.rand(4, 100, 20)
    loss = sigmoid_focal_loss(pred, target, reduction='mean')
    assert loss.ndim == 0, f'Expected scalar, got {loss.shape}'
    assert torch.isfinite(loss), 'Loss should be finite'
    print(f'sigmoid_focal_loss (mean): {loss.item():.4f} OK')

    loss_none = sigmoid_focal_loss(pred, target, reduction='none')
    assert loss_none.shape == pred.shape
    print(f'sigmoid_focal_loss (none): shape {loss_none.shape} OK')


def _test_reg_loss(fn, name):
    """Helper: test a regression loss with valid shapes and finite output."""
    pred = torch.randn(4, 200, 2)
    gt = torch.randn(4, 200, 2)
    loss = fn(pred, gt, reduction='sum')
    assert loss.ndim == 0, f'{name}: expected scalar, got {loss.shape}'
    assert torch.isfinite(loss), f'{name}: loss should be finite'
    print(f'{name}: {loss.item():.4f} OK')

    loss_none = fn(pred, gt, reduction='none')
    assert loss_none.shape == (4, 200), f'{name}: wrong none shape {loss_none.shape}'
    print(f'{name} (none): shape {loss_none.shape} OK')


def test_ctr_giou_loss_1d():
    _test_reg_loss(ctr_giou_loss_1d, 'ctr_giou_loss_1d')


def test_ctr_diou_loss_1d():
    _test_reg_loss(ctr_diou_loss_1d, 'ctr_diou_loss_1d')


def test_ctr_eiou_loss_1d():
    _test_reg_loss(ctr_eiou_loss_1d, 'ctr_eiou_loss_1d')


def test_ctr_alpha_diou_loss_1d():
    pred = torch.randn(4, 200, 2)
    gt = torch.randn(4, 200, 2)
    loss = ctr_alpha_diou_loss_1d(pred, gt, reduction='sum', alpha=3.0)
    assert loss.ndim == 0 and torch.isfinite(loss)
    print(f'ctr_alpha_diou_loss_1d (alpha=3): {loss.item():.4f} OK')


def test_ctr_focaler_diou_loss_1d():
    pred = torch.randn(4, 200, 2)
    gt = torch.randn(4, 200, 2)
    loss = ctr_focaler_diou_loss_1d(pred, gt, reduction='sum', d=0.0, u=0.95)
    assert loss.ndim == 0 and torch.isfinite(loss)
    print(f'ctr_focaler_diou_loss_1d: {loss.item():.4f} OK')


def test_quality_focal_loss():
    pred = torch.randn(4, 100, 20)
    target = torch.rand(4, 100, 20)
    loss = quality_focal_loss(pred, target, reduction='mean', beta=2.0)
    assert loss.ndim == 0 and torch.isfinite(loss)
    print(f'quality_focal_loss: {loss.item():.4f} OK')


def test_ctr_giou_perfect_match():
    """GIoU should be ~0 when pred == gt."""
    x = torch.randn(4, 100, 2)
    loss = ctr_giou_loss_1d(x, x, reduction='mean')
    assert loss.item() < 1e-5, f'GIoU for identical tensors should be ~0, got {loss.item()}'
    print(f'ctr_giou_loss_1d (identical): {loss.item():.10f} OK')


if __name__ == '__main__':
    test_sigmoid_focal_loss()
    test_ctr_giou_loss_1d()
    test_ctr_diou_loss_1d()
    test_ctr_eiou_loss_1d()
    test_ctr_alpha_diou_loss_1d()
    test_ctr_focaler_diou_loss_1d()
    test_quality_focal_loss()
    test_ctr_giou_perfect_match()
    print('\n=== All loss tests passed! ===')

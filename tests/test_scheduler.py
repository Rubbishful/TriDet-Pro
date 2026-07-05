"""Smoke tests for make_scheduler with warmup and non-warmup paths."""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import torch.optim as optim
from libs.utils.train_utils import make_scheduler


def _make_optimizer():
    """Create a dummy optimizer for scheduler testing."""
    model = torch.nn.Linear(10, 2)
    return optim.SGD(model.parameters(), lr=0.01)


def test_scheduler_cosine_warmup():
    """Cosine annealing with linear warmup."""
    opt = _make_optimizer()
    opt_cfg = {
        'warmup': True,
        'epochs': 30,
        'warmup_epochs': 5,
        'eta_min': 1e-6,
        'schedule_type': 'cosine',
    }
    sched = make_scheduler(opt_cfg, opt, num_iters_per_epoch=100, last_epoch=-1)
    assert sched is not None
    # Step a few iterations
    for _ in range(10):
        opt.step()
        sched.step()
    lr = opt.param_groups[0]['lr']
    assert lr > 0, f'LR should be positive, got {lr}'
    print(f'cosine+warmup: LR after 10 steps = {lr:.6f} OK')


def test_scheduler_multistep_warmup():
    """MultiStepLR with linear warmup."""
    opt = _make_optimizer()
    opt_cfg = {
        'warmup': True,
        'epochs': 30,
        'warmup_epochs': 5,
        'eta_min': 1e-6,
        'schedule_type': 'multistep',
        'schedule_steps': [10, 20],
        'schedule_gamma': 0.1,
    }
    sched = make_scheduler(opt_cfg, opt, num_iters_per_epoch=100, last_epoch=-1)
    for _ in range(10):
        opt.step()
        sched.step()
    lr = opt.param_groups[0]['lr']
    assert lr > 0
    print(f'multistep+warmup: LR after 10 steps = {lr:.6f} OK')


def test_scheduler_cosine_no_warmup():
    """Cosine annealing without warmup."""
    opt = _make_optimizer()
    opt_cfg = {
        'warmup': False,
        'epochs': 30,
        'warmup_epochs': 0,
        'eta_min': 1e-6,
        'schedule_type': 'cosine',
    }
    sched = make_scheduler(opt_cfg, opt, num_iters_per_epoch=100, last_epoch=-1)
    for _ in range(10):
        opt.step()
        sched.step()
    lr = opt.param_groups[0]['lr']
    assert lr > 0
    print(f'cosine (no warmup): LR after 10 steps = {lr:.6f} OK')


def test_scheduler_multistep_no_warmup():
    """MultiStepLR without warmup."""
    opt = _make_optimizer()
    opt_cfg = {
        'warmup': False,
        'epochs': 30,
        'warmup_epochs': 0,
        'eta_min': 1e-6,
        'schedule_type': 'multistep',
        'schedule_steps': [10, 20],
        'schedule_gamma': 0.1,
    }
    sched = make_scheduler(opt_cfg, opt, num_iters_per_epoch=100, last_epoch=-1)
    for _ in range(10):
        opt.step()
        sched.step()
    lr = opt.param_groups[0]['lr']
    assert lr > 0
    print(f'multistep (no warmup): LR after 10 steps = {lr:.6f} OK')


def test_scheduler_invalid_type():
    """Unsupported scheduler type should raise TypeError."""
    opt = _make_optimizer()
    opt_cfg = {
        'warmup': False,
        'epochs': 30,
        'warmup_epochs': 0,
        'eta_min': 1e-6,
        'schedule_type': 'invalid_type',
    }
    try:
        make_scheduler(opt_cfg, opt, num_iters_per_epoch=100)
        assert False, 'Should have raised TypeError'
    except TypeError:
        print('TypeError for invalid schedule_type: OK')


if __name__ == '__main__':
    test_scheduler_cosine_warmup()
    test_scheduler_multistep_warmup()
    test_scheduler_cosine_no_warmup()
    test_scheduler_multistep_no_warmup()
    test_scheduler_invalid_type()
    print('\n=== All scheduler tests passed! ===')

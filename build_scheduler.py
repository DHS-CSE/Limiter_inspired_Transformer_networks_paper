from __future__ import annotations
import torch

def build_scheduler(optimizer, args):
    name = getattr(args, 'scheduler', 'none').lower()
    if name in {'none', 'fixed', 'constant'}:
        return None
    elif name == 'rba':
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: 0.9 ** (epoch / 5000.0))
    elif name == 'brdr':
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=750, gamma=0.99)
    raise ValueError(f'Unknown scheduler: {name}')

def step_scheduler(scheduler, metric=None):
    if scheduler is None:
        return
    if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
        if metric is None:
            raise ValueError('ReduceLROnPlateau requires metric, e.g. total_loss.item().')
        scheduler.step(metric)
    else:
        scheduler.step()

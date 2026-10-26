from __future__ import annotations
import torch
import json
import time
from pathlib import Path
from typing import Any
import math
import argparse
import numpy as np
import matplotlib.pyplot as plt
from models.factory import build_adapter
from reference.reference_burgers2d import burgers2d_exact_solution
from reference.reference_guckenheimer2d_dirichlet import guckenheimer2d_exact_solution
from scipy.stats import qmc

def set_seed(seed: int, deterministic: bool=True):
    import os
    import random
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def current_lr(optimizer):
    return optimizer.param_groups[0]['lr']

def get_source_term(coords: torch.Tensor) -> torch.Tensor:
    x = coords[..., 0:1]
    return torch.zeros_like(x)

def u0_func(x: torch.Tensor) -> torch.Tensor:
    return x ** 2 * torch.cos(torch.pi * x)

def format_time(seconds):
    seconds = int(seconds)
    h = seconds // 3600
    m = seconds % 3600 // 60
    s = seconds % 60
    return f'{h:02d}:{m:02d}:{s:02d}'

def print_training_status(seed, epoch, num_epochs, loss, best_loss, rel_l2, loss_dict, lr, best_epoch, start_time):
    elapsed = time.time() - start_time
    if math.isnan(rel_l2):
        rel_l2_str = 'N/A'
    else:
        rel_l2_str = f'{rel_l2:.3e}'
    print('=' * 60)
    print(f'Seed        : {seed}')
    print(f'Epoch       : {epoch} / {num_epochs}')
    print('-' * 60)
    print(f'Loss        : {loss:.3e}')
    print(f'Best Loss   : {best_loss:.3e}')
    print(f'Rel L2      : {rel_l2_str}')
    print('-' * 60)
    for name, value in loss_dict.items():
        if name == 'total':
            continue
        print(f'{name.upper():<12}: {float(value):.3e}')
    print('-' * 60)
    print(f'LR          : {lr:.3e}')
    print(f'Best Epoch  : {best_epoch}')
    print(f'Elapsed     : {format_time(elapsed)}')
    print('=' * 60)

def make_allen_cahn_data(h_x: float, h_t: float, device: torch.device, batch_size: int=1) -> dict[str, torch.Tensor | float]:
    B = batch_size
    eps = 1e-08
    nx_inner = int(2 / h_x) + 1
    nt = int(1 / h_t) + 1
    x_inner = torch.linspace(-1.0, 1.0, nx_inner, device=device)
    x = torch.cat([torch.tensor([-1.0 - h_x], device=device), x_inner, torch.tensor([1.0 + h_x], device=device)])
    t = torch.linspace(0.0, 1.0, nt, device=device)
    X, T = torch.meshgrid(x, t, indexing='ij')
    X_inner, T_inner = torch.meshgrid(x_inner, t, indexing='ij')
    coords = torch.stack([X, T], dim=-1)
    coords_inner = torch.stack([X_inner, T_inner], dim=-1)
    domain_coords = coords.reshape(1, -1, 2)
    domain_coords_inner = coords_inner.reshape(1, -1, 2)
    domain_f = get_source_term(domain_coords)
    domain_f_inner = get_source_term(domain_coords_inner)
    x_all = domain_coords[..., 0]
    t_all = domain_coords[..., 1]
    mask_pde = ((-1.0 <= x_all) & (x_all <= 1.0)).squeeze(0)
    mask_boundary = ((torch.abs(x_all + 1.0) < eps) | (torch.abs(x_all - 1.0) < eps)).squeeze(0)
    mask_initial = (torch.abs(t_all - 0.0) < eps).squeeze(0)
    mask_initial_inner = ((torch.abs(x_all - (-1.0 - h_x)) > eps) & (torch.abs(x_all - (1.0 + h_x)) > eps) & (torch.abs(t_all - 0.0) < eps)).squeeze(0)
    bnd_coords = domain_coords[:, mask_boundary, :]
    ini_coords = domain_coords[:, mask_initial, :]
    ini_coords_inner = domain_coords[:, mask_initial_inner, :]
    bnd_values = torch.zeros(B, bnd_coords.shape[1], 1, device=device)
    ini_values = u0_func(ini_coords_inner[..., 0:1])
    mask_left_boundary_2step_ghost = ((torch.abs(x_all - (-1.0 - h_x)) < eps) | (torch.abs(x_all + 1.0) < eps) | (torch.abs(x_all - (-1.0 + h_x)) < eps)).squeeze(0)
    mask_right_boundary_2step_ghost = ((torch.abs(x_all - (1.0 - h_x)) < eps) | (torch.abs(x_all - 1.0) < eps) | (torch.abs(x_all - (1.0 + h_x)) < eps)).squeeze(0)
    left_bnd_coords_2step = domain_coords[:, mask_left_boundary_2step_ghost, :]
    right_bnd_coords_2step = domain_coords[:, mask_right_boundary_2step_ghost, :]
    bnd_coords_2step = torch.cat((left_bnd_coords_2step, right_bnd_coords_2step), dim=0)
    bnd_x = bnd_coords_2step[..., 0]
    boundary_mask = ((torch.abs(bnd_x + 1.0) < eps) | (torch.abs(bnd_x - 1.0) < eps)).squeeze(0)
    left_x = left_bnd_coords_2step[..., 0]
    right_x = right_bnd_coords_2step[..., 0]
    left_boundary_mask = (torch.abs(left_x + 1.0) < eps).squeeze(0)
    right_boundary_mask = (torch.abs(right_x - 1.0) < eps).squeeze(0)
    left_bnd_points = left_bnd_coords_2step[:, left_boundary_mask, :]
    right_bnd_points = right_bnd_coords_2step[:, right_boundary_mask, :]
    x_ini = ini_coords[..., 0]
    inner_initial_mask = ((-1.0 <= x_ini) & (x_ini <= 1.0)).squeeze(0)
    return {'h_x': h_x, 'h_t': h_t, 'domain_coords': domain_coords, 'domain_coords_inner': domain_coords_inner, 'domain_f': domain_f, 'domain_f_inner': domain_f_inner, 'bnd_coords': bnd_coords, 'bnd_values': bnd_values, 'ini_coords': ini_coords, 'ini_coords_inner': ini_coords_inner, 'ini_values': ini_values, 'bnd_coords_2step': bnd_coords_2step, 'left_bnd_coords_2step': left_bnd_coords_2step, 'right_bnd_coords_2step': right_bnd_coords_2step, 'left_bnd_coords': left_bnd_points, 'right_bnd_coords': right_bnd_points, 'pde_mask': mask_pde, 'boundary_inner_mask': boundary_mask, 'left_boundary_mask': left_boundary_mask, 'right_boundary_mask': right_boundary_mask, 'initial_mask': mask_initial, 'initial_inner_mask': inner_initial_mask}

def make_burgers_data(h_x, h_t, device, batch_size=1):
    B = batch_size
    eps = 1e-08
    nx = int(2 / h_x) + 1
    nt = int(1 / h_t) + 1
    x = torch.linspace(0, 2, nx, device=device)
    t = torch.linspace(0, 1, nt, device=device)
    x_ghost_plus = torch.cat((torch.tensor([-h_x], device=device), x, torch.tensor([2 + h_x], device=device)), dim=-1)
    X, T = torch.meshgrid(x_ghost_plus, t, indexing='ij')
    coords = torch.stack([X, T], dim=-1)
    domain_coords = coords.reshape(1, -1, 2)
    x_all = domain_coords[..., 0]
    t_all = domain_coords[..., 1]
    pde_mask = ((0.0 <= x_all) & (x_all <= 2.0)).squeeze(0)
    domain_coords_inner = domain_coords[:, pde_mask, :]
    domain_f = torch.zeros(B, domain_coords.shape[1], 1, device=device)
    domain_f_inner = torch.zeros(B, domain_coords_inner.shape[1], 1, device=device)
    boundary_mask = (torch.abs(x_all) < eps).squeeze(0)
    bnd_coords = domain_coords[:, boundary_mask, :]
    bnd_values = torch.full((B, bnd_coords.shape[1], 1), 2.5, device=device)
    initial_mask = (torch.abs(t_all) < eps).squeeze(0)
    initial_inner_mask = ((torch.abs(t_all) < eps) & ((0.0 < x_all) & (x_all <= 2.0))).squeeze(0)
    ini_coords = domain_coords[:, initial_mask, :]
    x_ini = ini_coords[..., 0]
    initial_inner_mask = ((0.0 < x_ini) & (x_ini <= 2.0)).squeeze(0)
    ini_coords_inner = ini_coords[:, initial_inner_mask, :]
    ini_values = torch.where(ini_coords_inner[..., 0:1] <= 0.5, 1.5, 0.5)
    boundary_mask_2step = ((torch.abs(x_all + h_x) < eps) | (torch.abs(x_all) < eps) | (torch.abs(x_all - h_x) < eps)).squeeze(0)
    bnd_coords_2step = domain_coords[:, boundary_mask_2step, :]
    bnd_x = bnd_coords_2step[..., 0]
    boundary_inner_mask = (torch.abs(bnd_x) < eps).squeeze(0)
    bnd_points = bnd_coords_2step[:, boundary_inner_mask, :]
    return {'h_x': h_x, 'h_t': h_t, 'domain_coords': domain_coords, 'domain_coords_inner': domain_coords_inner, 'domain_f': domain_f, 'domain_f_inner': domain_f_inner, 'bnd_coords': bnd_coords, 'bnd_coords_2step': bnd_coords_2step, 'bnd_values': bnd_values, 'ini_coords': ini_coords, 'ini_coords_inner': ini_coords_inner, 'ini_values': ini_values, 'pde_mask': pde_mask, 'boundary_inner_mask': boundary_inner_mask, 'boundary_mask_2step': boundary_mask_2step, 'initial_mask': initial_mask, 'initial_inner_mask': initial_inner_mask}

def make_burgers_lhs_random_data(h_x: float, h_t: float, device: torch.device | str, batch_size: int=1, num_random_x_points: int=127, seed: Optional[int]=None, dtype: torch.dtype=torch.float32):
    device = torch.device(device)
    if h_x <= 0.0:
        raise ValueError(f'h_x must be positive, got {h_x}.')
    if h_t <= 0.0:
        raise ValueError(f'h_t must be positive, got {h_t}.')
    if batch_size < 1:
        raise ValueError(f'batch_size must be >= 1, got {batch_size}.')
    if num_random_x_points < 1:
        raise ValueError('num_random_x_points must be positive.')
    B = batch_size
    nt = int(round(1.0 / h_t)) + 1
    t = torch.linspace(0.0, 1.0, nt, device=device, dtype=dtype)
    sampler = qmc.LatinHypercube(d=nt, scramble=True, seed=seed)
    x_unit_np = sampler.random(n=num_random_x_points).T
    boundary_gap = h_x
    x_inner_np = boundary_gap + (2.0 - boundary_gap) * x_unit_np
    x_random = torch.as_tensor(x_inner_np, device=device, dtype=dtype)
    fixed_x_values = torch.tensor([0.0, 2.0], device=device, dtype=dtype)
    fixed_x = fixed_x_values.view(1, 2).expand(nt, -1)
    x_all = torch.cat([x_random, fixed_x], dim=1)
    x_all = torch.sort(x_all, dim=1, stable=True).values
    nx_total = x_all.shape[1]
    x_all = x_all.transpose(0, 1).contiguous()
    t_all = t.unsqueeze(0).expand(nx_total, nt)
    coords_structured_base = torch.stack([x_all, t_all], dim=-1)
    domain_coords_structured = coords_structured_base.unsqueeze(0).expand(B, -1, -1, -1).clone()
    domain_coords = domain_coords_structured.reshape(B, nx_total * nt, 2)
    x_flat = domain_coords[0, :, 0]
    t_flat = domain_coords[0, :, 1]
    pde_mask = 0.0 < x_flat
    boundary_mask = x_flat == 0.0
    bnd_coords = domain_coords[:, boundary_mask, :]
    bnd_x = bnd_coords[..., 0]
    boundary_inner_mask = bnd_x == 0.0
    right_boundary_mask = x_flat == 2.0
    right_bnd_coords = domain_coords[:, right_boundary_mask, :]
    initial_mask = t_flat == 0.0
    ini_coords = domain_coords[:, initial_mask, :]
    ini_x = ini_coords[0, :, 0]
    initial_inner_mask = (ini_x > 0.0) & (ini_x <= 2.0)
    ini_coords_inner = ini_coords[:, initial_inner_mask, :]
    domain_coords_inner = domain_coords[:, pde_mask, :]
    domain_f = torch.zeros(B, domain_coords.shape[1], 1, device=device, dtype=dtype)
    domain_f_inner = torch.zeros(B, domain_coords_inner.shape[1], 1, device=device, dtype=dtype)
    bnd_values = torch.full((B, bnd_coords.shape[1], 1), 2.5, device=device, dtype=dtype)
    ini_values = torch.where(ini_coords_inner[..., 0:1] <= 0.5, torch.full_like(ini_coords_inner[..., 0:1], 1.5), torch.full_like(ini_coords_inner[..., 0:1], 0.5))
    expected_total_points = nx_total * nt
    if domain_coords.shape[1] != expected_total_points:
        raise RuntimeError(f'Unexpected flattened coordinate count: {domain_coords.shape[1]} != {expected_total_points}.')
    if ini_coords_inner.shape[1] != ini_values.shape[1]:
        raise RuntimeError(f'Initial coordinate/value count mismatch: {ini_coords_inner.shape[1]} != {ini_values.shape[1]}.')
    if bnd_coords.shape[1] != bnd_values.shape[1]:
        raise RuntimeError(f'Boundary coordinate/value count mismatch: {bnd_coords.shape[1]} != {bnd_values.shape[1]}.')
    return {'h_x': h_x, 'h_t': h_t, 'nt': nt, 'nx_random': num_random_x_points, 'nx_total': nx_total, 'grid_shape': (nx_total, nt), 'domain_coords_structured': domain_coords_structured, 'domain_coords': domain_coords, 'domain_coords_inner': domain_coords_inner, 'domain_f': domain_f, 'domain_f_inner': domain_f_inner, 'bnd_coords': bnd_coords, 'right_bnd_coords': right_bnd_coords, 'bnd_values': bnd_values, 'ini_coords': ini_coords, 'ini_coords_inner': ini_coords_inner, 'ini_values': ini_values, 'pde_mask': pde_mask, 'boundary_mask': boundary_mask, 'boundary_inner_mask': boundary_inner_mask, 'right_boundary_mask': right_boundary_mask, 'initial_mask': initial_mask, 'initial_inner_mask': initial_inner_mask}

def make_burgers2d_data(h_x, h_y, h_t, device, u_left=1.0, u_middle=0.6, u_right=0.0, discontinuity_distance=0.8, batch_size=1) -> dict[str, Any]:
    device = torch.device(device)
    dtype = torch.float32
    eps = 1e-08
    nx = int(round(1.0 / h_x)) + 1
    ny = int(round(1.0 / h_y)) + 1
    nt = int(round(1.0 / h_t)) + 1
    x_inner = torch.linspace(0.0, 1.0, nx, device=device, dtype=dtype)
    y_inner = torch.linspace(0.0, 1.0, ny, device=device, dtype=dtype)
    t = torch.linspace(0.0, 1.0, nt, device=device, dtype=dtype)
    x = torch.cat([x_inner.new_tensor([-h_x]), x_inner, x_inner.new_tensor([1.0 + h_x])], dim=0)
    y = torch.cat([y_inner.new_tensor([-h_y]), y_inner, y_inner.new_tensor([1.0 + h_y])], dim=0)
    X, Y, T = torch.meshgrid(x, y, t, indexing='ij')
    domain_coords = torch.stack([X, Y, T], dim=-1).reshape(1, -1, 3)
    X_inner, Y_inner, T_inner = torch.meshgrid(x_inner, y_inner, t, indexing='ij')
    domain_coords_inner = torch.stack([X_inner, Y_inner, T_inner], dim=-1).reshape(1, -1, 3)
    x_all = domain_coords[..., 0]
    y_all = domain_coords[..., 1]
    t_all = domain_coords[..., 2]
    pde_mask = ((x_all >= -eps) & (x_all <= 1.0 + eps) & (y_all >= -eps) & (y_all <= 1.0 + eps)).squeeze(0)
    if int(pde_mask.sum()) != domain_coords_inner.shape[1]:
        raise RuntimeError(f'pde_mask and domain_coords_inner have inconsistent sizes: {int(pde_mask.sum())} vs {domain_coords_inner.shape[1]}.')
    domain_f = torch.zeros_like(domain_coords[..., 0:1])
    domain_f_inner = torch.zeros_like(domain_coords_inner[..., 0:1])
    x_is_0 = torch.isclose(x_all, x_all.new_tensor(0.0), atol=eps, rtol=0.0)
    x_is_1 = torch.isclose(x_all, x_all.new_tensor(1.0), atol=eps, rtol=0.0)
    y_is_0 = torch.isclose(y_all, y_all.new_tensor(0.0), atol=eps, rtol=0.0)
    y_is_1 = torch.isclose(y_all, y_all.new_tensor(1.0), atol=eps, rtol=0.0)
    boundary_mask = ((x_is_0 | x_is_1 | y_is_0 | y_is_1) & (x_all >= -eps) & (x_all <= 1.0 + eps) & (y_all >= -eps) & (y_all <= 1.0 + eps)).squeeze(0)
    boundary_coords = domain_coords[:, boundary_mask, :]
    boundary_values = burgers2d_exact_solution(boundary_coords, u_left=u_left, u_middle=u_middle, u_right=u_right, discontinuity_distance=discontinuity_distance)
    initial_mask = torch.isclose(t_all, t_all.new_tensor(0.0), atol=eps, rtol=0.0).squeeze(0)
    initial_coords = domain_coords[:, initial_mask, :]
    x_initial = initial_coords[..., 0]
    y_initial = initial_coords[..., 1]
    initial_inner_mask = ((x_initial >= -eps) & (x_initial <= 1.0 + eps) & (y_initial >= -eps) & (y_initial <= 1.0 + eps)).squeeze(0)
    initial_coords_inner = initial_coords[:, initial_inner_mask, :]
    initial_values = burgers2d_exact_solution(initial_coords_inner, u_left=u_left, u_middle=u_middle, u_right=u_right, discontinuity_distance=discontinuity_distance)
    x_is_left_ghost = torch.isclose(x_all, x_all.new_tensor(-h_x), atol=eps, rtol=0.0)
    x_is_left_inner = torch.isclose(x_all, x_all.new_tensor(h_x), atol=eps, rtol=0.0)
    x_is_right_inner = torch.isclose(x_all, x_all.new_tensor(1.0 - h_x), atol=eps, rtol=0.0)
    x_is_right_ghost = torch.isclose(x_all, x_all.new_tensor(1.0 + h_x), atol=eps, rtol=0.0)
    y_is_bottom_ghost = torch.isclose(y_all, y_all.new_tensor(-h_y), atol=eps, rtol=0.0)
    y_is_bottom_inner = torch.isclose(y_all, y_all.new_tensor(h_y), atol=eps, rtol=0.0)
    y_is_top_inner = torch.isclose(y_all, y_all.new_tensor(1.0 - h_y), atol=eps, rtol=0.0)
    y_is_top_ghost = torch.isclose(y_all, y_all.new_tensor(1.0 + h_y), atol=eps, rtol=0.0)
    boundary_2step_mask = (x_is_left_ghost | x_is_0 | x_is_left_inner | x_is_right_inner | x_is_1 | x_is_right_ghost | y_is_bottom_ghost | y_is_0 | y_is_bottom_inner | y_is_top_inner | y_is_1 | y_is_top_ghost).squeeze(0)
    boundary_coords_2step = domain_coords[:, boundary_2step_mask, :]
    x_boundary_2step = boundary_coords_2step[..., 0]
    y_boundary_2step = boundary_coords_2step[..., 1]
    boundary_inner_mask = ((torch.isclose(x_boundary_2step, x_boundary_2step.new_tensor(0.0), atol=eps, rtol=0.0) | torch.isclose(x_boundary_2step, x_boundary_2step.new_tensor(1.0), atol=eps, rtol=0.0) | torch.isclose(y_boundary_2step, y_boundary_2step.new_tensor(0.0), atol=eps, rtol=0.0) | torch.isclose(y_boundary_2step, y_boundary_2step.new_tensor(1.0), atol=eps, rtol=0.0)) & (x_boundary_2step >= -eps) & (x_boundary_2step <= 1.0 + eps) & (y_boundary_2step >= -eps) & (y_boundary_2step <= 1.0 + eps)).squeeze(0)
    boundary_coords_inner = boundary_coords_2step[:, boundary_inner_mask, :]
    boundary_values_inner = burgers2d_exact_solution(boundary_coords_inner, u_left=u_left, u_middle=u_middle, u_right=u_right, discontinuity_distance=discontinuity_distance)
    x_boundary_inner = boundary_coords_inner[..., 0]
    y_boundary_inner = boundary_coords_inner[..., 1]
    left_boundary_mask = torch.isclose(x_boundary_inner, x_boundary_inner.new_tensor(0.0), atol=eps, rtol=0.0).squeeze(0)
    right_boundary_mask = torch.isclose(x_boundary_inner, x_boundary_inner.new_tensor(1.0), atol=eps, rtol=0.0).squeeze(0)
    bottom_boundary_mask = torch.isclose(y_boundary_inner, y_boundary_inner.new_tensor(0.0), atol=eps, rtol=0.0).squeeze(0)
    top_boundary_mask = torch.isclose(y_boundary_inner, y_boundary_inner.new_tensor(1.0), atol=eps, rtol=0.0).squeeze(0)
    reference_values = burgers2d_exact_solution(domain_coords_inner, u_left=u_left, u_middle=u_middle, u_right=u_right, discontinuity_distance=discontinuity_distance)
    data = {'domain_coords': domain_coords, 'domain_coords_inner': domain_coords_inner, 'domain_f': domain_f, 'domain_f_inner': domain_f_inner, 'pde_mask': pde_mask, 'boundary_coords': boundary_coords, 'boundary_values': boundary_values, 'boundary_mask': boundary_mask, 'boundary_coords_2step': boundary_coords_2step, 'boundary_coords_inner': boundary_coords_inner, 'boundary_values_inner': boundary_values_inner, 'boundary_inner_mask': boundary_inner_mask, 'left_boundary_mask': left_boundary_mask, 'right_boundary_mask': right_boundary_mask, 'bottom_boundary_mask': bottom_boundary_mask, 'top_boundary_mask': top_boundary_mask, 'initial_coords': initial_coords, 'initial_coords_inner': initial_coords_inner, 'initial_values': initial_values, 'initial_inner_mask': initial_inner_mask, 'bnd_coords': boundary_coords, 'bnd_values': boundary_values, 'bnd_coords_2step': boundary_coords_2step, 'ini_coords': initial_coords, 'ini_coords_inner': initial_coords_inner, 'ini_values': initial_values, 'h_x': h_x, 'h_y': h_y, 'h_t': h_t, 'reference_coords': domain_coords_inner, 'reference_values': reference_values}
    return data

def relative_l2_error(u_pred: torch.Tensor, u_ref: torch.Tensor | None) -> float:
    if u_ref is None:
        return float('nan')
    num = torch.linalg.norm((u_pred - u_ref).reshape(-1), ord=2)
    den = torch.linalg.norm(u_ref.reshape(-1), ord=2)
    return float((num / den).detach().cpu())

def evaluate_model_on_inner_grid(model, data, sharp_const=None) -> tuple[torch.Tensor, torch.Tensor]:
    adapter = build_adapter(model)
    model.eval()
    with torch.no_grad():
        coords = adapter.get_domain_coords(data)
        u_pred, aux_domain = adapter.forward(model, coords, data, sharp_const, data['pde_mask'])
        if hasattr(adapter, 'get_domain_inner_prediction'):
            u_pred = adapter.get_domain_inner_prediction(u_pred, data)
        u_inner = u_pred
    aux = {'u_inner': u_inner.detach()}
    if len(aux_domain) >= 8:
        aux.update({k: v.detach() for k, v in aux_domain.items() if torch.is_tensor(v)})
    return (data['domain_coords_inner'].detach(), u_pred.detach(), aux)

def save_checkpoint(path: Path, model: torch.nn.Module, optimizer: torch.optim.Optimizer, scheduler: Any, epoch: int, metric_value: float, args: argparse.Namespace, extra: dict[str, Any] | None=None):
    ckpt = {'epoch': epoch, 'metric_value': metric_value, 'model_state': model.state_dict(), 'optimizer_state': optimizer.state_dict() if optimizer is not None else None, 'scheduler_state': scheduler.state_dict() if scheduler is not None and hasattr(scheduler, 'state_dict') else None, 'args': vars(args), 'extra': extra or {}}
    torch.save(ckpt, path)

def save_prediction_plot(coords: torch.Tensor, u_pred: torch.Tensor, data: dict[str, Any], path: Path, title: str):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    h_x = float(data['h_x'])
    h_t = float(data['h_t'])
    nx_inner = int(2 / h_x) + 1
    nt = int(1 / h_t) + 1
    coords_cpu = coords.detach().cpu().reshape(nx_inner, nt, 2)
    u_cpu = u_pred.detach().cpu().reshape(nx_inner, nt)
    X = coords_cpu[..., 0].numpy()
    T = coords_cpu[..., 1].numpy()
    U = u_cpu.numpy()
    fig = plt.figure(figsize=(7.0, 4.8))
    ax = fig.add_subplot(111)
    cf = ax.contourf(X, T, U, levels=80)
    fig.colorbar(cf, ax=ax, label='u_pred')
    ax.set_xlabel('x')
    ax.set_ylabel('t')
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)

def save_prediction_burgers_plot(coords: torch.Tensor, u_pred: torch.Tensor, data: dict[str, Any], path: Path, title: str):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    h_x = float(data['h_x'])
    h_t = float(data['h_t'])
    nx = int(2 / h_x) + 1
    nt = int(1 / h_t) + 1
    coords_cpu = coords.detach().cpu().reshape(nx, nt, 2)
    u_cpu = u_pred.detach().cpu().reshape(nx, nt)
    X = coords_cpu[..., 0].numpy()
    T = coords_cpu[..., 1].numpy()
    U = u_cpu.numpy()
    fig1 = plt.figure(figsize=(6, 5), dpi=300)
    ax1 = fig1.add_subplot(111, projection='3d')
    cf1 = ax1.plot_surface(X, T, U, cmap='viridis', linewidth=0, antialiased=False)
    ax1.set_xlabel('x')
    ax1.set_ylabel('t')
    fig1.tight_layout()
    surf_path = path.with_name(path.stem + '_surf.png')
    fig1.savefig(surf_path, dpi=300)
    plt.close(fig1)
    fig2, ax2 = plt.subplots(figsize=(6, 5), dpi=300)
    im = ax2.imshow(U.T, origin='lower', extent=[0, 2, 0, 1], cmap='viridis', aspect='auto')
    ax2.set_box_aspect(1)
    ax2.set_xlabel('x')
    ax2.set_ylabel('t')
    fig2.tight_layout()
    heat_path = path.with_name(path.stem + '_heatmap.png')
    fig2.savefig(heat_path, bbox_inches='tight')
    plt.close(fig2)

def save_prediction_burgers2d_plot(coords: torch.Tensor, u_pred: torch.Tensor, data: dict[str, Any], path: Path, title: str):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    coords_flat = coords.detach().cpu().reshape(-1, 3)
    x_unique = torch.unique(coords_flat[:, 0], sorted=True)
    y_unique = torch.unique(coords_flat[:, 1], sorted=True)
    t_unique = torch.unique(coords_flat[:, 2], sorted=True)
    nx = len(x_unique)
    ny = len(y_unique)
    nt = len(t_unique)
    coords_cpu = coords.detach().cpu().reshape(nx, ny, nt, 3)
    u_cpu = u_pred.detach().cpu().reshape(nx, ny, nt)
    X = coords_cpu[..., 0].numpy()
    Y = coords_cpu[..., 1].numpy()
    U = u_cpu.numpy()
    save_dir = Path(path)
    save_dir.mkdir(parents=True, exist_ok=True)
    if nx != nt:
        time_indices = [0, nt // 3, 2 * nt // 3, nt - 1]
    else:
        time_indices = [0, nt // 4, nt // 2, 3 * nt // 4, nt - 1]
    for i in time_indices:
        fig, ax = plt.subplots(figsize=(6, 5), dpi=300)
        cf = ax.contourf(X[:, :, i], Y[:, :, i], U[:, :, i], levels=100, cmap='viridis')
        fig.colorbar(cf, ax=ax)
        t_value = float(t_unique[i])
        ax.set_title(f'{title} (t={t_value:.3f})')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_aspect('equal')
        fig.tight_layout()
        fig.savefig(save_dir / f'prediction_t{i:03d}.png', dpi=300, bbox_inches='tight')
        plt.close(fig)
        fig = plt.figure(figsize=(6, 5), dpi=300)
        ax = fig.add_subplot(111, projection='3d')
        surf = ax.plot_surface(X[:, :, i], Y[:, :, i], U[:, :, i], cmap='viridis', linewidth=0, antialiased=False)
        fig.colorbar(surf, ax=ax, shrink=0.6)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.invert_xaxis()
        ax.view_init(elev=32, azim=120)
        fig.tight_layout()
        fig.savefig(save_dir / f'prediction_surface_t{i:03d}.png', dpi=300, bbox_inches='tight')
        plt.close(fig)

def save_loss_plot(history: list[dict[str, Any]], path: Path, title: str):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    if len(history) == 0:
        return
    epochs = [r['epoch'] for r in history]
    fig = plt.figure(figsize=(7.0, 4.8))
    ax = fig.add_subplot(111)
    for key in ['total', 'pde', 'bc', 'bc_grad', 'ini']:
        if key in history[0]:
            ax.plot(epochs, [max(float(r[key]), 1e-30) for r in history], label=key)
    ax.set_yscale('log')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)

def save_loss_burgers_plot(history: list[dict[str, Any]], path: Path, title: str):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    if len(history) == 0:
        return
    epochs = [r['epoch'] for r in history]
    fig = plt.figure(figsize=(7.0, 4.8))
    ax = fig.add_subplot(111)
    for key in ['total', 'pde', 'bc', 'ini']:
        if key in history[0]:
            ax.plot(epochs, [max(float(r[key]), 1e-30) for r in history], label=key)
    ax.set_yscale('log')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)

def save_lr_plot(history: list[dict[str, Any]], path: Path, title: str):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    if len(history) == 0 or 'lr' not in history[0]:
        return
    epochs = [r['epoch'] for r in history]
    lrs = [float(r['lr']) for r in history]
    fig = plt.figure(figsize=(7.0, 4.0))
    ax = fig.add_subplot(111)
    ax.plot(epochs, lrs)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Learning rate')
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)

def save_error_curve(history: list[dict[str, Any]], path: Path, title: str):
    import math
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    rows = [r for r in history if 'relative_l2_error' in r and math.isfinite(float(r['relative_l2_error']))]
    if len(rows) == 0:
        return
    epochs = [r['epoch'] for r in rows]
    errors = [max(float(r['relative_l2_error']), 1e-30) for r in rows]
    fig = plt.figure(figsize=(7.0, 4.8))
    ax = fig.add_subplot(111)
    ax.plot(epochs, errors)
    ax.set_yscale('log')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Relative L2 error')
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)

def save_error_contour_plot(coords: torch.Tensor, u_pred: torch.Tensor, u_ref: torch.Tensor | None, data: dict[str, Any], path: Path, title: str):
    if u_ref is None:
        return
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    h_x = float(data['h_x'])
    h_t = float(data['h_t'])
    nx_inner = int(2 / h_x) + 1
    nt = int(1 / h_t) + 1
    coords_cpu = coords.detach().cpu().reshape(nx_inner, nt, 2)
    err_cpu = torch.abs(u_pred.detach().cpu().reshape(nx_inner, nt) - u_ref.detach().cpu().reshape(nx_inner, nt))
    X = coords_cpu[..., 0].numpy()
    T = coords_cpu[..., 1].numpy()
    E = err_cpu.numpy()
    fig = plt.figure(figsize=(6, 5))
    ax = fig.add_subplot(111)
    cf = ax.contourf(X, T, E, levels=80)
    fig.colorbar(cf, ax=ax, label='absolute error')
    ax.set_xlabel('x')
    ax.set_ylabel('t')
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)

def save_error_contour_burgers_plot(coords: torch.Tensor, u_pred: torch.Tensor, u_ref: torch.Tensor | None, data: dict[str, Any], path: Path, title: str):
    if u_ref is None:
        return
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    h_x = float(data['h_x'])
    h_t = float(data['h_t'])
    nx_inner = int(2 / h_x) + 1
    nt = int(1 / h_t) + 1
    coords_cpu = coords.detach().cpu().reshape(nx_inner, nt, 2)
    err_cpu = torch.abs(u_pred.detach().cpu().reshape(nx_inner, nt) - u_ref.detach().cpu().reshape(nx_inner, nt))
    X = coords_cpu[..., 0].numpy()
    T = coords_cpu[..., 1].numpy()
    E = err_cpu.numpy()
    fig = plt.figure(figsize=(6, 5), dpi=300)
    ax = fig.add_subplot(111)
    cf = ax.contourf(X, T, E, levels=80, cmap='RdBu')
    ax.set_box_aspect(1)
    ax.set_xlabel('x')
    ax.set_ylabel('t')
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)

def save_error_contour_burgers2d_plot(coords: torch.Tensor, u_pred: torch.Tensor, u_ref: torch.Tensor | None, data: dict[str, Any], path: Path, title: str):
    if u_ref is None:
        return
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    coords_flat = coords.detach().cpu().reshape(-1, 3)
    x_unique = torch.unique(coords_flat[:, 0], sorted=True)
    y_unique = torch.unique(coords_flat[:, 1], sorted=True)
    t_unique = torch.unique(coords_flat[:, 2], sorted=True)
    nx = len(x_unique)
    ny = len(y_unique)
    nt = len(t_unique)
    expected_points = nx * ny * nt
    actual_points = coords_flat.shape[0]
    if expected_points != actual_points:
        raise ValueError(f'Coordinates do not form a complete structured grid: nx={nx}, ny={ny}, nt={nt}, expected={expected_points}, actual={actual_points}')
    coords_cpu = coords_flat.reshape(nx, ny, nt, 3)
    u_pred_cpu = u_pred.detach().cpu().reshape(nx, ny, nt)
    u_ref_cpu = u_ref.detach().cpu().reshape(nx, ny, nt)
    X = coords_cpu[..., 0].numpy()
    Y = coords_cpu[..., 1].numpy()
    E = torch.abs(u_pred_cpu - u_ref_cpu).numpy()
    save_dir = path.with_suffix('')
    save_dir.mkdir(parents=True, exist_ok=True)
    if nx != nt:
        time_indices = [0, nt // 3, 2 * nt // 3, nt - 1]
    else:
        time_indices = [0, nt // 4, nt // 2, 3 * nt // 4, nt - 1]
    for i in time_indices:
        fig, ax = plt.subplots(figsize=(6, 5), dpi=300)
        cf = ax.contourf(X[:, :, i], Y[:, :, i], E[:, :, i], levels=100, cmap='RdBu')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_aspect('equal')
        fig.tight_layout()
        fig.savefig(save_dir / f'error_t{i:03d}.png', dpi=300, bbox_inches='tight')
        plt.close(fig)

def write_excel_report(excel_path: Path, summary_records: list[dict[str, Any]], history_records: list[dict[str, Any]], args: argparse.Namespace):
    try:
        import pandas as pd
    except ImportError:
        import csv
        csv_path = excel_path.with_suffix('.summary.csv')
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=sorted(summary_records[0].keys()))
            writer.writeheader()
            writer.writerows(summary_records)
        return
    summary_df = pd.DataFrame(summary_records)
    history_df = pd.DataFrame(history_records)
    config_df = pd.DataFrame([{'key': k, 'value': str(v)} for k, v in vars(args).items()])
    with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
        summary_df.to_excel(writer, index=False, sheet_name='summary')
        history_df.to_excel(writer, index=False, sheet_name='history')
        config_df.to_excel(writer, index=False, sheet_name='config')

def save_json(path: Path, obj: dict[str, Any]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)

def append_jsonl(path: Path, record: dict[str, Any]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'a', encoding='utf-8') as f:
        f.write(json.dumps(record, ensure_ascii=False) + '\n')

def save_history_jsonl(path: Path, history: list[dict[str, Any]]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        for rec in history:
            f.write(json.dumps(rec, ensure_ascii=False) + '\n')

def save_manifest(path: Path, args: argparse.Namespace, seed: int, status: str, epoch: int, best_loss: float, best_epoch: int, stop_reason: str, run_dir: Path, extra: dict[str, Any] | None=None):
    manifest = {'status': status, 'seed': seed, 'model_name': args.model_name, 'optimizer': args.optimizer, 'scheduler': args.scheduler, 'lr': args.lr, 'epoch': epoch, 'best_loss': best_loss, 'best_epoch': best_epoch, 'stop_reason': stop_reason, 'run_dir': str(run_dir), 'updated_at': time.strftime('%Y-%m-%d %H:%M:%S'), 'config': vars(args)}
    if extra:
        manifest.update(extra)
    save_json(path, manifest)

def save_analysis_tensors(save_path, epoch, aux, loss, rel_l2, lr, metadata=None, mode='epoch'):
    save_path = Path(save_path)
    if save_path.exists():
        analysis = torch.load(save_path)
    else:
        analysis = {'metadata': metadata or {}, 'epochs': {}, 'best': None, 'final': None}
    record = {'loss': float(loss), 'rel_l2': float(rel_l2), 'lr': float(lr)}
    for k, v in aux.items():
        if torch.is_tensor(v):
            record[k] = v.detach().cpu()
    if mode == 'epoch':
        analysis['epochs'][epoch] = record
    elif mode == 'best':
        record['epoch'] = epoch
        analysis['best'] = record
    elif mode == 'final':
        record['epoch'] = epoch
        analysis['final'] = record
    torch.save(analysis, save_path)

def count_parameters(model):
    total_params = sum((p.numel() for p in model.parameters()))
    trainable_params = sum((p.numel() for p in model.parameters() if p.requires_grad))
    return (total_params, trainable_params)

def make_guckenheimer2d_dirichlet_data(h_x: float, h_y: float, h_t: float, device, amplitude=0.15, d1=0.3, d2=0.8, batch_size: int=1, x_bounds=(0.0, 1.0), y_bounds=(0.0, 1.0), t_bounds=(0.0, 0.6)) -> dict[str, Any]:
    device = torch.device(device)
    dtype = torch.float32
    B = int(batch_size)
    eps = 1e-06
    xmin, xmax = map(float, x_bounds)
    ymin, ymax = map(float, y_bounds)
    tmin, tmax = map(float, t_bounds)
    if not (xmax > xmin and ymax > ymin and (tmax > tmin)):
        raise ValueError('Invalid Guckenheimer domain bounds.')
    if h_x <= 0.0 or h_y <= 0.0 or h_t <= 0.0:
        raise ValueError('h_x, h_y and h_t must all be positive.')
    nx = int(round((xmax - xmin) / h_x)) + 1
    ny = int(round((ymax - ymin) / h_y)) + 1
    nt = int(round((tmax - tmin) / h_t)) + 1
    if nx < 3 or ny < 3 or nt < 2:
        raise ValueError(f'Guckenheimer grid is too small for the slope-limiter stencil: nx={nx}, ny={ny}, nt={nt}.')
    x_inner = torch.linspace(xmin, xmax, nx, device=device, dtype=dtype)
    y_inner = torch.linspace(ymin, ymax, ny, device=device, dtype=dtype)
    t = torch.linspace(tmin, tmax, nt, device=device, dtype=dtype)
    dx = float((x_inner[1] - x_inner[0]).detach().cpu())
    dy = float((y_inner[1] - y_inner[0]).detach().cpu())
    x = torch.cat([x_inner.new_tensor([xmin - dx]), x_inner, x_inner.new_tensor([xmax + dx])])
    y = torch.cat([y_inner.new_tensor([ymin - dy]), y_inner, y_inner.new_tensor([ymax + dy])])
    X, Y, T = torch.meshgrid(x, y, t, indexing='ij')
    base = torch.stack([X, Y, T], dim=-1).reshape(1, -1, 3)
    domain_coords = base.expand(B, -1, -1).clone()
    Xi, Yi, Ti = torch.meshgrid(x_inner, y_inner, t, indexing='ij')
    inner_base = torch.stack([Xi, Yi, Ti], dim=-1).reshape(1, -1, 3)
    domain_coords_inner = inner_base.expand(B, -1, -1).clone()
    xa = base[..., 0]
    ya = base[..., 1]
    ta = base[..., 2]
    physical = (xa >= xmin - eps) & (xa <= xmax + eps) & (ya >= ymin - eps) & (ya <= ymax + eps)
    pde_mask = physical.squeeze(0)
    if int(pde_mask.sum()) != domain_coords_inner.shape[1]:
        raise RuntimeError(f'Guckenheimer pde_mask/domain_coords_inner mismatch: {int(pde_mask.sum())} vs {domain_coords_inner.shape[1]}.')
    domain_f = torch.zeros(B, base.shape[1], 1, device=device, dtype=dtype)
    domain_f_inner = torch.zeros(B, inner_base.shape[1], 1, device=device, dtype=dtype)
    xL = torch.isclose(xa, xa.new_tensor(xmin), atol=eps, rtol=0.0)
    xR = torch.isclose(xa, xa.new_tensor(xmax), atol=eps, rtol=0.0)
    yB = torch.isclose(ya, ya.new_tensor(ymin), atol=eps, rtol=0.0)
    yT = torch.isclose(ya, ya.new_tensor(ymax), atol=eps, rtol=0.0)
    boundary_mask = ((xL | xR | yB | yT) & physical).squeeze(0)
    boundary_coords = domain_coords[:, boundary_mask, :]
    boundary_values = guckenheimer2d_exact_solution(boundary_coords, amplitude, d1, d2)
    initial_mask = torch.isclose(ta, ta.new_tensor(tmin), atol=eps, rtol=0.0).squeeze(0)
    initial_coords = domain_coords[:, initial_mask, :]
    xi = initial_coords[..., 0]
    yi = initial_coords[..., 1]
    initial_inner_mask = ((xi >= xmin - eps) & (xi <= xmax + eps) & (yi >= ymin - eps) & (yi <= ymax + eps)).squeeze(0)
    initial_coords_inner = initial_coords[:, initial_inner_mask, :]
    initial_values = guckenheimer2d_exact_solution(initial_coords_inner, amplitude, d1, d2)
    xLm = torch.isclose(xa, xa.new_tensor(xmin - dx), atol=eps, rtol=0.0)
    xLp = torch.isclose(xa, xa.new_tensor(xmin + dx), atol=eps, rtol=0.0)
    xRm = torch.isclose(xa, xa.new_tensor(xmax - dx), atol=eps, rtol=0.0)
    xRp = torch.isclose(xa, xa.new_tensor(xmax + dx), atol=eps, rtol=0.0)
    yBm = torch.isclose(ya, ya.new_tensor(ymin - dy), atol=eps, rtol=0.0)
    yBp = torch.isclose(ya, ya.new_tensor(ymin + dy), atol=eps, rtol=0.0)
    yTm = torch.isclose(ya, ya.new_tensor(ymax - dy), atol=eps, rtol=0.0)
    yTp = torch.isclose(ya, ya.new_tensor(ymax + dy), atol=eps, rtol=0.0)
    boundary_2step_mask = (xLm | xL | xLp | xRm | xR | xRp | yBm | yB | yBp | yTm | yT | yTp).squeeze(0)
    boundary_coords_2step = domain_coords[:, boundary_2step_mask, :]
    xb = boundary_coords_2step[..., 0]
    yb = boundary_coords_2step[..., 1]
    xbL = torch.isclose(xb, xb.new_tensor(xmin), atol=eps, rtol=0.0)
    xbR = torch.isclose(xb, xb.new_tensor(xmax), atol=eps, rtol=0.0)
    ybB = torch.isclose(yb, yb.new_tensor(ymin), atol=eps, rtol=0.0)
    ybT = torch.isclose(yb, yb.new_tensor(ymax), atol=eps, rtol=0.0)
    boundary_inner_mask = ((xbL | xbR | ybB | ybT) & (xb >= xmin - eps) & (xb <= xmax + eps) & (yb >= ymin - eps) & (yb <= ymax + eps)).squeeze(0)
    boundary_coords_inner = boundary_coords_2step[:, boundary_inner_mask, :]
    boundary_values_inner = guckenheimer2d_exact_solution(boundary_coords_inner, amplitude, d1, d2)
    xbi = boundary_coords_inner[..., 0]
    ybi = boundary_coords_inner[..., 1]
    left_boundary_mask = torch.isclose(xbi, xbi.new_tensor(xmin), atol=eps, rtol=0.0).squeeze(0)
    right_boundary_mask = torch.isclose(xbi, xbi.new_tensor(xmax), atol=eps, rtol=0.0).squeeze(0)
    bottom_boundary_mask = torch.isclose(ybi, ybi.new_tensor(ymin), atol=eps, rtol=0.0).squeeze(0)
    top_boundary_mask = torch.isclose(ybi, ybi.new_tensor(ymax), atol=eps, rtol=0.0).squeeze(0)
    left_bnd_coords = boundary_coords_inner[:, left_boundary_mask, :]
    right_bnd_coords = boundary_coords_inner[:, right_boundary_mask, :]
    bottom_bnd_coords = boundary_coords_inner[:, bottom_boundary_mask, :]
    top_bnd_coords = boundary_coords_inner[:, top_boundary_mask, :]
    left_bnd_coords_2step = boundary_coords_2step[:, torch.isclose(boundary_coords_2step[..., 0], boundary_coords_2step.new_tensor(xmin), atol=eps, rtol=0.0).squeeze(0), :]
    right_bnd_coords_2step = boundary_coords_2step[:, torch.isclose(boundary_coords_2step[..., 0], boundary_coords_2step.new_tensor(xmax), atol=eps, rtol=0.0).squeeze(0), :]
    bottom_bnd_coords_2step = boundary_coords_2step[:, torch.isclose(boundary_coords_2step[..., 1], boundary_coords_2step.new_tensor(ymin), atol=eps, rtol=0.0).squeeze(0), :]
    top_bnd_coords_2step = boundary_coords_2step[:, torch.isclose(boundary_coords_2step[..., 1], boundary_coords_2step.new_tensor(ymax), atol=eps, rtol=0.0).squeeze(0), :]
    expected_perimeter = (2 * nx + 2 * ny - 4) * nt
    if boundary_coords_inner.shape[1] != expected_perimeter:
        raise RuntimeError(f'Guckenheimer physical boundary layout is incomplete: got {boundary_coords_inner.shape[1]} points, expected {expected_perimeter}.')
    for name, side_coords, tangential_axis in (('left', left_bnd_coords, 1), ('right', right_bnd_coords, 1), ('bottom', bottom_bnd_coords, 0), ('top', top_bnd_coords, 0)):
        n_tangent = torch.unique(side_coords[0, :, tangential_axis]).numel()
        n_time = torch.unique(side_coords[0, :, 2]).numel()
        if n_tangent < 2 or n_time != nt:
            raise RuntimeError(f'Invalid {name} boundary grid for slope limiter: n_tangent={n_tangent}, n_time={n_time}, nt={nt}.')
    return {'periodic': False, 'domain_coords': domain_coords, 'domain_coords_inner': domain_coords_inner, 'domain_f': domain_f, 'domain_f_inner': domain_f_inner, 'pde_mask': pde_mask, 'boundary_coords': boundary_coords, 'boundary_values': boundary_values, 'boundary_mask': boundary_mask, 'boundary_coords_2step': boundary_coords_2step, 'boundary_coords_inner': boundary_coords_inner, 'boundary_values_inner': boundary_values_inner, 'boundary_inner_mask': boundary_inner_mask, 'left_boundary_mask': left_boundary_mask, 'right_boundary_mask': right_boundary_mask, 'bottom_boundary_mask': bottom_boundary_mask, 'top_boundary_mask': top_boundary_mask, 'left_bnd_coords': left_bnd_coords, 'right_bnd_coords': right_bnd_coords, 'bottom_bnd_coords': bottom_bnd_coords, 'top_bnd_coords': top_bnd_coords, 'left_bnd_coords_2step': left_bnd_coords_2step, 'right_bnd_coords_2step': right_bnd_coords_2step, 'bottom_bnd_coords_2step': bottom_bnd_coords_2step, 'top_bnd_coords_2step': top_bnd_coords_2step, 'initial_coords': initial_coords, 'initial_coords_inner': initial_coords_inner, 'initial_values': initial_values, 'initial_inner_mask': initial_inner_mask, 'bnd_coords': boundary_coords, 'bnd_values': boundary_values, 'bnd_coords_2step': boundary_coords_2step, 'ini_coords': initial_coords, 'ini_coords_inner': initial_coords_inner, 'ini_values': initial_values, 'h_x': float(h_x), 'h_y': float(h_y), 'h_t': float(h_t), 'dx': dx, 'dy': dy, 'nx': nx, 'ny': ny, 'nt': nt}

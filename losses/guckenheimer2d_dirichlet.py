from __future__ import annotations
from typing import Any
import torch
import torch.nn as nn
from models.factory import build_adapter
from methods.sa_ntk import validate_loss_methods, adaptive_group_losses, residual_groups_for_weights

def calculate_loss_guckenheimer2d_dirichlet_ghost_sharp_analysis(model: nn.Module, data: dict[str, Any], sharp_const: float | None=None, rba=None, brdr=None, causal=None, update_weighting: bool=True, lambda_pde: float=1.0, lambda_bc: float=10.0, lambda_ini: float=5.0, sa=None, ntk=None, return_residual_graph: bool=False):
    validate_loss_methods(rba, brdr, causal, sa, ntk)
    if sum((w is not None for w in (rba, brdr, causal))) > 1:
        raise ValueError('Only one weighting method may be enabled.')
    adapter = build_adapter(model)
    domain_coords = adapter.get_domain_coords(data).clone().detach().requires_grad_(True)
    boundary_coords = adapter.get_boundary_coords(data).clone().detach().requires_grad_(True)
    initial_coords = adapter.get_initial_coords(data).clone().detach().requires_grad_(True)
    pde_mask = adapter.get_pde_mask(data)
    u_pred, aux = adapter.forward(model=model, coords=domain_coords, data=data, sharp_const=sharp_const, mask=data['pde_mask'], mode='domain')
    if u_pred.ndim == 2:
        u_pred = u_pred.unsqueeze(0)
    grads = torch.autograd.grad(u_pred, domain_coords, torch.ones_like(u_pred), create_graph=True, retain_graph=True)[0]
    ux = grads[..., 0:1][:, pde_mask, :]
    uy = grads[..., 1:2][:, pde_mask, :]
    ut = grads[..., 2:3][:, pde_mask, :]
    u = u_pred
    pde_residual = ut + u * ux + u.square() * uy - data['domain_f_inner']
    ub, _ = adapter.forward(model=model, coords=boundary_coords, data=data, sharp_const=sharp_const, mask=data['boundary_inner_mask'], mode='boundary')
    if ub.ndim == 2:
        ub = ub.unsqueeze(0)
    bc_residual = ub - data['boundary_values_inner']
    ui, _ = adapter.forward(model=model, coords=initial_coords, data=data, sharp_const=sharp_const, mask=data['initial_inner_mask'], mode='initial')
    if ui.ndim == 2:
        ui = ui.unsqueeze(0)
    ini_residual = ui - data['initial_values']
    weighting_aux = {}
    if sa is not None or ntk is not None:
        adaptive_losses, adaptive_aux = adaptive_group_losses({'pde': pde_residual, 'bc': bc_residual, 'ini': ini_residual}, data, sa, ntk, update_weighting)
        lp = adaptive_losses['pde']
        lb = adaptive_losses['bc']
        li = adaptive_losses['ini']
        weighting_aux.update(adaptive_aux)
    elif brdr is not None:
        groups = {'pde': pde_residual, 'bc': bc_residual, 'ini': ini_residual}
        weights = brdr.update(groups) if update_weighting else brdr.get_all()
        lp = torch.mean(weights['pde'] * pde_residual.square())
        lb = torch.mean(weights['bc'] * bc_residual.square())
        li = torch.mean(weights['ini'] * ini_residual.square())
    elif rba is not None:
        w = rba.update(pde_residual) if update_weighting else rba.get_current()
        lp = torch.mean((w * pde_residual).square())
        lb = bc_residual.square().mean()
        li = ini_residual.square().mean()
        weighting_aux['rba_pde_weight'] = w.detach()
    elif causal is not None:
        pde_time = domain_coords[:, pde_mask, 2:3]
        out = causal(pde_residual=pde_residual, pde_time_coords=pde_time, initial_residual=ini_residual, lambda_pde=lambda_pde, lambda_ini=lambda_ini)
        lp = out.pde_loss
        li = out.initial_loss
        lb = bc_residual.square().mean()
        weighting_aux.update({'causal_temporal_losses': out.temporal_losses.detach(), 'causal_temporal_weights': out.temporal_weights.detach(), 'causal_time_centers': out.time_centers.detach(), 'causal_min_weight': out.min_temporal_weight.detach(), 'causal_epsilon': pde_residual.new_tensor(causal.epsilon)})
    else:
        lp = pde_residual.square().mean()
        lb = bc_residual.square().mean()
        li = ini_residual.square().mean()
    total = lambda_pde * lp + lambda_bc * lb + lambda_ini * li
    if causal is not None:
        total = out.total_loss + lambda_bc * lb
    losses = {'pde': lp, 'bc': lb, 'ini': li, 'total': total}
    aux_out = {'pde_residual': pde_residual.detach(), 'bc_residual': bc_residual.detach(), 'ini_residual': ini_residual.detach(), **weighting_aux}
    if isinstance(aux, dict):
        aux_out.update(aux)
    if return_residual_graph:
        aux_out['adaptive_residual_groups'] = residual_groups_for_weights({'pde': pde_residual, 'bc': bc_residual, 'ini': ini_residual}, data, sa, ntk)
    return (total, losses, aux_out)

from __future__ import annotations
from typing import Any
import torch
import torch.nn as nn
from models.factory import build_adapter
from methods.sa_ntk import validate_loss_methods, adaptive_group_losses, residual_groups_for_weights

def calculate_loss_burgers2d_ghost_sharp_analysis(model: nn.Module, data: dict[str, Any], sharp_const: float | None=None, rba=None, brdr=None, causal=None, update_weighting: bool=True, lambda_pde: float=1.0, lambda_bc: float=10.0, lambda_ini: float=5.0, sa=None, ntk=None, return_residual_graph: bool=False) -> tuple[torch.Tensor, dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    validate_loss_methods(rba, brdr, causal, sa, ntk)
    number_of_weighting_methods = sum((weighting is not None for weighting in (rba, brdr, causal)))
    if number_of_weighting_methods > 1:
        raise ValueError('Only one weighting method can be enabled among RBA, BRDR, and Causal.')
    adapter = build_adapter(model)
    domain_coords = adapter.get_domain_coords(data).clone().detach().requires_grad_(True)
    boundary_coords = adapter.get_boundary_coords(data).clone().detach().requires_grad_(True)
    initial_coords = adapter.get_initial_coords(data).clone().detach().requires_grad_(True)
    pde_mask = adapter.get_pde_mask(data)
    u_pred_domain, aux_domain = adapter.forward(model=model, coords=domain_coords, data=data, sharp_const=sharp_const, mask=data['pde_mask'], mode='domain')
    if u_pred_domain.ndim == 2:
        u_pred_domain = u_pred_domain.unsqueeze(0)
    if u_pred_domain.shape[0] != domain_coords.shape[0] or u_pred_domain.shape[-1] != 1:
        raise ValueError(f'Unexpected domain prediction shape: {tuple(u_pred_domain.shape)}.')
    domain_gradients = torch.autograd.grad(outputs=u_pred_domain, inputs=domain_coords, grad_outputs=torch.ones_like(u_pred_domain), create_graph=True, retain_graph=True)[0]
    if domain_gradients.shape != domain_coords.shape:
        raise ValueError(f'Unexpected domain-gradient shape: gradient={tuple(domain_gradients.shape)}, coords={tuple(domain_coords.shape)}.')
    u_x = domain_gradients[..., 0:1][:, pde_mask, :]
    u_y = domain_gradients[..., 1:2][:, pde_mask, :]
    u_t = domain_gradients[..., 2:3][:, pde_mask, :]
    u_inner = u_pred_domain
    if not u_inner.shape == u_x.shape == u_y.shape == u_t.shape:
        raise ValueError(f'The physical prediction and derivative shapes do not match: u={tuple(u_inner.shape)}, u_x={tuple(u_x.shape)}, u_y={tuple(u_y.shape)}, u_t={tuple(u_t.shape)}.')
    source_inner = data['domain_f_inner']
    if source_inner.shape != u_inner.shape:
        raise ValueError(f'Source-term and prediction shapes do not match: source={tuple(source_inner.shape)}, prediction={tuple(u_inner.shape)}.')
    lhs = u_t + u_inner * u_x + u_inner * u_y
    pde_residual = lhs - source_inner
    u_pred_boundary, aux_boundary = adapter.forward(model=model, coords=boundary_coords, data=data, sharp_const=sharp_const, mask=data['boundary_inner_mask'], mode='boundary')
    if u_pred_boundary.ndim == 2:
        u_pred_boundary = u_pred_boundary.unsqueeze(0)
    boundary_values = data['boundary_values_inner']
    if u_pred_boundary.shape != boundary_values.shape:
        raise ValueError(f'Boundary prediction and target shapes do not match: prediction={tuple(u_pred_boundary.shape)}, target={tuple(boundary_values.shape)}.')
    boundary_residual = u_pred_boundary - boundary_values
    u_pred_initial, aux_initial = adapter.forward(model=model, coords=initial_coords, data=data, sharp_const=sharp_const, mask=data['initial_inner_mask'], mode='initial')
    if u_pred_initial.ndim == 2:
        u_pred_initial = u_pred_initial.unsqueeze(0)
    initial_values = data['initial_values']
    if u_pred_initial.shape != initial_values.shape:
        raise ValueError(f'Initial prediction and target shapes do not match: prediction={tuple(u_pred_initial.shape)}, target={tuple(initial_values.shape)}.')
    initial_residual = u_pred_initial - initial_values
    weighting_aux: dict[str, torch.Tensor] = {}
    if sa is not None or ntk is not None:
        adaptive_losses, adaptive_aux = adaptive_group_losses({'pde': pde_residual, 'bc': boundary_residual, 'ini': initial_residual}, data, sa, ntk, update_weighting)
        loss_pde = adaptive_losses.get('pde', pde_residual.new_zeros(()))
        loss_bc = adaptive_losses.get('bc', pde_residual.new_zeros(()))
        loss_ini = adaptive_losses.get('ini', pde_residual.new_zeros(()))
        weighting_aux.update(adaptive_aux)
        total_loss = lambda_pde * loss_pde + lambda_bc * loss_bc + lambda_ini * loss_ini
    elif brdr is not None:
        residual_groups = {'pde': pde_residual, 'bc': boundary_residual, 'ini': initial_residual}
        if update_weighting:
            weights = brdr.update(residual_groups)
        else:
            weights = brdr.get_all()
        loss_pde = torch.mean(weights['pde'] * pde_residual.square())
        loss_bc = torch.mean(weights['bc'] * boundary_residual.square())
        loss_ini = torch.mean(weights['ini'] * initial_residual.square())
        weighting_aux.update({'brdr_pde_weight': weights['pde'].detach(), 'brdr_bc_weight': weights['bc'].detach(), 'brdr_ini_weight': weights['ini'].detach()})
        total_loss = lambda_pde * loss_pde + lambda_bc * loss_bc + lambda_ini * loss_ini
    elif rba is not None:
        if update_weighting:
            rba_weight = rba.update(pde_residual)
        else:
            if not hasattr(rba, 'get_current'):
                raise RuntimeError('RBA must implement get_current() to be used inside an LBFGS closure.')
            rba_weight = rba.get_current()
        if rba_weight.shape != pde_residual.shape:
            raise ValueError(f'RBA weight and PDE-residual shapes do not match: weight={tuple(rba_weight.shape)}, residual={tuple(pde_residual.shape)}.')
        loss_pde = torch.mean((rba_weight * pde_residual).square())
        loss_bc = torch.mean(boundary_residual.square())
        loss_ini = torch.mean(initial_residual.square())
        total_loss = lambda_pde * loss_pde + lambda_bc * loss_bc + lambda_ini * loss_ini
        weighting_aux['rba_pde_weight'] = rba_weight.detach()
    elif causal is not None:
        pde_time_coords = domain_coords[:, pde_mask, 2:3]
        if pde_time_coords.shape[:-1] != pde_residual.shape[:-1]:
            raise ValueError(f'Causal time-coordinate and PDE-residual shapes do not match: time={tuple(pde_time_coords.shape)}, residual={tuple(pde_residual.shape)}.')
        causal_output = causal(pde_residual=pde_residual, pde_time_coords=pde_time_coords, initial_residual=initial_residual, lambda_pde=lambda_pde, lambda_ini=lambda_ini)
        loss_pde = causal_output.pde_loss
        loss_ini = causal_output.initial_loss
        loss_bc = torch.mean(boundary_residual.pow(2))
        total_loss = causal_output.total_loss + lambda_bc * loss_bc
        weighting_aux.update({'causal_temporal_losses': causal_output.temporal_losses.detach(), 'causal_temporal_weights': causal_output.temporal_weights.detach(), 'causal_time_centers': causal_output.time_centers.detach(), 'causal_min_weight': causal_output.min_temporal_weight.detach(), 'causal_epsilon': torch.tensor(causal.epsilon, device=pde_residual.device, dtype=pde_residual.dtype)})
    else:
        loss_pde = torch.mean(pde_residual.square())
        loss_bc = torch.mean(boundary_residual.square())
        loss_ini = torch.mean(initial_residual.square())
        total_loss = lambda_pde * loss_pde + lambda_bc * loss_bc + lambda_ini * loss_ini
    loss_dict = {'total': total_loss.detach(), 'pde': loss_pde.detach(), 'bc': loss_bc.detach(), 'ini': loss_ini.detach()}
    aux: dict[str, torch.Tensor] = {'u_inner': u_inner.detach(), 'u_boundary': u_pred_boundary.detach(), 'u_initial': u_pred_initial.detach(), 'lhs': lhs.detach(), 'u_x': u_x.detach(), 'u_y': u_y.detach(), 'u_t': u_t.detach(), 'pde_residual': pde_residual.detach(), 'bc_residual': boundary_residual.detach(), 'ini_residual': initial_residual.detach(), **weighting_aux}
    if isinstance(aux_domain, dict):
        aux.update({f'domain_{key}': value.detach() for key, value in aux_domain.items() if torch.is_tensor(value)})
    if isinstance(aux_boundary, dict):
        aux.update({f'boundary_{key}': value.detach() for key, value in aux_boundary.items() if torch.is_tensor(value)})
    if isinstance(aux_initial, dict):
        aux.update({f'initial_{key}': value.detach() for key, value in aux_initial.items() if torch.is_tensor(value)})
    if return_residual_graph:
        aux['adaptive_residual_groups'] = residual_groups_for_weights({'pde': pde_residual, 'bc': boundary_residual, 'ini': initial_residual}, data, sa, ntk)
    return (total_loss, loss_dict, aux)

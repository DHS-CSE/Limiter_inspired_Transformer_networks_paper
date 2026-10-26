from typing import Any
import torch
import torch.nn as nn
from models.factory import build_adapter
from methods.sa_ntk import validate_loss_methods, adaptive_group_losses, residual_groups_for_weights

def calculate_loss_burgers_ghost_sharp_analysis(model: nn.Module, data: dict[str, Any], sharp_const: float | None=None, rba=None, brdr=None, causal=None, update_weighting: bool=True, lambda_pde: float=1.0, lambda_bc: float=5.0, lambda_ini: float=5.0, sa=None, ntk=None, return_residual_graph: bool=False, lambda_ini1: float | None=None, lambda_ini2: float | None=None) -> tuple[torch.Tensor, dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    validate_loss_methods(rba, brdr, causal, sa, ntk)
    if rba is not None and brdr is not None:
        raise ValueError('RBA and BRDR cannot be enabled simultaneously.')
    adapter = build_adapter(model)
    domain_coords = adapter.get_domain_coords(data).clone().detach().requires_grad_(True)
    boundary_coords = adapter.get_boundary_coords(data).clone().detach().requires_grad_(True)
    initial_coords = adapter.get_initial_coords(data).clone().detach().requires_grad_(True)
    pde_mask = adapter.get_pde_mask(data)
    u_pred_domain, aux_domain = adapter.forward(model=model, coords=domain_coords, data=data, sharp_const=sharp_const, mask=data['pde_mask'])
    grads = torch.autograd.grad(outputs=u_pred_domain, inputs=domain_coords, grad_outputs=torch.ones_like(u_pred_domain), create_graph=True, retain_graph=True)[0]
    u_inner = u_pred_domain
    u_x = grads[..., 0:1][:, pde_mask, :]
    u_t = grads[..., 1:2][:, pde_mask, :]
    if u_inner.shape != u_x.shape:
        raise ValueError(f'Domain prediction and derivative shapes do not match: u={tuple(u_inner.shape)}, u_x={tuple(u_x.shape)}, u_t={tuple(u_t.shape)}.')
    pde_residual = u_t + u_inner * u_x
    u_pred_boundary, _ = adapter.forward(model=model, coords=boundary_coords, data=data, sharp_const=sharp_const, mask=data['boundary_inner_mask'])
    u_boundary = u_pred_boundary
    boundary_values = data['bnd_values']
    if u_boundary.shape != boundary_values.shape:
        raise ValueError(f'Boundary prediction and target shapes do not match: prediction={tuple(u_boundary.shape)}, target={tuple(boundary_values.shape)}.')
    bc_residual = u_boundary - boundary_values
    u_pred_initial, _ = adapter.forward(model=model, coords=initial_coords, data=data, sharp_const=sharp_const, mask=data['initial_inner_mask'])
    u_initial = u_pred_initial
    initial_values = data['ini_values']
    if u_initial.shape != initial_values.shape:
        raise ValueError(f'Initial prediction and target shapes do not match: prediction={tuple(u_initial.shape)}, target={tuple(initial_values.shape)}.')
    ini_residual = u_initial - initial_values
    split_initial = getattr(model, 'input_type', 'PINN') == 'PINN'
    groups = {'pde': pde_residual, 'bc': bc_residual, 'ini': ini_residual}
    ini_mask = None
    if split_initial:
        ini_coords_physical = data['ini_coords_inner']
        ini_mask = ini_coords_physical[0, :, 0] <= 0.5
        if not ini_mask.any() or ini_mask.all():
            raise ValueError('Burgers split IC requires points on both sides of x=0.5.')
        groups.pop('ini')
        groups.update(ini1=ini_residual[:, ini_mask, :], ini2=ini_residual[:, ~ini_mask, :])

    def initial_mean(values):
        if ini_mask is None:
            return values.mean()
        return values[:, ini_mask, :].mean() + values[:, ~ini_mask, :].mean()
    coefficient_ini1 = lambda_ini if lambda_ini1 is None else lambda_ini1
    coefficient_ini2 = lambda_ini if lambda_ini2 is None else lambda_ini2
    loss_ini1 = loss_ini2 = None

    def split_initial_terms(values):
        return (values[:, ini_mask, :].mean(), values[:, ~ini_mask, :].mean())
    weighting_aux: dict[str, torch.Tensor] = {}
    if sa is not None or ntk is not None:
        adaptive_losses, adaptive_aux = adaptive_group_losses(groups, data, sa, ntk, update_weighting, merge_initial=not split_initial)
        loss_pde = adaptive_losses.get('pde', pde_residual.new_zeros(()))
        loss_bc = adaptive_losses.get('bc', pde_residual.new_zeros(()))
        if split_initial:
            loss_ini1, loss_ini2 = (adaptive_losses['ini1'], adaptive_losses['ini2'])
            loss_ini = loss_ini1 + loss_ini2
        else:
            loss_ini = adaptive_losses.get('ini', pde_residual.new_zeros(()))
        weighting_aux.update(adaptive_aux)
    elif brdr is not None:
        residual_groups = {'pde': pde_residual, 'bc': bc_residual, 'ini': ini_residual}
        if update_weighting:
            weights = brdr.update(residual_groups)
        else:
            weights = brdr.get_all()
        loss_pde = torch.mean(weights['pde'] * pde_residual.pow(2))
        loss_bc = torch.mean(weights['bc'] * bc_residual.pow(2))
        initial_squared = weights['ini'] * ini_residual.pow(2)
        if split_initial:
            loss_ini1, loss_ini2 = split_initial_terms(initial_squared)
            loss_ini = loss_ini1 + loss_ini2
        else:
            loss_ini = initial_mean(initial_squared)
        weighting_aux.update({'brdr_pde_weight': weights['pde'].detach(), 'brdr_bc_weight': weights['bc'].detach(), 'brdr_ini_weight': weights['ini'].detach()})
    elif rba is not None:
        if update_weighting:
            rba_weight = rba.update(pde_residual)
        else:
            rba_weight = rba.get_current()
        loss_pde = torch.mean((rba_weight * pde_residual).pow(2))
        loss_bc = torch.mean(bc_residual.pow(2))
        if split_initial:
            loss_ini1, loss_ini2 = split_initial_terms(ini_residual.pow(2))
            loss_ini = loss_ini1 + loss_ini2
        else:
            loss_ini = initial_mean(ini_residual.pow(2))
        weighting_aux['rba_pde_weight'] = rba_weight.detach()
    elif causal is not None:
        pde_time_coords = domain_coords[:, pde_mask, 1:2]
        causal_options = {}
        if split_initial:
            loss_ini1, loss_ini2 = split_initial_terms(ini_residual.square())
            causal_options['weighted_initial_loss'] = coefficient_ini1 * loss_ini1 + coefficient_ini2 * loss_ini2
        causal_output = causal(pde_residual=pde_residual, pde_time_coords=pde_time_coords, initial_residual=ini_residual, lambda_pde=lambda_pde, lambda_ini=lambda_ini, **causal_options)
        loss_pde = causal_output.pde_loss
        loss_ini = loss_ini1 + loss_ini2 if split_initial else causal_output.initial_loss
        loss_bc = torch.mean(bc_residual.square())
        total_loss = causal_output.total_loss + lambda_bc * loss_bc
        weighting_aux.update({'causal_temporal_losses': causal_output.temporal_losses.detach(), 'causal_temporal_weights': causal_output.temporal_weights.detach(), 'causal_time_centers': causal_output.time_centers.detach(), 'causal_min_weight': causal_output.min_temporal_weight.detach(), 'causal_epsilon': torch.tensor(causal.epsilon, device=pde_residual.device, dtype=pde_residual.dtype)})
    else:
        loss_pde = torch.mean(pde_residual.pow(2))
        loss_bc = torch.mean(bc_residual.pow(2))
        if split_initial:
            loss_ini1, loss_ini2 = split_initial_terms(ini_residual.pow(2))
            loss_ini = loss_ini1 + loss_ini2
        else:
            loss_ini = initial_mean(ini_residual.pow(2))
    if split_initial:
        initial_contribution1 = coefficient_ini1 * loss_ini1
        initial_contribution2 = coefficient_ini2 * loss_ini2
        initial_contribution = initial_contribution1 + initial_contribution2
    else:
        initial_contribution = lambda_ini * loss_ini
    if causal is None:
        total_loss = lambda_pde * loss_pde + lambda_bc * loss_bc + initial_contribution
    loss_dict = {'total': total_loss.detach(), 'pde': loss_pde.detach(), 'bc': loss_bc.detach(), 'ini': loss_ini.detach()}
    if split_initial:
        loss_dict.update({'ini1': loss_ini1.detach(), 'ini2': loss_ini2.detach(), 'ini1_weighted': initial_contribution1.detach(), 'ini2_weighted': initial_contribution2.detach()})
    aux = {'u_inner': u_inner.detach(), 'u_boundary': u_boundary.detach(), 'u_initial': u_initial.detach(), 'pde_residual': pde_residual, 'bc_residual': bc_residual.detach(), 'ini_residual': ini_residual.detach(), **weighting_aux}
    if ini_mask is not None:
        aux['ini1_residual'] = groups['ini1'].detach()
        aux['ini2_residual'] = groups['ini2'].detach()
        aux['initial_split_mask'] = ini_mask.detach()
        loss_dict['ini1_unweighted'] = groups['ini1'].square().mean().detach()
        loss_dict['ini2_unweighted'] = groups['ini2'].square().mean().detach()
    if sa is not None:
        coordinate_groups = {'pde': data['domain_coords_inner'], 'bc': data['bnd_coords'], 'ini': data['ini_coords_inner']}
        for key, weight in sa.weights.items():
            coords = coordinate_groups['ini' if key.startswith('ini') else key]
            if key in ('ini1', 'ini2'):
                mask = coords[0, :, 0] <= sa.initial_split_x
                coords = coords[:, mask if key == 'ini1' else ~mask, :]
            if coords.shape[:-1] != weight.shape[:-1]:
                raise ValueError(f'SA coordinate/weight mismatch for {key}.')
            aux[f'sa_{key}_coords'] = coords.detach()
            aux[f'sa_{key}_effective_weight'] = (weight.square() if sa.mode == 'square' else weight).detach()
    if isinstance(aux_domain, dict):
        aux.update({key: value.detach() for key, value in aux_domain.items() if torch.is_tensor(value)})
    if return_residual_graph:
        aux['adaptive_residual_groups'] = residual_groups_for_weights(groups, data, sa, ntk)
    return (total_loss, loss_dict, aux)

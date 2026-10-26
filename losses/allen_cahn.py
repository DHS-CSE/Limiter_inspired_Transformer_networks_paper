import torch
from typing import Any
from models.factory import build_adapter
from methods.sa_ntk import validate_loss_methods, adaptive_group_losses, residual_groups_for_weights

def calculate_loss_allen_cahn_ghost_sharp_analysis(model: torch.nn.Module, data: dict[str, Any], sharp_const: float, rba=None, brdr=None, causal=None, update_weighting: bool=True, hard_periodic: bool=False, lambda_pde: float=1.0, lambda_bc: float=10.0, lambda_bc_grad: float=10.0, lambda_ini: float=100.0, rho: float=5.0, sa=None, ntk=None, return_residual_graph: bool=False) -> tuple[torch.Tensor, dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    validate_loss_methods(rba, brdr, causal, sa, ntk)
    adapter = build_adapter(model)
    coords_grad = adapter.get_domain_coords(data).clone().requires_grad_(True)
    ini_coords_grad = adapter.get_initial_coords(data).clone().requires_grad_(True)
    u_pred_domain, aux_domain = adapter.forward(model, coords_grad, data, sharp_const, data['pde_mask'])
    inner_index = adapter.get_pde_mask(data)
    grads = torch.autograd.grad(outputs=u_pred_domain, inputs=coords_grad, grad_outputs=torch.ones_like(u_pred_domain), create_graph=True, retain_graph=True)[0]
    grad_x = grads[..., 0:1][:, inner_index, :]
    grad_t = grads[..., 1:2][:, inner_index, :]
    if u_pred_domain.shape[1] == grad_x.shape[1]:
        u_inner = u_pred_domain
    elif u_pred_domain.shape[1] == coords_grad.shape[1]:
        u_inner = u_pred_domain[:, inner_index, :]
    else:
        raise ValueError('Prediction and physical derivative point counts differ.')
    grad_xx = torch.autograd.grad(outputs=grad_x, inputs=coords_grad, grad_outputs=torch.ones_like(grad_x), create_graph=True, retain_graph=True)[0][..., 0:1][:, inner_index, :]
    lhs = grad_t + rho * (u_inner ** 3 - u_inner) - 0.0001 * grad_xx
    pde_residual = -lhs - data['domain_f_inner']
    u_ini, _ = adapter.forward(model, ini_coords_grad, data, sharp_const, data['initial_inner_mask'])
    mask_initial_inner = adapter.get_initial_mask(data)
    u_ini_inner = u_ini
    ini_residual = u_ini_inner - data['ini_values']
    bc_residual = None
    bc_grad_residual = None
    if not hard_periodic:
        left_coords = adapter.get_left_boundary_coords(data).clone().detach().requires_grad_(True)
        right_coords = adapter.get_right_boundary_coords(data).clone().detach().requires_grad_(True)
        left_mask = adapter.get_left_boundary_mask(data)
        right_mask = adapter.get_right_boundary_mask(data)
        u_left_all, _ = adapter.forward(model=model, coords=left_coords, data=data, sharp_const=sharp_const, mask=left_mask)
        u_right_all, _ = adapter.forward(model=model, coords=right_coords, data=data, sharp_const=sharp_const, mask=right_mask)
        u_left = u_left_all
        u_right = u_right_all
        if u_left.shape != u_right.shape:
            raise ValueError(f'Left and right boundary prediction shapes do not match: left={tuple(u_left.shape)}, right={tuple(u_right.shape)}.')
        bc_residual = u_left - u_right
        u_left_x_all = torch.autograd.grad(outputs=u_left_all, inputs=left_coords, grad_outputs=torch.ones_like(u_left_all), create_graph=True, retain_graph=True)[0][..., 0:1]
        u_right_x_all = torch.autograd.grad(outputs=u_right_all, inputs=right_coords, grad_outputs=torch.ones_like(u_right_all), create_graph=True, retain_graph=True)[0][..., 0:1]
        u_left_x = u_left_x_all[:, left_mask, :]
        u_right_x = u_right_x_all[:, right_mask, :]
        if u_left_x.shape != u_right_x.shape:
            raise ValueError(f'Left and right boundary-gradient shapes do not match: left={tuple(u_left_x.shape)}, right={tuple(u_right_x.shape)}.')
        bc_grad_residual = u_left_x - u_right_x
    weighting_aux = {}
    if sa is not None or ntk is not None:
        adaptive_losses, adaptive_aux = adaptive_group_losses({'pde': pde_residual, 'ini': ini_residual, 'bc': bc_residual, 'bc_grad': bc_grad_residual}, data, sa, ntk, update_weighting)
        loss_pde = adaptive_losses.get('pde', pde_residual.new_zeros(()))
        loss_bc = adaptive_losses.get('bc', pde_residual.new_zeros(()))
        loss_bc_grad = adaptive_losses.get('bc_grad', pde_residual.new_zeros(()))
        loss_ini = adaptive_losses.get('ini', pde_residual.new_zeros(()))
        weighting_aux.update(adaptive_aux)
    elif brdr is not None:
        residual_groups = {'pde': pde_residual, 'ini': ini_residual}
        if not hard_periodic:
            residual_groups['bc'] = bc_residual
            residual_groups['bc_grad'] = bc_grad_residual
        if update_weighting:
            weights = brdr.update(residual_groups)
        else:
            weights = brdr.get_all()
        loss_pde = torch.mean(weights['pde'] * pde_residual.pow(2))
        loss_ini = torch.mean(weights['ini'] * ini_residual.pow(2))
        weighting_aux['brdr_pde_weight'] = weights['pde'].detach()
        weighting_aux['brdr_ini_weight'] = weights['ini'].detach()
        if not hard_periodic:
            loss_bc = torch.mean(weights['bc'] * bc_residual.pow(2))
            loss_bc_grad = torch.mean(weights['bc_grad'] * bc_grad_residual.pow(2))
            weighting_aux['brdr_bc_weight'] = weights['bc'].detach()
            weighting_aux['brdr_bc_grad_weight'] = weights['bc_grad'].detach()
        else:
            loss_bc = pde_residual.new_zeros(())
            loss_bc_grad = pde_residual.new_zeros(())
    elif rba is not None:
        global_indices = data.get('global_domain_indices', None)
        if update_weighting:
            if global_indices is None:
                rba_weight = rba.update(pde_residual)
            else:
                rba_weight = rba.update_batch(residual=pde_residual, global_indices=global_indices)
        elif global_indices is None:
            rba_weight = rba.get_current()
        else:
            rba_weight = rba.get_batch(global_indices=global_indices)
        loss_pde = torch.mean((rba_weight * pde_residual).pow(2))
        loss_ini = torch.mean(ini_residual.pow(2))
        weighting_aux['rba_pde_weight'] = rba_weight.detach()
        if not hard_periodic:
            loss_bc = torch.mean(bc_residual.pow(2))
            loss_bc_grad = torch.mean(bc_grad_residual.pow(2))
        else:
            loss_bc = pde_residual.new_zeros(())
            loss_bc_grad = pde_residual.new_zeros(())
    elif causal is not None:
        pde_time_coords = coords_grad[:, inner_index, 1:2]
        causal_output = causal(pde_residual=pde_residual, pde_time_coords=pde_time_coords, initial_residual=ini_residual, lambda_pde=lambda_pde, lambda_ini=lambda_ini)
        loss_pde = causal_output.pde_loss
        loss_ini = causal_output.initial_loss
        if hard_periodic:
            loss_bc = pde_residual.new_zeros(())
            loss_bc_grad = pde_residual.new_zeros(())
            total_loss = causal_output.total_loss
        else:
            loss_bc = torch.mean(bc_residual.square())
            loss_bc_grad = torch.mean(bc_grad_residual.square())
            total_loss = causal_output.total_loss + lambda_bc * loss_bc + lambda_bc_grad * loss_bc_grad
        weighting_aux.update({'causal_temporal_losses': causal_output.temporal_losses.detach(), 'causal_temporal_weights': causal_output.temporal_weights.detach(), 'causal_time_centers': causal_output.time_centers.detach(), 'causal_min_weight': causal_output.min_temporal_weight.detach(), 'causal_epsilon': torch.tensor(causal.epsilon, device=pde_residual.device, dtype=pde_residual.dtype)})
    else:
        loss_pde = torch.mean(pde_residual.pow(2))
        loss_ini = torch.mean(ini_residual.pow(2))
        if not hard_periodic:
            loss_bc = torch.mean(bc_residual.pow(2))
            loss_bc_grad = torch.mean(bc_grad_residual.pow(2))
        else:
            loss_bc = pde_residual.new_zeros(())
            loss_bc_grad = pde_residual.new_zeros(())
    if hard_periodic:
        total_loss = lambda_pde * loss_pde + lambda_ini * loss_ini
    else:
        total_loss = lambda_pde * loss_pde + lambda_bc * loss_bc + lambda_bc_grad * loss_bc_grad + lambda_ini * loss_ini
    loss_dict = {'total': total_loss.detach(), 'pde': loss_pde.detach(), 'bc': loss_bc.detach(), 'bc_grad': loss_bc_grad.detach(), 'ini': loss_ini.detach()}
    aux = {'u_inner': u_inner.detach(), 'u_initial': u_ini_inner.detach(), 'pde_residual': pde_residual.detach(), 'ini_residual': ini_residual.detach(), 'hard_periodic': torch.tensor(float(hard_periodic), device=total_loss.device), **weighting_aux}
    if not hard_periodic:
        aux.update({'bc_residual': bc_residual.detach(), 'bc_grad_residual': bc_grad_residual.detach()})
    if isinstance(aux_domain, dict):
        aux.update({key: value.detach() for key, value in aux_domain.items() if torch.is_tensor(value)})
    if return_residual_graph:
        aux['adaptive_residual_groups'] = residual_groups_for_weights({'pde': pde_residual, 'ini': ini_residual, 'bc': bc_residual, 'bc_grad': bc_grad_residual}, data, sa, ntk)
    return (total_loss, loss_dict, aux)

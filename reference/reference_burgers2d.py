import torch
import math

def burgers2d_exact_solution(
    coords: torch.Tensor, u_left: float=1.0, u_middle: float=0.6, u_right: float=0.0,
    discontinuity_distance: float=0.8
) -> torch.Tensor:
    x = coords[..., 0:1]
    y = coords[..., 1:2]
    t = coords[..., 2:3]
    sqrt_two = math.sqrt(2.0)
    eta = (x + y) / sqrt_two
    shock_speed_1 = sqrt_two / 2.0 * (u_left + u_middle)
    shock_speed_2 = sqrt_two / 2.0 * (u_middle + u_right)
    eta_1 = shock_speed_1 * t
    eta_2 = discontinuity_distance + shock_speed_2 * t
    solution = torch.full_like(eta, fill_value=u_middle)
    solution = torch.where(eta < eta_1, torch.full_like(eta, u_left), solution)
    solution = torch.where(eta > eta_2, torch.full_like(eta, u_right), solution)
    return solution

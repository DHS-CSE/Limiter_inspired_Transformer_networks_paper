import torch

def burgers_solution(points):
    x = points[..., 0:1]
    t = points[..., 1:2]
    u_exact = torch.zeros_like(x)
    x_s0 = 2.0 * t
    x_s1 = 0.5 + 1.0 * t
    x_s2 = 1.5 * t + 0.25
    mask_pre = t < 0.5
    u_exact[mask_pre & (x < x_s0)] = 2.5
    u_exact[mask_pre & (x >= x_s0) & (x < x_s1)] = 1.5
    u_exact[mask_pre & (x >= x_s1)] = 0.5
    mask_post = t >= 0.5
    u_exact[mask_post & (x < x_s2)] = 2.5
    u_exact[mask_post & (x >= x_s2)] = 0.5
    return u_exact

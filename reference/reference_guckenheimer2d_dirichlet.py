import math
import torch

def guckenheimer2d_exact_solution(coords, amplitude=0.15, d1=0.3, d2=0.8):
    x, y, t = (coords[..., 0:1], coords[..., 1:2], coords[..., 2:3])
    sqrt2 = math.sqrt(2.0)
    gap_rate = (109 / 75 - 21 / 50 + 2 * math.pi * abs(amplitude) * abs(9 / 50 - 11 / 75)) / sqrt2
    if not 2 * math.pi * abs(amplitude) < 1 or not d1 < d2:
        raise ValueError('Require 2*pi*abs(amplitude)<1 and d1<d2.')
    if torch.any(t < 0) or torch.any(d2 - d1 - gap_rate * t <= 0):
        raise ValueError('Exact solution requires nonnegative time before front interaction.')
    x1, y1 = (x - 4 / 5 * t, y - 49 / 75 * t)
    x2, y2 = (x - 3 / 10 * t, y - 3 / 25 * t)
    phi1 = (x1 + y1) / sqrt2 - amplitude * torch.sin(2 * torch.pi * (x1 - y1) / sqrt2)
    phi2 = (x2 + y2) / sqrt2 - amplitude * torch.sin(2 * torch.pi * (x2 - y2) / sqrt2)
    u = torch.full_like(x, 0.6)
    u = torch.where(phi1 < d1, torch.ones_like(u), u)
    return torch.where(phi2 > d2, torch.zeros_like(u), u)

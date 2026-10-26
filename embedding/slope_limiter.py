import torch
import torch.nn as nn
from .embedding import Trainable_FourierFeatureEmbedding

class FixedSharpSlopeLimiterEmbeddingAnalysis_NoSlope(nn.Module):

    def __init__(self, scale, in_dim, embed_dim):
        super().__init__()
        self.low_proj = nn.Sequential(nn.Linear(1, embed_dim), nn.Tanh(), nn.Linear(embed_dim, 1))
        self.high_proj = nn.Sequential(
            Trainable_FourierFeatureEmbedding(input_dim=in_dim, mapping_size=embed_dim // 2, scale=scale),
            nn.Linear(embed_dim, embed_dim)
        )
        self.alpha = nn.Parameter(torch.tensor(5.0))

    class Exponential(nn.Module):

        def __init__(self, alpha=1.0):
            super().__init__()
            self.alpha = alpha

        def forward(self, x):
            return torch.exp(-self.alpha * x)

    def forward(self, coords, sharp_const, mask):
        tol = 1e-08
        x = coords[..., 0]
        t = coords[..., 1]
        x_unique = torch.unique(x.detach())
        t_unique = torch.unique(t.detach())
        x_unique, _ = torch.sort(x_unique)
        t_unique, _ = torch.sort(t_unique)
        dx = x_unique[1:] - x_unique[:-1]
        dt = t_unique[1:] - t_unique[:-1]
        dx = dx[dx > tol]
        dt = dt[dt > tol]
        high = self.high_proj(coords)
        if len(x_unique) <= 1:
            return high
        high_grid = high.view(1, len(x_unique), len(t_unique), high.shape[-1])
        dx = dx.view(1, len(x_unique) - 1, 1, 1)
        a = (high_grid[:, 1:-1, :, :] - high_grid[:, :-2, :, :]) / dx[:, :-1, :, :]
        b = (high_grid[:, 2:, :, :] - high_grid[:, 1:-1, :, :]) / dx[:, 1:, :, :]
        grad_0 = (high_grid[:, 1, :, :] - high_grid[:, 0, :, :]) / dx[:, 0, :, :]
        grad_N = (high_grid[:, -1, :, :] - high_grid[:, -2, :, :]) / dx[:, -1, :, :]
        ab = a * b
        w = torch.tanh(10 * torch.relu(ab)) ** 2
        sigma_candidate = torch.where(
            torch.linalg.norm(a, dim=-1, keepdim=True) < torch.linalg.norm(b, dim=-1, keepdim=True), a, b
        )
        sigma = w * sigma_candidate
        sigma_full = torch.zeros_like(high_grid)
        sigma_full[:, 1:-1, :, :] = sigma
        sigma_full[:, 0, :, :] = grad_0
        sigma_full[:, -1, :, :] = grad_N
        limiter = sigma_full[:, 1:-1, :, :] - sigma_full[:, :-2, :, :]
        limiter = torch.sqrt(torch.mean(limiter ** 2, dim=-1, keepdim=True) + 1e-08)
        limiter_full = torch.zeros_like(high_grid[..., :1])
        limiter_full[:, 1:-1, :, :] = limiter
        limiter_full[:, 0, :, :] = limiter_full[:, 1, :, :]
        limiter_full[:, -1, :, :] = limiter_full[:, -2, :, :]
        limiter = limiter_full.reshape(high.shape[0], high.shape[1], 1)
        z = high
        z_inner = z[:, mask, :]
        l_inner = limiter[:, mask, :]
        return (z_inner, l_inner)

class FixedSharpSlopeLimiterEmbeddingAnalysis(nn.Module):

    def __init__(self, scale, in_dim, embed_dim):
        super().__init__()
        self.low_proj = nn.Sequential(nn.Linear(1, embed_dim), nn.Tanh(), nn.Linear(embed_dim, 1))
        self.high_proj = nn.Sequential(
            Trainable_FourierFeatureEmbedding(input_dim=in_dim, mapping_size=embed_dim // 2, scale=scale),
            nn.Linear(embed_dim, embed_dim)
        )
        self.alpha = nn.Parameter(torch.tensor(5.0))

    class Exponential(nn.Module):

        def __init__(self, alpha=1.0):
            super().__init__()
            self.alpha = alpha

        def forward(self, x):
            return torch.exp(-self.alpha * x)

    def forward(self, coords, sharp_const, mask):
        tol = 1e-08
        x = coords[..., 0]
        t = coords[..., 1]
        x_unique = torch.unique(x.detach())
        t_unique = torch.unique(t.detach())
        x_unique, _ = torch.sort(x_unique)
        t_unique, _ = torch.sort(t_unique)
        dx = x_unique[1:] - x_unique[:-1]
        dt = t_unique[1:] - t_unique[:-1]
        dx = dx[dx > tol]
        dt = dt[dt > tol]
        high = self.high_proj(coords)
        if len(x_unique) <= 1:
            return high
        high_grid = high.view(1, len(x_unique), len(t_unique), high.shape[-1])
        dx = dx.view(1, len(x_unique) - 1, 1, 1)
        a = (high_grid[:, 1:-1, :, :] - high_grid[:, :-2, :, :]) / dx[:, :-1, :, :]
        b = (high_grid[:, 2:, :, :] - high_grid[:, 1:-1, :, :]) / dx[:, 1:, :, :]
        grad_0 = (high_grid[:, 1, :, :] - high_grid[:, 0, :, :]) / dx[:, 0, :, :]
        grad_N = (high_grid[:, -1, :, :] - high_grid[:, -2, :, :]) / dx[:, -1, :, :]
        ab = a * b
        w = torch.tanh(10 * torch.relu(ab)) ** 2
        sigma_candidate = torch.where(
            torch.linalg.norm(a, dim=-1, keepdim=True) < torch.linalg.norm(b, dim=-1, keepdim=True), a, b
        )
        sigma = w * sigma_candidate
        sigma_full = torch.zeros_like(high_grid)
        sigma_full[:, 1:-1, :, :] = sigma
        sigma_full[:, 0, :, :] = grad_0
        sigma_full[:, -1, :, :] = grad_N
        limiter = sigma_full[:, 1:-1, :, :] - sigma_full[:, :-2, :, :]
        limiter = torch.sqrt(torch.mean(limiter ** 2, dim=-1, keepdim=True) + 1e-08)
        limiter_full = torch.zeros_like(high_grid[..., :1])
        limiter_full[:, 1:-1, :, :] = limiter
        limiter_full[:, 0, :, :] = limiter_full[:, 1, :, :]
        limiter_full[:, -1, :, :] = limiter_full[:, -2, :, :]
        limiter = limiter_full.reshape(high.shape[0], high.shape[1], 1)
        sigma_full = sigma_full.reshape(high.shape[0], high.shape[1], high.shape[2])
        gradient_norm = torch.linalg.norm(sigma_full, ord=2, dim=-1, keepdim=True)
        gate = torch.tanh(self.alpha * gradient_norm ** 1.5) ** 2
        low = self.low_proj(limiter)
        z = (1 + sharp_const * gate * low) * high
        z_inner = z[:, mask, :]
        gate_inner = gate[:, mask, :]
        limiter_inner = limiter[:, mask, :]
        low_inner = low[:, mask, :]
        return (z_inner, gate_inner, limiter_inner, low_inner)

class FixedSharpSlopeLimiterEmbeddingAnalysis_Superbee(nn.Module):

    def __init__(self, scale, in_dim, embed_dim):
        super().__init__()
        self.low_proj = nn.Sequential(nn.Linear(1, embed_dim), nn.Tanh(), nn.Linear(embed_dim, 1))
        self.high_proj = nn.Sequential(
            Trainable_FourierFeatureEmbedding(input_dim=in_dim, mapping_size=embed_dim // 2, scale=scale),
            nn.Linear(embed_dim, embed_dim)
        )
        self.alpha = nn.Parameter(torch.tensor(5.0))

    def forward(self, coords, sharp_const, mask):
        tol = 1e-08
        x = coords[..., 0]
        t = coords[..., 1]
        x_unique = torch.unique(x.detach())
        t_unique = torch.unique(t.detach())
        x_unique, _ = torch.sort(x_unique)
        t_unique, _ = torch.sort(t_unique)
        dx = x_unique[1:] - x_unique[:-1]
        dx = dx[dx > tol]
        high = self.high_proj(coords)
        if len(x_unique) <= 1:
            return high
        high_grid = high.view(1, len(x_unique), len(t_unique), high.shape[-1])
        dx = dx.view(1, len(x_unique) - 1, 1, 1)
        a = (high_grid[:, 1:-1, :, :] - high_grid[:, :-2, :, :]) / dx[:, :-1, :, :]
        b = (high_grid[:, 2:, :, :] - high_grid[:, 1:-1, :, :]) / dx[:, 1:, :, :]
        cond1 = b * (2 * a) > 0
        sigma1 = torch.where(
            cond1, torch.where(torch.linalg.norm(b) < torch.linalg.norm(2 * a), b, 2 * a), torch.zeros_like(b)
        )
        cond2 = 2 * b * a > 0
        sigma2 = torch.where(
            cond2, torch.where(torch.linalg.norm(2 * b) < torch.linalg.norm(a), 2 * b, a), torch.zeros_like(a)
        )
        cond3 = sigma1 * sigma2 > 0
        sigma_candidate = torch.where(
            cond3, torch.where(torch.linalg.norm(sigma1) > torch.linalg.norm(sigma2), sigma1, sigma2),
            torch.zeros_like(sigma1)
        )
        w = torch.tanh(10 * torch.relu(sigma1 * sigma2)) ** 2
        sigma = w * sigma_candidate
        grad_0 = (high_grid[:, 1, :, :] - high_grid[:, 0, :, :]) / dx[:, 0, :, :]
        grad_N = (high_grid[:, -1, :, :] - high_grid[:, -2, :, :]) / dx[:, -1, :, :]
        sigma_full = torch.zeros_like(high_grid)
        sigma_full[:, 1:-1, :, :] = sigma
        sigma_full[:, 0, :, :] = grad_0
        sigma_full[:, -1, :, :] = grad_N
        limiter = sigma_full[:, 1:-1, :, :] - sigma_full[:, :-2, :, :]
        limiter = torch.sqrt(torch.mean(limiter ** 2, dim=-1, keepdim=True) + 1e-08)
        limiter_full = torch.zeros_like(high_grid[..., :1])
        limiter_full[:, 1:-1, :, :] = limiter
        limiter_full[:, 0, :, :] = limiter_full[:, 1, :, :]
        limiter_full[:, -1, :, :] = limiter_full[:, -2, :, :]
        limiter = limiter_full.reshape(high.shape[0], high.shape[1], 1)
        sigma_full = sigma_full.reshape(high.shape[0], high.shape[1], high.shape[2])
        gradient_norm = torch.linalg.norm(sigma_full, dim=-1, keepdim=True)
        gate = torch.tanh(self.alpha * gradient_norm ** 1.5) ** 2
        low = self.low_proj(limiter)
        z = (1 + sharp_const * gate * low) * high
        z_inner = z[:, mask, :]
        gate_inner = gate[:, mask, :]
        limiter_inner = limiter[:, mask, :]
        low_inner = low[:, mask, :]
        return (z_inner, gate_inner, limiter_inner, low_inner)

class FixedSharpSlopeLimiterEmbeddingAnalysis_MC(nn.Module):

    def __init__(self, scale, in_dim, embed_dim):
        super().__init__()
        self.low_proj = nn.Sequential(nn.Linear(1, embed_dim), nn.Tanh(), nn.Linear(embed_dim, 1))
        self.high_proj = nn.Sequential(
            Trainable_FourierFeatureEmbedding(input_dim=in_dim, mapping_size=embed_dim // 2, scale=scale),
            nn.Linear(embed_dim, embed_dim)
        )
        self.alpha = nn.Parameter(torch.tensor(5.0))

    def forward(self, coords, sharp_const, mask):
        tol = 1e-08
        x = coords[..., 0]
        t = coords[..., 1]
        x_unique = torch.unique(x.detach())
        t_unique = torch.unique(t.detach())
        x_unique, _ = torch.sort(x_unique)
        t_unique, _ = torch.sort(t_unique)
        dx = x_unique[1:] - x_unique[:-1]
        dx = dx[dx > tol]
        high = self.high_proj(coords)
        if len(x_unique) <= 1:
            return high
        high_grid = high.view(1, len(x_unique), len(t_unique), high.shape[-1])
        dx = dx.view(1, len(x_unique) - 1, 1, 1)
        a = (high_grid[:, 1:-1, :, :] - high_grid[:, :-2, :, :]) / dx[:, :-1, :, :]
        b = (high_grid[:, 2:, :, :] - high_grid[:, 1:-1, :, :]) / dx[:, 1:, :, :]
        c = 0.5 * (a + b)
        d = 2.0 * a
        e = 2.0 * b
        norm_c = torch.linalg.norm(c)
        norm_d = torch.linalg.norm(d)
        norm_e = torch.linalg.norm(e)
        min_cd = torch.where(norm_c < norm_d, c, d)
        min_norm_cd = torch.minimum(norm_c, norm_d)
        sigma_candidate = torch.where(min_norm_cd < norm_e, min_cd, e)
        w = torch.tanh(10 * torch.relu(min_cd * e)) ** 2
        sigma = w * sigma_candidate
        grad_0 = (high_grid[:, 1, :, :] - high_grid[:, 0, :, :]) / dx[:, 0, :, :]
        grad_N = (high_grid[:, -1, :, :] - high_grid[:, -2, :, :]) / dx[:, -1, :, :]
        sigma_full = torch.zeros_like(high_grid)
        sigma_full[:, 1:-1, :, :] = sigma
        sigma_full[:, 0, :, :] = grad_0
        sigma_full[:, -1, :, :] = grad_N
        limiter = sigma_full[:, 1:-1, :, :] - sigma_full[:, :-2, :, :]
        limiter = torch.sqrt(torch.mean(limiter ** 2, dim=-1, keepdim=True) + 1e-08)
        limiter_full = torch.zeros_like(high_grid[..., :1])
        limiter_full[:, 1:-1, :, :] = limiter
        limiter_full[:, 0, :, :] = limiter_full[:, 1, :, :]
        limiter_full[:, -1, :, :] = limiter_full[:, -2, :, :]
        limiter = limiter_full.reshape(high.shape[0], high.shape[1], 1)
        sigma_full = sigma_full.reshape(high.shape[0], high.shape[1], high.shape[2])
        gradient_norm = torch.linalg.norm(sigma_full, dim=-1, keepdim=True)
        gate = torch.tanh(self.alpha * gradient_norm ** 1.5) ** 2
        low = self.low_proj(limiter)
        z = (1 + sharp_const * gate * low) * high
        z_inner = z[:, mask, :]
        gate_inner = gate[:, mask, :]
        limiter_inner = limiter[:, mask, :]
        low_inner = low[:, mask, :]
        return (z_inner, gate_inner, limiter_inner, low_inner)

class AP_FixedSharpSlopeLimiterEmbeddingAnalysis(nn.Module):

    def __init__(self, scale, in_dim, embed_dim, virtual_dx):
        super().__init__()
        self.virtual_dx = virtual_dx
        self.low_proj = nn.Sequential(nn.Linear(1, embed_dim), nn.Tanh(), nn.Linear(embed_dim, 1))
        self.high_proj = nn.Sequential(
            Trainable_FourierFeatureEmbedding(input_dim=in_dim, mapping_size=embed_dim // 2, scale=scale),
            nn.Linear(embed_dim, embed_dim)
        )
        self.alpha = nn.Parameter(torch.tensor(5.0))

    class Exponential(nn.Module):

        def __init__(self, alpha=1.0):
            super().__init__()
            self.alpha = alpha

        def forward(self, x):
            return torch.exp(-self.alpha * x)

    def minmod(self, a, b):
        ab = a * b
        w = torch.tanh(10 * torch.relu(ab)) ** 2
        sigma_candidate = torch.where(
            torch.linalg.norm(a, dim=-1, keepdim=True) < torch.linalg.norm(b, dim=-1, keepdim=True), a, b
        )
        return w * sigma_candidate

    def forward(self, coords, sharp_const):
        h = self.virtual_dx
        offsets = torch.tensor(
            [-2.0 * h, -1.0 * h, 0.0, 1.0 * h], device=coords.device, dtype=coords.dtype
        ).view(1, 4, 1)
        coords_virtual = coords.unsqueeze(1)
        coords_virtual = coords_virtual.expand(-1, 4, -1, -1).clone()
        coords_virtual[..., 0] += offsets
        coords_all = torch.cat(
            [coords_virtual[:, 0], coords_virtual[:, 1], coords_virtual[:, 2], coords_virtual[:, 3]], dim=1
        )
        high_all = self.high_proj(coords_all)
        high_m2, high_m1, high, high_p1 = torch.chunk(high_all, 4, dim=1)
        a0 = (high - high_m1) / h
        b0 = (high_p1 - high) / h
        am1 = (high_m1 - high_m2) / h
        bm1 = (high - high_m1) / h
        sigma = self.minmod(a0, b0)
        sigma_prev = self.minmod(am1, bm1)
        x = coords[..., 0]
        left_boundary_mask = (x == 0.0).unsqueeze(-1)
        sigma_prev = torch.where(left_boundary_mask, a0, sigma_prev)
        limiter = sigma - sigma_prev
        limiter = torch.sqrt(torch.mean(limiter ** 2, dim=-1, keepdim=True) + 1e-08)
        gradient_norm = torch.linalg.norm(sigma, ord=2, dim=-1, keepdim=True)
        gate = torch.tanh(self.alpha * gradient_norm ** 1.5) ** 2
        low = self.low_proj(limiter)
        z = (1.0 + sharp_const * gate * low) * high
        del high_m2, high_m1, high_p1
        return (z, gate, limiter, low)

class FixedSharpSlopeLimiterEmbeddingAnalysis3D(nn.Module):

    def __init__(self, scale, in_dim, embed_dim):
        super().__init__()
        self.low_proj = nn.Sequential(nn.Linear(2, embed_dim), nn.Tanh(), nn.Linear(embed_dim, 1))
        self.high_proj = nn.Sequential(
            Trainable_FourierFeatureEmbedding(input_dim=in_dim, mapping_size=embed_dim // 2, scale=scale),
            nn.Linear(embed_dim, embed_dim)
        )
        self.alpha = nn.Parameter(torch.tensor(5.0))

    def slope_limiter_2d(self, high, coords, axis='y'):
        tol = 1e-08
        coords_flat = coords[0]
        high_flat = high[0]
        if axis == 'y':
            s1 = coords_flat[:, 1]
            s2 = coords_flat[:, 2]
        else:
            s1 = coords_flat[:, 0]
            s2 = coords_flat[:, 2]
        s1_unique = torch.unique(s1.detach())
        s2_unique = torch.unique(s2.detach())
        s1_unique, _ = torch.sort(s1_unique)
        s2_unique, _ = torch.sort(s2_unique)
        idx = torch.argsort(s2, stable=True)
        idx = idx[torch.argsort(s1[idx], stable=True)]
        coords_flat = coords_flat[idx]
        high_flat = high_flat[idx]
        high = high_flat.unsqueeze(0)
        high_grid = high.view(1, len(s1_unique), len(s2_unique), high.shape[-1])
        ds = s1_unique[1:] - s1_unique[:-1]
        ds = ds[ds > tol]
        ds = ds.view(1, len(s1_unique) - 1, 1, 1)
        a = (high_grid[:, 1:-1, :, :] - high_grid[:, :-2, :, :]) / ds[:, :-1, :, :]
        b = (high_grid[:, 2:, :, :] - high_grid[:, 1:-1, :, :]) / ds[:, 1:, :, :]
        grad_0 = (high_grid[:, 1, :, :] - high_grid[:, 0, :, :]) / ds[:, 0, :, :]
        grad_N = (high_grid[:, -1, :, :] - high_grid[:, -2, :, :]) / ds[:, -1, :, :]
        ab = a * b
        w = torch.tanh(10 * torch.relu(ab)) ** 2
        sigma_candidate = torch.where(
            torch.linalg.norm(a, dim=-1, keepdim=True) < torch.linalg.norm(b, dim=-1, keepdim=True), a, b
        )
        sigma = w * sigma_candidate
        sigma_full = torch.zeros_like(high_grid)
        sigma_full[:, 1:-1, :, :] = sigma
        sigma_full[:, 0, :, :] = grad_0
        sigma_full[:, -1, :, :] = grad_N
        limiter = sigma_full[:, 1:-1, :, :] - sigma_full[:, :-2, :, :]
        limiter = torch.sqrt(torch.mean(limiter ** 2, dim=-1, keepdim=True) + 1e-08)
        limiter_full = torch.zeros_like(high_grid[..., :1])
        limiter_full[:, 1:-1, :, :] = limiter
        limiter_full[:, 0, :, :] = limiter_full[:, 1, :, :]
        limiter_full[:, -1, :, :] = limiter_full[:, -2, :, :]
        limiter = limiter_full.reshape(high.shape[0], high.shape[1], 1)
        sigma_full = sigma_full.reshape(high.shape[0], high.shape[1], high.shape[2])
        gradient_norm = torch.linalg.norm(sigma_full, dim=-1, keepdim=True)
        return (gradient_norm, limiter)

    def overwrite_corner(
        self, z, high, mask_corner, mask_boundary, lim_boundary, grad_boundary, mask_other, lim_other,
        grad_other, sharp_const
    ):
        corner_global_idx = torch.where(mask_corner[0])[0]
        if corner_global_idx.numel() == 0:
            return z
        boundary_global_idx = torch.where(mask_boundary[0])[0]
        other_global_idx = torch.where(mask_other[0])[0]
        boundary_local_idx = torch.searchsorted(boundary_global_idx, corner_global_idx)
        other_local_idx = torch.searchsorted(other_global_idx, corner_global_idx)
        valid_boundary = boundary_local_idx < boundary_global_idx.numel()
        valid_other = other_local_idx < other_global_idx.numel()
        if not torch.all(valid_boundary & valid_other):
            raise RuntimeError('Some corner points are not included in both boundary masks.')
        if not torch.equal(boundary_global_idx[boundary_local_idx], corner_global_idx):
            raise RuntimeError('Corner and first boundary indices are inconsistent.')
        if not torch.equal(other_global_idx[other_local_idx], corner_global_idx):
            raise RuntimeError('Corner and second boundary indices are inconsistent.')
        limiter_y = lim_boundary[:, boundary_local_idx, :]
        grad_y = grad_boundary[:, boundary_local_idx, :]
        limiter_x = lim_other[:, other_local_idx, :]
        grad_x = grad_other[:, other_local_idx, :]
        limiter = torch.cat([limiter_x, limiter_y], dim=-1)
        low = self.low_proj(limiter)
        gradient_norm = torch.sqrt(grad_x.square() + grad_y.square() + 1e-08)
        gate = torch.tanh(self.alpha * gradient_norm.pow(1.5)).square()
        z[:, corner_global_idx, :] = (1.0 + sharp_const * gate * low) * high[:, corner_global_idx, :]
        return z

    def forward(self, coords, sharp_const, mode, mask):
        if mode == 'domain' or mode == 'initial':
            tol = 1e-08
            x = coords[..., 0]
            y = coords[..., 1]
            t = coords[..., 2]
            x_unique = torch.unique(x.detach())
            y_unique = torch.unique(y.detach())
            t_unique = torch.unique(t.detach())
            x_unique, _ = torch.sort(x_unique)
            y_unique, _ = torch.sort(y_unique)
            t_unique, _ = torch.sort(t_unique)
            dx = x_unique[1:] - x_unique[:-1]
            dy = y_unique[1:] - y_unique[:-1]
            dx = dx[dx > tol]
            dy = dy[dy > tol]
            high = self.high_proj(coords)
            coords_flat = coords[0]
            high_flat = high[0]
            x = coords_flat[:, 0]
            y = coords_flat[:, 1]
            t = coords_flat[:, 2]
            idx = torch.argsort(t, stable=True)
            idx = idx[torch.argsort(y[idx], stable=True)]
            idx = idx[torch.argsort(x[idx], stable=True)]
            coords_flat = coords_flat[idx]
            high_flat = high_flat[idx]
            coords = coords_flat.unsqueeze(0)
            high = high_flat.unsqueeze(0)
            high_grid = high.view(1, len(x_unique), len(y_unique), len(t_unique), high.shape[-1])
            dx = dx.view(1, len(x_unique) - 1, 1, 1, 1)
            dy = dy.view(1, 1, len(y_unique) - 1, 1, 1)
            a_x = (high_grid[:, 1:-1, :, :, :] - high_grid[:, :-2, :, :, :]) / dx[:, :-1, :, :, :]
            b_x = (high_grid[:, 2:, :, :, :] - high_grid[:, 1:-1, :, :, :]) / dx[:, 1:, :, :, :]
            grad_x0 = (high_grid[:, 1, :, :, :] - high_grid[:, 0, :, :, :]) / dx[:, 0, :, :, :]
            grad_xN = (high_grid[:, -1, :, :, :] - high_grid[:, -2, :, :, :]) / dx[:, -1, :, :, :]
            ab_x = a_x * b_x
            w_x = torch.tanh(10 * torch.relu(ab_x)) ** 2
            sigma_x_candidate = torch.where(
                torch.linalg.norm(a_x, dim=-1, keepdim=True) < torch.linalg.norm(b_x, dim=-1, keepdim=True),
                a_x, b_x
            )
            sigma_x = w_x * sigma_x_candidate
            sigma_x_full = torch.zeros_like(high_grid)
            sigma_x_full[:, 1:-1, :, :, :] = sigma_x
            sigma_x_full[:, 0, :, :, :] = grad_x0
            sigma_x_full[:, -1, :, :, :] = grad_xN
            limiter_x = sigma_x_full[:, 1:-1, :, :, :] - sigma_x_full[:, :-2, :, :, :]
            a_y = (high_grid[:, :, 1:-1, :, :] - high_grid[:, :, :-2, :, :]) / dy[:, :, :-1, :, :]
            b_y = (high_grid[:, :, 2:, :, :] - high_grid[:, :, 1:-1, :, :]) / dy[:, :, 1:, :, :]
            grad_y0 = (high_grid[:, :, 1, :, :] - high_grid[:, :, 0, :, :]) / dy[:, :, 0, :, :]
            grad_yN = (high_grid[:, :, -1, :, :] - high_grid[:, :, -2, :, :]) / dy[:, :, -1, :, :]
            ab_y = a_y * b_y
            w_y = torch.tanh(10 * torch.relu(ab_y)) ** 2
            sigma_y_candidate = torch.where(
                torch.linalg.norm(a_y, dim=-1, keepdim=True) < torch.linalg.norm(b_y, dim=-1, keepdim=True),
                a_y, b_y
            )
            sigma_y = w_y * sigma_y_candidate
            sigma_y_full = torch.zeros_like(high_grid)
            sigma_y_full[:, :, 1:-1, :, :] = sigma_y
            sigma_y_full[:, :, 0, :, :] = grad_y0
            sigma_y_full[:, :, -1, :, :] = grad_yN
            limiter_y = sigma_y_full[:, :, 1:-1, :, :] - sigma_y_full[:, :, :-2, :, :]
            limiter_x = limiter_x[:, :, 1:-1, :, :]
            limiter_y = limiter_y[:, 1:-1, :, :, :]
            limiter_x_full = torch.zeros_like(high_grid)
            limiter_x_full[:, 1:-1, 1:-1, :, :] = limiter_x
            limiter_x_full[:, 0, :, :, :] = limiter_x_full[:, 1, :, :, :]
            limiter_x_full[:, -1, :, :, :] = limiter_x_full[:, -2, :, :, :]
            limiter_x_full[:, :, 0, :, :] = limiter_x_full[:, :, 1, :, :]
            limiter_x_full[:, :, -1, :, :] = limiter_x_full[:, :, -2, :, :]
            limiter_x_full = limiter_x_full.reshape(high.shape[0], high.shape[1], high.shape[2])
            limiter_y_full = torch.zeros_like(high_grid)
            limiter_y_full[:, 1:-1, 1:-1, :, :] = limiter_y
            limiter_y_full[:, 0, :, :, :] = limiter_y_full[:, 1, :, :, :]
            limiter_y_full[:, -1, :, :, :] = limiter_y_full[:, -2, :, :, :]
            limiter_y_full[:, :, 0, :, :] = limiter_y_full[:, :, 1, :, :]
            limiter_y_full[:, :, -1, :, :] = limiter_y_full[:, :, -2, :, :]
            limiter_y_full = limiter_y_full.reshape(high.shape[0], high.shape[1], high.shape[2])
            limiter_x_mag = torch.sqrt(torch.mean(limiter_x_full ** 2, dim=-1, keepdim=True) + 1e-08)
            limiter_y_mag = torch.sqrt(torch.mean(limiter_y_full ** 2, dim=-1, keepdim=True) + 1e-08)
            limiter = torch.cat([limiter_x_mag, limiter_y_mag], dim=-1)
            grad_x_norm = torch.linalg.norm(sigma_x_full, dim=-1, keepdim=True)
            grad_y_norm = torch.linalg.norm(sigma_y_full, dim=-1, keepdim=True)
            gradient_norm = torch.sqrt(grad_x_norm ** 2 + grad_y_norm ** 2 + 1e-08)
            gate = torch.tanh(self.alpha * gradient_norm ** 1.5) ** 2
            gate = gate.reshape(high.shape[0], high.shape[1], 1)
            low = self.low_proj(limiter)
            z = (1 + sharp_const * gate * low) * high
            z_inner = z[:, mask, :]
            gate_inner = gate[:, mask, :]
            m_inner = limiter[:, mask, :]
            low_inner = low[:, mask, :]
            return (z_inner, gate_inner, m_inner, low_inner)
        elif mode == 'boundary':
            high = self.high_proj(coords)
            x = coords[..., 0]
            y = coords[..., 1]
            tol = 1e-06
            if mask is None:
                raise ValueError('Boundary mode requires boundary_inner_mask to infer the physical x/y bounds.')
            boundary_mask = mask
            if boundary_mask.ndim > 1:
                boundary_mask = boundary_mask.reshape(-1)
            boundary_mask = boundary_mask.to(device=coords.device, dtype=torch.bool)
            if boundary_mask.numel() != coords.shape[1]:
                raise ValueError(f'boundary_inner_mask has incompatible size: {boundary_mask.numel()} vs {coords.shape[1]} boundary-support points.')
            if not torch.any(boundary_mask):
                raise ValueError('boundary_inner_mask contains no physical boundary points.')
            physical_boundary_coords = coords[:, boundary_mask, :]
            physical_x = physical_boundary_coords[..., 0]
            physical_y = physical_boundary_coords[..., 1]
            x0_value = physical_x.amin().detach()
            x1_value = physical_x.amax().detach()
            y0_value = physical_y.amin().detach()
            y1_value = physical_y.amax().detach()
            mask_x0 = torch.isclose(x, x0_value, atol=tol)
            mask_x1 = torch.isclose(x, x1_value, atol=tol)
            mask_y0 = torch.isclose(y, y0_value, atol=tol)
            mask_y1 = torch.isclose(y, y1_value, atol=tol)
            mask_corner00 = mask_x0 & mask_y0
            mask_corner01 = mask_x0 & mask_y1
            mask_corner10 = mask_x1 & mask_y0
            mask_corner11 = mask_x1 & mask_y1
            z = torch.zeros_like(high)
            coords_x0 = coords[:, mask_x0[0], :]
            high_x0 = high[:, mask_x0[0], :]
            gn_x0, lim_x0 = self.slope_limiter_2d(high_x0, coords_x0, axis='y')
            gate_x0 = torch.tanh(self.alpha * gn_x0 ** 1.5) ** 2
            lim_x0_cat = torch.cat([torch.zeros_like(lim_x0), lim_x0], dim=-1)
            low_x0 = self.low_proj(lim_x0_cat)
            z[:, mask_x0[0], :] = (1 + sharp_const * gate_x0 * low_x0) * high_x0
            coords_x1 = coords[:, mask_x1[0], :]
            high_x1 = high[:, mask_x1[0], :]
            gn_x1, lim_x1 = self.slope_limiter_2d(high_x1, coords_x1, axis='y')
            gate_x1 = torch.tanh(self.alpha * gn_x1 ** 1.5) ** 2
            lim_x1_cat = torch.cat([torch.zeros_like(lim_x1), lim_x1], dim=-1)
            low_x1 = self.low_proj(lim_x1_cat)
            z[:, mask_x1[0], :] = (1 + sharp_const * gate_x1 * low_x1) * high_x1
            coords_y0 = coords[:, mask_y0[0], :]
            high_y0 = high[:, mask_y0[0], :]
            gn_y0, lim_y0 = self.slope_limiter_2d(high_y0, coords_y0, axis='x')
            gate_y0 = torch.tanh(self.alpha * gn_y0 ** 1.5) ** 2
            lim_y0_cat = torch.cat([lim_y0, torch.zeros_like(lim_y0)], dim=-1)
            low_y0 = self.low_proj(lim_y0_cat)
            z[:, mask_y0[0], :] = (1 + sharp_const * gate_y0 * low_y0) * high_y0
            coords_y1 = coords[:, mask_y1[0], :]
            high_y1 = high[:, mask_y1[0], :]
            gn_y1, lim_y1 = self.slope_limiter_2d(high_y1, coords_y1, axis='x')
            gate_y1 = torch.tanh(self.alpha * gn_y1 ** 1.5) ** 2
            lim_y1_cat = torch.cat([lim_y1, torch.zeros_like(lim_y1)], dim=-1)
            low_y1 = self.low_proj(lim_y1_cat)
            z[:, mask_y1[0], :] = (1 + sharp_const * gate_y1 * low_y1) * high_y1
            z = self.overwrite_corner(
                z, high, mask_corner00, mask_x0, lim_x0, gn_x0, mask_y0, lim_y0, gn_y0, sharp_const
            )
            z = self.overwrite_corner(
                z, high, mask_corner01, mask_x0, lim_x0, gn_x0, mask_y1, lim_y1, gn_y1, sharp_const
            )
            z = self.overwrite_corner(
                z, high, mask_corner10, mask_x1, lim_x1, gn_x1, mask_y0, lim_y0, gn_y0, sharp_const
            )
            z = self.overwrite_corner(
                z, high, mask_corner11, mask_x1, lim_x1, gn_x1, mask_y1, lim_y1, gn_y1, sharp_const
            )
            z_inner = z[:, mask, :]
            return (z_inner, None, None, None)

class FixedSharpSlopeLimiterEmbeddingAnalysis3D_NoSlope(nn.Module):

    def __init__(self, scale, in_dim, embed_dim):
        super().__init__()
        self.high_proj = nn.Sequential(
            Trainable_FourierFeatureEmbedding(input_dim=in_dim, mapping_size=embed_dim // 2, scale=scale),
            nn.Linear(embed_dim, embed_dim)
        )

    def slope_limiter_2d(self, high, coords, axis='y'):
        tol = 1e-08
        coords_flat = coords[0]
        high_flat = high[0]
        if axis == 'y':
            s1 = coords_flat[:, 1]
            s2 = coords_flat[:, 2]
        else:
            s1 = coords_flat[:, 0]
            s2 = coords_flat[:, 2]
        s1_unique = torch.unique(s1.detach())
        s2_unique = torch.unique(s2.detach())
        s1_unique, _ = torch.sort(s1_unique)
        s2_unique, _ = torch.sort(s2_unique)
        idx = torch.argsort(s2, stable=True)
        idx = idx[torch.argsort(s1[idx], stable=True)]
        coords_flat = coords_flat[idx]
        high_flat = high_flat[idx]
        high = high_flat.unsqueeze(0)
        high_grid = high.view(1, len(s1_unique), len(s2_unique), high.shape[-1])
        ds = s1_unique[1:] - s1_unique[:-1]
        ds = ds[ds > tol]
        ds = ds.view(1, len(s1_unique) - 1, 1, 1)
        a = (high_grid[:, 1:-1, :, :] - high_grid[:, :-2, :, :]) / ds[:, :-1, :, :]
        b = (high_grid[:, 2:, :, :] - high_grid[:, 1:-1, :, :]) / ds[:, 1:, :, :]
        grad_0 = (high_grid[:, 1, :, :] - high_grid[:, 0, :, :]) / ds[:, 0, :, :]
        grad_N = (high_grid[:, -1, :, :] - high_grid[:, -2, :, :]) / ds[:, -1, :, :]
        ab = a * b
        w = torch.tanh(10 * torch.relu(ab)) ** 2
        sigma_candidate = torch.where(
            torch.linalg.norm(a, dim=-1, keepdim=True) < torch.linalg.norm(b, dim=-1, keepdim=True), a, b
        )
        sigma = w * sigma_candidate
        sigma_full = torch.zeros_like(high_grid)
        sigma_full[:, 1:-1, :, :] = sigma
        sigma_full[:, 0, :, :] = grad_0
        sigma_full[:, -1, :, :] = grad_N
        limiter = sigma_full[:, 1:-1, :, :] - sigma_full[:, :-2, :, :]
        limiter = torch.sqrt(torch.mean(limiter ** 2, dim=-1, keepdim=True) + 1e-08)
        limiter_full = torch.zeros_like(high_grid[..., :1])
        limiter_full[:, 1:-1, :, :] = limiter
        limiter_full[:, 0, :, :] = limiter_full[:, 1, :, :]
        limiter_full[:, -1, :, :] = limiter_full[:, -2, :, :]
        limiter = limiter_full.reshape(high.shape[0], high.shape[1], 1)
        sigma_full = sigma_full.reshape(high.shape[0], high.shape[1], high.shape[2])
        gradient_norm = torch.linalg.norm(sigma_full, dim=-1, keepdim=True)
        return (gradient_norm, limiter)

    def forward(self, coords, sharp_const, mode, mask):
        if mode == 'domain' or mode == 'initial':
            tol = 1e-08
            x = coords[..., 0]
            y = coords[..., 1]
            t = coords[..., 2]
            x_unique = torch.unique(x.detach())
            y_unique = torch.unique(y.detach())
            t_unique = torch.unique(t.detach())
            x_unique, _ = torch.sort(x_unique)
            y_unique, _ = torch.sort(y_unique)
            t_unique, _ = torch.sort(t_unique)
            dx = x_unique[1:] - x_unique[:-1]
            dy = y_unique[1:] - y_unique[:-1]
            dx = dx[dx > tol]
            dy = dy[dy > tol]
            high = self.high_proj(coords)
            coords_flat = coords[0]
            high_flat = high[0]
            x = coords_flat[:, 0]
            y = coords_flat[:, 1]
            t = coords_flat[:, 2]
            idx = torch.argsort(t, stable=True)
            idx = idx[torch.argsort(y[idx], stable=True)]
            idx = idx[torch.argsort(x[idx], stable=True)]
            coords_flat = coords_flat[idx]
            high_flat = high_flat[idx]
            coords = coords_flat.unsqueeze(0)
            high = high_flat.unsqueeze(0)
            high_grid = high.view(1, len(x_unique), len(y_unique), len(t_unique), high.shape[-1])
            dx = dx.view(1, len(x_unique) - 1, 1, 1, 1)
            dy = dy.view(1, 1, len(y_unique) - 1, 1, 1)
            a_x = (high_grid[:, 1:-1, :, :, :] - high_grid[:, :-2, :, :, :]) / dx[:, :-1, :, :, :]
            b_x = (high_grid[:, 2:, :, :, :] - high_grid[:, 1:-1, :, :, :]) / dx[:, 1:, :, :, :]
            grad_x0 = (high_grid[:, 1, :, :, :] - high_grid[:, 0, :, :, :]) / dx[:, 0, :, :, :]
            grad_xN = (high_grid[:, -1, :, :, :] - high_grid[:, -2, :, :, :]) / dx[:, -1, :, :, :]
            ab_x = a_x * b_x
            w_x = torch.tanh(10 * torch.relu(ab_x)) ** 2
            sigma_x_candidate = torch.where(
                torch.linalg.norm(a_x, dim=-1, keepdim=True) < torch.linalg.norm(b_x, dim=-1, keepdim=True),
                a_x, b_x
            )
            sigma_x = w_x * sigma_x_candidate
            sigma_x_full = torch.zeros_like(high_grid)
            sigma_x_full[:, 1:-1, :, :, :] = sigma_x
            sigma_x_full[:, 0, :, :, :] = grad_x0
            sigma_x_full[:, -1, :, :, :] = grad_xN
            limiter_x = sigma_x_full[:, 1:-1, :, :, :] - sigma_x_full[:, :-2, :, :, :]
            a_y = (high_grid[:, :, 1:-1, :, :] - high_grid[:, :, :-2, :, :]) / dy[:, :, :-1, :, :]
            b_y = (high_grid[:, :, 2:, :, :] - high_grid[:, :, 1:-1, :, :]) / dy[:, :, 1:, :, :]
            grad_y0 = (high_grid[:, :, 1, :, :] - high_grid[:, :, 0, :, :]) / dy[:, :, 0, :, :]
            grad_yN = (high_grid[:, :, -1, :, :] - high_grid[:, :, -2, :, :]) / dy[:, :, -1, :, :]
            ab_y = a_y * b_y
            w_y = torch.tanh(10 * torch.relu(ab_y)) ** 2
            sigma_y_candidate = torch.where(
                torch.linalg.norm(a_y, dim=-1, keepdim=True) < torch.linalg.norm(b_y, dim=-1, keepdim=True),
                a_y, b_y
            )
            sigma_y = w_y * sigma_y_candidate
            sigma_y_full = torch.zeros_like(high_grid)
            sigma_y_full[:, :, 1:-1, :, :] = sigma_y
            sigma_y_full[:, :, 0, :, :] = grad_y0
            sigma_y_full[:, :, -1, :, :] = grad_yN
            limiter_y = sigma_y_full[:, :, 1:-1, :, :] - sigma_y_full[:, :, :-2, :, :]
            limiter_x = limiter_x[:, :, 1:-1, :, :]
            limiter_y = limiter_y[:, 1:-1, :, :, :]
            limiter_x_full = torch.zeros_like(high_grid)
            limiter_x_full[:, 1:-1, 1:-1, :, :] = limiter_x
            limiter_x_full[:, 0, :, :, :] = limiter_x_full[:, 1, :, :, :]
            limiter_x_full[:, -1, :, :, :] = limiter_x_full[:, -2, :, :, :]
            limiter_x_full[:, :, 0, :, :] = limiter_x_full[:, :, 1, :, :]
            limiter_x_full[:, :, -1, :, :] = limiter_x_full[:, :, -2, :, :]
            limiter_x_full = limiter_x_full.reshape(high.shape[0], high.shape[1], high.shape[2])
            limiter_y_full = torch.zeros_like(high_grid)
            limiter_y_full[:, 1:-1, 1:-1, :, :] = limiter_y
            limiter_y_full[:, 0, :, :, :] = limiter_y_full[:, 1, :, :, :]
            limiter_y_full[:, -1, :, :, :] = limiter_y_full[:, -2, :, :, :]
            limiter_y_full[:, :, 0, :, :] = limiter_y_full[:, :, 1, :, :]
            limiter_y_full[:, :, -1, :, :] = limiter_y_full[:, :, -2, :, :]
            limiter_y_full = limiter_y_full.reshape(high.shape[0], high.shape[1], high.shape[2])
            limiter_x_mag = torch.sqrt(torch.mean(limiter_x_full ** 2, dim=-1, keepdim=True) + 1e-08)
            limiter_y_mag = torch.sqrt(torch.mean(limiter_y_full ** 2, dim=-1, keepdim=True) + 1e-08)
            limiter = torch.cat([limiter_x_mag, limiter_y_mag], dim=-1)
            z_inner = high[:, mask, :]
            m_inner = limiter[:, mask, :]
            return (z_inner, m_inner)
        elif mode == 'boundary':
            high = self.high_proj(coords)
            x = coords[..., 0]
            y = coords[..., 1]
            tol = 1e-06
            if mask is None:
                raise ValueError('Boundary mode requires boundary_inner_mask to infer the physical x/y bounds.')
            boundary_mask = mask
            if boundary_mask.ndim > 1:
                boundary_mask = boundary_mask.reshape(-1)
            boundary_mask = boundary_mask.to(device=coords.device, dtype=torch.bool)
            if boundary_mask.numel() != coords.shape[1]:
                raise ValueError(f'boundary_inner_mask has incompatible size: {boundary_mask.numel()} vs {coords.shape[1]} boundary-support points.')
            if not torch.any(boundary_mask):
                raise ValueError('boundary_inner_mask contains no physical boundary points.')
            physical_boundary_coords = coords[:, boundary_mask, :]
            physical_x = physical_boundary_coords[..., 0]
            physical_y = physical_boundary_coords[..., 1]
            x0_value = physical_x.amin().detach()
            x1_value = physical_x.amax().detach()
            y0_value = physical_y.amin().detach()
            y1_value = physical_y.amax().detach()
            mask_x0 = torch.isclose(x, x0_value, atol=tol)
            mask_x1 = torch.isclose(x, x1_value, atol=tol)
            mask_y0 = torch.isclose(y, y0_value, atol=tol)
            mask_y1 = torch.isclose(y, y1_value, atol=tol)
            mask_corner00 = mask_x0 & mask_y0
            mask_corner01 = mask_x0 & mask_y1
            mask_corner10 = mask_x1 & mask_y0
            mask_corner11 = mask_x1 & mask_y1
            m = torch.zeros_like(high[..., :2])
            _, lim_x0 = self.slope_limiter_2d(high[:, mask_x0[0], :], coords[:, mask_x0[0], :], axis='y')
            _, lim_x1 = self.slope_limiter_2d(high[:, mask_x1[0], :], coords[:, mask_x1[0], :], axis='y')
            _, lim_y0 = self.slope_limiter_2d(high[:, mask_y0[0], :], coords[:, mask_y0[0], :], axis='x')
            _, lim_y1 = self.slope_limiter_2d(high[:, mask_y1[0], :], coords[:, mask_y1[0], :], axis='x')
            m[:, mask_x0[0], :] = torch.cat([torch.zeros_like(lim_x0), lim_x0], dim=-1)
            m[:, mask_x1[0], :] = torch.cat([torch.zeros_like(lim_x1), lim_x1], dim=-1)
            m[:, mask_y0[0], :] = torch.cat([lim_y0, torch.zeros_like(lim_y0)], dim=-1)
            m[:, mask_y1[0], :] = torch.cat([lim_y1, torch.zeros_like(lim_y1)], dim=-1)
            for corner, side_x, lim_x, side_y, lim_y in (
                (mask_corner00, mask_x0, lim_x0, mask_y0, lim_y0),
                (mask_corner01, mask_x0, lim_x0, mask_y1, lim_y1),
                (mask_corner10, mask_x1, lim_x1, mask_y0, lim_y0),
                (mask_corner11, mask_x1, lim_x1, mask_y1, lim_y1)
            ):
                corner_idx = torch.where(corner[0])[0]
                if corner_idx.numel() == 0:
                    continue
                x_idx = torch.searchsorted(torch.where(side_x[0])[0], corner_idx)
                y_idx = torch.searchsorted(torch.where(side_y[0])[0], corner_idx)
                m[:, corner_idx, :] = torch.cat([lim_y[:, y_idx, :], lim_x[:, x_idx, :]], dim=-1)
            return (high[:, mask, :], m[:, mask, :])

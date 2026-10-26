import torch
import numpy as np
from scipy.interpolate import RegularGridInterpolator

def solve_allen_cahn_explicit(Nx=512, dt=0.001, T=1.0, D=0.0001, beta=5.0, save_every=10):
    L = 2.0
    x = -1.0 + L * np.arange(Nx) / Nx
    h = L / Nx
    phi = x ** 2 * np.cos(np.pi * x)
    nsteps = int(np.round(T / dt))
    times = []
    snaps = []

    def save(step):
        times.append(step * dt)
        snaps.append(phi.copy())
    save(0)
    if D > 0:
        dt_max_diff = h * h / (2.0 * D)
        if dt > dt_max_diff:
            print(f'[warning] dt={dt:g} is larger than diffusion stability bound ~ {dt_max_diff:g}. Consider reducing dt or increasing Nx.')
    for n in range(1, nsteps + 1):
        phi_left = np.roll(phi, 1)
        phi_right = np.roll(phi, -1)
        lap = (phi_left - 2.0 * phi + phi_right) / (h * h)
        reaction = -beta * (phi ** 3 - phi)
        phi = phi + dt * (reaction + D * lap)
        if n % save_every == 0:
            save(n)
    return (x, np.array(times), np.array(snaps))

class AllenCahnReference:

    def __init__(self, Nx=512, dt=0.001, T=1.0, D=0.0001, beta=5.0, save_every=10):
        self.x_ref, self.t_ref, self.phi_hist = solve_allen_cahn_explicit(
            Nx=Nx, dt=dt, T=T, D=D, beta=beta, save_every=save_every
        )
        self.interpolator = RegularGridInterpolator(
            (self.t_ref, self.x_ref), self.phi_hist, bounds_error=False, fill_value=None
        )

    def evaluate_tensor(self, coords):
        coords_np = coords.detach().cpu().numpy()
        t = coords_np[..., 1]
        x = coords_np[..., 0]
        points = np.stack([t.reshape(-1), x.reshape(-1)], axis=-1)
        u = self.interpolator(points)
        return torch.from_numpy(u.reshape(coords.shape[:-1] + (1,))).to(coords.device)

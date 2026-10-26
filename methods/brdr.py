from __future__ import annotations
import torch

class BRDRWeights:

    def __init__(self, beta_residual: float=0.999, beta_weight: float=0.999, eps: float=1e-14) -> None:
        if not 0.0 <= beta_residual < 1.0:
            raise ValueError('beta_residual must be in [0, 1).')
        if not 0.0 <= beta_weight < 1.0:
            raise ValueError('beta_weight must be in [0, 1).')
        if eps <= 0.0:
            raise ValueError('eps must be positive.')
        self.beta_residual = float(beta_residual)
        self.beta_weight = float(beta_weight)
        self.eps = float(eps)
        self.ema_residual_fourth: dict[str, torch.Tensor] = {}
        self.weights: dict[str, torch.Tensor] = {}
        self.num_updates = 0

    @torch.no_grad()
    def _initialize_group(self, name: str, residual: torch.Tensor) -> None:
        self.ema_residual_fourth[name] = torch.zeros_like(residual)
        self.weights[name] = torch.ones_like(residual)

    @torch.no_grad()
    def update(self, residuals: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        if not residuals:
            raise ValueError('residuals must contain at least one residual group.')
        self.num_updates += 1
        bias_correction = 1.0 - self.beta_residual ** self.num_updates
        inverse_decay: dict[str, torch.Tensor] = {}
        for name, residual in residuals.items():
            residual_detached = residual.detach()
            if name not in self.weights:
                self._initialize_group(name, residual_detached)
            if self.weights[name].shape != residual_detached.shape:
                raise ValueError(f'BRDR shape mismatch for {name!r}: stored={tuple(self.weights[name].shape)}, current={tuple(residual_detached.shape)}.')
            residual_fourth = residual_detached.pow(4)
            ema_fourth = self.beta_residual * self.ema_residual_fourth[name] + (1.0 - self.beta_residual) * residual_fourth
            self.ema_residual_fourth[name].copy_(ema_fourth)
            corrected_ema_fourth = ema_fourth / max(bias_correction, self.eps)
            inverse_decay[name] = residual_detached.square() / (torch.sqrt(corrected_ema_fourth) + self.eps)
        flat_inverse = torch.cat([value.reshape(-1) for value in inverse_decay.values()], dim=0)
        total_points = flat_inverse.numel()
        inverse_sum = flat_inverse.sum().clamp_min(self.eps)
        global_scale = total_points / inverse_sum
        for name, inverse in inverse_decay.items():
            target_weight = global_scale * inverse
            updated_weight = self.beta_weight * self.weights[name] + (1.0 - self.beta_weight) * target_weight
            self.weights[name].copy_(updated_weight)
        return {name: weight.detach() for name, weight in self.weights.items()}

    def get(self, name: str) -> torch.Tensor:
        if name not in self.weights:
            raise KeyError(f'BRDR group {name!r} has not been initialized.')
        return self.weights[name]

    def get_all(self) -> dict[str, torch.Tensor]:
        if not self.weights:
            raise RuntimeError('BRDR weights have not been initialized. Call update() at least once before get_all().')
        return {name: weight.detach() for name, weight in self.weights.items()}

    def state_dict(self) -> dict:
        return {
            'beta_residual': self.beta_residual, 'beta_weight': self.beta_weight, 'eps': self.eps,
            'num_updates': self.num_updates,
            'ema_residual_fourth': {key: value.clone() for key, value in self.ema_residual_fourth.items()},
            'weights': {key: value.clone() for key, value in self.weights.items()}
        }

    def load_state_dict(self, state: dict, device=None) -> None:
        self.beta_residual = float(state['beta_residual'])
        self.beta_weight = float(state['beta_weight'])
        self.eps = float(state['eps'])
        self.num_updates = int(state.get('num_updates', 0))
        self.ema_residual_fourth = {
            key: value.clone().to(device) for key, value in state['ema_residual_fourth'].items()
        }
        self.weights = {key: value.clone().to(device) for key, value in state['weights'].items()}

class BRDRScale:

    def __init__(self, beta_scale=0.999, eps=1e-14):
        self.beta_scale = beta_scale
        self.scale = 1.0
        self.eps = eps

    def get_scale(self):
        return self.scale

    @torch.no_grad()
    def update(self, loss, grad_norm_sq, lr, beta_scale):
        self.beta_scale = beta_scale
        s_max = self.scale * (2.0 * loss / (lr * grad_norm_sq + self.eps))
        self.scale = self.beta_scale * self.scale + (1 - self.beta_scale) * s_max.item()
        return self.scale

    def state_dict(self):
        return {'scale': self.scale, 'beta_scale': self.beta_scale, 'eps': self.eps}

    def load_state_dict(self, state):
        self.scale = float(state['scale'])
        self.beta_scale = float(state['beta_scale'])
        self.eps = float(state['eps'])

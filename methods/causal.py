from __future__ import annotations
from dataclasses import dataclass
import torch

@dataclass
class CausalLossOutput:
    total_loss: torch.Tensor
    pde_loss: torch.Tensor
    initial_loss: torch.Tensor
    temporal_losses: torch.Tensor
    temporal_weights: torch.Tensor
    time_centers: torch.Tensor
    min_temporal_weight: torch.Tensor

class CausalPINNWeighting:

    def __init__(
        self, epsilon: float=1.0, num_time_bins: Optional[int]=None, include_initial: bool=True,
        normalization: str='paper', eps: float=1e-12
    ) -> None:
        if epsilon <= 0.0:
            raise ValueError('epsilon must be positive.')
        if num_time_bins is not None and num_time_bins < 1:
            raise ValueError('num_time_bins must be positive.')
        if normalization not in {'paper', 'separate'}:
            raise ValueError("normalization must be 'paper' or 'separate'.")
        self.epsilon = float(epsilon)
        self.num_time_bins = num_time_bins
        self.include_initial = bool(include_initial)
        self.normalization = normalization
        self.eps = float(eps)
        self.last_temporal_losses: Optional[torch.Tensor] = None
        self.last_temporal_weights: Optional[torch.Tensor] = None
        self.last_time_centers: Optional[torch.Tensor] = None

    def set_epsilon(self, epsilon: float) -> None:
        if epsilon <= 0.0:
            raise ValueError('epsilon must be positive.')
        self.epsilon = float(epsilon)

    @staticmethod
    def _flatten_points(
        residual: torch.Tensor, time_coords: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if residual.ndim == 3:
            if residual.shape[0] != 1:
                raise ValueError('Expected batch dimension 1 for residual.')
            residual_flat = residual.squeeze(0).reshape(-1)
        else:
            residual_flat = residual.reshape(-1)
        if time_coords.ndim == 3:
            if time_coords.shape[0] != 1:
                raise ValueError('Expected batch dimension 1 for time coordinates.')
            time_flat = time_coords.squeeze(0).reshape(-1)
        else:
            time_flat = time_coords.reshape(-1)
        if residual_flat.numel() != time_flat.numel():
            raise ValueError(f'Residual and time coordinates must contain the same number of points: residual={residual_flat.numel()}, time={time_flat.numel()}.')
        return (residual_flat, time_flat)

    def _exact_time_groups(
        self, residual_sq: torch.Tensor, time_flat: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        unique_times, inverse = torch.unique(time_flat.detach(), sorted=True, return_inverse=True)
        num_times = unique_times.numel()
        loss_sum = torch.zeros(num_times, device=residual_sq.device, dtype=residual_sq.dtype)
        point_count = torch.zeros(num_times, device=residual_sq.device, dtype=residual_sq.dtype)
        loss_sum.scatter_add_(dim=0, index=inverse, src=residual_sq)
        point_count.scatter_add_(dim=0, index=inverse, src=torch.ones_like(residual_sq))
        temporal_losses = loss_sum / point_count.clamp_min(1.0)
        return (temporal_losses, unique_times)

    def _binned_time_groups(
        self, residual_sq: torch.Tensor, time_flat: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.num_time_bins is None:
            raise RuntimeError('num_time_bins is not defined.')
        t_min = time_flat.detach().amin()
        t_max = time_flat.detach().amax()
        if torch.isclose(t_min, t_max):
            return (residual_sq.mean().reshape(1), t_min.reshape(1))
        edges = torch.linspace(
            t_min, t_max, self.num_time_bins + 1, device=time_flat.device, dtype=time_flat.dtype
        )
        bin_ids = torch.bucketize(time_flat.detach(), edges[1:-1], right=False)
        num_bins = self.num_time_bins
        loss_sum = torch.zeros(num_bins, device=residual_sq.device, dtype=residual_sq.dtype)
        point_count = torch.zeros(num_bins, device=residual_sq.device, dtype=residual_sq.dtype)
        loss_sum.scatter_add_(0, bin_ids, residual_sq)
        point_count.scatter_add_(0, bin_ids, torch.ones_like(residual_sq))
        occupied = point_count > 0
        temporal_losses = loss_sum[occupied] / point_count[occupied]
        centers = 0.5 * (edges[:-1] + edges[1:])
        time_centers = centers[occupied]
        return (temporal_losses, time_centers)

    def temporal_residual_losses(
        self, residual: torch.Tensor, time_coords: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        residual_flat, time_flat = self._flatten_points(residual=residual, time_coords=time_coords)
        residual_sq = residual_flat.square()
        if self.num_time_bins is None:
            return self._exact_time_groups(residual_sq=residual_sq, time_flat=time_flat)
        return self._binned_time_groups(residual_sq=residual_sq, time_flat=time_flat)

    def compute_weights(self, causal_components: torch.Tensor) -> torch.Tensor:
        if causal_components.ndim != 1:
            raise ValueError('causal_components must be one-dimensional.')
        previous_sum = torch.cat(
            [
                torch.zeros(1, device=causal_components.device, dtype=causal_components.dtype),
                torch.cumsum(causal_components.detach()[:-1], dim=0)
            ],
            dim=0
        )
        exponent = -self.epsilon * previous_sum
        exponent = exponent.clamp(min=-80.0, max=0.0)
        return torch.exp(exponent).detach()

    def __call__(
        self, pde_residual: torch.Tensor, pde_time_coords: torch.Tensor, initial_residual: torch.Tensor,
        lambda_pde: float, lambda_ini: float, weighted_initial_loss: Optional[torch.Tensor]=None
    ) -> CausalLossOutput:
        temporal_losses, time_centers = self.temporal_residual_losses(
            residual=pde_residual, time_coords=pde_time_coords
        )
        initial_loss = initial_residual.square().mean()
        initial_component = float(lambda_ini) * initial_loss if weighted_initial_loss is None else weighted_initial_loss
        weighted_temporal_losses = float(lambda_pde) * temporal_losses
        if self.normalization == 'paper':
            if self.include_initial:
                causal_components = torch.cat([initial_component.reshape(1), weighted_temporal_losses], dim=0)
                all_weights = self.compute_weights(causal_components)
                total_loss = torch.sum(all_weights * causal_components) / temporal_losses.numel()
                temporal_weights = all_weights[1:]
            else:
                temporal_weights = self.compute_weights(weighted_temporal_losses)
                total_loss = torch.mean(temporal_weights * weighted_temporal_losses)
        else:
            if self.include_initial:
                previous_components = torch.cat(
                    [initial_component.reshape(1), weighted_temporal_losses], dim=0
                )
                all_weights = self.compute_weights(previous_components)
                temporal_weights = all_weights[1:]
            else:
                temporal_weights = self.compute_weights(weighted_temporal_losses)
            pde_loss = torch.mean(temporal_weights * temporal_losses)
            total_loss = initial_component + float(lambda_pde) * pde_loss
        pde_loss = torch.mean(temporal_weights * temporal_losses)
        self.last_temporal_losses = temporal_losses.detach()
        self.last_temporal_weights = temporal_weights.detach()
        self.last_time_centers = time_centers.detach()
        return CausalLossOutput(
            total_loss=total_loss, pde_loss=pde_loss, initial_loss=initial_loss,
            temporal_losses=temporal_losses, temporal_weights=temporal_weights, time_centers=time_centers,
            min_temporal_weight=temporal_weights.amin()
        )

    def state_dict(self) -> dict:
        return {
            'epsilon': self.epsilon, 'num_time_bins': self.num_time_bins,
            'include_initial': self.include_initial, 'normalization': self.normalization, 'eps': self.eps
        }

    def load_state_dict(self, state: dict) -> None:
        self.epsilon = float(state['epsilon'])
        self.num_time_bins = state['num_time_bins']
        self.include_initial = bool(state['include_initial'])
        self.normalization = str(state['normalization'])
        self.eps = float(state['eps'])

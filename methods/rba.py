import torch

class RBAWeights:

    def __init__(self, shape, device, gamma=0.999, eta=0.001, eps=1e-12, offset=0.0):
        self.gamma = gamma
        self.eta = eta
        self.eps = eps
        self.offset = offset
        self.lam = torch.zeros(shape, device=device)

    @torch.no_grad()
    def update(self, residual):
        r = residual.detach().abs()
        r = r / (r.max() + self.eps)
        self.lam.mul_(self.gamma)
        self.lam.add_(self.eta * r)
        return self.lam + self.offset

    @torch.no_grad()
    def update_batch(
        self, residual: torch.Tensor, global_indices: torch.Tensor, decay_all: bool=False
    ) -> torch.Tensor:
        r = residual.detach().abs()
        r = r / (r.amax() + self.eps)
        if decay_all:
            self.lam.mul_(self.gamma)
            current = self.lam.index_select(dim=1, index=global_indices)
            updated = current + self.eta * r
        else:
            current = self.lam.index_select(dim=1, index=global_indices)
            updated = self.gamma * current + self.eta * r
        self.lam.index_copy_(dim=1, index=global_indices, source=updated)
        return updated + self.offset

    def get_current(self):
        return self.lam + self.offset

    @torch.no_grad()
    def get_batch(self, global_indices: torch.Tensor) -> torch.Tensor:
        return torch.index_select(self.lam, dim=1, index=global_indices) + self.offset

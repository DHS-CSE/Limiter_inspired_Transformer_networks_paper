from embedding.sinusoidal import Sin
import torch.nn as nn
import torch

class Trainable_FourierFeatureEmbedding(nn.Module):

    def __init__(self, input_dim=2, mapping_size=64, scale=10.0):
        super().__init__()
        self.input_dim = input_dim
        self.mapping_size = mapping_size
        self.B = nn.Parameter(torch.randn(input_dim, mapping_size) * scale)

    def forward(self, x):
        projected = 2 * torch.pi * x @ self.B
        return torch.cat([torch.sin(projected), torch.cos(projected)], dim=-1)

class FourierWaveletActivation(nn.Module):

    def __init__(self, num_features, omega0=1.0):
        super(FourierWaveletActivation, self).__init__()
        self.omega0 = omega0
        self.w1 = nn.Parameter(torch.ones(1, num_features))
        self.w2 = nn.Parameter(torch.ones(1, num_features))

    def forward(self, x):
        x_scaled = self.omega0 * x
        return self.w1 * torch.sin(x_scaled) + self.w2 * torch.cos(x_scaled)

class LayerWiseLAAF(nn.Module):

    def __init__(self, activation=None, n=10.0, a_init=0.1):
        super().__init__()
        self.activation = activation if activation is not None else nn.Tanh()
        self.n = float(n)
        self.a = nn.Parameter(torch.tensor(float(a_init)))

    def forward(self, x):
        return self.activation(self.n * self.a * x)

    def effective_slope(self):
        return (self.n * self.a).detach().item()

class PointwiseMLP(nn.Module):

    def __init__(self, input_dim=1, hidden_dim=128, activation='GELU'):
        super().__init__()
        activation_name = activation.strip().lower()
        if activation_name not in {'gelu', 'tanh', 'sin', 'sine', 'silu', 'wavelet', 'laaf'}:
            raise ValueError(f'Unknown activation: {activation!r}')

        def make_activation():
            if activation_name == 'gelu':
                return nn.GELU()
            elif activation_name == 'tanh':
                return nn.Tanh()
            elif activation_name in {'sin', 'sine'}:
                return Sin()
            elif activation_name == 'silu':
                return nn.SiLU()
            elif activation_name == 'wavelet':
                return FourierWaveletActivation(hidden_dim)
            elif activation_name == 'laaf':
                return LayerWiseLAAF(n=10.0, a_init=0.1)
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), make_activation(), nn.Linear(hidden_dim, hidden_dim),
            make_activation(), nn.Linear(hidden_dim, hidden_dim), make_activation()
        )

    def forward(self, x):
        return self.net(x)

from embedding.embedding import Trainable_FourierFeatureEmbedding
import torch.nn as nn
import torch
import torch.nn.init as init
from .base import BaseAdapter
from registry import register_model

class Sine(nn.Module):

    def __init__(self, w0: float=1.0):
        super().__init__()
        self.w0 = w0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sin(self.w0 * x)

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

@register_model
class PINN(nn.Module):
    input_type = 'PINN'

    def __init__(self, layers, activation):
        super().__init__()
        if len(layers) < 2:
            raise ValueError('layers must contain input and output sizes.')
        activation_name = activation.strip().lower()
        if activation_name not in {'tanh', 'sin', 'sine', 'silu', 'laaf'}:
            raise ValueError(f'Unknown activation: {activation!r}')
        self.layers = nn.ModuleList([
            nn.Linear(in_features, out_features) for in_features, out_features in zip(layers[:-1], layers[1:])
        ])

        def make_activation():
            if activation_name == 'tanh':
                return nn.Tanh()
            elif activation_name in {'sin', 'sine'}:
                return Sine()
            elif activation_name == 'silu':
                return nn.SiLU()
            elif activation_name == 'laaf':
                return LayerWiseLAAF(n=10.0, a_init=1.0)
        self.activations = nn.ModuleList([make_activation() for _ in range(len(self.layers) - 1)])
        self.apply(self._init_weights)

    def forward(self, x):
        for layer, activation in zip(self.layers[:-1], self.activations):
            x = activation(layer(x))
        x = self.layers[-1](x)
        return x

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            init.xavier_normal_(m.weight)
            if m.bias is not None:
                init.constant_(m.bias, 0.0)

    @classmethod
    def from_args(cls, args):
        return cls(layers=args.layers, activation=args.activation)

@register_model
class PINNFourier(nn.Module):
    input_type = 'PINN'

    def __init__(self, layers, activation, input_dim, mapping_size, fourier_scale):
        super().__init__()
        self.fourier_feature = Trainable_FourierFeatureEmbedding(
            input_dim=input_dim, mapping_size=mapping_size, scale=fourier_scale
        )
        self.layers = nn.ModuleList()
        if activation == 'Tanh':
            self.activation = nn.Tanh()
        elif activation == 'Sin':
            self.activation = Sine()
        elif activation == 'Silu':
            self.activation = nn.SiLU()
        for i in range(len(layers) - 1):
            self.layers.append(nn.Linear(layers[i], layers[i + 1]))
        self.apply(self._init_weights)

    def forward(self, x):
        x = self.fourier_feature(x)
        for i in range(len(self.layers) - 1):
            x = self.activation(self.layers[i](x))
        x = self.layers[-1](x)
        return x

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            init.xavier_normal_(m.weight)
            if m.bias is not None:
                init.constant_(m.bias, 0)

    @classmethod
    def from_args(cls, args):
        return cls(
            layers=args.layers, activation=args.activation, input_dim=args.input_dim,
            mapping_size=args.embed_dim, fourier_scale=args.fourier_scale
        )

class PINNAdapter(BaseAdapter):

    def get_domain_coords(self, data):
        return data['domain_coords_inner']

    def get_domain_inner_coords(self, data):
        return data['domain_coords_inner']

    def get_boundary_coords(self, data):
        return data['bnd_coords']

    def get_left_boundary_coords(self, data):
        return data['left_bnd_coords']

    def get_right_boundary_coords(self, data):
        return data['right_bnd_coords']

    def get_top_boundary_coords(self, data):
        return data['top_bnd_coords']

    def get_bottom_boundary_coords(self, data):
        return data['bottom_bnd_coords']

    def get_initial_coords(self, data):
        return data['ini_coords_inner']

    def get_pde_mask(self, data):
        return slice(None)

    def get_left_boundary_mask(self, data):
        return slice(None)

    def get_right_boundary_mask(self, data):
        return slice(None)

    def get_top_boundary_mask(self, data):
        return slice(None)

    def get_bottom_boundary_mask(self, data):
        return slice(None)

    def get_initial_mask(self, data):
        return slice(None)

    def forward(self, model, coords, data, sharp_const=None, mask=None, mode=None):
        u = model(coords)
        return (u, {})

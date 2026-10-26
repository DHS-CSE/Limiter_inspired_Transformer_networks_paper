from embedding.embedding import FourierWaveletActivation, LayerWiseLAAF, PointwiseMLP
from embedding.slope_limiter import (
    AP_FixedSharpSlopeLimiterEmbeddingAnalysis, FixedSharpSlopeLimiterEmbeddingAnalysis,
    FixedSharpSlopeLimiterEmbeddingAnalysis3D, FixedSharpSlopeLimiterEmbeddingAnalysis3D_NoSlope,
    FixedSharpSlopeLimiterEmbeddingAnalysis_MC, FixedSharpSlopeLimiterEmbeddingAnalysis_NoSlope,
    FixedSharpSlopeLimiterEmbeddingAnalysis_Superbee,
)
from attention.attention import (
    Batched_MultiheadCrossAttention, Batched_MultiheadCrossAttention_Periodic_Single,
    Batched_MultiheadCrossAttention_Single,
)
from embedding.sinusoidal import Sin
import torch.nn as nn
import torch
from .base import BaseAdapter
from registry import register_model

@register_model
class SLFormerFixedSharpAnalysis(nn.Module):
    input_type = 'GhostTransformer'

    def __init__(self, fourier_scale=1.0, embed_dim=128, num_heads=4, depth=4):
        super().__init__()
        self.coord_proj = FixedSharpSlopeLimiterEmbeddingAnalysis(fourier_scale, 2, embed_dim)
        self.val_embedder = PointwiseMLP(input_dim=1, hidden_dim=embed_dim, activation='Sin')
        self.volume_layers = nn.ModuleList([Batched_MultiheadCrossAttention(embed_dim, num_heads) for _ in range(depth)])
        self.initial_layers = nn.ModuleList([Batched_MultiheadCrossAttention(embed_dim, num_heads) for _ in range(depth)])
        self.final_mlp = nn.Sequential(
            nn.Linear(2 * embed_dim, embed_dim), Sin(), nn.Linear(embed_dim, embed_dim), Sin(),
            nn.Linear(embed_dim, 1)
        )
        self._init_weights()

    def _init_weights(self):
        nn.init.uniform_(self.final_mlp[-1].weight, -0.01, 0.01)
        nn.init.zeros_(self.final_mlp[-1].bias)

    def forward(
        self, tar_coords, src_coords, src_values, bnd_coords, bnd_values, ini_coords, ini_values, sharp_const,
        mask, pde_mask, bnd_mask, ini_mask, mode
    ):
        Q_vol, gate_dom_q, m_dom_q, l_dom_q = self.coord_proj(tar_coords, sharp_const, mask)
        Q_ini = Q_vol.clone()
        K_vol, gate_dom_k, m_dom_k, l_dom_k = self.coord_proj(src_coords, sharp_const, pde_mask)
        V_vol = self.val_embedder(src_values)
        K_ini, gate_ini_k, m_ini_k, l_ini_k = self.coord_proj(ini_coords, sharp_const, ini_mask)
        V_ini = self.val_embedder(ini_values)
        for layer in self.volume_layers:
            Q_vol, _, _, H_dom = layer(Q_vol, K_vol, V_vol)
        for layer in self.initial_layers:
            Q_ini, _, _, H_ini = layer(Q_ini, K_ini, V_ini)
        H_total = torch.cat([Q_vol, Q_ini], dim=-1)
        return (
            self.final_mlp(H_total),
            {'H_dom': H_dom, 'H_ini': H_ini, 'gate_dom_q': gate_dom_q, 'm_dom_q': m_dom_q, 'l_dom_q': l_dom_q}
        )

    @classmethod
    def from_args(cls, args):
        return cls(
            fourier_scale=args.fourier_scale, embed_dim=args.embed_dim, num_heads=args.num_heads,
            depth=args.depth
        )

@register_model
class SLSingleFormerFixedSharpAnalysis(nn.Module):
    input_type = 'GhostTransformer'

    def __init__(self, fourier_scale=1.0, embed_dim=128, num_heads=4, depth=4):
        super().__init__()
        self.coord_proj = FixedSharpSlopeLimiterEmbeddingAnalysis(fourier_scale, 2, embed_dim)
        self.val_embedder = PointwiseMLP(input_dim=1, hidden_dim=embed_dim, activation='sin')
        self.decoder_layers = nn.ModuleList([Batched_MultiheadCrossAttention_Periodic_Single(embed_dim, num_heads) for _ in range(depth)])
        self.final_mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim), Sin(), nn.Linear(embed_dim, embed_dim), Sin(),
            nn.Linear(embed_dim, 1)
        )
        self._init_weights()

    def _init_weights(self):
        nn.init.uniform_(self.final_mlp[-1].weight, -0.01, 0.01)
        nn.init.zeros_(self.final_mlp[-1].bias)

    def forward(
        self, tar_coords, src_coords, src_values, bnd_coords, bnd_values, ini_coords, ini_values, sharp_const,
        mask, pde_mask, bnd_mask, ini_mask, mode
    ):
        Nd = tar_coords.shape[1]
        Ni = ini_coords.shape[1]
        Q_vol, gate_dom_q, m_dom_q, l_dom_q = self.coord_proj(tar_coords, sharp_const, mask)
        K_dom, gate_dom_k, m_dom_k, l_dom_k = self.coord_proj(src_coords, sharp_const, pde_mask)
        K_ini, gate_ini_k, m_ini_k, l_ini_k = self.coord_proj(ini_coords, sharp_const, ini_mask)
        K_vol = torch.cat((K_dom, K_ini), dim=1)
        key_values = torch.cat((src_values, ini_values), dim=1)
        V_vol = self.val_embedder(key_values)
        for layer in self.decoder_layers:
            Q_vol, _, _, H_dom_final, H_ini_final = layer(Q_vol, K_vol, V_vol, Nd, Ni)
        H_total = Q_vol
        return (
            self.final_mlp(H_total),
            {
                'H_dom': H_dom_final, 'H_ini': H_ini_final, 'gate_dom_q': gate_dom_q, 'm_dom_q': m_dom_q,
                'l_dom_q': l_dom_q
            }
        )

    @classmethod
    def from_args(cls, args):
        return cls(
            fourier_scale=args.fourier_scale, embed_dim=args.embed_dim, num_heads=args.num_heads,
            depth=args.depth
        )

@register_model
class SLFormerFixedSharpAnalysis_NoSlope(nn.Module):
    input_type = "GhostTransformer"
    def __init__(self, fourier_scale=1.0, embed_dim=128, num_heads=4, depth=4):
        super().__init__()
        
        # 1. Coord Embedder
        self.coord_proj = FixedSharpSlopeLimiterEmbeddingAnalysis_NoSlope(fourier_scale, 2, embed_dim)
        
        # 2. value Embedder
        self.val_embedder = PointwiseMLP(input_dim=1, hidden_dim=embed_dim, activation='Sin')
        
        # Path A: Domain Decoder
        self.volume_layers = nn.ModuleList([Batched_MultiheadCrossAttention(embed_dim, num_heads) for _ in range(depth)])

        self.initial_layers = nn.ModuleList([Batched_MultiheadCrossAttention(embed_dim, num_heads) for _ in range(depth)])
        
        # 5. Final Projection
        self.final_mlp = nn.Sequential(
            nn.Linear(2*embed_dim, embed_dim),
            Sin(),
            nn.Linear(embed_dim, embed_dim),
            Sin(),
            nn.Linear(embed_dim, 1),
        )

        self._init_weights()

    def _init_weights(self):
        nn.init.uniform_(self.final_mlp[-1].weight, -0.01, 0.01)
        nn.init.zeros_(self.final_mlp[-1].bias)

    def forward(self, tar_coords, src_coords, src_values, bnd_coords, bnd_values, ini_coords, ini_values, sharp_const, mask, pde_mask, bnd_mask, ini_mask, mode):
        # Q (coord Embedding)
        Q_vol, m_dom_q = self.coord_proj(tar_coords, sharp_const, mask)
        Q_ini = Q_vol.clone() # Initial Q

        # K (coord Embedding), V (value Embedding)
        K_vol, m_dom_k = self.coord_proj(src_coords, sharp_const, pde_mask)
        V_vol = self.val_embedder(src_values)

        K_ini, m_ini_k = self.coord_proj(ini_coords, sharp_const, ini_mask)
        V_ini = self.val_embedder(ini_values)
        
        # [Path A] Daomain Layers
        for layer in self.volume_layers:
            Q_vol, _, _, H_dom = layer(Q_vol, K_vol, V_vol)            

        # [Path C] Initial Layers
        for layer in self.initial_layers:
            Q_ini, _, _, H_ini = layer(Q_ini, K_ini, V_ini)
        
        # Prediction
        H_total = torch.cat([Q_vol, Q_ini], dim=-1)
        
        return self.final_mlp(H_total), {"H_dom": H_dom, "H_ini": H_ini, "m_dom_q": m_dom_q}
    
    @classmethod
    def from_args(cls, args):

        return cls(
            fourier_scale=args.fourier_scale,
            embed_dim=args.embed_dim,
            num_heads=args.num_heads,
            depth=args.depth,
        )

@register_model
class SLSingleFormerFixedSharpAnalysis_NoSlope(nn.Module):
    input_type = "GhostTransformer"
    def __init__(self, fourier_scale=1.0, embed_dim=128, num_heads=4, depth=4):
        super().__init__()
        
        # 1. Coord Embedder
        self.coord_proj = FixedSharpSlopeLimiterEmbeddingAnalysis_NoSlope(fourier_scale, 2, embed_dim)
        
        # 2. value Embedder
        self.val_embedder = PointwiseMLP(input_dim=1, hidden_dim=embed_dim, activation='sin')
        
        # Path A: Domain Decoder
        self.decoder_layers = nn.ModuleList([Batched_MultiheadCrossAttention_Periodic_Single(embed_dim, num_heads) for _ in range(depth)])
                
        # 5. Final Projection
        self.final_mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            Sin(),
            nn.Linear(embed_dim, embed_dim),
            Sin(),
            nn.Linear(embed_dim, 1),
        )

        self._init_weights()

    def _init_weights(self):
        nn.init.uniform_(self.final_mlp[-1].weight, -0.01, 0.01)
        nn.init.zeros_(self.final_mlp[-1].bias)

    def forward(self, tar_coords, src_coords, src_values, bnd_coords, bnd_values, ini_coords, ini_values, sharp_const, mask, pde_mask, bnd_mask, ini_mask, mode):
        Nd = tar_coords.shape[1]
        Ni = ini_coords.shape[1]

        # Q (coord Embedding)
        Q_vol, m_dom_q = self.coord_proj(tar_coords, sharp_const, mask)

        # K (coord Embedding), V (value Embedding)
        K_dom, _ = self.coord_proj(src_coords, sharp_const, pde_mask)
        K_ini, _ = self.coord_proj(ini_coords, sharp_const, ini_mask)
        K_vol = torch.cat((K_dom, K_ini), dim=1)
        key_values = torch.cat((src_values, ini_values), dim=1)
        V_vol = self.val_embedder(key_values)

        # [Path A] Damain Layers
        for layer in self.decoder_layers:
            Q_vol, _, _, H_dom_final, H_ini_final = layer(Q_vol, K_vol, V_vol, Nd, Ni)
        
        H_total = Q_vol
        
        return self.final_mlp(H_total), {"H_dom": H_dom_final, "H_ini": H_ini_final, "m_dom_q": m_dom_q}
    
    @classmethod
    def from_args(cls, args):

        return cls(
            fourier_scale=args.fourier_scale,
            embed_dim=args.embed_dim,
            num_heads=args.num_heads,
            depth=args.depth,
        )

@register_model
class BurgersSLFormerFixedSharpAnalysis(nn.Module):
    input_type = 'GhostTransformer'

    def __init__(self, fourier_scale=1.0, embed_dim=128, num_heads=4, depth=4):
        super().__init__()
        self.coord_proj = FixedSharpSlopeLimiterEmbeddingAnalysis(fourier_scale, 2, embed_dim)
        self.val_embedder = PointwiseMLP(input_dim=1, hidden_dim=embed_dim, activation='SiLU')
        self.volume_layers = nn.ModuleList([Batched_MultiheadCrossAttention(embed_dim, num_heads) for _ in range(depth)])
        self.boundary_layers = nn.ModuleList([Batched_MultiheadCrossAttention(embed_dim, num_heads) for _ in range(depth)])
        self.initial_layers = nn.ModuleList([Batched_MultiheadCrossAttention(embed_dim, num_heads) for _ in range(depth)])
        self.final_mlp = nn.Sequential(
            nn.Linear(3 * embed_dim, embed_dim), nn.SiLU(), nn.Linear(embed_dim, embed_dim), nn.SiLU(),
            nn.Linear(embed_dim, 1)
        )
        self._init_weights()

    def _init_weights(self):
        nn.init.uniform_(self.final_mlp[-1].weight, -0.01, 0.01)
        nn.init.zeros_(self.final_mlp[-1].bias)

    def forward(
        self, tar_coords, src_coords, src_values, bnd_coords, bnd_values, ini_coords, ini_values, sharp_const,
        coord_mask, pde_mask, bnd_mask, ini_mask, mode
    ):
        Q_vol, g_dom_q, m_dom_q, l_dom_q = self.coord_proj(tar_coords, sharp_const, coord_mask)
        Q_bnd = Q_vol.clone()
        Q_ini = Q_vol.clone()
        K_vol, g_dom_k, m_dom_k, l_dom_k = self.coord_proj(src_coords, sharp_const, pde_mask)
        V_vol = self.val_embedder(src_values)
        K_bnd, g_bnd_k, m_bnd_k, l_bnd_k = self.coord_proj(bnd_coords, sharp_const, bnd_mask)
        V_bnd = self.val_embedder(bnd_values)
        K_ini, g_ini_k, m_ini_k, l_ini_k = self.coord_proj(ini_coords, sharp_const, ini_mask)
        V_ini = self.val_embedder(ini_values)
        for layer in self.volume_layers:
            Q_vol, _, attn_dom, H_dom = layer(Q_vol, K_vol, V_vol)
        for layer in self.boundary_layers:
            Q_bnd, _, _, H_bnd = layer(Q_bnd, K_bnd, V_bnd)
        for layer in self.initial_layers:
            Q_ini, _, _, H_ini = layer(Q_ini, K_ini, V_ini)
        H_total = torch.cat([Q_vol, Q_bnd, Q_ini], dim=-1)
        return (
            self.final_mlp(H_total),
            {
                'H_dom': H_dom, 'H_bnd': H_bnd, 'H_ini': H_ini, 'Attn_Score': attn_dom, 'gate_dom_q': g_dom_q,
                'gate_dom_k': g_dom_k, 'gate_bnd_k': g_bnd_k, 'gate_ini_k': g_ini_k, 'm_dom_q': m_dom_q,
                'm_dom_k': m_dom_k, 'm_bnd_k': m_bnd_k, 'm_ini_k': m_ini_k, 'l_dom_q': l_dom_q,
                'l_dom_k': l_dom_k, 'l_bnd_k': l_bnd_k, 'l_ini_k': l_ini_k
            }
        )

    @classmethod
    def from_args(cls, args):
        return cls(
            fourier_scale=args.fourier_scale, embed_dim=args.embed_dim, num_heads=args.num_heads,
            depth=args.depth
        )

@register_model
class BurgersSLFormerFixedSharpAnalysis_Superbee(nn.Module):
    input_type = 'GhostTransformer'

    def __init__(self, fourier_scale=1.0, embed_dim=128, num_heads=4, depth=4):
        super().__init__()
        self.coord_proj = FixedSharpSlopeLimiterEmbeddingAnalysis_Superbee(fourier_scale, 2, embed_dim)
        self.val_embedder = PointwiseMLP(input_dim=1, hidden_dim=embed_dim, activation='SiLU')
        self.volume_layers = nn.ModuleList([Batched_MultiheadCrossAttention(embed_dim, num_heads) for _ in range(depth)])
        self.boundary_layers = nn.ModuleList([Batched_MultiheadCrossAttention(embed_dim, num_heads) for _ in range(depth)])
        self.initial_layers = nn.ModuleList([Batched_MultiheadCrossAttention(embed_dim, num_heads) for _ in range(depth)])
        self.final_mlp = nn.Sequential(
            nn.Linear(3 * embed_dim, embed_dim), nn.SiLU(), nn.Linear(embed_dim, embed_dim), nn.SiLU(),
            nn.Linear(embed_dim, 1)
        )
        self._init_weights()

    def _init_weights(self):
        nn.init.uniform_(self.final_mlp[-1].weight, -0.01, 0.01)
        nn.init.zeros_(self.final_mlp[-1].bias)

    def forward(
        self, tar_coords, src_coords, src_values, bnd_coords, bnd_values, ini_coords, ini_values, sharp_const,
        coord_mask, pde_mask, bnd_mask, ini_mask, mode
    ):
        Q_vol, g_dom_q, m_dom_q, l_dom_q = self.coord_proj(tar_coords, sharp_const, coord_mask)
        Q_bnd = Q_vol.clone()
        Q_ini = Q_vol.clone()
        K_vol, g_dom_k, m_dom_k, l_dom_k = self.coord_proj(src_coords, sharp_const, pde_mask)
        V_vol = self.val_embedder(src_values)
        K_bnd, g_bnd_k, m_bnd_k, l_bnd_k = self.coord_proj(bnd_coords, sharp_const, bnd_mask)
        V_bnd = self.val_embedder(bnd_values)
        K_ini, g_ini_k, m_ini_k, l_ini_k = self.coord_proj(ini_coords, sharp_const, ini_mask)
        V_ini = self.val_embedder(ini_values)
        for layer in self.volume_layers:
            Q_vol, _, attn_dom, H_dom = layer(Q_vol, K_vol, V_vol)
        for layer in self.boundary_layers:
            Q_bnd, _, _, H_bnd = layer(Q_bnd, K_bnd, V_bnd)
        for layer in self.initial_layers:
            Q_ini, _, _, H_ini = layer(Q_ini, K_ini, V_ini)
        H_total = torch.cat([Q_vol, Q_bnd, Q_ini], dim=-1)
        return (
            self.final_mlp(H_total),
            {
                'H_dom': H_dom, 'H_bnd': H_bnd, 'H_ini': H_ini, 'Attn_Score': attn_dom, 'gate_dom_q': g_dom_q,
                'gate_dom_k': g_dom_k, 'gate_bnd_k': g_bnd_k, 'gate_ini_k': g_ini_k, 'm_dom_q': m_dom_q,
                'm_dom_k': m_dom_k, 'm_bnd_k': m_bnd_k, 'm_ini_k': m_ini_k, 'l_dom_q': l_dom_q,
                'l_dom_k': l_dom_k, 'l_bnd_k': l_bnd_k, 'l_ini_k': l_ini_k
            }
        )

    @classmethod
    def from_args(cls, args):
        return cls(
            fourier_scale=args.fourier_scale, embed_dim=args.embed_dim, num_heads=args.num_heads,
            depth=args.depth
        )

@register_model
class BurgersSLFormerFixedSharpAnalysis_MC(nn.Module):
    input_type = 'GhostTransformer'

    def __init__(self, fourier_scale=1.0, embed_dim=128, num_heads=4, depth=4):
        super().__init__()
        self.coord_proj = FixedSharpSlopeLimiterEmbeddingAnalysis_MC(fourier_scale, 2, embed_dim)
        self.val_embedder = PointwiseMLP(input_dim=1, hidden_dim=embed_dim, activation='SiLU')
        self.volume_layers = nn.ModuleList([Batched_MultiheadCrossAttention(embed_dim, num_heads) for _ in range(depth)])
        self.boundary_layers = nn.ModuleList([Batched_MultiheadCrossAttention(embed_dim, num_heads) for _ in range(depth)])
        self.initial_layers = nn.ModuleList([Batched_MultiheadCrossAttention(embed_dim, num_heads) for _ in range(depth)])
        self.final_mlp = nn.Sequential(
            nn.Linear(3 * embed_dim, embed_dim), nn.SiLU(), nn.Linear(embed_dim, embed_dim), nn.SiLU(),
            nn.Linear(embed_dim, 1)
        )
        self._init_weights()

    def _init_weights(self):
        nn.init.uniform_(self.final_mlp[-1].weight, -0.01, 0.01)
        nn.init.zeros_(self.final_mlp[-1].bias)

    def forward(
        self, tar_coords, src_coords, src_values, bnd_coords, bnd_values, ini_coords, ini_values, sharp_const,
        coord_mask, pde_mask, bnd_mask, ini_mask, mode
    ):
        Q_vol, g_dom_q, m_dom_q, l_dom_q = self.coord_proj(tar_coords, sharp_const, coord_mask)
        Q_bnd = Q_vol.clone()
        Q_ini = Q_vol.clone()
        K_vol, g_dom_k, m_dom_k, l_dom_k = self.coord_proj(src_coords, sharp_const, pde_mask)
        V_vol = self.val_embedder(src_values)
        K_bnd, g_bnd_k, m_bnd_k, l_bnd_k = self.coord_proj(bnd_coords, sharp_const, bnd_mask)
        V_bnd = self.val_embedder(bnd_values)
        K_ini, g_ini_k, m_ini_k, l_ini_k = self.coord_proj(ini_coords, sharp_const, ini_mask)
        V_ini = self.val_embedder(ini_values)
        for layer in self.volume_layers:
            Q_vol, _, attn_dom, H_dom = layer(Q_vol, K_vol, V_vol)
        for layer in self.boundary_layers:
            Q_bnd, _, _, H_bnd = layer(Q_bnd, K_bnd, V_bnd)
        for layer in self.initial_layers:
            Q_ini, _, _, H_ini = layer(Q_ini, K_ini, V_ini)
        H_total = torch.cat([Q_vol, Q_bnd, Q_ini], dim=-1)
        return (
            self.final_mlp(H_total),
            {
                'H_dom': H_dom, 'H_bnd': H_bnd, 'H_ini': H_ini, 'Attn_Score': attn_dom, 'gate_dom_q': g_dom_q,
                'gate_dom_k': g_dom_k, 'gate_bnd_k': g_bnd_k, 'gate_ini_k': g_ini_k, 'm_dom_q': m_dom_q,
                'm_dom_k': m_dom_k, 'm_bnd_k': m_bnd_k, 'm_ini_k': m_ini_k, 'l_dom_q': l_dom_q,
                'l_dom_k': l_dom_k, 'l_bnd_k': l_bnd_k, 'l_ini_k': l_ini_k
            }
        )

    @classmethod
    def from_args(cls, args):
        return cls(
            fourier_scale=args.fourier_scale, embed_dim=args.embed_dim, num_heads=args.num_heads,
            depth=args.depth
        )

@register_model
class BurgersSLFormerFixedSharpAnalysis_Random(nn.Module):
    input_type = 'Transformer'

    def __init__(self, fourier_scale=1.0, embed_dim=128, num_heads=4, depth=4, virtual_dx=1 / 64):
        super().__init__()
        self.coord_proj = AP_FixedSharpSlopeLimiterEmbeddingAnalysis(fourier_scale, 2, embed_dim, virtual_dx)
        self.val_embedder = PointwiseMLP(input_dim=1, hidden_dim=embed_dim, activation='SiLU')
        self.volume_layers = nn.ModuleList([Batched_MultiheadCrossAttention(embed_dim, num_heads) for _ in range(depth)])
        self.boundary_layers = nn.ModuleList([Batched_MultiheadCrossAttention(embed_dim, num_heads) for _ in range(depth)])
        self.initial_layers = nn.ModuleList([Batched_MultiheadCrossAttention(embed_dim, num_heads) for _ in range(depth)])
        self.final_mlp = nn.Sequential(
            nn.Linear(3 * embed_dim, embed_dim), nn.SiLU(), nn.Linear(embed_dim, embed_dim), nn.SiLU(),
            nn.Linear(embed_dim, 1)
        )
        self._init_weights()

    def _init_weights(self):
        nn.init.uniform_(self.final_mlp[-1].weight, -0.01, 0.01)
        nn.init.zeros_(self.final_mlp[-1].bias)

    def forward(
        self, tar_coords, src_coords, src_values, bnd_coords, bnd_values, ini_coords, ini_values, sharp_const
    ):
        Q_vol, g_dom_q, m_dom_q, l_dom_q = self.coord_proj(tar_coords, sharp_const)
        Q_bnd = Q_vol.clone()
        Q_ini = Q_vol.clone()
        K_vol, g_dom_k, m_dom_k, l_dom_k = self.coord_proj(src_coords, sharp_const)
        V_vol = self.val_embedder(src_values)
        K_bnd, g_bnd_k, m_bnd_k, l_bnd_k = self.coord_proj(bnd_coords, sharp_const)
        V_bnd = self.val_embedder(bnd_values)
        K_ini, g_ini_k, m_ini_k, l_ini_k = self.coord_proj(ini_coords, sharp_const)
        V_ini = self.val_embedder(ini_values)
        for layer in self.volume_layers:
            Q_vol, _, attn_dom, H_dom = layer(Q_vol, K_vol, V_vol)
        for layer in self.boundary_layers:
            Q_bnd, _, _, H_bnd = layer(Q_bnd, K_bnd, V_bnd)
        for layer in self.initial_layers:
            Q_ini, _, _, H_ini = layer(Q_ini, K_ini, V_ini)
        H_total = torch.cat([Q_vol, Q_bnd, Q_ini], dim=-1)
        return (
            self.final_mlp(H_total),
            {
                'H_dom': H_dom, 'H_bnd': H_bnd, 'H_ini': H_ini, 'Attn_Score': attn_dom, 'gate_dom_q': g_dom_q,
                'm_dom_q': m_dom_q, 'l_dom_q': l_dom_q
            }
        )

    @classmethod
    def from_args(cls, args):
        return cls(
            fourier_scale=args.fourier_scale, embed_dim=args.embed_dim, num_heads=args.num_heads,
            depth=args.depth, virtual_dx=args.h_x
        )

@register_model
class BurgersSLFormerFixedSharpAnalysis_Single(nn.Module):
    input_type = 'GhostTransformer'

    def __init__(self, fourier_scale=1.0, embed_dim=128, num_heads=4, depth=4):
        super().__init__()
        self.coord_proj = FixedSharpSlopeLimiterEmbeddingAnalysis(fourier_scale, 2, embed_dim)
        self.val_embedder = PointwiseMLP(input_dim=1, hidden_dim=embed_dim, activation='SiLU')
        self.decoder_layers = nn.ModuleList([Batched_MultiheadCrossAttention_Single(embed_dim, num_heads) for _ in range(depth)])
        self.final_mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim), nn.SiLU(), nn.Linear(embed_dim, embed_dim), nn.SiLU(),
            nn.Linear(embed_dim, 1)
        )
        self._init_weights()

    def _init_weights(self):
        nn.init.uniform_(self.final_mlp[-1].weight, -0.01, 0.01)
        nn.init.zeros_(self.final_mlp[-1].bias)

    def forward(
        self, tar_coords, src_coords, src_values, bnd_coords, bnd_values, ini_coords, ini_values, sharp_const,
        coord_mask, pde_mask, bnd_mask, ini_mask, mode
    ):
        Q_vol, _, _, _ = self.coord_proj(tar_coords, sharp_const, coord_mask)
        K_dom, _, _, _ = self.coord_proj(src_coords, sharp_const, pde_mask)
        K_bnd, _, _, _ = self.coord_proj(bnd_coords, sharp_const, bnd_mask)
        K_ini, _, _, _ = self.coord_proj(ini_coords, sharp_const, ini_mask)
        K_vol = torch.cat((K_dom, K_bnd, K_ini), dim=1)
        key_values = torch.cat([src_values, bnd_values, ini_values], dim=1)
        V_vol = self.val_embedder(key_values)
        Nd = src_coords.shape[1]
        Nb = bnd_coords.shape[1]
        Ni = ini_coords.shape[1]
        for layer in self.decoder_layers:
            Q_vol, _, _, H_dom, H_bnd, H_ini = layer(Q_vol, K_vol, V_vol, Nd, Nb, Ni)
        H_total = Q_vol
        return (self.final_mlp(H_total), {'H_dom': H_dom, 'H_bnd': H_bnd, 'H_ini': H_ini})

    @classmethod
    def from_args(cls, args):
        return cls(
            fourier_scale=args.fourier_scale, embed_dim=args.embed_dim, num_heads=args.num_heads,
            depth=args.depth
        )

@register_model
class BurgersSLFormerFixedSharpAnalysis_NoSlope(nn.Module):
    input_type = 'GhostTransformer'

    def __init__(self, fourier_scale=1.0, embed_dim=128, num_heads=4, depth=4):
        super().__init__()
        self.coord_proj = FixedSharpSlopeLimiterEmbeddingAnalysis_NoSlope(fourier_scale, 2, embed_dim)
        self.val_embedder = PointwiseMLP(input_dim=1, hidden_dim=embed_dim, activation='SiLU')
        self.volume_layers = nn.ModuleList([Batched_MultiheadCrossAttention(embed_dim, num_heads) for _ in range(depth)])
        self.boundary_layers = nn.ModuleList([Batched_MultiheadCrossAttention(embed_dim, num_heads) for _ in range(depth)])
        self.initial_layers = nn.ModuleList([Batched_MultiheadCrossAttention(embed_dim, num_heads) for _ in range(depth)])
        self.final_mlp = nn.Sequential(
            nn.Linear(3 * embed_dim, embed_dim), nn.SiLU(), nn.Linear(embed_dim, embed_dim), nn.SiLU(),
            nn.Linear(embed_dim, 1)
        )
        self._init_weights()

    def _init_weights(self):
        nn.init.uniform_(self.final_mlp[-1].weight, -0.01, 0.01)
        nn.init.zeros_(self.final_mlp[-1].bias)

    def forward(
        self, tar_coords, src_coords, src_values, bnd_coords, bnd_values, ini_coords, ini_values, sharp_const,
        coord_mask, pde_mask, bnd_mask, ini_mask, mode
    ):
        Q_vol, m_dom_q = self.coord_proj(tar_coords, sharp_const, coord_mask)
        Q_bnd = Q_vol.clone()
        Q_ini = Q_vol.clone()
        K_vol, _ = self.coord_proj(src_coords, sharp_const, pde_mask)
        V_vol = self.val_embedder(src_values)
        K_bnd, _ = self.coord_proj(bnd_coords, sharp_const, bnd_mask)
        V_bnd = self.val_embedder(bnd_values)
        K_ini, _ = self.coord_proj(ini_coords, sharp_const, ini_mask)
        V_ini = self.val_embedder(ini_values)
        for layer in self.volume_layers:
            Q_vol, _, _, H_dom = layer(Q_vol, K_vol, V_vol)
        for layer in self.boundary_layers:
            Q_bnd, _, _, H_bnd = layer(Q_bnd, K_bnd, V_bnd)
        for layer in self.initial_layers:
            Q_ini, _, _, H_ini = layer(Q_ini, K_ini, V_ini)
        H_total = torch.cat([Q_vol, Q_bnd, Q_ini], dim=-1)
        return (self.final_mlp(H_total), {'H_dom': H_dom, 'H_bnd': H_bnd, 'H_ini': H_ini, 'm_dom_q': m_dom_q})

    @classmethod
    def from_args(cls, args):
        return cls(
            fourier_scale=args.fourier_scale, embed_dim=args.embed_dim, num_heads=args.num_heads,
            depth=args.depth
        )

@register_model
class BurgersSLFormer_Single_NoSlope(nn.Module):
    input_type = 'GhostTransformer'

    def __init__(self, fourier_scale=1.0, embed_dim=128, num_heads=4, depth=4):
        super().__init__()
        self.coord_proj = FixedSharpSlopeLimiterEmbeddingAnalysis_NoSlope(fourier_scale, 2, embed_dim)
        self.val_embedder = PointwiseMLP(input_dim=1, hidden_dim=embed_dim, activation='SiLU')
        self.decoder_layers = nn.ModuleList([Batched_MultiheadCrossAttention_Single(embed_dim, num_heads) for _ in range(depth)])
        self.final_mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim), nn.SiLU(), nn.Linear(embed_dim, embed_dim), nn.SiLU(),
            nn.Linear(embed_dim, 1)
        )
        self._init_weights()

    def _init_weights(self):
        nn.init.uniform_(self.final_mlp[-1].weight, -0.01, 0.01)
        nn.init.zeros_(self.final_mlp[-1].bias)

    def forward(
        self, tar_coords, src_coords, src_values, bnd_coords, bnd_values, ini_coords, ini_values, sharp_const,
        coord_mask, pde_mask, bnd_mask, ini_mask, mode
    ):
        Q_vol, m_dom_q = self.coord_proj(tar_coords, sharp_const, coord_mask)
        K_dom, _ = self.coord_proj(src_coords, sharp_const, pde_mask)
        K_bnd, _ = self.coord_proj(bnd_coords, sharp_const, bnd_mask)
        K_ini, _ = self.coord_proj(ini_coords, sharp_const, ini_mask)
        K_vol = torch.cat((K_dom, K_bnd, K_ini), dim=1)
        key_values = torch.cat([src_values, bnd_values, ini_values], dim=1)
        V_vol = self.val_embedder(key_values)
        Nd = src_coords.shape[1]
        Nb = bnd_coords.shape[1]
        Ni = ini_coords.shape[1]
        for layer in self.decoder_layers:
            Q_vol, _, _, H_dom, H_bnd, H_ini = layer(Q_vol, K_vol, V_vol, Nd, Nb, Ni)
        H_total = Q_vol
        return (self.final_mlp(H_total), {'H_dom': H_dom, 'H_bnd': H_bnd, 'H_ini': H_ini, 'm_dom_q': m_dom_q})

    @classmethod
    def from_args(cls, args):
        return cls(
            fourier_scale=args.fourier_scale, embed_dim=args.embed_dim, num_heads=args.num_heads,
            depth=args.depth
        )

@register_model
class Burgers2DSLFormerFixedSharpAnalysis(nn.Module):
    input_type = 'GhostTransformer'

    def __init__(self, fourier_scale=1.0, embed_dim=128, num_heads=4, depth=4, activation='SiLU'):
        super().__init__()
        self.coord_proj = FixedSharpSlopeLimiterEmbeddingAnalysis3D(fourier_scale, 3, embed_dim)
        self.val_embedder = PointwiseMLP(input_dim=1, hidden_dim=embed_dim, activation=activation)
        self.volume_layers = nn.ModuleList([Batched_MultiheadCrossAttention(embed_dim, num_heads) for _ in range(depth)])
        self.boundary_layers = nn.ModuleList([Batched_MultiheadCrossAttention(embed_dim, num_heads) for _ in range(depth)])
        self.initial_layers = nn.ModuleList([Batched_MultiheadCrossAttention(embed_dim, num_heads) for _ in range(depth)])
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
                return FourierWaveletActivation(embed_dim)
            elif activation_name == 'laaf':
                return LayerWiseLAAF(n=10.0, a_init=0.1)
        self.final_mlp = nn.Sequential(
            nn.Linear(3 * embed_dim, embed_dim), make_activation(), nn.Linear(embed_dim, embed_dim),
            make_activation(), nn.Linear(embed_dim, 1)
        )
        self._init_weights()

    def _init_weights(self):
        nn.init.uniform_(self.final_mlp[-1].weight, -0.01, 0.01)
        nn.init.zeros_(self.final_mlp[-1].bias)

    def forward(
        self, tar_coords, src_coords, src_values, bnd_coords, bnd_values, ini_coords, ini_values, sharp_const,
        coord_mask, pde_mask, bnd_mask, ini_mask, mode='domain'
    ):
        Q_vol, gate_dom_q, m_dom_q, l_dom_q = self.coord_proj(tar_coords, sharp_const, mode, coord_mask)
        Q_bnd = Q_vol.clone()
        Q_ini = Q_vol.clone()
        K_vol, _, _, _ = self.coord_proj(src_coords, sharp_const, 'domain', pde_mask)
        V_vol = self.val_embedder(src_values)
        K_bnd, _, _, _ = self.coord_proj(bnd_coords, sharp_const, 'boundary', bnd_mask)
        V_bnd = self.val_embedder(bnd_values)
        K_ini, _, _, _ = self.coord_proj(ini_coords, sharp_const, 'initial', ini_mask)
        V_ini = self.val_embedder(ini_values)
        for layer in self.volume_layers:
            Q_vol, _, _, H_dom = layer(Q_vol, K_vol, V_vol)
        for layer in self.boundary_layers:
            Q_bnd, _, _, H_bnd = layer(Q_bnd, K_bnd, V_bnd)
        for layer in self.initial_layers:
            Q_ini, _, _, H_ini = layer(Q_ini, K_ini, V_ini)
        H_total = torch.cat([Q_vol, Q_bnd, Q_ini], dim=-1)
        return (
            self.final_mlp(H_total),
            {
                'H_dom': H_dom, 'H_bnd': H_bnd, 'H_ini': H_ini, 'gate_dom_q': gate_dom_q, 'm_dom_q': m_dom_q,
                'l_dom_q': l_dom_q
            }
        )

    @classmethod
    def from_args(cls, args):
        return cls(
            fourier_scale=args.fourier_scale, embed_dim=args.embed_dim, num_heads=args.num_heads,
            depth=args.depth, activation=args.activation
        )

@register_model
class Burgers2DSLSingleFormerAnalysis(nn.Module):
    input_type = 'GhostTransformer'

    def __init__(self, fourier_scale=1.0, embed_dim=128, num_heads=4, depth=4, activation='SiLU'):
        super().__init__()
        self.coord_proj = FixedSharpSlopeLimiterEmbeddingAnalysis3D(fourier_scale, 3, embed_dim)
        self.val_embedder = PointwiseMLP(input_dim=1, hidden_dim=embed_dim, activation=activation)
        self.decoder_layers = nn.ModuleList([Batched_MultiheadCrossAttention_Single(embed_dim, num_heads) for _ in range(depth)])
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
                return FourierWaveletActivation(embed_dim)
            elif activation_name == 'laaf':
                return LayerWiseLAAF(n=10.0, a_init=0.1)
        self.final_mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim), make_activation(), nn.Linear(embed_dim, embed_dim),
            make_activation(), nn.Linear(embed_dim, 1)
        )
        self._init_weights()

    def _init_weights(self):
        nn.init.uniform_(self.final_mlp[-1].weight, -0.01, 0.01)
        nn.init.zeros_(self.final_mlp[-1].bias)

    def forward(
        self, tar_coords, src_coords, src_values, bnd_coords, bnd_values, ini_coords, ini_values, sharp_const,
        mask, pde_mask, bnd_mask, ini_mask, mode='domain'
    ):
        Q_vol, gate_dom_q, m_dom_q, l_dom_q = self.coord_proj(tar_coords, sharp_const, mode, mask)
        K_dom, gate_dom_k, m_dom_k, l_dom_k = self.coord_proj(src_coords, sharp_const, 'domain', pde_mask)
        K_bnd, gate_bnd_k, m_bnd_k, l_bnd_k = self.coord_proj(bnd_coords, sharp_const, 'boundary', bnd_mask)
        K_ini, gate_ini_k, m_ini_k, l_ini_k = self.coord_proj(ini_coords, sharp_const, 'initial', ini_mask)
        K_vol = torch.cat((K_dom, K_bnd, K_ini), dim=1)
        key_values = torch.cat((src_values, bnd_values, ini_values), dim=1)
        V_vol = self.val_embedder(key_values)
        Nd = src_coords.shape[1]
        Nb = bnd_coords.shape[1]
        Ni = ini_coords.shape[1]
        for layer in self.decoder_layers:
            Q_vol, _, _, H_dom_final, H_bnd_final, H_ini_final = layer(Q_vol, K_vol, V_vol, Nd, Nb, Ni)
        H_total = Q_vol
        return (
            self.final_mlp(H_total),
            {
                'H_dom': H_dom_final, 'H_bnd': H_bnd_final, 'H_ini': H_ini_final, 'gate_dom_q': gate_dom_q,
                'gate_dom_k': gate_dom_k, 'gate_bnd_k': gate_bnd_k, 'gate_ini_k': gate_ini_k,
                'm_dom_q': m_dom_q, 'm_dom_k': m_dom_k, 'm_bnd_k': m_bnd_k, 'm_ini_k': m_ini_k,
                'l_dom_q': l_dom_q, 'l_dom_k': l_dom_k, 'l_bnd_k': l_bnd_k, 'l_ini_k': l_ini_k
            }
        )

    @classmethod
    def from_args(cls, args):
        return cls(
            fourier_scale=args.fourier_scale, embed_dim=args.embed_dim, num_heads=args.num_heads,
            depth=args.depth, activation=args.activation
        )

@register_model
class Burgers2DSLFormer_NoSlope(nn.Module):
    input_type = 'GhostTransformer'

    def __init__(self, fourier_scale=1.0, embed_dim=128, num_heads=4, depth=4, activation='SiLU'):
        super().__init__()
        self.coord_proj = FixedSharpSlopeLimiterEmbeddingAnalysis3D_NoSlope(fourier_scale, 3, embed_dim)
        self.val_embedder = PointwiseMLP(input_dim=1, hidden_dim=embed_dim, activation=activation)
        self.volume_layers = nn.ModuleList([Batched_MultiheadCrossAttention(embed_dim, num_heads) for _ in range(depth)])
        self.boundary_layers = nn.ModuleList([Batched_MultiheadCrossAttention(embed_dim, num_heads) for _ in range(depth)])
        self.initial_layers = nn.ModuleList([Batched_MultiheadCrossAttention(embed_dim, num_heads) for _ in range(depth)])
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
                return FourierWaveletActivation(embed_dim)
            elif activation_name == 'laaf':
                return LayerWiseLAAF(n=10.0, a_init=0.1)
        self.final_mlp = nn.Sequential(
            nn.Linear(3 * embed_dim, embed_dim), make_activation(), nn.Linear(embed_dim, embed_dim),
            make_activation(), nn.Linear(embed_dim, 1)
        )
        self._init_weights()

    def _init_weights(self):
        nn.init.uniform_(self.final_mlp[-1].weight, -0.01, 0.01)
        nn.init.zeros_(self.final_mlp[-1].bias)

    def forward(
        self, tar_coords, src_coords, src_values, bnd_coords, bnd_values, ini_coords, ini_values, sharp_const,
        coord_mask, pde_mask, bnd_mask, ini_mask, mode='domain'
    ):
        Q_vol, m_dom_q = self.coord_proj(tar_coords, sharp_const, mode, coord_mask)
        Q_bnd = Q_vol.clone()
        Q_ini = Q_vol.clone()
        K_vol, _ = self.coord_proj(src_coords, sharp_const, 'domain', pde_mask)
        V_vol = self.val_embedder(src_values)
        K_bnd, _ = self.coord_proj(bnd_coords, sharp_const, 'boundary', bnd_mask)
        V_bnd = self.val_embedder(bnd_values)
        K_ini, _ = self.coord_proj(ini_coords, sharp_const, 'initial', ini_mask)
        V_ini = self.val_embedder(ini_values)
        for layer in self.volume_layers:
            Q_vol, _, _, H_dom = layer(Q_vol, K_vol, V_vol)
        for layer in self.boundary_layers:
            Q_bnd, _, _, H_bnd = layer(Q_bnd, K_bnd, V_bnd)
        for layer in self.initial_layers:
            Q_ini, _, _, H_ini = layer(Q_ini, K_ini, V_ini)
        H_total = torch.cat([Q_vol, Q_bnd, Q_ini], dim=-1)
        return (self.final_mlp(H_total), {'H_dom': H_dom, 'H_bnd': H_bnd, 'H_ini': H_ini, 'm_dom_q': m_dom_q})

    @classmethod
    def from_args(cls, args):
        return cls(
            fourier_scale=args.fourier_scale, embed_dim=args.embed_dim, num_heads=args.num_heads,
            depth=args.depth, activation=args.activation
        )

@register_model
class Burgers2DSLSingleFormer_NoSlope(nn.Module):
    input_type = 'GhostTransformer'

    def __init__(self, fourier_scale=1.0, embed_dim=128, num_heads=4, depth=4, activation='SiLU'):
        super().__init__()
        self.coord_proj = FixedSharpSlopeLimiterEmbeddingAnalysis3D_NoSlope(fourier_scale, 3, embed_dim)
        self.val_embedder = PointwiseMLP(input_dim=1, hidden_dim=embed_dim, activation=activation)
        self.decoder_layers = nn.ModuleList([Batched_MultiheadCrossAttention_Single(embed_dim, num_heads) for _ in range(depth)])
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
                return FourierWaveletActivation(embed_dim)
            elif activation_name == 'laaf':
                return LayerWiseLAAF(n=10.0, a_init=0.1)
        self.final_mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim), make_activation(), nn.Linear(embed_dim, embed_dim),
            make_activation(), nn.Linear(embed_dim, 1)
        )
        self._init_weights()

    def _init_weights(self):
        nn.init.uniform_(self.final_mlp[-1].weight, -0.01, 0.01)
        nn.init.zeros_(self.final_mlp[-1].bias)

    def forward(
        self, tar_coords, src_coords, src_values, bnd_coords, bnd_values, ini_coords, ini_values, sharp_const,
        mask, pde_mask, bnd_mask, ini_mask, mode='domain'
    ):
        Q_vol, m_dom_q = self.coord_proj(tar_coords, sharp_const, mode, mask)
        K_dom, _ = self.coord_proj(src_coords, sharp_const, 'domain', pde_mask)
        K_bnd, _ = self.coord_proj(bnd_coords, sharp_const, 'boundary', bnd_mask)
        K_ini, _ = self.coord_proj(ini_coords, sharp_const, 'initial', ini_mask)
        K_vol = torch.cat((K_dom, K_bnd, K_ini), dim=1)
        key_values = torch.cat((src_values, bnd_values, ini_values), dim=1)
        V_vol = self.val_embedder(key_values)
        Nd = src_coords.shape[1]
        Nb = bnd_coords.shape[1]
        Ni = ini_coords.shape[1]
        for layer in self.decoder_layers:
            Q_vol, _, _, H_dom_final, H_bnd_final, H_ini_final = layer(Q_vol, K_vol, V_vol, Nd, Nb, Ni)
        H_total = Q_vol
        return (
            self.final_mlp(H_total),
            {'H_dom': H_dom_final, 'H_bnd': H_bnd_final, 'H_ini': H_ini_final, 'm_dom_q': m_dom_q}
        )

    @classmethod
    def from_args(cls, args):
        return cls(
            fourier_scale=args.fourier_scale, embed_dim=args.embed_dim, num_heads=args.num_heads,
            depth=args.depth, activation=args.activation
        )

class TransformerAdapter(BaseAdapter):

    def get_domain_coords(self, data):
        return data['domain_coords']

    def get_domain_inner_coords(self, data):
        return data['domain_coords_inner']

    def get_boundary_coords(self, data):
        return data['bnd_coords']

    def get_left_boundary_coords(self, data):
        return data['left_bnd_coords']

    def get_right_boundary_coords(self, data):
        return data['right_bnd_coords']

    def get_initial_coords(self, data):
        return data['ini_coords_inner']

    def get_pde_mask(self, data):
        return slice(None)

    def get_boundary_mask(self, data):
        return slice(None)

    def get_initial_mask(self, data):
        return slice(None)

    def get_domain_inner_prediction(self, prediction, data):
        n_inner = data['domain_coords_inner'].shape[1]
        n_domain = data['domain_coords'].shape[1]
        if prediction.shape[1] == n_inner:
            return prediction
        if prediction.shape[1] != n_domain:
            raise ValueError(f'Unexpected prediction size: {prediction.shape[1]}. Expected {n_inner} or {n_domain}.')
        return prediction[:, data['pde_mask'], ...]

    def forward(self, model, coords, data, sharp_const, mask, mode='domain'):
        return model(
            coords, data['domain_coords'], data['domain_f'], data['bnd_coords'], data['bnd_values'],
            data['ini_coords_inner'], data['ini_values'], sharp_const
        )

class GhostTransformerAdapter(BaseAdapter):

    def get_domain_coords(self, data):
        return data['domain_coords']

    def get_domain_inner_coords(self, data):
        return data['domain_coords_inner']

    def get_boundary_coords(self, data):
        return data['bnd_coords_2step']

    def get_left_boundary_coords(self, data):
        return data['left_bnd_coords_2step']

    def get_right_boundary_coords(self, data):
        return data['right_bnd_coords_2step']

    def get_bottom_boundary_coords(self, data):
        return data['bottom_bnd_coords_2step']

    def get_top_boundary_coords(self, data):
        return data['top_bnd_coords_2step']

    def get_initial_coords(self, data):
        return data['ini_coords']

    def get_pde_mask(self, data):
        return data['pde_mask']

    def get_boundary_mask(self, data):
        return data['boundary_inner_mask']

    def get_left_boundary_mask(self, data):
        return data['left_boundary_mask']

    def get_right_boundary_mask(self, data):
        return data['right_boundary_mask']

    def get_top_boundary_mask(self, data):
        return data['top_boundary_mask']

    def get_bottom_boundary_mask(self, data):
        return data['bottom_boundary_mask']

    def get_initial_mask(self, data):
        return data['initial_inner_mask']

    def forward(self, model, coords, data, sharp_const, mask, mode='domain'):
        return model(
            coords, data['domain_coords'], data['domain_f_inner'], data['bnd_coords_2step'],
            data['bnd_values'], data['ini_coords'], data['ini_values'], sharp_const, mask, data['pde_mask'],
            data['boundary_inner_mask'], data['initial_inner_mask'], mode
        )

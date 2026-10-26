import torch
import torch.nn as nn
import math

class Batched_MultiheadCrossAttention(nn.Module):

    def __init__(self, d_model: int, n_heads: int, dropout: float=0.0):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.in_proj = nn.Linear(d_model, 3 * d_model, bias=True)
        self.out_proj = nn.Linear(d_model, d_model, bias=True)
        self.drop = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, q_in: torch.Tensor, k_in: torch.Tensor, v_in: torch.Tensor, alpha: float=1.0, key_mask: torch.Tensor | None=None):
        B, Nq, D = q_in.shape
        _, Nk, _ = k_in.shape
        W = self.in_proj.weight
        b = self.in_proj.bias
        W_q, W_k, W_v = W.chunk(3, dim=0)
        b_q, b_k, b_v = b.chunk(3, dim=0)
        Q = q_in @ W_q.T + b_q
        K = k_in @ W_k.T + b_k
        V = v_in @ W_v.T + b_v
        Q = Q.view(B, Nq, self.n_heads, self.d_head).transpose(1, 2)
        K = K.view(B, Nk, self.n_heads, self.d_head).transpose(1, 2)
        V = V.view(B, Nk, self.n_heads, self.d_head).transpose(1, 2)
        attn_logits = Q @ K.transpose(-2, -1) / math.sqrt(self.d_head)
        if key_mask is not None:
            if key_mask.dim() == 2:
                attn_logits = attn_logits.masked_fill(~key_mask[:, None, None, :], float('-inf'))
            else:
                attn_logits = attn_logits.masked_fill(~key_mask[:, None, :, :], float('-inf'))
        attn = torch.softmax(attn_logits / alpha, dim=-1)
        attn = self.drop(attn)
        head_out = attn @ V
        out = head_out.transpose(1, 2).contiguous()
        out = out.view(B, Nq, D)
        out = self.out_proj(out)
        final_out = self.norm(q_in + out)
        return (final_out, head_out, attn, out)

class Batched_MultiheadCrossAttention_Single(nn.Module):

    def __init__(self, d_model: int, n_heads: int, dropout: float=0.0):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.in_proj = nn.Linear(d_model, 3 * d_model, bias=True)
        self.out_proj = nn.Linear(d_model, d_model, bias=True)
        self.drop = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, q_in, k_in, v_in, Nd: int, Nb: int, Ni: int, alpha: float=1.0, key_mask=None):
        B, Nq, D = q_in.shape
        _, Nk, _ = k_in.shape
        W_q, W_k, W_v = self.in_proj.weight.chunk(3, dim=0)
        b_q, b_k, b_v = self.in_proj.bias.chunk(3, dim=0)
        Q = q_in @ W_q.T + b_q
        K = k_in @ W_k.T + b_k
        V = v_in @ W_v.T + b_v
        Q = Q.view(B, Nq, self.n_heads, self.d_head).transpose(1, 2)
        K = K.view(B, Nk, self.n_heads, self.d_head).transpose(1, 2)
        V = V.view(B, Nk, self.n_heads, self.d_head).transpose(1, 2)
        attn_logits = Q @ K.transpose(-2, -1) / math.sqrt(self.d_head)
        attn = torch.softmax(attn_logits / alpha, dim=-1)
        attn = self.drop(attn)
        V_dom = V[:, :, :Nd, :]
        V_bnd = V[:, :, Nd:Nd + Nb, :]
        V_ini = V[:, :, Nd + Nb:, :]
        A_dom = attn[:, :, :, :Nd]
        A_bnd = attn[:, :, :, Nd:Nd + Nb]
        A_ini = attn[:, :, :, Nd + Nb:]
        H_dom = A_dom @ V_dom
        H_bnd = A_bnd @ V_bnd
        H_ini = A_ini @ V_ini
        head_out = H_dom + H_bnd + H_ini
        H_dom_merge = H_dom.transpose(1, 2).contiguous().view(B, Nq, D)
        H_bnd_merge = H_bnd.transpose(1, 2).contiguous().view(B, Nq, D)
        H_ini_merge = H_ini.transpose(1, 2).contiguous().view(B, Nq, D)
        H_dom_final = self.out_proj(H_dom_merge)
        H_bnd_final = self.out_proj(H_bnd_merge)
        H_ini_final = self.out_proj(H_ini_merge)
        out = head_out.transpose(1, 2).contiguous()
        out = out.view(B, Nq, D)
        out = self.out_proj(out)
        out = self.norm(q_in + out)
        return (out, head_out, attn, H_dom_final, H_bnd_final, H_ini_final)

class Batched_MultiheadCrossAttention_Periodic_Single(nn.Module):

    def __init__(self, d_model: int, n_heads: int, dropout: float=0.0):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.in_proj = nn.Linear(d_model, 3 * d_model, bias=True)
        self.out_proj = nn.Linear(d_model, d_model, bias=True)
        self.drop = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, q_in, k_in, v_in, Nd: int, Ni: int, alpha: float=1.0, key_mask=None):
        B, Nq, D = q_in.shape
        _, Nk, _ = k_in.shape
        W_q, W_k, W_v = self.in_proj.weight.chunk(3, dim=0)
        b_q, b_k, b_v = self.in_proj.bias.chunk(3, dim=0)
        Q = q_in @ W_q.T + b_q
        K = k_in @ W_k.T + b_k
        V = v_in @ W_v.T + b_v
        Q = Q.view(B, Nq, self.n_heads, self.d_head).transpose(1, 2)
        K = K.view(B, Nk, self.n_heads, self.d_head).transpose(1, 2)
        V = V.view(B, Nk, self.n_heads, self.d_head).transpose(1, 2)
        attn_logits = Q @ K.transpose(-2, -1) / math.sqrt(self.d_head)
        attn = torch.softmax(attn_logits / alpha, dim=-1)
        attn = self.drop(attn)
        V_dom = V[:, :, :Nd, :]
        V_ini = V[:, :, Nd:, :]
        A_dom = attn[:, :, :, :Nd]
        A_ini = attn[:, :, :, Nd:]
        H_dom = A_dom @ V_dom
        H_ini = A_ini @ V_ini
        head_out = H_dom + H_ini
        H_dom_merge = H_dom.transpose(1, 2).contiguous().view(B, Nq, D)
        H_ini_merge = H_ini.transpose(1, 2).contiguous().view(B, Nq, D)
        H_dom_final = self.out_proj(H_dom_merge)
        H_ini_final = self.out_proj(H_ini_merge)
        out = head_out.transpose(1, 2).contiguous()
        out = out.view(B, Nq, D)
        out = self.out_proj(out)
        out = self.norm(q_in + out)
        return (out, head_out, attn, H_dom_final, H_ini_final)

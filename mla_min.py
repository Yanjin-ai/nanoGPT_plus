"""
Minimal DeepSeek-style Multi-head Latent Attention, with a KV cache that actually
stores the compressed latent.

Three ideas, in ~150 lines:
  1. compression  -- cache c_kv (kv_lora_rank dims), not K/V (2*n_head*head_dim dims)
  2. absorption   -- fold W_UK into Q and W_UV into the output, so the cached latent
                     is attended to directly and K/V are never materialized at decode
  3. decoupled RoPE -- a separate, head-shared k_rope path, because rotating an
                     absorbed key would make W_UK position-dependent

Run `python mla_min.py` to check absorbed == naive, cached decode == full forward,
and to print the cache saving vs MHA.
"""
import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

# -----------------------------------------------------------------------------

def precompute_freqs_cis(dim, end, theta=10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    return torch.polar(torch.ones(end, dim // 2), torch.outer(torch.arange(end, dtype=torch.float32), freqs))


def apply_rope(x, freqs_cis):
    """x: (B, H, T, d) -- freqs_cis: (T, d//2) complex."""
    xc = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    y = torch.view_as_real(xc * freqs_cis[None, None]).flatten(-2)
    return y.type_as(x)


class RMSNorm(nn.Module):
    def __init__(self, ndim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.eps = eps

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


@dataclass
class MLAConfig:
    n_embd: int = 256
    n_head: int = 4
    qk_nope_head_dim: int = 32   # per-head key dims that come from the latent (absorbable)
    qk_rope_head_dim: int = 16   # per-head query rope dims; the key side is shared across heads
    v_head_dim: int = 32
    kv_lora_rank: int = 32       # r -- the compressed dim that gets cached
    block_size: int = 128


class MLA(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        nh, r = cfg.n_head, cfg.kv_lora_rank
        self.qk_head_dim = cfg.qk_nope_head_dim + cfg.qk_rope_head_dim

        self.wq = nn.Linear(cfg.n_embd, nh * self.qk_head_dim, bias=False)
        # one projection produces both the cached latent and the shared rope key
        self.wkv_a = nn.Linear(cfg.n_embd, r + cfg.qk_rope_head_dim, bias=False)
        self.kv_norm = RMSNorm(r)
        self.wk_b = nn.Linear(r, nh * cfg.qk_nope_head_dim, bias=False)  # W_UK
        self.wv_b = nn.Linear(r, nh * cfg.v_head_dim, bias=False)        # W_UV
        self.wo = nn.Linear(nh * cfg.v_head_dim, cfg.n_embd, bias=False)

        self.register_buffer('freqs_cis', precompute_freqs_cis(cfg.qk_rope_head_dim, cfg.block_size), persistent=False)
        self.scale = 1.0 / math.sqrt(self.qk_head_dim)

    def forward(self, x, start_pos=0, cache=None, absorb=True):
        """
        cache: dict with preallocated 'c_kv' (B, block_size, r) and
               'k_rope' (B, block_size, qk_rope_head_dim), or None for a plain forward.
        """
        cfg = self.cfg
        B, T, _ = x.shape
        nh, r, dn = cfg.n_head, cfg.kv_lora_rank, cfg.qk_nope_head_dim
        end = start_pos + T
        freqs = self.freqs_cis[start_pos:end]

        q = self.wq(x).view(B, T, nh, self.qk_head_dim).transpose(1, 2)     # (B, nh, T, dn+dr)
        q_nope, q_rope = q.split([dn, cfg.qk_rope_head_dim], dim=-1)
        q_rope = apply_rope(q_rope, freqs)

        kv = self.wkv_a(x)
        c_kv, k_rope = kv.split([r, cfg.qk_rope_head_dim], dim=-1)
        c_kv = self.kv_norm(c_kv)                                           # (B, T, r)
        k_rope = apply_rope(k_rope.unsqueeze(1), freqs).squeeze(1)          # (B, T, dr), head-shared

        if cache is not None:
            cache['c_kv'][:B, start_pos:end] = c_kv
            cache['k_rope'][:B, start_pos:end] = k_rope
            c_kv = cache['c_kv'][:B, :end]
            k_rope = cache['k_rope'][:B, :end]

        mask = None
        if T > 1:  # prefill / training: causal within the new chunk, full access to the past
            mask = torch.full((T, end), float('-inf'), device=x.device).triu_(1 + start_pos)

        if absorb:
            # ---- fold W_UK into Q and W_UV into the output: never materialize K/V ----
            wk_b = self.wk_b.weight.view(nh, dn, r)
            wv_b = self.wv_b.weight.view(nh, cfg.v_head_dim, r)
            q_lat = torch.einsum('bhtd,hdr->bhtr', q_nope, wk_b)            # query lives in latent space
            scores = torch.einsum('bhtr,bsr->bhts', q_lat, c_kv) \
                   + torch.einsum('bhtd,bsd->bhts', q_rope, k_rope)
            if mask is not None:
                scores = scores + mask
            att = F.softmax(scores * self.scale, dim=-1)
            y_lat = torch.einsum('bhts,bsr->bhtr', att, c_kv)               # weighted sum in latent space
            y = torch.einsum('bhtr,hdr->bhtd', y_lat, wv_b)                 # up-project once, at the end
        else:
            # ---- naive path: up-project the latent to full K/V (what training does) ----
            S = c_kv.size(1)
            k_nope = self.wk_b(c_kv).view(B, S, nh, dn).transpose(1, 2)
            v = self.wv_b(c_kv).view(B, S, nh, cfg.v_head_dim).transpose(1, 2)
            k = torch.cat([k_nope, k_rope.unsqueeze(1).expand(-1, nh, -1, -1)], dim=-1)
            scores = torch.einsum('bhtd,bhsd->bhts', torch.cat([q_nope, q_rope], dim=-1), k)
            if mask is not None:
                scores = scores + mask
            y = torch.einsum('bhts,bhsd->bhtd', F.softmax(scores * self.scale, dim=-1), v)

        return self.wo(y.transpose(1, 2).reshape(B, T, nh * cfg.v_head_dim))

    def empty_cache(self, B, device):
        cfg = self.cfg
        return {
            'c_kv':   torch.zeros(B, cfg.block_size, cfg.kv_lora_rank, device=device),
            'k_rope': torch.zeros(B, cfg.block_size, cfg.qk_rope_head_dim, device=device),
        }


# -----------------------------------------------------------------------------

if __name__ == '__main__':
    torch.manual_seed(0)
    cfg = MLAConfig()
    m = MLA(cfg).eval()
    x = torch.randn(2, 16, cfg.n_embd)

    with torch.no_grad():
        y_absorb = m(x, absorb=True)
        y_naive = m(x, absorb=False)
        print(f"absorbed vs naive        max|diff| = {(y_absorb - y_naive).abs().max():.2e}")

        # token-by-token decode through the cache must match the full forward
        cache = m.empty_cache(2, x.device)
        steps = torch.cat([m(x[:, t:t+1], start_pos=t, cache=cache) for t in range(x.size(1))], dim=1)
        print(f"cached decode vs full    max|diff| = {(steps - y_absorb).abs().max():.2e}")

    mla_per_tok = cfg.kv_lora_rank + cfg.qk_rope_head_dim
    mha_per_tok = 2 * cfg.n_head * cfg.qk_nope_head_dim
    print(f"\ncache per token: MLA {mla_per_tok} floats vs MHA {mha_per_tok} -> {mha_per_tok/mla_per_tok:.1f}x smaller")
    print(f"DeepSeek-V2 scale: MLA {512+64} vs MHA {2*128*128} -> {2*128*128/(512+64):.1f}x smaller")

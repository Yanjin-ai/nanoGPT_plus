"""
Full definition of a GPT Language Model with Multi-head Latent Attention (MLA).

MLA is introduced in DeepSeek-V2: https://arxiv.org/abs/2405.04434

Differences from the vanilla nanoGPT `model.py`:

1. K/V are computed from a shared low-rank latent `c_kv` instead of full-rank
   projections. Only `c_kv` (kv_lora_rank dims) needs to be cached per token,
   which shrinks the KV cache dramatically.
2. Q can optionally be computed from a low-rank latent `c_q` (q_lora_rank > 0).
3. Learned absolute position embeddings (`wpe`) are replaced by Rotary Position
   Embeddings (RoPE), with support for DeepSeek's *decoupled* RoPE: the rope
   part of the key comes from its own projection `W_KR` applied to x (a single
   head shared by every attention head) and is cached next to `c_kv`, while the
   nope part comes from `c_kv`. See the "Why decoupled RoPE" note below.
4. `v_head_dim` may differ from the q/k head dim (as in DeepSeek-V2/V3, where
   qk = 128 nope + 64 rope = 192 but v = 128).
5. An actual KV cache with a prefill/decode split, plus an "absorbed" decode
   path that never materialises K or V.
6. `GPT.from_pretrained` is removed: OpenAI GPT-2 checkpoints do not map onto
   MLA weights (different projection layouts). Train from scratch or resume a
   checkpoint produced by this file.

The file is otherwise a drop-in replacement for `model.py`: the public API
(`GPT`, `GPTConfig`, `configure_optimizers`, `estimate_mfu`, `generate`) and
the training-time call convention `logits, loss = model(x, y)` are unchanged.

Reference for the MLA weight layout (per DeepSeek-V2):
    c_q  = norm_q( x @ W_DQ^T )            # (B, T, q_lora_rank)
    q    = c_q @ W_UQ^T                    # (B, T, n_head * head_dim)
    c_kv = norm_kv( x @ W_DKV^T )          # (B, T, kv_lora_rank)  <- cached
    k_C  = c_kv @ W_UK^T                   # nope part, per head
    k_R  = RoPE( x @ W_KR^T )              # rope part, ONE head shared  <- cached
    v    = c_kv @ W_UV^T                   # (B, T, n_head * v_head_dim)
    k    = [k_C ; k_R]   q = [q_C ; q_R]

Why decoupled RoPE (i.e. why k_R does not come from c_kv)
---------------------------------------------------------
The attention score can be rewritten so that the cache is never decompressed:

    q_i . k_j = (W_UQ c_q_i)^T (W_UK c_kv_j) = c_q_i^T (W_UQ^T W_UK) c_kv_j

`W_UQ^T W_UK` is a constant matrix, so it can be folded ("absorbed") into the
query side once, and the score becomes a dot product directly in latent space.
RoPE breaks this: with rotations R_m / R_n the middle term becomes
`W_UQ^T R_{n-m} W_UK`, which depends on the relative distance and therefore
cannot be precomputed. DeepSeek's fix is to split each head into a nope part
(absorbable) and a small rope part computed from x by a separate projection.
`_attn_absorbed` below implements the absorbed decode path.

KV cache (per token, per layer, in numbers stored)
--------------------------------------------------
    vanilla MHA : 2 * n_head * head_dim
    MLA         : kv_lora_rank + qk_rope_head_dim
"""

import math
import inspect
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F

# -----------------------------------------------------------------------------
# RoPE helpers
# -----------------------------------------------------------------------------

def precompute_freqs_cis(dim, end, theta=10000.0):
    """Precompute complex rotary embeddings: (end, dim//2) complex64."""
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32)[: (dim // 2)] / dim))
    t = torch.arange(end, dtype=torch.float32)
    freqs = torch.outer(t, freqs)  # (end, dim//2)
    return torch.polar(torch.ones_like(freqs), freqs)  # complex64


def apply_rotary_emb(x, freqs_cis):
    """
    x: (B, nh, T, d) where d is the (even) rotary dim.
    freqs_cis: (T, d//2) complex64 -- indexed by ABSOLUTE position.
    """
    dtype = x.dtype
    xf = x.float().reshape(*x.shape[:-1], -1, 2)
    xc = torch.view_as_complex(xf)                  # (B, nh, T, d//2)
    freqs_cis = freqs_cis.unsqueeze(0).unsqueeze(0)  # (1, 1, T, d//2)
    yc = torch.view_as_real(xc * freqs_cis).flatten(-2)  # (B, nh, T, d)
    return yc.to(dtype)


class RMSNorm(nn.Module):
    """Root mean square layer norm, used on the MLA latent vectors."""

    def __init__(self, ndim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.eps = eps

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


class LayerNorm(nn.Module):
    """ LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False """

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)


class CausalSelfAttention(nn.Module):
    """
    MLA attention with three modes, selected by config:

      kv_lora_rank == 0                     -> vanilla MHA (full-rank K/V)
      kv_lora_rank >  0, qk_rope_head_dim==0 -> MLA, whole head rotated, k_R from c_kv
                                                (legacy layout; absorption impossible)
      kv_lora_rank >  0, qk_rope_head_dim >0 -> MLA + DeepSeek decoupled RoPE
                                                (absorption possible)
    """

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head   # q/k per-head dim
        self.block_size = config.block_size
        self.dropout = config.dropout

        # ---- MLA knobs ----
        self.q_lora_rank = config.q_lora_rank
        self.kv_lora_rank = config.kv_lora_rank
        self.use_rope = config.use_rope
        self.v_head_dim = config.v_head_dim or self.head_dim
        self.absorb = config.absorb

        # rope / nope split of the q-k head dim
        if self.use_rope and config.qk_rope_head_dim > 0:
            # partial rotary: only the last qk_rope_head_dim dims of each head are rotated
            self.split_rope = True
            self.qk_rope_head_dim = config.qk_rope_head_dim
            self.qk_nope_head_dim = self.head_dim - self.qk_rope_head_dim
            assert self.qk_nope_head_dim > 0, \
                "qk_rope_head_dim must be < head_dim, otherwise there is no absorbable nope part"
        else:
            # legacy: rotate the whole head (or no rope at all)
            self.split_rope = False
            self.qk_rope_head_dim = self.head_dim if self.use_rope else 0
            self.qk_nope_head_dim = self.head_dim - self.qk_rope_head_dim
        # `decoupled` is the full DeepSeek layout: k_R gets its OWN projection from x
        # (one head shared by all heads) and is cached beside c_kv. It only exists when
        # there is a latent to decouple from, i.e. in MLA mode.
        self.decoupled_rope = self.split_rope and self.kv_lora_rank > 0
        assert self.qk_rope_head_dim % 2 == 0, "qk_rope_head_dim must be even for RoPE"

        # query projection: low-rank (MLA) or direct
        if self.q_lora_rank > 0:
            self.wq_a = nn.Linear(self.n_embd, self.q_lora_rank, bias=False)
            self.q_norm = RMSNorm(self.q_lora_rank)
            self.wq_b = nn.Linear(self.q_lora_rank, self.n_head * self.head_dim, bias=False)
        else:
            self.wq = nn.Linear(self.n_embd, self.n_head * self.head_dim, bias=config.bias)

        # key/value projection: shared low-rank (MLA) or direct
        if self.kv_lora_rank > 0:
            self.wkv_a = nn.Linear(self.n_embd, self.kv_lora_rank, bias=False)
            self.kv_norm = RMSNorm(self.kv_lora_rank)
            # wk_b produces only the nope part when RoPE is decoupled
            k_out = self.n_head * (self.qk_nope_head_dim if self.decoupled_rope else self.head_dim)
            self.wk_b = nn.Linear(self.kv_lora_rank, k_out, bias=False)
            self.wv_b = nn.Linear(self.kv_lora_rank, self.n_head * self.v_head_dim, bias=False)
            if self.decoupled_rope:
                # ONE head, shared by every attention head -- cached next to c_kv
                self.wkr = nn.Linear(self.n_embd, self.qk_rope_head_dim, bias=False)
        else:
            self.wk = nn.Linear(self.n_embd, self.n_head * self.head_dim, bias=config.bias)
            self.wv = nn.Linear(self.n_embd, self.n_head * self.v_head_dim, bias=config.bias)

        # output projection: n_head*v_head_dim -> n_embd (NOT necessarily square)
        self.c_proj = nn.Linear(self.n_head * self.v_head_dim, self.n_embd, bias=config.bias)
        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        # flash attention make GPU go brrrrr but support is only in PyTorch >= 2.0
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash:
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
            # causal mask to ensure that attention is only applied to the left in the input sequence
            self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                        .view(1, 1, config.block_size, config.block_size))

        # ---- KV cache (lazily allocated, never part of state_dict) ----
        # MLA mode caches `c_kv` (+ the rotated `k_R` when RoPE is decoupled);
        # MHA fallback caches full K and V.
        self.cache_c_kv = None
        self.cache_k_rope = None
        self.cache_k = None
        self.cache_v = None

    # -- KV cache plumbing ----------------------------------------------------

    def reset_cache(self):
        self.cache_c_kv = self.cache_k_rope = self.cache_k = self.cache_v = None

    def _cache_update(self, name, value, start_pos):
        """Write `value` at [start_pos, start_pos+T) and return the whole prefix [0, start_pos+T)."""
        buf = getattr(self, name)
        B, T, D = value.shape
        assert start_pos + T <= self.block_size, \
            f"KV cache overflow: {start_pos}+{T} > block_size={self.block_size}"
        if buf is None or buf.size(0) < B or buf.dtype != value.dtype or buf.device != value.device:
            assert start_pos == 0, "cache was (re)allocated mid-sequence; call reset_cache() first"
            buf = torch.zeros(B, self.block_size, D, dtype=value.dtype, device=value.device)
            setattr(self, name, buf)
        buf[:B, start_pos:start_pos + T] = value
        return buf[:B, :start_pos + T]

    def kv_cache_numbers_per_token(self):
        """How many scalars this layer caches per token. The headline MLA metric."""
        if self.kv_lora_rank > 0:
            return self.kv_lora_rank + (self.qk_rope_head_dim if self.decoupled_rope else 0)
        return self.n_head * self.head_dim + self.n_head * self.v_head_dim

    # -- forward --------------------------------------------------------------

    def forward(self, x, freqs_cis=None, start_pos=0, use_cache=False):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)

        # ---- query ----
        if self.q_lora_rank > 0:
            q = self.wq_b(self.q_norm(self.wq_a(x)))
        else:
            q = self.wq(x)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)  # (B, nh, T, hd)

        # RoPE on the query's rope slice, at its ABSOLUTE positions
        if self.use_rope:
            q_freqs = freqs_cis[start_pos:start_pos + T]
            if self.split_rope:
                q_nope, q_rope = q.split([self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)
                q = torch.cat([q_nope, apply_rotary_emb(q_rope, q_freqs)], dim=-1)
            else:
                q = apply_rotary_emb(q, q_freqs)

        # ---- key / value ----
        if self.kv_lora_rank > 0:
            c_kv = self.kv_norm(self.wkv_a(x))          # (B, T, r) -- the thing we cache
            if self.decoupled_rope:
                # rotate BEFORE caching, so the cache stores position-correct k_R
                k_rope = self.wkr(x).view(B, T, 1, self.qk_rope_head_dim).transpose(1, 2)
                k_rope = apply_rotary_emb(k_rope, freqs_cis[start_pos:start_pos + T])
                k_rope = k_rope.transpose(1, 2).reshape(B, T, self.qk_rope_head_dim)
            if use_cache:
                c_kv = self._cache_update('cache_c_kv', c_kv, start_pos)
                if self.decoupled_rope:
                    k_rope = self._cache_update('cache_k_rope', k_rope, start_pos)
            T_kv = c_kv.size(1)

            # absorbed decode path: never materialise K or V
            if self._can_absorb(T, use_cache):
                return self._attn_absorbed(q, c_kv, k_rope, B, T)

            if self.decoupled_rope:
                k_nope = self.wk_b(c_kv).view(B, T_kv, self.n_head, self.qk_nope_head_dim).transpose(1, 2)
                k_r = k_rope.view(B, T_kv, 1, self.qk_rope_head_dim).transpose(1, 2)
                k = torch.cat([k_nope, k_r.expand(B, self.n_head, T_kv, self.qk_rope_head_dim)], dim=-1)
            else:
                k = self.wk_b(c_kv).view(B, T_kv, self.n_head, self.head_dim).transpose(1, 2)
                if self.use_rope:
                    k = apply_rotary_emb(k, freqs_cis[:T_kv])
            v = self.wv_b(c_kv).view(B, T_kv, self.n_head, self.v_head_dim).transpose(1, 2)
        else:
            # vanilla MHA fallback: cache full K/V
            v = self.wv(x)
            k = self.wk(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
            if self.use_rope:
                k_freqs = freqs_cis[start_pos:start_pos + T]
                if self.split_rope:
                    k_nope, k_r = k.split([self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)
                    k = torch.cat([k_nope, apply_rotary_emb(k_r, k_freqs)], dim=-1)
                else:
                    k = apply_rotary_emb(k, k_freqs)
            k = k.transpose(1, 2).reshape(B, T, self.n_head * self.head_dim)
            if use_cache:
                k = self._cache_update('cache_k', k, start_pos)
                v = self._cache_update('cache_v', v, start_pos)
            T_kv = k.size(1)
            k = k.view(B, T_kv, self.n_head, self.head_dim).transpose(1, 2)
            v = v.view(B, T_kv, self.n_head, self.v_head_dim).transpose(1, 2)

        return self._attn(q, k, v, B, T, start_pos)

    def _causal_args(self, T, T_kv, start_pos, device):
        """
        SDPA's is_causal aligns the triangle to the TOP-LEFT corner, i.e. query row i
        may see key column j<=i counting from 0. That is only what we want when the
        query block starts at position 0. During decode (T==1, T_kv>1) it would mask
        away the entire history, so we must not use it.
        """
        if T_kv == T and start_pos == 0:
            return None, True                      # training / prefill
        if T == 1:
            return None, False                     # single new token sees all of the cache
        q_pos = torch.arange(start_pos, start_pos + T, device=device).unsqueeze(1)
        k_pos = torch.arange(0, T_kv, device=device).unsqueeze(0)
        return (q_pos >= k_pos), False             # chunked prefill

    def _attn(self, q, k, v, B, T, start_pos):
        T_kv = k.size(2)
        if self.flash:
            attn_mask, is_causal = self._causal_args(T, T_kv, start_pos, q.device)
            y = torch.nn.functional.scaled_dot_product_attention(
                q, k, v, attn_mask=attn_mask,
                dropout_p=self.dropout if self.training else 0,
                is_causal=is_causal)
        else:
            # NOTE: scale uses the Q/K head dim, never v_head_dim
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))
            att = att.masked_fill(self.bias[:, :, start_pos:start_pos + T, :T_kv] == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, self.n_head * self.v_head_dim)
        return self.resid_dropout(self.c_proj(y))

    # -- absorbed decode path -------------------------------------------------

    def _can_absorb(self, T, use_cache):
        return (self.absorb and use_cache and T == 1 and not self.training
                and self.decoupled_rope and self.kv_lora_rank > 0)

    def _attn_absorbed(self, q, c_kv, k_rope, B, T):
        """
        Score/aggregate entirely in latent space -- K and V are never built.

            q . k_C = q_nope^T (W_UK c_kv) = (W_UK^T q_nope)^T c_kv
            y       = sum_j a_j (W_UV c_kv_j) = W_UV (sum_j a_j c_kv_j)
        """
        nh, r = self.n_head, self.kv_lora_rank
        q_nope, q_rope = q.split([self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)
        w_uk = self.wk_b.weight.view(nh, self.qk_nope_head_dim, r)   # (nh, nope, r)
        w_uv = self.wv_b.weight.view(nh, self.v_head_dim, r)         # (nh, vd,   r)

        q_latent = torch.einsum('bhtn,hnr->bhtr', q_nope, w_uk)      # fold W_UK into Q
        scores = torch.einsum('bhtr,bsr->bhts', q_latent, c_kv)
        scores = scores + torch.einsum('bhtd,bsd->bhts', q_rope, k_rope)
        scores = scores / math.sqrt(self.head_dim)

        att = F.softmax(scores.float(), dim=-1).to(q.dtype)          # (B, nh, T, T_kv)
        o_latent = torch.einsum('bhts,bsr->bhtr', att, c_kv)         # aggregate in latent space
        y = torch.einsum('bhtr,hvr->bhtv', o_latent, w_uv)           # lift once, at the end
        y = y.transpose(1, 2).contiguous().view(B, T, nh * self.v_head_dim)
        return self.resid_dropout(self.c_proj(y))


class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu    = nn.GELU()
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x, freqs_cis=None, start_pos=0, use_cache=False):
        x = x + self.attn(self.ln_1(x), freqs_cis, start_pos, use_cache)
        x = x + self.mlp(self.ln_2(x))
        return x


@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50304 # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = True # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster
    # --- MLA ---
    q_lora_rank: int = 0      # low-rank dim for Q compression; 0 = direct Q projection
    kv_lora_rank: int = 512   # low-rank dim shared by K and V; 0 = fall back to vanilla MHA
    qk_rope_head_dim: int = 0 # >0 = DeepSeek decoupled RoPE (separate W_KR, shared single head).
                              # 0 = legacy: rotate the whole head, k_R derived from c_kv.
                              #     Legacy has no nope part, so the absorbed decode path is
                              #     unavailable -- prefer a small positive value (e.g. 16).
    v_head_dim: int = 0       # 0 = same as the q/k head dim (n_embd // n_head)
    use_rope: bool = True     # use RoPE instead of learned absolute position embeddings
    rope_theta: float = 10000.0
    absorb: bool = True       # use the absorbed (never-decompress) path during cached decode


class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = LayerNorm(config.n_embd, bias=config.bias),
        ))
        # learned absolute position embeddings are only used when RoPE is off
        if not config.use_rope:
            self.transformer.wpe = nn.Embedding(config.block_size, config.n_embd)

        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # with weight tying when using torch.compile() some warnings get generated:
        # "UserWarning: functional_call was passed multiple values for tied weights.
        # This behavior is deprecated and will be an error in future versions"
        # not 100% sure what this is, so far seems to be harmless. TODO investigate
        self.transformer.wte.weight = self.lm_head.weight # https://paperswithcode.com/method/weight-tying

        # one shared RoPE table for the whole model (not one per layer)
        if config.use_rope:
            rope_dim = self.transformer.h[0].attn.qk_rope_head_dim
            self.register_buffer(
                "freqs_cis",
                precompute_freqs_cis(rope_dim, config.block_size, config.rope_theta),
                persistent=False,  # derived from config, not a learned weight
            )
        else:
            self.freqs_cis = None

        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))

        # report number of parameters
        print("number of parameters: %.2fM" % (self.get_num_params()/1e6,))

    def get_num_params(self, non_embedding=True):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding and hasattr(self.transformer, 'wpe'):
            n_params -= self.transformer.wpe.weight.numel()
        return n_params

    # ---- KV cache -----------------------------------------------------------

    def reset_cache(self):
        """Drop all cached state. MUST be called before starting a new sequence."""
        for block in self.transformer.h:
            block.attn.reset_cache()

    def kv_cache_numbers_per_token(self):
        """Scalars cached per token across the whole model (all layers)."""
        return sum(b.attn.kv_cache_numbers_per_token() for b in self.transformer.h)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None, start_pos=0, use_cache=False):
        device = idx.device
        b, t = idx.size()
        assert start_pos + t <= self.config.block_size, \
            f"Cannot forward sequence of length {t} at position {start_pos}, block size is only {self.config.block_size}"

        # forward the GPT model itself
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (b, t, n_embd)
        if hasattr(self.transformer, 'wpe'):
            # learned absolute position embeddings (RoPE disabled) -- absolute, so offset by start_pos
            pos = torch.arange(start_pos, start_pos + t, dtype=torch.long, device=device)
            x = self.transformer.drop(tok_emb + self.transformer.wpe(pos))
        else:
            # RoPE handles positions inside the attention layers
            x = self.transformer.drop(tok_emb)
        freqs_cis = self.freqs_cis.to(device) if self.freqs_cis is not None else None
        for block in self.transformer.h:
            x = block(x, freqs_cis, start_pos, use_cache)
        x = self.transformer.ln_f(x)

        if targets is not None:
            # if we are given some desired targets also calculate the loss
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.lm_head(x[:, [-1], :]) # note: using list [-1] to preserve the time dim
            loss = None

        return logits, loss

    def crop_block_size(self, block_size):
        # model surgery to decrease the block size if necessary
        assert block_size <= self.config.block_size
        self.config.block_size = block_size
        if hasattr(self.transformer, 'wpe'):
            self.transformer.wpe.weight = nn.Parameter(self.transformer.wpe.weight[:block_size])
        if self.freqs_cis is not None:
            self.freqs_cis = self.freqs_cis[:block_size]
        for block in self.transformer.h:
            block.attn.block_size = block_size
            block.attn.reset_cache()
            if hasattr(block.attn, 'bias'):
                block.attn.bias = block.attn.bias[:, :, :block_size, :block_size]

    # NOTE: `from_pretrained` is intentionally removed. OpenAI GPT-2 checkpoints use
    # a single fused c_attn projection + learned wpe, which does not map onto MLA's
    # low-rank (wkv_a/wk_b/wv_b) layout. Train from scratch, or resume a checkpoint
    # saved by this file (`init_from='resume'`).

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        # start with all of the candidate parameters
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        print(f"using fused AdamW: {use_fused}")

        return optimizer

    def estimate_mfu(self, fwdbwd_per_iter, dt):
        """ estimate model flops utilization (MFU) in units of A100 bfloat16 peak FLOPS """
        # approximate: counts the dominant matmuls; MLA's low-rank matmuls are the same order
        N = self.get_num_params()
        cfg = self.config
        L, H, Q, T = cfg.n_layer, cfg.n_head, cfg.n_embd//cfg.n_head, cfg.block_size
        flops_per_token = 6*N + 12*L*H*Q*T
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        # express our flops throughput as ratio of A100 bfloat16 peak flops
        flops_achieved = flops_per_iter * (1.0/dt) # per second
        flops_promised = 312e12 # A100 GPU bfloat16 peak flops is 312 TFLOPS
        mfu = flops_achieved / flops_promised
        return mfu

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None, use_cache=True):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.

        With use_cache=True this runs as prefill (the whole prompt, start_pos=0) followed by
        decode (one token at a time, start_pos advancing). Mathematically identical to
        use_cache=False -- see test_mla.py::test_kv_cache_equivalence.
        """
        if not use_cache:
            for _ in range(max_new_tokens):
                # if the sequence context is growing too long we must crop it at block_size
                idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
                logits, _ = self(idx_cond)
                idx = torch.cat((idx, self._sample(logits, temperature, top_k)), dim=1)
            return idx

        assert idx.size(1) + max_new_tokens <= self.config.block_size, (
            f"cached generation needs prompt({idx.size(1)}) + new({max_new_tokens}) "
            f"<= block_size({self.config.block_size}); pass use_cache=False for the sliding-window path")
        self.reset_cache()   # a stale cache from a previous call would be read as history
        pos, cur = 0, idx
        for _ in range(max_new_tokens):
            logits, _ = self(cur, start_pos=pos, use_cache=True)
            pos += cur.size(1)                       # first pass is prefill, the rest are 1 token
            cur = self._sample(logits, temperature, top_k)
            idx = torch.cat((idx, cur), dim=1)
        return idx

    @staticmethod
    def _sample(logits, temperature, top_k):
        # pluck the logits at the final step and scale by desired temperature
        logits = logits[:, -1, :] / temperature
        # optionally crop the logits to only the top k options
        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float('Inf')
        if temperature == 0.0:
            return logits.argmax(dim=-1, keepdim=True)
        probs = F.softmax(logits, dim=-1)
        return torch.multinomial(probs, num_samples=1)

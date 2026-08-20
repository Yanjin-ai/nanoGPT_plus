"""
Full definition of a GPT Language Model with Multi-head Latent Attention (MLA).

MLA is introduced in DeepSeek-V2: https://arxiv.org/abs/2405.04434

Differences from the vanilla nanoGPT `model.py`:

1. K/V are computed from a shared low-rank latent `c_kv` instead of full-rank
   projections. Only `c_kv` (kv_lora_rank dims) needs to be cached per token,
   which shrinks the KV cache dramatically.
2. Q can optionally be computed from a low-rank latent `c_q` (q_lora_rank > 0).
3. Learned absolute position embeddings (`wpe`) are replaced by Rotary Position
   Embeddings (RoPE), with support for decoupled RoPE: only the first
   `qk_rope_head_dim` dims of each head are rotated.
4. `GPT.from_pretrained` is removed: OpenAI GPT-2 checkpoints do not map onto
   MLA weights (different projection layouts). Train from scratch or resume a
   checkpoint produced by this file.

The file is otherwise a drop-in replacement for `model.py`: the public API
(`GPT`, `GPTConfig`, `configure_optimizers`, `estimate_mfu`, `generate`) and
the training-time call convention `logits, loss = model(x, y)` are unchanged.

Reference for the MLA weight layout (per DeepSeek-V2):
    c_q  = norm_q( x @ W_DQ^T )            # (B, T, q_lora_rank)
    q    = c_q @ W_UQ^T                    # (B, T, n_head * head_dim)
    c_kv = norm_kv( x @ W_DKV^T )          # (B, T, kv_lora_rank)  <- cached
    k    = c_kv @ W_UK^T                   # (B, T, n_head * head_dim)
    v    = c_kv @ W_UV^T                   # (B, T, n_head * head_dim)
The first qk_rope_head_dim dims of q/k get RoPE; the rest stay non-rotated.
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
    freqs_cis: (T, d//2) complex64.
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

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head
        self.dropout = config.dropout

        # ---- MLA knobs ----
        self.q_lora_rank = getattr(config, 'q_lora_rank', 0)
        self.kv_lora_rank = getattr(config, 'kv_lora_rank', 0)
        self.use_rope = getattr(config, 'use_rope', True)
        self.qk_rope_head_dim = getattr(config, 'qk_rope_head_dim', 0) or self.head_dim
        assert self.qk_rope_head_dim <= self.head_dim
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
            self.wk_b = nn.Linear(self.kv_lora_rank, self.n_head * self.head_dim, bias=False)
            self.wv_b = nn.Linear(self.kv_lora_rank, self.n_head * self.head_dim, bias=False)
        else:
            self.wk = nn.Linear(self.n_embd, self.n_head * self.head_dim, bias=config.bias)
            self.wv = nn.Linear(self.n_embd, self.n_head * self.head_dim, bias=config.bias)

        # output projection
        self.c_proj = nn.Linear(self.n_embd, self.n_embd, bias=config.bias)
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

        # rotary position embeddings (persistent=False: derived from config, not saved)
        if self.use_rope:
            self.register_buffer(
                "freqs_cis",
                precompute_freqs_cis(self.qk_rope_head_dim, config.block_size, config.rope_theta),
                persistent=False,
            )

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)

        # ---- project to Q, K, V ----
        if self.q_lora_rank > 0:
            q = self.wq_b(self.q_norm(self.wq_a(x)))
        else:
            q = self.wq(x)

        if self.kv_lora_rank > 0:
            c_kv = self.kv_norm(self.wkv_a(x))  # (B, T, kv_lora_rank) -- the only thing to cache
            k = self.wk_b(c_kv)
            v = self.wv_b(c_kv)
        else:
            k = self.wk(x)
            v = self.wv(x)

        # move head forward to be the batch dim: (B, nh, T, hs)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        # ---- decoupled rotary position embeddings ----
        if self.use_rope:
            freqs_cis = self.freqs_cis[:T].to(x.device)  # (T, qk_rope_head_dim//2)
            q_rope = apply_rotary_emb(q[..., :self.qk_rope_head_dim], freqs_cis)
            k_rope = apply_rotary_emb(k[..., :self.qk_rope_head_dim], freqs_cis)
            q = torch.cat([q_rope, q[..., self.qk_rope_head_dim:]], dim=-1)
            k = torch.cat([k_rope, k[..., self.qk_rope_head_dim:]], dim=-1)

        # ---- causal self-attention ----
        if self.flash:
            # efficient attention using Flash Attention CUDA kernels
            y = torch.nn.functional.scaled_dot_product_attention(
                q, k, v, attn_mask=None,
                dropout_p=self.dropout if self.training else 0,
                is_causal=True)
        else:
            # manual implementation of attention
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))
            att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y


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

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
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
    qk_rope_head_dim: int = 0 # per-head dim that receives RoPE; 0 = rotate the whole head
    use_rope: bool = True     # use RoPE instead of learned absolute position embeddings
    rope_theta: float = 10000.0


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

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=device) # shape (t)

        # forward the GPT model itself
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (b, t, n_embd)
        if hasattr(self.transformer, 'wpe'):
            # learned absolute position embeddings (RoPE disabled)
            pos_emb = self.transformer.wpe(pos) # position embeddings of shape (t, n_embd)
            x = self.transformer.drop(tok_emb + pos_emb)
        else:
            # RoPE handles positions inside the attention layers
            x = self.transformer.drop(tok_emb)
        for block in self.transformer.h:
            x = block(x)
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
        for block in self.transformer.h:
            if hasattr(block.attn, 'bias'):
                block.attn.bias = block.attn.bias[:,:,:block_size,:block_size]
            if block.attn.use_rope:
                block.attn.freqs_cis = block.attn.freqs_cis[:block_size]

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
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """
        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at block_size
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            # forward the model to get the logits for the index in the sequence
            logits, _ = self(idx_cond)
            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature
            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            # append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)

        return idx

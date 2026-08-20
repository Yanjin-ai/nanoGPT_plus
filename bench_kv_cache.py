"""
Measure what MLA actually buys: KV cache footprint and decode speed.

The headline MLA claim is "we cache kv_lora_rank numbers per token instead of
2 * n_head * head_dim". That claim is only real if there IS a cache, so this
script (a) re-verifies cached == uncached on the trained weights, then
(b) reports the footprint and the wall-clock difference.

Run:
$ python bench_kv_cache.py --out_dir=out-shakespeare-char-mla --device=cpu
$ python bench_kv_cache.py --device=cpu --kv_lora_rank=0   # untrained MHA reference
"""
import os
import time
from contextlib import nullcontext

import torch

from model_mla import GPT, GPTConfig

# -----------------------------------------------------------------------------
out_dir = ''          # load a checkpoint from here; empty = random init from the knobs below
device = 'cuda' if torch.cuda.is_available() else 'cpu'
dtype = 'float32'
seed = 1337
prompt_len = 32
max_new_tokens = 256
# used only when out_dir is empty
block_size, vocab_size, n_layer, n_head, n_embd = 512, 65, 6, 6, 384
kv_lora_rank, q_lora_rank, qk_rope_head_dim, v_head_dim = 128, 0, 16, 0
exec(open('configurator.py').read())
# -----------------------------------------------------------------------------

torch.manual_seed(seed)
device_type = 'cuda' if 'cuda' in device else 'cpu'
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

if out_dir:
    ckpt = torch.load(os.path.join(out_dir, 'ckpt.pt'), map_location=device, weights_only=False)
    conf = GPTConfig(**ckpt['model_args'])
    model = GPT(conf)
    sd = {k[10:] if k.startswith('_orig_mod.') else k: v for k, v in ckpt['model'].items()}
    model.load_state_dict(sd)
    print(f"loaded {out_dir}/ckpt.pt  (iter {ckpt['iter_num']}, best val {float(ckpt['best_val_loss']):.4f})")
else:
    conf = GPTConfig(block_size=block_size, vocab_size=vocab_size, n_layer=n_layer, n_head=n_head,
                     n_embd=n_embd, dropout=0.0, bias=False, kv_lora_rank=kv_lora_rank,
                     q_lora_rank=q_lora_rank, qk_rope_head_dim=qk_rope_head_dim,
                     v_head_dim=v_head_dim)
    model = GPT(conf)
model.eval().to(device)

prompt = torch.randint(0, conf.vocab_size, (1, prompt_len), device=device)

# --- 1. equivalence on the TRAINED weights, not just a random init --------------
with ctx:
    a = model.generate(prompt, max_new_tokens, temperature=0.0, use_cache=False)
    b = model.generate(prompt, max_new_tokens, temperature=0.0, use_cache=True)
same = torch.equal(a, b)
print(f"\ncached == uncached : {'YES' if same else 'NO  <-- BUG'}")
assert same, "KV cache changed the model's output; it must be an identity transform"

# --- 2. footprint ---------------------------------------------------------------
attn0 = model.transformer.h[0].attn
per_tok_mla = model.kv_cache_numbers_per_token()
per_tok_mha = conf.n_layer * (attn0.n_head * attn0.head_dim + attn0.n_head * attn0.v_head_dim)
bytes_per = torch.empty(0, dtype=ptdtype).element_size()
ctx_len = conf.block_size

print(f"\nKV cache, whole model, {ctx_len} tokens, batch 1, {dtype}:")
print(f"  MLA (cached c_kv{' + k_R' if attn0.decoupled_rope else ''})"
      f"   {per_tok_mla:6d} numbers/token   {per_tok_mla * ctx_len * bytes_per / 1024:8.1f} KiB")
print(f"  MHA equivalent (K and V)   {per_tok_mha:6d} numbers/token   {per_tok_mha * ctx_len * bytes_per / 1024:8.1f} KiB")
print(f"  reduction                  {per_tok_mha / per_tok_mla:6.2f}x")

live = sum(buf.numel() for blk in model.transformer.h
           for buf in (blk.attn.cache_c_kv, blk.attn.cache_k_rope, blk.attn.cache_k, blk.attn.cache_v)
           if buf is not None)
print(f"  actually allocated after the run: {live * bytes_per / 1024:.1f} KiB")

# --- 3. wall clock --------------------------------------------------------------
def timed(use_cache, absorb=True):
    for blk in model.transformer.h:
        blk.attn.absorb = absorb
    with ctx:
        model.generate(prompt, 16, temperature=0.0, use_cache=use_cache)  # warmup
        if device_type == 'cuda':
            torch.cuda.synchronize()
        t0 = time.time()
        model.generate(prompt, max_new_tokens, temperature=0.0, use_cache=use_cache)
        if device_type == 'cuda':
            torch.cuda.synchronize()
    return time.time() - t0

t_nocache = timed(False)
t_cache_naive = timed(True, absorb=False)
t_cache_absorb = timed(True, absorb=True)
print(f"\ndecode {max_new_tokens} tokens from a {prompt_len}-token prompt ({device}):")
print(f"  no cache          {t_nocache*1000:8.1f} ms   ({t_nocache/max_new_tokens*1000:.2f} ms/token)   1.00x")
print(f"  cache, naive      {t_cache_naive*1000:8.1f} ms   ({t_cache_naive/max_new_tokens*1000:.2f} ms/token)   {t_nocache/t_cache_naive:.2f}x")
if attn0.decoupled_rope:
    print(f"  cache, absorbed   {t_cache_absorb*1000:8.1f} ms   ({t_cache_absorb/max_new_tokens*1000:.2f} ms/token)   {t_nocache/t_cache_absorb:.2f}x")
else:
    print("  cache, absorbed   n/a (needs qk_rope_head_dim > 0; this checkpoint uses the legacy layout)")

"""
Correctness tests for model_mla.py.

The KV cache and the absorbed decode path are *pure optimisations*: they change
how the model is computed, never what it computes. So every test here is an
equivalence assertion -- some fast path must agree with the slow reference path.

That is deliberate. Position bugs, causal-mask bugs, stale-cache bugs and
train/eval-mode bugs are all SILENT: they do not raise, they just make the
output worse, which is easy to misread as "the model is small". Turning them
into a boolean is the whole point.

Run:  python test_mla.py       (or: pytest test_mla.py)
"""

import torch

from model_mla import GPT, GPTConfig

torch.manual_seed(1337)

BASE = dict(block_size=64, vocab_size=97, n_layer=3, n_head=4, n_embd=64, dropout=0.0, bias=False)

# The configurations we care about, each exercising a different code path.
CONFIGS = {
    'mla-legacy-rope':  dict(kv_lora_rank=16, qk_rope_head_dim=0),                 # whole head rotated
    'mla-decoupled':    dict(kv_lora_rank=16, qk_rope_head_dim=4),                 # DeepSeek layout
    'mla-decoupled-v':  dict(kv_lora_rank=16, qk_rope_head_dim=4, v_head_dim=12),  # v_head_dim != head_dim
    'mla-qlora':        dict(kv_lora_rank=16, q_lora_rank=24, qk_rope_head_dim=4),
    'mla-norope':       dict(kv_lora_rank=16, use_rope=False),                     # learned wpe path
    'mha-fallback':     dict(kv_lora_rank=0,  qk_rope_head_dim=4),                 # full-rank K/V
    'mha-norope':       dict(kv_lora_rank=0,  use_rope=False),
}


def build(**kw):
    model = GPT(GPTConfig(**{**BASE, **kw}))
    model.eval()   # dropout must be off: a cached noisy c_kv would poison the whole suffix
    return model


def _report(name, ok, detail=''):
    print(f"  [{'ok' if ok else 'FAIL'}] {name}{'  ' + detail if detail else ''}")
    assert ok, f"{name} failed {detail}"


# ---------------------------------------------------------------------------

@torch.no_grad()
def test_kv_cache_equivalence():
    """Cached generation must reproduce uncached generation token for token."""
    print("test_kv_cache_equivalence")
    for name, kw in CONFIGS.items():
        model = build(**kw)
        prompt = torch.randint(0, BASE['vocab_size'], (2, 5))
        # temperature=0.0 -> greedy. Sampling would diverge on RNG state alone.
        a = model.generate(prompt, 24, temperature=0.0, use_cache=False)
        b = model.generate(prompt, 24, temperature=0.0, use_cache=True)
        _report(name, torch.equal(a, b), f"(mismatch at {int((a != b).float().argmax())})" if not torch.equal(a, b) else '')


@torch.no_grad()
def test_absorbed_matches_naive():
    """The absorbed decode path folds W_UK into Q and W_UV into the output; same answer."""
    print("test_absorbed_matches_naive")
    for name, kw in CONFIGS.items():
        if kw.get('qk_rope_head_dim', 0) == 0 or kw.get('kv_lora_rank', 1) == 0:
            continue  # absorption needs a nope part and a latent cache
        model = build(**kw)
        prompt = torch.randint(0, BASE['vocab_size'], (2, 5))
        for b in model.transformer.h:
            b.attn.absorb = False
        naive = model.generate(prompt, 24, temperature=0.0, use_cache=True)
        for b in model.transformer.h:
            b.attn.absorb = True
        absorbed = model.generate(prompt, 24, temperature=0.0, use_cache=True)
        _report(name, torch.equal(naive, absorbed))


@torch.no_grad()
def test_chunked_prefill():
    """Prefilling in two chunks must equal prefilling in one (exercises the offset mask)."""
    print("test_chunked_prefill")
    for name, kw in CONFIGS.items():
        model = build(**kw)
        idx = torch.randint(0, BASE['vocab_size'], (2, 12))
        model.reset_cache()
        one, _ = model(idx, start_pos=0, use_cache=True)
        model.reset_cache()
        model(idx[:, :7], start_pos=0, use_cache=True)
        two, _ = model(idx[:, 7:], start_pos=7, use_cache=True)
        _report(name, torch.allclose(one, two, atol=1e-4, rtol=1e-3),
                f"maxdiff={(one - two).abs().max():.2e}")


@torch.no_grad()
def test_cache_is_reset_between_generations():
    """Two back-to-back generations must not contaminate each other."""
    print("test_cache_is_reset_between_generations")
    for name, kw in CONFIGS.items():
        model = build(**kw)
        p1 = torch.randint(0, BASE['vocab_size'], (1, 4))
        p2 = torch.randint(0, BASE['vocab_size'], (1, 6))
        first = model.generate(p1, 12, temperature=0.0, use_cache=True)
        second = model.generate(p2, 12, temperature=0.0, use_cache=True)
        model.reset_cache()
        fresh = model.generate(p2, 12, temperature=0.0, use_cache=True)
        _report(name, torch.equal(second, fresh))


@torch.no_grad()
def test_causality():
    """Changing token t must not move any logit at position < t."""
    print("test_causality")
    for name, kw in CONFIGS.items():
        model = build(**kw)
        a = torch.randint(0, BASE['vocab_size'], (1, 16))
        b = a.clone()
        b[0, -1] = (b[0, -1] + 1) % BASE['vocab_size']
        la, _ = model(a, a)
        lb, _ = model(b, b)
        delta = (la[:, :-1] - lb[:, :-1]).abs().max().item()
        _report(name, delta == 0.0, f"delta={delta:.2e}")


def test_train_step():
    """Every config must forward+backward with a gradient on every parameter."""
    print("test_train_step")
    for name, kw in CONFIGS.items():
        model = GPT(GPTConfig(**{**BASE, **kw, 'dropout': 0.1}))
        model.train()
        x = torch.randint(0, BASE['vocab_size'], (2, 16))
        y = torch.randint(0, BASE['vocab_size'], (2, 16))
        _, loss = model(x, y)
        loss.backward()
        missing = [n for n, p in model.named_parameters() if p.grad is None]
        _report(name, not missing and torch.isfinite(loss), f"loss={loss.item():.3f}")


@torch.no_grad()
def test_cache_footprint():
    """Report the headline MLA number: scalars cached per token."""
    print("test_cache_footprint (informational)")
    mha = build(**CONFIGS['mha-fallback']).kv_cache_numbers_per_token()
    for name, kw in CONFIGS.items():
        n = build(**kw).kv_cache_numbers_per_token()
        print(f"  {name:18s} {n:5d} numbers/token   ({mha / n:.2f}x smaller than MHA)")


if __name__ == '__main__':
    import io, contextlib
    tests = [test_kv_cache_equivalence, test_absorbed_matches_naive, test_chunked_prefill,
             test_cache_is_reset_between_generations, test_causality, test_train_step,
             test_cache_footprint]
    for t in tests:
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):   # swallow GPT.__init__'s param printout
            pass
        t()
    print("\nall tests passed")

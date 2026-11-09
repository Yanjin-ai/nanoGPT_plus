# Every Metric Said My MLA Worked. It Delivered Nothing.

*What implementing DeepSeek's Multi-head Latent Attention on nanoGPT taught me about the half of an architecture nobody checks.*

---

I set out to learn the DeepSeek stack the only way I trust: by rebuilding it on something small enough to hold in my head. nanoGPT — 300 lines of model, 300 lines of training loop — was the obvious host. First target: **MLA**, Multi-head Latent Attention, the thing that lets DeepSeek-V2 serve enormous context windows without drowning in KV cache.

A few days later I had it working. Here's what my run said:

| | baseline (MHA) | MLA |
|---|---|---|
| parameters | 10.65 M | **9.76 M** (−8.4%) |
| best val loss | 1.4716 | 1.4757 (+0.004) |
| final val loss | 1.7127 | **1.6636** |

Fewer parameters. Same quality. *Less overfitting.* Every number I could see said it worked.

It delivered **zero** of what MLA exists for.

Not "less than expected." Zero. Exactly 0% of the benefit, and I couldn't tell, because nothing I was measuring could have told me.

This post is about why — and about the general lesson underneath, which is much bigger than MLA.

---

## What MLA is, in one minute

When a language model generates text, it re-reads its own history at every step. To avoid recomputing that history, you cache the keys and values for every past token. That cache grows linearly with context and with batch size, and on real deployments it becomes the thing that decides how many users a GPU can serve.

The family of fixes:

| approach | idea |
|---|---|
| **MQA** | all heads share one K/V. Huge savings, real quality cost. |
| **GQA** | heads share in groups. The interpolation Llama uses. |
| **MLA** | store a **low-rank latent**, expand to K/V on demand. |

MQA and GQA subtract in the *head* dimension — they take away head independence. MLA compresses in the *feature* dimension — every head keeps its own K and V, they're just constrained to a low-dimensional subspace.

That distinction matters more than it sounds. Preserving head independence is what makes MLA gentler on quality, and it's also what makes it more complicated to implement correctly. Everything hard about MLA follows from that trade.

**The cache math:**

```
MHA:  2 × n_head × head_dim   numbers per token per layer
MLA:  kv_lora_rank            numbers per token per layer
```

For DeepSeek-V2 that's 32,768 versus 576. About **57×**.

---

## The steps

Here's the sequence I'd recommend, which is *not* the sequence I actually used. Four stages:

| stage | what changes | changes the model? | needs retraining? | how you verify |
|---|---|---|---|---|
| **A. Training-side layout** | low-rank K/V projections | yes | yes | val loss vs baseline |
| **B. KV cache** | inference path | **no** | **no** | equivalence assertion |
| **C. Decoupled RoPE** | separate rope projection | yes | yes | val loss + reuse B's tests |
| **D. Absorption** | decode path only | **no** | **no** | reuse B's tests |

Two things about this ordering are worth stealing.

**B before C.** The KV cache is a pure inference-side change. You can validate it against a checkpoint you already have — no retraining, no waiting. And it produces a test harness that then protects stages C and D. **Build the safety net before you touch the structure.**

**Notice the "changes the model?" column.** That single question determines how you're allowed to verify. I'll come back to it, because it's the most portable idea in this whole post.

I did A, shipped it, and declared victory. Stages B, C and D didn't exist in my head as work items. That's the failure, and it's worth naming precisely.

---

## The trap: an architecture has two halves

My change touched exactly two places: layer definitions in `__init__`, and tensor shapes in `forward`. Nothing about state. Nothing about the inference path. Nothing about how training and generation differ.

For most architecture changes you'd get away with that. **MLA is the one where you can't**, because 100% of MLA's benefit lands on the inference path. It offers nothing at training time — it's actually *slightly slower*, since low-rank projections add a matmul. The parameter savings I was so pleased about are a side effect, not the point.

And here's the part that still bothers me. My own code had this comment in it:

```python
c_kv = self.kv_norm(self.wkv_a(x))  # <- the only thing to cache
```

I'd read the paper. I knew which tensor was supposed to be cached. I wrote it down.

And then I never cached anything, because I read that line as a *description* when it was a *promise that needed code*.

nanoGPT's `generate()` re-runs the entire prefix at every step. No cache, by design — Karpathy optimized that repo for training clarity, and he's right to. But there's an asymmetry I completely missed:

> For vanilla GPT, a missing KV cache is a **performance** problem.
> For MLA, a missing KV cache is a **missing feature**.

Same absent code. Totally different meaning, depending on what you built on top of it.

---

## The five pitfalls, and how to handle the tricky ones

Once I started building the inference path, I hit a wall of bugs that share one nasty property: **none of them raise an exception.** They just make the output worse — which, on a small model, is trivially misread as "well, it's a small model."

Here they are, in the order they bit me.

### 1. RoPE positions (silent)

RoPE stores *absolute* positions and produces relative behavior through the dot product. Without a cache, `freqs[:T]` works because your input always starts at position 0.

With a cache, you're feeding in one token whose real position might be 500. If you still index `freqs[:1]`, you've told the model that token is at position 0.

**The tell:** generation starts fine and degrades. The prompt (prefill) is correct, so the first few tokens look plausible, then it drifts into repetition.

**The fix**, and note that query and key need *different* slices:

```python
q_freqs = freqs[start_pos : start_pos + T]   # the new token's true position
k_freqs = freqs[: start_pos + T]             # the whole history
```

### 2. The causal mask (silent, and the trickiest one)

This is the one I'd flag hardest, because the API actively misleads you.

PyTorch's `scaled_dot_product_attention(..., is_causal=True)` builds a triangular mask aligned to the **top-left corner**: query row `i` sees key columns `0..i`, both indexed from zero.

During training `q_len == k_len`, so top-left alignment equals diagonal alignment. Correct.

During decode, `q_len = 1` and `k_len = 501`. Top-left alignment means your single query row sees key column 0 — and *only* column 0. **Your entire history is masked away.** Silently.

**The fix** is to stop treating `is_causal` as a constant:

```python
if T_kv == T and start_pos == 0:  return None, True    # training / prefill
if T == 1:                        return None, False   # decode: see everything cached
# chunked prefill needs an explicit offset mask
q_pos = torch.arange(start_pos, start_pos + T).unsqueeze(1)
k_pos = torch.arange(0, T_kv).unsqueeze(0)
return (q_pos >= k_pos), False
```

And don't forget the manual attention fallback — `self.bias[:, :, :T, :T]` carries the exact same assumption, and on CPU that's the branch you're actually running.

### 3. Stale cache between generations (silent)

State means leftovers. If your sampling script loops to draw N samples, sample #1 is clean and every sample after it inherits the previous one's history.

**The fix:** reset inside `generate()`, at the entry point. Don't rely on callers remembering. I nearly shipped this one because I only ever tested with `num_samples=1`.

### 4. Forgetting `model.eval()` (silent, and amplified)

Without a cache, leaving dropout on just makes generation a bit noisier. With a cache, a dropout-corrupted latent gets **written into the cache** and poisons every subsequent token. The cache doesn't just expose the bug, it persists it.

### 5. Running past `block_size` (loud, if you let it be)

Original nanoGPT handles overflow with a sliding window: `idx[:, -block_size:]`. With a cache that gets genuinely complicated — you'd need a rolling buffer and recomputed positions.

**The fix I recommend: refuse.**

```python
assert idx.size(1) + max_new_tokens <= self.config.block_size, \
    "cached generation needs prompt+new <= block_size; use_cache=False for the sliding-window path"
```

Drawing a boundary explicitly beats pretending to support something you haven't built. This is the one pitfall I turned *loud* on purpose, and it's the only one I never debugged.

---

## The tool that catches all five

Notice what four of those five have in common. They're silent. No stack trace, no NaN, no shape error. Just worse output.

If your only verification method is "run it and see if the text looks okay," you cannot catch any of them. And that *was* my only method, which is the actual root cause of everything above.

Here's the fix, and it generalizes far beyond MLA.

**Every change you make to a model falls into exactly one of three categories:**

| category | examples | how you're allowed to verify |
|---|---|---|
| **① Changes the model** (`f' ≠ f`) | MHA→MLA, adding MoE, changing head dims | **retrain**, compare val loss |
| **② Identity transform** (`f' = f`) | KV cache, absorption, FlashAttention, `torch.compile` | **assert equivalence**, instantly |
| **③ Approximation** (`f' ≈ f`) | quantization, sliding-window attention, pruning | assert an error bound, then check downstream |

A KV cache is category ②. It is *mathematically required* to produce identical output, because the things it stores are deterministic functions of tokens that are already fixed. It's memoization. `fib(n)` with a lookup table returns the same number.

Which means:

> **Verifying a category-② change by retraining is pure waste.** Cache bugs don't affect training loss — training never touches the cache. You'd be measuring a quantity that is causally disconnected from the bug.

So write the assertion instead:

```python
a = model.generate(prompt, N, temperature=0.0, use_cache=False)  # slow reference
b = model.generate(prompt, N, temperature=0.0, use_cache=True)   # fast path
assert torch.equal(a, b)
```

Two non-negotiable details, both of which I got wrong on the first attempt:

- **Greedy decoding** (`temperature=0.0`). With sampling, the two paths diverge on RNG state alone and the test means nothing.
- **`model.eval()`**. With dropout on, the paths *must* differ, and you've written a test that can never pass.

That two-line assertion catches every one of the five pitfalls. Position bug? Different logits, fails. Mask bug? Wildly different logits, fails. Stale cache? Second call fails. Dropout on? Fails immediately.

**It collapses five invisible failure modes into one boolean.**

I ended up with five assertions across seven configurations — 38 checks total, covering the decoupled and legacy RoPE layouts, decoupled `v_head_dim`, low-rank queries, and the plain MHA fallback. That suite took maybe an hour to write and has caught every regression since.

Here's the rule I'd hand to anyone doing this kind of work:

> **If you claim a change is "just an optimization," you must be able to write an equivalence assertion for it. If you can't, it isn't just an optimization — it changes the model, and you haven't noticed yet.**

---

## The tricky one: RoPE breaks the good part

One more pitfall deserves its own section, because it's the one where I misread the paper rather than the API — and because it's a genuinely beautiful piece of design.

MLA's real superpower isn't just a smaller cache. It's that you never have to *decompress* it. Watch:

```
q · k = (W_UQ c_q)ᵀ (W_UK c_kv) = c_qᵀ (W_UQᵀ W_UK) c_kv
                                       └─── constant ───┘
```

That middle matrix is fixed. Precompute it once, fold it into the query side, and now you're scoring **directly in latent space**. The key vector is never built. Same trick on the value side. This is called **absorption**, and since decoding is bandwidth-bound, not compute-bound, it's where a large chunk of the real speedup lives.

RoPE ruins it. Rotations `R_m` and `R_n` leave you with:

```
q · k = c_qᵀ W_UQᵀ R_{n-m} W_UK c_kv
                   └──┬──┘
          depends on relative distance — can't be precomputed
```

DeepSeek's answer: split each head. Most dimensions carry no positional rotation and stay absorbable. A small slice carries the position — and critically, **that slice gets its own projection from `x`, as a single head shared across all attention heads, cached alongside the latent.**

My first version implemented the *appearance* of this — some dimensions rotated, some not — but derived the rotated part by slicing the decompressed key. Which meant caching still worked, but absorption was impossible. I'd copied the shape of the solution without the mechanism.

Worse, my default config rotated the *entire* head, so there were zero absorbable dimensions. I had implemented decoupled RoPE with the decoupling set to zero.

The lesson I took from this is a reading heuristic I now use constantly:

> **When a paper contains a step that looks needlessly complicated — "why not just slice it out of the latent, that's so much simpler?" — that step is usually the whole point.** Authors simplify wherever they can; page space is expensive. When you spot an obvious simplification, assume you've misunderstood the design, not that the authors were sloppy.

---

## The numbers, finally

After building the inference path, here's what my model actually does. Small model, CPU, 6 layers, `n_embd=384`, `kv_lora_rank=128`:

```
cached == uncached : YES          ← assert this BEFORE you benchmark anything

KV cache          768 vs 4608 numbers/token        6.00× smaller
decode 200 tok    1296 ms → 333 ms                 3.89× faster
```

And with the decoupled layout plus absorption, over 400 tokens:

```
no cache          4498 ms         1.00×
cache, naive       894 ms         5.03×
cache, absorbed    730 ms         6.16×      ← absorption adds 1.22× on top
```

Three things to read off this:

**The cache carries 100% of MLA's inference benefit.** Without it, MLA is strictly worse than MHA at inference — same speed, less expressive.

**Decoupling costs a little cache to buy absorption.** 768 → 864 numbers per token, in exchange for a decode path that never decompresses. Clearly worth it.

**My 6× is not DeepSeek's 57×, and that's fine.** The ratio scales with `n_head × head_dim`. On a 6-head, 64-dim model there simply isn't as much to compress. If I hadn't measured it myself I'd have spent a week wondering why my numbers "didn't match the paper."

That last point is the general one: **run the measurement on your own model.** Paper ratios are properties of the paper's configuration, not of the technique.

---

## What I'd tell myself on day one

Not "learn MLA better." Three structural things:

**1. Write `forward(x, start_pos, use_cache)` from the very beginning**, even if both extra arguments go unused for weeks. Retrofitting state into a stateless function means editing every caller in the chain — block, model, generate, sampling script. Doing it upfront costs nothing.

**2. Write one equivalence test before writing the optimization it validates.** Not after. Writing the assertion forces you to articulate *why* the two paths should agree, and that articulation is where you notice you don't actually understand the state machine yet.

**3. Keep the fallbacks.** `kv_lora_rank=0` → plain MHA. `use_rope=False` → learned positions. I did do this one right, and it's the only reason I have a baseline comparison at all. **Delete the old path and you've deleted your control group.**

And the umbrella question, the one that reorganized how I think about this work:

> Before adding anything, ask: **does this change the model, or just how it's computed?**
>
> Changes the model → retrain, and decide up front which axis you're holding constant.
> Just how it's computed → assert equality against the slow path. Never "run it and see."

Half of doing this well is knowing which question you're allowed to answer with a loss curve.

---

## Next

MTP (multi-token prediction) is next, then MoE. MTP is the cleaner follow-up — it's orthogonal to attention, it has no distributed-training landmines, and this time I have a test suite waiting for it.

Full code, the four-stage implementation guide, the experiment log, and a longer write-up of everything above:
**[github.com/Yanjin-ai/nanoGPT_plus](https://github.com/Yanjin-ai/nanoGPT_plus)**

---

## References

**Primary**

- Liu et al., *DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model* — [arXiv:2405.04434](https://arxiv.org/abs/2405.04434). §2.1 is MLA, including the decoupled-RoPE derivation. Three pages, and the only source you strictly need.
- DeepSeek-AI, *DeepSeek-V3 Technical Report* — [arXiv:2412.19437](https://arxiv.org/abs/2412.19437). MLA at scale, plus MoE and MTP.

**Understanding the family**

- Shazeer, *Fast Transformer Decoding: One Write-Head is All You Need* (MQA) — [arXiv:1911.02150](https://arxiv.org/abs/1911.02150)
- Ainslie et al., *GQA: Training Generalized Multi-Query Transformer Models* — [arXiv:2305.13245](https://arxiv.org/abs/2305.13245). Read it to see the ceiling on subtracting in the head dimension, which is what motivates MLA's different direction.

**Inference-side context**

- Dao et al., *FlashAttention* — [arXiv:2205.14135](https://arxiv.org/abs/2205.14135). Another identity transform; orthogonal to KV caching and stacks with it.
- Kwon et al., *Efficient Memory Management for LLM Serving with PagedAttention* (vLLM) — [arXiv:2309.06180](https://arxiv.org/abs/2309.06180). What production-grade cache management looks like.

**Code**

- Karpathy, *nanoGPT* — [github.com/karpathy/nanoGPT](https://github.com/karpathy/nanoGPT)
- This project — [github.com/Yanjin-ai/nanoGPT_plus](https://github.com/Yanjin-ai/nanoGPT_plus)

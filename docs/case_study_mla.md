# 技术学习案例：在 nanoGPT 上从零实现 MLA

> 从 Karpathy 的 nanoGPT（标准 MHA）出发，完整实现 DeepSeek-V2 的 Multi-head Latent
> Attention，包括训练侧的低秩权重布局、推理侧的 KV cache、decoupled RoPE 和矩阵吸收，
> 并用等价性测试和实测数据验证每一步。
>
> 配套代码：[`model_mla.py`](../model_mla.py) · [`test_mla.py`](../test_mla.py) ·
> [`bench_kv_cache.py`](../bench_kv_cache.py) · [`mla_min.py`](../mla_min.py)
> 相关文档：[MLA 深度解析](mla_deep_dive.md) · [推理路径](inference_path.md) ·
> [实验记录](experiments.md) · [MoE 踩坑清单](moe_pitfalls.md)

## 目录

- [0. 这个案例是什么](#0-这个案例是什么)
- [1. 起点：问题在哪](#1-起点问题在哪)
- [2. 解法谱系：MHA → MQA → GQA → MLA](#2-解法谱系mha--mqa--gqa--mla)
- [3. MLA 的三个核心思想](#3-mla-的三个核心思想)
- [4. 从公式到 shape：完整维度表](#4-从公式到-shape完整维度表)
- [5. 分四个阶段实施](#5-分四个阶段实施)
- [6. 五个静默陷阱](#6-五个静默陷阱)
- [7. 验证方法论：恒等变换与等价性测试](#7-验证方法论恒等变换与等价性测试)
- [8. 测量：把论文数字变成你自己的数字](#8-测量把论文数字变成你自己的数字)
- [9. 复盘：三个认知缺口](#9-复盘三个认知缺口)
- [10. 复现步骤](#10-复现步骤)
- [11. 延伸阅读](#11-延伸阅读)

---

## 0. 这个案例是什么

**目标读者**：读过 transformer 代码、能看懂 nanoGPT，想真正搞懂 MLA 而不只是"知道它压缩了 KV cache"的人。

**读完你应该能做到**：

1. 从"为什么需要"推导出 MLA 的每一个设计决策，而不是照抄论文的公式；
2. 说清楚**吸收（absorption）**是什么、RoPE 为什么会破坏它、DeepSeek 怎么绕过；
3. 自己写出一个带 KV cache 的推理路径，并知道会踩哪五个静默坑；
4. 用**等价性测试**而不是"跑一下看 loss"来验证一个优化是否正确；
5. 把论文里的比例数字换算成你自己模型上的实测数字。

**这个案例的特点**：它不是一份"正确答案"，而是一次**真实的、带弯路的**改造记录。第一版实现只做对了一半——训练能跑、参数量下降、val loss 持平，看起来完全成功，但 MLA 存在的理由完全没有兑现。第 9 节专门复盘这个弯路，因为**那个弯路比正确答案更有教学价值**。

---

## 1. 起点：问题在哪

### 1.1 自回归解码在重复做什么

生成时是一个 token 一个 token 出的。已有前缀 `t₁..t₅`，要出 `t₆`：

- 位置 6 的 query 要和位置 1..6 的 key 算注意力，再对 1..6 的 value 加权；
- **位置 1..5 的 K/V 完全不依赖 t₆**——它们只由已经确定的 t₁..t₅ 决定。

每步重算一遍，是纯粹的重复劳动：

| | 每步计算量 | 生成 T 个 token 总量 | 额外显存 |
|---|---|---|---|
| 无 cache | O(t²) | **O(T³)** | 0 |
| 有 cache | O(t) | **O(T²)** | 存 K/V |

拿显存换时间，换得极其划算，所以所有推理框架都做。

### 1.2 于是 KV cache 成了推理的主要显存开销

标准 MHA 每 token 每层要缓存：

```
2 × n_head × head_dim        （K 一份，V 一份）
```

对 DeepSeek-V2 的规模（`n_head=128, head_dim=128`）就是 **32768 个数**。乘上 60 层、乘上上下文长度、乘上并发请求数——KV cache 会迅速超过模型权重本身，成为决定"这张卡能同时服务多少个用户"的硬约束。

### 1.3 更要命的是带宽

decode 阶段每步只处理 1 个 token，却必须把**全部权重 + 整个 KV cache** 从显存读一遍。算术强度极低，算力单元大部分时间在等数据。

**所以 decode 阶段：省显存读取量 ≈ 省时间。** 这一条决定了后面所有设计的价值判断。

### 1.4 nanoGPT 的起点

```python
# model.py:52 —— 纯函数，无状态
def forward(self, x):
    q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
    ...

# model.py:312 —— 每步重跑整个前缀
for _ in range(max_new_tokens):
    idx_cond = idx if idx.size(1) <= block_size else idx[:, -block_size:]
    logits, _ = self(idx_cond)
```

**nanoGPT 没有 KV cache，一行都没有。** 这不是疏漏——它是训练仓库，`generate()` 只是个便利函数，为可读性放弃了推理效率。

**但这里有个关键的不对称，是整个案例的引子：**

| | 缺 KV cache 的后果 |
|---|---|
| 原版 nanoGPT（MHA） | **性能问题**。慢，但架构完整、正确 |
| 改成 MLA 之后 | **功能缺失**。MLA 存在的目的就是压缩 cache，没有 cache 就等于没做 |

---

## 2. 解法谱系：MHA → MQA → GQA → MLA

要理解 MLA，先看它在解法谱系里的位置。所有方法都在回答同一个问题：**能不能少存一点 K/V？**

| 方法 | 思路 | 每 token 每层缓存 | 代价 |
|---|---|---|---|
| **MHA** | 每个 head 独立的 K/V | `2·n_head·head_dim` | 基准 |
| **MQA** | 所有 head **共享一份** K/V | `2·head_dim` | 压缩率极高，但质量损失明显 |
| **GQA** | 分成 g 组，组内共享 | `2·g·head_dim` | MQA 和 MHA 之间的插值，Llama 2/3 用它 |
| **MLA** | 存**低秩 latent**，用时再升维 | `kv_lora_rank (+ rope 维)` | 压缩率高且质量损失小 |

**关键区别**：MQA/GQA 是在 **head 维度上做减法**——直接砍掉独立性，各 head 被迫看同样的 K/V。MLA 是在 **特征维度上做压缩**——每个 head 仍然有自己的 K 和 V（由 `W_UK[h]`、`W_UV[h]` 从共享 latent 升维得到），只是这些 K/V 被约束在一个低维子空间里。

> **一句话**：MQA/GQA 减少 head 的数量，MLA 减少每个 head 的信息熵。后者对质量更友好，因为 head 之间的差异性被保留了。

这个对比也解释了为什么 MLA 更复杂——**保留 head 独立性是要付出实现代价的**，后面的吸收和 decoupled RoPE 都是这个代价的一部分。

---

## 3. MLA 的三个核心思想

### 3.1 思想①：压缩——缓存 latent 而不是 K/V

数据流：

```
x  ──W_DKV──>  c_kv  ──W_UK──>  k
  (n_embd)      (r)   ──W_UV──>  v
```

三个事实：

1. `c_kv` 是 `r` 维（比如 128），`k`+`v` 合起来是 `2·n_head·head_dim` 维（比如 768）；
2. `W_UK`、`W_UV` 是**固定权重**，推理时不变，所以 k、v 是 `c_kv` 的**确定性函数**；
3. **存一个量和存它的函数值，信息量完全相同**，但占用差好几倍。

**结论：存上游那个更窄的，需要时再算下游。**

这不是一个实现技巧，**这是 MLA 这个架构的定义**。低秩瓶颈的全部意义就是人为制造一个信息瓶颈，让整条序列的历史能被压进一个窄向量。

> **反过来说**：如果你缓存 `k`/`v`，MLA 相对 MHA 就只剩"参数少一点"了——而参数少从来不是 MLA 的目的，它甚至是代价（低秩限制表达能力）。你会得到一个**纯亏**的架构。

latent 上还要加一层 **RMSNorm**（`kv_norm`）：`c_kv` 是整条序列历史的信息瓶颈，它的尺度必须稳定，否则梯度会在 `W_DKV`/`W_UK` 之间来回震荡。

### 3.2 思想②：吸收——为什么可以永不解压

注意力分数是 q·k。展开：

```
q_i · k_j = (W_UQ c_q_i)ᵀ (W_UK c_kv_j)
          = c_q_iᵀ (W_UQᵀ W_UK) c_kv_j
                    └──────┬──────┘
                    固定矩阵，可离线预乘
```

令 `W' = W_UQᵀ W_UK`，分数变成 `c_q_iᵀ W' c_kv_j`——**直接在 latent 空间里算完了，`k_j` 从头到尾没有被造出来过。**

value 侧同理：

```
y_i = Σ_j a_ij v_j = Σ_j a_ij (W_UV c_kv_j) = W_UV (Σ_j a_ij c_kv_j)
```

先在 latent 空间加权求和，最后统一升维一次。

**这才是 MLA 的完全体**：解码时每读一个历史 token，只从显存里读 `r` 个数、只在 `r` 维里算，永远不解压。省的不只是容量，更是**带宽**——而 1.3 节说过，decode 是彻头彻尾的带宽瓶颈场景。

代价要说清楚：吸收后 score 的点积维度从 `nope`（48）变成 `r`（128），**单次点积反而更贵**。它划算是因为省掉了 `O(T_kv · r · nope)` 的解压矩阵乘——序列越长越划算。

本仓库的实测：吸收在朴素 cache 之上再快 **1.22×**（894 ms → 730 ms）。

### 3.3 思想③：解耦 RoPE——调和①和②的冲突

**RoPE 会把吸收打死。** 位置 m 的 q 乘 `R_m`，位置 n 的 k 乘 `R_n`：

```
q_i · k_j = (R_m W_UQ c_q_i)ᵀ (R_n W_UK c_kv_j)
          = c_q_iᵀ W_UQᵀ (R_mᵀ R_n) W_UK c_kv_j
          = c_q_iᵀ W_UQᵀ  R_{n-m}   W_UK c_kv_j
                          └───┬───┘
                     依赖相对距离，卡在中间合并不掉
```

需要给每个可能的距离都存一个矩阵——比不吸收还贵。**吸收失效。**

**DeepSeek 的解法：让一部分维度负责位置，另一部分负责内容。**

| | 维度（V2） | 来源 | 能否吸收 |
|---|---|---|---|
| **nope 部分** | 128 | `W_UK · c_kv`（从 latent 升维） | ✅ |
| **rope 部分** | 64 | **`RoPE(W_KR · x)`，直接从 x 投影** | ❌，但很小 |

三个关键细节：

1. `k_R` **不从 `c_kv` 出**，而是由独立的 `W_KR` 从 `x` 投影；
2. `k_R` 是**所有 head 共享的单个 head**（只有 64 维一份，不是 `n_head` 份）；
3. `k_R` 和 `c_kv` **一起被缓存**。

最终分数 = `q_nope · k_nope`（吸收，latent 空间）+ `q_rope · k_rope`（64 维，共享，直接算）。

**这里是本案例最重要的一个判断信号：**

> 当论文里出现一个看起来"多余的复杂性"——"为什么不直接从 `c_kv` 里切一段做 rope？那多简单"——**那个复杂性通常就是整篇论文的关键**。凡是能简化掉的地方，作者早就简化了（论文页数是稀缺资源）。
>
> **看到一个可以简化的设计，先假设自己没理解，而不是假设作者啰嗦。**

第一版实现踩的正是这个坑：实现了"一部分维度旋转、一部分不旋转"的**表象**，但 `k_rope` 是从解压后的 `k` 里切的。结果是缓存 `c_kv` 仍然合法（省容量还在），但吸收彻底做不了（省带宽没了）。

### 3.4 附带的第四个设计：`v_head_dim` 独立

普通 MHA 里 q/k/v 每头维度相同，是因为它们都由 `n_embd / n_head` 切出来，**不是数学要求**。真正的约束只有：

- `dim(q) == dim(k)`（要做点积）；
- `dim(v)` **完全自由**——它只决定每个 head 输出多宽，`W_O` 把 `n_head × v_head_dim` 映回 `n_embd`。

DeepSeek 选 `qk = 128 nope + 64 rope = 192`，而 `v_head_dim = 128`。理由：那 64 维 rope 是**为携带位置信息付的开销**，只在打分时用得上；value 侧不参与打分，没理由跟着变宽，否则 `W_UV` 和 `W_O` 白白多付 50%。

---

## 4. 从公式到 shape：完整维度表

把上面三个思想落到具体张量。记 `B` batch、`T` 新 token 数、`S` 缓存长度、`H` = `n_head`、`E` = `n_embd`、`d` = `head_dim`、`r` = `kv_lora_rank`、`dr` = `qk_rope_head_dim`、`dn` = `d - dr`、`dv` = `v_head_dim`。

### 权重

| 张量 | shape | 说明 |
|---|---|---|
| `wq` / `wq_b` | `(H·d, E)` / `(H·d, q_lora_rank)` | q_lora_rank>0 时走低秩 |
| `wkv_a` (W_DKV) | `(r, E)` | 下投影，产出 latent |
| `kv_norm` | `(r,)` | latent 上的 RMSNorm |
| `wk_b` (W_UK) | **`(H·dn, r)`** | 只产 nope 部分！ |
| `wv_b` (W_UV) | `(H·dv, r)` | |
| `wkr` (W_KR) | **`(dr, E)`** | 单头，所有 head 共享 |
| `c_proj` (W_O) | `(E, H·dv)` | **不再是方阵** |

### 前向（decoupled + cache）

| 步骤 | shape |
|---|---|
| `x` | `(B, T, E)` |
| `q = wq(x)` → view/transpose | `(B, H, T, d)` |
| `q_nope, q_rope = q.split([dn, dr])` | `(B,H,T,dn)`, `(B,H,T,dr)` |
| `q_rope = RoPE(q_rope, freqs[start:start+T])` | 不变 |
| `c_kv = kv_norm(wkv_a(x))` | `(B, T, r)` **← 写入 cache** |
| `k_rope = RoPE(wkr(x), freqs[start:start+T])` | `(B, T, dr)` **← 写入 cache** |
| 读回 cache | `c_kv: (B,S,r)`, `k_rope: (B,S,dr)` |
| `k_nope = wk_b(c_kv)` | `(B, H, S, dn)` |
| `k = cat([k_nope, k_rope.expand(H)])` | `(B, H, S, d)` |
| `v = wv_b(c_kv)` | `(B, H, S, dv)` |
| `SDPA(q, k, v)` | `(B, H, T, dv)` |
| reshape + `c_proj` | `(B, T, E)` |

### 吸收路径（decode，`T=1`）

| 步骤 | shape | 备注 |
|---|---|---|
| `w_uk = wk_b.weight.view(H, dn, r)` | `(H, dn, r)` | 只是 view，零开销 |
| `q_latent = einsum('bhtn,hnr->bhtr', q_nope, w_uk)` | `(B,H,T,r)` | **W_UK 折进 Q** |
| `scores = einsum('bhtr,bsr->bhts', q_latent, c_kv)` | `(B,H,T,S)` | latent 空间打分 |
| `+ einsum('bhtd,bsd->bhts', q_rope, k_rope)` | 同上 | rope 项 |
| `/ sqrt(d)` | | **scale 用 q/k 维，不是 dv** |
| `o_latent = einsum('bhts,bsr->bhtr', att, c_kv)` | `(B,H,T,r)` | latent 空间聚合 |
| `y = einsum('bhtr,hvr->bhtv', o_latent, w_uv)` | `(B,H,T,dv)` | **最后才升维一次** |

**注意 `k` 和 `v` 在这条路径上从未出现。**

### 缓存量

```
MHA :  2 · H · d          （本仓库 6 层配置：2 × 6 × 64 = 768 / 层）
MLA :  r + dr             （128 + 16 = 144 / 层）
```

DeepSeek-V2 的真实比例：`576 vs 32768 ≈ 56.9×`。**head 数越多、head_dim 越大，MLA 收益越大**——这是它在大模型上才真正发力的原因。

---

## 5. 分四个阶段实施

顺序不是随意的。**每个阶段有独立的验收标准，且后一阶段依赖前一阶段建立的安全网。**

| 阶段 | 改什么 | 是否改变模型 | 是否需要重训 | 验收 |
|---|---|---|---|---|
| **A. 训练侧** | 低秩权重布局 | ✅ | ✅ | val loss 对照 |
| **B. 推理侧** | KV cache | ❌ 纯优化 | ❌ | 等价性断言 |
| **C. 正确的解耦** | W_KR / v_head_dim | ✅ | ✅ | val loss + 复用 B 的断言 |
| **D. 吸收** | 解码路径 | ❌ 纯优化 | ❌ | 复用 B 的断言 |

**为什么 B 排在 C 前面**：B 是纯推理侧改动，**可以用现有 checkpoint 立刻验证**，不需要重训。它产出的等价性测试会成为 C 和 D 的安全网。**先建安全网，再动结构。**

### 阶段 A：训练侧的权重布局

改 `__init__` 和 `forward` 的张量变换。要点：

```python
if self.kv_lora_rank > 0:
    self.wkv_a  = nn.Linear(n_embd, kv_lora_rank, bias=False)
    self.kv_norm = RMSNorm(kv_lora_rank)          # latent 上的 norm，不能省
    self.wk_b   = nn.Linear(kv_lora_rank, H * dn, bias=False)
    self.wv_b   = nn.Linear(kv_lora_rank, H * dv, bias=False)
else:
    self.wk = nn.Linear(n_embd, H * d, bias=config.bias)   # ← fallback 必须保留
    self.wv = nn.Linear(n_embd, H * dv, bias=config.bias)
```

**这一步最重要的工程决策是保留 fallback 开关**：`kv_lora_rank=0` 退回 MHA、`use_rope=False` 退回学习式 `wpe`。**没有开关就没有对照实验**，而没有对照实验，你的所有结论都是"感觉上"。

**验收**：和 baseline 跑同样超参、同样步数，比 val loss。本案例的结果：

| | baseline | MLA |
|---|---|---|
| 参数量 | 10.65 M | **9.76 M**（−8.4%） |
| best val | **1.4716** | 1.4757（+0.004） |
| final val (5000) | 1.7127 | **1.6636** |

结论：参数少 8.4%，质量持平，**且过拟合更轻**——低秩瓶颈在小数据集上起到了正则化作用。

**这一阶段的陷阱**：做完这一步很容易觉得"MLA 做完了"。训练能跑、参数下降、loss 持平——所有指标都好看。但**推理侧一行没写，MLA 的目的一点没实现**。第 9 节详谈。

### 阶段 B：KV cache

这是把**无状态函数变成有状态对象**。三件事：

**① 两个新参数**

```python
def forward(self, x, freqs_cis=None, start_pos=0, use_cache=False):
```

- `start_pos`：解码时模块只收到 `(B,1,C)` 的张量，**无法判断这是第 1 个还是第 500 个 token**。位置对 RoPE 必需、对 cache 写入位置必需，所以必须从外面告诉它。
- `use_cache`：训练/推理两条路的分岔口。**这个参数的存在本身就是提醒：一个模块可以有两种行为模式。**

**② 预分配 buffer 而不是 `torch.cat`**

```python
def _cache_update(self, name, value, start_pos):
    buf = getattr(self, name)
    B, T, D = value.shape
    assert start_pos + T <= self.block_size, "KV cache overflow"
    if buf is None or buf.size(0) < B or buf.dtype != value.dtype or buf.device != value.device:
        assert start_pos == 0, "cache was (re)allocated mid-sequence; call reset_cache() first"
        buf = torch.zeros(B, self.block_size, D, dtype=value.dtype, device=value.device)
        setattr(self, name, buf)
    buf[:B, start_pos:start_pos + T] = value      # 写：新的
    return buf[:B, :start_pos + T]                # 读：全部历史
```

`torch.cat` 每步分配新显存并拷贝整段历史：O(T²) 拷贝量 + 显存碎片。预分配 + 原地写是 O(1)。

**注意最后两行——cache 的全部本质就是"写新的，读全部"。**

**③ `generate()` 变成两阶段**

```python
self.reset_cache()      # 上次生成的残留会被当成历史读进来
pos, cur = 0, idx
for _ in range(max_new_tokens):
    logits, _ = self(cur, start_pos=pos, use_cache=True)
    pos += cur.size(1)   # 第一次是 prefill（整个 prompt），之后每次 1 个 token
    cur = self._sample(logits, temperature, top_k)
    idx = torch.cat((idx, cur), dim=1)
```

**验收**：等价性测试（第 7 节）。**用现有 checkpoint 就能跑，不需要重训。**

### 阶段 C：正确的 decoupled RoPE + `v_head_dim`

三件事一起做，因为互相依赖，拆开等于重训三次：

```python
self.split_rope = use_rope and config.qk_rope_head_dim > 0
self.qk_rope_head_dim = config.qk_rope_head_dim
self.qk_nope_head_dim = self.head_dim - self.qk_rope_head_dim
assert self.qk_nope_head_dim > 0, "否则没有可吸收的 nope 部分"

# 完整的 DeepSeek 布局只在有 latent 可解耦时才成立（即 MLA 模式）
self.decoupled_rope = self.split_rope and self.kv_lora_rank > 0
if self.decoupled_rope:
    self.wkr = nn.Linear(n_embd, self.qk_rope_head_dim, bias=False)   # 单头，共享
```

**注意 `split_rope` 和 `decoupled_rope` 是两个概念**：前者是"head 被切成 nope+rope"（MHA 模式下也成立，就是 partial rotary），后者是"`k_R` 有自己的投影并单独缓存"（只在 MLA 模式下有意义）。第一版把两者混为一谈，导致 MHA fallback 路径直接崩掉。

**`k_R` 要在写入 cache 前就旋转好**，读出来直接可用：

```python
k_rope = self.wkr(x).view(B, T, 1, dr).transpose(1, 2)
k_rope = apply_rotary_emb(k_rope, freqs_cis[start_pos:start_pos + T])   # 先转
k_rope = k_rope.transpose(1, 2).reshape(B, T, dr)
if use_cache:
    k_rope = self._cache_update('cache_k_rope', k_rope, start_pos)      # 再存
```

**`v_head_dim` 解耦的两个必踩点**：

1. 最后的 reshape 会炸——`view(B, T, C)` 隐含假设 `n_head*head_dim == n_embd`。改成 `view(B, T, self.n_head * self.v_head_dim)`。**好消息：这个坑会立刻报错。**
2. softmax scale 必须用 **q/k 维**，不是 v 维。走 SDPA 时默认 scale 取 `query.size(-1)`，**自动正确**；手写路径必须自己改。**这个坑不报错。**

**验收**：val loss 重新对照（改变了模型）+ 复用阶段 B 的等价性测试（确保 cache 逻辑没被改坏）。

### 阶段 D：吸收

只在**逐 token 解码**时启用：

```python
def _can_absorb(self, T, use_cache):
    return (self.absorb and use_cache and T == 1 and not self.training
            and self.decoupled_rope and self.kv_lora_rank > 0)
```

五个条件缺一不可。**训练时不需要吸收**——训练是并行的，materialize K/V 是正常的，SDPA/FlashAttention 更快。DeepSeek 官方推理代码里也是 naive / absorb 两条路并存。

核心实现（完整版见 [`model_mla.py`](../model_mla.py) 的 `_attn_absorbed`）：

```python
w_uk = self.wk_b.weight.view(nh, self.qk_nope_head_dim, r)
w_uv = self.wv_b.weight.view(nh, self.v_head_dim, r)

q_latent = torch.einsum('bhtn,hnr->bhtr', q_nope, w_uk)      # W_UK 折进 Q
scores   = torch.einsum('bhtr,bsr->bhts', q_latent, c_kv)    # latent 空间打分
scores   = scores + torch.einsum('bhtd,bsd->bhts', q_rope, k_rope)
scores   = scores / math.sqrt(self.head_dim)

att      = F.softmax(scores.float(), dim=-1).to(q.dtype)
o_latent = torch.einsum('bhts,bsr->bhtr', att, c_kv)         # latent 空间聚合
y        = torch.einsum('bhtr,hvr->bhtv', o_latent, w_uv)    # 最后升维一次
```

**验收**：`吸收路径 == 朴素路径`（复用阶段 B 的测试框架，换个断言）。

---

## 6. 五个静默陷阱

**全部不报错，全部只让生成质量变差**——而"生成得不好"在小模型上太容易被归因成"模型太小/训得不够"。这才是 KV cache 的真正难度所在：不是内存管理，是**状态一致性**。

### 陷阱 1：RoPE 位置切片

**根因**：RoPE 存的是**绝对位置**。`freqs_cis[i]` 是位置 i 的旋转量。"相对"是点积算出来的结果，不是存储方式。

无 cache 时 `freqs_cis[:T]` 是对的——`x` 永远从位置 0 开始，第 j 行天然是位置 j。加了 cache 后，解码时 `x` 只含 1 个 token，真实位置是 `start_pos`。写 `freqs_cis[:1]` 等于告诉模型"这个 token 在位置 0"。

**症状**：形状完全对得上，PyTorch 毫无怨言。prefill 部分是对的所以开头几个 token 还像样，越往后越退化成重复或语法崩坏。

**修法**：q 和 k 需要**不同的切片**：

```python
q_freqs = freqs_cis[start_pos : start_pos + T]   # 新 token 的真实位置
k_freqs = freqs_cis[: start_pos + T]             # 覆盖全部历史
```

### 陷阱 2：`is_causal` 的对齐语义

**根因**：SDPA 的 `is_causal=True` 生成的是**左上角对齐**的三角掩码——query 第 i 行只能看 key 第 0..i 列，**下标都从 0 数**。

训练时 `q_len == k_len`，左上角对齐 = 主对角线对齐，正确。解码时 `q_len=1, k_len=501`：**唯一那行 query 只能看 key[0]，整段 500 个 token 的历史被全部屏蔽。**

**修法**：

```python
if T_kv == T and start_pos == 0:  return None, True    # 训练 / prefill
if T == 1:                        return None, False   # 单 token 看全部历史
q_pos = torch.arange(start_pos, start_pos + T).unsqueeze(1)
k_pos = torch.arange(0, T_kv).unsqueeze(0)
return (q_pos >= k_pos), False                         # 分块 prefill
```

**别忘了手写路径**：`self.bias[:,:,:T,:T]` 有同样的左上角假设，要改成 `[:,:,start_pos:start_pos+T, :T_kv]`。在没有 FlashAttention 的环境（比如 mac CPU）走的正是这条路。

### 陷阱 3：cache 生命周期

**根因**：有了状态就有了残留。

- **每次新的 generate 必须 reset**，否则上次的历史被读进来；
- `sample.py` 里 `num_samples` 是循环调 generate 的——**第一个样本正常，后面的全被污染**。这个 bug 会让你以为"模型不稳定"。

**修法**：把 reset 放进 `generate()` 入口，不依赖调用方记得。

### 陷阱 4：忘记 `model.eval()`

**根因**：无 cache 时忘了 eval 只是随机性略大；**有 cache 后，带 dropout 噪声的 latent 会被存进 cache，污染整段后续生成**。后果被 cache 放大并持久化了。

即使 `wkv_a → kv_norm` 这一小段没有 dropout，`x` 也来自带 `resid_dropout` 的残差流。

### 陷阱 5：长度越界

**根因**：原代码靠 `idx[:, -block_size:]` 滑窗兜底超长序列。有 cache 后滑窗变复杂（要么不支持，要么实现 rolling cache 并重算所有 RoPE 位置）。

**修法**：**最小实现直接 assert 拒绝**，别装作支持：

```python
assert idx.size(1) + max_new_tokens <= self.config.block_size, (
    f"cached generation needs prompt+new <= block_size; pass use_cache=False for the sliding-window path")
```

**主动划定边界比做全更重要。**

---

## 7. 验证方法论：恒等变换与等价性测试

这一节是整个案例最可迁移的部分——它适用于任何模型优化，不只是 MLA。

### 7.1 什么是恒等变换

**恒等变换 = 改变计算过程，但不改变计算结果的重写。** `f'(x) = f(x)`，允许浮点误差，不允许语义差异。

`a*b + a*c → a*(b+c)` 是。记忆化递归是。数据库加索引是。**KV cache 就是记忆化**。

三条理由让它成为**定理而不是愿望**：

1. 被缓存的量是"过去"的确定性函数，已永久固定；
2. 它不改变任何可学习参数，cache 里只有中间激活值；
3. 注意力对历史的读取方式没变，只是这些数**从哪来**变了。

### 7.2 三类改动，必须分清

| 类别 | 例子 | 验收方式 |
|---|---|---|
| **① 改变模型** `f' ≠ f` | MHA→MLA、加 MoE、改 RoPE 布局、改 `v_head_dim` | **必须重训 + 比 val loss** |
| **② 恒等变换** `f' = f` | KV cache、吸收、FlashAttention、`torch.compile` | **等价性断言**，立刻可验 |
| **③ 近似** `f' ≈ f` | 量化、滑窗注意力、剪枝 | 断言 + 误差上界，再比下游 |

**这个分类直接决定你该怎么验证：**

- ① **必须重训**，不重训的结论都是假的；
- ② **绝不该重训**，重训只会掩盖 bug；
- ③ 先断言误差在预期范围内，再看下游。

> **把 ② 当成 ① 来验证，是最常见的浪费**：花两小时重训、看 loss 差不多、说"cache 应该没问题"。而 cache 的 bug 根本不影响训练 loss（训练不走 cache）——你验证的是一个和 bug 无关的量。

**反例值得记**：滑窗和量化**不是**恒等的，是近似；PagedAttention **是**恒等的；投机解码整体是恒等的（草稿必须被主模型逐 token 验证接受）——**用一个不精确的东西，通过验证机制，得到精确的结果**。

### 7.3 落地：五组断言

```python
1. 有 cache 的生成      == 无 cache 的生成          # 位置 / mask / 索引 / 模式
2. 吸收路径             == 朴素路径                 # W_UK、W_UV 折叠的数学
3. 分两块 prefill       == 一次 prefill             # 偏移 mask
4. 连续两次 generate    == 第二次单独跑             # cache 重置
5. 改动未来 token       不影响过去位置的 logits      # cache 破坏因果性
```

**两个必须的实现细节**：

- **必须用贪心（`temperature=0.0`）**。采样有随机性，两条路会因 RNG 状态不同而分叉，测试就没意义了。
- **必须 `model.eval()`**。否则 dropout 让两条路必然不同，你会得到一个永远失败的测试。

**必须覆盖所有配置分支**。本仓库测 7 种：legacy / decoupled / `v_head_dim≠head_dim` / q-LoRA / 无RoPE / MHA回退 / MHA无RoPE。7 × 5 = 38 项断言全过。少测一个分支，那个分支就是没验证的。

### 7.4 可迁移的规则

> **凡是声称"只是优化"的改动，都必须能写出一条等价性断言。写不出来，说明它其实改变了模型，你只是还没意识到。**

这条规则的威力在于它**把模糊问题变成确定性问题**：把"只能靠肉眼看生成质量"变成"过 / 不过"。五个静默错误，全部收敛成一个布尔值。

---

## 8. 测量：把论文数字变成你自己的数字

论文说 93% 削减，你的模型上是多少？**必须自己测**，因为比例强依赖 `n_head` 和 `head_dim`。

**测量的第一条纪律：每次测速前先断言正确性**，否则你测的是一个坏实现的速度。

```python
a = model.generate(prompt, N, temperature=0.0, use_cache=False)
b = model.generate(prompt, N, temperature=0.0, use_cache=True)
assert torch.equal(a, b), "KV cache changed the output; it must be an identity transform"
```

### 本案例的实测结果

**已训练 checkpoint（legacy 布局，6 层，`n_embd=384`，`kv_lora_rank=128`，CPU）**：

```
cached == uncached : YES

KV cache, 256 tokens, batch 1, float32:
  MLA (cached c_kv)            768 numbers/token       768.0 KiB
  MHA equivalent (K and V)    4608 numbers/token      4608.0 KiB
  reduction                   6.00x

decode 200 tokens from a 32-token prompt:
  no cache            1296.0 ms   (6.48 ms/token)   1.00x
  cache, naive         333.2 ms   (1.67 ms/token)   3.89x
```

**decoupled 布局（`qk_rope_head_dim=16`）**：

```
  MLA (cached c_kv + k_R)      864 numbers/token      5.33x   ← 多缓存 16 维换来吸收
decode 400 tokens:
  no cache            4497.7 ms   1.00x
  cache, naive         894.1 ms   5.03x
  cache, absorbed      730.0 ms   6.16x              ← 吸收再快 1.22x
```

### 怎么读这些数字

1. **cache 是 MLA 收益的全部载体**。没有 cache 时，MLA 相对 MHA 的推理优势严格为 0。
2. **decoupled 用 12% 的额外 cache 换来了吸收路径**（768 → 864），净赚。
3. **无 cache → 有 cache 的加速会随生成长度增长**（O(T³) vs O(T²)），所以 200 token 测出 3.89×、400 token 测出 5.03×。报数字时必须说明生成长度。
4. **CPU 的带宽/算力比和 GPU 差异很大，加速比不可直接外推**。诚实标注测试环境。

### 和 DeepSeek-V2 的比例对照

```
MLA : kv_lora_rank(512) + qk_rope_head_dim(64)  =   576
MHA : 2 × n_head(128) × head_dim(128)           = 32768
                                                  ≈ 56.9x
```

**你的小模型只有 6×，V2 有 57×。差异全部来自 `n_head` 和 `head_dim`。** 理解这一点，才不会因为"我这里只有 6 倍"而怀疑 MLA 的价值。

---

## 9. 复盘：三个认知缺口

第一版实现只做对了一半，但**从表面指标看完全正常**：训练能跑、参数少 8.4%、val loss 持平、过拟合更轻。所有能看到的数字都是好的。

而实际上 MLA 存在的理由一点没兑现。这个落差值得拆开看。

### 缺口 1：把"架构"理解成"层的连接方式"

第一版的所有改动都落在两个地方：`__init__` 里的层定义、`forward` 里的张量变换。**没有一处涉及状态、推理路径、或训练/推理的行为差异。**

但完整的架构还包括：

- 训练路径和推理路径的**行为差异**
- **状态管理**
- **数值精度的处理**

**MLA 恰好是一个收益 100% 落在推理路径上的架构**，所以这个缺口在它身上被放大到极致——你会得到一个"训练时完全正确、但没有在做它该做的事"的 MLA。

同一个缺口在其他架构上会换个形式出现：MoE 是训练时的路由行为 vs eval 时的路由行为；MTP 是训练时用、推理时丢弃。**这些架构都要求你同时持有两条路径的心智模型。**

### 缺口 2："读懂论文"和"兑现论文"之间的距离

第一版的注释写着：

```python
c_kv = self.kv_norm(self.wkv_a(x))  # (B, T, kv_lora_rank) -- the only thing to cache
```

**这句话是对的**——论文这部分读懂了。但它被当成了一句**描述**，而它其实是一个**需要代码兑现的承诺**。

decoupled RoPE 上是同一个模式：实现了"一部分维度旋转、一部分不旋转"的表象，但没有追问"论文为什么让 `k_R` 从 `x` 单独投影，而不是从 `c_kv` 里切一段？后者明明简单得多。"

那个"为什么"里藏着吸收，也就是 MLA 的另一半收益。

**可直接用的判断信号**（第 3.3 节已经给过，值得重复）：

> 当论文里出现一个看起来"多余的复杂性"，那个复杂性通常就是关键。凡是能简化掉的地方，作者早就简化了。**看到可以简化的设计，先假设自己没理解，而不是假设作者啰嗦。**

### 缺口 3：没有建立"等价性测试"这个工具（这条是杠杆）

第一版整个改动过程没有一行断言。验证正确性的唯一手段是"跑一下看 loss 像不像样"。

**这个手段对静默错误完全无效**——第 6 节那五个坑全都是静默的。它们不报错，只是让结果变差，然后被归因成"模型太小"。

**为什么说这条是杠杆**：写等价性测试会逼你回答"这两条路径为什么应该相同"。而回答那个问题的过程，本身就在补缺口 1 和 2——**你必须把两条路径的行为差异想清楚，才写得出这个断言。**

所以正确的做法是**先写测试、再写 cache**，而不是反过来。

### 反过来，第一版做对了什么

复盘不能只挑毛病。这些是真的做对了，而且不是偶然：

1. **低秩权重布局正确**，包括 latent 上 RMSNorm 的位置；
2. **保留了 fallback 开关**（`kv_lora_rank=0`、`use_rope=False`）——很多人改架构会直接删掉老路径，**没有开关就没有对照实验**；
3. **主动删掉 `from_pretrained` 并写清理由**，而不是留一个会静默出错的函数；
4. resume 时用 `if k in checkpoint_model_args` 保护 MLA 专有 key；
5. `crop_block_size` 里同步裁了 `freqs_cis`——极容易漏；
6. `freqs_cis` 用 `persistent=False`——知道哪些东西不该进 checkpoint；
7. **真的跑了对照实验并留了日志**。

**所以问题不在于粗心**——工程习惯是细的。缺口是结构性的：认知里没有"推理路径"这个对象，改的时候压根没往那儿看。

---

## 10. 复现步骤

```bash
# 1. 数据
python data/shakespeare_char/prepare.py

# 2. baseline（MHA + 学习式 wpe）
python train.py config/train_shakespeare_char.py --device=cpu --compile=False 2>&1 | tee baseline_train.log

# 3. MLA
python train_mla.py config/train_shakespeare_char_mla.py --device=cpu --compile=False 2>&1 | tee mla_train.log

# 4. 正确性：7 配置 × 5 断言
python test_mla.py

# 5. 实测 cache 与解码速度（先断言 cached == uncached）
python bench_kv_cache.py --out_dir=out-shakespeare-char-mla --device=cpu --max_new_tokens=200

# 6. 独立的最小参考实现（160 行，可脱离仓库阅读）
python mla_min.py

# 7. 采样
python sample_mla.py --out_dir=out-shakespeare-char-mla --device=cpu --num_samples=2
```

**关键配置项**（[`config/train_shakespeare_char_mla.py`](../config/train_shakespeare_char_mla.py)）：

```python
kv_lora_rank = 128    # 0 = 退回 MHA
q_lora_rank = 0       # Q 侧压缩，小模型可以不用
qk_rope_head_dim = 16 # >0 = DeepSeek decoupled RoPE（可吸收）；0 = legacy 整头旋转
v_head_dim = 0        # 0 = 同 q/k head dim
use_rope = True       # False = 退回学习式 wpe
```

---

## 11. 延伸阅读

**必读**

- **DeepSeek-V2 技术报告**，arXiv:2405.04434 §2.1 —— MLA 的一手来源，包括 decoupled RoPE 的完整推导。
- **DeepSeek-V3 技术报告**，arXiv:2412.19437 —— MLA 在更大规模上的配置，以及 MoE / MTP 的设计。

**理解解法谱系**

- **MQA**：Shazeer, "Fast Transformer Decoding: One Write-Head is All You Need", arXiv:1911.02150
- **GQA**：Ainslie et al., arXiv:2305.13245 —— 读它是为了理解"在 head 维度做减法"的极限在哪，从而理解 MLA 为什么换了个方向。

**推理侧**

- **FlashAttention**，arXiv:2205.14135 —— 另一个经典的恒等变换，和 KV cache 正交、可叠加。
- **PagedAttention / vLLM**，arXiv:2309.06180 —— KV cache 的内存管理怎么做到工业级。

**本仓库**

| 文档 | 内容 |
|---|---|
| [mla_deep_dive.md](mla_deep_dive.md) | 概念梳理：KV cache 本质、MHA 谱系、FlashAttention、RoPE 为什么两维一组 |
| [inference_path.md](inference_path.md) | 推理路径的完整定义、成熟系统的五层分工、从 0 做模型的清单 |
| [experiments.md](experiments.md) | 实验记录与实测数据 |
| [moe_pitfalls.md](moe_pitfalls.md) | 下一步：MoE 的工程坑 |

---

## 附：一页速查

| 问题 | 答案 |
|---|---|
| MLA 缓存什么？ | `c_kv`（+ decoupled 时的 `k_R`），**不是** K/V |
| 为什么能只存 `c_kv`？ | K/V 是它的确定性函数，信息量相同、占用更小 |
| 什么是吸收？ | `W_UK` 折进 Q、`W_UV` 折到输出，全程在 latent 空间算，K/V 从不出现 |
| RoPE 为什么破坏吸收？ | `R_{n-m}` 卡在 `W_UQᵀ` 和 `W_UK` 之间，依赖相对距离，合并不掉 |
| 怎么绕过？ | 拆成 nope（可吸收）+ rope（`W_KR` 单独投影、共享单头、单独缓存） |
| 训练时需要吸收吗？ | **不需要**。训练是并行的，SDPA 更快。吸收只在逐 token 解码有意义 |
| KV cache 是什么类别的改动？ | **恒等变换**。必须写等价性断言，不能靠重训验证 |
| 五个静默坑？ | 位置切片、`is_causal` 对齐、cache 未重置、忘 `eval()`、长度越界 |
| 等价性测试的两个前提？ | 贪心解码（`temperature=0.0`）+ `model.eval()` |
| 实施顺序？ | 训练侧 → **KV cache（先建安全网）** → 正确的解耦 → 吸收 |

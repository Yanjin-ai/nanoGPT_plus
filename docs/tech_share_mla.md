# 在 nanoGPT 上把 MHA 换成 MLA：核心理念、改动清单，和那些不报错的坑

> 这是一篇技术分享帖，记录一次真实的改造：把 Karpathy 的 nanoGPT（标准 MHA）
> 换成 DeepSeek-V2 的 **Multi-head Latent Attention（MLA）**。
>
> 前半部分讲清楚 MLA 到底在解决什么问题、它的核心理念是什么；
> 后半部分是一份「坑地图」——MLA 的坑几乎都有一个共同点：**不报错，只会让生成变差**。
>
> 配套代码：[`model_mla.py`](../model_mla.py) · [`mla_min.py`](../mla_min.py) · [`test_mla.py`](../test_mla.py) · [`bench_kv_cache.py`](../bench_kv_cache.py)

---

## 目录

- [一、先说清楚：MLA 在解决什么问题](#一先说清楚mla-在解决什么问题)
- [二、MLA 的核心理念（三个思想）](#二mla-的核心理念三个思想)
- [三、从 MHA 到 MLA：具体改了哪些东西](#三从-mha-到-mla具体改了哪些东西)
- [四、坑地图：五个静默陷阱](#四坑地图五个静默陷阱)
- [五、最隐蔽的一个坑：RoPE 会杀死 MLA 最好的部分](#五最隐蔽的一个坑rope-会杀死-mla-最好的部分)
- [六、v_head_dim 解耦的两个坑（一个报错，一个不报）](#六v_head_dim-解耦的两个坑一个报错一个不报)
- [七、怎么证明自己没写错：等价性测试](#七怎么证明自己没写错等价性测试)
- [八、实测数字](#八实测数字)
- [九、复盘：这次改造教会我的三件事](#九复盘这次改造教会我的三件事)
- [参考](#参考)

---

## 一、先说清楚：MLA 在解决什么问题

### 1.1 自回归解码在重复算历史

大模型生成是**一个 token 一个 token 吐**的。要生成第 `t` 个 token，就得让它的 query 去和
第 `0..t-1` 个 token 的 key/value 算注意力。而历史 token 的 K/V **只由已经确定的输入决定**，
不会因为后面多了一个 token 而改变。

所以如果不缓存，每一步都要把整个前缀重新跑一遍：

| | 每步计算量 | 生成 T 个 token 总量 |
|---|---|---|
| 无 cache | O(t²) | **O(T³)** |
| 有 cache | O(t) | **O(T²)** |

缓存 K/V 就是拿一小块显存，换掉这个重复劳动——**KV cache 本质上是记忆化（memoization）**。

### 1.2 KV cache 成了推理的主要开销

标准 MHA 每 token 每层要缓存：

```
K 一份 + V 一份 = 2 × n_head × head_dim
```

放到 DeepSeek-V2 的规模（`n_head=128, head_dim=128`），就是 **32768 个数 / token / 层**。
再乘上几十层、乘上几万 token 的上下文、乘上并发请求数——KV cache 会迅速超过模型权重本身，
成为「这张卡能同时服务多少个用户」的硬约束。

### 1.3 更要命的是：decode 是带宽瓶颈

decode 阶段每步只处理 1 个新 token，却必须把**全部权重 + 整个 KV cache** 从显存读一遍。
算术强度极低，算力单元大部分时间在等数据。

> **decode 阶段：省显存读取量 ≈ 省时间。**

这句话是整个 MLA 设计的价值锚点。后面你会看到，MLA 的两个核心动作——压缩和吸收——
一个省容量，一个省带宽，都是冲着这一条去的。

---

## 二、MLA 的核心理念（三个思想）

### 2.1 先看它在解法谱系里的位置

所有方法都在回答同一个问题：**能不能少存一点 K/V？**

| 方法 | 思路 | 每 token 每层缓存 | 代价 |
|---|---|---|---|
| **MHA** | 每个 head 独立的 K/V | `2·n_head·head_dim` | 基准 |
| **MQA** | 所有 head 共享一份 K/V | `2·head_dim` | 压缩率高，质量损失明显 |
| **GQA** | 分成 g 组，组内共享 | `2·g·head_dim` | MHA 与 MQA 的折中（Llama 用） |
| **MLA** | 存**低秩 latent**，用时再升维 | `kv_lora_rank (+ rope 维)` | 压缩率高且质量损失小 |

关键区别一句话：

> **MQA/GQA 在 head 维度上做减法（砍 head 独立性）；MLA 在特征维度上做压缩（每个 head 的 K/V 被约束进低维子空间）。**

保留 head 独立性是 MLA 质量更好的原因，也是它实现更复杂的原因——后面所有的「坑」都源于这个取舍。

### 2.2 思想①：压缩——缓存 latent，而不是 K/V

MLA 的数据流是这样的：

```
x ──W_DKV──> c_kv ──W_UK──> k
  (n_embd)    (r)   ──W_UV──> v
```

- `c_kv` 是 `r` 维（低秩 rank，比如 128）；
- 完整的 K + V 是 `2·n_head·head_dim` 维（比如 768）；
- `W_UK`、`W_UV` 是**固定权重**，所以 K、V 是 `c_kv` 的**确定性函数**。

存一个量和存它的函数值，信息量完全相同，但占用差好几倍。所以：

> **存上游那个窄的 `c_kv`，需要时再算下游的 K/V。这就是 MLA 的定义，不是实现技巧。**

两个关键点：

1. `c_kv` 上要加一层 **RMSNorm**（`kv_norm`）。它是整条序列历史的信息瓶颈，尺度必须稳定，
   否则梯度会在下投影和上投影之间来回震荡。
2. **反过来想**：如果你缓存的是解压后的 K/V，那 MLA 相对 MHA 就只剩「参数少一点」了——而参数少
   从来不是 MLA 的目的，低秩反而是它付出的代价。你会得到一个**纯亏**的架构。

### 2.3 思想②：吸收——为什么可以永不解压

注意力分数是 `q·k`。把低秩结构代进去：

```
q_i · k_j = (W_UQ c_q_i)ᵀ (W_UK c_kv_j)
          = c_q_iᵀ (W_UQᵀ W_UK) c_kv_j
                    └──────┬──────┘
                    固定矩阵，可离线预乘
```

令 `W' = W_UQᵀ W_UK`，分数变成 `c_q_iᵀ W' c_kv_j`——**直接在 latent 空间算完了，K 从头到尾没被造出来。**

value 侧同理：

```
y_i = Σ_j a_ij v_j = Σ_j a_ij (W_UV c_kv_j) = W_UV (Σ_j a_ij c_kv_j)
```

先在 latent 空间加权求和，最后统一升维一次。

这就是**吸收（absorption）**：把 `W_UK` 折进 Q、把 `W_UV` 折到输出之后。解码时每个历史 token
只读 `r` 个数、只在 `r` 维里算，永远不解压。**省的不只是容量，更是带宽**——正是 1.3 节说的关键。

代价要讲清楚：吸收后单次点积的维度从 `nope`（比如 48）变成 `r`（128），**单次点积反而更贵**。
它划算是因为省掉了 `O(T_kv·r·nope)` 的解压矩阵乘——序列越长越划算。

### 2.4 思想③：解耦 RoPE——调和①和②的冲突

**RoPE 会把吸收打死。** 位置 m 的 q 乘 `R_m`，位置 n 的 k 乘 `R_n`：

```
q_i · k_j = c_q_iᵀ W_UQᵀ (R_mᵀ R_n) W_UK c_kv_j
          = c_q_iᵀ W_UQᵀ  R_{n-m}   W_UK c_kv_j
                          └───┬───┘
                     依赖相对距离，卡在中间合并不掉
```

中间那个旋转矩阵依赖相对距离，不能预计算——吸收失效。

**DeepSeek 的解法：让一部分维度负责位置，另一部分负责内容。**

| | 维度（V2） | 来源 | 能否吸收 |
|---|---|---|---|
| **nope 部分** | 128 | `W_UK · c_kv`（从 latent 升维） | ✅ |
| **rope 部分** | 64 | `RoPE(W_KR · x)`，直接从 x 投影 | ❌，但很小 |

三个关键细节：

1. `k_R` **不从 `c_kv` 出**，而是由独立的 `W_KR` 从 `x` 投影；
2. `k_R` 是**所有 head 共享的单个 head**（一份 64 维，不是 n_head 份）；
3. `k_R` 和 `c_kv` **一起被缓存**。

最终分数 = `q_nope·k_nope`（吸收，latent 空间）+ `q_rope·k_rope`（很小，直接算）。

> **判断信号**：当论文里出现一个看起来「多余的复杂性」——「为什么不直接从 `c_kv` 里切一段做 rope？那多简单」——
> **那个复杂性通常就是整篇论文的关键。** 作者能简化早就简化了。看到一个可简化的设计，先假设自己没理解，而不是假设作者啰嗦。

### 2.5 缓存数学对比

```
MHA :  2 × n_head × head_dim          = 2 × 128 × 128 = 32768 个数 / token / 层
MLA :  kv_lora_rank + qk_rope_head_dim =     512 + 64 =   576 个数 / token / 层
                                                  ≈ 56.9× 更小
```

head 数越多、head_dim 越大，MLA 的相对收益越大——这是它在大模型上才真正发力的原因。

---

## 三、从 MHA 到 MLA：具体改了哪些东西

以 `model.py`（vanilla nanoGPT）为对照，`model_mla.py` 动了这些地方：

### 3.1 权重布局：从融合投影到低秩三件套

MHA 版（融合 `c_attn`）→ MLA 版拆成：

```python
if self.kv_lora_rank > 0:
    self.wkv_a  = nn.Linear(n_embd, kv_lora_rank, bias=False)      # W_DKV：下投影
    self.kv_norm = RMSNorm(kv_lora_rank)                            # latent 上的 norm
    self.wk_b   = nn.Linear(kv_lora_rank, H * dn, bias=False)       # W_UK：升维到 nope 部分
    self.wv_b   = nn.Linear(kv_lora_rank, H * dv, bias=False)       # W_UV：升维到 value
    if self.decoupled_rope:
        self.wkr = nn.Linear(n_embd, qk_rope_head_dim, bias=False)  # W_KR：单头，共享
else:
    self.wk = nn.Linear(n_embd, H * d, bias=config.bias)            # ← fallback 必须保留
    self.wv = nn.Linear(n_embd, H * dv, bias=config.bias)
```

Q 侧也可选低秩（`q_lora_rank > 0`），走 `wq_a + q_norm + wq_b`。

### 3.2 位置编码：学习式 wpe → RoPE

- `use_rope=False` 时保留学习式 `wpe`（原 nanoGPT 路径）；
- `use_rope=True` 时换成旋转位置编码，并支持 DeepSeek 的**解耦 RoPE**（`qk_rope_head_dim > 0`）。

### 3.3 v_head_dim 独立

普通 MHA 里 q/k/v 每头维度相同，是「由 `n_embd / n_head` 切出来」的**结果**，不是数学要求。
真正的约束只有 `dim(q) == dim(k)`；`dim(v)` 完全自由。DeepSeek 选
`qk = 128 nope + 64 rope = 192`、`v_head_dim = 128`——那 64 维 rope 只用于打分，value 侧没必要跟着变宽。

### 3.4 forward 签名：从纯函数到有状态

```python
# 原来：def forward(self, x):
# 现在：def forward(self, x, freqs_cis=None, start_pos=0, use_cache=False):
```

- `start_pos`：解码时模块只收到 `(B,1,C)`，无法判断这是第 1 个还是第 500 个 token；
- `use_cache`：训练 / 推理两条路的分岔口。

### 3.5 generate：变成 prefill + decode 两阶段

```python
self.reset_cache()      # 上次生成的残留会被当成历史读进来
pos, cur = 0, idx
for _ in range(max_new_tokens):
    logits, _ = self(cur, start_pos=pos, use_cache=True)
    pos += cur.size(1)  # 第一次是 prefill（整个 prompt），之后每次 1 个 token
    cur = self._sample(logits, temperature, top_k)
    idx = torch.cat((idx, cur), dim=1)
```

---

## 四、坑地图：五个静默陷阱

这四个（外加一个）坑有一个共同点，也正是 MLA 改造最危险的地方：**没有一个会抛异常**。
它们只会让生成质量变差，而「生成得不好」在小模型上太容易被归因成「模型太小 / 训得不够」。

### 坑 1：RoPE 位置错位（静默）

RoPE 存的是**绝对位置**，靠点积产生相对行为。没有 cache 时 `freqs[:T]` 恰好对，因为输入总从 0 开始。

有了 cache，你喂进来的那个 token 真实位置可能是 500。如果你还写 `freqs[:1]`，就等于告诉模型它在位置 0。

**表现**：生成开头正常（prefill 是对的），然后漂移、开始重复。

**修法**——注意 query 和 key 需要**不同的切片**：

```python
q_freqs = freqs[start_pos : start_pos + T]   # 新 token 的真实位置
k_freqs = freqs[: start_pos + T]             # 整段历史
```

### 坑 2：因果 mask（静默，且最阴险）

PyTorch 的 `scaled_dot_product_attention(..., is_causal=True)` 会把三角 mask 对齐到**左上角**：
query 第 i 行只能看 key 第 0..i 列。

- 训练时 `q_len == k_len`，左上角对齐 = 对角线对齐，**正确**；
- decode 时 `q_len=1`、`k_len=501`，你的单个 query 只能看到 key 第 0 列——**整段历史被静默屏蔽**。

**修法**：别把 `is_causal` 当常数：

```python
if T_kv == T and start_pos == 0:  return None, True    # 训练 / prefill
if T == 1:                        return None, False   # decode：看全部历史
# 分块 prefill 需要显式偏移 mask
q_pos = torch.arange(start_pos, start_pos + T).unsqueeze(1)
k_pos = torch.arange(0, T_kv).unsqueeze(0)
return (q_pos >= k_pos), False
```

别忘手写 attention 的 fallback 分支——`self.bias[:, :, :T, :T]` 带着同样的假设，而 CPU 上你跑的就是这条分支。

### 坑 3：两次生成之间 cache 残留（静默）

状态意味着残留。如果采样脚本循环画 N 条样本，第 1 条是干净的，之后每条都会继承上一条的历史。

**修法**：在 `generate()` 入口处 `reset_cache()`，不要指望调用方记得。

### 坑 4：忘了 `model.eval()`（静默，且被放大）

没有 cache 时，留着 dropout 顶多让生成带点噪声。有了 cache，**被 dropout 污染的 latent 会被写进 cache**，
毒害后面每一个 token。cache 不只暴露了这个 bug，还把它**持久化**了。

### 坑 5：跑过 block_size（这个是「响」的，故意让它响）

原版 nanoGPT 用滑动窗口 `idx[:, -block_size:]` 处理超长。带 cache 后这件事真变复杂了（滚动 buffer + 重算位置）。

**建议的修法是：拒绝。**

```python
assert idx.size(1) + max_new_tokens <= self.config.block_size, \
    "cached generation needs prompt+new <= block_size; use_cache=False for the sliding-window path"
```

主动划边界，比假装支持一个你没实现的东西强。这是五个坑里唯一一个我**故意让它报错**的。

---

## 五、最隐蔽的一个坑：RoPE 会杀死 MLA 最好的部分

这个坑值得单独一节，因为它是「读错论文」而不是「读错 API」，而且它背后的设计很漂亮。

MLA 真正的超能力不只是 cache 小，而是**你根本不用解压**。但 2.4 节算过：RoPE 的旋转矩阵卡在
`W_UQᵀ` 和 `W_UK` 中间，依赖相对距离，合并不掉，吸收失效。

DeepSeek 的答案是**切头**：大部分维度不旋转、可吸收；一小部分携带位置，并且**用独立的 `W_KR` 从 x 投影**、
作为单头共享、和 latent 一起缓存。

我第一次实现只做了「一部分维度旋转、一部分不旋转」的**表象**，但 rope 部分是从解压后的 K 里切的。
结果：缓存 `c_kv` 仍然合法（省容量还在），但**吸收彻底做不了**（省带宽没了）。我抄了答案的形状，没抄到机制。

更糟的是，我当时的默认配置把**整个 head 都旋转了**，nope 部分 = 0，吸收维度 = 0——我实现的解耦 RoPE，解耦度设成了零。

> **教训**：当论文里有一个看起来可以简化的步骤，那个步骤往往就是全部要点。

---

## 六、v_head_dim 解耦的两个坑（一个报错，一个不报）

让 `v_head_dim != head_dim` 时有两个必踩点：

1. **最后的 reshape 会炸**——`view(B, T, C)` 隐含假设 `n_head*head_dim == n_embd`。
   要改成 `view(B, T, self.n_head * self.v_head_dim)`。
   **好消息：这个坑会立刻报错。**

2. **softmax scale 必须用 q/k 维，不是 v 维。** 走 SDPA 时默认 scale 取 `query.size(-1)`，
   **自动正确**；手写路径必须自己写 `1/sqrt(head_dim)`。
   **这个坑不报错。**

---

## 七、怎么证明自己没写错：等价性测试

上面的坑 99% 是**静默**的。如果唯一验证手段是「跑一下看看像不像样」，你一个都抓不到。
而「跑一下看看」正是最容易犯的错。

这里引入一把尺子，把改动分三类：

| 类别 | 定义 | 例子 | 验收方式 |
|---|---|---|---|
| **① 改变模型** | `f' ≠ f` | MHA→MLA、改 RoPE 布局、改 v_head_dim | 重训 + 比 val loss |
| **② 恒等变换** | `f' = f`，只是更快/更省 | KV cache、吸收、FlashAttention | **等价性断言** |
| **③ 近似** | `f' ≈ f` | 量化、滑动窗口、剪枝 | 断言 + 误差上界 |

**KV cache 和吸收都是类别②**——它们是记忆化，数学上必须和慢路径逐位相同。所以正确的验证方式不是重训，而是断言：

```python
a = model.generate(prompt, N, temperature=0.0, use_cache=False)  # 慢参考
b = model.generate(prompt, N, temperature=0.0, use_cache=True)   # 快路径
assert torch.equal(a, b)
```

两个不可妥协的细节：

- **必须贪心（`temperature=0.0`）**：采样有随机性，两条路会因 RNG 状态不同而分叉，测试就没意义。
- **必须 `model.eval()`**：否则 dropout 让两条路必然不同，你会得到一个永远失败的测试。

`test_mla.py` 里落地了五组断言，覆盖 7 种配置：

```
1. 有 cache 的生成 == 无 cache 的生成          (贪心，逐 token 相等)
2. 吸收路径      == 朴素路径                    (W_UK 折进 Q，W_UV 折到输出后)
3. 分两块 prefill == 一次 prefill               (验证偏移 mask)
4. 连续两次 generate == 第二次单独跑             (验证 cache 重置)
5. 改动未来 token 不影响过去位置的 logits        (因果性没被 cache 破坏)
```

这五条断言，把坑 1~5 全部收敛成一个布尔值。位置错了？失败。mask 错了？失败。cache 残留？失败。dropout 没关？立刻失败。

> **规则**：凡是声称「只是优化」的改动，都必须能写出一条等价性断言。写不出来，说明它其实改变了模型，你只是还没意识到。

---

## 八、实测数字

小模型、CPU，只比同硬件下的相对差异。

**训练质量**（`shakespeare_char`，6 层，`kv_lora_rank=128`，5000 步）：

| | baseline (MHA) | MLA |
|---|---|---|
| 参数量 | 10.65 M | **9.76 M**（−8.4%） |
| best val loss | **1.4716** | 1.4757（+0.004） |
| final val loss | 1.7127 | **1.6636** |

参数更少、质量持平、过拟合更轻——低秩瓶颈在小数据集上起到了正则化作用。

**推理成本**（先断言 `cached == uncached` 再测）：

```
KV cache:       768 → 864 numbers/token        (5.33× 更小)
decode 400 tok: 4498 ms → 894 ms (naive) → 730 ms (absorbed)
                                  5.03×            6.16×
```

三点读法：

1. **cache 是 MLA 收益的全部载体**——没有 cache，MLA 相对 MHA 的推理优势严格为 0，甚至略慢。
2. **解耦用 12% 的额外 cache 换来了吸收路径**，吸收在朴素 cache 上再快 1.22×。
3. **本仓库的 6× 不是 DeepSeek 的 57×**。比例是配置的函数，不是技术的函数。自己模型上自己测。

---

## 九、复盘：这次改造教会我的三件事

1. **从一开始就写 `forward(x, start_pos, use_cache)`**，哪怕两个参数几个月没人用。事后给无状态函数
   补状态，要改遍 `Block` / `GPT` / `generate` / 采样脚本整条调用链。
2. **先写等价性测试，再写它要验证的优化**。写断言的过程会逼你说清「两条路为什么该一样」，
   而正是在这里你会发现你其实还没搞懂状态机。
3. **保留 fallback 开关**：`kv_lora_rank=0` → 退回 MHA，`use_rope=False` → 退回学习式 wpe。
   **删掉旧路径，就等于删掉了你的对照组。**

最后是一把总尺子：

> 每加一个东西，先问：**它改变模型，还是只改变计算方式？**
>
> 改变模型 → 重训，并提前想清楚对齐哪根轴。
> 只改变计算方式 → 断言它和慢路径相等，永远不要「跑一下看看」。

---

## 参考

- Liu et al., *DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model* — [arXiv:2405.04434](https://arxiv.org/abs/2405.04434)。§2.1 就是 MLA，含 decoupled RoPE 的推导。
- DeepSeek-AI, *DeepSeek-V3 Technical Report* — [arXiv:2412.19437](https://arxiv.org/abs/2412.19437)
- Shazeer, *Fast Transformer Decoding: One Write-Head is All You Need*（MQA）— [arXiv:1911.02150](https://arxiv.org/abs/1911.02150)
- Ainslie et al., *GQA: Training Generalized Multi-Query Transformer Models* — [arXiv:2305.13245](https://arxiv.org/abs/2305.13245)
- Dao et al., *FlashAttention* — [arXiv:2205.14135](https://arxiv.org/abs/2205.14135)
- Karpathy, *nanoGPT* — [github.com/karpathy/nanoGPT](https://github.com/karpathy/nanoGPT)

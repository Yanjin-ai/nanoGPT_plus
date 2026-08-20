# MLA 深度解析（nanoGPT 改造版）

> 本文档配套仓库里的 `model_mla.py`（Multi-head Latent Attention 改造），
> 回答"为什么这么改"背后的所有基础概念：KV cache、MHA 起源、head 的本质、
> 各种注意力变体、RoPE、维度关系、对外接口、weight decay 等。
>
> 阅读顺序建议按目录从上到下，每节都在为后面做铺垫。

---

## 目录

1. [KV cache 的本质：为什么推理要缓存 K/V](#1-kv-cache-的本质)
2. [MHA 之前是什么：注意力机制简史](#2-mha-之前是什么)
3. [head 的本质与架构制约](#3-head-的本质与架构制约)
4. [为什么 KV 是显存重点，共享的来龙去脉](#4-为什么-kv-是显存重点)
5. [FlashAttention 是怎么加速的](#5-flashattention-是怎么加速的)
6. [Linear Attention 是什么](#6-linear-attention-是什么)
7. [各种注意力能成立的本质原因](#7-各种注意力能成立的本质原因)
8. [为什么维度 = n_head × head_dim](#8-为什么维度--n_head--head_dim)
9. [代码解析：`nn.Linear(kv_lora_rank, n_head * head_dim)`](#9-代码解析)
10. [QKᵀ 来自哪里，来自不同 token 吗](#10-qk-来自哪里)
11. [学习式 wpe 为什么训练中能自动学出来](#11-学习式-wpe-为什么能学)
12. [RoPE 为什么两维一组旋转](#12-rope-为什么两维一组旋转)
13. [维度、head、head_dim 的关系](#13-维度headhead_dim-的关系)
14. [对外接口与各脚本职责](#14-对外接口与各脚本职责)
15. [weight decay 与 L2 正则化](#15-weight-decay-与-l2-正则化)
16. ["改 GPT 类"是什么意思](#16-改-gpt-类是什么意思)
17. [一句话总结](#17-一句话总结)

---

## 1. KV cache 的本质

### 为什么要缓存 K/V

自回归生成是**一个 token 一个 token 地吐**。生成第 t 个 token 时，模型要计算
"第 t 个 token 对前 0…t-1 个 token 的注意力"，这需要历史所有 token 的 K 和 V。

关键问题：**如果不缓存，会怎样？**

那每一步都要把整个前缀（0…t-1）重新跑一遍注意力前向，把它们的 K/V 重新算一遍。
生成 N 个 token，总计算量会变成 O(N³)（每步重算 O(t²) 的注意力，t 从 1 到 N）。
而缓存之后，每步只需要：

1. 算新 token 自己的 Q、K、V（O(1) 个新 token）；
2. 把新 K/V 拼到缓存里；
3. 新 Q 对缓存里的所有 K 做注意力。

这样总计算量降为 O(N²)。

> **本质：KV cache 是"时间换空间"的反向操作——用一小块显存，换来避免每一步都重算历史的巨大计算量。**
> 缓存不是"必须"，但不缓存意味着计算爆炸。

### 必须存在 GPU 里吗

**不必须。**

- 存在 GPU 显存（HBM）：最快，但显存有限；
- 存在 CPU 内存：可以（叫 offloading），但每次计算要把 K/V 搬回 GPU，通信开销大、变慢；
- 存在硬盘：更慢，极端场景才用。

所以 KV cache 的"位置"是速度与容量的权衡。MLA 的价值就在这：**它让同样 1GB 显存能装下更长的 KV cache**，因为每个 token 缓存的东西变小了。

---

## 2. MHA 之前是什么

### MHA 是哪一代模型的基础

MHA（Multi-Head Attention）是 **Transformer** 的基础，来自 2017 年的论文
《Attention Is All You Need》。所以"用 MHA 的 GPT"都源于 Transformer 这一代。

### MHA 之前是什么

| 年代 | 模型 | 怎么处理序列 |
|---|---|---|
| 2014 前 | RNN / LSTM / GRU | **串行**：一个隐藏状态从左到右一步步传递信息 |
| 2014 | seq2seq | encoder-decoder，靠最后一个隐藏状态压缩整句 |
| 2015 | Bahdanau Attention | 在 seq2seq 里加"对齐注意力"，让 decoder 能回看 encoder 所有位置 |
| 2017 | **Transformer + MHA** | **抛弃循环**，纯注意力，一步看全序列 |

### MHA 带来什么

1. **训练可并行**：RNN 必须按时间步串行，Transformer 所有 token 同时算，GPU 吃饱；
2. **长距离依赖**：RNN 传了 100 步信息会衰减，注意力一步直达任意距离的 token；
3. **可扩展**：堆层、堆头、堆宽度都能规模化，催生了 GPT/BERT。

> MHA 的核心突破一句话：**把"顺序传递信息"换成"每个位置直接按权重聚合所有位置"。**

---

## 3. head 的本质与架构制约

### head 的本质

把 n_embd 维的向量**切成 n_head 份**，每份 head_dim 维，**每一份独立做一次注意力**。

直觉：一个头可能关注"语法"，一个头关注"指代"，一个头关注"位置近邻"……
多头 = 多个并行的子空间，让模型能同时关注不同类型的关系。

### head 和架构的相互制约

1. **整除约束**：`n_embd = n_head × head_dim`，必须整除，否则切不匀；
2. **此消彼长**：n_embd 固定时，头多 → 每头窄（表达能力弱），头少 → 每头宽；
3. **参数量不变**：QKV 投影是 `n_embd × 3n_embd`，和头数无关（拼接起来总量一样），
   所以"加头"不改变参数量，只改变子空间切分方式；
4. **计算量**：attention 内部是 O(T² × head_dim × n_head) = O(T² × n_embd)，也与切分方式无关。

> 所以 head 数是一个"切蛋糕"的选择：切几块、每块多宽，蛋糕总量（n_embd）不变。

---

## 4. 为什么 KV 是显存重点

### 为什么是 K/V，而不是 Q

自回归生成第 t 个 token 时：

- **Q**：只有"当前这个新 token"的 Q 被用到，历史 token 的 Q 用完就丢，**不需要缓存**；
- **K/V**：所有历史 token 的 K/V 都要被"当前 token 的 Q"反复查询，**必须缓存**。

所以缓存里装的是 K 和 V 两份（V 用于最后加权求和）。

### 为什么能共享（MQA / GQA）

观察：多个头各自有一份 K/V，但它们的 K/V 高度冗余——都在描述同一个序列，只是投影不同。
于是提出：**让多个 Q 头共用同一份 K/V**。

- **MQA**（Multi-Query）：所有头共用 1 份 K/V，缓存里 K/V 的头数从 n_head 降到 1；
- **GQA**（Grouped-Query）：分 g 组，每组内共享，头数降到 g（MHA 和 MQA 的折中）。

### 共享与不共享的区别

| | 缓存头数 | 表达能力 | 显存 |
|---|---|---|---|
| MHA（不共享） | n_head | 最强 | 最大 |
| GQA（分组共享） | g | 中等 | 中等 |
| MQA（全共享） | 1 | 最弱 | 最小 |
| MLA（低秩） | 压缩成 c_kv | 强（靠低秩而非砍头） | 小 |

### 能有多少种变化

KV 共享的头数 `n_head_kv` 可以在 **1 到 n_head 之间任意取值**，是连续谱：
MQA 是一端（=1），MHA 是另一端（=n_head），GQA 是中间任意值。
MLA 则是另一个维度上的变化：**不砍头，而是把每个头的 K/V 压成低维 c_kv**。

---

## 5. FlashAttention 是怎么加速的

### 瓶颈在哪

标准注意力要显式算出 `softmax(QKᵀ/√d)`，这是 `(T, T)` 的大矩阵，要写回 GPU 的
高带宽内存（HBM）。当 T 很大时，**瓶颈不是算力（FLOPS），而是内存读写带宽**（memory-bound）。

### FlashAttention 优化了什么

两个核心技巧：

1. **分块（tiling）**：把 Q/K/V 切成小块，每块小到能塞进**片上缓存 SRAM**（带宽远高于 HBM）。
   在 SRAM 内完成"算分数 → softmax → 乘 V"，只把最终结果写回 HBM。
   **避免把整个 (T,T) 中间矩阵写回 HBM**。

2. **online softmax**：分块算 softmax 时，维护一个 running max 和 running sum，
   边算边归一化，不需要把完整矩阵算两遍。

### 结果

- 数学结果**基本不变**（数值上可忽略的微小差异）；
- HBM 读写大幅减少 → 快 2-4 倍；
- 显存占用从 O(N²) 降到 O(N)（因为不再物化 (T,T) 矩阵）。

> FlashAttention 不是新的"注意力机制"，它**没改公式，只改了计算公式的方式**（IO 优化）。

---

## 6. Linear Attention 是什么

标准注意力：`softmax(QKᵀ)V`，其中 QKᵀ 是 (T,T)，复杂度 O(T²)。

Linear Attention 的思路：**把 softmax 换成一个"可分离"的核函数**：

```
softmax(QKᵀ)  →  φ(Q) · φ(K)ᵀ     （φ 是某个特征映射）
```

于是利用矩阵结合律：

```
(φ(Q) φ(K)ᵀ) V = φ(Q) (φ(K)ᵀ V)
                   └── 先算这个小矩阵 (d×d)
```

先算 `φ(K)ᵀ V`（d×d，与 T 无关），再乘 φ(Q)，复杂度变成 **O(T·d²)**，**线性于序列长度**。

### 代价

丢掉 softmax 的显式归一化和"稀疏尖峰"特性（softmax 能让注意力聚焦到少数几个 token），
因此部分任务上长程建模能力会下降。

---

## 7. 各种注意力能成立的本质原因

剥开所有变体，注意力只有**三件事**：

1. **算相似度**：Q 和 K 怎么匹配（点积 / 核函数 / 加性……）；
2. **归一化**：把相似度变成权重（softmax / 其他归一化 / 无归一化……）；
3. **聚合**：用权重对 V 加权求和。

所有变体改的都是这三件事的某一部分：

| 变体 | 改了什么 |
|---|---|
| MHA / MQA / GQA | 改 Q、K、V 的**投影结构**（谁和谁共享） |
| MLA | 改 K/V 的**投影维度**（低秩压缩） |
| FlashAttention | 什么都不改，只改**计算方式**（IO） |
| Linear Attention | 改**相似度函数**（核函数）和**归一化** |
| Sliding Window | 改**注意力范围**（只看局部窗口） |

> 本质一句话：**注意力 = "按相似度加权聚合"。变体只是换了"相似度怎么算、权重怎么归一、聚合哪些 V"。**

---

## 8. 为什么维度 = n_head × head_dim

这是多头拆分的**定义**：

```
n_embd（总宽度） = n_head（切几块） × head_dim（每块多宽）
```

- `n_head` = 并行的注意力子空间数量；
- `head_dim` = 每个子空间的宽度；
- 两者乘积固定为 `n_embd`，所以是**此消彼长**的关系。

它们共同决定：

1. **模型宽度**（n_embd 本身）；
2. **每个头的表达能力**（head_dim 越大，单头能表达越复杂的关系）；
3. **并行子空间数量**（head 越多，能同时关注的关系类型越多）。

在 `model_mla.py` 里，低秩升维的目标 `n_head * head_dim` 之所以等于 `n_embd`，
就是因为**完整的 K 就是"n_head 个头、每个 head_dim 维"拼接起来**。

---

## 9. 代码解析

```python
self.wk_b = nn.Linear(kv_lora_rank, n_head * head_dim)
```

### 语法意思

`nn.Linear(in_features, out_features)` 创建一个线性层，做：

```
y = x @ Wᵀ + b
```

- 输入 `x` 形状：`(..., in_features)`；
- 输出 `y` 形状：`(..., out_features)`；
- 权重 `W` 形状：`(out_features, in_features)`（注意是反的）。

### 这行具体在干什么

```
kv_lora_rank       →  n_head * head_dim
（压缩后的 c_kv）      （完整的 K）
```

c_kv 是 `(B, T, kv_lora_rank)`，经过这层变成 `(B, T, n_head*head_dim)`。
因为完整 K 的宽度就是"n_head 个头 × 每个 head_dim 维"，所以输出维度写成
`n_head * head_dim`——**它就是把压缩的 c_kv 升维回完整 K 的矩阵**。

同理 `wkv_a` 是把完整 x 压下去：

```python
self.wkv_a = nn.Linear(n_embd, kv_lora_rank)   # x → c_kv（压缩）
self.wk_b  = nn.Linear(kv_lora_rank, n_head*head_dim)  # c_kv → K（升维）
self.wv_b  = nn.Linear(kv_lora_rank, n_head*head_dim)  # c_kv → V（升维）
```

---

## 10. QKᵀ 来自哪里

### Q 和 K 都来自输入 x

```python
q = self.wq(x)   # (B, T, n_head*head_dim)
k = self.wk(x)   # (B, T, n_head*head_dim)
```

它们都是**同一个输入序列 x 的线性投影**，只是用了不同的权重矩阵。

### QKᵀ 是什么

把 Q 和 K 都 reshape 成 (B, n_head, T, head_dim) 后：

```
att = q @ k.transpose(-2, -1)   # (B, n_head, T, T)
```

矩阵乘法的第 `[i, j]` 个元素：

```
att[i][j] = q_i · k_j = 第 i 个 token 的 query 和 第 j 个 token 的 key 的点积
```

所以：

- **是的，QKᵀ 来自不同 token 的两两交互**——它是一个 (T, T) 的"匹配分数表"；
- 第 i 行第 j 列 = "token i 有多想关注 token j"；
- 因果注意力里还会加 mask，只保留 j ≤ i（只看过去和现在，不看未来）。

---

## 11. 学习式 wpe 为什么能学

```python
self.wpe = nn.Embedding(block_size, n_embd)
```

- `nn.Embedding(block_size, n_embd)` 内部是一个形状 `(block_size, n_embd)` 的**参数矩阵**；
- 这个矩阵是 `nn.Parameter`，**会被梯度更新**；
- 第 i 行就是"位置 i"的向量，初始是随机的。

训练时目标是 next-token prediction，loss 的梯度会一路反传到这个矩阵，
于是模型**自动学会**："位置 0 的向量该长什么样、位置 1 的向量该长什么样"。
这就是"学习式"的含义——位置向量不是人设计出来的，是训出来的。

（对比：RoPE 是"固定公式算出来的"位置编码，不是学出来的。）

---

## 12. RoPE 为什么两维一组旋转

RoPE 的数学：把向量相邻两维 `(x_{2i}, x_{2i+1})` 看成一个 2D 平面上的点，然后旋转一个角度：

```
[ x_{2i}'   ]   [ cos θ_i   -sin θ_i ] [ x_{2i}   ]
[ x_{2i+1}' ] = [ sin θ_i    cos θ_i ] [ x_{2i+1} ]
```

**旋转是一个 2D 操作，必须两个坐标一起转**（转一个点需要 x、y 两个分量）。
所以维度要两两配对：第 0、1 维一组，第 2、3 维一组……每个维度对转不同的角度 θ_i。

这就是为什么 `qk_rope_head_dim` 必须是**偶数**——否则最后一维落单，没有配对对象，公式算不了。

（代码里用复数实现：`(x_{2i} + i·x_{2i+1}) × (cos θ_i + i·sin θ_i)`，复数乘法天然等价于 2D 旋转。）

---

## 13. 维度、head、head_dim 的关系

```
n_embd（总宽度 768）
   └── 切成 n_head（12）份
         └── 每份 head_dim = 768 / 12 = 64 维
```

- 每个头**只拥有** head_dim 维的向量（因为总宽度被切成了 n_head 份）；
- RoPE 作用在"每个头的向量内部"，所以它的可操作范围**最多就是 head_dim 维**；
- 所以 `qk_rope_head_dim ≤ head_dim`——旋转不能越界到别的头；
- `qk_rope_head_dim < head_dim` 时就是"解耦 RoPE"：头的前一部分维度管位置，后一部分管内容。

---

## 14. 对外接口与各脚本职责

### 对外接口

| 接口 | 职责 | 谁调用 |
|---|---|---|
| `forward(idx, targets) -> (logits, loss)` | 前向计算；有 targets 算 loss，没有就只算最后位置 logits | 训练循环、采样 |
| `generate(idx, ...)` | 自回归采样，内部反复调 forward | sample.py |
| `configure_optimizers(...)` | 生成优化器 + 参数分组 | train.py |
| `get_num_params()` | 统计参数量 | 日志、MFU |

这些接口是"模型对外的契约"。只要它们不变，外部脚本主体不用动。

### train.py 在干嘛

训练循环：

1. **取数据**：`get_batch` 从磁盘读 token 序列；
2. **前向**：`model(X, Y)` 得 loss；
3. **反传**：`loss.backward()`；
4. **梯度累积**：多个 micro-batch 累加梯度再更新（模拟大 batch）；
5. **更新**：optimizer 步进、梯度裁剪；
6. **评估与保存**：每隔 eval_interval 算 val loss，好的话存 checkpoint。

### sample.py 在干嘛

1. 加载 checkpoint（`model_args` → 重建 `GPTConfig` → 重建模型 → 读权重）；
2. 把 prompt 编码成 token；
3. 循环调用 `model.generate` 补全新 token；
4. 解码并打印。

### 日志在干嘛

训练循环里 `print(f"iter {iter_num}: loss {lossf:.4f}, ... mfu {running_mfu*100:.2f}%")`，
让你**监控训练是否正常**：loss 在降吗？速度多少？GPU 利用率多高？

### MFU 在干嘛

MFU = Model FLOPs Utilization（**模型算力利用率**）。

```
MFU = 实际每秒 FLOPs / 硬件峰值 FLOPs（A100 是 312 TFLOPS）
```

它回答："我花的 GPU 算力，有多大比例真的用在模型计算上了？"
MFU 低说明有瓶颈（数据搬运、通信、显存），不是 GPU 没吃饱。

---

## 15. weight decay 与 L2 正则化

### L2 正则化是什么

防止权重过大、防止过拟合的手段。在原始 loss 上加一项惩罚：

```
total_loss = 原始 loss + λ · ‖w‖²
```

λ 是正则强度。求梯度后，惩罚项贡献 `2λw`，所以权重更新时额外向 0 收缩。

### weight decay 是什么

weight decay 是 **L2 正则化的"实现形式"**：每次更新时，先把权重乘以一个略小于 1 的系数：

```
w ← w · (1 - η·λ) - η·∇loss
         └── 这个 (1-η·λ) 就是"衰减"
```

效果等价于 L2 正则化，让权重越来越小。

### 和注意力的联系

实践中有个经验法则：**2D 矩阵权重要衰减，1D 向量（bias、LayerNorm/RMSNorm 的 scale）不衰减**。

`configure_optimizers` 按 `p.dim() >= 2` 分组：

- MLA 新增的 `wkv_a / wk_b / wv_b` 是 2D 矩阵 → **进 decay 组**；
- `RMSNorm.weight` 是 1D → **不进 decay 组**。

这就是为什么改 MLA 时 optimizer 分组**不用改代码**——它按维度自动分，而 MLA 的新层恰好落在正确的那组。

---

## 16. "改 GPT 类"是什么意思

`GPT` 是**顶层模型类**，负责把所有零件组装起来：

```
GPT
 ├── transformer.wte   （token embedding）
 ├── transformer.wpe   （位置 embedding，MLA 里可选）
 ├── transformer.h     （n_layer 个 Block，每个 = attention + MLP）
 ├── transformer.ln_f  （最后 LayerNorm）
 └── lm_head           （输出层）
```

"改 GPT 类"指的是改这个组装层的 4 个地方：

1. **`wpe` 条件创建**：`use_rope=True` 时根本不建 wpe（位置交给 RoPE）；
2. **`forward` 条件叠加**：有 wpe 就 `tok_emb + pos_emb`，没有就只用 `tok_emb`；
3. **`crop_block_size` / `get_num_params` 加保护**：wpe 可能不存在，要 `hasattr` 判断；
4. **删 `from_pretrained`**：GPT-2 权重布局对不上 MLA。

> 换句话说：**改 `CausalSelfAttention` 是改"注意力怎么算"，改 `GPT` 类是改"顶层怎么把位置编码和注意力拼起来"。** 两者配合，MLA 才算改完整。

---

## 17. 一句话总结

- **注意力本质** = 按相似度加权聚合：`softmax(QKᵀ/√d)·V`；
- **KV cache 本质** = 用显存换"避免每步重算历史"，不缓存则计算爆炸，放 GPU/CPU/硬盘只是速度与容量的权衡；
- **MHA** = Transformer（2017）的多头并行子空间，取代了串行的 RNN；
- **head 本质** = 把 n_embd 切成 n_head 个 head_dim 宽的独立子空间，两者乘积固定、此消彼长；
- **KV 共享/压缩** = 因为 KV 是缓存的唯一大头，MQA/GQA 砍头数、MLA 降维度；
- **FlashAttention** = 不改公式，只优化 IO（分块 + online softmax）；
- **Linear Attention** = 换掉 softmax 用可分离核函数，复杂度降到线性；
- **MLA 本质** = 低秩压缩 K/V 成 c_kv 省缓存 + 解耦 RoPE 保位置建模；
- **本次改动收敛为两处**：投影布局（融合 c_attn → 低秩 K/V）+ 位置编码（wpe → 解耦 RoPE），
  其余全部不动，靠 `model_args` 字典和 `GPTConfig(**model_args)` 贯穿训练与采样两端。

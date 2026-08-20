# 实验记录

> 相关：[MLA 深度解析](mla_deep_dive.md) · [推理路径](inference_path.md) · [MoE 踩坑清单](moe_pitfalls.md)

所有实验在 shakespeare-char 数据集、MacBook CPU 上跑。绝对数字没有意义（MFU ~0.2%），
**只看同一硬件下的相对差异**。

---

## 实验 1：MLA vs 稠密 MHA baseline（训练质量）

### 配置

两次运行只差注意力实现，其余完全相同：

```
dataset = shakespeare_char      n_layer = 6      n_head = 6       n_embd = 384
block_size = 256                batch_size = 64  dropout = 0.2
learning_rate = 1e-3            max_iters = 5000 warmup_iters = 100
min_lr = 1e-4                   beta2 = 0.99     always_save_checkpoint = False
```

| | baseline | MLA |
|---|---|---|
| 脚本 | `train.py` + `model.py` | `train_mla.py` + `model_mla.py` |
| 配置 | `config/train_shakespeare_char.py` | `config/train_shakespeare_char_mla.py` |
| 注意力 | MHA，融合 `c_attn` | MLA，`kv_lora_rank=128` |
| 位置编码 | 学习式 `wpe` | RoPE（`qk_rope_head_dim=0`，整头旋转的 legacy 布局） |
| 原始日志 | `baseline_train.log` | `mla_train.log` |

### 结果

| 指标 | baseline | MLA | 差异 |
|---|---|---|---|
| 参数量 | 10.65 M | **9.76 M** | **−8.4 %** |
| 衰减参数张量数 | 26 | 43 | 低秩把 1 个融合投影拆成了多个 |
| **best val loss** | **1.4716**（step 1750） | 1.4757（step 1250） | +0.0041 |
| final val loss（step 5000） | 1.7127 | **1.6636** | −0.049 |
| final train loss | 0.6178 | 0.6602 | +0.042 |

### 结论

1. **参数少 8.4%，best val 基本持平（+0.004）。** 低秩瓶颈在这个规模下几乎没有造成质量损失。
2. **MLA 过拟合更轻**：final train 更高、final val 更低。低秩 K/V 起到了正则化作用，这与
   「瓶颈限制表达能力」的直觉一致——在小数据集上，这个限制反而是好事。
3. MLA 的 best val 出现得更早（1250 vs 1750），也是同一现象的表现。

### 这个实验**不能**说明什么

- **不能**说明 MLA 更省显存 —— 训练时 K/V 照样要 materialize，省的是**推理** cache（见实验 2）。
- **不能**外推到大模型 —— `kv_lora_rank=128`、`head_dim=64` 在这个尺度下瓶颈很松。
- 参数量差异里有一部分来自去掉 `wpe`（256 × 384 ≈ 98K），不全是低秩的功劳。

---

## 实验 2：KV cache 与吸收解码（推理成本）

`bench_kv_cache.py`。**每次测量前都先断言 `cached == uncached`**，否则测的是一个坏实现的速度。

### 2a. 已训练的 checkpoint（legacy RoPE 布局）

```bash
python bench_kv_cache.py --out_dir=out-shakespeare-char-mla --device=cpu --max_new_tokens=200
```

```
cached == uncached : YES

KV cache, whole model, 256 tokens, batch 1, float32:
  MLA (cached c_kv)            768 numbers/token       768.0 KiB
  MHA equivalent (K and V)    4608 numbers/token      4608.0 KiB
  reduction                   6.00x

decode 200 tokens from a 32-token prompt (cpu):
  no cache            1296.0 ms   (6.48 ms/token)   1.00x
  cache, naive         333.2 ms   (1.67 ms/token)   3.89x
  cache, absorbed   n/a (legacy layout has no nope part)
```

### 2b. decoupled RoPE 布局（`qk_rope_head_dim=16`，随机初始化，仅测成本）

```bash
python bench_kv_cache.py --device=cpu --block_size=512 --max_new_tokens=400
```

```
cached == uncached : YES

KV cache, whole model, 512 tokens, batch 1, float32:
  MLA (cached c_kv + k_R)      864 numbers/token      1728.0 KiB
  MHA equivalent (K and V)    4608 numbers/token      9216.0 KiB
  reduction                   5.33x

decode 400 tokens from a 32-token prompt (cpu):
  no cache            4497.7 ms   (11.24 ms/token)   1.00x
  cache, naive         894.1 ms   ( 2.24 ms/token)   5.03x
  cache, absorbed      730.0 ms   ( 1.83 ms/token)   6.16x
```

### 结论

1. **cache 是 MLA 收益的全部载体。** 没有 cache 时，MLA 相对 MHA 的推理优势严格为 0。
2. **6.00× 的 cache 削减**（legacy）/ **5.33×**（decoupled，因为多缓存了 16 维 `k_R`）。
   decoupled 用 12% 的额外 cache 换来了吸收路径。
3. **吸收在朴素 cache 之上再快 1.22×**（894 → 730 ms）。它省掉的是把 `c_kv` 解压成 K/V 的
   那次矩阵乘 —— 也就是把「省容量」变成同时「省带宽」。
4. 无 cache → 有 cache 的 3.89×~5.03× 加速会随生成长度增长（无 cache 是 O(T³)，有 cache 是 O(T²)）。

### 对照 DeepSeek-V2 的真实比例

本仓库的模型太小，`kv_lora_rank=128` 相对 `2·384=768` 只有 6×。V2 的比例是：

```
MLA : kv_lora_rank(512) + qk_rope_head_dim(64)      =   576
MHA : 2 × n_head(128) × head_dim(128)               = 32768
                                                      ≈ 56.9x
```

**head 数越多、head_dim 越大，MLA 的相对收益越大。** 这是它在大模型上才真正发力的原因。

---

## 实验 3：等价性测试

`test_mla.py`。7 种配置 × 5 组断言，全部通过。

| 配置 | 覆盖的代码路径 |
|---|---|
| `mla-legacy-rope` | 整头旋转、`k_R` 来自 `c_kv` |
| `mla-decoupled` | DeepSeek 布局：独立 `W_KR`、共享单头、可吸收 |
| `mla-decoupled-v` | 上面 + `v_head_dim ≠ head_dim`（非方阵 `c_proj`） |
| `mla-qlora` | `q_lora_rank > 0` 的 Q 侧压缩 |
| `mla-norope` | RoPE 关闭，走学习式 `wpe`（`start_pos` 偏移） |
| `mha-fallback` | `kv_lora_rank=0`，全秩 K/V 直接缓存 |
| `mha-norope` | 最朴素的对照路径 |

| 断言 | 检出的 bug 类型 |
|---|---|
| 有 cache 生成 == 无 cache 生成 | 位置偏移、mask、索引、模式 |
| 吸收路径 == 朴素路径 | `W_UK`/`W_UV` 折叠的数学错误 |
| 分块 prefill == 一次 prefill | 偏移 causal mask |
| 连续两次 generate == 单独跑第二次 | cache 未重置 |
| 改未来 token 不影响过去 logits | cache 破坏因果性 |

**为什么必须贪心（`temperature=0.0`）**：采样有随机性，两条路会因 RNG 状态不同而分叉。
**为什么必须 `model.eval()`**：dropout 会让两条路必然不同。

`mla_min.py` 是一个 160 行的独立最小实现，用同样的思路自检（吸收 vs 朴素、cache vs 全量前向），
可以脱离整个仓库单独阅读。

---

## 待办

- [ ] **用 `qk_rope_head_dim=16` 重跑训练对照。** 实验 1 的 MLA 结果来自 legacy 整头旋转布局；
      配置默认值已改为 decoupled，需要重训一次才能更新表格。CPU 上约 2 小时。
- [ ] 等参数量对照：给 MLA 加宽 `n_embd` 或加层，补一组「等参数量」而非「等超参」的对照。
- [ ] 在 GPU 上重测实验 2 —— CPU 的带宽/算力比和 GPU 差异很大，加速比不可直接外推。
- [ ] MTP（D=1, λ=0.3），见 [inference_path.md](inference_path.md) 第 7 节的清单。
- [ ] MoE，见 [moe_pitfalls.md](moe_pitfalls.md)。

# MoE 落地踩坑清单（nanoGPT / model_mla.py 基础上）

> 本文只记录**工程坑**，不讲 MoE 原理。
> 适用前提：在 `model_mla.py` / `train_mla.py` 的基础上，把 `MLP` 换成 top-k 路由的 `MoELayer`。
> 相关：[MLA 深度解析](mla_deep_dive.md)

## 目录

1. [DDP 会炸：未激活专家没有梯度](#1-ddp-会炸未激活专家没有梯度)
2. [MFU 虚高：total params vs activated params](#2-mfu-虚高total-params-vs-activated-params)
3. [优化器分组：router 权重与 aux-loss-free bias](#3-优化器分组router-权重与-aux-loss-free-bias)
4. [负载统计的跨卡同步](#4-负载统计的跨卡同步)
5. [其他小坑](#5-其他小坑)
6. [实验设计：对齐哪个参数量口径](#6-实验设计对齐哪个参数量口径)
7. [验收清单](#7-验收清单)

---

## 1. DDP 会炸：未激活专家没有梯度

### 现象

多卡训练启动后抛出：

```
RuntimeError: Expected to have finished reduction in the prior iteration
before starting a new one. This error indicates that your module has
parameters that were not used in producing loss.
```

或者干脆卡死不动（各卡在等一个永远凑不齐的 bucket）。

### 根因

`DistributedDataParallel` 在包装模型时，给**每一个 `requires_grad=True` 的参数**挂 autograd hook。
参数被分成若干 bucket，某个 bucket 里的梯度全部就绪时立刻发起 `all_reduce`，
让通信和反向计算重叠 —— 这是 DDP 快的原因，但它的前提是
**每次 backward 每个注册过的参数都会拿到梯度**。

top-k 路由破坏了这个前提：在某一个 micro-batch、某一张卡上，
完全可能出现某个专家一个 token 都没被分到 →
该专家的权重不在计算图里 → hook 不触发 → bucket 永远凑不齐。

### 为什么单卡看不出来

单卡没有 DDP 包装，没有 hook。未使用参数的 `p.grad` 保持 `None`，
AdamW 遇到 `grad is None` 直接跳过 —— **静默、无害、什么都不发生**。
所以本地 mac 上跑 shakespeare 永远不会遇到，一上多卡必现。

### 相关代码位置

`train_mla.py:215`

```python
model = DDP(model, device_ids=[ddp_local_rank])   # find_unused_parameters 默认 False
```

### 三种解法

| 方案 | 做法 | 代价 |
|---|---|---|
| **A. 开关**（推荐给学习项目） | `DDP(model, device_ids=[...], find_unused_parameters=True)` | 每次迭代额外遍历一遍计算图找未使用参数；关掉部分 bucket 优化；**与 `static_graph=True` 冲突** |
| **B. 零权重触碰** | loss 里加 `0.0 * sum(p.sum() for p in experts)` | 骗过 DDP，但每步遍历全部专家参数，纯浪费。不推荐 |
| **C. 专家并行 EP** | 专家**分片**到各卡而非复制，token 经 all-to-all 发到拥有该专家的卡 | 真实系统做法（DeepSeek 即是），远超 nanoGPT 复杂度预算 |

### 额外注意

nanoGPT 用 `require_backward_grad_sync` 手动控制梯度累积的同步时机
（`train_mla.py:295-301`）。该机制与 `find_unused_parameters=True`
的交互是已知脆弱点 —— **开了 A 之后，必须专门测一次 `gradient_accumulation_steps > 1` 的路径**。

---

## 2. MFU 虚高：total params vs activated params

### 现象

MFU 数字看起来很漂亮，但和实测吞吐对不上。

### 根因

MFU = 实际达到的 FLOPs / 硬件峰值 FLOPs。nanoGPT 用 PaLM 论文附录的估算：

```python
flops_per_token = 6*N + 12*L*H*Q*T
```

- `6*N` —— 每个参数前向贡献 2 FLOPs（一次乘加），反向 4 FLOPs，共 6。
- 第二项 —— 注意力随序列长度平方增长的部分。

稠密模型里 `N = 总参数量` 是对的，因为**每个 token 都要经过每一个参数**。

MoE 的定义就是每个 token 只激活一小部分参数，这个等式不再成立。

### 影响有多大

设 8 专家、top-2，注意力占非嵌入参数 1/3、FFN 占 2/3：

```
实际激活比例 = 1/3 + 2/3 × (2/8) = 1/2
```

`flops_per_token` 高估 **2 倍** → 报出的 MFU 是真实值的两倍。

### 相关代码位置

`train_mla.py:328` → `model_mla.py: GPT.estimate_mfu()` → 内部调 `get_num_params()`

### 怎么改

**① `get_num_params` 支持两种口径**

```python
def get_num_params(self, non_embedding=True, activated=False):
    """
    activated=False: 全部参数（报模型规模用）
    activated=True : 每 token 实际经过的参数
                     稠密部分全算 + 专家部分 × (top_k / n_experts) + router 全算
    """
```

**② `estimate_mfu` 改用激活口径**

```python
N = self.get_num_params(activated=True)
```

**③ 初始化时两个数都打印**

```
number of parameters: 45.20M total, 12.80M activated per token
```

这不是美化。MoE 论文一律双口径报数 —— DeepSeek-V3 是「**671B 总参数 / 37B 激活参数**」。
MoE 语境下只报一个数没有意义。

---

## 3. 优化器分组：router 权重与 aux-loss-free bias

### 现有规则

`model_mla.py: configure_optimizers()`

```python
decay_params   = [p for n, p in param_dict.items() if p.dim() >= 2]
nodecay_params = [p for n, p in param_dict.items() if p.dim() <  2]
```

粗糙但有效的启发式：**2 维以上 = 矩阵乘权重和嵌入 → 衰减；1 维 = bias 和 norm gain → 不衰减**。
对 bias / norm 做 weight decay 有害，因为它们的「零」有特殊含义。

### 坑 3.1 —— router 权重是 2D，会被 weight decay

router 通常是 `nn.Linear(n_embd, n_experts, bias=False)`，2 维，落进 decay 组。

**这是好是坏？** router 输出的是 logits，**重要的是相对大小**。
weight decay 持续把权重拉向零 → logits 整体变小 → softmax 后趋于均匀 →
路由越来越不敢做出坚定选择。既是一种微弱的隐式均衡压力，
也是对 router 学习能力的持续压制。

不致命，但**要有意识**。常见做法：

- 把 gate 排除出 decay 组，或
- 给它单独的（更小的）学习率。

另外两条实践经验：

- router 初始化 std 应比常规更小；
- **gate 的计算保持 fp32** —— 路由是离散的 top-k 决策，对数值精度敏感，
  bf16 下可能出现跨卡决策不一致。

### 坑 3.2 —— aux-loss-free bias（DeepSeek-V3 做法）

**机制**：传统 MoE 靠辅助损失（auxiliary loss）逼专家负载均衡，
但该损失的梯度会和语言建模主损失打架，损害模型质量。

V3 的替代方案：每个专家维护一个偏置 `b_e`，
**只在 top-k 选择时加到路由分数上，不参与最终组合输出的权重计算**。
每步之后手动更新：

```
专家过载 → b_e -= γ
专家欠载 → b_e += γ
```

纯手动、无梯度。相当于给冷门专家加分、热门专家减分来调节流量，而不污染梯度。

**落地方式**：

```python
self.register_buffer('expert_bias', torch.zeros(n_experts))
```

用 `register_buffer` 最干净：不进 `named_parameters()`，
所以 `configure_optimizers` 天然看不到它；同时 **buffer 会进 `state_dict`**，
checkpoint 保存/恢复正常。

如果偏好 `nn.Parameter(..., requires_grad=False)` 也可以 ——
`configure_optimizers` 的第二行正好会过滤掉：

```python
param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
```

**这一点现有代码是兼容的，不需要改。**

### 注意

V3 仍保留一个**系数极小**（α ≈ 1e-4）的 sequence-wise 辅助损失，
防止单条序列内部出现极端不均衡。别以为 V3 把辅助损失彻底删干净了。

---

## 4. 负载统计的跨卡同步

### 根因

更新 `expert_bias` 依据的是「各专家收到多少 token」，
而**每张卡看到的 token 不同，统计出的负载也不同**。
各卡各自更新 → `expert_bias` 逐渐发散 → 每张卡在跑不同的模型。

这是一个**静默**故障：不报错，只是模型慢慢坏掉。

### 正确做法

```python
# 必须在 optimizer.step() 之后
if ddp:
    dist.all_reduce(expert_load_counts, op=dist.ReduceOp.SUM)
update_expert_bias(expert_load_counts)
```

同理，**打印/记录专家负载分布时也要先 all_reduce**，
否则你看到的是 rank 0 的局部视角，不是全局分布。

---

## 5. 其他小坑

### 5.1 梯度裁剪

`clip_grad_norm_(model.parameters(), grad_clip)` 在 MoE 下仍然可用，
但专家梯度是稀疏且尖峰的，全局范数波动会明显变大。
不需要改，但看到 grad norm 抖动加剧不要惊讶。

### 5.2 checkpoint 体积

专家参数会让 checkpoint 迅速膨胀。`.gitignore` 里已有 `*.pt`，确认它继续生效。

### 5.3 eval 与 train 的路由一致性

`estimate_loss()` 里 `model.eval()` 会关掉 dropout，但**不会**改变路由行为。
如果实现里让 bias 更新或负载统计依赖 `self.training`，务必确认 eval 路径不会污染统计。

### 5.4 专家负载必须被记录

每次 eval 时打印各专家的 token 占比。**专家塌缩**（少数专家吃掉绝大多数 token）
是 MoE 最常见的失败模式，而它**不会体现在 loss 曲线上** ——
loss 照常下降，只是模型实际容量退化成了几个专家。
不打印就等于没有观测。

### 5.5 MTP 模块里的专家

如果先做了 MTP：MTP 模块内部用的是同一个 `Block` 类，
所以它会**自动获得 MoE**（这与 DeepSeek 的设计一致）。
但那些专家同样要纳入负载统计和 bias 更新 —— 别漏掉。

---

## 6. 实验设计：对齐哪个参数量口径

有了双口径之后，跑对照实验前必须先决定**对齐哪一根轴**：

| 对照方式 | 预期结论 |
|---|---|
| **等激活参数量**（MoE 总参数远大于 baseline） | MoE 应该赢 —— 这才是 MoE 的卖点：同样的计算成本，更大的容量 |
| **等总参数量**（MoE 激活量远小于 baseline） | MoE 大概率打平或略输，但速度快得多 |

选错对照轴，跑出来的结论无法解读。**这个决定要写进实验记录。**

---

## 7. 验收清单

做完 MoE 后逐条确认：

- [ ] 单卡训练 loss 正常下降
- [ ] 每个专家的 token 占比被打印出来，且**没有塌缩**（没有专家长期接近 0）
- [ ] `number of parameters` 同时打印 total 和 activated
- [ ] `estimate_mfu` 用的是 activated 口径
- [ ] router 的 weight decay 归属是**有意识的决定**，不是默认掉进去的
- [ ] `expert_bias` 不在优化器参数组里，但在 `state_dict` 里
- [ ] checkpoint 存→ 读→ 继续训练，loss 无跳变
- [ ] 若上多卡：`find_unused_parameters=True`，且**专门测过梯度累积路径**
- [ ] 若上多卡：负载统计有 `all_reduce`
- [ ] 实验记录里写明对齐的是 total 还是 activated 参数量

# Open-Jev 是如何把 Qwen 做成类型化决策模型的

> 代码库：<https://github.com/Zefan-Cai/Open-Jev>
> 对应 README 第 3-4 行：*Give your app a decision, with probabilities... get typed
> probabilities directly, without autoregressive answer generation or parsing generated JSON.*
> 本文按 `jev/` 的实际代码逐环节说明这句话的实现方式；文中出现的文件名与行号（`api.py:48`、
> `model.py:108-111`、`prefix_cache.py:212` 等）均相对该仓库根目录。
>
> **同名配对文档**：另一个项目 RYOTIDE 的 [`docs/ryotide/typed-decision-model.md`](../ryotide/typed-decision-model.md)。两者是同一个目标
> （封闭的概率分布、不生成 token）的两条实现路线，本篇刻意与它同名以便并读对照。
> §0 先讲清分歧，§9 把它的四条不变量逐条落位到 Open-Jev。

## 0. 与 RYOTIDE 的分野

两套系统都承诺**输出是一个封闭的概率分布，全程不生成 token**，但达成方式相反：

| | RYOTIDE | Open-Jev |
|---|---|---|
| 权重来源 | 公开 LLM（Qwen3.5 / Gemma 4），全程 frozen | 公开基座（走查用的那一档是 `Qwen/Qwen3.5-2B`）+ **自训的 LoRA 与标量决策头**；上游发布的三个包是 `ZefanCai/Open-Jev-2B` / `-9B` / `-27B-v1.1`（基座 `Qwen/Qwen3.5-2B` / `Qwen/Qwen3.5-9B` / `Qwen/Qwen3.8-27B`，`README:33-35`），且**只含组件、不含合并后的 Qwen 权重**（`README:389`） |
| 决策机制 | 读**标签 token** 在最后一个位置的 logits，掩码 + softmax | **逐候选点式打分**，再对候选分数做 softmax |
| 需要训练 | 否，权重全程 frozen | 是：LoRA + 一个标量决策头 |
| 「类型化」来自 | 标签必须是单 token（否则映射成 `A/B/C` 标记） | 请求里声明的 `answer_keys` + 响应校验 |
| 候选数上限 | 260（`A0`–`Z9` 两步读） | 255（`api.py:48`），一步打分 |
| 候选顺序偏置 | 需跑多个 order 再平均来消除 | 不存在该变量（pointwise 无顺序） |
| 置信度来源 | 一次前向 + 事后拟合的 temperature | 软目标训练学出分布，temperature 再校正 |
| 读位置标定 | 运行期扫 9 前缀 × 2 标记（`marker_mass`） | 无标定；`Yes or No` 是固定问句 |
| 诊断量 | `marker_mass` | 概率校验 + `max_length` 硬报错 + 拒绝服务未加载模型 |
| 共同底线 | **`output_tokens = 0`** | 同（`serving.py:93`、`predict.py:48`） |

一句话：RYOTIDE 是**「不改权重，只换读法」**；Open-Jev 是**「把决策拆成 N 个独立的是非题，
再训练一个小头把它读成校准打分」**。

## 1. 端到端数据流

```
                        ┌────────────── serving.py / predict.py ──────────────┐
POST /v1/systemone      │ Predictor.predict()                                  │
{state, questions}      │   compile_request() → records                        │
                        │   candidate_batches() → 按 batch_size 切分大 Choice   │
                        └───────────────────────┬─────────────────────────────┘
                                                v
        ┌──────────────────── api.py ────────────────────┐
        │ compile_request()  choice/score/noul → record   │  ← 「类型化」在此定义
        │ candidate_prompts() 每个候选一条独立序列         │  ← pointwise，无顺序变量
        │ format_response()  校验 + 组装类型化答案         │  ← 封闭性的守门人
        └───────────────────────┬─────────────────────────┘
                                v
        ┌──────── model.py / prefix_cache.py ────────────┐
        │ apply_chat_template(enable_thinking=False)      │
        │ forward()/score_cached() 取隐状态最后一位         │
        │ head(h_last) → 每个候选一个标量 s_i               │
        └───────────────────────┬─────────────────────────┘
                                v
        softmax([s_1..s_n], T)  ← 候选之间归一化 = 决策       metrics.py:25
                                v
        format_response() → 决策 JSON（无生成 token，output_tokens = 0）
```

## 2. 「typed」从哪来：请求声明的答案空间 + 响应校验

RYOTIDE 的类型化依赖分词器（标签必须落在单 token 上）。Open-Jev **完全不依赖分词器**，
类型化由两端夹住：

**入口**（`api.py:38-74`）：三种问题类型，各自固定 `answer_keys`：

| kind | 候选约束 | `answer_keys` | 输出 |
|---|---|---|---|
| `choice` | `1 <= len(criteria) <= 255`（`api.py:48`） | `criteria` 的键，保持作者顺序 | argmax 标签 + 全分布 + confidence |
| `score` | `2 <= len(criteria) <= 10`（`api.py:59`） | `"0"`…`"n-1"` | **期望等级** `Σ i·p_i` + 分布 + legend |
| `noul` | 布尔 | `["false","true"]` | `P(yes)` |

**出口**（`api.py:106-111`）：概率必须有限、落在 `[0,1]`、**和为一**（`abs_tol=1e-6`），否则抛错：

```python
if any(not math.isfinite(value) or not 0 <= value <= 1 for value in probs):
    raise ValueError("probabilities must be finite and in [0, 1]")
if not math.isclose(total, 1.0, rel_tol=1e-6, abs_tol=1e-6):
    raise ValueError("probabilities must sum to one")
```

选择用校验而非掩码来封闭输出，直接后果是**候选可以是任意自由文本**——`choice` 的候选渲染为
`name: description`（`api.py:54-57`），多 token、含标点、含 Unicode 都无所谓。RYOTIDE 需要
`A0`–`Z9` 两 token 码加两步读才能表达的 255 个选项，这里只是「255 条独立序列」。

`state`、`instructions`、`criteria` 都允许是**对象或数组**（`api.py:10-21`），用
`json.dumps(..., sort_keys=True)` 确定性渲染——这是请求哈希可复现的前提。

## 3. 决策头：从 Yes-minus-No 播种

这是全仓最关键、也最容易被忽略的一处设计。

`model.py:108-111` 先要求 `"Yes"`/`"No"` 各自是**单 token**——这是唯一残留的「标签必须单 token」
约束，且只用于初始化：

```python
yes = self.tokenizer.encode("Yes", add_special_tokens=False)
no  = self.tokenizer.encode("No",  add_special_tokens=False)
if len(yes) != 1 or len(no) != 1:
    raise ValueError("The Yes/No baseline requires single-token labels")
```

`model.py:112-126` 读出语言模型自己 `lm_head` 的两行之差，`model.py:182-186` 用它播种决策头：

```python
initial = (_ow[yes[0]] - _ow[no[0]]).detach().float().clone()   # [hidden_size]
...
self.head = nn.Linear(hidden_size, 1, bias=True, dtype=torch.float32)
with torch.no_grad():
    self.head.weight.copy_(initial.unsqueeze(0))
    self.head.bias.zero_()
```

因为 `head(h) = ⟨W_yes − W_no, h⟩`，而模型自身在末位置的 Yes/No logit 差正是
`⟨W_yes − W_no, h_last⟩ + (b_yes − b_no)`（偏置置零后完全相等）——**初始化瞬间，这个标量头
逐字复现了「直接问模型 Yes 还是 No」的 logit margin**。

这条设计有三个可验证的推论：

1. `serving.py:139-142` 在不加载任何 checkpoint 时把方法名报为
   **`pretrained_yes_minus_no_no_training`**——此时 Open-Jev 退化成一个**零训练分类器**，
   正是 RYOTIDE 那套「读 logits」思路的特例。训练做的是把这个方向从「模型的 yes/no 偏好」
   精化成「该候选是否正确的校准打分」。
2. 训练不是从随机头起步，所以 baseline 有意义：`train.py:389-396` 在**任何优化器更新之前**
   先 evaluate 未更新的权重，得到 `baseline_test/ood/calibration.jsonl` 与
   `baseline_temperature`，用于事后对照（`train.py:485-488`）。
3. 输出头是 `float32` 的 `Linear(hidden, 1)`，输入 `hidden.float()`（`model.py:262`）——
   打分本身不经过低精度路径。

LoRA 只作用于注意力/线性注意力投影（`model.py:187-199`）：`q_proj, k_proj, v_proj, o_proj,
in_proj_qkv, out_proj`，`r=8`、`alpha=16`、`dropout=0.0`、`bias="none"`；backbone 主体
`requires_grad_(False)`（`model.py:132`）。

## 4. 独立候选：pointwise 而不是 listwise

`api.py:78-85` 是全仓最短也最关键的一段：

```python
def candidate_prompts(record: dict) -> list[str]:
    prefix = f"Context:\n{_render(record['state'])}\n\nQuestion: {_render(record['question'])}\n"
    if record["kind"] == "noul":
        return [prefix + "Is the answer to this question yes? Answer Yes or No."]
    # Candidate order, question IDs, adjacent score levels and targets are absent.
    return [prefix + f"Proposed answer: {_render(option)}\nIs this proposed answer correct? Answer Yes or No."
            for option in record["options"]]
```

三个直接后果：

- **候选数不受词表约束**。上限 255 是接口约定（`api.py:48`），不是模型限制。
- **候选顺序不是变量**（源码注释直说 "Candidate order ... are absent"）。这是对 RYOTIDE §6
  「顺序去偏」的**架构性替代**：RYOTIDE 必须跑多个 order 再平均，才能消除
  `A. yes / B. no` 与反向 72% vs 21% 的顺序偏置；pointwise 打分里根本不存在这个变量。
- **代价是 N 倍前向**。RYOTIDE 的 N 个 order 共享同一段 state 前缀，选项块只有几十个 token，
  顺序平均「几乎免费」；Open-Jev 每个候选都是一条完整序列。这是它必须有 §7 前缀缓存的根本原因。

问题 ID、相邻等级、目标都不进 prompt：`candidate_prompts` 只读 `state`、`question`、`options`
（`api.py:79` 注释：*using only declared model input fields*）。`answer_keys` 与 `legend` 是
纯软件侧元数据，从不传给模型。

## 5. 只读最后一个位置 → 标量 → softmax

`model.py:259-262`：

```python
hdev = outputs.last_hidden_state.device
idx = torch.arange(len(prompts), device=hdev)
hidden = outputs.last_hidden_state[idx, lengths.to(hdev) - 1]   # 每条序列最后一个真实位置
scores = self.head(hidden.float().to(self.head.weight.device)).squeeze(-1)
```

`padding_side = "right"`（`model.py:44`），注意力是因果的，所以右填充不会污染真实末位置——
这与 RYOTIDE 不变量 #3（只读最后一个位置）同构，只是读出的是一行隐状态而非一整行词表 logits。

**`noul` 的构造**（`model.py:266-269`）：logits 被拼成 `[0, s]`，于是

```
softmax([0, s]) = [1−σ(s), σ(s)]   ⇒   P(yes) = sigmoid(s)
```

这就是 `metrics.py:15-22` 那个 `sigmoid` 的来历。

**归一化**在 `serving.py:92`（与 `predict.py:45`）：

```python
result = format_response(records, [softmax(row, self.temperature) for row in logits])
```

`metrics.py:25-36` 的 `softmax` 接受 `temperature`，并显式允许 `-inf`（用于屏蔽类别）：

```python
weights = [math.exp((value - maximum) / temperature) for value in logits]
```

**confidence 不是校准正确率**。`api.py:119` 用 `choice_confidence`，`metrics.py:69-79` 明确写着
*"Official system-one-adapter formula, not calibrated correctness"*：

```python
uniform = 1.0 / len(probabilities)
return (max(probabilities) - uniform) / (1.0 - uniform)   # 均匀 → 0，确定 → 1
```

`score_confidence`（`metrics.py:81-91`）是围绕众数等级的集中度。两者都只描述分布形状，
不能被读成「正确答案的概率」。

## 6. 训练：软目标交叉熵 + Brier

`train.py:413-421`：

```python
logits = model([row])[0].float()
target = torch.tensor(row["target"], device=logits.device)
loss = -(target * logits.log_softmax(-1)).sum()
loss = loss + args.brier_weight * ((logits.softmax(-1) - target) ** 2).sum()
```

`brier_weight` 默认 `0.1`（`train.py:558`）。关键点是 **`target` 可以是软分布**——
`metrics.py:12` 的类型是 `Target = int | Sequence[float]`，数据侧大量使用非 one-hot 目标：

| 目标类型 | 位置 |
|---|---|
| `deterministic_explicit_rules` | `data.py:202` |
| `defined_uniform_latent_worlds` | `case_customer.py:218` |
| `Uniform best available actions under full-graph shortest-path teacher` | `case_wikiracing.py:232` |
| `exact_programmatic_oracle` | `case_reasoning.py:299` |

这是 Open-Jev 与 RYOTIDE 在**置信度来源**上的根本分野：RYOTIDE 的置信度是**读出来的**，
只有 temperature 能改数值；Open-Jev 的置信度是**训练出来的**——软目标交叉熵等价于让模型
拟合一个参考分布，Brier 项额外惩罚分布的平方距离。temperature 是训练之后的二次校正。

优化器分两组学习率（`train.py:397-400`）：LoRA 用 `args.lr`，决策头用 `args.head_lr`；
梯度裁剪 1.0；任何非有限 loss 或 grad 直接抛出（`train.py:417-424`，`Nonfinite loss; run invalid`）。

**评估与温度**：

- `metrics.py:94-164` `evaluate_probabilities`：`n_bins=15`（`metrics.py:98`）等宽分箱的
  top-label ECE，另给 NLL、Brier、expected_brier 与逐阈值 coverage。注意 RYOTIDE/vendor
  JevBench 用的是 **10 箱**——两边的 ECE 数字不可直接互比。
- `metrics.py:167-226` `fit_temperature`：在 `log T ∈ [log 0.05, log 20]` 上取 25 点网格，
  再做 24 步黄金分割，最小化 calibration split 的交叉熵。
- `train.py:443-447` **无条件写入**拟合结果：

  ```python
  temperature = fit_temperature([r["logits"] for r in cal], [r["target"] for r in cal])
  (out / "checkpoint/temperature.json").write_text(json.dumps({
      "temperature": temperature, "split": "calibration", "n": len(cal), ...
  ```

  这里**没有接受/拒绝门槛**。对照 RYOTIDE §9.2：它预先注册了「留出集 ECE 必须下降才采纳」的规则，
  因此 Qwen3.5-4B 拟合出 1.592 却保持 1.0。Open-Jev 当前把拟合值直接发布。
  实际发布的 2B 值是 **1.7360720317205194**（§10 两条实例都来自这份 checkpoint）。
  `train.py:475` 有一句克制的声明：*"Calibration temperature is not a held-out capability or
  safety result"*。

**可恢复训练**：`train.py:293-296` 要求 40 位 revision；快照写入 `.incomplete-*` 后原子重命名；
run identity 绑定数据字节、行 ID/顺序、优化器设置、相关实现文件哈希与运行时版本
（`docs/training.md`）。快照标 `inference_ready: false`，不含 `temperature.json`。

## 7. 前缀缓存：两级 LCP 与混合架构的分支

`prefix_cache.py` 是性能主体，思路与 RYOTIDE §7「多问题共享一次 state」一致但更细。

**两级复用**：

1. `shared_length = common_prefix_length(sequences)`（`prefix_cache.py:212`）——所有候选的公共
   前缀，即 `Context:` 段；
2. 每个问题内部再算 `prefix_length`（`prefix_cache.py:253`）——该问题候选之间的公共前缀，
   即 `Context + Question` 段。

`common_prefix_length`（`prefix_cache.py:42-51`）强制 `limit = min(len) - 1`，**至少留一个 token
让分支去扩展**；且 LCP 永远在**完整 chat template 分词后**的 token 序列上求，绝不在单独分词的
字符串上求。每个候选只前向自己的增量后缀（`prefix_cache.py:292`），右填充成 batch，按各行真实
长度 `lengths-1` 读（`prefix_cache.py:296-299`）。

**难点是 Qwen3.5 的混合架构**：既有全注意力 KV（`DynamicLayer.keys/values`），又有线性注意力层
（gated delta）的 `conv_states` / `recurrent_states`——后两者是**就地更新**的。`fork_cache`
（`prefix_cache.py:54-86`）因此手写：

```python
if type(layer) is DynamicLayer:                      fields = ("keys", "values")
elif type(layer) in (LinearAttentionLayer, RecurrentDtypeLayer):
                                                     fields = ("conv_states", "recurrent_states")
...
setattr(cloned, name, tensor.repeat_interleave(batch_size, dim=0))
```

注释直说了为什么不能用 `DynamicCache.batch_repeat_interleave`：transformers 5.10.2 里
`LinearAttentionLayer` 没实现它；而浅拷贝或只拷 KV 会让递推状态**别名（alias）**，
把后续候选污染掉。这与 RYOTIDE §8 里 `ArraysCache`/`RotatingKVCache` 的两个 CAS 坑是同一类
问题的不同模型版本。

`RecurrentDtypeLayer`（`prefix_cache.py:19-32`）是一个精度补丁：transformers 5.10.2 用
**conv dtype**（bf16）分配递推状态，而 delta-rule 状态本应是 FP32；每次前缀切分都会把 FP32
状态舍入一次。子类强制保留 kernel 输出的 dtype。

**而且他们诚实地标记了它不等价**（`docs/prefix-caching.md`）：2B/H100 上 11 个负载里 **9 个
超出预注册的 `1e-4` 概率误差容限**，最大 `0.005703`；选中的决策全部一致，但概率不被认为等价。
因此 `--prefix-cache` **默认关闭**（`predict.py:22-23`、`serving.py:129`）。这与 RYOTIDE 用
`marker_mass` 让「读错位置无法静默发生」是同一种工程态度。

## 8. Fail-loud：这批代码最鲜明的风格

几乎每个「看起来能跑」的错误路径都被显式拒绝：

| 场景 | 位置 | 行为 |
|---|---|---|
| 输入超长 | `model.py:239-240` | `no silent truncation` 硬报错 |
| 概率非法 / 和不为 1 | `api.py:106-110` | 直接抛 |
| 后端返回错误行数 | `serving.py:85-90` | `backend returned the wrong number of rows` |
| 前缀缓存用于不支持的后端 | `prefix_cache.py:104-105` | 明确抛错，**不静默回退** |
| 缓存层类型未知 | `prefix_cache.py:72-73` | `Unsupported prefix cache layer` |
| 缓存被 offload | `prefix_cache.py:64` | 拒绝 |
| 缓存序列数不为 1 | `prefix_cache.py:79-80` | 拒绝 |
| 未填充的前缀分词含 padding | `prefix_cache.py:125-126` | 拒绝 |
| 训练出现非有限 loss/grad | `train.py:417-424` | `run invalid` |
| checkpoint 重载误差 > 0.05 | `train.py:462-464` | 拒绝 |
| **模型未真正加载完** | `model.py:149-179` | 拒绝启动服务 |
| meta 参数残留 | `model.py:215-224` | `refusing to serve unloaded meta parameters` |

`model.py:149-179` 那段注释是一条完整的踩坑记录：`device_map="auto"` 是**均衡**而非填满优先，
18 GiB 权重分到两张卡后**还有 12 个模块落到了 disk**，训练好的决策头留在 `meta`，LoRA 适配器被
拷进 meta 参数（约三分之一从未生效）——**而服务仍然对 `/health` 回 "ready"，并且会从一个未加载
的模型返回自信的数字**。这就是「看似合理却是错的」那一类失效，和 RYOTIDE 用 `marker_mass` 抓
`<think>` 读错位置是同一个主题。

**那 18 GiB 不是本篇走查的那一档**：`model.py:83-88` 把这一档的 bf16 权重逐项拆过账——
`model.language_model` 12.89 GiB + embeddings 1.89 GiB = 14.78 GiB 会活过 `del full`，视觉塔 / `mtp` /
`lm_head` 合计 3.19 GiB 被丢掉，**全量 17.98 GiB**。这是 27B 那一档（`ZefanCai/Open-Jev-27B-v1.1` 的基座
`Qwen/Qwen3.8-27B`，27,781,427,952 参数 ⇒ bf16 ≈ 55.6 GB），而本文的实例走查用的是 2B（2,274,069,824 参数
⇒ bf16 4,548,139,648 B ≈ 4.55 GB）。引自 `model.py:149-179` 的那句注释讲的是**多卡切分在 27B 上**
才会遇到的失效，**不要把它读成 2B 的体量**。

## 9. 四条不变量对照

把 RYOTIDE §13 的四条不变量逐条落到 Open-Jev：

1. **读位置必须落在标记上** —— 概念不适用。这里没有「标记 token」，打分量是 head 对隐状态的
   标量输出，不是词表分布，因此没有 `marker_mass` 这个诊断量。等价的守卫是概率校验（§2）
   与 `max_length` 硬报错。
2. **前缀与标记表面形式耦合** —— 不适用。没有 `SUFFIXES`/`PATTERNS` 扫描，**也没有运行期标定**；
   `Yes or No` 是固定问句。Open-Jev 少掉了一整块复杂度，代价是引入了训练。
3. **只读最后一个位置** —— 同样成立（`model.py:261`、`prefix_cache.py:298`）。
4. **比较概率前先确认 temperature** —— 同样成立，且落地方式一致：`temperature.json` 随 checkpoint
   发布（`train.py:444-447`），推理时写进响应 `metadata.temperature`（`serving.py:94`）。
   §10 实例② 会把这条不变量的代价具体化：同一条记录在 T=1.736 与 T=1 下的顶部概率是
   **0.107 与 0.576**。

## 10. 实例走查：同一条流水线上的两个候选规模

下面用两条**真实存盘记录**各走一遍完整处理过程。分界取在**候选规模**——这正是 §4 的
pointwise 打分与 RYOTIDE 单 token 标记方案分道扬镳的地方：

| | 实例 ① | 实例 ② |
|---|---|---|
| 记录 | `formal_logic-01b` | `examples/community/tariff_255.json` |
| 候选数 | 4（`choice`） | **255**（`choice`，接口上限） |
| 候选形态 | 单字母键 + 语义描述 | `synthetic_000`…`synthetic_254` + 描述 |
| 模型 | `Qwen/Qwen3.5-2B`，`lora_decision_head` | 同 |
| temperature | 1.7360720317205194 | 同 |
| 结果 | 模型答 `D`，gold 是 `C`（**错**） | 答 `synthetic_173`（**对**） |

实例 ① 走完整六步；实例 ② 只走**与 ① 不同的那几步**，其余逐字相同。

### 数据来源（务必先读）

- **记录了温度的那条声明**：两条实例来自**同一份 checkpoint**
  （`checkpoint_sha256` = `1d3c0e4e37bf04731edee447363ddc02d084a9c1f10ab829ac4d424f7d7c413b`，
  `base_revision` = `15852e8c16360a2fea060d615a32b45270f8a8fc`），因此两边概率可比。
  实例 ① 取自 `reports/pilot-frontier-n1-16k/2b/frontier/outcomes.jsonl`（`max_length` 16384），
  实例 ② 取自 `reports/integration/2b-api-check.jsonl`（`max_length` 4096）——**两者的
  `max_length` 配置不同**，但它们都远未触及上限，不影响本条记录。
- **prompt 是本次实跑得到的**，不是复述：用 `jev.api.compile_request` +
  `jev.api.candidate_prompts` 对实例 ① 的原始请求直接调用。这两个函数是纯 Python
  （只依赖 `jev.metrics`），不需要 torch。
- **概率读自已存盘结果**，不是本次重新前向。本机没有 `torch` 与 `transformers`，**没有**重新
  加载模型或分词器，因此本文**不声称**任何 token 数或 token id——只声称「这两条记录的结果如此」。
- confidence 与温度换算两项数值是本文**现算**的，脚本见 §10 末尾。

### 实例 ①：`formal_logic-01b`，4 个候选

#### (a) 输入记录

```json
{
  "model": "open-jev",
  "state": {
    "premises": ["not P"],
    "query": "P",
    "semantics": "Classical Boolean logic. not, or, and have their Python meanings; != means exclusive-or for Booleans; == means equivalence."
  },
  "questions": {
    "answer": {
      "type": "choice",
      "instructions": "Classify the query: true in all satisfying assignments = entailed; false in all = contradicted; both possibilities = undetermined; no satisfying assignments = inconsistent_premises.",
      "criteria": {
        "A": "undetermined", "B": "inconsistent_premises",
        "C": "contradicted",   "D": "entailed"
      }
    }
  }
}
```

注意一个容易读错的地方：`criteria` 的**键是 `A`–`D`（真正的 `answer_keys`）**，值是语义描述。
`state` 是对象而非字符串，所以它会被 `sort_keys=True` 确定性渲染。

#### (b) 编译后的 record（实跑输出）

```
kind         choice
id           answer
answer_keys  ["A", "B", "C", "D"]
options      ["A: undetermined", "B: inconsistent_premises",
              "C: contradicted", "D: entailed"]
```

`answer_keys` 与 `legend` 是软件侧字段——`candidate_prompts` 不会把它们传给模型。

#### (c) 真实 prompt（实跑 `candidate_prompts` 输出，4 条）

```
Context:
{"premises": ["not P"], "query": "P", "semantics": "Classical Boolean logic. not, or, and have their Python meanings; != means exclusive-or for Booleans; == means equivalence."}

Question: Classify the query: true in all satisfying assignments = entailed; false in all = contradicted; both possibilities = undetermined; no satisfying assignments = inconsistent_premises.
Proposed answer: A: undetermined
Is this proposed answer correct? Answer Yes or No.
```

后三条把 `Proposed answer:` 一行依次换成
`B: inconsistent_premises` / `C: contradicted` / `D: entailed`，其余**逐字节相同**。

三点在这里同时可见：

1. **`state` 在每条候选里都完整重复一次**。这是 pointwise 的代价：4 个候选 = 4 份 context。
   也正因如此，§7 的 `shared_length` 前缀复用才成为必需而非优化。
2. **prompt 里没有顺序信息**——没有 `A.`/`B.` 的枚举前缀，也没有「以下 4 个选项中」。候选之间
   唯一的关系是它们共享同一个 `Context + Question` 前缀。这就是 §4 所说「顺序不是变量」的字面形态。
3. **末尾是 `Answer Yes or No.`**——固定问句，不随模型或候选数变化。RYOTIDE 靠 §3 的标定去
   探测「读哪个位置」；这里直接命令模型进入 Yes/No 位置，且 `enable_thinking=False`
   （`model.py:234`）保证叙事/推理块不会吞掉这个位置。

#### (d) 打分与归一化

```
4 条序列 → forward() → 每条取 last_hidden_state[lengths-1]
         → head(·) → 4 个标量 s_A, s_B, s_C, s_D
         → softmax([s_A..s_D], T=1.7360720317205194)     serving.py:92
```

#### (e) 真实结果

```
model        Qwen/Qwen3.5-2B    method  lora_decision_head
choice       D        gold  C        correct  false
probabilities  A 0.2188917927851033   B 0.19540758564593178
               C 0.2660079848525914   D 0.3196926367163735
confidence   0.09292351562183132
usage        {input_tokens: 490, output_tokens: 0}   elapsed 167 ms
metadata     temperature 1.7360720317205194   candidate_sequences 4
             inference_seconds 0.15950932539999485   max_length 16384
             code_commit fb9d51731f1d2e720bfc44dd58ee4edf0adfc875
```

四处值得对照代码读：

- **`output_tokens: 0`**：整个决策没有生成任何 token。这是「读」而非「写」的字面含义。
- **gold 是 `C`，模型答 `D`，而且错得很有信息量**：`not P` 与 `P` 中，`P` 在所有满足赋值里
  都为假，正确答案是 `C: contradicted`；模型选了 `D: entailed`，即把「被否定」读成了「被蕴含」。
  四个概率是 `[0.219, 0.195, 0.266, 0.320]`——**接近均匀**。confusion 在分布形状上可见，
  这正是 `confidence` 这个量存在的意义：0.093 表示「几乎不比瞎猜好」。
- **`confidence` 公式验算**：`(0.31969 − 0.25)/0.75 = 0.09292`，与记录的 `0.09292351562183132`
  吻合（`metrics.py:69-79`）。
- **`input_tokens: 490`，4 个候选 = 每个约 122 token**。实测字符数：`state` 渲染后 177 字符、
  `instructions` 181 字符，单条候选 prompt 459–472 字符，四条合计 1857 字符——约 3.8 字符/token，
  正符合 §10(c) 的「同一段 context 重复 4 次」形态。这也是 §4 所说 N 倍前向代价的直接度量。

#### (f) 温度还原：谁是「原始分布」

记录里的 `temperature` 是 **1.7360720317205194**，不是 1.0。按 §6，发布概率是
`p = softmax(s/T)`，因此还原 T=1 的分布就是 `normalize(p^T)`：

| 候选 | 已发布（T=1.736） | 还原 T=1 |
|---|---|---|
| A undetermined | 0.2189 | 0.1940 |
| B inconsistent_premises | 0.1954 | 0.1593 |
| C contradicted（gold） | 0.2660 | 0.2722 |
| **D entailed（选中）** | **0.3197** | **0.3745** |

两列各自都对，**放在一起比就是错的**。argmax 不变（`D` 在两列都是最大），这正是 RYOTIDE §9.2
所说的「temperature 只能改置信度、不能改 argmax」在 Open-Jev 上的同一个性质——但在这里
temperature 不是唯一改数值的环节，**训练本身也改数值**；T 只是训练之后的二次校正。

### 实例 ②：`tariff_255`，255 个候选

#### (g) 输入与边界

`examples/community/tariff_255.json` 是一个**接口规模压测**用例（`state.adapter` =
`tariff_synthetic_scale_stress`，`notice` 明示 *"These are not real tariff codes or customs advice"*）。
它的 `criteria` 有 **255 个键**：`synthetic_000` … `synthetic_254`，每个值是
`Synthetic group N; not an official tariff code`。

255 正是硬上限（`api.py:48`）：

```python
if not isinstance(criteria, Mapping) or not 1 <= len(criteria) <= 255:
    raise ValueError("Choice requires between 1 and 255 candidates")
```

prompt 骨架与实例 ① **逐字相同**，只是候选块多出 251 条；`state` 也很小（约 244 字符），
描述 `"The mock catalog explicitly assigns this item to synthetic group 173."`。

#### (h) 真实结果

```
model        Qwen/Qwen3.5-2B      method  lora_decision_head
choice       synthetic_173   ← 与 state 里声明的 group 173 一致
confidence   0.1035312339614464
top4         synthetic_173 0.107047 | synthetic_017 0.021998
             synthetic_001 0.008707 | synthetic_073 0.008668
非零候选数   255 / 255           概率和 1.0000000000000002
usage        {input_tokens: 30235, output_tokens: 0}   elapsed 0.737 s
metadata     temperature 1.7360720317205194   candidate_sequences 255
             inference_seconds 0.7330010579898953   max_length 4096
             code_commit bfd53dd896963b55548ea4e5402e53830196be50
```

五处值得对照代码读：

- **一步打分，没有第二步**。RYOTIDE 在 >26 选项时必须切换到两 token 代码 `A0`–`Z9` 并做两步读
  `P(字母) × P(数字|字母)`（或者用近似的 `grouped` 读法）。这里 **255 个候选仍是一次
  `softmax`**：候选是自由文本，不需要被压缩进任何标记方案。这是两条路线最尖锐的对照。
- **`confidence` 只有 0.1035，但答案是对的**。按 §5 的公式，
  `(0.107047 − 1/255)/(1 − 1/255) = 0.10353`，与记录吻合。255 个候选下的均匀分布是 0.0039，
  所以模型确实显著高于均匀；但绝对概率只有 0.107。**低置信 ≠ 错误**——这正是 RYOTIDE §9.6
  那条立场的另一侧证据：低置信标记的是「不值得据以行动的决策」，不是「可以翻转的答案」。
- **255 个概率全部非零**（和 = 1.0000000000000002）。没有掩码、没有截断，输出是真正封闭在
  255 维单纯形上的分布。`api.py:106-111` 的校验正好覆盖这个精度尾差（`abs_tol=1e-6`）。
- **成本**：`input_tokens: 30235` ≈ 每个候选约 119 token，与实例 ① 的每候选 122 token 一致
  （候选自身长度不同而已）。**255 个候选只用了 0.737 s**，而实例 ① 的 4 个候选用了 0.167 s——
  序列数是 64 倍、时间是 4.4 倍，差距来自 `serving.py:82` 的有界批处理
  （`--batch-size` 默认 32 → 255 个候选切成 8 批）与 §7 的前缀复用。
- **`max_length` 是 4096**（实例 ① 是 16384）。这条记录没有触及上限，但它是配置指纹的一部分，
  与 `checkpoint_sha256`、`temperature`、`code_commit` 一起写进每条响应的 `metadata`
  （`serving.py:150-159`），使「这条数字是哪个配置算出来的」始终可追溯。

#### (i) 温度还原（同 §10(f) 的方法）

| 候选 | 已发布（T=1.736） | 还原 T=1 |
|---|---|---|
| **synthetic_173（选中，正确）** | **0.107047** | **0.575983** |
| synthetic_017 | 0.021998 | 0.036931 |
| top1/top2 比值 | 4.87 | 15.60 |

这一条把不变量 #4 的代价放大了：**同一条记录、同一个 argmax，顶部概率可以是 0.107 或 0.576**。
在 255 个候选上，T>1 的压平效应远比实例 ① 明显（实例 ① 只从 0.3197 → 0.3745）。
任何关于「这个模型在 255 选项中是否自信」的结论，都必须先确认记录里的 `runtime` 温度。

### (j) 复现本文的数值（纯 Python，无需 torch）

```python
import json, sys
sys.path.insert(0, ".")
from jev.api import compile_request, candidate_prompts
from jev.metrics import choice_confidence

T = 1.7360720317205194

# 实例① 的 prompt 是实跑的
req = json.loads(open("reports/pilot-frontier-n1-16k/2b/frontier/outcomes.jsonl").readline())["request"]
print(candidate_prompts(compile_request(req["state"], req["questions"])[0]))

# confidence 与温度换算
def restore(p, T):                      # normalize(p^T) 还原 T=1 分布
    raw = [v ** T for v in p.values()]; Z = sum(raw)
    return {k: v / Z for k, v in zip(p, raw)}
```

## 11. 文件对照

| 文件 | 职责 |
|---|---|
| `api.py` | 请求编译（三种问题类型 → 记录）、候选渲染、类型化响应与概率校验 |
| `model.py` | `DecisionModel`：backbone 加载、Yes−No 播种的决策头、LoRA、分片守卫、`forward` |
| `train.py` | 训练循环（软 CE + Brier）、分组学习率、可恢复快照、calibration 与温度拟合、held-out 评估 |
| `metrics.py` | softmax/sigmoid、两种 confidence、ECE/coverage/NLL/Brier、温度拟合 |
| `serving.py` | `Predictor`（批处理、校验、prompt_hash 式 provenance）、`TorchScorer`、`load_predictor` |
| `prefix_cache.py` | 两级 LCP、混合架构的 cache 分支、请求内前缀复用（实验性，默认关） |
| `predict.py` | 单文件请求的 CLI 入口 |
| `server.py` | HTTP 服务（`/v1/systemone`、`/health`） |
| `client.py` | 标准库客户端（校验响应问题 ID 与请求一致） |
| `fast_backend.py` | `triton-tail` / `fast-cuda` 加速后端与最终行 RMSNorm/head 融合 |
| `kernels/final_score.py`、`kernels/triton_tail.py` | 原生的 Triton final-row 内核 |
| `third_party/open_jev_fast/` | 归属的第三方快速推理树（CUDA 内核限 27B profile） |
| `sharded_model.py`、`inference_initialization.py` | 多卡分片与推理初始化 |
| `data.py` | 合成数据生成（四条规则族）、split 划分、验证、导入外部分类数据 |
| `case_*.py`、`community*.py`、`*_controls_v*.py` | 各领域的决策数据集与控制实验 |
| `docs/prefix-caching.md` | 前缀缓存的误差声明与对比协议 |
| `docs/training.md` | 可恢复训练的身份绑定与快照格式 |

## 12. 亲手跑一遍

```bash
# 起服务（无 checkpoint → 零训练的 Yes-minus-No 基线）
python -m jev.server --model Qwen/Qwen3.5-2B \
  --revision 15852e8c16360a2fea060d615a32b45270f8a8fc --max-length 4096

# 起服务（加载训练好的 checkpoint，带 calibration 温度）
python -m jev.server --checkpoint /path/to/checkpoint --max-length 4096 --batch-size 32

# 单文件请求：state + questions → 类型化概率
python -m jev.predict --checkpoint /path/to/checkpoint \
  --request examples/community/tariff_255.json

# 开前缀缓存做 A/B（默认关闭，见 §7 的误差声明）
python -m jev.predict --checkpoint /path/to/checkpoint \
  --request examples/email-selection.json --prefix-cache

# 训练（软 CE + 0.1 Brier）
python -m jev.train --model Qwen/Qwen3.5-2B \
  --revision 15852e8c16360a2fea060d615a32b45270f8a8fc \
  --data data/your-validated-mixture --output runs/2b \
  --train-rows 0 --steps 27000 --accumulation 4 --max-length 4096
```

看结果里 `usage.output_tokens` 是否为 0，以及 `metadata.temperature` 是多少——这两个数就能
确认「读」确实发生了，以及你手上的概率是哪个温度下的。

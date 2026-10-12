# Kev 是如何把「用户给的选项」直接读成一次前向的分布的

> 代码库：<https://github.com/jaredpalmer/kev>
> 对应 README 第 2 行：*Small Jev-like decision models you can train and run yourself.*
> 与 README 第 318 行（How It Works）：*The pointer head scores each option's `</opt>` hidden state
> against the question's `<decide>` hidden state. A softmax turns those scores into probabilities.*
> 本文按 `kev/` 的实际代码逐环节说明这句话的实现方式；文中出现的文件名与行号（`model.py:87`、`api.py:102`、
> `serve.py:216` 等）均相对该仓库根目录。
>
> **同名配对文档**：RYOTIDE 的 [`docs/ryotide/typed-decision-model.md`](../ryotide/typed-decision-model.md)。
> 本文凡称「RYOTIDE §N」均指该篇的节号；§0 先讲清分歧，§10 把它的四条不变量逐条落位。
>
> **与 `docs/openjev-hf/` 那篇的区别**：那篇也是「Qwen3.x + 一个头」，但它读的是**第一个输出位置的 52 个字母
> token**，词表维度是存在的，只是被 `allowed_token_ids` 掩掉。Kev 走的更彻底一步：它加载骨干时用
> `AutoModelForCausalLM.from_pretrained(...).model` **直接丢掉 vocab head**（`model.py:257`，注释是
> *backbone only (no vocab head): we never generate text*），所以这条路径上**连一个 `[1, n_tokens, vocab]`
> 的张量都不会被算出来**。这是十篇里唯一一条「读法在词表之外」的实现。

与另外九篇最大的不同有两条，前面先说清：

1. **「typed」不由分词器给，也不由读位置给，而由用户声明的选项跨度（span）给。** 每个选项是
   `<opt> … </opt>` 包起来的一**段** token，长度不限、可多 token（`model.py:110`）。候选集不是从词表里
   挑出来的 K 个 id，而是请求里 `criteria` 的顺序（`api.py:94-99`）。
2. **因此没有「掩码 + 重归一化」这一步。** RYOTIDE 是因为读出了整个词表分布、必须把质量收回到标记上才需要
   重归一化；Kev 的 softmax 直接作用在头给出的**恰好 K 个标量**上（`model.py:216-218`），分布天然封闭。
   §5 展开。

## 本文的事实来源

| 类别 | 来源 | 本文中的标记 |
|---|---|---|
| 代码引用 | `kev/model.py`、`kev/api.py`、`kev/serve.py`、`kev/predictors.py`、`kev/metrics.py`、`kev/train.py`、`kev/data.py`、`kev/checkpoint.py`、`kev/anchors.py`、`kev/calibrate.py`、`scripts/calibrate_checkpoint.py` | `文件:行` |
| 可实跑得到的产物 | `runs/release/kev-1.0-verify/hub-{4b,0.8b}/rows.json`、`runs/rel9-public/1-jaredpalmer_kev-9b/report.json`、`runs/rel27-public/{1,2}-jaredpalmer_kev-27b/`、`runs/r4-kev-4b-v4-one/report.json`、`runs/r15-08b-a-docs/rows.json`、`runs/fam-4b-breadth/rows.json`、`docs/claims.json`、`runs/release/kev-1.0-manifest.json` + `kev-1.0-SHA256SUMS.txt` — 都是 **committed 的 JSON/文本**，容量小、纯数据、可直接重算 | 标注「实测」，并注明是**在 committed 产物上重算 / 对照** |
| 上游记录的数字 | `README.md`、`docs/model-cards/kev-*.md`、`docs/releases/kev-1.0.md`、`AGENTS.md`、各 `runs/*/report.json` | 引原文并给出 `文件:行`，或给出 run 标签与 temperature |
| 未跑 | **本机无权重、无 GPU、无 checkpoint，本文没有跑过任何前向** | 全文**不给任何由本机前向产生的概率数值**；所有概率数字都来自上游 committed 的记录，并标 run 与 T |

本文所有「实测」只有两类动作：**在 committed 的 rows/report 上重算公式**（§12.2、§13.3），
**把 committed 的模型卡与发布清单的 sha256 对照**（§8.4）。两类都不需要权重，也不运行上游的任何程序——
`kev.*` 一个模块都没有 import 过。这一点在 §12 会把命令原样列出。

---

## 0. 与 RYOTIDE 的分野

| | RYOTIDE | Kev |
|---|---|---|
| 决策机制 | 读**标签 token** 在末位的词表 logits → 掩码到标记 → softmax | 读**每个选项自己的 `</opt>` 位置**的隐状态，与 `<decide>` 位置做点积 → **恰好 K 个标量** → softmax（`model.py:216-218`） |
| 需要训练 | 否，权重全程 frozen | **是**：LoRA + 一个 `PointerHead`（`model.py:205-223`、`model.py:276`）；Kev-27B 是整骨干全量微调 |
| 是否物化词表维度 | 是，`marker_mass` 就是对它的诊断 | **否**：骨干不带 vocab head（`model.py:257`） |
| 「typed」来自 | 分词器：标签必须单 token，否则映射成字母标记 | 请求里 `criteria` 的**顺序**；每个选项是一段 `<opt>…</opt>` 跨度，长度不限（`model.py:110`、`api.py:94-99`） |
| 候选数上限 | 260（`A0`–`Z9` 两步读；>26 另有 `grouped` 近似读法） | **255**，但这是 **API 契约的上限**不是分词器的上限（`api.py:14` 的 `MAX_OPTIONS`，`api.py:30`、`api.py:37` 两处校验）——选项是跨度，撞不到词表 |
| 读位置怎么定 | **运行期标定**：扫 9 前缀 × 2 表面形式，取 `marker_mass` 最高 | **训练期固定**：位置是 `encode` 在构造序列时写下的下标（`model.py:121-125` 的 `decide_idx` / `opt_idx`），运行期没有任何标定 |
| 顺序偏置 | 多 order 共享前缀 fork，折叠回真实标签后**概率平均** | **训练期打散**（每 epoch 重排 + 随机插 none/distractor，`data.py:341`、`data.py:333-340`），另有一个**构造性不变**的开关 `option_isolation`（`model.py:116-118`），运行期只**度量**（§6） |
| 置信度来源 | **读出来的**：只有 temperature 能改数值 | **训出来的**：头的 q/k 权重；temperature 是训练之后施加的单调缩放，改数值不改 argmax（§9） |
| 诊断量 | `marker_mass` | `raw_probability_sum` + `zero_count`（每条记录，`benchmark.py:61`）、`permutation.{n,mean_max_delta,flip_rate}`（每次 benchmark，`benchmark.py:92-93`）、`/v1/models` 里的 prefix-cache 与 graph 计数（`serve.py:308-311`） |
| 多问题共享 state | `run_many()`：公共前缀预填充一次 | 三段并存：packed block-causal 掩码 / 行形式 / 运行期的 state 前缀 LRU（`model.py:316-324`、`model.py:404-414`、`serve.py:38-90`）（§7） |
| 服务化 | FastAPI `/v1/systemone` + `/health` | FastAPI `/v1/systemone` + `/v1/models` + 两个 demo 端点；`422` 拒绝而不是静默截断（`serve.py:139-142`）（§8） |
| 新增维度 | — | ①`date_facts` 确定性预处理（`api.py:66-91`）；②MLX 第二后端（`mlx_model.py`）；③发布数字的可追溯清单 `docs/claims.json` + `scripts/verify_claims.py`（§14） |

一句话立场：**Kev 把 RYOTIDE 那条「读词表再掩码」的读法整个换掉，改成「读用户给的每一段选项」**——
它付的代价是必须**训一个头**（位置和打分函数不再免费），换来的是候选集完全不受分词器约束、
且读位置在构造期就写死、运行期没有任何可标定的自由度。

---

## 1. 端到端数据流（一次请求）

```
POST /v1/systemone                                   serve.py:249-252
  {state, model, questions:{id:{type,instructions,criteria}}}
        │
        ▼
SystemOneRequest (pydantic)                          api.py:43-46
  · questions 至少 1 个（Field(min_length=1)）
  · Choice 校验 1..255 个选项；Score 校验 1..255 级   api.py:28-37
        │
        ▼  prepare()：只有 KEV_DATE_FACTS=1 时才改 state   serve.py:223-225, api.py:84-91
        ▼
to_record(req)                                       api.py:102-117
  · render() 把 object/array 摊成带字段名的文本        api.py:49-55
  · choice → [option_text(k,v)…]；noul → ["no","yes"]
    score → [render(x)…]                             api.py:108-115
  · meta 记下每题的 {id, type, keys, legend}          api.py:107
        │
        ▼  请求线程把 record 交给唯一模型线程（每批 ≤64 条）  serve.py:132-145, 158-160
admit(model, tok, rec)                               model.py:130-142
  · state > 65,536 token → ContextOverflow → HTTP 422（除非 KEV_TRUNCATE_STATES=1）
  · 任一题的 row（state+该题分支）> 73,728 → 422
        │
        ▼
encode(tok, rec)                                     model.py:87-127
  [ <state> …state… | <q> instr <opt> o1 </opt> <opt> o2 </opt> … <decide> | …每题一段 ]
        │
        ▼  模型线程：model.probs_batch(encs, prefixes, keep)   model.py:464-492
前向（三选一，见 §4/§7）：
  A. packed block-causal 掩码，一次读所有题               model.py:316-324, 326-327
  B. 每题一行因果序列（混合骨干唯一可选）                 model.py:364-383
  C. 行 + state 前缀 KV 缓存（命中时只跑分支）            model.py:404-414, 446-462
        │
        ▼
PointerHead：z[j] = (k(h_opts[j]) · q(h_decide)) / sqrt(dp)   model.py:216-218, 220-223
  非训练且 temperature≠1 时 z ← z / temperature               model.py:218
        │
        ▼  softmax(z, -1)，K 个标量 → 分布                   model.py:402, 413, 444, 462, 505
        │
        ▼
to_answers(probs, meta)                              api.py:149-160
  · noul   → {"noul": round(p[1],4)}
  · choice → {"choice": argmax(p), "confidence": (p_max−1/K)/(1−1/K), "probabilities": {…}}
  · score  → {"score": Σ i·p_i, "legend": …, "confidence": …, "probabilities": {…}}
        │
        ▼
Server._body                                          serve.py:211-220
  usage.input_tokens  = 编码后的全部 token（含 state）  serve.py:189
  usage.output_tokens = **序列化后答案 JSON 的 token 数**（不是生成的 token，见 §9.4）
  latency_ms          = 本条请求所在**那一批**的模型时间
```

---

## 2. 「typed」从哪来

### 2.1 候选集 = 请求里 `criteria` 的顺序

`to_record` 把三件事摊平（`api.py:102-117`），`question_keys` 决定答案用什么键回报（`api.py:94-99`）：

| 题型 | options 文本 | 回报的键 |
|---|---|---|
| `noul` | `["no", "yes"]`（可带描述：`"no: …"`） | `["false", "true"]` |
| `choice` | `[f"{name}: {desc}" …]`，desc 为空则只有 name | `criteria` 的键，**按请求顺序** |
| `score` | 每级的文本 | `["0","1",…,"L-1"]` |

三者共用同一个内部表示 `{"state": str, "questions": [{"instr","options","label"}]}`（`api.py:117`），
训练数据也走同一函数（`data.py:397-399`，其文档串第 1-2 行：*so the training format is byte-identical to
what /v1/systemone feeds the model*）。**训练与服务的文本格式是同一个函数产出的**，这不是约定，是同一行代码。

### 2.2 选项是一段跨度，不是一个 token——所以分词器不参与「typed」

`encode` 为每个选项写一段 `[<opt>] + user_tokens(o) + [</opt>]`（`model.py:110`）。
选项文本多长都不影响候选集，唯一的上限是 API 契约的 255（`api.py:14`）。
这与 RYOTIDE（标签必须落成单 token，否则降级成字母标记）、OpenJev(HF)（52 个字母 token）是两种完全不同的
封闭方式：**前者把标签集封在分词器上，Kev 把标签集封在头的输出个数上**。

### 2.3 五个分隔符是复用 Qwen 的闲置特殊 token

`SPECIAL = ["<|fim_prefix|>", "<|fim_middle|>", "<|box_start|>", "<|box_end|>", "<|fim_suffix|>"]`
依次作 `<state> <q> <opt> </opt> <decide>`（`model.py:11`、`model.py:104`、`model.py:106`）。
理由写在注释里：*so no embedding rows need to be added/trained; LoRA adapts their meaning*
（`model.py:9-10`）。**分词器没有被扩充**——`AGENTS.md:412-413` 另记了一条工程理由：
peft 的 `trainable_token_indices` 在 MPS 上漏内存。

### 2.4 用户文本造不出分隔符（这一条是「typed」的守卫）

`user_tokens()` 在分词前把 `<|name|>` 改写成 `<¦name¦>`（`model.py:78-81`）：

```python
def user_tokens(tok, text):
    """Tokenize caller-supplied text so it can never produce delimiter/control tokens (option boundaries are unforgeable).
    The fast tokenizer ignores split_special_tokens, so `<|name|>` is rewritten to `<¦name¦>` before tokenizing."""
    return tok(_SPECIAL_RE.sub(r"<¦\1¦>", text), add_special_tokens=False).input_ids
```

state、instructions、选项文本三条路都过它（`model.py:101`、`model.py:109`、`model.py:110`）。
`AGENTS.md:416-417` 复述了同一条（并说明为什么不能用 `split_special_tokens`）。
所以**「选项边界不可伪造」不是靠约定，是靠一次正则改写**。README 第 284 行对外也是这么说的：
*Delimiter-like strings in user input are escaped before tokenization.*

---

## 3. 读位置

`encode` 返回的五个数组把「读哪里」写成了下标（`model.py:126-127`）：

| 字段 | 含义 |
|---|---|
| `decide_idx [Q]` | 第 q 题的 `<decide>` token 在整条序列里的下标 |
| `opt_idx [Q][K]` | 第 q 题第 k 个选项的 `</opt>` token 下标 |
| `opt` | 逐 token 的选项归属（`OPT_NONE=-1` 状态/指令，`0..K-1` 选项跨度，`OPT_DECIDE=-2`） |
| `seg` | 逐 token 段号（`0` = state，`k` = 第 k 题） |
| `pos` | 位置 id；**每题的分支位置从 state 结束后重新起算**（`model.py:114` 的 `p0 = len(S)`） |

`rows_of` 再把它们折成「行」的**分支内相对偏移**（`model.py:188-202`）：
`rows[k] = {"ids","pos","decide","opts"}`，其中 `decide` 与 `opts` 已经减掉了 state 长度。

三个结论：

1. **位置在构造期就定死，运行期没有标定**。整条 `kev/` 里没有「扫前缀找标记」这种代码——
   `SPECIAL` 是编译期常量，下标由 `encode` 算术算出。这与 RYOTIDE 的「9 前缀 × 2 表面形式扫描」
   形成最直接的对照：**Kev 把这个自由度用「自己拼 token 序列」换掉了**。
2. **`pos` 重启是有意义的**：每题位置从 state 之后重新起算（`model.py:114`、`model.py:120`），
   所以一道题的分支形状与它前面有几道题无关——这是 packed 形式能成立的前提。
3. **没有 `<decide>` 就没有分布**。softmax 的输入只有 `opt_idx` 那 K 个位置上的标量，
   `<decide>` 不产生 logit，它只是被点积的 query（`model.py:217`）。

---

## 4. 一次前向：读 K+1 个构造出来的位置，且不物化词表维度

### 4.1 打分函数

`PointerHead` 只有两个线性层（`model.py:205-223`）：

```python
class PointerHead(nn.Module):
    def __init__(self, d, dp=256):
        """dp = pointer dimension (head capacity knob)."""
        self.q, self.k = nn.Linear(d, dp), nn.Linear(d, dp)
        self.scale = 1 / math.sqrt(dp)
        self.temperature = 1.0

    def forward(self, h_decide, h_opts):  # [d], [K,d] -> logits [K]
        z = (self.k(h_opts) @ self.q(h_decide)) * self.scale
        return z if self.training or self.temperature == 1.0 else z / self.temperature
```

- `dp = 256` 是默认头容量（`model.py:206`，`DecisionModel.__init__` 的 `head_dim=256`，`model.py:244`）；
- 打分是 `<decide>` 的 query 与每个 `</opt>` 的 key 的**点积**，量纲被 `1/sqrt(dp)` 归一；
- **`temperature` 只在 `not self.training` 时生效**（`model.py:218`）——训练永远看 T=1。
  这条与 §9 的「温度是训练之后施加的」是同一件事的两面。

`many()` 是同一函数的批量版（`model.py:220-223`），服务端一次算完一题的所有选项；
`_readout_many` 更进一步，把一批请求的**全部题、全部选项**摊成一个 `[ΣK, d]`，
用 `index_put_` 一次读出（`model.py:494-505`）。两者都只是把同一个点积重排，不改变数值。

### 4.2 没有词表维度

`DecisionModel.__init__`（`model.py:257`）：

```python
self.lm = AutoModel.from_pretrained(weights, **load) if weights else AutoModelForCausalLM.from_pretrained(name, revision=revision, **load).model
```

取的是 `.model`——**丢掉 `lm_head`**。所以：

- 前向返回的是 `last_hidden_state`（`model.py:324`、`model.py:360`），不是 logits；
- 这条路径上不存在 `[1, n_tokens, vocab]` 张量，也不存在「词表上 renormalise」这个动作；
- 头的参数量是 `2·d·dp`（Qwen3.5-4B：2 × 2560 × 256 ≈ 1.3 M），与词表大小无关。

`LocalPredictor.__call__` 里返回给上层的是**这两组数**（`predictors.py:140-147`）：

```python
ps = [torch.softmax(z, -1).cpu() for z in logits]
zs = [z.float().cpu() for z in logits]
...
"probabilities": {...}, "logits": {...}, "inference_temperature": self.temperature, ...
```

即 `logits` 是**已被温度除过**的 K 个标量。这一条在 §9.5 会变成一个口径坑。

---

## 5. softmax 落在 K 个标量上：没有词表掩码

RYOTIDE 的三个动作——「读词表」「掩码到标记」「重归一化」——在 Kev 里**只剩第一个动作的等价物**，
而且是消失在构造里的：

| RYOTIDE 的步骤 | Kev 里对应什么 |
|---|---|
| 读末位词表 logits | 不需要：头直接产出 K 个标量（`model.py:216-218`） |
| 掩码到已知标签集 | **不需要**：`options` 就是头要打分的 K 段跨度，没有「第 K+1 个候选」这回事 |
| 重归一化 | **不需要**：`softmax` 的输出天然和为 1（`model.py:402` 等） |

唯一的 `-inf` 出现在**批内选项数不齐时的填充**上（`model.py:504`）：

```python
Z = torch.full((len(ks), max(ks)), float("-inf"), device=self.device).index_put_((own, sl), z)
return torch.softmax(Z, -1)[own, sl].cpu().split(ks)
```

这不是「掩掉词表」，而是把一批题铺成矩形时给短行补的位——补位随后被 `[own, sl]` 索引丢弃。
**它是批形状的产物，不是标签集的守卫**；同一题单独算就没有这个张量。

另一处掩码是**注意力**的，不是读出的：`branch_mask_batch`（`model.py:154-185`）实现
`attend(i,j) iff j<=i and (seg[j]==0 or seg[j]==seg[i])`，即**任何 token 只能看 state 和它自己那道题**。
`option_isolation=True` 时再叠一层：选项跨度内的 token 只能看 state、指令和自己那一段（`model.py:176-183`）。
两者都作用于注意力，不作用于 logits。

所以 §10 的 #1（读位置必须落在标记上）在 Kev 上是**概念不适用**——没有「质量落在标记上」这个量。
Kev 自己的守卫换成了**分布自检**：每条 benchmark 记录都写 `raw_probability_sum` 与 `zero_count`
（`benchmark.py:61`）。实测值见 §13.3。

---

## 6. 顺序偏置：训练期打散 + 一个构造性不变的开关 + 一个逐请求探针

这是 Kev 与别篇差别最大的一节，因为它同时有**三种不同性质**的东西，且**都不在发布卡片上给数字**。

### 6.1 训练期：选项顺序每 epoch 重排，K 本身也在变

`data.augment` 的文档串（`data.py:320-322`）：

```python
def augment(req, rng, p_none=0.1, p_none_distract=0.12, p_distract=0.15):
    """Choice only: permute option order (always); sometimes add a 'none of the above' option, either as the correct
    answer (true option removed) or as a wrong alternative (true option kept); sometimes add an irrelevant distractor."""
```

- 顺序**无条件**打乱：`keys = list(crit); rng.shuffle(keys)`（`data.py:341`）；
- `K` 会变：插入「none of the above」当作正确答案（真选项被删）或当作错误项（真选项保留），
  以及插无关干扰项（`data.py:333-340`，候选池在 `data.py:48-56`）；
- **每个 epoch 重新抽一次**：`record_variants` 按 `f"{epoch}:{id}"` 播种再调 `augment`（`train.py:295-296`）。

另外还有一条独立的置换项：`permuted_copy`（`train.py:29-35`）把每个 K≥3 的 Choice 题重排，
配一个对称 KL 项 `--perm_kl`（`train.py:336-338`、`train.py:359-362`）。
**但它的默认值是 0.0**（`train.py:381`），`--perm_frac` 默认 0.3（`train.py:382`）——即默认关闭（见 §14.2）。

模型卡把这一条对外写成（`docs/model-cards/kev-4b.md:149`）：*Option order is shuffled,
"none of the above" options and distractors are inserted at random*。

### 6.2 一个构造性不变的开关：`option_isolation`

`encode(..., option_isolation=True)`（`model.py:93-96`、`model.py:116-118`）把每个选项跨度变成
**自己的子分支**，所有选项跨度**共享同一组位置 id**，`<decide>` 落在最长跨度之后的一个固定位置：

```python
longest = max(len(sp) for sp in spans)
br_pos = (list(range(p0, p0 + len(instr)))
          + [p0 + len(instr) + i for sp in spans for i in range(len(sp))]
          + [p0 + len(instr) + longest])
```

配合 `branch_mask_batch` 的 `opts` 参数（`model.py:176-183`），每个选项**只看得到 state + 指令 + 自己**。
于是「按选项排列的表示」与「`<decide>` 在其上的注意」**按构造就是置换不变的**——
源码原话：*the per-option representations and `<decide>`'s attention over them are permutation-invariant by
construction*（`model.py:96`）。

代价写在 `model.py:263`：混合骨干**不支持**它（`option_isolation needs the packed mask`），
而当前全家族都是混合骨干（Qwen3.5 / Qwen3.8）。它也是**默认关**的（`train.py:395` 的 `--option_isolation`
默认 0）。所以这是一条**存在、可用、但没有发布**的路径——归入 §14.2。

### 6.3 运行期：只度量，且度量工具是公开的

三样东西：

1. **逐请求探针** `POST /v1/systemone/permute`（`serve.py:262-275`）：把一道 Choice 题跑 `n_perm`
   种顺序（默认 6，上限 64），返回每次的 `probabilities` / `choice`，外加
   `argmax_stable`（所有顺序 argmax 是否同一）与 `spread`（每个选项概率的极差）。
2. **每次 benchmark 的置换统计**（`benchmark.py:92-93`）：`permutation = {n, mean_max_delta, flip_rate}`，
   即对固定的 36（或 60/48）个探针题算「最大概率位移的平均」与「argmax 翻转率」。
3. **测试期顺序平均** `RotationAveraged`（`predictors.py:213-251`）：对每个 Choice 题跑前 `rotations`
   个循环位移，在 logits 上取平均（softmax 的几何平均 = 平均 logits 的 softmax，所以概率与 logits 保持一致）。
   成本是每记录 `rotations` 次前向，构造时要求 `rotations >= 2`。

### 6.4 发布卡片上**没有**翻转率

这是与 OpenJev(HF) 最刺眼的一处对照。OpenJev(HF) 的卡片直接写「18.5% → 2.3%」；
Kev 的模型卡只写一句定性的话（`docs/model-cards/kev-4b.md:283`、`README.md:434`）：

> **Option order** can change an answer; question isolation does not prevent this.

翻转率确实被算了出来，但**留在 runs 里**。committed 的发布读数给出了这些（§12.3 有重算命令）：

| 记录 | 检查点 | T（其 report 里记的） | 探针 n | flip_rate | mean_max_delta |
|---|---|---|---|---|---|
| `runs/rel9-public/1-jaredpalmer_kev-9b/report.json` | `jaredpalmer/kev-9b`（Kev 1.0，T 与卡片 2.19 一致） | 2.193649959389252 | 36 | **2/36 = 0.0556** | 0.0681 |
| `runs/rel27-public/1-jaredpalmer_kev-27b/report.json` | `jaredpalmer/kev-27b`（卡片 1.32） | 1.319507910772894 | 36 | **0/36 = 0.0** | 0.0322 |
| `runs/rel27-public/2-jaredpalmer_kev-27b/report.json` | 同上，更大的一批 | 1.319507910772894 | 60 | **1/60 = 0.0167** | 0.0288 |
| `runs/r4-kev-4b-v4-one/report.json` | `jaredpalmer/kev-4b`（**早于 1.0 的 T=2.1435**） | 2.1435469250725863 | 36 | 3/36 = 0.0833 | 0.0401 |

**前三行的温度与卡片一致，可以当作「发布版在发布温度下」的读数；第四行不是**——
它的 T=2.1435 与 Kev 1.0 的 2.4061 不同，那是同一个 Hub 名下的**更早一次**读数。
本表把这一列原样列出就是为了让这种错配当场可见（§5 / 其 §2）。

---

## 7. 多问题共享 state：三套机制，按骨干能力选

Kev 在这件事上有**三种实现**，选哪一种是**骨干能力决定的**，不是配置决定的。

### 7.1 packed + block-causal 掩码（纯注意力骨干）

一遍前向算完所有题（`model.py:316-324`）。可行性靠 `pos` 重启（§3）与 `seg` 掩码：
题与题之间互相看不见，state 每个 token 只看得到自己之前的 state（因果）。
源码把这个等价性写成注释并配了测试（`model.py:155-156`、`model.py:260-261`：
*the two agree to fp32 noise (tests/test_model.py::test_rows_match_packed)*）。

### 7.2 每题一行因果序列（混合骨干的唯一选择）

`is_hybrid()` 判定 `layer_types` 里有没有 `linear_attention`（`model.py:69-72`）。
Qwen3.5 的 Gated DeltaNet 层是**递推**的、忽略注意力掩码，所以 packed 掩码没有意义：
`rows_of` 把记录拆成「state + 第 k 题」的因果行（`model.py:188-202`），一行一题地跑
（`forward_rows_batch`，`model.py:364-383`）。

代价被源码明写在两处：

- **训练**：state 会按题数重复计算 `Q × state_tokens`，注释说 *which training accepts*（`model.py:367`），
  `shared_prefix` 与 `full_ft` 默认一起打开就是为了消掉它（`train.py:440`，`shared_prefix.py:1-19`）；
- **运行**：不重复算，靠下面的前缀缓存。

### 7.3 运行期的 state 前缀 LRU

`DecisionModel.prefix()` 只跑 state 并返回 `(n_state_tokens, KV cache, state hidden)`（`model.py:416-426`），
`probs_with_prefix()` 只跑分支并把 cache 裁回 state（`model.py:446-462`，裁剪在 `finally` 里，
`model.py:443` 与 `model.py:461`）。**精确性靠构造**：分支之间互不可见、state 也看不见分支，
所以 state 的隐状态与 KV 与「有没有分支」无关（`model.py:404-406` 的注释）。

`Server` 侧是一个 LRU（`serve.py:38-90`）：

- 键是 `(state token ids 元组, option_isolation 布尔)`（`serve.py:55`）；
- 两个上限：至多 `KEV_PREFIX_CACHE=4` 个 state，合计至多 `KEV_PREFIX_MAX_TOKENS=65536` 个 token
  （`serve.py:27-29`）——注释把预算算给你看：*One 64k state (Kev-27B: ~1.3 GB of keys, values and
  DeltaNet states), or four 16k ones, not four 64k ones*；
- `min_tokens` 默认取模型的 `prefix_min_tokens`：**注意力骨干 384，混合骨干 0**
  （`model.py:283-288`）。理由也写在注释里：混合骨干的 miss 路径否则会按题数重算 state
  （*Kev-0.8B bf16 on MPS, 5 questions: 1011 -> 413 ms*）；
- `make_room` 在**这批跑之前**就把注定被挤掉的条目删掉，避免新老 64k state 同时驻留
  （`serve.py:67-78`，实测记录见注释：*Kev-4B on MLX: a 64k state's cache is 2.2 GB*）；
- `_run` 有一个 OOM 重试环：带着缓存跑 OOM 时**清掉缓存再跑一次**（`serve.py:172-191`），
  注释指向 issue #75。

### 7.4 缓存没有改答案——但缓存**有**一个关于答案的坑

`_rows_hidden` 复制 cache 时，对线性注意力层要**显式深拷递推状态**（`model.py:346-357`）：

```python
for source, target in zip(cache.layers, replica.layers):
    if isinstance(source, LinearAttentionCacheLayerMixin):
        target.conv_states = source.conv_states.copy()
        target.recurrent_states = source.recurrent_states.copy()
        ...
```

这**正是** §4.3 那张表里别家踩过的坑（浅拷贝会让递推状态别名、污染后续分支）——
Kev 的 `rows_per_pass` 分批（`model.py:41-47`）会让同一批里多个 chunk 复制同一份 cache，
所以这里必须是深拷。`replica.reorder_cache(torch.zeros(len(part)))`（`model.py:357`）
的语义与 SemIf-OpenJev 的 `reorder_cache(zeros)` 相同：**把第 0 行复制 N 份**。

---

## 8. 服务化与 fail-loud

### 8.1 请求线程 → 单模型线程

所有前向都归一个线程（`serve.py:120-121`）。请求线程只做 `admit` + 入队 + 等 Future
（`serve.py:132-145`）；模型线程把自己空闲时队列里的东西**一次全取**（上限 `MAX_BATCH=64`，
`serve.py:158-160`），跑完再把结果分回各自的 Future（`serve.py:166-168`）。
异步端点用 `asyncio.wrap_future(self.submit(rec))`，注释说明了理由：
*so a container takes as many concurrent requests as its batches can absorb*（`serve.py:205-209`）。
另外它把 Python 的 GIL 切换间隔调到 0.5 ms（`serve.py:115-119`），
注释给了实测：默认 5 ms 时一个 batch 的模型时间被拉长约 2×。

### 8.2 拒绝而不是静默截断

`admit`（`model.py:130-142`）与 `Server.submit`（`serve.py:139-142`）把上下文溢出变成 **HTTP 422**，
错误消息里带**实际 token 数**与**上限**，并给出两条出路（缩短文档 / 切分 / 开 `KEV_TRUNCATE_STATES=1`）。

```python
raise HTTPException(422, str(e) + (f"; or start the server with KEV_TRUNCATE_STATES=1 to read only its first {e.max_state:,} tokens (responses then say truncated: true)" if state else ""))
```

`KEV_TRUNCATE_STATES=1` 是**显式选择**：截断之后每条响应都带 `truncated` 与
`usage.state_tokens` / `state_tokens_used`（`serve.py:217-219`）；两个 demo 端点从任一子回答里继承这些标记
（`truncate_marks`，`serve.py:289-295`）。这条「不许静默截断」在 Kev 1.0 的发布说明里是一件专项
（`docs/releases/kev-1.0.md:34`）：*The server used to cut a state longer than its limit without saying so.*

### 8.3 每一条响应都带请求 id 与服务端计时

`typesafe` 中间件（`serve.py:232-242`）在**所有**响应上写 `x-typesafe-request-id`
（回显调用方的或新生成）与 `server-timing`，后者注释写明用途：
*time inside this process, for telling it from the network*。鉴权只在设了 `KEV_API_KEY` 时启用，
比较用 `hmac.compare_digest`（`serve.py:236`）。

### 8.4 发布数字的可追溯清单（本文的「实测」之一）

`docs/claims.json` 是「已发布数字 → committed 证据」的清单，`scripts/verify_claims.py` 校验它
（`scripts/verify_claims.py:1-14`）：每条 claim 记 `{"printed": 文档里那个字符串, "in": [必须出现它的文件],
"source": 证据 JSON, "path": 取值的键路径}`，校验方式是**取值格式化到该精度后与 printed 逐字符相等**，
并且 printed 必须**真的出现在**那些文件里。清单自己的文档串诚实标注了它的边界：
*The text check is presence, not position: it proves the number is published and traces to evidence,
not that every occurrence of those digits means this claim.*（`scripts/verify_claims.py:7-9`）

**实测**（§12.4）：committed 的三张模型卡与 `runs/release/kev-1.0-manifest.json` 里
「该 Hub 仓的 `README.md`」条目**逐字节相符**——三张卡的 sha256 分别等于清单里的
`kev-0.8b` / `kev-4b` / `kev-9b` 的 `README.md` 值，而发布说明第 93 行给出的上传命令正是
`hf upload jaredpalmer/kev-<size> docs/model-cards/kev-<size>.md README.md`。
第四张卡（`kev-27b.md`）**不在清单里**——与发布说明第 83 行「27B 的 51 GB 超过 GitHub 单资产 2 GB 上限，
不做资产、只在 Hub 上」一致。这是**我算的**，不是抄的；命令在 §12.4。

---

## 9. 置信度从哪来，又怎么被改写

### 9.1 头部：**训出来的**

`PointerHead` 的 `q` / `k` 是训练参数（`model.py:209`），损失是**每题在 K 个选项上的交叉熵**
（`train.py:41-61`）：

```python
y = torch.tensor([q["label"]], device=dev)
loss = F.cross_entropy(z[None], y, label_smoothing=label_smoothing)
```

`label` 是**选项键列表里的下标**（`data.py:401-403`）：

```python
y = src_q["label"]
q["label"] = m["keys"].index(y) if m["type"] == "choice" else int(y)
```

所以「训练本身」这一环同时改数值与 argmax——这与 RYOTIDE 的「只读」是根本区别。
`--full_ft 1`（Kev-27B）连骨干一起训（`README.md:328`：*Every weight of the base is fine-tuned*）。

### 9.2 温度：训练之后施加的一个**单调**缩放

`head.temperature` 在训练里**没有任何角色**：`PointerHead.forward` 只在 `not self.training` 时才除它
（`model.py:218`），而 `kev/train.py` 全文没有 `temperature` 这个词。
它由脚本拟合后写进 `head.pt`（`checkpoint.py:61` 的 `Meta.temperature`），
加载时落到头上（`checkpoint.py:241`）：

```python
m.head.temperature = meta.temperature if opts.temperature is None else opts.temperature
```

除以一个正数**不改变 argmax**。Kev 侧的表述有两处：模型/README 的话
`README.md:385`：*A temperature never changes which answer wins.*；
以及 `metrics.py:361` 的一句行内注释 *argmax is temperature-invariant*。
**这句在 Kev 上没有断言也没有专项测试**（`metrics.py` 全文只有这一处），
但可以从数学上确认（正标量单调），且能用 committed 产物实测（§13.3）。

**但有一条 Kev 自己记下的反例边界**（`AGENTS.md:445`）：

> Temperature can reorder confidence across multiclass questions, even at fixed option count.

即：温度不改**同一题内**的 argmax，但会改**跨题**的置信度排序。
`README.md:431` 与发布说明第 54 行把这条列进限制（*One temperature per model cannot reorder confidences*），
指的是这条的另一面：温度是单参数的，无法让 A 题更自信、B 题更不自信。

### 9.3 拟合的判据与「被拒绝的温度」

`scripts/calibrate_checkpoint.py` 的文档串（`scripts/calibrate_checkpoint.py:1-28`）记了完整判据：

- 拟合集必须是**留出数据集**，不能是训练语料的留出条目。脚本**拒绝**这类行，除非 `--allow-in-distribution`；
  拒绝能力来自 `kev.rounds.pool_conflicts`（对照检查点自己的训练记录）；
- 拒绝是**因为一次真实的失败**：round 19 的 SFT 臂在 sft-v1 development 行上拟合出 T=0.955，
  在 breadth-v1 上 ECE 0.059；换成留出数据集池后同一检查点 0.0085；
- 也会因**拼写错误**拒绝：`path:source,...` 允许列表里的 source 不在该 suite 里就报错，
  否则会静默丢行（`scripts/calibrate_checkpoint.py:52-56`）；
- 无条件写值要走 `--temperature T --reason …`，理由记进 `head.pt["temperature_fit"]`。

Kev 1.0 的四个温度**全部来自留出条目**：0.8B 2.35 / 4B 2.41 / 9B 2.19 是**训练语料的留出条目**，
27B 1.32 是**留出数据集**（`docs/releases/kev-1.0.md:39`）；发布说明同时说，
两个小模型在留出数据集上的重拟合**被评估过、都没采纳**（第 39 行），
4B 的判据是 Brier 差 −0.0001 [−0.0005, +0.0003]、ECE 0.025 vs 0.024
（`docs/model-cards/kev-4b.md:239`）。

### 9.4 输出的 `confidence` 不是准确率，也不是概率本身

`api.py:120-140` 的两个公式**逐字照抄 TypeSafe 的参考适配器**
`system-one-adapter` 0.2.1 的 `confidence_metrics.py`（注释见 `api.py:120-121`，README 第 282 行也这么说）：

```python
def choice_confidence(p):  # (p_max - 1/K) / (1 - 1/K): 0 at uniform, 1 at certainty
    K = len(p)
    return 1.0 if K == 1 else (max(_normalize(p)) - 1 / K) / (1 - 1 / K)
```

- 先 `_normalize` 把和归一（全零 → 均匀，`api.py:122-124`）；
- `choice` 用 `(p_max − 1/K)/(1 − 1/K)`；`score` 用 `max(0, 1 − E|level − mode| / D)`；
- `noul` **没有** `confidence` 字段，直接给 `p(true)`（`api.py:153`）；
- 序列化精度是 `round(·, 4)`（`api.py:143-146`），理由也写在注释里：
  255 选项下 `255 × 0.00005 < 0.02`，落在 TypeSafe 的容差内。

**校验的是未舍入的值**：`to_answers` 先按未舍入的 `p` 取 argmax
（`api.py:156`：`max(range(len(p)), key=lambda i: p[i])`），再把各字段舍到 4 位。
所以**舍入不会改变服务端报出的 `choice`**，但会让下游从 `probabilities` 自己重算 argmax 时可能不一致——
性质与 OpenJev(HF) 的第 ④ 环相同。

### 9.5 「谁在改发布出去的数值」——Kev 的链表

| 环节 | 位置 | 改 argmax？ |
|---|---|---|
| ① **训练本身**（`PointerHead` 的 q/k；27B 还有整个骨干） | `train.py:41-61`、`train.py:342-365` | **是** |
| ② `head.temperature`（训练后拟合，加载时落到头上） | `checkpoint.py:241`、`model.py:218` | 否（正标量单调） |
| ③ TypeSafe 的 `confidence` 变换 | `api.py:127-140` | 否（`choice` 是 `p_max` 的单调函数） |
| ④ `round(·, 4)` | `api.py:143-146`、`api.py:153-159` | 否（选中的是未舍入的 argmax），但下游重算可能不一致 |
| ⑤ `RotationAveraged`（测试期顺序平均） | `predictors.py:213-251` | **会**——但它要 `rotations >= 2` 才生效，而发布的读数是 `rotations = 1`（§14.2） |

**另有一条口径坑**（写在 committed 产物里，值得单独指出）：
benchmark 落盘的 `logits` 是**已被温度除过**的（`predictors.py:141`、`predictors.py:147`），
`metrics.py:241` 还原原始 logits 的办法是**乘**回去：

```python
z = np.asarray(row["logits"], dtype=float) * temperature
```

所以「纪录里的 `logits` 是原始 logits」这个在其他实现里常见的直觉，**在这里是错的**——
要看 `inference_temperature` 是不是 1.0。§13.3 会在 committed 行上实测这条约定。

### 9.6 两套 readout 并存，但只有一套在发布路径上

`kev/anchors.py` 实现的是**RYOTIDE 式读法**：冻结基座在 `Answer:` 之后的**下一 token 字母 logits**
（`anchors.py:48`：`logits = model(**enc).logits[:, -1].float()`，字母 id 取自 `" " + L`，
`anchors.py:40`），softmax 到 K 个字母上作为**教师分布**（`anchors.py:51-52`）。
它经 `--anchor_w` 作为一个 KL 项参与训练（`train.py:355-358`、`train.py:64`、`train.py:408-409`），
目的是「微调会侵蚀基座能力」（`anchors.py:1-3`）。

**也就是说：RYOTIDE 那条读法在 Kev 里确实存在——但它只出现在训练期的教师信号里，从不出现在服务路径上。**
`scripts/base_mmlu_probe.py` 用同一套读法（`anchors.py:2-3`）。
`--anchor_w` 默认 0.0（`train.py:409`），且 `--anchor` 与 `--anchor_w` 必须同时给或同时不给（`train.py:450`）。

---

## 10. 四条不变量的落位

RYOTIDE §13 的四条，逐条落到 Kev：

| RYOTIDE §13 不变量 | Kev 的落位 |
|---|---|
| **#1** 读位置必须落在标记上（诊断量 `marker_mass`） | **概念不适用**：读出的不是词表分布，没有「落在标记上的质量份额」这个量。等价命题改写成「**softmax 的输入必须恰好是那 K 个选项跨度的标量**」——它在构造上成立（`model.py:216-218` 只接受 `h_opts`，`opt_idx` 由 `encode` 算出）。Kev 自己的守卫换成**分布自检**：每条记录写 `raw_probability_sum` 与 `zero_count`（`benchmark.py:61`），实测见 §13.3 |
| **#2** 前缀与标记表面形式耦合，必须成对扫 | **不适用**：标记是 token **id**（`SPECIAL` 编译期常量 → `convert_tokens_to_ids`，`model.py:104`、`model.py:106`），不存在「表面形式」这个自由度。同类耦合换成「**选项文本不可伪造分隔符**」——用一次正则改写把用户的 `<|name|>` 变成 `<¦name¦>`（`model.py:78-81`），并有 `AGENTS.md:416-417` 记录 |
| **#3** 只读最后一个位置 | **改写**：「**只读每题构造出来的那 K+1 个位置**」——K 个 `</opt>`（读出）+ 1 个 `<decide>`（只当 query，不产生 logit）（`model.py:121-125`、`model.py:217`）。它**不读末位**，也不读 `/opt` 之外的任何位置。「不物化词表维度」这一半成立得更彻底：骨干连 vocab head 都没有（`model.py:257`） |
| **#4** 比较概率 / ECE 之前先确认 temperature | **成立，而且有两个方向要确认**：①T 是多少（`head.pt` 的 `Meta.temperature`，`checkpoint.py:61`；`/v1/models` 会报出来，`serve.py:307`）；②**`logits` 是不是已经除过 T**——Kev 的 rows 里是除过的，还原要**乘**（`metrics.py:241`），直觉方向相反（§9.5） |
| （Kev 新增的） | **①「发布数字」与「服务默认」不是同一条精度路径**：所有已发布准确率走 **fp32 evaluation path**，而服务在 CUDA/MPS 上默认 **bf16**（`README.md:383`：*Published accuracy numbers use fp32 evaluation, not the bf16 serving path*；`checkpoint.py:89-92`）；<br>**②「选项顺序」在训练期是被打散的、在运行期只被度量**——所以任何「Kev 对顺序不敏感」的说法都超出了它的记录（§6.4） |

---

## 11. 文件对照表

| 文件 | 行数 | 职责 |
|---|---|---|
| `kev/model.py` | 508 | `SPECIAL` 分隔符、`encode`、`rows_of`、`branch_mask_batch`、`PointerHead`、三套前向与 state 前缀 |
| `kev/api.py` | 165 | TypeSafe 请求/响应形状、`render`、`to_record`、`to_answers`、两个 `confidence` 公式、`output_tokens` |
| `kev/serve.py` | 358 | FastAPI、请求线程→模型线程、`PrefixCache`、422、三个端点、bearer 鉴权 |
| `kev/predictors.py` | 341 | `LocalPredictor`（fp32-exact 开关、长行内核）、`RemotePredictor`、`JevPredictor`、`RotationAveraged` |
| `kev/metrics.py` | 426 | ECE（10 等宽箱）、Brier、NLL、`fit_temperature`、`raw_row` / `tempered_row`、`paired_bootstrap` |
| `kev/train.py` | 680 | 交叉熵损失、`augment`/`permuted_copy`、anchor KL、perm KL、LoRA 目标 |
| `kev/data.py` | 410 | 公开数据集 → TypeSafe 形状、`augment`（打乱 + none/distractor）、`materialize` |
| `kev/checkpoint.py` | 362 | `head.pt` 布局、`Meta.temperature`、`LoadOptions`（fp32 默认 / 服务 bf16） |
| `kev/anchors.py` | 71 | RYOTIDE 式字母读法，只做**训练期教师** |
| `kev/calibrate.py` | 73 | 逐 workload 四臂报告（raw / shipped / workload / workload_oof） |
| `kev/mlx_model.py` | 229 | Apple Silicon 第二后端，复用同一个 `encode` / `PointerHead` |
| `scripts/calibrate_checkpoint.py` | — | 把温度写进 `head.pt`，带**留出集判据**与拒绝路径 |
| `scripts/verify_claims.py` | — | 校验 `docs/claims.json`：已发布数字 ↔ committed 证据 |
| `docs/claims.json` | — | 「printed 字符串 → 证据文件 + 键路径」清单 |

---

## 12. 亲手跑一遍

### 12.1 起服务（上游给的命令）

```bash
git clone https://github.com/jaredpalmer/kev.git && cd kev && uv sync --extra serve
uv run --extra serve python -m kev.serve --run jaredpalmer/kev-4b@v1.0 --port 8009   # README.md:114
```

MPS 上混合检查点默认走 MLX（`serve.py:340`）；`KEV_BACKEND=torch` 可以拒绝。
读一眼服务端自己报的常数（含温度与缓存状态）：

```bash
curl -s localhost:8009/v1/models | python3 -m json.tool    # serve.py:298-312
```

### 12.2 一条请求 / 一次顺序探针

```bash
curl -s localhost:8009/v1/systemone -H 'content-type: application/json' -d '{
  "state": "I was charged twice for order 1182. Please refund one of the charges.",
  "questions": {"team": {"type": "choice", "instructions": "Which team should handle this?",
    "criteria": {"billing": "Charges and refunds", "shipping": "Deliveries", "returns": "Exchanges"}}}
}' | python3 -m json.tool

# 同一道题换 6 种选项顺序：看 argmax_stable 与 spread     serve.py:262-275
curl -s localhost:8009/v1/systemone/permute -H 'content-type: application/json' \
  -d '{"request": {…同上…}, "question": "team", "n_perm": 6, "seed": 0}' | python3 -m json.tool
```

想要未校准的原始概率：`KEV_TEMPERATURE=1.0`（`checkpoint.py:144`、`README.md:296`）。

### 12.3 自己算一遍置换统计与温度（不 import 上游，只用 committed 产物）

```bash
python3 - <<'PY'
import json
for f in ["runs/rel9-public/1-jaredpalmer_kev-9b/report.json",
          "runs/rel27-public/1-jaredpalmer_kev-27b/report.json",
          "runs/r4-kev-4b-v4-one/report.json"]:
    r = json.load(open(f))
    print(f.split("/")[1], r["run"], "T =", r["calibration"]["inference_temperature"], r["permutation"])
PY
```

### 12.4 「实测」：模型卡与发布清单逐字节对照

```bash
python3 - <<'PY'
import hashlib, json
man = json.load(open("runs/release/kev-1.0-manifest.json"))
want = {a["repo"]: a["files"]["README.md"] for a in man["assets"]}
for size, repo in [("0.8b","jaredpalmer/kev-0.8b"),("4b","jaredpalmer/kev-4b"),("9b","jaredpalmer/kev-9b")]:
    h = hashlib.sha256(open(f"docs/model-cards/kev-{size}.md","rb").read()).hexdigest()
    print(size, h == want[repo], h[:12])
print("kev-27b.md 在清单里吗:", "jaredpalmer/kev-27b" in want)   # False：27B 不是发布资产
PY
# 结果：0.8b True / 4b True / 9b True；kev-27b.md 不在清单里（与发布说明第 83 行一致）
```

清单与 `SHA256SUMS.txt` 也互相印证（三个 tar.gz 的 sha256 两处相同）：

```bash
python3 -c "
import json; m=json.load(open('runs/release/kev-1.0-manifest.json'))
s=dict(l.split()[::-1] for l in open('runs/release/kev-1.0-SHA256SUMS.txt'))
print([s[a['file']]==a['sha256'] for a in m['assets']])"   # [True, True, True]
```

### 12.5 复现一篇发布的读数（需要权重，本文没跑）

```bash
uv run python -m kev.benchmark --run jaredpalmer/kev-4b --suite evals/v8/decision-v8 --split development --out runs/mine
uv run python -m kev.calibrate --rows runs/mine/rows.json       # kev/calibrate.py:1-2
uv run python scripts/verify_claims.py                          # scripts/verify_claims.py:1-14
```

---

## 13. 实例走查

### 13.1 发布检查点：Kev-4B 在 77 个选项上的读数

来源：`runs/release/kev-1.0-verify/hub-4b/rows.json`（**committed**，2 条记录，`run` 字段是
`jaredpalmer/kev-4b@v1.0`，`calibration.inference_temperature = 2.406050072164233`——与
`docs/model-cards/kev-4b.md:302` 记的 `head.pt` 里的 `2.4061` 同一个数）。

| 字段 | 值 |
|---|---|
| `id` | `banking77/test/1962` |
| `keys` | 77 个 intent 名（`activate_my_card`、`age_limit`、…） |
| `label` | 63 |
| `inference_temperature` | 2.406050072164233 |
| `p[63]` | 0.9082994462298021（最大） |
| `logits[63]` | 见下 |
| `raw_probability_sum` | 1.0000000654858923 |

三点值得注意：

1. **77 个选项是一条 `choice` 题的常态**——候选数上限 255 是 API 契约，不是分词器约束（§2.2）。
   同一份代码在 §13.2 里又是 9 选项。
2. `raw_probability_sum = 1.0000000654858923`，即服务端报出的分布**归一化到 1e-7 量级**——
   这是 `softmax` 天然和为一的体现，不是任何 renormalise 步骤。
3. **该行的 `logits` 是已经除以 2.4061 的**（§9.5）：把 `p` 与 `softmax(logits)` 比，差 2.8e-09；
   若按「logits 是原始的」去算 `softmax(logits/T)`，会得到一条完全不同的、更平的分布（最大差 0.68）。
   哪个对由代码定：`predictors.py:140-141` 先 `softmax(z)` 再原样存 `z`，而 `z` 已由
   `model.py:218` 除过温度。

### 13.2 有状态可读的一条：documents-v1 的 CFPB 投诉

来源：`evals/documents-v1/development.jsonl`（committed 的记录，含 state 与选项文本）
join `runs/r15-08b-a-docs/rows.json`（committed 的读数，920 行 / 568 记录，`T = 1.0`）。
这条能给出**完整的一条链**：state → 选项文本 → 概率 → 对错。

| 字段 | 值 |
|---|---|
| `id` | `cfpb/6623062`（state 1773 字符，CFPB 消费者投诉叙述） |
| 题 | `product`，instructions = *Which kind of financial product is this complaint about?* |
| 9 个选项 | `credit_reporting` / `debt_collection` / `mortgage` / `bank_account` / `card` / `student_loan` / `money_transfer` / `vehicle_loan` / `personal_loan`，每个带一句描述（`criteria` 是 map，`api.py:112` 用 `name: desc` 渲染） |
| label | `card`（下标 4） |
| `inference_temperature` | 1.0（该检查点未带温度） |
| `logits` | `[-2.96205472946167, -2.626758098602295, -5.337671756744385, -3.385693073272705, 7.858955383300781, -4.287512302398682, -5.272880554199219, -3.7289485931396484, -1.8867435455322266]` |
| `p` | `[1.997e-05, 2.793e-05, 1.857e-06, 1.308e-05, **0.9998620652442762**, 5.306e-06, 1.981e-06, 9.276e-06, 5.854e-05]` |
| `zero_count` / `raw_probability_sum` | 0 / 1.000000009609039 |

按 §9.4 的公式，这一题服务端会报出的东西是：
`choice = "card"`、`confidence = (0.9998620652442762 − 1/9)/(1 − 1/9) = 0.9998448233998107 → 0.9998`、
`probabilities = {"credit_reporting": 0.0, …, "card": 0.9999, …}`（`round(·,4)`）。
**其中 `credit_reporting: 0.0` 只是序列化的结果**——落盘的 `p` 是 1.997e-05，`zero_count` 记的是
「恰好等于 0 的个数」，这一行是 0。引用这条题的分布时，用 rows 里的 `p`，不要用响应里舍过的 4 位。

### 13.3 「实测」：在 committed 行上重算

在 §12.3/§12.4 之外，这两条也能在 committed 产物上做完，本文做过了：

1. **`p == softmax(logits)`**：在 `runs/release/kev-1.0-verify/hub-{4b,0.8b}/rows.json` 的 4 行、
   以及 §13.2 那一行上成立，最大差 ≤ **1.5e-08**（0.8B 第 2 行）——证明记录下来的 `logits` 与 `p`
   **处在同一个温度上**，与 `metrics.py:241` 「乘回温度以还原」的写法一致。
2. **argmax 对温度不变**：同一批行上 `argmax(p) == argmax(logits)` 全部成立；
   而 §6.4 表格里 9B 的 `mean_max_delta = 0.0681` 说明**数值**确实被顺序改动了——
   两件事不矛盾：**温度**是每题的单调缩放（不动 argmax），**选项顺序**改的是前向本身（会动）。

**本文没有做的**：任何需要权重的事（前向、更细的温度、别的 workload 上的顺序敏感性）。
那需要 0.8B 起的一组权重与一个 GPU，本机没有，所以**本文不给任何由本机前向产生的概率数值**。

---

## 14. Kev 引入的、别篇没有的三个维度

### 14.1 `date_facts`：把「算不出来的日期减法」变成状态里的一行

`api.date_facts`（`api.py:66-81`）扫描 state 里的绝对日期（两种格式，`api.py:62-63`），
为**每一对**日期生成一句 `"August 3, 2026 is 12 days after July 22, 2026."`；
`with_date_facts` 把它挂成 object 的 `date_facts` 字段、array 的一个元素、或字符串末尾的段落（`api.py:84-91`）。
理由写在注释里（`api.py:67-69`）：*The model cannot subtract dates reliably (issue #8); it can use a stated day count.*

这是十篇里少见的「**读之前改 state**」的机制，而且它是 **opt-in** 的：
服务端要 `KEV_DATE_FACTS=1`（`serve.py:31`、`serve.py:223-225`），benchmark 要 `--date_facts`
（`api.py:86`），每次 report 都记 `date_facts` 这个布尔（§13 引的三份 report 里都是 `False`）。
`README.md:297` 与模型卡把它列进限制（日期算术是最弱的家族：`deadline` 政策 0.65 vs Jev 0.95，
`docs/releases/kev-1.0.md:50`）。

### 14.2 四条「写了但没发布」的路径——Kev 全部记在明处

§4.5 那张表收的是各篇「算得出来 ≠ 接进来了」。Kev 有**四条**，性质各不相同，且都能从代码与
committed 产物直接读出：

| 路径 | 位置 | 状态 |
|---|---|---|
| `option_isolation`（构造性置换不变） | `model.py:116-118`、`model.py:176-183` | **默认关**（`train.py:395` 默认 0），且混合骨干**不支持**（`model.py:263`）。当前全家族都是混合骨干，所以这条路径在发布版上不可达 |
| `--perm_kl`（两个选项顺序下的对称 KL） | `train.py:336-338`、`train.py:359-362` | **默认 0.0**（`train.py:381`）；`--perm_frac` 默认 0.3 |
| `RotationAveraged`（测试期顺序平均） | `predictors.py:213-251` | 要 `rotations >= 2` 才生效；committed 的 report 里 `rotations` 全是 **1** |
| 三个可选损失 `--label_smoothing` / `--brier_w` / `--focal_gamma` | `train.py:52-56` | **一次注册过的筛选筛完没有候选通过**，`AGENTS.md:446` 原话：*the `--label_smoothing/--brier_w/--focal_gamma` losses exist but ship in nothing* |

最后一行是这一类里最诚实的一条：不是「忘了接线」，而是**接过、测过、记录了负结果**。
（对照 §4.5 里 Laya 的 `soft-ce` 与 RYOTIDE 的 `calibration_bias`。）

### 14.3 MLX 第二后端：同一个 `encode` 与同一个头，换 Metal 内核

`kev/mlx_model.py` 的文档串（`mlx_model.py:1-19`）说明它**复用** `kev.model.encode`、`rows_of`
与**同一个 `PointerHead`**（fp32、带检查点的温度），只把骨干换成 mlx-lm 的 Metal 实现
（MPS 上 PyTorch 走参考实现，Kev-4B 每请求约 0.8 s）。
两个实现的契约由 `SCORING_INTERFACE` 固定（`model.py:228-229`），
`tests/test_mlx.py` 按这个列表检查 MLX 类，并对 fp32 torch 路径测 `max |dp|` 与 argmax 翻转。
`LoadOptions` 的文档串明确写了**benchmark 从不用它**：
*never by the benchmark, whose reported numbers stay on the fp32 torch path*（`checkpoint.py:106-112`）——
即**发布数字只有一条后端**，MLX 是服务选项。

---

## 附：本文与上游文档的关系

- 本文**没有**引用上游任何一句结论性的技术判断；`README.md` 的「How It Works」只用来定位
  （第 2 行、第 318 行），所有机制描述都回到 `kev/**.py` 的行号。
- 上游**模型卡与发布说明**用来取**已发布的数字**（温度、准确率、上下文长度、拒绝/采纳的判据），
  这些地方一律标了 `文件:行` 或 run 标签。
- 「九条/十条」以本仓库 `docs/comparison.md` 为准；本文是该总表的第 10 列（收录时该表刚扩到 CLM 那条，Kev 是第 10 条）。

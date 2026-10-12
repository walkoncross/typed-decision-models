# Laya 是如何用一次前向把 state 读成类型化决策的

> 代码库：<https://github.com/NandaKishorM/laya>
> 对应 README 第 8 行：*Multilingual, non-autoregressive System 1 decision engine. Typed decisions
> over 100+ languages in a single forward pass — 33 ms — trained with reinforcement learning
> against strictly proper scoring rules (RLCD), with a router that picks the right checkpoint per
> request.*
> 本文按 `laya/` 的实际代码逐环节说明这句话的实现方式；文中出现的文件名与行号（`common.py:634`、
> `train.py:499`、`serve.py:1079` 等）均相对该仓库根目录。

与另外几篇最大的不同：**Laya 不读「最后一个位置」**。它把每个选项铺成序列里的一个 `[MASK]` 槽位，
在**一次前向**里同时取出 K 个槽位各一个标量。§0 先把这条分歧讲清楚，§4 展开。

## 本文的事实来源

| 类别 | 来源 | 本文中的标记 |
|---|---|---|
| 代码引用 | `laya/**`（本仓） | `文件:行` |
| 可实跑得到的产物 | 路由、语言检测、置信度与门控层是**刻意 torch-free** 的：`import laya` 成功且 `laya.__version__` = `0.4.4`，全程不加载 torch（`__init__.py:40-42` 的注释说明了为什么） | 标注「实测」 |
| 训练侧机制 | `train.py`、`calibrate.py` | `文件:行` |
| 未跑 | **本机无 torch、无 checkpoint，本文没有跑过任何前向** | 照例不给任何前向产生的概率数值 |

`import laya` 不拉 torch 不是巧合：`__init__.py:43-81` 把 torch-backed 的名字全部登记进 `_LAZY_ATTRS`，
由 `__getattr__` 按需导入。所以「路由 + 语言 + 置信度」这一层可以脱离 torch 单独执行——本文全部实测都来自这一层。

模型侧的权重与配置（`temperature`、`act_costs`、`max_len`）需要从 Hub 取，本机没有，第 14 节给出取法。

---

## 0. 与 RYOTIDE 的分野

对照对象是 [`docs/ryotide/typed-decision-model.md`](../ryotide/typed-decision-model.md)——本文凡称「RYOTIDE §N」均指该篇的节号。

| | RYOTIDE | Laya |
|---|---|---|
| 骨架 | 因果解码器，读**最后一个位置** | **双向编码器**（ModernBERT 系），读**每个选项的槽位** |
| 读几个位置 | 1 个 | **K 个**（K = 选项数），一次前向 |
| 读出的量 | 词表 logits → 取标记 token 的那几维 | 每个槽位一个**标量**，由共享 `scorer` 打分 |
| 「typed」怎么保证 | 标签必须落在**单 token** | 槽位是 `[MASK]`，**与选项文本多长无关** |
| 读位置怎么定 | 运行期扫 9 前缀 × 2 表面形式 | 训练期固定，**无运行期标定** |
| 选项上限 | 260（`A0`–`Z9` 两步读） | 无硬上限，受 token 预算与「槽位不许塌陷」约束 |
| 顺序偏置 | 多 order 平均 | **训练时消解**（`parallel` 布局，§6） |
| 训练 | 无，权重全程 frozen | **有**，RLCD（§11） |
| 路由 | 无，逐 preset 选后端 | **有**，逐请求挑 checkpoint（§8） |

一句话：**RYOTIDE 在「模型会写下答案的地方」读词表分布；Laya 在「模型被要求填空的地方」读一个标量。**
前者借用了生成模型已有的答案位置，后者直接把决策改造成填空题。

两者的共同点只剩最本质的一条：**一次前向、不生成任何 token、输出封闭在已知选项集上**。

---

## 1. 端到端数据流

```
                    ┌──────────── serve.py ────────────┐
POST /v1/systemone  │ BODY_CONTROLS 白名单(serve.py:157)│
{state, questions}  │ BODY_REFUSALS 显式拒绝(serve.py:158)│
                    └───────────────┬──────────────────┘
                                    v
        ┌───────────────── router.py（§8）──────────────────┐
        │ _route(): 纯 Python 启发式，不加载也不跑任何模型    │
        │ 显式 model > task > lang > lang_guess > 检测 > 默认 │
        └───────────────────────┬───────────────────────────┘
                                v
        ┌──────────────── common.py / agent.py ─────────────────┐
        │ build_head()  [CLS] <type> ins [SEP] [MASK]opt0 [MASK]opt1 … │
        │ build_sequence()  再接 [SEP] state [SEP]，裁到 max_len  │
        │ _encode_state() state 只分词一次，逐题复用 state_ids     │
        │ collate_items()  → input_ids/attention_mask/marker_pos/marker_mask/qtype │
        └───────────────────────┬───────────────────────────────┘
                                v
        ┌──────── backends/：eager / compile / tilelang / onnx ────────┐
        │ encoder(input_ids, attention_mask) → last_hidden_state        │
        │ + type_emb(qtype)，过 head 的 2 层 TransformerEncoder          │
        │ torch.gather(h, marker_pos) → scorer(·) → 每选项一个 logit      │
        │ masked_fill(~marker_mask, -1e4)                               │
        └───────────────────────┬───────────────────────────────────────┘
                                v
        ┌──────── agent._decode_answers()：softmax(logit / T) ─────────┐
        │ T = temperature_by_options[temp_bucket(qt,k)] 或 temperature[qt] │
        │ answer_confidence = max(p)；confidence = 1 - H/log k            │
        └───────────────────────┬───────────────────────────────────────┘
                                v
   {model, answers, usage{input_tokens, output_tokens: 0}, routing}
```

`output_tokens` 在**模型层硬编码**：`agent.py:1550` 的 `"output_tokens": 0,`，另有 `agent.py:1871`、
`onnx_agent.py:610`、`onnx_agent.py:818` 三处同类赋值。`serve.py` 只是透传与在批处理路径上求和
（`serve.py:1450`）——它不含这个常量，值来自上面那几行。

---

## 2. 「typed」从哪来：`[MASK]` 槽位，而不是单 token 标签

RYOTIDE 的类型化是**借来的**：它要求标签能落到分词器的单个 token 上，落不到就退化成字母标记。
Laya 不需要这条约束，因为**被评分的不是标签文本**。

`common.py:169` 的 `build_sequence` docstring 就是权威定义：

```
Format: [CLS] <type> instructions [SEP] [MASK] opt0 [MASK] opt1 ... [SEP] state [SEP]
```

每个选项前面放一个 `[MASK]`（`common.py:251` 的 `opt_ids.append([tok.mask_token_id] + opt_tokens)`），
并记下它的绝对位置（`common.py:261-264`）。这个位置列表就是后面要 `gather` 的东西。

三个直接后果：

1. **选项文本可以任意长**。它只是被编码进序列，不参与「能不能被评分」。真正被评分的是它前面那个 `[MASK]`。
   对照 RYOTIDE §2：那里选项文本必须先压成单 token 才可能被读到。
2. **选项数不撞词表**。RYOTIDE 超过 26 就要切到 `A0`–`Z9` 两步读（上限 260）；Laya 没有这个台阶。
3. **`[MASK]` 是模型极其熟悉的符号**。RYOTIDE §9.3 实测过：把标记换成与 prompt 嵌入最远的字符，
   标记质量从 0.9998 掉到 0.9210、准确率同时下降。Laya 用的 `[MASK]` 正是预训练目标里被问了千万次的
   那个符号——它天然是「此处待填」的位置，不需要额外论证。

### 槽位塌陷：一条只有机械检查才抓得到的失效

选项被逐条裁到 48 token（`common.py:244-249` 的 `truncation=True, max_length=48`），
head 预算不够时还会整体按 `per = max(4, (head_max_len - 16) // len(opt_ids))` 再收
（`common.py:252-258`）。于是**两个共享前缀的选项可能被裁成同一个 token 跨度**。

`common.py:266-271` 记下了为什么不能靠事后检查补救：

> Two options that share a prefix can come out of the cut as the same token span: the marker
> count still matches the option count, so the guard in `Agent._encode_state` passes and
> nothing downstream can tell that the question lost the ability to name them apart. Counted
> on the capped option ids, before assembly: re-slicing the finished sequence cannot close
> the last option's span -- it runs on into the serialized state, which differs per request,
> so the last option always looks distinguishable however it collided (#538).

所以判据在**裁剪后的选项 id 上**先算出来，作为 `build_head` 的第三个返回值带着走：

```python
return ids, markers, {
    "options": len(opt_ids),
    "options_distinct": len({tuple(o) for o in opt_ids}),
    "tokens_per_option": per_option,
}
```

`options_distinct < options` 就是槽位塌陷。这条在服务侧**会拒绝请求**（`agent.py:1161-1174` 直接抛
`ValueError`，说明「只有 %d 个选项标记放进了 max_len=%d」），训练侧**会丢弃该样本**
（`train.py:309-310` 的 `return None, "options_collapsed"`）。

这是 RYOTIDE §13 不变量 #1「读位置必须落在标记上」在 Laya 里的对应物，而且 Laya **有诊断量**
（`options_distinct` / `tokens_per_option`）——这一点比 Jeff 强，Jeff §13 自陈「结构性成立，但无诊断量」。

---

## 3. 读位置：没有标定，位置是训练时就固定下来的

RYOTIDE 花了一整节（其 §3）在运行期扫 `SUFFIXES × PATTERNS`，用 marker mass 挑读位置，还记下
「同一条 prompt 单独测得 0.9988、跨集合却是 0.0001」的教训。

**Laya 没有这一步。** 读位置不是一个需要探测的量，而是**构造出来的**：`markers` 在 `build_head`
里被显式记录，`marker_pos` 一路传到 forward。位置不是「猜模型会在哪里写答案」，而是「我就在这里放了一个
`[MASK]`」。

代價与 RYOTIDE 恰好互换：

| | RYOTIDE | Laya |
|---|---|---|
| 换模型 / 换模板 | 重新标定即可 | **不行**——布局是权重的属性 |
| 布局在哪里 | 运行期决定 | `option_layout` 写在 checkpoint config 里 |

`common.py:311-326` 的 `uses_parallel_layout` 说明了为什么布局必须进 config：

> The layout is a property of the weights -- a checkpoint trained on one layout reads the
> other as a different input -- so it lives in the config, defaulting to the sequential layout
> every published checkpoint was trained on.

于是 `agent.py:1209-1214` 有一条硬拒绝：`option_layout='parallel'` 的 checkpoint
**只能**用 eager 后端跑，fast/compile 路径会拿着五输入的顺序布局跑这个 batch 而不带它的 mask——
那不是慢，是错。

---

## 4. 一次前向：读 K 个槽位，而不是最后一个位置

这是与 RYOTIDE 最根本的结构性分歧，也是本仓与另外四篇都不同的地方。

`common.py:609-658` 的 `DecisionModel.forward` 是全部：

```python
h = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
h = h + self.type_emb(qtype)[:, None, :]           # 题型嵌入，3 种题型共用一套权重
if self.head is not None:                          # 2 层 TransformerEncoder，带 padding mask
    for layer in self.head.layers:
        h = layer(h, src_key_padding_mask=pad)
idx = marker_pos.clamp(min=0)[:, :, None].expand(-1, -1, h.size(-1))
m = torch.gather(h, 1, idx)                         # ← 每个选项取一个位置的隐状态
logits = self.scorer(m).squeeze(-1).float()         # ← 每个选项压成一个标量
logits = logits.masked_fill(~marker_mask, -1e4)
```

逐点对照：

- **`gather` 而不是取末位**。RYOTIDE 的 `logits_at_end` 只取 `[0, -1, :]`，因为那个位置是模型「将要写下答案」的地方；
  Laya 取的是 K 个 `[MASK]` 位置。**读的位置数从 1 变成 K**，但这 K 个位置在**同一条序列、同一次前向**里。
- **不物化词表维度**。RYOTIDE 要小心 248k 词表的 `[1, n, vocab]`（其 §4 记了 3.7k token 会产生约 3.7 GB logits）；
  Laya 的 `scorer` 是 `LayerNorm → Linear(d,d) → GELU → Linear(d,1)`（`common.py:597`），
  输出直接是 `(batch, K)`。**两边都避开了词表维度，但避的方式不同**：RYOTIDE 靠只取一行，Laya 靠根本不投影到词表。
- **`type_emb` 让三种题型共用一套编码器**（`common.py:596` 的 `nn.Embedding(3, d)`）。题型是一个加到每个位置上的向量，
  不是 prompt 里的一句话。RYOTIDE 靠 prompt 文本区分题型。
- **abstain 走独立的 `act_head`**。`logits` 之外还有第二个输出（`common.py:657` 的 `act_logits`），
  特征里包括 top1、top1−top2、归一化熵和 `k / 255.0`（`common.py:650`）。
  它在 `agent.py:1365` 被成形为每个答案的 `action.act_probability`。

---

## 5. 掩码 + softmax：决策本身

只有两步：

```python
logits = logits.masked_fill(~marker_mask, -1e4)      # common.py:634
p = np.exp(z - z.max()); p = p / p.sum()             # agent.py:1346-1347
```

- **无效槽位是 `-1e4` 而不是 `-inf`**。这与掩码 softmax 等价（`exp(-1e4)` 下溢到 0），
  但它让填充行也有一个有限分布——`backends/base.py:65-66` 明确解释了为什么：全掩码行会变成 NaN，
  而**批里有一个 NaN 就足以毒掉 CUDA graph 的 replay**。RYOTIDE 用的是 `-inf`，
  因为它不批量跑图。
- **`k` 被 clamp 到最小 2**（`common.py:637` 的 `marker_mask.sum(-1).clamp(min=2)`），
  熵与 act 特征都按这个 k 归一化。单选项题（`k=1`）在 `common.py:641-649` 有专门分支，
  因为 `p.topk(2, ...)` 在只有一列时没有第二个可选项，会直接抛。
- **一个 NaN 级的坑被注释留下**：`agent.py:1345-1347` 是**先除温度再 softmax**，
  而 RYOTIDE 是在**已经 softmax 过的概率上**重做 `softmax(log p / T)`（其 §9.2）。
  两者数学等价，但顺序不同——RYOTIDE 那样做在 `p=0` 时 `log` 需要 epsilon 兜底。

---

## 6. 顺序偏置：训练时消解，而不是运行时平均

JevBench 记过 open-alternative-jev 在 `A. yes, B. no` 上 72%、顺序反过来 21%。三条路线的应答方式完全不同：

| | 做法 | 代价 |
|---|---|---|
| RYOTIDE | 跑多个 order，折叠回真实标签后**概率平均** | N 倍读（靠共享前缀压到「几乎免费」） |
| Open-Jev | **pointwise**，顺序不是变量 | N 倍前向 |
| Jeff | 只给 4 种 yes/no 渲染**开关**，自己不平均 | 去偏推给训练或评测方 |
| **Laya** | **`parallel` 布局**：让顺序在结构上不可见 | 布局进 config，换布局等于换权重 |

`common.py:279-305` 的 `parallel_layout` 是这套做法的实现：

```python
position_ids, option_ids = list(range(start)), [0] * start
for s, (a, b) in enumerate(spans):
    position_ids += range(start, start + b - a)      # ← 每个选项都从同一个位置开始
    option_ids += [s + 1] * (b - a)                  # ← 用槽位号而不是位置号区分
```

配套的 `parallel_option_masks`（`common.py:329-347`）再切断选项之间的互相注意：

```python
other_option = (option_ids[:, :, None] > 0) & (option_ids[:, None, :] > 0) \
    & (option_ids[:, :, None] != option_ids[:, None, :])
full = keys & ~other_option
```

docstring 把动机和实测都写下来了（`common.py:280-288`）：

> In the sequential layout option `s` sits at positions after option `s-1`, and every option
> attends to every other, so where an option is listed changes its embedding: the slot logits
> of five identical options spread by 3.59 (`tests/test_option_order.py`). Here every option
> starts at the same position... Together with `parallel_option_masks`, which stops an option
> from attending to another, reordering the options only reorders the marker embeddings.

**3.59 是「五个完全相同的选项，仅因排列不同而产生的 logit 离散度」**——这就是顺序偏置在**没有**这套布局时的量级。
滑动窗口层也跟着改：窗口按 `position_ids` 而非序列下标度量（`common.py:345-346`），
否则 `option_ids` 相同的窗口会让 slot 0 比 slot 3 多看到一些 state。

---

## 7. 多问题共享一次 state

一个请求里同一段 state 问多个问题，四篇的处理各不相同。Laya 的答案在 `Agent._encode_state`：

```python
state_ids = encode_text(self.tok, serialize_state(state)...)["input_ids"]   # 只分词一次
for qid in ids:
    seq, markers, stats, state_stats, *layout = build_sequence(
        ..., state_ids=state_ids, ...)                                      # 逐题复用
```

`common.py:170-172` 写明了这个参数为什么存在：

> `state_ids` lets a caller tokenize the shared state once and reuse it across every question,
> instead of re-serializing and re-tokenizing the same document per question.

但**计费口径与 RYOTIDE 相反**：RYOTIDE 的 `run_many()` 把共享 state 只算一次（其 §7）；
Laya 每个问题都是一条**完整序列**（state 是序列的一部分，不是可选前缀），所以 `input_tokens`
**逐题累计**——与 Jeff §7、Quyet §8 相同。

编码器这里还有一层 RYOTIDE 没有的代价，`common.py:425-428` 说得很直接：

> The cost is not the capping, it is the shape of the request: 80% of every sequence is the
> question, and because the encoder is bidirectional the head cannot be computed once and
> reused -- its representations depend on the state it is paired with.

**因果模型可以复用前缀的 KV，双向编码器不能**：每个 token 的表示都依赖整段 state，换个 state 就得重算。
这是「共享 state」在两条路线里含义不同的根本原因。

长文档走 `predict_long`：把 state 切窗扫描。`common.py:416-424` 记了一组实测——
英文 checkpoint、100 个四词选项：`room` 只剩 102（`max_len=512`），窗口从 312 被压到 102，
扫描从 11 个窗口变成 36 个，**1902 ms → 5858 ms，3.1 倍**。

---

## 8. 路由：本仓独有的一个维度

另外四篇都没有这一层。Laya 每次请求要**先决定用哪个 checkpoint**，而这件事本身**不跑模型**。

`router.py:1212` 的 docstring 一句话说清：

```python
"""Decide which checkpoint to use, without loading or running anything.
```

三个 canonical 名字（`router.py:52-56`）+ 一组别名（`router.py:80-85`）：

| canonical | 别名 |
|---|---|
| `english` | `en`, `laya`, `default` |
| `multilingual` | `multi`, `ml`, `laya-multilingual` |
| `typed-decisions` | `typed`, `typed_decisions`, `decisions` |

优先级（`router.py:1214-1215`）：

```
explicit `model` > explicit `task` > detected workflow (opt-in) >
explicit `lang` > `lang_guess` > detected script/language > default
```

默认路径走 `det = analyse(state)`（`router.py:1264`），判据是**脚本**而不是语言：

```python
if det["script"] == "unknown":
    key = self.default
elif det["script"] != "latin":
    key = "multilingual"      # 非拉丁脚本，英文 checkpoint 读不了
elif not det["is_english"]:
    key = "multilingual"
```

脚本判定的重要性有一条实测背书（`README.md:650`）：英文 checkpoint 在非拉丁脚本上会**塌陷**——
高棉语得分 **0.000 准确率、0.952 置信度**（原始温度下的数字；经 clamp 后发布为 0.705）。
README 的结语是 *Because the model stays confident while being wrong, confidence gating cannot save you.*
**模型一边错一边保持自信**，所以弃权门控救不了这个失效——路由必须先看脚本，这不是优化，是避免一个自信的错误答案。

`lang.py:1` 自陈是 `Dependency-free language/script detection`，只 import `re` 和 `unicodedata`
（`lang.py:13-14`）。脚本检测是**精确的 Unicode 区间查表**（`lang.py:20-46` 约 25 个区间）；拉丁语系的语种猜测是**停用词 + 变音符启发式**，且刻意保守——`lang.py:474` 要求
`best >= max(2, en + 2)`（必须比英文高出余量、且至少命中一个别的语言没有的词），
短输入按设计返回 `None`（`lang.py:455-457`）。

`typed-decisions` **从不自动选中**（`router.py:26-28`）：

> it is fine-tuned on four specific synthetic workflows and should not be a silent default.

这是个值得记的设计立场：一个在窄分布上微调的 checkpoint，宁可让它被显式点名，也不要让它成为沉默的默认值。

---

## 9. 服务化：fail-loud 到了罕见的密度

`serve.py` 是 FastAPI，三条路由：`GET /health`（`serve.py:1182`）、`POST /v1/systemone`（`serve.py:1248`）、
`POST /v1/systemone/batch`（`serve.py:1345`）。`mcp/server.py` 是 **stdio** MCP 服务，不是 HTTP。

**白名单 + 显式拒绝表**（`serve.py:157-158`）：

```python
BODY_CONTROLS = ("model", "max_len", "head_max_len", "task", "lang", "lang_guess", "min_confidence")
BODY_REFUSALS = ("hooks", "on_predict_start", "on_predict_end", "hooks_raise", "hooks_timeout")
```

被拒的那几个是**在服务进程内跑的钩子**，`serve.py:494-497` 的解释是：它们「run inside the server process
and cannot be sent to this endpoint」。两张表由 `tests/test_serve.py` 拿 `inspect.signature` 钉住，
所以「新增一个控制项」必须先在「转发还是拒绝」这条线上选边，测试才会绿。

**并发是串行的，而且是刻意的**：

```python
pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="laya-infer")   # serve.py:1079
```

`serve.py:1072-1078` 的理由是「Inference is synchronous torch... One worker, because one forward pass
at a time is what a single CPU or GPU Agent wants」。与 RYOTIDE 用一把锁包住决策是同一个结论、不同的理由。

准入另有一道闸：`DEFAULT_MAX_CONCURRENT = 16` 的信号量，超了返回 **503** 而不是排队
（`serve.py:134-136`：每个请求在推理前最多缓冲 `MAX_BODY_BYTES`，不设上界的话「many concurrent
near-cap requests OOM the worker even though every request is individually valid (#330)」）。

**拒绝而不是截断**：state 超过 50000 字符直接 413（`MAX_STATE_CHARS`），`max_len`/`head_max_len`
超过 8192 直接 422（`DEFAULT_MAX_TOKEN_BUDGET`）。真正的 token 级截断发生在模型内，且**上报不告警**：
`usage` 里带 `state_tokens_dropped` / `truncated` / `truncated_questions`（`agent.py:1548-1558`）。

### 一批「看似合理却是错的」的守卫

这个仓的注释密度不常见，而且几乎每条都对应一次真实故障。挑几条最狠的（均逐字引自源码）：

| 位置 | 内容 |
|---|---|
| `serve.py:1162-1168` | 认证用 `hmac.compare_digest` 比较 **bytes 而非 str**：Starlette 按 latin-1 解头部，`Authorization: Bearer s\xe9cret` 是合法请求，却让比较本身抛异常——**任何未认证客户端一个字节就能打出 500**。 |
| `serve.py:785-791` | 用 `repr` 估 state 长度**低报了近一倍**：`repr` 把值里的 `"` 算作 1 个字符，而 JSON 转义成 `\"` 两个。`{"body": '"'*49988}` 的 `len(str())` 正好 50000，通过了闸门，实际序列化成 99988。 |
| `serve.py:833-838` | `serialize_state(None)` 就是 `json.dumps(None)` = 四个字符 `null`，于是**缺 `state` 字段的请求会被当成「对字面文本 null 的决策」**回答，HTTP 200，约 0.94 置信度，与显式发送 `"state": "null"` 逐字节相同。 |
| `serve.py:766-772` | 长度下界只认**精确类型**：`dict` 子类可以覆写 `values()` 而 `json.dumps` 读真实条目，会让下界**超过**真值——实测一个 13 字符的 state 被判成 `60000 > 50000`。 |
| `serve.py:1265-1271` | 孤立代理对（`\udXXX` 无配对）是**合法 JSON 但无法 UTF-8 编码**，会在 `build_sequence` 里抛 `TypeError`，于是客户端的错回来变成 500「inference failed」。 |
| `serve.py:942-945` | 走代理对用**显式栈而非递归**：`json.loads` 接受的嵌套深度远超 Python 递归上限，递归走法把一个解析器能处理的请求变成 `RecursionError`——**一个 500 换成另一个 500**。 |
| `serve.py:963-969` | `Content-Length` **不能**是唯一的体量闸门：它是客户端自己写的值，且在 `Transfer-Encoding: chunked` 下完全缺失（HTTP/2、HTTP/3 根本没有这个头）。 |
| `serve.py:334-336` | 模型 id 打错时**曾经静默自动路由**，用路由碰巧选中的 checkpoint 回答——「which is a wrong answer」。现在这类请求是 422。 |
| `__init__.py:5-22` | 在 import 任何东西之前 `setdefault("USE_TF", "0")`：transformers 4.x 的某条导入链会 `import tensorflow`，在「装了 TF 但加载不了」的环境里那是**原生崩溃**（`Fatal Python error: Bus error`），不是一个异常。 |

`serve.py:81-84` 还说明了为什么失败一律报固定 500：

> A failed inference is reported to the client as a fixed 500 so nothing about paths, weights or
> memory state leaks, which leaves the server log as the only place the actual cause can appear.

---

## 10. 置信度从哪来，又怎么被改写

### 10.0 两个都叫 confidence 的量

这是 Laya 最容易读错的地方：**同一条答案里有两个不同的数**。

| 字段 | 公式 | 是什么 | 温度会改它吗 |
|---|---|---|---|
| `answer_confidence` | `max(p)`（`common.py:779-799`） | 报出那个答案上的概率质量 | **是**——温度拟合拟合的就是它，ECE 量的也是它 |
| `confidence` | `1 - H(p)/log k`（`common.py:802-812`） | 整个分布的集中度 | 否 |

`common.py:802-806` 对第二个的定性是「Useful, but not calibrated: it is not what temperature scaling
fits and not what the reported ECE measures」。`confidence.py:24-28` 进一步禁止它们互相顶替：

> So this deliberately does **not** fall back to it: reporting the entropy number under the
> `answer_confidence` name would be worse than reporting nothing, because the name is what a
> caller filters on.

服务侧还会注入第三个：`x_jev_confidence`（`serve.py:241`），用 TypeSafe 自己的公式重算一遍
（`confidence.py:38-76`）。**实测**（本文实跑，3 选项 `{0.62, 0.25, 0.13}`）：

```
answer_confidence (max p) = 0.6200
jev_confidence            = 0.4300      # (max(p) - 1/n) / (1 - 1/n)
Laya confidence (entropy) = 0.4200      # 巧合地接近，但公式完全不同
```

后两者在这条样本上数值接近，纯属巧合——它们的定义毫无关系。`confidence.py:41-43` 的警告是：
「a threshold carried over from Jev selects about twice as much on Laya」。

### 10.1 谁在改发布数值

| 环节 | 代码 | 改数值？ | 改 argmax？ |
|---|---|---|---|
| 训练（RLCD，§11） | `train.py:499-518` | **是** | **是** |
| 温度 | `agent.py:1345` | 是 | 否，按构造成立 |
| 逐语言温度 | `agent.py:1342-1344` | 是 | 否 |
| 分箱（binning map） | `agent.py:1359-1362` | 是（只改 `answer_confidence` 的报告值） | 否 |

与 RYOTIDE 的差别是根本的：RYOTIDE **零训练**，所以「发布出去的概率数值只被 temperature 改变」是一条**可验证的强性质**；
Laya 训练本身就在改数值、也可能改 argmax，温度只是**训练之后的二次校正**。这与 Jeff、Open-Jev 同类。

### 10.2 温度：拟合了，但**没有接受/拒绝门槛**

拟合器是 LBFGS 最小化 NLL（`calibrate.py:101-138`）：

```python
log_t = torch.zeros(1, requires_grad=True)
opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=100)
def closure():
    loss = -(t_mat * torch.log_softmax(z_mat / log_t.exp(), -1)).sum(-1).mean()
    ...
return clamp_temperature(fitted, TEMP_MIN, TEMP_MAX)
```

两个粒度：逐题型标量（`temperature`，样本少于 `MIN_TYPE_N = 10` 不拟合）与逐选项数分桶
（`temperature_by_options`，少于 `MIN_BUCKET_N = 2000` 不拟合，`calibrate.py:37-39`）。
分桶的键就是 `temp_bucket`（`common.py:815-817`）：`choice:2` / `choice:3-5` / `choice:6-10` / `choice:11+`——
**与 Quyet 的 `size_bucket` 是同一套思路**，且都在推理时按实际的 k 取桶（`agent.py:1341`）。

拟合用的**不是训练集**：`train.py:474-488` 在训练前就把 calibration slice 切出去了
（`calib_frac=0.1`、`calib_max=400`、`calib_seed=20260922`），注释写明「Fitting temperatures on items
the run has trained on measures the fit, not the calibration」。

**但发布路径没有接受 / 拒绝判据**：`train.py:1138` 的 `fitted = fit_temperature_map(records)`
用默认的 `compute_ece=False`，于是它在**全部** record 上拟合并把结果写进 `rl_agent_config.json`
（`train.py:1219-1223`）。唯一的约束是样本量下限与 `clamp_temperature`。
检测到的问题只发 `RuntimeWarning`（`train.py:1140-1142`），**不阻断发布**。

这与 RYOTIDE §9.2 的对照很尖锐：RYOTIDE **预先注册**了「留出集 ECE 必须下降才采纳」的规则，
因此 Qwen3.5-4B 拟合出 1.592 却被拒、保持 1.0；Laya 会把拟合值直接发出去。

### 10.3 clamp：一条承认自己「没有校准、只是限制了损害」的注释

`common.py:820-825` 是本仓最诚实的一段：

```python
# A fitted temperature below 1 sharpens the logits instead of softening them. The shipped
# `choice:11+` bucket is 0.1006, which multiplies them ~10x: a 0.24 top probability is published as
# 0.99, so a caller gating on confidence is told a coin flip is a certainty. No honest calibration
# needs to sharpen this hard, so refuse to apply one that does.
TEMP_MIN = 0.5
TEMP_MAX = 5.0
```

而 `common.py:789-792` 明说 clamp 之后的后果：

> `clamp_temperature` refuses it and applies `TEMP_MIN` (0.5) instead, which still sharpens 2x and
> publishes that same 0.24 as 0.4993 -- so a threshold applied to these numbers still selects below
> model accuracy (issue #394). **The clamp bounds the damage; it does not calibrate anything.**

**「限界了损害，但没有校准任何东西」**——这句话值得与四处「confidence 不是校准正确率」的声明并读
（Jeff 的 serving rule、Open-Jev 的 `choice_confidence`、Quyet 的 `confidence = max(p)`、RYOTIDE 的 `margin`）。
四篇都在同一个位置立了同一块牌子。

### 10.4 逐语言温度：多语言带来的新一层

`resolve_lang_temperatures`（`common.py:846-886`）允许 `{"de": {"temperature": [...], "temperature_by_options": {...}}}`。
`agent.py:1342-1344` 在解码时按 `lang` 的**主语言子标签**（`lang.split("-")[0].lower()`）覆盖温度。
注意覆盖分桶时**跳过 binning map**（`agent.py:1358-1362` 的 `lang_override`）——否则两套校准会叠在一起。

### 10.5 弃权门控

`min_confidence` 是 opt-in 的（`confidence.py:249`），且可以按选项数分桶给不同阈值
（`confidence.py:123-146`，键就是 `temp_bucket` 的那 12 个）。它报的是**三态**而不是布尔
（`confidence.py:242-246`）：

```
passed | abstained | unevaluated
```

`unevaluated` 是布尔表达不了的那个：闸门跑了，但这条答案没带可用的置信度，于是闸门**无法判断**。
`confidence.py:263-266` 的理由是「Reporting that as a pass is the same lie as reporting it as a flag」。

**实测**（本文实跑，阈值 0.7）：

```
a (answer_confidence 0.90) -> passed
b (answer_confidence 0.55) -> abstained      low_confidence=True
c (无 answer_confidence)   -> unevaluated
```

未设 `min_confidence` 时**什么都不写**——`abstention` 字段的**存在**本身就是「跑过闸门」的信号
（`confidence.py:267-272`），不是靠第四个取值。

---

## 11. 训练：RLCD 到底是什么

README 说的是「trained with reinforcement learning against strictly proper scoring rules (RLCD)」。
读代码后可以精确到：**它是一步 bandit 上的策略梯度 + 软交叉熵，奖励来自一个严格 proper 的打分规则**。

默认损失 `rlcd_loss`（`train.py:499-518`）：

```python
eps = torch.randn((samples,) + logits.shape, ...) * sigma * fmask     # samples=4
eps = (eps - eps.sum(-1, keepdim=True) / k) * fmask                   # 去掉均值，留在单纯形切空间
z = logits.detach().unsqueeze(0) + eps
q = torch.softmax(z.masked_fill(~mask, _MASKED), -1)
with torch.no_grad():
    r = proper_reward(q, target.unsqueeze(0), qtype, fmask, w_sph=w_sph, w_rps=w_rps)
    adv = (r - r.mean(0, keepdim=True)) / (adv.std() + 1e-6)          # 标准化优势
logp = -(((z - logits.unsqueeze(0)) ** 2) * fmask).sum(-1) / (2 * sigma ** 2)
return -(adv * logp).mean() + soft_ce_loss(logits, target, mask)       # ← 第二项是普通软 CE
```

三点必须说清，否则「RL」这个词会被读大：

1. **确实有策略梯度项**。对 logit 加噪 → 用 proper 打分规则给每个样本发奖励 → 标准化成优势 →
   用优势给高斯对数密度加权。这就是 GRPO 式的 REINFORCE + baseline。
2. **但没有环境、没有价值网络、没有 rollout**。奖励信号**只有**一个 proper 打分规则，
   是「对 logit 扰动的一步 bandit」，不是序列决策。
3. **`logits.detach()`（`train.py:511`）**：采样那支不回传到 logits，梯度只走噪声路径 `logp`；
   真正把监督信号送进网络的是第二项 `soft_ce_loss`（`train.py:493-496`）。

奖励本体在 `common.py:719-745`：**log score + 球面分数（spherical）+ `score` 题型额外减 RPS**。

```python
r = log_score + w_sph * sph
if is_score.any():
    r = r - w_rps * rps * is_score
```

三者都是严格 proper 的（RPS 带序数 CDF，只在 `score` 上生效，因为等级有序）。
`train.py` 传 `w_sph=0.75, w_rps=1.0`，比 `proper_reward` 自己的默认 `w_sph=0.5` 更重球面项。
噪声 `sigma` 从 0.4 线性退火到 0.1（`train.py:101-102`、`521-523`）。

**目标不是 one-hot**。`target_from_gold`（`train.py:167-193`）从 benchmark 的 gold 分布取，
归一化后直接当软目标；只有 CSV 那条路径是 one-hot，且 `label_smoothing` 默认 0.0。
这一点与 Open-Jev 一致（软目标），与 RYOTIDE 无从比较（它不训练）。

一个诚实的自我否定也记在代码里：`train.py:9-10` 说 `loss="soft-ce"` 这条路「#741 measured no gain
from the extra term on the typed-decisions split」——**加了策略梯度项，在那个 split 上没测出收益**。

---

## 12. 四条不变量的落位

RYOTIDE §13 的四条，在 Laya 上**两条落位、两条根本不适用**——这正是 §0 那条「骨架不同」的直接后果。

| RYOTIDE §13 不变量 | Laya 的对应物 | 状态 |
|---|---|---|
| **#1** 读位置必须落在标记上（诊断量 `marker_mass`） | 读位置**就是**构造出来的 `[MASK]` 槽位 | **结构性成立，且有诊断量**：`options_distinct` vs `options`、`tokens_per_option`（§2） |
| **#2** 前缀与标记表面形式耦合，必须成对扫 | 没有文本前缀、没有标记表面扫描 | **不适用**：标记是 token id，不是字符串，不存在「表面形式」这个自由度 |
| **#3** 只读最后一个位置 | 读**每个选项的**一个位置，K 个位置一次前向 | **改写**：不是「只读一个位置」，而是「每条序列只读它自己的 K 个槽位」；不物化词表维度这条**同样成立**，但机制不同（§4） |
| **#4** 比较概率或 ECE 之前先确认 temperature | T 在 `rl_agent_config.json`，且分题型 / 分桶 / 分语言 | **成立，且更复杂**：要确认的不止 T，还有**两个 confidence 量**与 `x_jev_confidence` 用哪个（§10.0） |
| （新增：训练侧） | **布局是权重的属性** | `option_layout` 写死在 config；换布局等于换 checkpoint，且 fast/compile 后端拒绝跑 `parallel`（§3） |
| （新增：口径） | **归一化常量** | act 特征用 `k / 255.0`（`common.py:650`）；温度/阈值的分桶边界是 `2 / 3-5 / 6-10 / 11+`——跨实现比数字前要对齐这两个口径（§13） |

---

## 13. 文件对照

| 文件 | 行数 | 职责 |
|---|---|---|
| `laya/common.py` | 945 | 序列构造（`build_sequence` / `build_head` / `parallel_layout`）、`DecisionModel`、`proper_reward`、两个 confidence、温度 clamp 与分桶、`collate_items` |
| `laya/agent.py` | 2151 | `Agent`：加载与校验、`_encode_state`、`_infer`（含 OOM 降级）、`_decode_answers`、`predict_batch` / `predict_long` |
| `laya/router.py` | 1893 | 逐请求挑 checkpoint：别名、优先级、纯 Python 启发式路由 |
| `laya/lang.py` | 867 | 免依赖的脚本 / 语言检测（Unicode 区间表 + 停用词启发式） |
| `laya/serve.py` | 1510 | FastAPI 服务：三条路由、白名单与拒绝表、准入与串行化、体积与代理对守卫 |
| `laya/confidence.py` | 302 | 免 torch 的置信度校验、分桶阈值、**三态**弃权门控 |
| `laya/structured.py` | 457 | JSON schema / pydantic → 问题集；`decide` / `decide_batch` / `DecisionResult` |
| `laya/calibrate.py` | 707 | 温度（LBFGS/NLL）、弃权阈值、分箱图的拟合与存取 |
| `laya/train.py` | 1331 | 训练循环：`rlcd_loss` / `soft_ce_loss`、软目标、calibration slice、发布 config |
| `laya/backends/` | — | `eager` / `compile` / `tilelang` / `onnx` 四种 forward，一份 contract（`base.py:79-142`）与填充策略（`:21-38`） |
| `laya/revisions.py` | — | 供应链完整性：`PINNED_REVISIONS` 三个 reviewed SHA、`LAYA_REVISION=reviewed`、SHA-256 校验 |
| `laya/presets.py` | 210 | 五个内置预设（triage / email / guard / moderation / router） |
| `laya/mcp/server.py` | 545 | stdio MCP 服务（8 个工具），**不加载模型**的薄客户端形态 |

---

## 14. 亲手跑一遍

**不加载模型、不需要 torch 即可复现本文的全部实测**（路由 / 语言 / 置信度层是刻意 torch-free 的）：

```bash
cd /path/to/laya
python3 - <<'PY'
import sys; sys.path.insert(0, '.')
import laya
from laya import presets, lang
from laya.confidence import bucket_keys, apply_confidence_gate

print("version", laya.__version__)                      # 0.4.4，且未加载 torch

for name in ("triage_questions", "email_questions", "guard_questions",
             "moderation_questions", "router_questions"):
    qs = getattr(presets, name)()
    print(name, len(qs), [q["type"] for q in qs.values()])

for s in ("My payment failed twice", "我的订单还没到", "Мой заказ не пришёл"):
    print(repr(s), lang.analyse(s)["script"])           # latin / han / cyrillic

print("阈值桶:", sorted(bucket_keys()))                 # 12 个
rs = [{"answers": {"a": {"type": "choice", "probabilities": {"x": .9, "y": .1},
                         "answer_confidence": .9}}}]
apply_confidence_gate(rs, 0.7)
print(rs[0]["answers"]["a"]["abstention"])              # passed
PY
```

**实测输出**（本文实跑，节选）：

```
version 0.4.4
triage_questions 5 ['choice', 'noul', 'score', 'noul', 'noul']
email_questions 5 ['choice', 'noul', 'noul', 'score', 'noul']
guard_questions 5 ['noul', 'noul', 'noul', 'score', 'choice']
moderation_questions 5 ['noul', 'noul', 'noul', 'noul', 'score']
router_questions 4 ['score', 'choice', 'noul', 'noul']
'My payment failed twice'     latin
'我的订单还没到'                 han
'Мой заказ не пришёл'         cyrillic
阈值桶: ['choice:11+', 'choice:2', 'choice:3-5', 'choice:6-10', 'noul:11+', ...]   # 共 12 个
passed
```

语言检测的耗时也实测了（同一台机器，热态）：拉丁 0.053–0.821 ms、汉 0.087 ms、西里尔 0.019 ms、希伯来 0.027 ms。
README 第 650 行声称的「<0.5 ms pure Python」与这个量级一致。注意 `analyse` 只给脚本，
拉丁语系的 `language` 字段在短输入上按设计返回 `None`（`lang.py:455-457`）——上面三条都没给出语种，这是预期行为，不是 bug。

**跑模型前向**（需要 torch、CUDA/CPU 均可，以及从 Hub 取权重）：

```bash
pip install laya            # 或 pip install "laya[fast]" 走 tilelang
laya "My payment failed twice" --preset triage
```

第一个请求会下载 checkpoint。建议同时钉住版本以便复现——`revisions.py:26-30` 有三个 reviewed SHA：

| repo | reviewed SHA |
|---|---|
| `convaiinnovations/laya` | `55cf4c4ebb4ebe31b2550e8bdf3bd21b99753851` |
| `convaiinnovations/laya-multilingual` | `e4e9ddf21a7b1903b7acffd8814ad4307bf63a67` |
| `convaiinnovations/laya-typed-decisions` | `1a793eb568e6718f15941d08f85432581df534e3` |

```bash
export LAYA_REVISION=reviewed        # 或直接给 commit
export LAYA_SHA256_DIGESTS='{"model.safetensors": "<sha256>"}'   # 可选，加载前校验
```

看结果里三件事：

- `usage.output_tokens` **必须是 0**——这一条确认读确实发生了（值来自 `agent.py:1550` 的硬编码）；
- `answer_confidence` 与 `confidence` **是两个数**，别混用（§10.0）；
- `usage` 里若出现 `truncated: true` / `state_tokens_dropped > 0`，说明 state 被裁过（§9）。

**注意本文没有给出任何概率数值。** 本机无 torch、无 checkpoint，全文的概率相关结论都来自读代码，
包括 §6 的 3.59 与 §7 的 3.1 倍——那两组数字是**源码注释里记录的实测**，不是本文跑出来的，引用时请按
`common.py:280-288` 与 `common.py:416-424` 回源核对。

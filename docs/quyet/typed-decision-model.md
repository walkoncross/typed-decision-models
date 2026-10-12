# Quyet 是怎样把「一个 state + 一个类型化问题」读成一个分布的

> 代码库：<https://github.com/ncchinh/quyet>
> 对应 README 第 4–5 行：*Give it a state (any text, JSON or conversation) and one or more typed questions; it picks
> one option per question and returns calibrated probabilities. Nothing is generated.*
> 本文按 `src/quyet/` 的实际代码逐环节说明这句话的实现方式；文中出现的文件名与行号（`answers.py:30`、
> `questions.py:38`、`llm/runtime.py:87-91` 等）均相对该仓库根目录。本文还在每个环节与
> [`docs/ryotide/typed-decision-model.md`](../ryotide/typed-decision-model.md) 的路线做一次对照——两条路线要解决的问题相同，采纳的机制几乎处处相反。

## 0. 与 RYOTIDE 的分叉点

RYOTIDE 的立场是「**读**现成 LLM，不训练」；Quyet 的立场是「**训**一个决策模型，runtime 只负责把答案读出来」。
一句话的差别，落到代码上是四处结构性不同：

| 维度 | RYOTIDE | Quyet |
|---|---|---|
| 权重来源 | 公开 LLM（Qwen3.5 / Gemma 4），全程 frozen | 自训权重（`chinhnc/Quyet-1.0-*`），HF 上发布 |
| 「typed」从哪来 | 借来的 tokenizer：标签必须能落到**单 token**，落不到就映射成字母标记 | 训练时就固定的槽位：encoder 用 `[MASK]` 槽位，LLM 用字母 `A..J` |
| 读位置的确定 | **运行期标定**（扫 9 个 answer prefix × 2 个标记表面形式，取 marker mass 最高） | **训练期固定**，运行期只有一个常量：`prompt_version` |
| 后端 | MLX + torch 两个后端，一套 `jevbench_adapter` 适配器 | encoder（ModernBERT + 打分头）与 LLM（字母 logits）两条**并行** runtime，只有问题解析/答案成形/温度三处共享 |
| 置信度来源 | 运行期读位置标定 + 逐 preset 钉死的 temperature（拟合自 benchmark 留出集） | 逐模型、逐题型的 temperature，写在模型自带的 `quyet_config.json` 里 |

共同点只有一条，而且是最本质的那条：**产出分布的方法完全一样**——一次前向、只读最后一个位置、
把 logits 掩码到已知选项集、softmax 重归一化。`answers.py:30` 的 `"output_tokens": 0` 是这条立场最直白的证据。

## 1. 端到端数据流

```
                        ┌──────────────── base.py / cli.py ────────────────┐
                        │ predict(state, questions) → predict_many([...])  │
                        │ 全请求一次性准备；任一请求非法则整体抛 QuestionError │
                        └───────────────────┬─────────────────────────────┘
                                            v
                    ┌───────────────── questions.py ─────────────────┐
                    │ parse_questions → {qid: Decision(type,          │
                    │   instructions, labels, texts)}；2..10 个选项    │
                    └────────┬──────────────────────────────┬─────────┘
                             v                              v
        ┌──────── encoder/runtime.py ────────┐  ┌──────── llm/runtime.py ────────┐
        │ inputs.build_item  单序列布局        │  │ prompt.build_prompt  文本 prompt │
        │  CLS <type> question: … SEP         │  │  State / Question / Options A..J │
        │  {MASK label[:] text}* SEP state SEP│  │  chat template(enable_thinking=F)│
        │ collate → 长度分桶批                │  │ 左填充 → 长度分桶批               │
        │ DecisionModel.forward               │  │ forward(logits_to_keep=1)        │
        │  → 每个槽位一个标量 logit            │  │  → 只取 [A..J] 的 logit           │
        │ masked_fill(非槽位, -1e4)           │  │                                  │
        └──────────────┬─────────────────────┘  └──────────────┬──────────────────┘
                       └───────────────┬───────────────────────┘
                                       v
                    ┌────────── calibration.py ──────────┐
                    │ softmax_t(logits, temperature(...)) │  ← 唯一改发布数值的步骤
                    └───────────────┬─────────────────────┘
                                    v
                    ┌──────────── answers.py ─────────────┐
                    │ choice / score / noul → Jev 响应形状  │
                    │ usage{input_tokens, output_tokens:0} │
                    └──────────────────────────────────────┘
```

## 2. 类型化的入口：`questions.py`

`parse_question()`（`questions.py:38`）是两条 runtime 共用的唯一入口，把 wire 上的 JSON 收敛成一个不可变对象：

```python
Decision(type, instructions, labels, texts)          # questions.py:20
```

三种题型的 `labels` 顺序是**协议的一部分**，不是实现细节：

| type | criteria 输入 | labels | texts |
|---|---|---|---|
| `choice` | `{label: description}` 或 `[label, ...]` | 给定顺序 | 描述或 `None` |
| `score` | `[level 描述, ...]`（低→高） | `"0".."K-1"` | 每级描述 |
| `noul` | `{"true": ..., "false": ...}`（两者都可省） | `("true","false")` | true/false 描述 |

两条硬边界（`questions.py:64-67`）：**至少 2 个、至多 10 个选项**（`MAX_OPTIONS = 10`，`questions.py:17`），
超出**直接报错而不是截断**。10 这个数字与 LLM 侧 `MAX_LETTERS = 10`（`prompt.py:15`）是同一个约束的两端：
字母表只有 `A..J`，第 11 个选项没有槽位可放。这与 RYOTIDE 为 >26 选项做的两 token 代码（`A0..Z9`）形成对照——
Quyet 选择了「不支持」，因为多出来的选项意味着模型没有训练过的槽位，读出来的概率没有意义。

`describe()`（`questions.py:28`）把描述统一成文本：**falsy 值（`None`/`""`/`0`/`False`/`{}`/`[]`）一律等价于「没有描述」**，
其余 JSON 值以紧凑 JSON 渲染。这条 `text or None` 语义是被评估过的路径上的既有行为，测试
（`tests/test_questions.py:70`）把它钉住了。

## 3. 两条 runtime

`hub.py:32` 的 `load()` 读 `quyet_config.json` 的 `kind` 字段分派：`encoder` → `EncoderModel`，
`llm` → `LLMModel`。同一个包、同一个 `predict()` 接口，底下是完全不同的两套模型。

### 3.1 encoder 路径：一条序列、一个槽位一个 logit

**序列布局**（`encoder/inputs.py:3` 的 docstring 就是权威定义）：

```
CLS  "<type> question: <instructions>"  SEP  { MASK  " <label>[:]"  [" <text>"] }*  SEP  <state>  SEP
```

- `build_item()`（`inputs.py:50`）逐选项记录 **marker 位置**、**标签+描述占的 token 跨度** `[start, end)`
  （`inputs.py:59-63`），以及 `q_end` / `st_start` / `st_end`。
- 描述为空时标签**不带冒号**、跨度退化为空（`text is None` 分支），测试断言 `decode == " a:"` 与空跨度（`tests/test_encoder_inputs.py:28`）。
- `Tokens.encode()`（`inputs.py:43`）以 `split_special_tokens=True` 编码，并在结果里出现任何 special id 时**报错**：
  用户文本里的 `[CLS]`/`[SEP]`/`[MASK]` 只能当字面文本（`test_control_strings_stay_text`）。这是防注入的硬保证——
  否则一段 state 就能伪造结构 token 去命中别的选项槽位。

**打分头**（`encoder/modeling.py`）三种实现由 config 的 `head` 字段选择（`make_head()`，`modeling.py:95`）：

| head | 结构 |
|---|---|
| `option_query_v1` | 编码器状态 → `Linear(d, 384)` + 题型 embedding；选项 query = `GELU(Linear(2h,h)[marker; 描述均值])`；1 组（交叉注意力 → CLS/问题/state，自注意力 → 选项之间）；`LayerNorm→Linear→GELU→Linear` 打分 |
| `option_query_wide` | 同上，h = 1024、16 头、2 组 |
| `refine_pool_v1` | 全序列过 2 层 pre-norm Transformer，再取 marker + 描述均值融合打分（结构跟随 Laya，Apache-2.0） |

两个机制值得单独看：

- **`evidence_mask`（`inputs.py:90-93`）只标 CLS、问题 token、（截断后的）state token 为证据。**
  选项块**不是**证据：交叉注意力（`modeling.py:71` 的 `cross(q, x, b["evidence_mask"])`）让选项 query 只去读
  「问题 + state」，选项之间则通过第二个自注意力块互相比较。这样「选项文本」只作为被评分的对象，不会污染证据侧。
- **`span_pool()`（`modeling.py:22`）** 对每个选项的描述跨度做均值池化，空跨度回退到 marker 向量——
  这就是「没有描述」的选项仍然有一个可评分表示的原因。

最后 `DecisionModel.forward()`（`modeling.py:116`）把非槽位位置 `masked_fill(~marker_mask, MASKED_LOGIT=-1e4)`，
再交给 runtime。**注意这里是防御性的**：`encoder/runtime.py:42` 只取 `lg[j, :len(markers)]`，
补齐出来的槽位本来就进不了 softmax，所以 `MASKED_LOGIT` 在实际路径上不会改变任何数值——与 RYOTIDE §9.3 的
`calibration_bias()` 类似，是一段「写了但不在主路径上生效」的代码，区别是这里它只是冗余，而非被绕过。

### 3.2 LLM 路径：prompt 里的字母，就是标签槽位

`prompt.py:62` 的 `build_prompt()` 拼接的是训练时逐字用过的文本：

```
State:
<state>

Question: <instructions>
<按题型的 KIND 指令>

Options:
A. <选项 0 文本>
B. <选项 1 文本>
```

两个 `prompt_version`（`prompt.py:32`，由 `quyet_config.json` 选择）：

| version | 形态 | 差别 |
|---|---|---|
| 1 | 有 `SYSTEM` 消息、结尾 `"Answer with one letter."`、JSON state 缩进 1 格 | 训练时原样（Quyet-1.0-Medium） |
| 2 | 无 system、无结尾句、JSON state 紧凑 | Quyet-1.0-Large 的「diet」prompt；每次决策省约 68 个输入 token，留出集准确率低约 0.3 点，temperature 已重新拟合 |

`letter_ids()`（`prompt.py:76`）在加载时对 `A..J` 逐个断言 `len(encode(letter)) == 1`，否则**抛异常拒绝加载模型**。
这与 RYOTIDE `LabelSet.resolve()` 的 `MultiTokenLabelError` 是同一类检查，区别是 Quyet 在**加载期**就把
「标签落不到单 token」变成不可加载，而不是等到第一次决策。

读法本身只有三行（`llm/runtime.py:87-91`）：

```python
logits = self.model(input_ids=ids, attention_mask=att, logits_to_keep=1).logits[:, -1, :]
sel = logits.float()[:, lid]          # lid = [A..J] 的 token id
out[b] = sel[r, :seqs[b][1]].cpu().numpy()   # 只取前 k 个字母
```

`logits_to_keep=1` 让 transformers 只保留最后一个位置的 logits——与 RYOTIDE 的 `PREFILL_CHUNK` +
「最后一个 token 单独前向」是同一个目的（不物化 `[1, n, vocab]`），只是 Quyet 把它交给库的 `logits_to_keep` 参数，
因为这里每个请求都是**单次完整前向**，没有 KV cache 复用，不需要自己写分块。

`chat template` 上有一个与 RYOTIDE §5 完全同源的坑，两边独立踩到、独立修好：

```python
return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
except TypeError:                     # 模板不接受该参数时回退
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
```

不加 `enable_thinking=False`，读位置会落进 `<think>` 块，概率质量到不了选项字母上。RYOTIDE 是用 `marker_mass`
这个诊断量发现的，Quyet 则是直接在 prompt 构造处写死。

## 4. 两条路径的槽位顺序（一处容易读错的地方）

标签的**最终顺序**在两条路径上一致（`probs[0]` 对 `noul` 永远是 P(true)，`answers.py:20`），但**槽位顺序不同**：

| 路径 | `choice` / `score` | `noul` |
|---|---|---|
| LLM | 字母按 `labels` 顺序 | A = true 描述，B = false 描述（`prompt.py:23`），不折回 |
| encoder | 槽位按 `labels` 顺序 | 槽位 0 = **false**、1 = **true**（`inputs.py:32-33`），靠 `to_label = [1, 0]` 折回 |

encoder 侧的折回发生在 `runtime.py:71-73`：

```python
for c, x in enumerate(canon):
    p[items[qid]["to_label"][c]] = x
```

`choice`/`score` 的 `to_label` 是恒等（`inputs.py:27,30`），所以这段代码只在 `noul` 上真正干活。
读代码时如果只看 `format_answer`，会以为 `noul` 的 `probs[0]` 是「第一个槽位」——实际上两条路径都
显式保证了它是 P(true)，只是保证的位置不同。

## 5. 只读最后一个位置，然后掩码 + softmax + 温度

这一步两条路径共享，代码量极小，但决定了「分布」而非「文本」：

```python
def softmax_t(logits, t):                    # calibration.py:16
    z = np.asarray(logits, np.float64) / t
    p = np.exp(z - z.max())                  # 减去最大值：防溢出，测试覆盖 [1000, 0]
    return (p / p.sum()).tolist()
```

- **输入只有选项槽位的 logit**：encoder 侧是 `lg[j, :len(markers)]`，LLM 侧是 `sel[:, :k]`。词表其余部分
  从头到尾没有被读取，所以「没有东西需要解析，也没有东西可以幻觉」在两套实现里都是结构性成立的。
- **温度在这里、且只在这里生效**，与 RYOTIDE §9.2 的立场完全相同：`softmax(logit/T)` 不改变 argmax，
  只改变分布的陡峭程度（`test_softmax_t` 断言 T=100 时概率被压平）。
- 区别在温度的**来源**：RYOTIDE 的 T 是逐 preset 写死的常量，由离线拟合 + 预先注册的留出集判据决定；
  Quyet 的 T 是**模型自带**的，写在 `quyet_config.json` 的 `temperatures` 里，随权重一起发布。

温度的分桶策略是两条路径唯一不同的地方（`calibration.py:8-13`）：

```python
def size_bucket(k):   # 2 | 3-5 | 6-10 | 11+
def temperature(temps, qtype, k):
    return float(temps.get(f"{qtype}:{size_bucket(k)}", temps.get(qtype, 1.0)))
```

| 路径 | 键 | 原因 |
|---|---|---|
| encoder | `"choice:3-5"` 这样的 `<type>:<桶>` | 同一个打分头服务所有选项数，选项越多分布越平，需要分桶校准 |
| LLM | `"choice"` / `"score"` / `"noul"` 纯题型键 | 字母读法的形状随选项数的变化已经由训练吸收 |

查找顺序是「先查带桶的键、再退到纯题型键、最后 1.0」，所以一个 encoder config 也可以只用题型键
（`tests/test_calibration.py:10`）。**缺键 = 1.0 = 未校准**，这是显式的默认，不是错误。

## 6. 批处理与正确性

两条路径都把**整个 `predict_many` 调用里的所有问题**收集起来统一切批，而不是逐请求串行，因此
batch invariance 是核心正确性要求（`test_encoder_runtime.py:32`、`test_llm.py:53` 都在单条与批量之间逐位比对）。

**encoder**（`inputs.py:101` `length_batches`）：按长度稳定排序后贪心切批，约束是
`max(批内最长, 新条目) × 批内行数 ≤ 32768`（`runtime.py:25`，另加 256 行上限）。
一条超长条目**可以独占一批**（`if cur and ...` 的分支只拒绝「加入」，不拒绝「开新批」），所以不会因为
单条过长而报错。批内右填充、`pad_id` 位置靠 `attention_mask` 屏蔽。

**LLM**（`llm/runtime.py:65-93`）：按长度排序后，用同一套「填充 token 预算 = 批内最长 × 行数 ≤ 8000」
切批，然后**左填充**到等长，并把 `attention_mask` 传给模型。

左填充 + `attention_mask` 在这里是**逐位精确**的，不是近似。仓库没有为此写注释，但机制可以摊开：
模型内部拿到的位置 id 是 `arange(0, L-1)`（transformers 4.57.3 `models/qwen2/modeling_qwen2.py:356`），
左填充让整行的位置整体平移一个常量；RoPE 只看**相对距离**，而因果掩码按序列序构造、填充位又被
`attention_mask` 屏蔽，所以同一个真实 token 看到的注意力加权和完全不变。若某个版本改为从
`attention_mask` 推导位置 id（左填充下位置完全一致），结论只会更强，因此这个精确性与库版本无关。实测（随机初始化 2 层 Qwen2，
`transformers 4.57.3` / `torch 2.8.0`）：

| 配置 | 与不填充单条前向的最大 logit 差 |
|---|---|
| 同一行、位置整体 +36 | **0.0**（RoPE 相对性的直接验证） |
| 左填充 36 位 + `attention_mask` | 5.96e-08（浮点噪声） |
| 左填充 36 位、**漏传** `attention_mask` | 0.166（明显错误） |

这条与 RYOTIDE §7 形成有趣对照：RYOTIDE 的 torch 路径用**右填充且不加 mask**，靠「因果模型看不到后面的
padding」成立；Quyet 用**左填充且必须加 mask**，靠「相对位置不变 + padding 被屏蔽」成立。两条路都对，
但依赖的性质不同，所以一边能省掉 mask、另一边不能。

## 7. 截断：只有 state 会被截断

两条路径都遵守同一条语义（`base.py:24` 的 `truncation_warning`）：**只截 state，问题与选项永不截断**，
且 conversation（list）保**尾部**、其他 state 保**开头**。触发时响应里带 `truncated: true` + `warnings`，
`strict=True` 则改为抛错。

但两条路径的**预算机制**不同：

| | encoder | LLM |
|---|---|---|
| 位置 | `inputs.py:65` | `llm/runtime.py:43-57` |
| 方式 | 一次算术：`room = max_len - len(head) - 1`，`head` 是问题+选项块 | **迭代收缩**：先按 `cap = max_state_tokens` 试，若最长 prompt 超过 `max_prompt_tokens` 就 `cap = min(cap, n_state) - over - 16` 重试，最多 4 轮，仍超则报错 |
| 兜底 | `room < 0` 时直接报「问题与选项本身超出窗口」 | `cap < min_state_tokens` 时报「问题与选项挤掉了 state」 |
| 截断标记 | 纯 token 切片，无标记 | 重新 decode 成文本，list 加前缀 `"… "`、其他加后缀 `" …"`（`prompt.py:59`） |
| 警告里的 `kept_tokens` | 实际保留数 `kept_state_tokens` | **预算值 `cap`**，不是 decode/re-encode 后的真实数目 |

最后一行是元数据上的一个小近似：LLM 路径因为要「切成 token → 解码回文本 → 再交给 chat template 重新分词」，
往返后不一定正好是 `cap`，所以它报的是预算上限。这只影响可读性，不影响决策。

## 8. 答案成形：`answers.py`

`format_answer()`（`answers.py:9`）把概率向量翻译成 Jev 形状，三条规则：

| type | 主体 | confidence |
|---|---|---|
| `choice` | `choice` = argmax 标签 | 该标签的概率 |
| `score` | `score` = **期望等级** `Σ i·pᵢ` | **max(p)** |
| `noul` | `noul` = P(true) | `max(P, 1−P)` |

`score` 的两行要分开读：报出的 `score` 是期望值（可以是 1.3 这样的非整数），而 `confidence` 是**单个等级的
最大概率**，不是「所报等级的概率」。想拿「这个答案有多可信」时，用 `probabilities` 自己算更稳。

全部概率在输出时 `round(..., 4)`（`answers.py:6`）；需要未舍入的分布用 `base.py:45` 的 `raw_probabilities()`。
响应外层 `usage` 固定 `output_tokens: 0`——决策没有生成任何 token。

`input_tokens` 的口径值得注意：两条路径都是**逐问题累计**（encoder `runtime.py:74` 累加每个问题的序列长度，
LLM `runtime.py:108` 累加每个 prompt 长度）。同一个 state 问 3 个问题，state 会被计入 3 次——因为两条路径
都确实把 state 重复编码了 3 遍。这与 RYOTIDE §7「多问题共享一次 state 前缀，计费只算一次」是相反的取舍：
Quyet 用「整批前向 + 左填充」换吞吐，代价是 state 重复计算且如实计入。

## 9. 四条不变量

违反任意一条，输出都会「看起来像个正常分布」但其实是错的：

1. **只有选项槽位的 logit 进入 softmax。** 词表其余部分从不被读取，所以分布封闭在已知标签集上。
2. **标签必须能落到单 token（LLM）/ 单个 MASK 槽位（encoder）。** LLM 侧在加载期断言（`prompt.py:76`），
   encoder 侧由布局保证；选项数上限 10 是这条不变量的直接推论。
3. **只有 state 被截断，问题与选项永不截断。** 截断必须被上报（`truncated` + `warnings`），
   `strict=True` 时改为失败——把「静默降级」变成「显式失败」。
4. **比较概率之前先确认 temperature。** 同一个模型在不同 `temperatures` 配置下 argmax 相同、概率数值不同，
   而 T 只出现在模型自带的 `quyet_config.json` 里。跨模型比较 `probabilities` 前，先确认各自的 T。

## 10. 文件对照

| 文件 | 职责 |
|---|---|
| `src/quyet/base.py` | 共享接口：`predict` / `predict_many` / `raw_probabilities`、state 类型检查、截断警告 |
| `src/quyet/questions.py` | Jev 问题 → 校验过的 `Decision`；2..10 选项，`describe()` 的 falsy 语义 |
| `src/quyet/answers.py` | 概率 → Jev 答案形状；`score` 期望值、`noul` = P(true)、`output_tokens: 0` |
| `src/quyet/calibration.py` | 温度查找（`<type>:<桶>` 与纯题型两种键）与带防溢出的 `softmax_t` |
| `src/quyet/hub.py` | 定位模型（本地目录或 HF repo）、校验 `quyet_config.json`（`quyet_format`）、按 `kind` 分派 |
| `src/quyet/cli.py` | `quyet predict MODEL`：JSON Lines 进、JSON Lines 出，逐行捕获错误，任一行失败则退出码 1 |
| `src/quyet/encoder/inputs.py` | **训练时的布局**：序列拼接、marker/跨度、证据掩码、分桶切批 |
| `src/quyet/encoder/modeling.py` | 三种打分头与 `span_pool`；推理专用结构 |
| `src/quyet/encoder/runtime.py` | 编码器 runtime：CUDA bf16 autocast（评估时的设置）/ CPU fp32、批量前向、折回标签顺序 |
| `src/quyet/llm/prompt.py` | 两个 `prompt_version` 的 prompt、state 序列化与截断、字母单 token 检查 |
| `src/quyet/llm/runtime.py` | LLM runtime：迭代收缩的 state 预算、左填充批量、`logits_to_keep=1` 读字母 |
| `src/quyet/errors.py` | `QuestionError`：问题非法、选项过多、state 没有容身之处 |
| `examples/gem_run/` | 端到端演示（见下） |

## 11. `examples/gem_run`：把 runtime 放进真实时钟里

这个示例不改变 runtime，只是把 `predict()` 摆到一个**有截止时间**的场景里，用公开 SDK 驱动，
顺便与 Laya（另一个公开决策模型 SDK）同场对照。

- **进程隔离**（`run.py:75` `ProcessWorker`）：每个模型一个 disposable 子进程，父进程只跑游戏时钟与录制；
  启动有 900 s 上限（覆盖下载/加载/warmup），推理挂死也不会拖住父进程。
- **warmup 不计时**（`run.py:41-44`，默认 5 次）：与 RYOTIDE `server.py` 启动时打一条不计时请求同理。
- **单条推理在飞**（`engine.py:153` 的 `execution`）：截止时间不会为推理让路，超时的行直接判 0 分并保留原来的车道。
- **回答先验证再使用**（`backends.py:37` `normalize_answer`）：答案必须在合法动作集合里、概率有限且和为 1、
  标签集合完全一致，**并且 `warnings` 或 `truncated` 一律拒绝**——被截断的决策不许进入计分。
  这是 §9 第 3 条不变量在消费端的落地，作用与 RYOTIDE 用 `marker_mass` 挡住「读错位置」相同：
  让一个看似合法的错误答案无法静默通过。
- **录制可复现**（`replay.py`）：只用标准库把 raw records 与素材打包成一个自包含 HTML，浏览器只回放录制，不调用模型。

## 12. 亲手确认「读」确实发生了

```bash
pip install -e .
# 一行一个请求：{"state": ..., "questions": {...}}
echo '{"state": {"message": "Please close my card, I lost it yesterday."},
       "questions": {"intent": {"type": "choice", "instructions": "What does the customer want?",
                                "criteria": {"cancel": "close the card", "limit": "change the limit"}}}}' \
  | quyet predict chinhnc/Quyet-1.0-Small
```

看两个数字就够了：

- `usage.output_tokens` 必须是 **0**——没有任何 token 被生成；
- `answers.*.probabilities` 的所有键必须**恰好等于你给的选项**，和必须是 1。

再对照 `src/quyet/encoder/inputs.py` 顶部的 docstring 与 `src/quyet/llm/prompt.py` 的 `build_prompt()`，
就能确认服务里跑的是不是你训练时用的那条布局。

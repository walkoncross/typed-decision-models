# vLLM Semantic Router 是如何把「读一次前向」做成一套生产运行时的

> 代码库：<https://github.com/vllm-project/semantic-router>
> 对应 `src/model-runtime/README.md` 第 3 行：*`vllm-srun` serves the router's models (decision models,
> classifiers, embedders and rerankers) behind one HTTP contract. One process can serve several models.*
> 本文按 `src/model-runtime/vllm_srun/` 与 `src/semantic-router/pkg/` 的实际代码逐环节说明这句话的实现方式；
> 文中出现的文件名与行号（`systemone.py:24`、`families/decision1/vela.py:119`、
> `classifier_signal_decision_model.go:58` 等）均相对该仓库根目录。

与另外七篇最大的不同：**这不是一个「把某个 LLM 读成分类器」的小实现，而是这类读法的生产运行时。**
同一个 HTTP 契约（`/v1/decisions`）底下挂着三代模型、三套读出协议、四档数值精度的 profile，
外加一个用这套读法做路由决策的 Go 控制面。本文因此有两层：**Python 侧怎么读**（§2–§5、§10），
**Go 侧怎么用**（§1、§10.4、§15）。

先说结论式的三句话，后面逐条给代码位置：

1. 「typed」不来自分词器，也不只来自请求：它来自**模型包里固化的读法**——候选集合由请求声明并校验，
   但「在哪里读、读成什么」写死在 checkpoint 的布局里（三套，见 §3）。
2. 读完之后的 softmax / 掩码 / 重归一化**全部在 Python 运行时内完成**；Go 侧一个数都不改，
   只做阈值与一致性校验（§10.4）。
3. 这条线上**没有任何运行期标定**：读位置、prompt 布局、温度、阈值全是训练期或打包期定死的常量（§3、§10）。

## 本文的事实来源

| 类别 | 来源 | 本文中的标记 |
|---|---|---|
| 代码引用 | `src/model-runtime/vllm_srun/**`（Python 运行时）、`src/semantic-router/pkg/**`（Go 控制面）、`tools/calibration/**` | `文件:行` |
| 可实跑得到的产物 | `registry/golden_answers_*.json` 是**committed 的发布产物**：容量小、纯 JSON、可直接重算 | 标注「实测」，并注明是「在 committed 产物上重算」 |
| 上游记录的数字 | `src/model-runtime/docs/records/*.md`（parity / performance 记录）、`tools/calibration/systemone_auto/README.md` | 引原文并给出 `文件:行` |
| 未跑 | **本机没有权重、没有取模型包，本文没有跑过任何前向** | 照例不给任何前向产生的概率数值 |

本文所有「实测」都只有一类动作：**从 committed 的 golden answers 里读数字、用标准库算术重算，
再与记录值比对**（§14）。没有运行上游的任何程序，也没有下载任何模型。这一点在 §14 会把命令原样列出，
任何人都可以只用一个 Python 标准库复现。

---

## 0. 与 RYOTIDE 的分野

对照对象是 [`docs/ryotide/typed-decision-model.md`](../ryotide/typed-decision-model.md)——本文凡称「RYOTIDE §N」均指该篇的节号。

| | RYOTIDE | vLLM Semantic Router |
|---|---|---|
| 是什么 | 一个读法（脚本 + 服务） | 一个**运行时**：三代模型 × 三套读法 × 四档 profile，外加一个消费它的 Go 路由器 |
| 权重 | 公开 LLM，全程 frozen | 自家发布的模型包（`vllm-sr/Decision-1.0-*`、`Decision-2.0-*`、`Vela-2.0-*`），`src/training/` 里的训练代码**不产出这些包**（见下） |
| 「typed」怎么保证 | 标签必须落在**单 token** | **三套并列**：encoder 的 `[MASK]` 槽位、decoder 的选项段末位、Vela 2.0 的 8 个标记 token（§2、§3） |
| 读几个位置 | 1 个（末位） | Vela：每题 K 个槽位；Qwen：K 个端点 + 1 个 query；Vela 2.0：每题若干标记 token（§4） |
| 读位置怎么定 | **运行期标定**（扫 9 前缀 × 2 表面形式） | **训练期定义在 checkpoint 布局里，运行期零标定**（§3） |
| 谁的代码在跑 | RYOTIDE 自己写的前向 | **重实现**：`never imports the bundled Python`（`families/decision1/family.py:9`），数值要对到字节相同 |
| 一致性怎么保证 | 无（没有可比的参考实现） | **加载期 golden 自检**：跑不过就不 ready（§9.2） |
| 顺序偏置 | 多 order 平均 | **既不消除也不度量**（§6） |
| 单位 | 一个进程一个模型 | 一个进程可服务多个模型，且**多模型共享 GPU 时的 kernel 选择**是记录在案的坑（§9.4） |
| 置信度 | 读出来的，只有 T 能改 | **三套公式并存**，且 Go 侧**不用**模型报的 `confidence`（§10） |

一句话：**RYOTIDE 回答「怎么读」，这一路回答「怎么把一种读法做成能上线的系统」。**
两者共享的底线还是那一条——一次前向、不生成任何 token、输出封闭在已知选项集上（§11）。

**一个必须先说清的边界**：本仓 `src/training/` 里**没有** Decision 系列模型的训练代码。
它装的是 Vela 家族与各类安全/路由分类器的 HF/LoRA 训练（`src/training/README.md:11-19` 的归属表）。
Decision 包在仓里只以「推理 + 打包 + 标定评测」三种形态出现：
`families/decision1|decision2|vela2/`（运行时）、`registry/tables/*.py`（版本与摘要钉住）、
`tools/calibration/systemone_auto/`（拿 JevBench 做闭环评测）。
所以本文凡涉及「训练改了什么数值」，都只能从**发布包里的标定文件**反推，不能从训练代码读——这与
Jeff / OpenJev(HF) 那两篇的处境相反，那两篇能直接读训练侧。

---

## 1. 端到端数据流

一次带决策信号的请求，从 Go 控制面到 Python 运行时再回到决策引擎：

```
  Go 请求 ──► pkg/classification
              evaluateDecisionModelSignals()            classifier_signal_decision_model.go:36
              ├─ 按 deployment 分组，state := wholeText   :58
              ├─ Fan(...) 多 deployment 并行              :62
              └─ evaluateDecisionDeployment()
                   ├─ 每题 DecisionQuestion(rule)          :81-86
                   ├─ question.Truncate = true             :86
                   └─ callCtx = decisionCallContext(rules) :87   ← 取该 deployment 各 rule 的**最短**超时
                              │
                              ▼  HTTP（客户端是 openapi.yaml 的生成产物）
   ┌──────────────── pkg/modelservice ────────────────┐
   │ encodeDecisionRequest()      client.go:200        │
   │   api.DecisionRequest{Model, State, Questions,    │
   │                       Options{DeadlineMs,         │
   │                               MaxTokens}}         │
   │ labelCriteria.MarshalJSON()  client.go:224-253    │ ← 标签**按声明顺序**序列化
   │ CreateDecisionsWithResponse  surfaces.go:217      │
   └───────────────────────┬──────────────────────────┘
                           ▼  POST /v1/decisions（别名 /v1/systemone）
   ┌──────────────── src/model-runtime/vllm_srun ───────────────────┐
   │ api/app.py:197-201   两条路由指向同一个 surface_route("decisions")│
   │ runtime.call → prepare(): 校验 → plan_surface()  runtime.py:906 │
   │   DecisionModel.plan_surface  plugins/decisions.py:236           │
   │     ├─ refuse_unanswerable：全题无效 → 400                      │
   │     └─ family.plan(state, questions)                            │
   │          ├─ systemone.read_question()   ← 题型与字段的封闭校验   │
   │          ├─ render(row, state, tokens)  ← 三套布局之一（§3）     │
   │          └─ RequestPlan{items, errors, input_tokens}             │
   │ scheduler → profile → family.run(items) → 每题的候选概率列表     │
   │   ├─ Vela：  softmax on device，**不**在 host 重归一化（§5.2）   │
   │   └─ Qwen：  softmax(logits/T) on device → **host 重归一化**      │
   │ finish_surface() → 每题一个 typed answer    decisions.py:281     │
   │ runtime.finish(): usage{input_tokens, output_tokens: 0}  :961-963│
   └───────────────────────┬─────────────────────────────────────────┘
                           ▼
   { model, answers{id → {type, choice|noul|score, probabilities,
                          confidence, legend?}}, usage, meta? }
                           │
                           ▼
   ┌────────── Go 侧解码与消费 ──────────┐
   │ decodeAnswer()      client.go:370    │
   │ checkAnswers()      answer_check.go:17│ ← 不符的整条改判 invalid_model_output
   │ applyDecisionAnswer()  classifier_signal_decision_model.go:167
   │   Choice → 每选项一条信号 + 选中项一条  :187-197
   │   Noul   → P(true)                     :171-176
   │   Score  → expected level              :177-182
   └──────────────────┬───────────────────┘
                      ▼
   decision.SignalMatches → pkg/decision/engine.go:424 → 选型 / 路由
```

两条值得单独记住的性质：

- **`state` 就是请求原文**（`classifier_signal_decision_model.go:58` 的 `stateOf[deployment] = wholeText`），
  不是被 Go 侧加工过的摘要。决策模型看到的是 agent harness 发来的那一整段文本。
- **失败是 fail-open 的**：`evaluateDecisionModelSignals` 的 docstring 明写 *A failed or late call
  leaves its signals unknown; it never fails the request*（`:33-35`），错误被写进
  `results.SignalErrors["decision:"+name]`（`:109`、`:118`）。这与 Python 侧几乎处处置错的风格正好相反（§9）。

---

## 2. 「typed」从哪来：请求声明的 `criteria`，加包里固化的读法

RYOTIDE 的类型化是**借来的**：它要求标签能落到分词器的单个 token 上、落不到就退化成字母标记。
这里没有这条约束——被评分的从来不是标签文本，而是**为它准备的一个位置**。

### 2.1 标签集来自请求，并在两个地方被封闭

`systemone.read_question()`（`systemone.py:143-193`）是题型与字段的单一入口：

```python
QUESTION_TYPES = ("choice", "noul", "score")            # systemone.py:24
MIN_OPTIONS, MAX_OPTIONS = 2, 255                       # systemone.py:22-23
MIN_LEVELS, MAX_LEVELS = 2, 10                          # systemone.py:25
FIELDS = {
    "choice": frozenset({"type", "instructions", "criteria", "choices"}),
    "noul":   frozenset({"type", "instructions", "criteria", "choices"}),
    "score":  frozenset({"type", "instructions", "criteria", "levels"}),
}                                                       # systemone.py:27-31
```

`unknown = set(question) - FIELDS[kind] - extra_fields - {"require_full_input"}`（`systemone.py:159`）——
**多一个字段就报错**，不是忽略。标签集就是 `criteria` 的键（Choice / Noul）或 `levels` 的列表（Score），
上限 255 与 RYOTIDE 的 260、Open-Jev 的 255 同一量级，但**来源不同**：这里不需要标签落到词表上，
255 是接口约定的上限而不是分词器的上限。

第二处封闭在答案侧：`families/decision1/answers.py:49-57` 要求
`len(values) == len(keys)`、每个值有限且 ∈[0,1]、且 `fsum(values) ≈ 1.0`（`abs_tol=2e-5`，
`answers.py:19`），否则整条答案变成 `invalid_model_output`。**这条校验与 RYOTIDE 的 `marker_mass` 是两种东西**：
前者验的是「分布自洽」，后者验的是「质量落在标记上」。

Vela 2.0 把题型扩到 5 种：`QUESTION_TYPES = (*SYSTEM_ONE_TYPES, *LABELLED_TYPES)`（`families/vela2/request.py:53`），
多出 `set` 与 `span`——它们的标签是 `labels` 字段，同样按声明顺序成为槽位（§3.3）。

### 2.2 三套读法，各自的「选项」是什么

| runtime | 家族 / 模型 | 一个选项在模型输入里是什么 | 被评分的量 |
|---|---|---|---|
| `vela-encoder` | Decision 1.0 的 Kai / Lex / Route（`families/decision1/vela.py`） | 一个 `[marker]`（= 分词器的 MASK token）+ 该选项的文本 | 该 `[marker]` 位置过题型头后的**一个标量** |
| `qwen3.5-decision` | Decision 1.0 的 Eos / Sol / Nox / Lux（`families/decision1/qwen.py`） | 一个 `<option>…</option>` 段 | 该段**末位**的 hidden state 对 query 行的打分 |
| `vela2-unified` / `vela2-decoder` | Vela 2.0 的 0.3B / 0.8B / 4B / 9B（`families/vela2/`） | 一个 `[O]` 标记（Choice / Score）或一个 `[E]` 标签标记 | 标记位置经 `MarkerHead` 的标量 |

三套读法的共同点是：**模型永远不会「写出」标签**。`output_tokens` 在响应组装处硬编码为 0
（`runtime.py:961-963` 的 `{"input_tokens": plan.input_tokens, "output_tokens": 0}`），
契约里也把它列为 required（`api/openapi.yaml:605-611`）。

### 2.3 一个反例：Decision 2.0 的 Kai 不是 encoder 而是 decoder

Decision 1.0 与 Decision 2.0 的 Kai 同名不同族：1.0 的 Kai 走 `vela-encoder`
（`registry/tables/decision1.py:17-20`，571,909,635 参数），2.0 的 Kai 走 `decision2` 家族
（`registry/tables/decision2.py:9-14`，597,103,104 参数）。**同一个名字，两套读法。**
跨代引用数字之前必须先确认是哪一代——这是本仓最容易踩的口径坑。

---

## 3. 读位置：三套布局，全部写死在包里，运行期零标定

RYOTIDE 花了一整节在运行期扫 `SUFFIXES × PATTERNS`、用 marker mass 挑读位置。
**这一路完全没有这一步**：位置是包的属性，运行期只负责按包的声明重建它。

### 3.1 Vela encoder：每个候选前面放一个 `[marker]`

`families/decision1/vela.py:100-139` 的 `render()` 是全部：

```python
ids = [special["bos"], *tokens(f"{row.kind} question: {instructions}"), special["sep"]]
markers = []
for index, candidate in enumerate(row.candidates):
    markers.append(len(ids))                     # ← 记住 [marker] 的绝对位置
    text = candidate_text(row.kind, index, candidate.key, candidate.description)
    ids += [special["marker"], *tokens(text), special["sep"]]
state_ids = tokens(state)
room = max_length - len(ids) - 1
if room < 1 or len(state_ids) > room:
    raise QuestionError(MAX_LENGTH_EXCEEDED, ...)  # ← 不截断
ids += [*state_ids, special["sep"]]
```

`special` 来自 `special_ids()`（`vela.py:66-86`），它按 `SPECIAL_TOKENS`
（`vela.py:45-46`：`bos→cls_token|bos_token`、`sep→sep_token|eos_token`、`pad→pad_token`、`marker→mask_token`）
从 tokenizer 的 config / special_tokens_map 里查 id，查不到就 `PackageError`。

这是与 Laya 最像、与 RYOTIDE 最不像的一点：**被评分的不是候选文本本身，而是它前面那个 `[MASK]`。**
选项文本多长都无所谓（Laya §2 的同一结论），而读位置在 `render` 里被显式记录，不靠探测。

### 3.2 Qwen decoder：每个选项段的末位，加一个全局 query 行

`text/segments.py` 是三套里唯一被两个家族共用的（Decision 2.0 也用同一套布局）：

```python
SUFFIX = "\n\nSelect the single option best supported by the context and instructions.\nDecision:"   # :21
prefix = f"Context:\n{payload(state)}\n\nTask type: {kind}\nQuestion:\n{payload(instructions)}\nOptions:"  # :30
rendered = ["\n<option>\n" + canonical({"key":…, "description":…}) + "\n</option>" for option in options]  # :31-36
```

`encode()`（`:40-69`）**逐段独立分词**（不带 special token），于是：

- `endpoints.append(len(ids) - 1)`（`:59`）——每个选项段自己的**最后一个 token**；
- `query = len(ids) - 1`（`:69`）——整条序列的最后一个 token，落在 `Decision:` 后面。

`collate()` 再把这些位置收成两个张量（`:96-104`）：
`candidate_positions`（每行宽 = 该批最大选项数，用 0 补齐）与 `query_positions`。
`collate` 里还有一道便宜的守卫：`len(set(item.gather)) != count` 或端点越界就 `ValueError`（`:88-91`）——
**端点必须互不相同且都落在 query 之前**。

### 3.3 Vela 2.0：8 个标记 token，`[ABS]` 是弃权位

`families/vela2/encoder_layout.py:38-47` 的 `MARKERS` 是八元组：

```
("[Q]", "[O]", "[ABS]", "[SEP_SCHEMA]", "[SEG_user]", "[SEG_answer]", "[E]", "[SEG_context]")
```

布局全貌写在模块 docstring（`:1-11`）：

```
<bos> ([Q] question ([O] option)* [ABS]?)* ([E] label)* [SEP_SCHEMA] ([SEG_role] part)* <eos>
```

`config.json` 里的 `marker_ids` 必须**恰好**是这八个名字，否则 `PackageError`
（`encoder_layout.py:105-108`）——标记是 token **id**，用户文本永远造不出一个。
`[ABS]` 之后软化成 `abstain_probability` 报出（§10.2），不参与 argmax。

### 3.4 代价：位置不可换，换布局等于换权重

三套布局都由包的字节决定：Vela 的 `markers` 位置、Qwen 的 `<option>` 段落结构、
Vela 2.0 的 `marker_ids`，全都在包里。family 层还各有一道「只支持这一种布局」的硬拒绝：

| runtime | 断言 | 位置 |
|---|---|---|
| vela-encoder | `ARM = {"arm": "all22", "training_arm": "S22", "type_order": ["choice","noul","score"]}`，不符即 `PackageError` | `vela.py:42`、`vela.py:49-63` |
| qwen3.5-decision | `prompt_version` 必须是 `structured-segmented-candidate-endpoints-global-query-v2` | `qwen.py:29`、`qwen.py:40-47` |
| decision2 | `head_variant` 必须是 `shared`，且 `readout == "label_token"` 直接拒绝 | `families/decision2/package.py:251-254` |

最后一条尤其说明问题：**这个运行时明确不支持「按标签 token 读」那种（更接近 RYOTIDE 的）读法**——
它被写在拒绝清单里，不是没实现，是选定了不做。

---

## 4. 一次前向：三种读法，三次不同的 gather

### 4.1 Vela：一次 encode，三套层堆，读 K 个槽位

`VelaDecisionModel.run`（`families/decision1/model.py:162-177`）很短：

```python
with torch.inference_mode():
    scores = vela.marker_logits(self.engine_model.encode, self.readout, items,
                                self.special["pad"], self.engine_model.device, self.exit_layer)
    if not torch.isfinite(scores).all():
        return [None] * len(items)                     # ← 非有限值整批判无效
    return [scores[slot, : len(item.keys)].softmax(-1).cpu().tolist()
            for slot, item in enumerate(items)]
```

`marker_logits`（`vela.py:296-332`）的关键在「**每种题型各跑一遍整批**」：

```python
kinds = [item.task_type for item in items]
for kind in KINDS:
    rows = [slot for slot, item_kind in enumerate(kinds) if item_kind == kind]
    if not rows:
        continue
    hidden = encode(EncoderBatch(input_ids=..., attention_mask=..., branch=BRANCH[kind])).hidden[exit_layer]
    scores = readout(kind, hidden.index_select(0, index), padding.index_select(0, index),
                     batch["markers"].index_select(0, index))
    output = output.index_copy(0, index, scores)
return output.masked_fill(~batch["valid"], torch.finfo(torch.float32).min)
```

`BRANCH = {"choice": "choice", "noul": None, "score": "score"}`（`vela.py:44`）——
**Noul 读共享编码器，Choice 与 Score 各读自己的分支**。三个分支装在同一个 backbone 里
（`families/decision1/family.py:157-166` 的 `BackboneSpec.branches`），
共享 embedding 与 rotary 表，所以「一次前向」在这里的准确含义是
**一次 tokenize + 一次 embedding + 三套各 22 层的堆**。

题型头本身在 `heads/typed.py`：`KINDS = ("choice", "noul", "score")`（`:20`），
每种题型一套 `TypeHeadLayer`（pre-norm `TransformerEncoderLayer`，`:23-53`）+ 一个
`LayerNorm → Linear(d,d) → GELU → Linear(d,1)` 的 scorer（`:70-80`）。
forward 里 `hidden = hidden.float() + offset`（`:93-94`）——**加到每个 token 上的是题型嵌入，
不是 prompt 里的一句话**；这是 Laya `type_emb` 的同一手法。

`readout()` 最后用 `torch.gather` 把 K 个槽位的 hidden 取出来（`:100-103`），
所以「读几个位置」在这里是显式的 K。

### 4.2 Qwen / Decision 2.0：backbone + 一个候选头

`heads/candidate.py` 是 Decision 1.0 的 decoder 路径与**整个 Decision 2.0** 共用的读出：

```python
logit = (key(c) · query(q)) / sqrt(head_dim) + scalar(gelu(candidate_mlp(c) + query_mlp(q)))
```

（`candidate.py:45-59`，`c` = 选项端点行的 LayerNorm 后表示，`q` = query 行的）。
头部参数名与包里的完全一致，且 `load_head` 用 `set(tensors) != HEAD_TENSORS` 精确比对
（`candidate.py:19-30, 62-74`）——**多一个少一个张量都拒载**。

`forward_logits`（`candidate.py:85-105`）把 backbone 前向与打分接起来：

```python
output = engine_model.forward(ForwardBatch(input_ids=..., attention_mask=...,
                                           gather=batch["candidate_positions"],
                                           query=batch["query_positions"],
                                           lengths=lengths, shared_prefix=shared_prefix))
mask = batch["candidate_mask"].to(output.gathered.device)
with torch.inference_mode():
    return logits(head, output.gathered, output.query, mask)
```

`ForwardBatch` 的 docstring（`plugins/base.py:336-353`）说明 `gather` 是「每行的选项端点、
用 0 补齐到最宽行」，`query` 是「每行一个 query 位置」，`shared_prefix > 0` 才让引擎走共享上下文
（§7）。**解码器路径不物化词表维度**——与 Jeff、Laya、SemIf-OpenJev 同侧，与 RYOTIDE 不同。

### 4.3 分块与批：`PHYSICAL_BATCH = 8` 与 32 的填充倍数

两条 runtime 都把请求切成 8 题一组的**物理批**，但切法不同：

| | 切法 | 位置 |
|---|---|---|
| vela-encoder | 按题型**稳定排序**后每 8 题一组；批内填充到最长行（不做倍数对齐） | `vela.py:30`、`vela.py:160-168` |
| qwen3.5-decision | 保持请求顺序每 8 题一组；填充到 **32 的倍数** | `qwen.py:26-27`、`qwen.py:91-107` |

qwen 的 `_within()`（`qwen.py:110-126`）还会在超过前向 token 预算时把一批再切小，
而 docstring 明说发布的包**从不会触发它**（八行 × 输入上限仍在预算内）。

---

## 5. 掩码 + softmax：两个 runtime 在这里分道

### 5.1 掩码写在哪一层，两边不一样

| runtime | 掩码位置 | 掩码值 | 代码 |
|---|---|---|---|
| vela-encoder | 读出后的**候选槽位** | `torch.finfo(torch.float32).min` | `vela.py:237`、`vela.py:332` |
| qwen3.5-decision | 候选头的**补齐槽位** | `-inf` | `candidate.py:77-82` |

**两边都用不到「词表掩码」**：候选集合是通过 gather 出 K 个位置得到的，而不是从词表里挑标签 id。
RYOTIDE §5 那种「把词表里非标记的位置打成 `-inf`」在这里没有对应物——
`masked_fill` 打的永远是**补齐出来的假选项**，不是真选项。

两个掩码值不同（`finfo.min` vs `-inf`）**不是风格差异，是被各自那条路径的有限性检查逼出来的**：

- Vela 路径在**切片之前**就对整张分数表做 `torch.isfinite(scores).all()`
  （`model.py:172-173`、近似档的 `model.py:205`），一旦非有限就**整批**返回 `None`，
  于是该批每一题都变成 `invalid_model_output`。若把补齐槽位打成 `-inf`，
  任何「批内选项数不齐」的正常请求都会触发这条分支——所以这里**必须**用一个有限值（`finfo.min`）。
- Qwen / Decision 2.0 路径没有这个检查：`-inf` 只写在候选头上（`candidate.py:82`），
  而结果一取出来就按 `[:len(item.keys)]` 切片（`model.py:175`、`decision2/family.py:344`），
  非有限检查是在**切片之后**的概率上做的（`qwen.py:145`、`decision2/answers.py:35-39`）。
  补齐槽位在进 softmax 之前就没了，`-inf` 不会变成 NaN。

顺带解决另一个疑问：既然补齐槽位总在 softmax 之前被切掉，**掩码值本身对真实选项没有任何影响**。
`marker_logits` 里 `output = torch.empty(...)` 先建空张量、只在「该题型出现过」的行上 `index_copy`
（`vela.py:311-331`），所以掩码真正兜住的是**从未被写过的格子**——未初始化内存，
而不是「被屏蔽的候选」。

### 5.2 重归一化：一个 runtime 做，另一个不做（**可实测**）

这是本文最具体的一处发现，而且可以在 committed 产物上直接验证：

```python
# Vela：设备侧 softmax，之后直接 tolist，host 上不碰
scores[slot, : len(item.keys)].softmax(-1).cpu().tolist()          # model.py:175

# Qwen：设备侧 softmax(logits/T)，再在 **host 上重归一化一遍**
staged = [(scores[slot, : len(item.keys)].float() / temperatures[item.task_type]).softmax(-1) …]
host = torch.cat(staged).tolist()
…
total = sum(values)
out.append([value / total for value in values])                     # qwen.py:139,148-149
```

后果可以在 `registry/golden_answers_decision1.json` 上量出来（**实测，§13.1 给出完整命令**）：

```
encoder（Kai / Lex / Route，14 条分布）：max |Σp − 1| = 1.016e-07
                                        64/64 个记录值都**精确等于**它的 float32 舍入
decoder（Eos / Sol / Nox / Lux，16 条分布）：max |Σp − 1| = 0.0（16 条全部精确为 1）
                                        48/48 个记录值都**不是** float32（都是 float64 运算的产物）
```

也就是说：**同一个 `probabilities` 字段，在 encoder 路径上是「float32 softmax 的原始输出」，
在 decoder 路径上是「host 上 float64 重归一化后的结果」。** 这条判据同时被两个独立的量证实——
和的偏差，以及数值本身能不能被 float32 精确表示。两者都在契约允许的 ±2e-5 内
（`families/decision1/answers.py:19`），所以谁也不会报错；但拿这批概率做逐位比对时，
encoder 一栏永远差一点点，decoder 一栏则严丝合缝。RYOTIDE §5 留下的「掩码 softmax 之后还要不要
再归一化」在这里有了一条**分叉的答案**：同一个运行时里，一条路径做、另一条不做。

### 5.3 温度在 softmax 之前，且只有 decoder 有

- Vela 路径**没有温度**：`vela.py` 里没有任何 temperature 符号，包也不许带标定——
  `file_map()` 明写 `if runtime == VELA and calibration: raise PackageError("vela-encoder packages are uncalibrated")`
  （`families/decision1/package.py:139-140`）。
- Qwen 路径的 softmax 分母就是温度：`softmax(logits / temperatures[task_type])`（`qwen.py:134`）。
  温度来自包：要么一个常数，要么一个 per-type 文件（`package.py:178-191`）。

所以「温度改不改数值」这件事在这个仓里**逐 runtime 不同**，不逐模型不同。见 §10.3。

---

## 6. 顺序偏置：既不消除，也不度量，只有一个 tie-break 是顺序相关的

JevBench 记过 open-alternative-jev 在 `A. yes, B. no` 上 72%、顺序反过来 21%。
另外七篇给的答案各不相同（RYOTIDE 多 order 平均、Laya 训练期消解、OpenJev(HF) 训练期压低、
SemIf-OpenJev 只度量不消除）。**这一路给出的答案是：不处理。**

把话说准一点，分三层：

**(1) 运行期没有任何顺序处理。** 全仓对 `order_bias` / `option order` / `position bias` 的命中只有三处，
没有一处在这条读法上：

| 命中 | 是什么 | 与本文的关系 |
|---|---|---|
| `pkg/promptcompression/compressor.go:69` | prompt 里**句子**的中间位置惩罚（lost-in-the-middle） | 压的是上下文，不是选项 |
| `pkg/shadowdataset/judgments.go:157` | 让每对候选**两种顺序各出现一次**，统计 `SlotFollowingPairs` | 属于**裁判（LLM judge）路径**，有 `output_tokens` 与 `cost_usd`，不是类型化读法 |
| `pkg/modelservice/modelservice.go:30` | 注释：`Choices keep Choice / Noul option order` | 说明顺序**是语义**，不是缺陷 |

**(2) 顺序确实是变量，而且是**被刻意保留**的变量。** Go 侧把 Set / Span 的标签**按声明顺序**序列化，
注释写得毫不含糊（`modelservice/client.go:224-225`）：

> `labelCriteria encodes Set and Span labels as the criteria object in label order: a label's position is part of the question the model reads.`

Python 侧同样按请求顺序铺槽位（`families/decision1/questions.py:99` 的 `Candidate(*option)`
直接沿用 `criteria.items()` 的顺序；`vela.py:119` 的 `for index, candidate in enumerate(...)`）。
**顺序一路传到底，没有任何一处打乱或平均。**

**(3) 唯一的顺序依赖是一条写明的 tie-break。** Decision 2.0 在概率精确并列时
（`abs(p − max) <= TIE_TOLERANCE`，`TIE_TOLERANCE = 1e-8`）先判 `choice = None`，
再由 `product_answer` 用「请求顺序里的第一个」补上，注释同样明写
（`families/decision2/answers.py:91-94`）：

```python
elif answer["choice"] is None:
    # Exact ties resolve to the first option in request order; the distribution is unchanged.
    answer["choice"] = keys[probabilities.index(maximum)]
```

Decision 1.0 的 `answers.py:62` 则直接 `max(range(len(keys)), key=values.__getitem__)`——
Python 的 `max` 本来就在并列时取第一个，行为一致但没写注释。

**因此对照总表里这一格只能填「不消除也不度量」。** 这一路把顺序偏置当成**训练期的责任**：
模型包是别人训好发出来的，运行时只保证「按请求给的顺序原样读」。
与 Laya 的区别是根本的：Laya 把顺序在**结构上**变得不可见（`parallel` 布局）；
这里连「顺序是变量」这件事都不在运行期的视野里。

---

## 7. 多问题共享 state

一个请求带多道题（上限 `MAX_QUESTIONS = 1024`，`families/decision1/questions.py:17`），
state 是同一段文本。这一路的做法与前三篇都不同：**它把「要不要共享」做成了一个可选的 profile。**

### 7.1 `states`：一个请求里的多段 state

除了多题共享一段 state，契约还允许**一个请求带多段具名 state**
（`plugins/decisions.py:83-109` 的 `split_states` / `:112-143` 的 `join_states`）：
`{"state": …, "states": {"b": {...}}}` 会被拆成「每个 state 一个请求」，请求自己的排在最前，
首个失败即整体失败，最后把各段的结果合回一个响应体的 `states` 字段。
`plan_surface` 只在**非 part** 的顶层调用 `refuse_unanswerable`（`decisions.py:276-277`）——
分片的拒绝留给 `join_states` 统一做。

### 7.2 `shared_context` profile：树模式

`profiles/shared_context.py` 是这一路独有的东西。它的做法不是「缓存前缀的 KV」，
而是**把公共前缀与各题的续写拼成一条行**（`TreeBatch`，`plugins/base.py:406-422`）：

```python
def shared_prefix(items, align=1) -> int:
    """Tokens every item starts with, before any option endpoint, rounded down to align."""
    ids = [item.ids for item in items]
    low, high = min(ids), max(ids)
    common = next((i for i, (a, b) in enumerate(zip(low, high, strict=False)) if a != b),
                  min(len(low), len(high)))
    limit = min(min(item.gather) for item in items)      # ← 不许切进任何一个选项段
    return min(common, limit) // align * align
```

共享前缀是「所有题目的最长公共 token 前缀，**切在第一个选项端点之前**」（`:44-54`）。
`share()`（`:98-113`）再叠上一组闸门：至少 `min_questions = 2` 题、
省下的前缀 token `(questions - 1) * prefix` 必须超过**实测的盈亏平衡点**、
且拼起来的行要装得进前向 token 预算。

盈亏平衡点是硬编码的一张表（`:28-31`）：

```python
DENSE_BREAK_EVEN = (3072, 1024)                                     # (带图, eager)
HYBRID_BREAK_EVEN = ((1024, 6144, 2048), (2048, 3072, 1536), (3072, 1536, 1024))
WIDE_HYBRID_BREAK_EVEN = 512
```

`auto_shared_tokens()`（`:56-72`）按 backbone 的 `hidden_size` 与有没有 `linear_attention` 层查表，
注释写明这是 *the break-evens measured on one MI325X for the released runtime's switch*——
**上游记录的实测值，不是本文跑出来的**。它是「共享是有代价的」这句话的少见例子：
共享要多发 launches（一条打包行、每层两个 attention 或 gated-delta kernel），
小 backbone 要够大的请求才回本。

### 7.3 共享是**近似**的

`SharedContextProfile.numerics = "approximate"`（`:76`），docstring 也直说
*Answers can differ from the exact path by rounding*（`:12-13`）。
记录在案的数字（`docs/records/decision1-parity.md`，**上游记录**）：

> `shared_context`, decoders, ROCm: Eos max |Δp| 0.046 / 决策变化 0/0；Sol 0.031 / 10 of 320、15 of 640；
> Nox 0.025 / 0/0；Lux 0.032 / 5 of 320、0。每个变化的决定都是 bundled runtime 自己答案里的 near-tie
> （top-two margin ≤ 0.0032（Sol）、0.034（Lux））。

与 SemIf-OpenJev §7 的「8.6× 加速换来 5–6/777 的 argmax 漂移」是同一类交易，
但这里把「要不要做这笔交易」变成了**启动参数**而不是默认行为（§8）。

### 7.4 计费口径

`RequestPlan.input_tokens = sum(len(item.ids) for item in items)`（`families/decision1/model.py:116`）——
**逐题累计**，即使 `shared_context` 只算了一次前缀。这与 RYOTIDE 的 `run_many()` 相反、
与 Jeff / Quyet / Laya 相同。共享省的是**算力**，不是账面上的 `input_tokens`。

---

## 8. 四档 profile：exact 是默认，三档近似是选项

这是本仓相对另外七篇**新增的一个维度**：同一个模型，四档数值语义。

| profile | 谁用 | 语义 | 代码 |
|---|---|---|---|
| `exact` | 所有模型（**默认，golden 自检也跑在它上面**） | 复现发布包的批形状与数值，目标是**逐字节相同** | `profiles/exact.py`、`runtime.py:500-508` |
| `batching` | encoders | 每种题型只跑自己的行、打包不填充（`run_approximate`） | `families/decision1/model.py:179-212` |
| `max_speed` | encoders | `batching` + 预打包的 `float32-packed` 权重副本 | `profiles/max_speed.py`、`accel/*` |
| `shared_context` | decoders | 树模式共享前缀（§7.2） | `profiles/shared_context.py` |

`exact` 不是「随便跑一遍」：`runtime.py:500-508` 的 `_profiles()` 说明它**总是**被装进来，
注释是 *``exact`` (golden checks run on it) and the model's own profile*。
`vela.py:180-212` 的 `run_approximate` docstring 则把代价写在明处：

> The released runtime runs every type's stack over the whole padded batch; rows of other types and
> padding are wasted work, but dropping them changes GEMM and attention shapes, so this is for
> approximate profiles…

**为什么非拆不可**：记录里量到了批不变性并不成立（`docs/records/decision1-parity.md`，上游记录）：

> ROCm：640 行里 0 行相同，max |Δp| 0.08；CPU (MKL)：320 行里 244 行相同，max |Δp| 1.4e-3。

也就是说「把每题的题型单独跑」不是优化，是**换了一套数值**——在 ROCm 上 640 行没有一行逐位相等。
所以 `exact` 必须原样保留发布时的混合批形状。这条与 Laya §7 的「双向编码器无法复用前缀」
是同一枚硬币的两面：**批与布局是数值的一部分，不是调度的自由。**

近似档的记录同样是上游记录（`docs/records/decision1-parity.md`）：`batching` 在 ROCm 四个面板上
Kai 10,253/10,653 逐字节相同、决策变化 0、max |Δp| 9.6e-6；`max_speed` 在 CPU 上
Kai 70/1,431、决策变化 0、max |Δp| 1.1e-5。

---

## 9. 服务化：fail-loud 到了「不通过自检就不 ready」的程度

### 9.1 契约与错误码

`api/openapi.yaml` 是唯一的契约（`API_VERSION = "2.3.0"`，`api/app.py:27`），
Go 侧的客户端是它的生成产物（`pkg/modelservice/api/openapi.gen.go:3` 的 `DO NOT EDIT`）。
错误分两层（`errors.py:1-6`）：

- **请求级** → HTTP 状态 + `{"error": {"code", "message"}}`：
  `invalid_request 400` / `model_not_found 404` / `request_too_large 413` /
  `unsupported_surface 422` / `overloaded 429` / `internal_error 500` / `not_ready 503`（`errors.py:33-41`）。
- **条目级** → 200 响应里就地标注，不牵连同请求的其他题：
  `invalid_question` / `invalid_input` / `max_length_exceeded` / `scan_budget_exceeded` /
  `invalid_model_output` / `deadline_exceeded` / `unavailable`（`errors.py:13-31`）。

`RuntimeServiceError.__init__` 会拒绝任何不在表里的 code（`errors.py:47-48`）——
**新增一个错误码必须先选边。**

### 9.2 加载期 golden 自检：跑不过就不 ready

这是本仓最接近「fail-loud 到骨子里」的地方。`ServedModel._start()`（`runtime.py:402-452`）
在模型装好之后、置 ready 之前，**真的跑一遍 golden 请求**：

```python
self.health.set("warming", "running the golden check")
self.health.golden = golden_check(self.golden_surface, model.golden_compare,
                                  self.family.golden(self.package),
                                  placement.device.accelerator)
if self.health.golden.status == "failed":
    raise VerificationError(self.health.golden.detail or "golden check failed")
```

`golden_check` 的容差按设备类别给（`supervision/readiness.py:12-14, 115`）：

```python
GPU_TOLERANCE = 0.02
CPU_TOLERANCE = 1e-3
tolerance = CPU_TOLERANCE if device_class == "cpu" else GPU_TOLERANCE
```

golden 请求本身是固定的（`systemone.py:36-58`）——一段固定的 state 加三种题型各一题：

```python
GOLDEN_STATE = "Write a Python function that merges two sorted lists and explain its running time."
def golden_questions(other):   # domain(choice: code/math/other)、reasoning(noul)、difficulty(score: 3 levels)
```

参考值就是 §14 要重算的 `registry/golden_answers_*.json`。**三态而不是两态**：

| 状态 | 含义 | 位置 |
|---|---|---|
| `matched` | 与记录值在容差内一致 | `readiness.py:97-144` |
| `failed` | 超容差 → **`VerificationError`，模型不 ready** | `runtime.py:447-448` |
| `unverified` | golden 过了，但 kernel choices 没能装上（`unpinned`），所以「无法为发布答案背书」 | `runtime.py:449-452`、`runtime.py:458-475` |

`unverified` 这个状态是本仓独有的设计：它承认「我对得上记录，但我不是在发布时那套 kernel 上算的」。
`runtime.py:461-463` 的注释是原话：*A model whose choices can't be applied still loads,
but its golden result says ``unverified``: it can't vouch for the released answers.*
这与 Laya §9 的 `unevaluated`、SemIf-OpenJev 的 `allowed_token_mass`「只上报不参与」是同一类诚实的第三态。

### 9.3 「绝不截断」与显式的完整性证明

- **state 超长不截断，直接判错**。Vela 路径 `render()` 里 `room < 1 or len(state_ids) > room`
  就 `MAX_LENGTH_EXCEEDED`（`vela.py:124-129`），docstring 写着 *nothing is truncated*（`:107`）；
  Qwen 路径同理（`segments.py:64-68`，`no truncation`）。
  包侧还有一条硬约束：`packing.state_truncation` 必须是 `"error"`（`vela.py:55-56`）。
- **一条超长题拖垮整个请求**：`Decision1Model.plan()` 捕到 `MAX_LENGTH_EXCEEDED` 后
  **给每一道题都写同一条错误、并清空 items**（`model.py:94-104`），
  而 request 级的 `ValueError` 会变成 400。docstring 把这条列为「bundled runtime 的准入规则」：
  *one over-long question fails every question of the request, nothing is truncated*（`model.py:3-8`）。
- **`require_full_input`**：opt-in 的完整性契约，成功时在答案上盖 `input_coverage: "complete"`
  （`plugins/decisions.py:298-303`）。它与 `overflow: truncate` 互斥（`systemone.py:138-139`），
  且**只有这个证明能让调用方知道「每个 part 都被完整读过」**——
  上游 README 第 49-53 行专门警告 *An accepted request or its token usage does not prove that
  every supplied part was read*。

### 9.4 一个只有机械检查抓得到的坑：多模型共享 GPU 时的 kernel 选择

`docs/records/decision1-parity.md` 记录了一次**静默的数值污染**（上游记录）：

> 默认布局把一张 GPU 上的每个模型服务在同一个 runtime 进程里，而 FLA 的 autotuner 是**进程级**的。
> 在按模型钉住 kernel choices（`accel/autotune.KernelChoices`）之前，后加载的解码器会跑**先加载模型**的
> 选择。47/55 对内置 GPU 模型的 choices 在共享调优键上不一致，**选错就会改答案**。
> 修正前「Sol 跑在 Eos 之后」：10,653 条里只有 75 条逐字节相同、79 条决策变化、max |Δp| 0.193；
> 修正后 **10,653/10,653 逐字节相同**。而**这些跑全都通过了 golden 自检**（3/3 在 GPU 容差 0.02 内）。

这是「0.193 的概率位移照样过 golden 检查」的一个实例——**容差是给设备差异留的，
不是给「跑错了 kernel」留的**。它与 Open-Jev §8 的 `device_map="auto"`（服务照样回 ready，
并从没加载的模型返回自信的数字）是同一种失效，但这里的检测手段是记录里的 A/B 而非运行期断言。

---

## 10. 置信度从哪来，又怎么被改写

### 10.1 三套公式，同一个字段名

`confidence` 在三代里是三个不同的函数——**这是本仓内部的口径坑**，跨代比较概率之前必须先对齐：

| 家族 | choice 的 `confidence` | score 的 `confidence` | noul | 位置 |
|---|---|---|---|---|
| decision1（Vela 与 Qwen 共用） | **top-two margin**：`first - second`（截到 [0,1]） | `1 − var/uniform`，uniform = `(k²−1)/12` | 直接报 `P(true)`，无 confidence 字段 | `families/decision1/answers.py:22-39` |
| decision2 | **`1 − H/log k`**（归一化熵） | **同一条归一化熵** | 同上 | `families/decision2/answers.py:81-83` |
| vela2 | top-two margin | `1 − var/uniform` | `P(yes)`，可再过一层可选标定 | `families/vela2/answers.py:82-99` |

decision1 的 docstring 给这套统计起了个名字：`decision_type_aware_v1`（`answers.py:1-8`）。
**而在 committed 产物上可以直接验证这确实是两条不同的公式**（实测，§14）：

```
decision2 的 golden answers：|1 − H/log k − 记录值| 的最大值 = 0
                             |1 − var/uniform − 记录值| 的最大值 = 0.2817739191896361
```

也就是：Decision 2.0 的记录 `confidence` 与归一化熵**逐位吻合**，
而与 Decision 1.0 / Vela 2.0 用的离散度公式**最多差 0.28**。
同一个 JSON 字段名 `confidence`，隔一代就是另一个定义。

**同一个答案里还有第三个数，而且来自另一次归一化。** Vela 2.0 的 `abstain_probability`
是「包含弃权位的那次 softmax 的最后一维」（`families/vela2/answers.py:76-80`），
而同一行的 `probabilities` 是「只对真实选项做的那次 softmax」（`:66-68`）。
两次归一化的分母不同（一次含 `[ABS]`，一次不含），所以
`Σprobabilities + abstain_probability ≠ 1`——**这两个数不能相加，也不是同一个分布的切片**。
契约里也把它标成 *not calibrated*（`api/openapi.yaml:549-552`）。

### 10.2 谁在改发布出去的数值

| 环节 | 在哪 | 改数值？ | 改 argmax？ |
|---|---|---|---|
| **训练本身** | 不在本仓（§0 的边界） | 是 | 是 |
| 温度（仅 decoder 路径） | `qwen.py:134`，值来自包 | 是 | 否（单调） |
| `score_bias`（仅 Decision 2.0 的 score） | `families/decision2/family.py:352-355` | 是 | 不适用（Score 无 argmax，但有期望值与 confidence） |
| vela2 的 `noul_calibration` | `families/vela2/calibration.py:135-142` | 是 | **可以**（`sigmoid(logit(p)/T + b)` 非单调映射到 [0,1] 后可能跨过 0.5） |
| vela2 的阈值族 | `calibration.py:70-133` | 是（Set / Span 的选中集合） | **是**（阈值就是 argmax 的替代品） |

三条与另外七篇不同的性质：

1. **Vela encoder 完全没有标定**（§5.3）。包一带 calibration 就拒载
   （`families/decision1/package.py:139-140`）。
2. **Decision 2.0 的 `score_bias` 是「按等级数查表」的 logit 偏移**：
   `apply_score_bias(offsets, logits, level_count)` 取 `offsets[level_count]` 逐位相加
   （`families/decision2/answers.py:14-25`），**加法发生在除以温度之前**（`family.py:352-360`）。
   表本身绑定在 checkpoint 摘要上——`load_score_bias` 要求
   `report["model_sha256"] == model_sha256`（`package.py:285-286`），
   且文件摘要必须同时匹配 `MODEL_MANIFEST.json` 与包内文件（`:273-277`、`:287-289`）。
   这是七篇里少见的「**标定产物绑权重哈希**」，比 OpenJev(HF) 的「文件默认值 ≠ 发布值」更硬。
3. **vela2 的阈值不是常数而是规则族**（`families/vela2/calibration.py`）：
   `set_threshold` 按 `set:<question_id>` → `set:*` 回退（`:70-72`）；
   `span_threshold` 先判是不是 PII（`:74-75`：question id 是 `pii`，或标签集 ⊆ 17 个训练过的 PII 类型），
   PII 走**对数-对数分段线性插值**的长度规则（`pii_length_threshold`，`:77-87`：
   在 `log(tokens)` 与 `log(threshold)` 空间里 `np.interp`），
   再叠一道**稀疏文档闸门** `sparse_gate`（`:112-133`）：
   如果文档在 probe 阈值（默认 0.5）下解出的 span 数 ≤ K，就把阈值放宽到 `t_sparse`。
   全部命中都会回一个 `rule` 字符串（`"pii:length"` / `"pii:<bucket>"` / `"span:*"` / `"halu"` / …），
   并且**这个字符串会随答案一起发布**（`answers.py:180` 写进 `thresholds[question.id]`；
   `report_heads` 打开时还写 `span_heads`）。

### 10.3 温度：两种包格式，两种守卫强度，且都只在 decoder 上存在

同一个仓里，Decision 1.0 与 Decision 2.0 对「温度」这件事的管法**不一样**：

| | Decision 1.0 的 decoder | Decision 2.0 |
|---|---|---|
| 缺省 | **不许缺**：`(有常数) == (有文件)` 即 `PackageError` | **缺省为 1.0**：`manifest` 里没有 `calibration` 就 `dict.fromkeys(TASK_TYPES, 1.0)` |
| 形状 | 一个常数，或一个文件（文件里给 `{"temperatures": {…}}`，或只给一个常数键） | 只认 `temperature_by_type`，且**必须三型齐全** |
| 值域 | 有限且为正（布尔值也拒） | **`0.05 ≤ T ≤ 20.0`，越界即 `PackageError`** |
| 格式标识 | 无 | `calibration_version` 必须是 `decision2-per-type-temperature/1` |
| 绑权重 | 无 | 报告里的 `model_sha256` 必须等于 checkpoint 摘要 |
| 绑上下文 | 无 | 报告里的 `inference.max_length` 必须等于包的输入上限 |

（Decision 1.0 侧：`families/decision1/package.py:135-140`、`:167-175`、`:178-191`；
Decision 2.0 侧：`families/decision2/package.py:64`、`:67`、`:319-356`。）

三点值得单独说：

1. **「缺键 = 1.0 = 未校准」这条约定与 Quyet 相同**，但只适用于 Decision 2.0。
   Decision 1.0 的 decoder 反过来要求**必须显式给**——不给就拒载。
2. **值域检查是「拒」而不是「夹」**。Laya 的 `clamp_temperature` 把 `choice:11+` 的 0.1006
   悄悄改成 `TEMP_MIN = 0.5` 再发布（Laya §10.3）；这里越界直接 `PackageError`，
   **包根本装不上**。两者都是「不信任拟合值」，但一个改数、一个拒载。
3. **仍然没有接受 / 拒绝判据**——值域只是合理性边界，不是「留出集必须变好」。
   在这一点上它与 Jeff / Open-Jev / Laya / OpenJev(HF) 同侧，与 RYOTIDE 的预先注册判据相反。

温度的实际数值**本文不给**：它们在各模型包的 `config.json` / `MODEL_MANIFEST.json` 里，
本机没有这些包。§13.2 给出取法，取到之后按上面两张表自查是哪一种形状。

### 10.4 Go 侧**不用**模型报的那个 `confidence`

这是本文第二处「算得出来 ≠ 接进来了」（对照总表 §4.5 的口径），并且可以逐行核实：

模型报的 `confidence` 在 Go 侧只被**解码和校验**，从不参与信号：

| 动作 | 位置 |
|---|---|
| 解码进 `Answer.Confidence` | `pkg/modelservice/client.go:408-409` |
| 范围校验（必须 ∈[0,1]，否则整条判 `invalid_model_output`） | `pkg/modelservice/answer_check.go:35` |
| 有限性校验 | `pkg/modelservice/client.go:427-428` |

而信号里的 confidence 是 Go **自己从 `probabilities` 重算的**
（`pkg/classification/classifier_signal_decision_model.go:167-200`）：

| 题型 | SignalValues | SignalConfidences |
|---|---|---|
| Noul | `answer.Noul` | `answer.Noul` |
| Score | `answer.Score` | **`maxProbability(answer.Probabilities)`** |
| Choice | 每个选项一条 + 选中项一条 | **`answer.Probabilities[answer.Choice]`**（即选中项的概率） |

也就是说：**Score 的 `confidence` 从「1 − 归一化方差」换成了「最大概率」，
Choice 的从「top-two margin」换成了「选中项的概率」。** 模型的统计量被算出来了、
被契约要求必须有（缺失解码为 0.0，落在 [0,1] 内也照样通过）、
但**决定路由的那两个数不是它**。

后果是具体的：如果调用方按文档理解 `confidence` 是「这个答案有多可信」，
他拿到的数与路由器实际用于门控和排序的数**是两个不同的量**。
这正是另外七篇反复立的同一块牌子——Jeff 的 serving rule、Open-Jev 的 `choice_confidence`、
Quyet 的 `confidence = max(p)`、RYOTIDE 的 `margin`——在**同一个系统内部**又出现了一次，
而且这次是「生产者和消费者各有一套」。

Go 侧的聚合还有一层：AND 节点只在「恰好一个 evidence 叶上报了 confidence」时沿用它的值，
否则取均值并把 `scored` 置 false（`pkg/decision/engine.go:761-763`）；
未上报的信号取结构常量 1.0（`:451-459`）；排序时只有「全员已评分且 ScoreKind 相同」的池可比
（`pkg/decision/selection.go:48-77`）。

---

## 11. 四条不变量的落位

RYOTIDE §13 的四条，在这一路上的落位**整齐得出乎意料**——因为它的设计目标就是把发布包的读法原样复现。

| RYOTIDE §13 不变量 | 在这一路的对应物 | 状态 |
|---|---|---|
| **#1** 读位置必须落在标记上（诊断量 `marker_mass`） | 位置是 `render` / `collate` **构造并记录**出来的（`markers` / `endpoints`）；Qwen 路径还有端点互不相同、必须落在 query 之前的显式断言（`segments.py:88-91`） | **结构性成立**；诊断量换成「端点合法性断言」而不是质量占比。**没有任何质量占比类的诊断量**——因为候选不是从词表里挑的（§5.1） |
| **#2** 前缀与标记表面形式耦合，必须成对扫 | 标记是 token **id**（Vela 的 `mask_token`、Vela 2.0 的 `marker_ids`），不存在「表面形式」这个自由度 | **不适用**（与 Laya 同侧）。Vela 2.0 还多一条：`marker_ids` 的名字必须**恰好**是那八个（`encoder_layout.py:105-108`） |
| **#3** 只读最后一个位置 | Vela 读 K 个 `[marker]`；Qwen 读 K 个端点 + 1 个 query；Vela 2.0 读每题若干标记 | **改写**：不是「只读一个位置」，而是「一次前向只读构造出来的那些位置」。**「不物化词表维度」这条在 decoder 路径成立**（候选头输出 `[rows, width]`），在 Vela 路径同样成立（scorer 输出标量） |
| **#4** 比较概率 / ECE 前先确认 temperature | 温度**逐 runtime 不同**：Vela 无温度、Qwen 必有、vela2 在 `calibration.json` 里还有阈值族 | **成立，但要确认的不止 T**：还要确认**是哪一套 `confidence` 公式**（§10.1 三套）与**Go 侧用的是哪一个数**（§10.4） |
| （各自新增） | **profile 是数值的一部分** | 同一个模型四档：`exact` 目标逐字节相同，另三档是近似的（§8）。跨实现引用延迟或概率前，先确认跑的是哪一档 |
| （各自新增） | **标定产物必须绑权重哈希** | Decision 2.0 的 `score_bias` 报告里带 `model_sha256`，与 checkpoint 不符即拒载（`families/decision2/package.py:285-286`） |
| （各自新增） | **多模型同进程的 kernel 选择是全局状态** | 一张 GPU 上多个模型共用一个进程时，autotuner 的选择是进程级的；选错会改答案且**照样通过 golden 自检**（§9.4） |

---

## 12. 文件对照

| 文件 | 行数 | 职责 |
|---|---|---|
| `src/model-runtime/vllm_srun/systemone.py` | 271 | System One 的请求语义：题型、选项上限、state / question 校验。**五个决定系列共用的唯一一层** |
| `…/families/decision1/family.py` | 305 | Decision 1.0 家族：包校验、两条 runtime 的分派、golden 请求 |
| `…/families/decision1/model.py` | 268 | `Decision1Model` 与两条子类：`plan`（含「一题超长拖垮全请求」）、`run`、`run_approximate` |
| `…/families/decision1/vela.py` | 332 | `vela-encoder`：prompt 布局、`[marker]` 位置、8 题物理批、三题型层堆、长度分组读出 |
| `…/families/decision1/qwen.py` | 150 | `qwen3.5-decision`：`prompt_version`、8 题物理批（32 对齐）、`softmax(logits/T)` + host 重归一化 |
| `…/families/decision1/answers.py` | 78 | 概率 → typed answer：`decision_type_aware_v1` 的两条 confidence 公式、±2e-5 的自洽校验 |
| `…/families/decision1/questions.py` | 107 | question / preset → `Row`，Noul 的默认描述，`MAX_QUESTIONS = 1024` |
| `…/families/decision1/package.py` | 304 | 包格式 `vllm-sr-decision/1`：文件图、逐文件 SHA-256、温度（常数或 per-type 文件）、`QUESTIONS.json` 预设 |
| `…/families/decision2/family.py` + `package.py` + `answers.py` | 364 + 401 + 95 | Decision 2.0：共享候选头、LoRA/适配器打包、`score_bias` 绑哈希、归一化熵 confidence |
| `…/families/vela2/*` | ~3 600 | Vela 2.0：8 个标记的布局、窗口读、Set / Span、阈值族与 PII 规则 |
| `…/heads/typed.py` | 119 | Vela 的题型头：题型嵌入 + 2 层 pre-norm Transformer + MLP scorer，强制 FP32 |
| `…/heads/candidate.py` | 105 | 共享候选头（Decision 2.0 全部 + Decision 1.0 decoder），双线性 + 门控 MLP |
| `…/heads/marker.py` | 84 | Vela 2.0 的标记头：cosine / mlp / both 三种读出 |
| `…/text/segments.py` | 105 | 分段的「候选端点 + 全局 query」布局，被 Decision 1.0 decoder 与 Decision 2.0 共用 |
| `…/plugins/decisions.py` | 338 | `decisions` surface：`plan` / `answer` 契约、`states` 拆分与合并、golden 比较 |
| `…/plugins/base.py` | 991 | 引擎与家族的抽象：`ForwardBatch`（`gather` / `query` / `shared_prefix`）、`TreeBatch`、`Profile` |
| `…/runtime.py` | 1 360 | 进程级运行时：加载 → 校验 → 装 profile → **golden 自检** → ready；`output_tokens: 0` 在这里 |
| `…/supervision/readiness.py` | — | `golden_check` 与两个容差（CPU 1e-3 / GPU 0.02） |
| `…/profiles/exact.py` `batching.py` `max_speed.py` `shared_context.py` | 109 / 58 / 56 / 128 | 四档数值语义 |
| `…/api/openapi.yaml` + `app.py` | 1 539 + 289 | 唯一契约；`/v1/decisions` 与 `/v1/systemone` 是同一条路由 |
| `…/registry/tables/decision1.py` `decision2.py` `vela2.py` | 192 / 78 / 123 | 每个发布模型的 revision、逐文件 SHA-256、参数量、kernel choices |
| `…/registry/golden_answers_decision1.json` 等 | — | **committed 的加载期参考值**，也是本文 §14 的唯一数据源 |
| `src/semantic-router/pkg/modelservice/client.go` | — | Go 侧的 HTTP 客户端（契约生成）；`encodeDecisionRequest` / `decodeAnswer` |
| `…/pkg/modelservice/answer_check.go` | — | Go 侧的一致性校验（Σp 容差 **0.02**，比 Python 侧的 2e-5 宽五个数量级） |
| `…/pkg/classification/classifier_signal_decision_model.go` | — | state = 请求原文；typed answer → 信号；**fail-open** |
| `…/pkg/decision/engine.go` | — | 消费信号与 confidence：AND/OR 聚合、可比池、排序 |
| `tools/calibration/systemone_auto/*` | ~3 000 | 拿 JevBench 做闭环评测与级联门标定（§15） |
| `src/model-runtime/docs/records/decision1-parity.md` `decision1-performance.md` | — | **上游记录**的 parity / 性能数字，本文凡引用都注明 |

---

## 13. 亲手跑一遍

### 13.1 复现本文的全部实测（不需要权重、不需要 torch）

§14 的每一条都来自这一条命令的形状——**只读 committed 的 golden answers，用标准库算术重算**：

```bash
cd /path/to/semantic-router/src/model-runtime/vllm_srun/registry
python3 - <<'PY'
import json, math
# 1) Decision 1.0：choice 的 confidence = top-two margin，score 的 confidence = 1 - var/uniform
doc = json.load(open("golden_answers_decision1.json"))
worst = 0.0
for repo, entry in doc.items():
    for dev, ans in entry["answers"].items():
        for qid, a in ans.items():
            if a.get("type") not in ("choice", "score"): continue
            v = list(a["probabilities"].values()); n = len(v)
            if a["type"] == "choice":
                d = sorted(v, reverse=True); c = min(1.0, max(0.0, d[0]-d[1]))
            else:
                m = math.fsum(i*p for i, p in enumerate(v))
                var = math.fsum(p*(i-m)**2 for i, p in enumerate(v))
                c = min(1.0, max(0.0, 1.0 - var/((n*n-1)/12)))
            worst = max(worst, abs(c - a["confidence"]))
print("decision1 max|recomputed - recorded| =", worst)

# 2) Decision 2.0：confidence = 1 - H/log k（另一条公式）
doc = json.load(open("golden_answers.json"))
ent = var = 0.0
for repo, entry in doc.items():
    for dev, ans in entry["answers"].items():
        for qid, a in ans.items():
            if a.get("type") not in ("choice", "score"): continue
            v = list(a["probabilities"].values()); n = len(v)
            H = -sum(p*math.log(p) for p in v if p > 0)
            m = math.fsum(i*p for i, p in enumerate(v))
            V = math.fsum(p*(i-m)**2 for i, p in enumerate(v))
            ent = max(ent, abs(max(0.0, min(1.0, 1.0 - H/math.log(n))) - a["confidence"]))
            var = max(var, abs(max(0.0, min(1.0, 1.0 - V/((n*n-1)/12))) - a["confidence"]))
print("decision2 |1-H/log k - recorded| =", ent, " |1-var/unif - recorded| =", var)

# 3) host 重归一化的分叉：encoder 的 Σp 差一点、值是 float32；decoder 精确为 1、值不是 float32
import struct
ENC = ("Kai", "Lex", "Route")
f32 = lambda x: struct.unpack('f', struct.pack('f', x))[0]
doc = json.load(open("golden_answers_decision1.json"))
sums, vals = {}, {}
for repo, entry in doc.items():
    key = "encoder" if any(e in repo for e in ENC) else "decoder"
    for dev, ans in entry["answers"].items():
        for qid, a in ans.items():
            if not a.get("probabilities"): continue
            p = a["probabilities"]
            sums.setdefault(key, []).append(abs(math.fsum(p.values()) - 1))
            vals.setdefault(key, []).extend(v for v in p.values() if v != 0.0)
for k in sums:
    off = sum(1 for v in vals[k] if f32(v) != v)
    print(f"{k}: n={len(sums[k])}  max|Σp-1|={max(sums[k]):.3e}  "
          f"值不是 float32 的个数={off}/{len(vals[k])}")
PY
```

**实测输出**（本文实跑，Python 3 标准库，未加载 torch、未下载任何权重）：

```
decision1 max|recomputed - recorded| = 0.0
decision2 |1-H/log k - recorded| = 0.0  |1-var/unif - recorded| = 0.2817739191896361
decoder: n=16  max|Σp-1|=0.000e+00  值不是 float32 的个数=48/48
encoder: n=14  max|Σp-1|=1.016e-07  值不是 float32 的个数=0/64
```

三条结论都由此得到：§10.1 的「隔一代换一条 confidence 公式」（0 对 0.2818）、
§5.2 的「encoder 不重归一化、decoder 重归一化」（和与 float32 可表示性两个角度同时命中）、
以及 §5.1 的「候选概率就是设备侧 softmax 的原始输出」。

### 13.2 跑真的前向（需要权重与 CPU/GPU）

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install ./src/model-runtime
vllm-srun serve vllm-sr/Decision-2.0-Kai-0.6B --device cpu --port 8100
```

（`src/model-runtime/README.md:18-20` 给出的就是这几条命令。）然后：

```bash
curl -s localhost:8100/v1/decisions -H 'content-type: application/json' -d '{
  "state": "Write a Python function that merges two sorted lists.",
  "questions": {
    "domain":    {"type": "choice", "instructions": "Which domain is this?",
                  "criteria": {"code": "Programming", "math": "Mathematics", "other": "Anything else"}},
    "reasoning": {"type": "noul",   "instructions": "Does this need multi-step reasoning?"}
  }
}'
```

看四件事：

- `usage.output_tokens` **必须是 0**——值来自 `runtime.py:961-963` 的硬编码，不是模型报的；
- `usage.input_tokens` 是**逐题累计**的，即使你开了 `shared_context`（§7.4）；
- `GET /health` 的 `golden` 必须是 `matched`——**跑不过它模型根本不会 ready**（§9.2）；
- `GET /v1/models` 里有 `profiles` 与 `golden` 状态；请求体里加 `"options": {"return_meta": true}` 时
  响应会带 `meta.model_sha256` / `meta.profile` / `meta.numerics`，这是确认「你跑的到底是哪一档」的
  唯一途径。同一处还能逐请求换档：`"options": {"profile": "shared_context"}`（`runtime.py:895-901` 校验
  它必须是已装载的那几档之一）。

模型侧的参考值可以直接看 committed 文件（`registry/golden_answers_decision1.json` 等），
或者用官方的记录工具重新采一份：

```bash
python3 tools/golden_answers.py vllm-sr/Decision-2.0-Kai-0.6B --device cpu
```

**本文没有给出任何前向产生的概率数值。** 本机没有这些模型包（`vllm-sr/Decision-1.0-*` 等），
§14 的所有数字都来自 committed 的 golden answers 及其重算，
§8/§9.4 的数字来自 `src/model-runtime/docs/records/` 里上游记录的实测。

---

## 14. 实例走查：在 committed golden answers 上重算

记录文件：`src/model-runtime/vllm_srun/registry/golden_answers_decision1.json`。
它是各模型的**加载期参考值**，由 `tools/golden_answers.py` 采集，每个模型按 `cpu` / `rocm`
各存一份，并带该模型的 40 位 revision。下面是其中最完整的一条（Decision-1.0-Eos-0.8B，
revision `2ca39a23e3f4500873ebcd8f97394268e27d3167`，`golden_answers_decision1.json:2,67`）。

### 14.1 一条 Score 答案（`difficulty`，`legend` = Trivial / Moderate / Hard）

| 字段 | 记录值（cpu） | 本文重算 | 一致 |
|---|---|---|---|
| `probabilities` | `{0: 0.23944071711776505, 1: 0.5190608346742143, 2: 0.2414984482080207}` | — | — |
| `score` | `1.0020577310902556` | `0·p₀ + 1·p₁ + 2·p₂` = 1.0020577310902556 | 是（|Δ| = 0） |
| `confidence` | `0.2785976033971811` | `1 − var/uniform`，var = 0.4809672…，uniform = 8/12 | 是（|Δ| = 0） |
| `Σp` | — | `1.0000000000000000` | 是 |

同一模型的 rocm 一档是 `score = 1.0015055536775126`、`confidence = 0.27395595051005683`——
**同一段输入、同一份权重，两个设备类别的答案不是同一个数**。这正是 golden 检查要按设备类别
给两个容差（CPU 1e-3 / GPU 0.02）的原因（§9.2）。

### 14.2 一条 Choice 答案（`domain`，3 选项）

```
probabilities = {code: 0.9920034703201491, math: 0.0061639904034322245, other: 0.0018325392764186542}
choice        = "code"
confidence    = 0.9858394799167169
```

重算：`top-two margin = 0.9920034703201491 − 0.0061639904034322245 = 0.9858394799167169` —— **逐位相同**。
注意这个 0.9858 是 **top-two margin**，不是最大概率 0.9920。若把它当成「模型有多确信」，
你会比实际多算 0.006；若把它换到 Decision 2.0 的模型上，**字段同名但公式不同**（§10.1）。

### 14.3 一条 Noul 答案

```
reasoning = {noul: 0.6144479513168335, type: "noul"}
```

Noul **没有 `confidence` 字段**（`families/decision1/answers.py:58-59` 只返回 `type` 与 `noul`）。
Go 侧因此拿 `P(true)` 本身当 confidence（§10.4）；而**路由规则里 Noul 的默认阈值是 0.5**
（`pkg/config/decision_signal.go:40`）。同一段 golden state、同一道题，七个模型给出的 `P(true)` 是：

```
Kai-0.6B   0.323871    Lex-0.6B  0.096317    Route-0.6B 0.338411   （三条 vela-encoder）
Eos-0.8B   0.614448    Sol-2B    0.397374    Nox-4B     0.816205
Lux-9B     0.674865                                               （四条 qwen3.5-decision）
```

在默认阈值 `0.5`（`>=` 判真）下，**七个模型里只有 Eos 0.614、Nox 0.816、Lux 0.675 三个判「是」**，
另外四个判「否」——而它们的题面、选项、契约字段名一个字都没变，只是 checkpoint 不同。
这是「阈值必须与模型配对」最直白的一处：**换一个 Decision 1.0 权重，同一道题的 `P(true)` 从 0.096 走到 0.816。**

### 14.4 一条多选项 Choice（Route 的 preset，14 个选项）

`vllm-sr/Decision-1.0-Route-0.6B` 的 `preset:domain` 有 14 个选项（`golden_answers_decision1.json:365-385`），
最高项 `computer science = 0.9981111288070679`、次高 `engineering = 0.0013850410468876362`：

```
记录 confidence = 0.9967260877601802
重算 d₀ − d₁    = 0.9981111288070679 − 0.0013850410468876362 = 0.9967260877601802   ← 逐位相同
```

14 个选项把 `MAX_OPTIONS = 255` 的余量留得很足，而 margin 是 **0.9967**——
按 `answers.py:27` 的 `min(1.0, max(0.0, …))` 截断，这个数离上界只剩 0.0033。
它是本文**唯一一条多选项样本**：三个选项那两条（§14.1/14.2）证明不了「选项数不改变公式」，
这一条可以。**30 条记录里，重算与记录不等的有 0 条。**

### 14.5 这批记录能证明什么、不能证明什么

- **能**：`score`、`confidence` 两条公式在 Decision 1.0 上被 30 条记录**逐位证实**
  （最大偏差 0.0，§14.1–14.4）；Decision 2.0 的归一化熵公式被 24 条记录逐位证实，
  而在同一批 24 条上离散度公式最大差 0.2818（§10.1）；
  encoder / decoder 的重归一化分叉被 30 条分布从两个独立角度证实（§5.2）。
  这些都是**可复算的恒等式**，不是跑分。
- **不能**：这些数字**不是本文跑出来的**，是别人在 CPU 与 MI325X 上采的。它们证明的是
  「发布出去的这批数字符合源码里的公式」，不证明「公式是对的」。要复现记录本身，
  得按 §13.2 装上模型包重跑 `tools/golden_answers.py`。

---

## 15. 它拿 JevBench 做了什么

另外七篇都在**被 JevBench 评**。这一路多了一层：它还**拿 JevBench 反过来标定自己的决策门**。
代码在 `tools/calibration/systemone_auto/`（约 3 000 行，纯标准库 + 一个 Go 小工具）。

### 15.1 钉死的上游

`jevbench_suite.py:17-30` 把上游钉成常量：

```python
COMMIT = "b6b8fff7e345b98c060ad26c13308860ddc67004"
THRESHOLD = 0.6059704079536342
ARMS = ("direct_kai", "direct_vega", "auto")
PUBLIC_ITEM_COUNT = 231
PUBLIC_GROUP_COUNT = 195
REVISIONS = {"kai": "cd49ea3813fd8ba0928a9a23ef6c9a0f2f0cd764",
             "vega": "7aec49ae11a18741706da549ab626b9052795fe7"}
DATA_SHA256 = {"easy": "231df3c2…", "original": "5c2414ed…", "hard": "89e9e6be…"}
```

`load_suite()` 会 `git rev-parse HEAD` 与实际对比，不符就 `ValueError`（`:33-42`）——
**评测集的身份是硬校验，不是文档约定。**

### 15.2 打分口径：三处必须与上游一致

`metrics.py:62-93` 的 `_question_result` 是答案 → 分数的全部：

| 题型 | prediction | loss |
|---|---|---|
| Choice | `answer["choice"]`，必须落在声明的 `criteria` 里 | `float(prediction != gold["label"])` |
| Noul | `value >= NOUL_DECISION_THRESHOLD`（`0.5`，`metrics.py:11`） | 同上 |
| Score | `math.floor(score + 0.5)`（对 expected level 取整） | `abs(score − gold_level) / max_level` |

而**概率层的校准用的是众数答案，不是取整后的 expected Score**（`metrics.py:118`
的注释原话：*Probability calibration uses the modal answer, not a rounded expected Score*）。
上游 README 专门为此留了一段警告，说这两种口径会给出不同的数字——
**「Score 答案对不对」在这个评测里有两个互不相等的定义。**

`Set` 与 `Span` 在这条链上是**不支持的**：`raise ValueError("pilot supports Choice, Noul and Score only")`
（`metrics.py:93`）。也就是说 Vela 2.0 那半套题型没有进 JevBench 评测。

### 15.3 级联门：一个被明确声明「不是正确率」的阈值

`THRESHOLD = 0.6059704079536342` 是 **Kai → Vega 级联**的早退门：Kai 先答，
整个 bundle 的每个答案都过门才不升级到 Vega。上游 README 对它的定性值得逐字引用：

> Its frozen early-exit threshold is `0.6059704079536342` for every Choice, Score and Noul answer.
> The entire bundle must pass. This value was selected on a separate 192-source calibration set
> before evaluating the public suite; **it is not a probability of correctness or a fixed fast-path quota.**

**「它不是正确率的概率」**——这块牌子前面已经立过四处（RYOTIDE 的 margin、Jeff 的 serving rule、
Open-Jev 的 `choice_confidence`、Laya 的 clamp 注释），这里是第五处，而且是最直白的一处。
它与 §10.4 的发现并读特别有味道：**模型报的 `confidence` 路由器不用，
路由器门控用的是自己从 `probabilities` 重算的那个数，而这个门的阈值是在另一套数据上拟合的。**

### 15.4 这一节本文不做的事

- **不引用任何质量数字**：上游 README 明说 *The initial quality–cost target remains an experiment,
  not a claimed result*，而仓里也没有 committed 的 JevBench 结果文件。
- **不把 `THRESHOLD` 当成可移植的常数**：README 说 *A different serving profile or dataset
  calls for fresh validation*，而 §8 已经证明 profile 会改数值——**换 profile 就等于换一套标定。**
- **不把 231 条当成完整榜**：README 说的是 *This is a public-suite evaluation; it is **not** the full
  official v1.6.1 Intelligence, Capability or Composite leaderboard score.*

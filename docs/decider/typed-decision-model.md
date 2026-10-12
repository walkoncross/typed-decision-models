# decider 是如何用一次前向在 K 个答案槽位上读出类型化决策的

> 代码库：<https://github.com/Mapika/decider>
> 对应 `README.md` 第 7–11 行：*A language model that does not generate text. It reads a **state** and a set of
> **typed questions** and returns, from one forward pass, a probability distribution for every question.*
> 本文按 `decider/`（读数、引擎、服务、训练、校准）的实际代码逐环节说明这句话的实现方式；
> 文中出现的文件名与行号（`prompt.py:153`、`model.py:28`、`serve.py:546`、`shared_prefix.py:206` 等）
> 均相对该仓库根目录。
>
> **同名配对文档**：RYOTIDE 的 [`docs/ryotide/typed-decision-model.md`](../ryotide/typed-decision-model.md)。
> §0 先讲清分歧，§11 把它的四条不变量逐条落位。

decider 的读法在结构上**最接近 RYOTIDE**：把冻结或微调过的 LLM 的 `lm_head` 当作标签分类器，
一次前向、只取若干位置的 logits、掩码到已知标签集、softmax。它在这一条线上往前推的是三件事：

1. **一次前向读 K 个槽位，而不是只读最后一个位置**。K 个问题各有一个 `Answer k: (` 槽位，
   全部落在同一行里，所以「多问一个问题」不改变任何已有答案（§4、§7）。
2. **同一个读法读两种权重**：自训的 0.8B/2B/4B/12B/35B-A3B，以及**不训练、只靠一个 `decider_config.json`
   的 chat 布局**读进来的 stock 指令模型（`decider-chat-*`）（§8）。后者使它同时是一条「推理技巧」。
3. **置信度有两个同名公式，而且同一条读法上确实用了两个**：`decide()` 报的是 `p_max`，
   `system_one()` 报的是 TypeSafe 的 rescale 公式，两者都能在 committed 产物上复算（§10、§14）。

先说三句结论，后面逐条给代码位置：

- 「typed」来自**分词器**：标签表是 255 个**单 token** 的大写标签（`A`–`J`，再 `A`–`Z` 与 26×26
  里能单 token 化的双字母），窄渲染保持 v1 原样，宽渲染**用 token id 拼**以保证每个标签恰好一个 token（§2）；
- 读位置是**训练期固定的构造**，运行期没有任何标定；槽位是 `Answer k: (` 这个**左括号 token**，
  读它的 hidden state 再用 `lm_head` 的标签行投影，等价于「下一个 token 是哪个字母」（§3）；
- 概率是**训练之后**由逐题型温度改写的，温度来自模型包里的 `decider_config.json`，
  拟合**不会自动写回**（`calibrate.py` 只 print JSON，写回是 `scripts/evaluate.sh:13` 注释里的人工步骤）（§10）。

## 本文的事实来源

| 类别 | 来源 | 本文中的标记 |
|---|---|---|
| 代码引用 | `decider/**`（`prompt.py`、`model.py`、`systemone.py`、`infer.py`、`engine*.py`、`schema_engine.py`、`shared_prefix.py`、`batching.py`、`serve*.py`、`temperature.py`、`calibrate.py`、`train.py`、`metrics.py`、`evaluate.py`、`data/**`、`vision/model.py`） | `文件:行` |
| 可实跑得到的产物 | `tests/data/temperature_1_3_0_pins.json` 与 `tests/data/plain_layout_pins.json` 是**committed 的存盘产物**（后者是 sha256 摘要，前者是**概率数值**）；`tests/test_systemone.py` 里的 `REPORT` 是一份**存盘的 issue 记录** | 标注「实测」，并注明是「在 committed 产物上重算」 |
| 公式与常量 | `README.md`、`MODEL_CARD*.md`、`docs/RESULTS.md`、`docs/HISTORY.md`、`docs/CHANGELOG.md`、`docs/SERVING.md`、`docs/benchmarks/*.json` | `README:行`、`docs/...:行` |
| 未跑 | **本机没有 GPU、没有下载任何 `Mapika/decider-*` 权重、没有起过 `decider.serve` / `serve_vllm`、没有跑过 pytest** | 照例不给任何前向产生的概率数值 |

本文所有「实测」只有一种动作：**读 committed 的 JSON，用标准库（`json` + `math`）算术重算**（§14）。
没有运行上游的任何程序——没有 `import decider`、没有跑 `tests/temperature_cases.py`、没有起服务。
命令在 §13、§14 原样列出，并标明哪些**本文没有执行**。

---

## 0. 与 RYOTIDE 的分野

| 维度 | RYOTIDE | decider |
|---|---|---|
| **权重来源** | 公开 LLM，全程 frozen | **两种**：自训（Qwen3.5-0.8B/2B/4B/35B-A3B 微调；12B 是 Gemma-4-12B-it + 合并 LoRA）**与**不改权重、只用 `decider_config.json` 读进来的 stock 指令模型（`decider-chat-gemma4-31b`、`decider-chat-qwen3.6-27b`）（§8） |
| **是否训练** | 否，零新增参数 | **两边都有**：自训 checkpoint 训**全部参数**（含 `lm_head`，`train.py:95`）；stock 路径零训练。它自己把后者归为「inference techniques」（`README:137`） |
| **决策机制** | 读**标签 token** 的 logits → 掩码 → softmax | 读 `Answer k: (` 槽位的 hidden state → 用 `lm_head` 的标签行投影 → 掩码 → `softmax(logits / T)`（`model.py:28-31`）；**字母从不生成**，K 个槽位同一次前向 |
| **「typed」从哪来** | 分词器：标签必须单 token，否则映射成字母标记 | 分词器：**255 个单 token 标签**（`A`–`J` + `A`–`Z` + 229 个双字母），由 `label_table` 断言「恰好 255 个互不相同的 id」（`prompt.py:52`）；请求侧再限 2–255（`systemone.py:56-57`）（§2） |
| **读位置怎么定** | **运行期标定**：扫 9 前缀 × 2 表面形式，取 marker mass 最高 | **训练期固定**：槽位是 prompt 构造出来的 `Answer k: (` 的**左括号 token**（`prompt.py:153`）；运行期**没有标定**，也没有扫（§3）。唯一的位置「寻找」在 vision 分支：按 token 扫描并取最后 K 个（`vision/model.py:40-47`） |
| **读出来的是什么** | 末位**词表 logits**，再按标记 id 取出 | 只算**标签行**的 logits：`W = lm_head.weight[letters]`，`F.linear(hs, W)`（`model.py:28-29`）——**词表维度不物化**（§4） |
| **候选数上限** | 260（`A0`–`Z9` 两 token 两步读；>26 另有 `grouped` 近似） | **255**（`MAX_OPTIONS`，`prompt.py:12`）；>10 用宽渲染，每选项一个单 token 标签。255 是**分词器的上限**（`label_table` 断言），不是接口约定（§2） |
| **顺序偏置** | 多 order 共享前缀 fork，折叠回真实标签后**概率平均** | **训练时随机打乱**选项（`prompt.py:140`、`:65`、`:282`，训练入口传 `rng`，`train.py:16`）；运行期另有**可选**的「反转重读 + log 域平均」（`serve_vllm.py:139-159`），默认关。仓里**没有对顺序偏置的度量**（§6） |
| **置信度来源** | **读出来的**：只有 temperature 能改数值 | **训出来的**（微调本身改数值、也可能改 argmax）+ **逐题型温度**；此外还有 TypeSafe 的两个**公式**（不是学出来的），且 `decide()` 与 `system_one()` 用的是**不同公式**（§10） |
| **诊断量 / 守卫** | `marker_mass`（全词表落在标记上的份额） | **没有等价量**，因为全词表 logits 根本不出现，`marker_mass` 恒等于 1、无信息（§11 #1）。守卫改在**分词器与布局**上：`label_table` 的 255 唯一性断言、`ChatTemplate` 的「标签必须单 token 且不与该模板尾巴合并」断言（`prompt.py:229-233`）、`resolve_layout` 遇未知布局**报错而不是猜**（`prompt.py:183-185`） |
| **多问题共享 state** | `run_many()`：公共前缀预填充一次，**计费只算一次** | `independent=True` 每题一行共享同一段 state；`Engine.score_shared` 把前缀缓存**按内存预算分块 fork**，只跑问题后缀（`shared_prefix.py:186-235`）；`input_tokens` 用 LCP 去重（`systemone.py:192-199`）（§7） |
| **后端** | MLX（参考）+ torch | torch（CUDA graphs / `torch.compile` / FP8）、MPS、**llama.cpp（GGUF）**、**vLLM 0.29.0**（服务 stock 大模型）（§9） |
| **服务接口** | `/v1/systemone` + `/health`（含 `prompt_hash`），另有 MCP | `/v1/systemone` + `/decide` + `/v1/models` + `/health` + `/stats`；**两个服务器**：`serve.py`（自研引擎）与 `serve_vllm.py`（vLLM，只答 independent 题）（§9） |

一句话总结立场：**decider 是「RYOTIDE 的读法 + 可训练 + 工程化到两种权重与多种后端」**。
它没有改 RYOTIDE 的读数原理（仍是「标签行 logits → 掩码 → softmax」），改的是
**读几个位置**、**谁来训**、**温度怎么发**、**前缀缓存怎么分叉**这四件事。

---

## 1. 端到端数据流

```text
POST /v1/systemone  {"state": ..., "questions": {id: {type, instructions, criteria}}, "independent": true}
   │
   ├─ render_question()      题型校验：choice 2..255 / score 2..10 / noul 可省 instructions   systemone.py:40-74
   ├─ render_state()         长数组写进 "_index"，把「数第 47 个」变成查表                      systemone.py:22-37
   ├─ plan_rows()            score 且 isolated → 每个等级一行 yes/no；否则每题一行              systemone.py:106-115
   ├─ render_question 的 neutralize_options()  "none of the above" → "not listed here"        infer.py:33-44
   │
   ├─ check_size()          413（行数 / 单行 token / 总 token）                                serve.py:170-186
   ├─ _admit()              503（排队行数）                                                    serve.py:188-194
   │
   ├─ 一次前向 ──────────────────────────────────────────────────────────────────────────┐
   │   EngineV2（serve.py）／Engine（infer.py）／GGUFEngine／Decider eager                  │
   │   last_hidden_state [B,T,H]                                                            │
   │        → h[slot_batch, slot_idx]                    只取槽位                            model.py:27
   │        → W = lm_head.weight[letters]  [K,H]         只算标签行                          model.py:28
   │        → logits = cap_logits(linear(hs, W))         Gemma 的 softcap 在这里             model.py:29
   │        → masked_fill(ar >= nopts, -inf)             非法选项                            model.py:30-31
   │        → softmax(logits / T)                        T 逐槽位                            temperature.py:171-180
   └───────────────────────────────────────────────────────────────────────────────────────┘
   │
   ├─ assemble()             落回 names；isolated 的等级概率归一化 + fit_mass                  systemone.py:127-136
   └─ format_answer()        choice / noul / score + confidence / x_p_max / certainty         systemone.py:176-189
```

与 RYOTIDE 的流程图相比，这里多了两个分叉点（**哪条布局**、**哪个引擎**），少了一个环节（**没有标定**）。
两条布局（state-first / schema-first）决定槽位前面是「先题面后 state」还是「先 state 后题面」，
后者让题面成为一个**与 state 无关的前缀**，从而可以整段缓存（§4.1）。
三个引擎（`Engine` / `EngineV2` / `GGUFEngine`）共享同一套 prompt 与同一套后处理，
差别只在「怎么把 [B,T,K] 算出来」（§4.2）。

---

## 2. 「typed」从哪来：255 个单 token 标签 + 请求声明的 criteria

### 2.1 标签表由分词器定，不由接口定

`prompt.py:10-13` 三个常量划定了全部标签空间：

```python
LETTERS = "ABCDEFGHIJ"
NARROW = len(LETTERS)      # <= NARROW options: the original "(A) .. (J)" rendering, tokenized as a string (unchanged since v1)
MAX_OPTIONS = 255          # width of the label head. > NARROW options: "wide" rendering, one label token per option:
                           # A..Z then the first 229 two-letter upper-case strings that are single tokens (AA, AB, ...)
```

`label_table`（`prompt.py:38-55`）构造它们：按 `A..Z` 再 `AA, AB, ...` 枚举，**只保留能单 token 化的**，
凑满 255 个后断言唯一性：

```python
assert len(out) == 255 and len({i for _, i in out}) == 255          # prompt.py:52
```

于是「typed」的封闭性**不在掩码里，在分词器里**：能被读到的标签，就是这张表；
表外的字符串根本没有对应的 logits 行。这与 RYOTIDE 的机制同源（标签必须落到单 token），
但 decider 把它推到了 255，而且**同一张表的两段用的是两套渲染**：

- **窄渲染**（≤10 个选项）：`"\n(A) billing\n(B) technical..."` 整串 encode 成字符串（`prompt.py:71`），
  注释明说 *unchanged since v1*——即 v1 以来窄问题的 token 序列逐字节不变，这是它能拿 1.1.3 的
  摘要 pin 住老模型的原因（§14）。
- **宽渲染**（>10）：`"\n(" + <label token> + ") 选项文本"`，其中 `<label token>` 是**直接用 id 拼进去的**
  （`prompt.py:72-74`、`:146-151`）：

```python
_, lab_ids, open_ids = label_table(tok)
for j, oi in enumerate(opts):
    out += open_ids + [lab_ids[j]] + _enc_opt(tok, q.options[oi])       # prompt.py:73-74
```

注释解释了为什么不能拼字符串：*built from ids so every label is one token*（`prompt.py:146`）。
若拼字符串，`(AA)` 里的 `AA` 可能被 BPE 合并成别的切分，标签就不是一个 token 了——
这正是 RYOTIDE §2「必须成对扫前缀与表面形式」那个自由度，decider 用「按 id 拼」**消掉**了它。

### 2.2 请求侧的封闭性：三个题型、两个数

`systemone.py:12` 定义上限，`render_question`（`:40-74`）把三种题型都收敛到同一形状
`(question, options, names, type)`：

| 题型 | criteria 形状 | 选项数 | 落到 prompt 的 options |
|---|---|---|---|
| `choice` | map / list | **2–255**（`MAX_CHOICE`，`systemone.py:56-57`） | `"name"` 或 `"name: 描述"`（`:58`） |
| `score` | 有序列表 / `{"0": ..}` legend | **2–10**（`MAX_LEVELS`，`systemone.py:62-63`） | `"0: none"`、`"1: low"`…（`:64`） |
| `noul` | 可选 `{"true": .., "false": ..}` | 恒为 2（`:68-70`） | `"no"` / `"yes"`（可带描述） |

`noul` 的 `instructions` 在 TypeSafe 的 OpenAPI 里是可选的，decider 用一句固定问句补上
（`systemone.py:44-48`、`:77-80`：`NOUL_WITHOUT_INSTRUCTIONS = "Which answer fits the context?"`），
并要求 criteria 至少描述 true/false 之一，否则 `ValueError`。**问题 id 从不进 prompt**
（`systemone.py:8`：*ids are never shown to the model*）——答案按 id 组装回去，模型看不到它。

### 2.3 Score 的「逐级隔离」：一个题型，两种行数

`score` 默认走 **isolated** 路径（`systemone.py:74` 的 `isolated=bool(spec.get("isolated", True))`）：
每个等级单独成一行 yes/no 题，**不带等级编号、不带邻居**（`systemone.py:83-96`）：

```python
ISOLATED = "{q}\nProposed answer: {level}\nDoes the proposed answer fit?"     # systemone.py:84
```

等级概率 = 各行 `P(yes)` 归一化，另外**上报未归一化的总质量** `fit_mass`
（`combine_isolated`，`systemone.py:99-103`：*near 1 when exactly one level fits, low when none does,
high when several do*）。所以：

- **一个 score 问题会展开成 n 行**——这是 §9 里所有「行数上限」与「每行一个温度」的来源；
- 温度按**题型**给，一条 score 的 n 个等级行**都用 `score` 的温度**（`temperature.py:16-18`；
  `tests/test_temperature.py:358` 把这写成断言：`[[1.1, 1.1, 0.9, 0.9, 0.9, 1.1]]  # choice, noul, 3 isolated score levels, choice`）。

---

## 3. 读位置：训练期固定，槽位是 `Answer k: (` 的左括号

decider 没有标定环节——**没有任何代码去扫「哪个位置最像标记」**。位置是 prompt 构造时算出来的：

```python
ids.extend(piece)
slots.append(len(ids) - 1)          # position of " (" token        prompt.py:152-153
```

槽位落在 `"…\nAnswer k: ("` 的**最后一个 token**，也就是那个左括号。为什么读它就能得到字母分布：
因果 LM 在位置 *i* 的 hidden state 预测的是位置 *i+1* 的 token，而 *i+1* 正是字母该出现的地方。
`model.py:27-29` 把这个性质直接写成一次投影：

```python
hs = h[slot_batch, slot_idx]                       # [N,H]   ← 槽位的 hidden state
W = self.lm.lm_head.weight[self.letters]           # [K,H]   ← 标签 token 的 lm_head 行
logits = cap_logits(self, F.linear(hs, W).float()) # [N,K]
```

**这就是 RYOTIDE 的做法，只差一步**：RYOTIDE 先算全词表 logits 再按标记 id 取出，
decider 直接把 `lm_head` 的标签行取出来做线性层。数值上等价（同一组权重行），
但**词表维度 [B,T,V] 从未出现**，于是 §11 #1 的 `marker_mass` 在这里无处可测。

三条布局的槽位都在同一处语义位置，但拼接方式不同：

| 布局 | 槽位怎么放 | 行号 |
|---|---|---|
| state-first（默认） | `Context: <state>` → 每题 `Question k: … Options: …` → 每题 `Answer k: (` | `prompt.py:128-157` |
| schema-first | 先所有题面块（**可缓存前缀**）→ `\n\nContext:\n<state>` → 每题 `Answer k: (` | `prompt.py:78-92`、`:95-116` |
| chat（stock 模型） | 模板 head + 内容 + `chat.tail` → 每题一个 answer piece，仍读**最后一个 `(`** | `prompt.py:274-289`、`:235-237` |

chat 布局是唯一对位置有**额外断言**的地方——`ChatTemplate.__init__` 逐一验证
前 10 个标签在 `"Answer: ("` 之后确实各自单 token，并且**模板尾巴不会与答案前缀合并**
（`prompt.py:229-233`）：

```python
if tok.encode(f"{answer}: ({n}", add_special_tokens=False) != pre + [i]:
    raise ValueError(f"chat layout: the label {n!r} does not tokenize as one token after {answer!r}: (")
if tok.encode(tail + f"{answer}: (", add_special_tokens=False) != self.tail + pre:
    raise ValueError("chat layout: the chat template tail merges with the answer prefix")
```

这是本文里最接近 RYOTIDE「前缀与标记表面形式必须成对扫」的守卫，只是它**不扫，直接断言**。

**vision 分支是唯一的例外**：图像版本没有 token 记账可依赖（processor 会插入 image pad token），
于是它**扫描 token 序列找槽位**（`vision/model.py:40-47`）：找 `" ("` 且前一个 token 是 `":"`
且前 5 个 token 里出现过 `"Answer"`；匹配数不足即 `assert` 失败；**取最后 K 个**，
注释写明理由：

```python
found = found[-nq[bi]:]     # a state may quote "Answer 1: (" itself (agent traces); the real slots come last
```

这是一处少见的「解析式定位」——多数实现（含 decider 自己的文本路径）都靠构造而非解析。

---

## 4. 一次前向：K 个槽位一并进行

### 4.1 两条布局，一半一半地训

`train.py:16-17` 按 `schema_first_prob`（`scripts/train.sh:12` 传 **0.5**）为每条样本抽一种布局；
所以发布的模型**两种布局都会**，运行期选哪一种是一个开关：

- 默认 **state-first**（`infer.py:209`）；
- **schema-first 要显式打开**，而且它会**掉准确率**——`infer.py:139-141` 的注释直接写了代价：

```python
self.schema_first = bool(cfg.get("schema_first", False)) and self.eng is not None and not self.gguf   # default layout. Questions-first (the cacheable one) costs accuracy
self.T_schema = T_schema                                              # (about 1.5 points on fixed label sets, more elsewhere): opt in with schema()
```

schema-first 的收益在 `schema_engine.py`：题面/选项块是**与 state 无关的前缀**，其缓存
（6 个全注意力层的 K/V + 18 个 delta-net 层的 conv/递推状态，`schema_engine.py:4-8`）算一次，
之后每个请求只跑 `Context: <state>` 加若干槽位，且每种 (batch, 长度) 形状捕获一张 CUDA graph
（`schema_engine.py:84-108`）。`prepare` 支持 `independent=True/False` 两种前缀布局
（`schema_engine.py:56-78`：一个前缀装下所有题，或每题一个前缀）。

### 4.2 三个引擎，同一种后处理

| 引擎 | 前向怎么算 | 文件 |
|---|---|---|
| `DecisionModel.slot_logits`（eager，训练与 CPU/MPS 用） | 直接 `lm.model(...).last_hidden_state` → 取槽位 → 投影 | `model.py:23-32` |
| `Engine`（库路径，CUDA） | 先算**整张** `[B,T,K]`，再 `read_slots` 一次性 gather + 掩码 + softmax + 一次 D2H 拷贝 | `engine.py:107-109`、`:48-56` |
| `EngineV2`（服务路径） | 同上，但分桶到 8192、`warmup()` 捕获整张网格、`seal()` 之后**永不再捕获** | `engine_v2.py:30-53`、`:191-209` |
| `GGUFEngine` | llama.cpp 解码，只在槽位请求 logits，按 `llama_get_logits_ith` 取标签列 | `engine_gguf.py:95-115` |

四个引擎共享的只有两行数学：`W = lm_head.weight[letters]`、`softmax(linear(h, W) / T)`。
`Engine`/`EngineV2` 把它物化成 `self.W` 并在加载期 `detach().clone()` 冻住
（`engine.py:93`、`engine_v2.py:47`）——注意**冻结的只是读出头的那份副本**：
训练时 `W` 是从 `lm_head` 现取的、**带梯度**的（`train.py:95` 对 `model.parameters()` 建优化器，
不过滤任何参数；全仓没有 `requires_grad_(False)`）。

分桶是性能设计而非语义设计：右填充 + 因果层意味着 pad 位置不会影响更早的槽位，
所以一张 graph 可以在所有 ≤ 桶长的行上复用（`engine.py:1-6`、`engine_v2.py:10-12`）。
`GRAPH_MAX_T = 2048` 之上走 eager（`engine.py:14`），`EngineV2` 的阶梯到 8192、
更长则按 `LONG_STEP = 1024` 补齐后 eager（`engine_v2.py:24-27`）。

### 4.3 一处与 Gemma 有关的数值细节：softcap 要在 softmax 之前

`cap_logits`（`model.py:38-50`）把 Gemma 的 `final_logit_softcapping` 施加在**标签 logits**上，
`cap * tanh(x / cap)`，注释写明理由：*as its own LM head output has it*（`model.py:20`）。
`README:87-88` 记录了后果：1.7.0 之前读 Gemma checkpoint「太尖」，加回 softcap 才对齐。
这条被 `tests/test_softcap.py` 钉住。它不是可选的装饰——不加就是另一个分布。

### 4.4 `output_tokens = 0` 是字面量

三个服务与库入口都把这一项写死：

```python
"usage": {"input_tokens": unique_tokens(items), "output_tokens": 0}          # infer.py:221
"usage": {"input_tokens": len(row[0]) * h.P, "cached_tokens": sum(h.tps), "output_tokens": 0}   # serve.py:525（schema 分支）
"usage": {"input_tokens": unique_tokens(items, ctx_len), "output_tokens": 0}                     # serve.py:546
```

`serve_vllm.py:407` 与 `serve_v1.py:275,291` 同理。**唯一会生成 token 的是 `serve_vllm` 的读数实现**：
它给 vLLM 发 `max_tokens=1, temperature=0.0, logprob_token_ids=...`（`serve_vllm.py:190`），
但只取 `final.outputs[0].logprobs[0]`，**不消费那个 token**（`:195`）——这与 SemIf-OpenJev 的浏览器路径、
OpenJev(HF) 的 `max_tokens: 1` 是同一类交易：借一个能返回 logprobs 的接口读一次前向，
而响应里的 `output_tokens: 0` 仍是硬编码（其 `serve_vllm.py:407`）。

---

## 5. 掩码 + softmax：`-inf` 掩码与逐槽温度

掩码就一行（`model.py:30-31`，`engine.py:54`、`engine_gguf.py:145` 是同式的副本）：

```python
ar = torch.arange(MAX_OPTIONS, device=logits.device)[None, :]
logits = logits.masked_fill(ar >= nopts[:, None], float("-inf"))
```

`nopts` 是**这一槽位实际有几个选项**。所以 softmax 的分母里只有合法选项，
`-inf` 的那些贡献 0——**重归一化是掩码的副产品，不是单独一步**。
这与 RYOTIDE §5、SemIf-OpenJev §5 是同一个结构，区别是 decider 把「多少个合法」做成了
**逐槽位**的张量 `nopts`，因为一行里不同题目的选项数不同（`prompt.py:155`）。

温度也是逐槽位的（`temperature.py:147-180`）：

```python
def scaled_softmax(lg, temperature):
    """softmax(lg / T) over the last axis.  A scalar T is the 1.3.0 expression unchanged; a list gives one T per row of lg."""
    if isinstance(temperature, (list, tuple)):
        if len(temperature) != lg.shape[0]:
            raise ValueError(f"temperature: {len(temperature)} values for {lg.shape[0]} slots")
        t = torch.tensor(temperature, dtype=lg.dtype, device=lg.device)[:, None]
        return torch.softmax(lg / t, -1)
    return torch.softmax(lg / temperature, -1)
```

`slot_temperatures` 负责把「逐请求 → 逐项 → 逐槽位」的形状对齐，长度对不上就 `ValueError`
（`temperature.py:147-163`）；`score_shared` 与 `score_items` 都在**任何前向之前**先调它一次，
让形状错误在花钱之前失败（`shared_prefix.py:203`、`engine_v2.py:158`）。
设备上也有一处刻意的写法：温度张量**在目标设备上构造**，注释说明原因是 MPS 上
「从临时 host 张量异步拷贝会读到已释放内存」（`temperature.py:177-178`，issue #21）。

GGUF 路径有一处不同：标签列可能**不足 255**，于是先补 `-inf` 再掩码
（`engine_gguf.py:142-145`）——补的那部分既不在 `nopts` 内、值也是 `-inf`，
所以是恒等的冗余。

**§14 会验一件事**：这些掩码 + softmax 的结果在 committed pins 里逐行和为 1。

---

## 6. 顺序偏置：训练时打乱，运行期可选「反转重读平均」

这是 decider 与 RYOTIDE 差别最大的环节之一，因为它**训练**：

- **训练期消解**：每次渲染都 `rng.shuffle(opts)`，三个入口各一次——
  state-first `prompt.py:140`、schema-first 经 `_select` `prompt.py:65`、chat 布局 `prompt.py:282`；
  训练入口把同一个 `rng` 传进去（`train.py:16-17`）。所以模型见过的每个选项在**任何位置**都出现过。
- **评测期固定**：`evaluate.py:21` 用 `random.Random(1234)`，同一 `rng` 复用——评测可复现，
  但因此**评测里的顺序不是随机的**，引用评测数字时要记得这一点。
- **运行期保序**：库路径传 `_NoShuffle`（`infer.py:47-49`、`infer.py:252`），
  即「按调用方给的顺序」；`prompt.build` 的采样分支在 `system_one` 路径上**不可达**，
  因为 `render_question` 已经把 choice 截到 255（`prompt_fast.py:55-56` 注释明说）。
- **可选的反转重读**（`serve_vllm.py`）：第一次读完后，**最大概率低于阈值**的那些题，
  把选项**整个反转**再读一次，两次的 log-softmax 取平均，再按 `T(n) = max(min, a + b ln n)` softmax
  （`serve_vllm.py:139-159`）：

```python
sel = [i for i, p in enumerate(probs) if max(p) < below]                       # serve_vllm.py:143
first = _lsm([T1 * math.log(max(p, 1e-30)) for p in probs[i]])                 # 由概率还原 log-softmax
n = len(first); T2 = max(lo, a + b * math.log(max(n, 2)))
out[i] = softmax_T([(x + y) / 2 for x, y in zip(first, _lsm(t.result()[::-1]))], T2)   # :156-157
```

注意平均发生在 **log 域**（几何平均），不是概率域；而且它**只对低于阈值的题**做，
所以同一个响应里可能混着「读过两次」与「读过一次」的答案。配置来自
`decider_config.json` 的 `second_reading: {below, temperature_by_options}`，默认关，
且必须同时给 `temperature_by_options` 的 a、b，否则加载就报错（`serve_vllm.py:119-122`）。
`README:57-61` 把它列为 1.9.0 的新特性（配 `decider-31b`）。

**没有的东西**：仓里**没有任何对顺序偏置的度量**——没有 SemIf-OpenJev 那种
「反转顺序翻转 N/M 的 argmax」的 committed 报告，也没有 OpenJev(HF) 那种「翻转率 18.5% → 2.3%」的对比。
`probes/independence.py` 量的是**题目顺序**（`packed` / `reversed` / `independent` 三种问法下答案怎么动），
不是选项顺序。所以「这个实现把选项顺序偏置压到了多少」在本文里**未核实**——仓里的记录只到
「训练时打乱 + 评测里固定 seed + 运行期可选反转平均」这三件事。

---

## 7. 多问题共享 state：独立行 + 有界分叉

`system_one` 的 `independent` 参数决定多问题是「一行全装」还是「一题一行」（`infer.py:213-216`）：

```python
"""independent=True scores every question in its own row (state + that question only), so adding, removing or
reordering questions cannot change any other answer; the state is run once and its cache forked to every
question (Engine.score_shared).  independent=False packs all questions behind one copy of the state in one row
(later questions can then see earlier question texts)."""
```

`_system_one_items`（`infer.py:243-257`）把这个选择落成 `rows = [[r] for r in flat] if independent else [flat]`。
默认是 `True`（`infer.py:208`），`probes/independence.py` 是这条声明的探针：
它跑三种问法（packed / reversed / independent），报告 `dp_order_*`（同一批题换顺序答案动多少）
与 `flips_pack_vs_ind`（打包 vs 独立翻转多少）。**该探针的输出不在仓里**（要权重才能跑），
所以本文只记录「它有这个探针」，不给数值。

### 7.1 前缀只跑一次，缓存按内存预算分块复制

`independent=True` 时所有行共享同一段 state，于是 `Engine.score_shared` / `EngineV2.score_shared`
把公共前缀跑一次，再把它复制给各行（`shared_prefix.py:186-235`）。预算控制是这里的工程核心：

```python
DEFAULT_FORK_GB = 8.0                        # shared_prefix.py:45
DEFAULT_SUFFIX_TOKENS = 65536                # shared_prefix.py:46
STATE_NAMES = ("keys", "values", "indexer_keys", "conv_states", "recurrent_states")    # :47
```

- **前缀复制按 GB 预算分块**：`fork_budget_bytes` 取 `DECIDER_SHARED_FORK_GB`（默认 8 GB），
  并以「设备当前空闲的一半」为上限；`chunk_rows` 算出每块几行（`shared_prefix.py:114-137`）。
  模块 docstring 记了改前的代价：*a 31k-token state with 32 questions costs n times the prefix cache,
  which is 133 GB on a 31B model*（`shared_prefix.py:5-7`）。
- **后缀也按 token 预算分块**（1.8.1）：一块的行数 × 最长后缀 ≤ `DECIDER_SHARED_SUFFIX_TOKENS`
  （默认 65,536 填充 token），理由是**注意力掩码本身** `b × Ts × (prefix + Ts)` 会吃掉几十 GB
  （`shared_prefix.py:12-18`，`README:63-65` 记了这次修复：32 篇 ~26k 文档的请求从 OOM 降到 68.5 GB 峰值）。
- **分块会移动数值**，源码自己写明：「答案可能按 bf16 归约顺序的量移动」，
  但数学上不依赖它（右填充 + 因果层）（`shared_prefix.py:16-18`）。
- **分叉必须给每行自己的存储**：`fork_cache` 逐状态张量做 `index_select`，
  注释说明原因是线性注意力层用 `copy_()` 更新状态，共享存储会写花前缀（`shared_prefix.py:24-26`、`:157-177`）。

### 7.2 混合架构的 cache：枚举已知状态名，未知布局降级但不静默

Qwen3.5 的 cache 是**混合**的：24 层里既有带 `keys`/`values` 的注意力层，也有带
`conv_states`/`recurrent_states` 的 delta-net（线性注意力）层（`shared_prefix.py:20-26`，
`schema_engine.py:4-8` 给出 6 + 18 的分布）。decider 不假设布局，而是**枚举已知状态名**
（`cache_state_tensors`，`shared_prefix.py:70-87`），并处理三类边界：

| 情形 | 处理 | 行号 |
|---|---|---|
| 层上有 `STATE_NAMES` 之外的**带 batch 维**张量 | **不切块**：一次复制全部 n 行（1.1.0 的行为，正确但内存无界），并计数 `shared_unchunked_layout` | `shared_prefix.py:28-30`、`:95-111`、`:207-210` |
| Gemma-4 滑窗层的 `_sliding_window_tensor`（0 维，无 batch 维） | 1.8.1 起识别为「每行共享的常数」，不再触发上面的降级 | `shared_prefix.py:99-101` |
| 前缀长度 < `min_prefix`（默认 192）或只有一行 | 直接返回 `None`，调用方退回 `score_items` | `shared_prefix.py:198-202` |

这是一条**「降级但计数」**的路线：不像 RYOTIDE 的 `trim_to()` 那样直接报错，
也不像 SemIf-OpenJev 那样在缓存类型不对时报错（`shared_prefix.py` 里没有对应断言），
而是退到无界但正确的路径，并把次数留在 `engine.stats["shared_unchunked_layout"]` 里。

计费口径与 RYOTIDE §7 同类：`unique_tokens` 用 LCP 把公共前缀只算一次
（`systemone.py:192-199`；`prompt_fast.py:80-90` 是不重走整段 context 的快版），
`serve.py` 的 schema 分支另报 `cached_tokens`（`serve.py:525`）。

---

## 8. 两份权重：自训 checkpoint 与 stock chat 模型

这是 decider 相对前九篇**新增的一个维度**：同一套读法，两种权重来源。

**自训**：`train.py` 的损失是**掩码后选项 logits 上的交叉熵**，不是语言建模
（`train.py:42-51`）：

```python
def loss_fn(logits, golds, nopts, brier_w=0.0, label_smooth=0.0):
    ok = golds >= 0
    logits, golds, nopts = logits[ok], golds[ok], nopts[ok]
    ce = F.cross_entropy(logits, golds, label_smoothing=label_smooth)
    loss = ce
    if brier_w > 0: ...           # 可选 Brier 项，默认关（train.py:66）
    return loss, ce.detach()
```

`golds < 0`（gold 被丢弃的题）被剔除；非法选项已经在 logits 里是 `-inf`。
优化器对**全部参数**建（`train.py:95`），配方常数在 `scripts/train.sh:10-12`：
`--epochs 1 --lr 1e-5 --warmup 150 --max_tokens 16384 --accum 2 --max_options 255 --max_ctx 16384
--none_prob 0.1 --schema_first_prob 0.5`。数据侧 `decider/data/mixture.py:36-40` 的 `MIX` 定了各组件规模，
另有 6 个 teacher 域完全留出（`HELD_DOMAINS = set(DOMAINS[-6:])`，`:38`）
与两个「任何域都没训过」的规则族（`data/rules.py:69`：`HELD_FAMILIES = ["compare", "count"]`）。

**stock chat**：不训练，靠**布局开关**把同一个读法套到没有 `decider_config.json` 的 checkpoint 上。
`resolve_layout` / `with_layout`（`prompt.py:175-202`）是两个入口，
`DECIDER_LAYOUT=chat` 让服务端这么读；`serve_vllm.py` 就是为它存在的
（`README:281-295`：*for large checkpoints such as a stock instruct model read in the chat layout*）。
发布的两个 `decider-chat-*` 就是「stock 权重 + 一个 config」：

| 模型 | base | 它带的东西 | 文档数值 |
|---|---|---|---|
| `decider-chat-gemma4-31b` | Gemma-4-31B-it，权重未改 | `temperature_by_options: {"a": 10.124, "b": -1.633, "min": 0.05}` → `T(n) = max(0.05, 10.124 − 1.633 ln n)` | Decision Index v0.2.1 **57.33，第 2/70**，ECE 0.047（`README:169`、`:130`） |
| `decider-chat-qwen3.6-27b` | Qwen3.6-27B，权重未改 | 单一温度 **1.943** | Decision Index 51.35，第 8/70，ECE 0.021（`README:170`、`:132`） |

`README:137-139` 自己点出了这件事的含义：*The index lists them as inference techniques.
On the index the base model sets most of the score: our trained 35B, 4B and 2B are below these two stock models
read the same way.* 也就是说，在这条读法上**基座的知识量比读法的打磨更决定分数**。
`temperature_by_options` 这条曲线只在 state-first 布局与 state-first 答案上生效，
schema 缓存上不适用（`temperature.py:8-11`）。

---

## 9. 服务化 / fail-loud 风格

### 9.1 三个服务器，各管一段

| 服务器 | 引擎 | 端点 | 说明 |
|---|---|---|---|
| `serve.py` | `EngineV2`（捕获整张网格后 `seal()`） | `/decide`、`/v1/systemone`、`/v1/models`、`/health`、`/stats` | 主服务器：连续批处理 + schema 缓存 + shared-prefix 路径 |
| `serve_v1.py` | `Engine` | 同上 | 1.0.x 的旧服务器，docstring 明说 **no request-size or queue limits**（`serve_v1.py:8`） |
| `serve_vllm.py` | vLLM 0.29.0（`vllm_worker.py` 装 worker 扩展） | **只有** `/v1/systemone`（independent）、`/v1/models`、`/health`、`/stats` | `/decide` 与 schema 缓存都**不在**这里；`independent: false` 被 422 拒（`serve_vllm.py:369-370`） |

`/health` 会**回声温度**（`serve.py:554-559`）：`temperature` 与
`temperature_by_type: TT.effective(...)`，schema 打开时再加
`temperature_schema_first_by_type`——这正是 comparison.md §5 第 1 条要确认的东西，从接口就能拿到。

### 9.2 三个状态码，各有一条明确判据

```python
MAX_ROWS = 1024；MAX_ROW_TOKENS = MAX_STATE_TOKENS + 4096；MAX_REQUEST_TOKENS = 1 << 20；MAX_QUEUE_ROWS = 4096   # serve.py:84-88
```

`check_size`（`serve.py:170-186`）按顺序判「行数 / 单行 token / 总 token」，
触发即 `rejected_too_large += 1` 并抛 **413**，错误消息里**带上环境变量名**便于定位。
`_admit`（`serve.py:188-194`）在排队行数超限时抛 **503** 并同样计数。
参数非法（`render_question` / `_check_schema` / `resolve_layout` 的 `ValueError`）在请求入口处
转成 **422**（`serve.py:488-490`、`:511-513`、`:528-530`）。`serve_vllm` 多两个：
单行 token ≥ `MAX_MODEL_LEN` 时 413（`serve_vllm.py:323-324`）、客户端断连时 **499**（`:388`）。

值得注意的是 `/decide` 的一个短路：空 schema **不进任何前向**就返回 `{}`（`serve.py:491-492`）。

### 9.3 fail-loud 的边界：五处「不报错但你会想它报错」

| 位置 | 行为 | 行号 |
|---|---|---|
| `neutralize_options` | 把 `none of the above` 等**改写成** `not listed here` 再喂模型，答案里映射回去 | `infer.py:33-44`（+ `cfg["neutralize_none"]` 开关，`infer.py:122`） |
| GGUF `n_seq_max > 1` | 打包多行进一次 decode 会**移动概率**（源码记：BF16 最多 0.02、Q4_K_M 最多 **0.16**），所以**默认 1** 且每次 decode 前清空 memory | `engine_gguf.py:9-12`、`:98` |
| 未知 cache 布局 | 不报错，退到**一次复制全部行**，只计数 | `shared_prefix.py:207-210` |
| `abstain_below` | `p_max` 低于阈值时把 `choice` 报成 `None`，而 `probs` 照给 | `infer.py:177` |
| schema 缓存淘汰 | 超过 **128** 个 schema 时丢最旧的（含它的 graph），**不通知调用方** | `serve.py:306-308` |

前两条是「会移动数值但默认关」；第三条是「降级但计数」；第四条是「改了 `choice` 不改 `probs`」；
第五条是「静默 LRU」。另外 `serve.py` 的 cuDNN SDPA 关闭（`engine.py:66-74`）记的是一个**真错**：
在 Blackwell + torch 2.14 上，掩码矩形注意力的 cuDNN 后端会返回**错误的有限值**，
源码给了对照观测（*the cached path answered a wrong option at p=0.93 where the full forward …
give option_4 at p=0.95*），并强调必须在 `torch.compile` 与图捕获**之前**设置。

---

## 10. 置信度从哪来，又怎么被改写

### 10.1 一条链

```text
① 训练本身（train.py:95，全参数；CE over 掩码选项 logits）   → 改数值、也可能改 argmax
② 逐题型温度（decider_config.json → temperature.py:89-113）   → 只改数值（softmax 单调）
③ 报告的公式（systemone.py:155-173，TypeSafe 的两个式子）      → 只改"报出去的那个数"，不改分布
④ serve_vllm 的 second_reading（反转重读、log 域平均）         → 改数值，且**可以**改 argmax（阈值下才触发）
⑤ calibrate.py / report.py 的拟合                              → 产出 ②，**不自动写回**
```

### 10.2 温度：逐题型常量，且分「state-first / schema 两套」

`decider_config.json` 的键在 `temperature.py:1-28` 的模块 docstring 里列全：`temperature`、
`temperature_by_type`、`temperature_by_options`、`temperature_schema_first`、
`temperature_schema_first_by_type`。解析规则（`from_config`，`temperature.py:89-113`）有三处容易踩：

1. **schema 缓存有自己的温度**；不给 `temperature_schema_first` 时回落到 state-first 的
   （`temperature.py:109-112`）；
2. **显式覆盖会关掉 by-type map**：`Decider(temperature=…)` 或 `DECIDER_TEMPERATURE` 一给，
   `temperature_by_type` 就不生效（`temperature.py:96-98`，docstring `:22-24`）；
3. 没有 map 时**传进去的就是那个标量本身**，概率与 1.3.0 逐位相同（`temperature.py:26-27`、
   `for_items`，`:137-144`）。

**发布值**在文档里，不在代码里：`docs/RESULTS.md:184-185` 记 decider-4b v2.1 是
`temperature` 1.099 + by-type `choice 1.110 / noul 1.560 / score 1.287`，decider-2b v11 是
1.145 + `1.164 / 1.624 / 1.124`。而**代码与脚本里的默认值全都不同**：

| 来源 | 默认 T | 行号 |
|---|---|---|
| `temperature.py` 无配置时 | **1.0** | `temperature.py:100` |
| `scripts/evaluate.sh` 给没有 config 的 checkpoint 生成的 | **1.15**（并设 `schema_first: true`、`isolated_levels: true`、`neutralize_none: false`） | `scripts/evaluate.sh:7` |
| `serve.py` 的模块级默认 | `TEMP = 1.0`、`ISOLATED = False`、`NEUTRALIZE_NONE = True`、`SCHEMA_FIRST = False` | `serve.py:90-91` |

所以「照 `evaluate.sh` 生成的 config 跑」与「照服务默认值跑」是两套数——
这与 comparison.md §5 第 6 条（代码默认值 ≠ 发布配方）是同一类问题，只是发生在这里的
是**评测脚本的默认**而不是发布配方本身。

### 10.3 两个同名 `confidence`，以及第三个叫 `certainty` 的数

`systemone.format_answer`（`systemone.py:176-189`）按 TypeSafe 的定义报：

```python
def choice_confidence(p):        # (n * p_max - 1) / (n - 1)：均匀 → 0，全质量在一项 → 1        systemone.py:155-159
def score_confidence(p):         # 1 - Σ pᵢ·|i-k| / D，D = 各等级到刻度中点的平均距离            systemone.py:162-173
def certainty(p):                # 1 - 归一化熵（与上面两个都不是一回事）                      systemone.py:139-142
```

一个 choice 答案同时带 `confidence`（TypeSafe 式）、`x_p_max`（`p_max`）、`certainty`（归一化熵）
——`README:257-262` 解释了三者的关系，并说明 `x_p_max` 是 1.3.0 之前 `confidence` 的含义。

但 **`decide()` / `decide_json()` 报的 `confidence` 是 `p_max`，不是 TypeSafe 式**：

```python
res.append(dict(choice=..., confidence=p[j], probs={...}, probs_list=p))     # infer.py:177  ← p[j] 即 p_max
...
o[qtext] = {"choice": ..., "confidence": round(r["confidence"], 4), ...}    # infer.py:333  choice 沿用
o[qtext] = {"score": ..., "confidence": round(p[j], 4), "type": "scale"}    # infer.py:340  scale 用 p_max
```

同一份代码库里，`/v1/systemone` 的 `confidence` 与 `decide()` 的 `confidence` 是**两个公式**，
而且 `decide()` 那侧**没有 `x_p_max` 可对照**。二者都有 committed 断言钉住：
`tests/test_serve_http.py:193` 断言 systemone 的 `confidence == (n*x_p_max-1)/(n-1)`，
而 §14 会在 pins 上验证 `decide_json` 的 `confidence == p_max`。
这与 comparison.md §5 第 7 条（同名 `confidence` 是不同公式）同型，只是这里的两个公式在**同一个包里**。

### 10.4 温度怎么被拟合、又怎么被采纳

`calibrate.py` 只认一种输入：**温度 1 下读出的答案**（`calibrate.py:6-10`）：

```text
{"type": "choice"|"noul"|"score", "logits": [...], "gold": i}
{"type": "score", "level_logits": [[no, yes], ...], "gold": i}        # isolated levels
```

拟合目标是对 gold 的 **NLL**（`calibrate.py:50-52`、`:121-142`），先在一张
**0.05 到 20 的 801 点对数网格**上找最小值，再用黄金分割在相邻区间细化
（`GRID`，`calibrate.py:27`；`fit`，`:145-162`）。`fit_by_type` 按题型各拟一个，
行数少于 `min_rows`（默认 50）的题型跳过（`:165-179`）。

**关键的一步在哪儿断开**：`calibrate.py` **只 print JSON**，从不写文件
（`calibrate.py:216-218`）；写回 `decider_config.json` 是 `scripts/evaluate.sh:13` 的注释所规定的人工步骤
（*fits the temperature on in-task data; put it into decider_config.json*），
而 `report.py:20` 的 `fit_temperature` 用的是另一张更窄的网格（0.3–5、120 点）。
发布前只有一句存在性断言（`scripts/stage_release.py:16`），**没有**「温度是否已采纳/是否有留出判据」的检查。

**没有的东西**：仓里没有针对温度的接受/拒绝判据（不像 RYOTIDE 的预设判据），
也没有 SemIf-OpenJev 那种折外 ECE + bootstrap 区间；`metrics.py` 提供 ECE 等度量，
但**是否因 ECE 变差而拒绝某个 T**，代码里没有这一步。

### 10.5 校准怎么被测量

`metrics.py` 的 ECE 是**等宽 15 桶、左开右闭、未重归一化**（`metrics.py:4-11`）：

```python
def ece(conf, correct, bins=15):
    edges = np.linspace(0, 1, bins + 1)
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any(): e += m.mean() * abs(conf[m].mean() - correct[m].mean())
```

`bins=15` 与 Open-Jev、Laya 同口径，与 RYOTIDE / SemIf-OpenJev 的 10 箱**不同**
（comparison.md §5 第 2 条）。`summarize`（`metrics.py:27-37`）另外给
`acc / nll / brier / aurc / acc@80 / acc@50 / chance / mean_conf`；
`aurc`（`:14-18`）是风险-覆盖曲线下面积。`evaluate.py:45-62` 按 `D.TASKS[...]["heldout"]`
把任务拆成 in-task / held-out 两组分别聚合——这就是 `README` 里那一堆
「in-task 0.805 / held-out 0.755」数字的来源。

文档侧给出的**带 run 标签与温度**的数值（皆为文档，非代码）：`docs/RESULTS.md:151-162`
的 94-task 表逐行标了 T（`T=1.30`、`T=1.36`、`T=1.18`、`T=1.03`…）；
`docs/RESULTS.md:104-105` 说明 v2 的 **1.935** 是「在 67 个 in-task 任务里的 61 个上拟合」；
`docs/CHANGELOG.md:9-10` 记了 31b 的 **64.27 / ECE_bw 0.039**（*our run*）与 B300 上
单请求 **median 39 ms**；`docs/CHANGELOG.md:63` 记 JevBench public hard **0.712** 与 top-label ECE **0.147**。
引用这些数字时，run 标签与温度必须一起带上——它们各自的 T 不同（§13 的复现命令里会写明哪一种）。

---

## 11. 四条不变量的落位

RYOTIDE §13 的四条在这条路线上的落位**有三条要改写**，其中 #1 是**概念不适用**：

| RYOTIDE §13 不变量 | 在 decider 里的状态 |
|---|---|
| **#1** 读位置必须落在标记上，诊断量 `marker_mass` | **概念不适用**：`marker_mass` 量的是「全词表质量有多少落在标记上」，而 decider **从不物化词表维度**——`F.linear(hs, W)` 里 `W` 就是 `lm_head` 的 255 个标签行（`model.py:28`），标签外的质量在数学上不存在，`marker_mass ≡ 1` 恒成立、无信息。**改写后的等价命题**应变成：「`lm_head` 的哪 255 行是读出头」——这由分词器决定，而它**只有加载期的 `assert`**（`prompt.letter_ids`，`prompt.py:292-296`，逐一验证 `A`–`J` 各是单 token）加上 `label_table` 的 255 唯一性断言（`prompt.py:52`）。**没有**任何运行期诊断量能告诉你「读错了行」 |
| **#2** 前缀与标记表面形式耦合，必须成对扫 | **改写为「断言掉这个自由度」**：宽渲染按 **token id** 拼标签（`prompt.py:73-74`、`:146-151`），所以不存在「表面形式」可选；chat 布局再用两条 `assert` 钉死「标签在 `Answer: (` 后单 token」与「模板尾巴不与答案前缀合并」（`prompt.py:229-233`）。这与 SemIf-OpenJev 的 `direct.py:42` 是同一种做法（断言而非扫描） |
| **#3** 只读最后一个位置 | **改写**：读的是**构造出来的 K 个槽位**（每题一个），它们**不是**末位；同一行里 K 个槽位在一次前向里全读到（`model.py:23-32`）。「不物化词表维度」这半条**成立**且比 RYOTIDE 更彻底 |
| **#4** 比较概率 / ECE 前先确认 temperature | **成立，而且这里要确认的是三件事**：① 哪一层温度（state-first 还是 schema 缓存，`temperature.py:109-112`）；② 哪一型（by-type map 是否存在，且显式覆盖会关掉它，`:96-98`）；③ 哪条 API 的 `confidence`（`decide` 是 `p_max`，`system_one` 是 TypeSafe 式，§10.3）。外加 ECE 是 **15 等宽箱**（`metrics.py:4`） |
| （decider 新增的） | **标签表与分词器绑定**：换分词器就要重验 255 个单 token 标签，仓里用加载期断言而非配置记录来保证（`prompt.py:52`、`:292-296`）；**标签数上限是分词器的上限**（255 来自「能单 token 化的双字母有多少」），不是接口约定；**`output_tokens = 0` 是响应组装处的字面量**（`serve.py:525`、`:546`），而唯一的例外 `serve_vllm` 会向 vLLM 发 `max_tokens=1` 却不消费它（`serve_vllm.py:190-195`，§4.4） |

---

## 12. 文件对照表

| 文件 | 行数 | 职责 |
|---|---|---|
| `decider/prompt.py` | 301 | 两条文本布局 + chat 布局、`label_table`（255 单 token 标签）、答案槽位、`resolve_layout` |
| `decider/prompt_fast.py` | 90 | `system_one` 路径的行构造：state 只 tokenize 一次 |
| `decider/model.py` | 70 | `DecisionModel`：backbone → 槽位 hidden state → 标签行 logits；`cap_logits`（Gemma softcap） |
| `decider/systemone.py` | 199 | 三题型 → prompt 行；`plan_rows`（isolated 展开）；`assemble`；两个 TypeSafe 置信度公式 + `certainty` |
| `decider/infer.py` | 385 | `Decider`：`decide` / `decide_json` / `system_one` / `schema()`；设备与 dtype 选择 |
| `decider/engine.py` | 216 | `Engine`：CUDA graph 分桶、`score_items`、`score_shared`；cuDNN SDPA 策略 |
| `decider/engine_v2.py` | 213 | `EngineV2`（服务用）：整张网格预捕获 + `seal()`、长度阶梯到 8192 |
| `decider/engine_gguf.py` | 153 | `GGUFEngine`：llama.cpp 上的同一读数 |
| `decider/schema_engine.py` | 152 | schema 缓存：前缀缓存（K/V + conv/递推态）+ 每形状一张图 |
| `decider/shared_prefix.py` | 235 | 共享前缀的有界分叉：GB 预算切块、后缀 token 预算、未知布局降级计数 |
| `decider/batching.py` | 95 | 连续批处理的**代价模型 + DP 分组**（`plan_batches`，overhead 512） |
| `decider/temperature.py` | 180 | 五个温度键的解析与校验；`scaled_softmax`；逐槽位温度 |
| `decider/calibrate.py` | 222 | 由温度 1 的记录拟 by-type 温度（NLL 网格 + 黄金分割）；**只 print** |
| `decider/report.py` | 76 | 另一条拟合路径（标量 T，网格 0.3–5） |
| `decider/train.py` | 142 | 掩码选项 CE 微调；warmup + cosine；梯度裁剪 |
| `decider/metrics.py` | 45 | `ece`（15 等宽箱）、`aurc`、`acc@coverage`、`summarize` |
| `decider/evaluate.py` | 105 | 评测主循环；in-task / held-out 聚合 |
| `decider/serve.py` | 566 | 主服务器：`/decide`、`/v1/systemone`、连续批处理、schema 缓存、shared 路径、413/503 |
| `decider/serve_vllm.py` | 423 | vLLM 服务器：只答 independent `/v1/systemone`；`second_reading`、prefill-first、LCP |
| `decider/serve_v1.py` | 307 | 1.0.x 旧服务器（无限制） |
| `decider/batching.py` + `serve.py:243-273` | — | `adaptive_ms`：一次收集里有多于一个未完成请求时才多等 2 ms |
| `decider/data/mixture.py` | 227 | 训练混合的组成与规模（`MIX`）+ 4 个 teacher 数据源 |
| `decider/data/rules.py` | 398 | 规则读取任务生成：12 个族、3 个留出域、2 个留出族 |
| `decider/vision/model.py` | 84 | 图像分支：**扫描**找槽位，多模态 backbone，同一套标签行读数 |
| `decider/probes/independence.py` | 89 | 「答案是否依赖同批其他问题」的探针（packed / reversed / independent） |
| `decider/probes/isolated.py` | 105 | listwise vs 逐级隔离的对照探针 |
| `decider/data/core.py` | 696 | 任务注册表（`@task(name, heldout=...)`）与数据缓存 |
| `tests/temperature_cases.py` | 133 | **committed pins 的生成器**：确定性替身骨干 + 字节分词器 |
| `tests/data/temperature_1_3_0_pins.json` | — | 1.3.0 的存盘概率（4 个配置 × 21 条路径）；§14 在它上面重算 |
| `tests/data/plain_layout_pins.json` | — | 1.1.3 的 359 条 token id sha256 摘要 |
| `scripts/train.sh` / `evaluate.sh` / `stage_release.py` | — | 训练配方常数 / 评测默认 config（T=1.15）/ 发布目录装配（断言 config 存在） |
| `moe/` | — | 35B-A3B 的冻结专家 Muon 训练与 NVFP4 量化 |

---

## 13. 亲手跑一遍

以下命令**取自仓库自带的脚本与 docstring**。本文**只执行了 §14 的两条**（标准库重算），
其余**一条都没有执行**（没有 GPU、没有下权重）。

```bash
# 0) 安装（本文未执行）
pip install decider-ai                                  # 或: git clone https://github.com/Mapika/decider && pip install -e ".[serve]"

# 1) 库入口：一次前向、K 个问题（README:213-230 的同一个例子；本文未执行）
python -c "
from decider.infer import Decider
d = Decider('Mapika/decider-2b')
print(d.decide('My card was charged twice.', [{'question': 'Which team?', 'options': ['billing','technical','sales']}]))
print(d.system_one({'ticket': {'messages': [{'from':'customer','text':'I was charged twice.'}]}},
                   {'department': {'type':'choice','instructions':'Which team should handle this?','criteria': {'billing':'Charges, invoices','technical':None,'other':None}},
                    'frustration': {'type':'score','instructions':'How frustrated?','criteria':['calm','frustrated','very frustrated']}}))"

# 2) HTTP 服务（README:269-271；本文未执行）
scripts/serve.sh Mapika/decider-2b 8000        # 默认只监听 127.0.0.1；DECIDER_HOST=0.0.0.0 才对外开放（无鉴权）
curl -s localhost:8000/health                  # 回声每型温度（§9.1）

# 3) stock 大模型走 vLLM（README:286-292；本文未执行）
DECIDER_MODEL=Qwen/Qwen3.6-27B DECIDER_LAYOUT=chat DECIDER_TEMPERATURE=1.943 \
    uvicorn decider.serve_vllm:app --host 127.0.0.1 --port 8000

# 4) 重新拟合逐题型温度（calibrate.py:214-218；只 print，不落盘）
python -m decider.calibrate records.jsonl        # records 格式见 calibrate.py:6-10

# 5) 训练与评测（scripts/train.sh；本文未执行）
scripts/train.sh full                            # → data/mixture_full.pkl → 1 epoch → scripts/evaluate.sh
python -m pytest -q tests                        # 本文未执行；CI 里也跑（.github/workflows/tests.yml）
```

**本文执行过的两条**（只读 committed 文件、只用标准库，§14 会贴出）：

```bash
# A) 在 committed 的 1.3.0 pins 上重算温度律、掩码归一与 argmax 不变性
python3 - <<'PY'
import json, math
C = json.load(open("tests/data/temperature_1_3_0_pins.json"))["cases"]
A, B = C["default"]["decide_batch/eager"], C["t1.935"]["decide_batch/eager"]
PY
# B) 在 committed 的 issue 记录与 README 示例上重算 TypeSafe 置信度
```

这**不加载任何权重、不 import 上游包**——与 CLM 那篇 §13.2 的做法同类。

---

## 14. 实例走查：在 committed 产物上重算

`tests/data/temperature_1_3_0_pins.json` 的 `generated_with` 写明它是怎么来的：

```json
"generated_with": "decider-ai 1.3.0 (origin/main 5f91c01), tests/temperature_cases.py, CPU"
```

而 `tests/temperature_cases.py:17-51` 显示它的「模型」是**确定性替身**：
16 维骨干、种子 0 与 1 初始化的 embedding 与 head、字节分词器；
`tests/temperature_cases.py:42` 给激活乘了 4，注释是 *confident enough that T matters*。
所以这些数字**能证明读数链的内部一致，不能证明模型质量**——本文只做前者。

用例是固定的一段 state 与两组请求（`tests/temperature_cases.py:79-89`）：

```python
STATE = "My card was charged twice and checkout is down."
REQUESTS = [(STATE, [{"question": "Which queue?", "options": ["billing","technical","sales"]},
                     {"question": "Urgent?",    "options": ["no","yes"]}]),
            ("hey turn the lights off", [{"question": "Intent?", "options": ["lights","alarm","music","none of the above"]}])]
```

四个配置分别是 `default {}`、`t1.05_schema1.18`、`t1.3_iso {isolated_levels: true, temperature: 1.3}`、
`t1.935 {temperature: 1.935}`。

### 14.1 温度就是 `logits / T`（三项核对）

```text
== 每行是否归一（掩码后 softmax 的必然结果）==
  default item0 row0: sum=0.9999999665  n=3
  default item0 row1: sum=1.0000000447  n=2
  default item1 row0: sum=1.0000000503  n=4
== 同一位移下 argmax 是否随 T 变 ==
  item0 row0: argmax default=1  t1.935=1  t1.3iso=1
  item0 row1: argmax default=1  t1.935=1  t1.3iso=1
  item1 row0: argmax default=3  t1.935=3  t1.3iso=3
== p(T) ∝ p(1)**(1/T) 是否成立（用 T 还原 logits）==
  item0 row0: max|pred - pinned(t1.935)| = 2.060e-08
  item0 row1: max|pred - pinned(t1.935)| = 7.692e-09
  item1 row0: max|pred - pinned(t1.935)| = 2.019e-08
```

三件事同时成立，且都是**从存盘数字反推的**：

1. **每一行的概率和是 1**（误差 ≤ 5e-8，float32 存 float64 的量级）——
   这是 §5 的「`-inf` 掩码 + softmax」在数学上的必然结果，掩码外的质量不可能漏出来；
2. **argmax 在四个温度下完全不变**（4 条行、0 次翻转）——这是「温度只改数值不改答案」的实证，
   与 Jeff §9.1 的 100000 组抽样、SemIf-OpenJev §12 的 0/144 是同一类证据；
3. **把 T=1 的概率取对数当 logits，除以 1.935 再 softmax，能逐位复现已存盘的 `t1.935` 那一列**
   （最大偏差 ~2e-8）。这就直接验证了 `scaled_softmax(lg, T) = softmax(lg / T)`（`temperature.py:171-180`）
   ——温度是**除以 logits**，不是对概率做幂次，也不是别的参数化。

### 14.2 同一条答案上两个 `confidence` 公式（pins 里就能看出来）

```text
  [default] Which team?: confidence=0.7597 p_max=0.7597 | TypeSafe(n=3)=0.6396  -> == p_max
  [default] Mood?:       confidence=0.7764 p_max=0.7764 | TypeSafe score_confidence=0.6646 -> == p_max
  [t1.935]  Which team?: confidence=0.5736 p_max=0.5736 | TypeSafe(n=3)=0.3604  -> == p_max
  [t1.935]  Mood?:       confidence=0.5882 p_max=0.5882 | TypeSafe score_confidence=0.3823 -> == p_max
```

`decide_json` 路径存下来的 `confidence` **恰好等于 `p_max`**，而 TypeSafe 的两个式子会给出
明显不同的数（0.6396 / 0.6646）。这与 `tests/test_serve_http.py:193` 对 systemone 的断言
（`confidence == (n*x_p_max-1)/(n-1)`）**同时成立**——因为它们量的是两条不同的 API。
这正是 §10.3 那条分叉的 committed 证据：**同一个字段名，`decide` 与 `system_one` 报两个数**。

### 14.3 反方向：在 committed 的「问题记录」上重算 TypeSafe 式子

`tests/test_systemone.py:104-118` 存了一份 issue #15 的记录（一个 5 选项问题）：

```python
REPORT = {"Red": 0.195, "Green": 0.1994, "Blue": 0.2252, "Yellow": 0.2286, "Purple": 0.1518}   # issue #15
...
assert a["choice"] == "Yellow" and a["x_p_max"] == 0.2286
assert a["confidence"] == round((5 * 0.2286 - 1) / 4, 4) == 0.0358   # the report's reference value, 0.036
```

重算：`sum = 1.0000`、`argmax = Yellow (0.2286)`、`(5×0.2286 − 1)/4 = 0.03575 → 0.0358`。
三项全对。同一文件的 `:74` 也把 score 的式子钉住：
`score_confidence([0.1, 0.5, 0.4]) = 1 − 0.5/(2/3) = 0.25`，注释写成
`# 1 - (0.1 + 0.4) / (2/3)`。也就是说 **TypeSafe 的两个公式在 decider 里是照抄实现的**
（`systemone.py:155-173` 的 docstring 直接点名 *system-one-adapter-python*），
而 §14.2 表明**只有 `system_one` 那条路径用它**。

### 14.4 README 示例的自洽性（引文核对，非实测）

`README.md:223-225` 的示例里，`department` 报 `confidence: 0.34`、`x_p_max: 0.56`，
概率 `{returns: 0.44, billing: 0.56, other: 0.00}`（n=3）：
`(3×0.56 − 1)/2 = 0.34` ✓——**文档示例本身符合代码里的 TypeSafe 式子**。
同一示例里还有两处值得记下的**不一致**：

- `frustration` 报 `score: 0.76`，但显示的概率算出的期望等级是
  `0×0.34 + 1×0.55 + 2×0.10 = 0.75`（差 0.01，落在显示概率的四舍五入范围内，**不构成矛盾**）；
- 同一答案里 `certainty: 0.37`，而按 `certainty()` 对显示的概率算是 **0.3756**
  （`format_answer` 的 `nd=4` 会印 `0.3756`；两位小数则应是 `0.38`）。示例的 `0.37`
  与它自己印出的概率不是同一组数的舍入——大概率是示例被手工截短过
  （该行注释正写着 *v10 weights; "returns" also mentions refunds, so the mass is split*）。
  **这是一处文档与代码的小不符，不是读法问题**，列在这里以免有人拿它当复算基准。

### 14.5 这一节能证明什么、不能证明什么

- **能证明**：温度确实是 `logits / T`（§14.1）；掩码后的分布确实归一；温度确实不改 argmax；
  `decide` / `decide_json` 的 `confidence` 确实是 `p_max` 而非 TypeSafe 式；
  TypeSafe 的两个公式确实是照抄实现且只在 `system_one` 路径用（§14.3）。
- **不能证明**：这些数字来自一个 16 维的替身骨干，**与任何发布的 `decider-*` 权重无关**；
  它不能说明准确率、ECE 或任何模型质量；`t1.3_iso` 这一列里 isolated levels 的作用
  只能从形状（每条 score 展开成 3 行）看出来，**其数值本身也需要跑上游代码才能重算**
  （本文未执行 `tests/temperature_cases.py`）。
- **未核实**：`plain_layout_pins.json` 的 359 条摘要是**分词器相关**的（Qwen3.5-2B 的 tokenizer），
  本机没有该分词器，本文没有复算；`docs/benchmarks/*.json` 那几个 committed 的评测记录
  （如 `mps-heldout.json` 的 `acc 0.7553 / ece 0.0438`）需要模型权重，同理未复算。

---

## 15. 本文不做什么

- 不给任何**前向产生的概率数值**：本文的「实测」只到「在 committed 的 pins/记录上做算术」为止（§14）。
  所有准确率、ECE、延迟数字都标了来源与 run 标签，且都是**文档数值**（`README:行`、`docs/...:行`）。
- 不判断 decider 的读数是否**优于** RYOTIDE：两者的对比只在结构层面（§0、§11），
  质量对比需要同一 harness 下的实测，这里没有。
- 不覆盖 `decider/vision`（图像分支只说到「它靠扫描找槽位」为止）、
  `decider/games`（十个文本游戏 + Mario，属于 RL/数据生成侧）、
  `moe/`（35B 的冻结专家训练与 NVFP4 量化）与 `decider/data/**` 的数据构造细节——
  它们与「一次前向怎么读成类型化决策」这条主轴无关，本文只在 §8 记了训练混合的规模与留出设计。
- 不解释 `docs/RL.md` 那条把 v8 变成 v10 的 RL 阶段：仓库自己写明它依赖一个**不在本包里的**
  独立研究仓与在线 MiniWoB++/游戏环境（`README:331-332`），本文无法核实其实现。

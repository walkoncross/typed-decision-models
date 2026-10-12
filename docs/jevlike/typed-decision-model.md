# Jevlike 是如何让每个选项自己去读一次 state 的

> 代码库：<https://github.com/vinnylarouge/jevlike>
> 对应 README 第 3 行：*Train a small model that chooses among a changing list of text options.*
> 以及 README 第 5 行：*A Jev-like model takes a piece of text and a list of `N` text options. It
> returns one probability for each option. It does this in one pass instead of writing an answer word
> by word.*，与 README 第 31 行（Architecture）：*Each option becomes a query vector … The query
> assigns attention weights to the context tokens. Those weights make one context vector for that
> option. A shared dot product turns each option and context pair into one score. A softmax … runs
> across the options.*
> 本文按 `jevlike/` 与 `examples/` 的实际代码逐环节说明这句话的实现方式；文中出现的文件名与行号
> （`model.py:24`、`data.py:49`、`vision.py:57` 等）均相对该仓库根目录。
>
> **同名配对文档**：RYOTIDE 的 [`docs/ryotide/typed-decision-model.md`](../ryotide/typed-decision-model.md)。
> 本文凡称「RYOTIDE §N」均指该篇的节号；§0 先讲清分歧，§10 把它的四条不变量逐条落位。

与另外几篇最大的不同有四条，前面先说清：

1. **这条链上没有 LLM。** 默认骨干是**从头训的字节嵌入**（`nn.Embedding(257)`，`model.py:49`），
   可选路径是「**冻结的预训练编码器 + 掩码均值池化**」（`model.py:66-94`）。两条路都**不读词表分布**，
   所以 RYOTIDE 那一套「读位置 / `marker_mass` / 掩码到标签集」在这里不是「被绕开」，而是
   **没有对象**——它比 CLM 与 Kev 那两条「词表维度从不出现」的路线还要靠前一步：那两条至少还有
   一个语言模型骨干，这里连骨干都不是 LM。
2. **标签集就是这一行的选项列表。** 训练标签是**行内下标**（`data.py:30-31`），
   `F.cross_entropy(model(batch), batch["labels"])` 直接落在这一行的 N 个标量上（`train.py:74`）。
   于是「标签必须单 token」「字母表上限」「候选数上限」这三类约束**同时消失**——不是被放宽，
   而是没有全局类别空间可以让它们成立。
3. **决策量不是「从 state 里读一个位置」，而是「每个选项自己 query 一次 state」**：
   `logits[n] = q_n · (Σ_l softmax_l(q_n·k_l/√r) · v_l) / √r`（`model.py:34-42`）。
   一次前向里有**两次 softmax**：内层在 state 的位置上（这是「读」），外层在选项上（这是「决策」，
   由调用方施加）。
4. **同一个头能读图像。** `examples/` 里 `DoomScorerV2` 复用**同一个 `AttentionHead`** 读 8×10 个画面
   patch，选项是一张**12 行的共享控制器表**（Doom 0–6、象棋 7–11，`vision.py:15-17`），
   并用一份数据交替训两个游戏（`examples/doom/train_joint.py`）。§11 专门讲这一支。

第 2 条是这一篇与其余各篇最实质的差别：别家在讨论「**怎么把模型的输出逼进一个固定标签集**」，
这一篇的答案是**不要固定标签集**。

## 本文的事实来源

| 类别 | 来源 | 本文中的标记 |
|---|---|---|
| 代码引用 | `jevlike/**`、`examples/**`（上游仓，本机 checkout 在 `../jevlike`） | `文件:行` |
| 可复跑得到的产物 | 三个 committed 检查点的**容器层算术**（`unzip -l` 的逐条长度、pickle 里的键名字符串）与**权重张量的逐行算术**（标准库 `struct` 直接读 `data/N`，**不解 pickle**） | 标注「实测」 |
| 公式与常量 | `README.md`、`examples/*/README.md`、`AGENTS.md`、`pyproject.toml` | `README:行` |
| 未跑 | **本机 `import torch` 失败（Python 3.14.7，无 torch）、无 Stockfish、无 GPU**，本文没有跑过任何上游程序，也不给任何前向产生的概率数值 | 一律不填 |

三个检查点是本仓**唯一**的 committed 数值产物。上游 `runs/` 被 gitignore（`.gitignore:7`），
`examples/*/README.md` 里的准确率与对局结果**没有随仓提交任何原始记录**——
这与 CLM、Kev、decider、Nimble 那几篇「可在 committed 产物上重算」的处境不同，
本文能重算的只有**权重与容器本身**，不能重算任何评测数字。

---

## 0. 与 RYOTIDE 的分野

对照对象是 [`docs/ryotide/typed-decision-model.md`](../ryotide/typed-decision-model.md)。

| | RYOTIDE | Jevlike |
|---|---|---|
| 骨干 | 冻结的因果 LLM（Qwen3.5 / Gemma 4） | **没有 LLM**：`nn.Embedding(257)` 字节表，或冻结编码器 + 一个注意力头 |
| 决策量怎么来 | 读末位**词表 logits**，按标记 id 取出 | 每个选项**一个标量**：`q_n · attended_n / √rank`（`model.py:42`） |
| 「typed」从哪来 | 分词器：标签必须**单 token** | **不需要**：标签是**行内下标**，类别集就是这一行的选项 |
| 读几个位置 | 1 个（末位） | **0 个**——不是「读位置」，是每个选项对全部未掩位置的一次**加权平均** |
| 掩码 | 掩到标签集（词表 → K） | 两种掩码，**都不掩标签**：state 侧掩 padding、选项侧掩 padding（§5） |
| 重归一化 | 必须（把词表其余部分的质量收回） | **自动**：掩在 softmax 之前，padding 位概率恰为 0 |
| 训练 | 无，权重全程 frozen | **有**：tiny 路径训全部参数；hf 路径冻编码器、只训头 |
| 候选数上限 | 260（`A0`–`Z9` 两步读） | **无**：候选集由实例给出，不撞任何表 |
| 顺序偏置 | 多 order 共享前缀 fork，概率平均 | **构造性不存在**（选项之间零交互，严格等变，§6） |
| 置信度 | 读出来的：只有 temperature 能改数值 | **训出来的**，而且**链条上只有训练这一环**：无温度、无标定产物（§9） |
| 服务 | `/v1/systemone` + `/health` + MCP | **没有服务**：四个 console script（§8） |
| 诊断量 | `marker_mass`（全词表落在标记上的份额） | 视觉侧有一套厚诊断；**文本侧没有等价量**（§3、§11） |

一句话：**RYOTIDE 借冻结 LLM 已有的答案位置读出分布；Jevlike 不给 LLM 留位置，它让每个选项自己去
state 里取一次。** 前者的「读」是选一个位置，后者的「读」是学一张权重分布。

共同点只剩最本质的那一条：**一次前向、不生成任何 token、输出封闭在已知选项集上**——
而且是收录的几条路线里**封闭性最便宜**的一条：它连「越界」这件事都没有对象，因为没有词表。

---

## 1. 端到端数据流

两条路（文本 CLI 与视觉决策）共用同一个头，但数据流不同，画在一起：

```
── 文本：jevlike-predict（predict.py:15-35）/ jevlike-eval（eval.py:51-67）/ jevlike-train（train.py:33-91）
   --context "…" --option "…" --option "…"            （predict.py:18-23，少于两个选项直接 parser.error）
        │
        ▼
   ChoiceExample(context, options, label)                        data.py:17-22
   validate()  ① context 是非空 str  ② options ≥2 且每个非空
               ③ label 是 0..len(options)-1 的 int               data.py:24-32
        │
        ▼
   collator = ByteCollator | HuggingFaceCollator                 model.py:110-125（由 config 选）
        ├─ tiny：text.encode("utf-8")[:context_tokens] → 每个 byte +1     data.py:49-50
        └─ hf  ：AutoTokenizer(..., truncation=True,
                        max_length=…, add_special_tokens=True)            data.py:72-79
        │
        ▼
   _tensor_batch → context_ids / context_mask
                   option_ids  / option_token_mask / option_mask       data.py:87-107
        │
        ▼
   model.forward
        ├─ TinyScorer: embedding(257)+position → context (B,L,W)
        │              option_tokens 掩码均值池化 → options (B,N,W)     model.py:53-63
        └─ FrozenTransformerScorer: encoder 两次（state 一行；选项展平后一批）
                                    → 两次掩码均值池化                  model.py:75-94
        │
        ▼
   AttentionHead                                                   model.py:24-43
        LayerNorm(context) / LayerNorm(options)                     model.py:29-30
        q = query(options)  k = key(context)  v = value(context)    model.py:34-36
        scores[n,l] = q_n · k_l / √rank                             model.py:37
        scores.masked_fill(~context_mask, finfo.min)   ← 掩码 ①     model.py:38-40
        attended[n] = Σ_l softmax_l(scores[n]) · v_l                model.py:41
        logits[n]   = q_n · attended[n] / √rank        ← 一个选项一个标量
        logits.masked_fill(~option_mask, finfo.min)    ← 掩码 ②     model.py:42-43
        │
        ▼
   调用方 softmax(-1) → 每个选项一个概率
        predict.py:31（打印 JSON）/ eval.py:24（算 top1/top3/ECE）
        train.py:74（喂 cross_entropy，不自己 softmax）

── 视觉：examples/doom/*.py、examples/chess/*.py
   屏幕 uint8 (120,160,4) → observation_tensor → (B,4,120,160) float/255   vision.py:192-194
        │
        ▼
   DoomScorerV2.encode: 逐样本 RGB 标准化 + motion 通道 → conv stem → patch conv
                        → (B, 80, width) 共 8×10 个 patch                 vision.py:65-76
        │
        ▼
   option_vectors = options.weight[option_ids]  ← 选项是**查表**，不是编码器   vision.py:91-99
        │
        ▼
   AttentionHead（同一个类），但 key = key(context) + positions        vision.py:107
        │
        ▼
   logits (B, len(option_ids)) + value (B,)                            vision.py:114-116
        │
        ▼
   argmax / multinomial（play.py:84-88）/ softmax 取概率
```

---

## 2. 「typed」从哪来：这一行的选项列表就是类别集

这是本篇最该先记住的一条。

### 2.1 标签是行内下标，不是全局 id

`ChoiceExample` 的第三个字段是 `label: int`，`validate()` 只要求它落在 `0..len(options)-1`
（`data.py:30-31`）；训练时它直接进 `F.cross_entropy(model(batch), batch["labels"])`（`train.py:74`），
即 **softmax 的列就是这一行的 N 个选项**。JSONL 的一行长这样（`README:45`）：

```json
{"context":"The customer needs a refund.","options":["refund","sales","technical support"],"label":0}
```

`README:48` 明说 *Each row may have a different number of options, with a minimum of two.*

这条设计选择的后果是一条链的：

| 别家要处理的事 | 这里为什么不必 |
|---|---|
| 标签必须单 token（RYOTIDE、decider、SemIf-OpenJev、Nimble） | 标签是下标，不进分词器 |
| 标签表要固定（Jeff 的 255、Laya 的槽位、Kev 的 `<opt>` 跨度） | 类别集**逐行不同**，没有表 |
| 候选数上限（各家 16 / 52 / 255 / 260） | **没有上限**，只受显存与 token 预算约束 |
| 掩码到标签集 + 重归一化 | 头**只输出 K 个标量**，没有更大的空间要掩（§5） |
| `marker_mass` 这类诊断量 | 恒等于 1，无信息（§10 表 #1） |

代价同样清楚，而且这一篇里没有对应机制去补：**标签的含义不跨行**。
`label=0` 在第 1 行指「refund」、在第 2 行指别的；所以任何跨行的「类别混淆矩阵」都不存在，
`eval.py` 也只能报 top-1 / top-3 / ECE，不能报逐类召回（对比 chess 那份数据是有逐类召回的，
`examples/chess/train.py:36-39` 的 `per_key_recall`——因为那 5 个键是**固定**的）。

### 2.2 两类编码器，两份 collator，同一个 `make_system`

`make_system(config, device)`（`model.py:107-126`）是**训练与服务唯一的分叉点**，
两侧都从同一份 `config` 构造模型与 collator：

| | tiny（默认） | hf |
|---|---|---|
| 模型 | `TinyScorer(width, rank, context_tokens)` | `FrozenTransformerScorer(hf_model, rank)` |
| 输入表示 | `nn.Embedding(257, width)`，`padding_idx=0` | 冻结编码器的 `last_hidden_state` |
| 位置 | `nn.Embedding(context_tokens, width)` 学习式绝对位置（`model.py:50`） | 交给编码器自己 |
| 选项向量 | 选项 token 的**掩码均值池化**（`model.py:59`） | 选项展平后过一次编码器，再掩码均值池化（`model.py:83-90`） |
| 截断 | `context_tokens` **字节** / `option_tokens` **字节** | `max_length=context_tokens` **token** / `option_tokens` **token** |
| 头宽 | 由 `--width` 定 | 由 `encoder.config.hidden_size` 定，**`config["width"]` 在这条分支上从不被读**（§13.3） |
| 可训练参数 | 全部 | 只有 `head.*`（`encoder.requires_grad_(False)`，`model.py:72`） |

默认值是 `--width 64 --rank 64 --context-tokens 192 --option-tokens 32`（`train.py:40-43`）。
按这组默认值手算 tiny 的参数量：`257×64 + 192×64 + (2×64 + 2×64 + 3×64×64)` =
`16 448 + 12 288 + 12 544` = **41 280**。整个文本模型四万参数——这是「不读 LLM」换来的直接结果。

**正例先记下来**：训练与服务走的是**同一个函数**（`train.py:57`、`predict.py:25`、`eval.py:59`
都调 `make_system` 或 `load_checkpoint`），所以「训练与推理的输入渲染不一致」这个坑在这里
没有对象。这与 CLM 靠共享 `build_pairs`、Semantic Router 靠共享 `render/collate` 是同一种做法，
是这一族里少见的、被**结构**而不是被**约定**保证的一致性。

### 2.3 负例：截断到处都是，而且一处也不报

这是本篇唯一一条「别家做对了、这里没做」的对称项。

- **字节路径**：`_bytes(text, length)` 是 `text.encode("utf-8", errors="replace")[:length]`（`data.py:49-50`）。
  `errors="replace"` 管的是 lone surrogate（编码失败），**管不了截断**——切的是**字节串**，
  所以会切在多字节字符中间。手算：一个 CJK 字符 3 字节，`--option-tokens 32` 会留下 10 个完整字符
  加第 11 个字符的头 2 个字节；`--context-tokens 192` 同理。
  切完的字节序列仍然是一个合法的 token 序列，**没有任何地方会发现它是残的**。
- **`tokenizer` 路径**：`truncation=True`（`data.py:73-74`、`:77-79`）静默截断。
- **全仓没有一处截断计数或告警**。对比 Quyet 那一篇：它只有 state 被截断，但把 `truncated`
  与 `warnings` 上报出来；这里连「有多少条被截断了」这个数都不存在——全仓 `truncat*` 的命中只有
  collator 里那两处 `truncation=True`（`data.py:73`、`:78`）与 `README:81` 的一句用户须知，
  **没有任何一处统计或告警**。
- 上游把这件事写在文档里当**用户责任**：*Raise `--context-tokens` or `--option-tokens` when your text
  needs more room.*（`README:81`）。

一个具体的量级差，顺手算一下：`build_wikispeedia` 写出的 context 是
`f"Target article: …\nCurrent article: …\n{body[:2048]}"`（`data.py:188`）——正文留了 **2048 字符**；
而默认 `--context-tokens 192` 在字节路径上只保留前 **192 字节**，约 40 个 ASCII 字符的正文。
也就是说 **quickstart 的默认配置会把 wikispeedia 的正文几乎全部丢掉**。
`README:117` 报的那个 26% 用的是哪一套预算**未核实**（它写的是「frozen Qwen2.5-0.5B encoder plus the
scorer」，那是 token 预算而不是字节预算）。

### 2.4 hf 路径上的一处 `pad_token` 副作用（未核实）

`make_system` 在分词器没有 pad token 时把它设成 eos（`model.py:118-119`），
而 `_tensor_batch` 用 `context_ids.ne(pad_id)` 算 `context_mask`（`data.py:103-104`）。
于是**当 `pad_token_id` 原本为 `None` 时，每行由 `add_special_tokens=True` 追加的那个 EOS
也会被判为 padding 并屏蔽掉**——它同时影响编码器自己的 `attention_mask`（`model.py:78-80`）
和这个头的注意力掩码（`model.py:38-40`）。

本文**未核实**这条路径的实际后果：本机没有分词器，也不下载权重，无法确认示例用的
`Qwen/Qwen2.5-0.5B` 的 `pad_token_id` 是否为 `None`。可以确定的只有代码形状：
字节路径不受影响（pad id 是 0，而真实字节都被 `+1` 抬到 ≥1，`data.py:50`），
两条路径在这个副作用上是**不对称**的。若 `pad_token_id` 本就有值（与 eos 不同），则整条问题不存在。
另外，`torch.full(..., fill_value=None)` 会直接抛错——是 `model.py:118-119` 那一句兜住了这个 `None`。

---

## 3. 读位置：这个变量不存在，取而代之的是一张权重分布

RYOTIDE 的第 3 节在问「读哪个位置」。这里没有这个问题可以问，因为**没有任何一步读位置**。

### 3.1 「位置」被替换成了什么

头里的 `context` 是一个 `(B, L, W)` 的张量，`L` 是这一段 state 的 token/字节数。
`L` 从哪来：

- tiny：`_bytes` 按**字节**切到 `context_tokens`；位置嵌入是**学习式的绝对位置**，
  表长恰好等于 `context_tokens`（`model.py:50`），所以「位置」在这里只有「第几个字节」这一个含义。
- hf：由分词器决定（`add_special_tokens=True` 会加 BOS/EOS），位置交给编码器。
- 视觉：`position_2d(rows, columns, width)` 造**固定的二维正弦位置**（`vision.py:20-30`），
  8 行 10 列共 80 个 patch。

视觉侧有一处值得单说的设计：位置是**加在 key 上**的，而且是在学出来的投影**之后**加：

```python
key = head.key(context) + positions          # vision.py:107
```

源码在这一行上面留了理由（`vision.py:105-106`）：
*The fixed term is deliberately added after the learned projection. It cannot be washed out by
context normalisation or routed only into V.* `AGENTS.md:60` 把这件事记成一次真实的修复：
*a convolutional stem plus fixed two-dimensional positions added directly to the attention keys
fixed the clearest representation bug. Keep both.*

### 3.2 「读位置」变成「读的权重分布」，而它是可观测的

内层 softmax 的输出来自 `scores[n,l]`（`model.py:37-41`），形状 `(B, N, L)`：
**每个选项一条、长度等于 state 的分布**。这相当于 RYOTIDE 的「读位置」被摊开成了一张图，
而且是逐选项的。视觉侧的 trace 把它暴露出来（`vision.py:119-126` 的 `attention_map`），
`examples/doom/play.py:98-118` 与 `examples/chess/play.py:126-141` 把每个决策的注意力矩阵
写进 trace（`grid_rows × grid_columns`），再由 `examples/film/*.mjs` 画成热力图。

**文本侧没有任何等价物。** `jevlike-eval` 只输出 top1 / top3 / ECE（`eval.py:44-48`），
`jevlike-predict` 只输出每个选项一个概率（`predict.py:32-35`）。所以：

| | 视觉侧 | 文本侧 |
|---|---|---|
| 注意力分布 | `attention_map` 进 trace，另有 `attention_entropy_by_action`（`vision.py:216-227`） | **无** |
| 分布是否非均匀 | `normalised_entropy_mean` / `peak_mean`（`audit.py:141-144`） | **无** |
| 分布是否跟踪任务变量 | `x_enemy_correlation_by_button`（`audit.py:145-148`） | **无** |
| 输入是否真被用上 | 真实 / 空白 / 平均 / 打乱 patch 四组对照的 KL（`audit.py:161-164`） | 只有 shuffled-context 一个对照（`eval.py:65`、`model.py:31-33`） |

上表最后一行是这一篇里唯一一个**文本侧的守卫**，它值得单独讲：
`AttentionHead.forward(..., shuffle_context=True)` 会把整个 batch 的 context **滚一格**
（`model.py:31-33`），于是每个菜单配上**另一个样本的 context**；`eval.py:63-67` 把两组指标并排打印。
`README:71` 的说法是 *A useful model should beat that control.*

两个实现细节要记住：①它要求 `context.shape[0] > 1`（`model.py:31`），
所以 batch 为 1 时这个开关是**静默 no-op**；②`eval.py` 的 loader 没有 `shuffle=True`
（`eval.py:60-62`），所以这个对照是**确定性的轮转**，不是随机配对。

---

## 4. 一次前向：不是「只读一个位置」，是「每个选项读一次」

### 4.1 三行 einsum

```python
scores   = torch.einsum("bnr,blr->bnl", query, key) / math.sqrt(self.rank)      # model.py:37
attended = torch.einsum("bnl,blr->bnr", scores.softmax(-1), value)              # model.py:41
logits   = (query * attended).sum(-1) / math.sqrt(self.rank)                    # model.py:42
```

`b` = 批，`n` = 选项，`l` = state 位置，`r` = rank。三点值得标出来：

1. **内层 softmax 在 `l` 上做**（`:41`），归一化的是「这个选项怎么分配注意力到 state 各个位置」；
2. **外层 softmax 不在头里**。头返回 logits，由调用方施加（`predict.py:31`、`eval.py:24`、
   `vision.py:125`）。所以「一次前向里两个 softmax，只有一个是对选项的决策分布」这句话，
   在代码里就是**两个不同层级的函数**在各自负责——这与 decider 那篇要仔细区分「哪条 API 的
   哪个 `confidence`」是同类问题，但这里分得干净：`softmax(-1)` 的轴不同，语义不可能混。
3. **`/ √rank` 出现两次**（`:37` 与 `:42`），维度上是对的：`scores` 是点积除以 `√r`，
   `logits` 也是点积除以 `√r`。

### 4.2 选项之间零交互，这一点比 CLM 还强

`query` 是**逐选项**算的（`model.py:34`），`scores`/`attended`/`logits` 的每个 `n` 只依赖
**它自己的 query** 与 `context`。`context` 完全不受选项影响。于是：

> 对选项做任意排列 π，`logits(π(options)) = π(logits(options))`，**精确成立**。

CLM 那一篇的结论是「候选之间零交互，所以逐候选独立编码 + 逐候选点积」；
这里更彻底——不是两个向量比一次，而是**每个选项各自拿一次 state 的加权平均**，
彼此之间连共享的被比较对象都没有。§6 讲这条性质的代价。

### 4.3 两条编码路径各自的开销

| | 每次前向要跑几次编码器 | 选项向量的来源 |
|---|---|---|
| `TinyScorer` | 0 次（就是查表） | `embedding(option_ids)` 后掩码均值池化（`model.py:57-59`） |
| `FrozenTransformerScorer` | **2 次**：state 一次（`model.py:78-81`），**全部选项展平成一个 batch** 再一次（`model.py:82-87`） | 池化后 reshape 回 `(B, N, W)`（`model.py:88-90`） |
| `DoomScorerV2` | 0 次 | `self.options(option_ids)`，12 行表里查（`vision.py:91-99`） |

第三条是 CLM 的正对面：CLM 把「可缓存的候选向量」当成整个设计的支点，
而视觉侧这里**候选向量干脆就是一张 12 行的表**——共享的是**参数**，不是状态。
hf 路径则是反面：**每一批都要把选项重新过一遍编码器**，同一个选项文本在下一批会重新算一遍，
没有任何缓存。`README:98` 承认了另一层代价：*The checkpoint stores the trained scorer head and the
encoder name. It does not copy the frozen encoder weights.*——所以检查点里只有一个**模型名字符串**，
没有 revision、没有 sha256（`model.py:138-142` 只校验 `state_dict` 的形状）。这与 CLM 那篇
「检查点里不记录编码器身份」是同一个坑的较轻版本：名字记了，**版本没钉**。

---

## 5. 掩码 + softmax：掩码在，但它掩的不是标签集

这一篇里有两个掩码，**都不掩标签**，这一点必须和其他几篇分开看。

| 掩码 | 位置 | 掩的对象 | 不掩会怎样 |
|---|---|---|---|
| ① `context_mask` | `model.py:38-40` | state 侧的 padding 位 | **会读到 padding**：这个头对 state 是**双向**的（没有因果掩码），右填充**不自动安全** |
| ② `option_mask` | `model.py:43` | 选项侧的 padding 槽位 | 外层 softmax 会把概率分给不存在的槽位，`eval.py` 的 top-k 与 ECE 会读到它们 |

① 与 Laya、Semantic Router 的 encoder 是**同一处境**：一旦读法不是因果的，
「右填充所以不用传 mask」这条省事的推理就失效了。这里传了，而且两个 mask 是两套：

- `context_mask = context_ids.ne(pad_id)`（`data.py:103`）
- `option_token_mask = option_ids.ne(pad_id)`，逐 token（`data.py:104`）
- `option_mask` 是**逐行有几个选项**的布尔（`data.py:95-99`），与上面两个不同层级

② 的效果就是「重归一化」，但是**自动的**：`masked_fill(finfo.min)` 在 softmax 之前
（`model.py:43`），`exp(-3.4e38 - max) = 0`，所以被掩的槽位概率**恰好为 0**，
真实选项的概率和本来就是 1。RYOTIDE 那种「先算出整个词表分布、再把质量收回到标记上」
在这里没有对象——**没有更大的空间**可以溢出。这也就是 §10 表 #1 说 `marker_mass`
恒等于 1 的原因。

数值上的两个安全点：`torch.finfo(dtype).min` 而不是 `-inf`（避免 `-inf - (-inf)` 出 NaN），
以及每一行至少有 2 个真实选项（`data.py:28-29` 挡住），所以 `logsumexp` 不会落在全掩行上。

还有一处**参数级的截断**值得记在案：`predict.py:31` 的 `softmax(-1)[0, :len(args.option)]`
在单样本 batch 上是 no-op（`_tensor_batch` 会按这行的选项数定 `max_options`），
它的作用是给批内长度不齐时留一道保险——而 `eval.py` 没有这道保险，
它直接在**补齐后**的行上取 `max`（`eval.py:30`），靠的正是掩码把补齐位压成 0 概率。

---

## 6. 顺序偏置：构造性不存在

按 §4.2 的等变性，把选项表打乱只会把输出 logits 同样打乱。所以这一节不是「怎么消除」，
而是「这条性质买到了什么、又付掉了什么」。

**买到的**：不需要 RYOTIDE 的多 order 平均、不需要 Laya 的 `parallel` 布局、
不需要 Kev 的每 epoch 打散与 `option_isolation` 开关、不需要 decider 的 `_NoShuffle` + 反转重读。

**付掉的**，两条，而且都没有守卫：

1. **两个文本相同的选项在数学上不可区分**（同样的 `option_ids` → 同样的 query → 同样的 logit），
   于是并列，而并列由 `argmax` / `topk` 按**索引顺序**裁决（`eval.py:25-26`、`predict.py:31`
   打印的是原顺序的数组）。这与 CLM 那篇记的是同一个坑：「两条相同文本会静默并列并被顺序裁决」。
   `validate()` 只检查**每个选项非空**（`data.py:28`），**不检查互不相同**。
2. **运行期不保序也不打乱，而训练数据只在一处打乱**：`synthetic_example` 用 `rng.shuffle(options)`
   （`data.py:122`），`build_wikispeedia` 用 `menu` 两次 `rng.shuffle`（`data.py:183-184`）。
   但**评测集里的顺序是固定的**（`JsonlDataset` 原样读入，`data.py:35-46`），
   而且**全仓没有任何顺序偏置的度量**——没有 `/permute`（Kev）、没有 `flip_rate`
   （Kev、Semantic Router）、没有 `argmax_stable`。语义偏置这一栏在这里是「构造上不存在」，
   但「不存在」是**证明**出来的，不是**量**出来的。

视觉侧的「顺序」是另一回事，得区分开：12 行共享表里，Doom 用 id 0–6、象棋用 id 7–11
（`vision.py:15-16`），`option_ids` 决定**读哪几行**。所以这里的选择是**子集**，不是排列——
而 `logits.shape == (B, len(option_ids))`（`examples/chess/test_smoke.py:39`）说明
被选中的行按传入顺序排列，未选中的行根本不参与。这带来一个文本侧没有的能力：
**同一个进程里两种游戏的选项词表互不干扰**（§11）。

---

## 7. 多问题共享 state：不做

- **文本侧**：一个 JSONL 行 = 一次前向。没有多题循环，没有 `run_many` 的对应物。
- **不做也不亏**：tiny 路径下一次前向的成本是「查表 + 三次 `width×rank` 的投影 + 两次小 einsum」，
  没有可复用的 state 编码；hf 路径**理论上**可以复用 state 的编码器输出，
  但代码里没有任何缓存，`forward` 每次都重算（`model.py:77-90`）。
- **真正被共享的东西不在这一层**：是视觉侧的**一张 12 行嵌入表**——`DoomScorerV2.options`
  被两个游戏共用（`examples/chess/train.py:62`、`examples/doom/train_joint.py:59`、`train_joint_ppo.py:32` 建的是 12 行，
  而 Doom 的两个单游戏脚本建的是 7 行：`train_imitation.py:81`、`train_dagger.py:67`）。
  `examples/chess/README.md:3` 的原话是 *The network has one table of 12 option embeddings.*
  共享的粒度是**参数**，不是状态。
- **延迟只在一个地方被量**：视觉侧的 `benchmark()`（`vision.py:197-213`，200 次取均值）、
  `play.py` 的 `online_inference_ms`、`chess/eval.py` 的 `ms_per_key`（`eval.py:196-208`）。
  **文本侧一行延迟代码都没有**——`jevlike-eval` 的输出里没有时间字段（`eval.py:63-67`）。
  所以「一次前向比逐 token 生成快多少」在这个仓里**只有视觉侧有测量**。
- `README:117` 那一句 *At eight options, one pass was about 100 times faster than a small decoder
  forced to write 400 tokens* 是**上游自述的实验**，既没有脚本也没有记录随仓提交，本文不核对。

---

## 8. 服务化：这一篇**没有服务**

收录的路线里第一条完全没有 HTTP 面的——连 `/health` 都没有。

`pyproject.toml:27-31` 只有四个 console script：

| 入口 | 文件 | 输入 | 输出 |
|---|---|---|---|
| `jevlike-data` | `data.py:201-223` | `synthetic` / `wikispeedia` 子命令 | JSONL |
| `jevlike-train` | `train.py:33-91` | train + validation JSONL | `.pt` + 逐 epoch 一行 JSON 到 stdout |
| `jevlike-eval` | `eval.py:51-67` | checkpoint + JSONL | 一个 JSON（`model` 与 `shuffled_context` 两组指标） |
| `jevlike-predict` | `predict.py:15-35` | checkpoint + `--context` + `--option`×N | 一个 JSON 数组（选项与概率） |

游戏侧也不是服务，是脚本：`play.py`（输出 mp4 / trace JSON）、`eval.py`（棋局统计）、
`audit.py`（诊断 JSON）。**没有任何常驻进程、没有请求校验层、没有鉴权、没有响应模型。**

于是 fail-loud 只能落在三个地方，而且都是「早失败」而不是「运行期守卫」：

| 位置 | 检查 | 失败方式 |
|---|---|---|
| `predict.py:22-23` | `--option` 少于两次 | `parser.error` |
| `data.py:24-32` | context 是 str、options ≥2 且非空、label 在范围内 | `ValueError` |
| `data.py:39-40` | JSONL 里一条样本都没有 | `ValueError` |
| `model.py:139-141` | 检查点与当前 config 不匹配：`missing` 里除 `encoder.` 之外的任何名字、或任何 `unexpected` | `ValueError` |
| `examples/doom/train_dagger.py:69-72` | 只允许 `extra_heads.*` 缺失 | `ValueError` |
| `examples/doom/environment.py:59-60` | 按钮数不等于 7 | `RuntimeError` |
| `examples/chess/data.py:64` | 找不到 positions | `FileNotFoundError`（`data.py:157` 还会在起 worker 池**之前**先探一次，注释是 *fail fast in the caller*） |
| `examples/chess/data.py:107` | teacher 走 40 步还没落子 | `AssertionError` |

值得夸一处、也该点一处：

- `load_checkpoint`（`model.py:136-142`）用 `strict=False` 但**手工把缺口收紧**——
  允许缺 `encoder.*`（因为冻结编码器的权重本来就不存，`model.py:129-133`），
  其余任何缺口或任何多余键都抛错。所以「rank 或 width 改了」这类错配**不会静默加载成随机头**。
  这比语义相近的几条路线（比如 `device_map="auto"` 让头留在 `meta` 却照样回 ready）要干净。
- 但**没有任何东西检查「这个 checkpoint 是哪个 encoder 训的」**：
  `hf_model` 只是一个名字（`model.py:119`），且 `config["width"]` 在 hf 分支从不被读（§2.2 表）。
  换一个 revision 的同一名字模型，检查点照样加载成功，head 的输入宽度只要一致就不会报错。

---

## 9. 置信度从哪来，又怎么被改写

### 9.1 答：训出来的，而且链条上只有一环

`logits.softmax(-1)` 就是这个模型对外的概率，不存在任何二次校正：

| 环节 | 有没有 | 证据 |
|---|---|---|
| ① 训练本身 | **有** | tiny 训全部参数（`train.py:66-67` 收集 `requires_grad` 的参数）；hf 只训头（`model.py:72`） |
| ② 温度 | **没有** | 全仓 `temperature` 只出现在 `examples/chess/eval.py:72-73`（见 §9.3） |
| ③ bias / 重加权 / 保序平均 / 舍入 | **没有** | grep 零命中 |
| ④ ECE 或别的标定产物 | **没有** | `eval.py:37-43` 只**算** ECE，从不写回任何东西，也不产生文件 |
| ⑤ 接受 / 拒绝判据 | **没有** | 没有标定环节，因此没有判据 |

所以「跨实现比较概率之前先确认 temperature」这条不变量在这里**成立但空洞**：
没有可确认的量。这与 CLM 那篇的处境相邻但更极端——CLM 有一个每请求的入参 `temperature`
（默认 1.0、不回声）；**这里连入参都没有**，`softmax` 的输入就是头吐出来的 logits。

### 9.2 ECE：10 个等宽箱，只报不改

`eval.py:37-43` 的分箱是 `torch.linspace(0, 0.9, 10)`，即 `[0,0.1), [0.1,0.2), … , [0.9,1.0)`，
**10 个等宽箱**，按箱内样本占比加权求和 `|acc − conf|`。`confidence` 是
`probabilities.max(-1)`（`eval.py:30`），即这一行的最大概率。

口径上与谁同、与谁不同（对照 `docs/comparison.md` §5 第 2 条）：

| 10 箱 | 15 箱 |
|---|---|
| RYOTIDE（vendored JevBench）、SemIf-OpenJev、**Jevlike** | Open-Jev、Laya、decider |

另外注意两件事：①文本侧的 ECE 是在**逐行候选数不同**的混合批上算的（一行 2 个选项、
一行 8 个选项都进同一个箱子），而候选数本身强烈影响 `p_max`——这个口径在任何一家的文档里
都没有对齐过；②视觉侧的 `audit.py` 不算 ECE，它算的是另一族量（§11）。

### 9.3 全仓唯一叫 `temperature` 的东西：下棋时采样按键

```python
# sample=True is temperature 1; a float is the softmax temperature
probs = (logits.float() / float(sample)).softmax(-1).cpu().numpy().astype(np.float64)
#                                                                     examples/chess/eval.py:72-73
```

CLI 是 `--sample [T]`（`eval.py:264-265`），裸用即 1.0，不给就贪心取 `argmax`（`eval.py:70-71`）。
`examples/chess/play.py:112-115` 里也有一个不带温度的 `torch.multinomial` 分支。
**这是「怎么下这一步棋」的采样旋钮，不是「发布出去的概率」的校正**——
它只影响游玩时选哪个键，不影响任何被报出去的置信度。把它和 RYOTIDE/Jeff/decider 那些
标定温度并列会是错的，本文把它单列在这里就是为了避免这个误读。

顺带一条同源的性质：**贪心比采样差得多**。`examples/chess/README.md:78` 记了
*Choosing the most likely key every time is much worse than sampling: for a checkpoint halfway
through step 3, 68% of moves hit the key budget*——因为屏幕是唯一的状态，一个不改变屏幕的按键
（或两个互相抵消的按键）会无限循环（`examples/chess/README.md:78`）。
这是这一族里少见的、**「取 argmax 反而不如采样」的实测**，也解释了 `README:9` 的 demo 为什么用采样。

---

## 10. 四条不变量的落位

| RYOTIDE §13 不变量 | Jevlike 的落位 |
|---|---|
| **#1** 读位置必须落在标记上（诊断量 `marker_mass`） | **概念不适用**：没有词表分布，`marker_mass` 恒等于 1、无信息。**改写后的等价命题**：「softmax 的输入必须恰好是这一行的 K 个选项标量」——构造上成立（`model.py:42-43`，选项之间零交互，§4.2）。这个改写命题**在视觉侧有诊断量**（逐选项的注意力归一化熵与 peak，`vision.py:216-227`、`audit.py:138-144`），**在文本侧一个都没有**（其 §3.2） |
| **#2** 前缀与标记表面形式耦合，必须成对扫 | **不适用**：没有标记字符串，也没有 prompt——标签是下标（`data.py:30-31`），选项文本各自独立编码。**改写后的等价命题**：「**候选文本必须完整进入编码，截断必须可见**」，而这是本篇唯一的**负例**：两处静默截断（按字节切与 `truncation=True`，`data.py:50`、`:73-79`），无计数、无告警、无 `truncated` 字段（其 §2.3）。正例那一半是「训练与服务共用同一个 `make_system`」（其 §2.2） |
| **#3** 只读最后一个位置 | **改写**：不是「读一个位置」，而是**每个选项对全部未掩位置的一次加权平均**（`model.py:37-41`）；「不物化词表维度」这半条成立得**最彻底**——连骨干都不是语言模型（其 §4.1）。顺带一处与 Kev 同类、与 Laya 同因的细节：state 侧的注意力**不是因果的**，所以右填充必须靠掩码（其 §5） |
| **#4** 比较概率 / ECE 前先确认 temperature | **成立但无可确认**：没有温度、没有标定产物、没有发布配方、没有入参（其 §9.1）。要确认的只剩「这份概率是哪次训练产出的」——而检查点里连 encoder 的 revision 都没钉（其 §4.3） |
| （各自新增的） | **①候选文本必须完整进入编码，截断必须可见**——本仓**没有**这条守卫（其 §2.3）；**②同序并列必须被显式裁决**——两个文本相同的选项在数学上不可区分，裁决发生在 `argmax`/`topk` 的索引顺序上，没有任何地方记录或告警（其 §6）；**③「读」这个动作本身需要守卫**——因为读的不是位置而是分布，所以守卫的形式也变了：不是「位置对不对」，而是「这份分布是不是非均匀、是不是跟着任务变量走」（`audit.py:137-148`，视觉侧有、文本侧没有，其 §11.4） |

---

## 11. 视觉分支：同一个头读画面

这是这一篇独有的维度，不是「另一种读法」，而是**同一个读法换一种输入**。
`examples/doom/README.md:3` 与 `examples/chess/README.md:3` 都把这一点写在第一段。

### 11.1 输入与 8×10 网格

输入是一帧 `(120, 160, 4)` 的 uint8：RGB 三通道 + 一个由相邻两帧差编码出来的 motion 通道
（`vision.py:181-189`）。`play.py:71-74` 在抓 640×480 的录像时，**策略仍然只看缩回 160×120 的画面**
（`examples/doom/README.md:19`）。

网格形状是算出来的，手算一遍（这几步全部只用 `vision.py:48-56` 的卷积参数）：

| 步骤 | H | W | 公式 |
|---|---|---|---|
| 输入 | 120 | 160 | 一帧 |
| `F.pad(image, (0,0,4,4))` | 128 | 160 | 上下各 4（`vision.py:73`） |
| `Conv2d(4,16,5,stride=2,padding=2)` + GroupNorm + SiLU | 64 | 80 | `⌊(n+2p−k)/s⌋+1` |
| `Conv2d(16,width,3,stride=2,padding=1)` + GroupNorm + SiLU | 32 | 40 | 同上 |
| `Conv2d(width,width,4,stride=4)` | **8** | **10** | `⌊(n−k)/s⌋+1` |
| `permute(0,2,3,1).flatten(1,2)` | — | — | → `(B, 80, width)`（`vision.py:74`） |

`self.grid_rows, self.grid_columns = 8, 10`（`vision.py:46-47`）必须等于这张表的结果，
否则 `key + positions` 会**直接广播失败**（`vision.py:107`）。三处独立证据互相钉住：
`position_2d(8,10,width)` 的输出是 80×32（§14.2 在检查点里数到了这 2560 个浮点数）、
`PlainConvPolicy` 的 `Linear(width*32*40)`（`vision.py:148`，正是 stem 的输出形状）、
以及 smoke 测试里的 `trace["attention_map"].shape == (1, 7, 80)`
（`examples/doom/test_smoke.py:27`）。

另外两个数值细节：RGB 标准化用的是**逐样本**的 `mean/std`，`std` 有 `clamp_min(0.08)` 下限
（`vision.py:68-70`）；motion 通道先 `(x−0.5)*2` 拉到 `[-1,1]`（`vision.py:71`）。
`examples/doom/audit.py:89-91` 造对照帧时正是按这个约定造的：
`blank` 是全 0 且 motion 置 128、`average` 是全批均值。

### 11.2 12 行共享表与 `load_expanded`

`TOTAL_OPTIONS = 12`，`DOOM_OPTION_IDS = 0..6`，`CHESS_OPTION_IDS = 7..11`（`vision.py:15-17`），
选项就是 `nn.Embedding(12, width)`（`vision.py:57`）。两个游戏各用一段 id，
所以「同一个模型打两个游戏」不需要任何路由或任务标记——`examples/chess/README.md:3`：
*Nothing tells the network which game it is playing except the frame.*

把 7 行的 Doom 检查点与 12 行的表拼起来的是 `load_expanded`（`examples/doom/train_joint_ppo.py:28-49`）：

```python
if name == "options.weight":
    rows = min(len(value), len(target[name]))
    target[name][:rows] = value[:rows]          # 前 rows 行照抄，缺的留初始化
elif name in target and target[name].shape == value.shape:
    target[name] = value                        # 其余张量形状一致才搬
model.load_state_dict(target)
...
model.options.weight[7:12].copy_(chess_options[7:12])   # 象棋那 5 行单独从象棋检查点取
```

三处要点：①它是一个**容错搬运**（`min` 取小、形状不符就跳过），不是 `load_state_dict` 的严格匹配
——所以它不会因为行数不同而报错；②象棋检查点若不足 12 行则**显式抛错**（`:45-46`，
*chess checkpoint does not contain the joint 12-option table*）；③对选项表的操作是**逐行覆盖**
而非拼接，未覆盖的行保留 `DoomScorerV2.__init__` 的初始化。

### 11.3 `reads` / `extra_heads`：一次实验，且被记录成负结果

`DoomScorerV2(reads=k)` 会多造 `k−1` 个 `AttentionHead`（`vision.py:59-61`），
前向时把它们的 logits **取平均**（`vision.py:101-114`），trace 里也把 query/key/value/attention
一并平均（`vision.py:119-126`）。`train_dagger.py:73-77` 新增 read 时先**克隆主头**
再叠加 `--read-init-noise`（默认 0.01）的对称性破坏噪声。参数在 `--reads` 上限制为 1..4
（`train_dagger.py:52-53`）。

结论记在 `AGENTS.md:63`：*Duplicating an attention read with identical weights did not help;
symmetry-breaking noise still gave no clear gain.* 也就是**接了、测了、没有收益**——
与 Laya 那篇 `soft-ce` 那条是同一类（源码里明记负结果）。

### 11.4 诊断：这一族里最厚的一套，而且它是**审计脚本**不是服务守卫

`examples/doom/audit.py` 的 `analyse()` 一次产出（`:151-183`）：

| 量 | 位置 | 回答什么问题 |
|---|---|---|
| `kl_nats.real_vs_blank` / `real_vs_average_frame` | `:161`、`:163` | 分布是不是只在「有东西」时才变 |
| `kl_nats.real_vs_shuffled_patches` | `:162` | **画面被拆散后策略变不变**——这是「到底看没看屏幕」的判据 |
| `kl_nats.real_vs_mean_policy` | `:164` | 逐帧分布与批次平均分布的距离 |
| `probability_ranges`（逐按钮 min/max/range/std/mean） | `:133-136` | 有没有哪个按钮退化成常数 |
| `attention.normalised_entropy_mean` / `by_button` | `:138`、`:142-143` | 注意力是不是几乎均匀（除以 `ln 80`，`:138`） |
| `attention.peak_mean` | `:144` | 最大注意力权重 |
| `attention.x_enemy_correlation_by_button` | `:145-148` | 注意力的水平重心与敌人屏幕位置是否相关 |
| `enemy_probe.attack_margin` / `turn_left_margin` / `turn_right_margin` | `:167-176` | 概率是不是真的随敌人位置改变 |

判据写得很硬，`AGENTS.md:33`：*Reward above random does not count as visual control if
shuffled-patch KL and enemy-conditioned margins are near zero.*
`AGENTS.md:59` 还记了这条判据抓到的东西：*PPO from pixels learned a useful fixed action prior
but almost ignored the screen. Attention was nearly uniform and shuffled patches barely changed
the policy.*

要说清的是它的**位置**：`audit.py` 是一个**事后脚本**，不在任何前向路径上，
也不产生任何门控；而且它只对**视觉**模型成立。文本侧没有对应物，
`jevlike-eval` 只给 top1/top3/ECE 与一个 shuffled-context 对照（§3.2）。

### 11.5 上游自己记下的负结果清单

`AGENTS.md:55-66` 有一节 *What has been tried*，逐条写「试过、没用/有用」。
其中三条值得从别处引一次，因为它们与本文的机制判断互相印证：

- *A flat convolutional policy learned combat faster than the one-read-per-option head on
  identical DAgger data. The option head remains a capacity bottleneck*（`AGENTS.md:62`）——
  **一次读一个选项**这个结构本身是瓶颈。这条结论的对照臂就是 §13.1 说的那个**不可达分支**。
- *A short 20-update joint imitation run largely erased chess*（`AGENTS.md:64`）——
  与发布说明里「joint 检查点丢掉了大部分棋力」是同一件事，§14.3 在权重上量到了它。
- *Log attention entropy during RL. If representations collapse again…*（`AGENTS.md:66`）——
  诊断量是被当成**回归守卫**用的。

---

## 12. 文件对照表

| 文件 | 行数 | 职责 |
|---|---|---|
| `jevlike/model.py` | 142 | `AttentionHead`（唯一的打分器）、`TinyScorer`、`FrozenTransformerScorer`、`select_device`、`make_system`、`trainable_state`、`load_checkpoint` |
| `jevlike/data.py` | 223 | JSONL 读入与 `validate`、`ByteCollator`、`HuggingFaceCollator`、`_tensor_batch`、合成数据、wikispeedia 构造 |
| `jevlike/train.py` | 95 | 训练循环、按 validation NLL 留最佳 state、写检查点 |
| `jevlike/eval.py` | 71 | top1/top3/ECE + shuffled-context 对照 |
| `jevlike/predict.py` | 39 | 单次打分 CLI |
| `jevlike/vision.py` | 227 | `position_2d`、`DoomScorerV2`（含 `reads`）、`PlainConvPolicy`、`observation`、`benchmark`、`attention_entropy_by_action` |
| `jevlike/__init__.py` | 7 | 只导出 `vision` 的四个名字 + `__version__` |
| `examples/doom/environment.py` | 87 | ViZDoom 环境与七个按钮、tics 常量 |
| `examples/doom/diagnostics.py` | 44 | `kl`、`correlation`、`enemy_observation`（模型无关的小工具） |
| `examples/doom/audit.py` | 206 | 屏幕依赖审计（§11.4） |
| `examples/doom/train_imitation.py` | 173 | labels-buffer 专家模仿 + 左右镜像增强 |
| `examples/doom/train_dagger.py` | 182 | DAgger + `reads` 扩展 |
| `examples/doom/train_ppo.py` | 337 | PPO、形状奖励、梯度范数分组、随机基线 |
| `examples/doom/train_joint.py` | 194 | 交替 DAgger CE（**committed 的 12 选项检查点出自这里**，§14.3） |
| `examples/doom/train_joint_ppo.py` | 339 | `load_expanded`、`JOINT_ACTIONS`、`chess_frames`、混合 PPO |
| `examples/doom/play.py` | 170 | 定速游玩、录像、trace（注意力矩阵进 trace） |
| `examples/chess/render.py` | 183 | 把棋盘画成 120×160 的画面（含合法落点小方块） |
| `examples/chess/keys.py` | 94 | 五键控制器与 `press` 事件（`edge`/`empty`/`illegal` 就是 wasted press） |
| `examples/chess/positions.py` | 87 | Stockfish 标注位置（含 noisy self-play 与 vs-random 两条） |
| `examples/chess/data.py` | 193 | teacher walk → 逐帧下一个键的标签、DAgger 混合、固定评测集 |
| `examples/chess/train.py` | 107 | 预训练（只动 7–11 行）、逐键召回、注意力熵 |
| `examples/chess/dagger.py` | 145 | 采模型自己走到的状态并用 Stockfish 标注 |
| `examples/chess/eval.py` | 291 | 对局评测（胜/和/负、wasted、keys/move、CPL、按颜色分） |
| `examples/chess/play.py` | 168 | 定速下棋 trace（带 `clean_windows` 选片） |
| `examples/chess/stem.py` | 29 | 膨胀卷积 stem 实验（**全仓零调用点**，§13.2） |
| `scripts/get_wikispeedia.sh` | 22 | 下载 SNAP 归档并调 `jevlike-data wikispeedia` |
| `examples/film/make-film.sh` | 14 | `build-film.mjs` → `render-film.mjs` 两步 |
| `examples/film/build-film.mjs` | 93 | 从 trace 选窗口（或接受指定窗口）、生成 `replay.html` |
| `examples/film/render-film.mjs` | 20 | Playwright 驱动 `setFrame(t)`、24 fps 管道给 ffmpeg |
| `tests/test_smoke.py` | 23 | tiny scorer 能收敛 + 概率归一 |
| `examples/*/test_smoke.py` | 41 + 44 | 视觉前向的形状、12 行表的选择、teacher walk 复演 |

---

## 13. 「算得出来 ≠ 接进来了」：这一篇有三条

三条性质各不相同，且**都写在明处**，不是靠读者从缺口反推。

**13.1 `PlainConvPolicy` 在仓内产物上不可达。** `vision.py:130-178` 是一个明确的架构对照臂
（*Architecture control: the same visual stem followed by a flat MLP*），
`audit.py:72-74` 会在 `payload.get("model_type") == "plain_conv"` 时构造它。
但**全仓没有任何一处写 `model_type`**（grep 只命中 `audit.py:72` 这一处读点），
也没有任何脚本能产出 `PlainConvPolicy` 的检查点。于是这个分支在本仓的任何产物上**都不可达**，
`AGENTS.md:62` 那条「扁平卷积策略在同样的 DAgger 数据上学得更快」的结论
因此**不在本仓的复现路径上**——它是一次已经做过、但没有留下可跑通路的实验。

**13.2 `examples/chess/stem.py` 的 `widen()` 零调用点，且它的自陈与代码不符。**
`widen(model)`（`stem.py:27-29`）把三个膨胀卷积块接到 stem 后面，用来把每个 token 的感受野
从约 21 px 拉到约 133 px（`stem.py:3-5`）。但：①**全仓没有任何地方调用 `widen`**；
②模块 docstring 最后一句是 *checkpoints record it as `payload["stem"] = "wide"`*，
而**全仓没有任何地方写或读 `payload["stem"]`**（grep 零命中，`chess/train.py` 的 `save()`
只写 `version`/`model`/`width`/`rank`/`chess_option_ids`/`steps`/`history`，`:42-44`）。
所以这一句不是「写了没接线」，而是**接线描述本身在 HEAD 上不成立**——
引用它的时候不能把这句话当成事实。

**13.3 `config["width"]` 在 hf 分支从不被读。** `make_system` 的 hf 段（`model.py:115-125`）
只用 `config["hf_model"]`、`config["rank"]`、`config["context_tokens"]`、`config["option_tokens"]`；
头的输入宽度取 `self.encoder.config.hidden_size`（`model.py:73`）。
`--width` 照样能传、照样被写进 `config`（`train.py:52-56`）与检查点，
但在冻结编码器这条路上**改变不了任何行为**——它是 tiny 分支的专属旋钮。
这一条的危害比前两条小（改它不会得到错数），但它会让「我刚把 width 调大了」这种判断失效。

三条都属于 `docs/comparison.md` §4.5 那张表的「写了没生效」大类，
但各自的成因不同：13.1 是**读者在、写者从没有过**，13.2 是**描述与代码脱节**，
13.3 是**同一份 config 服务两条分支、只有一条读它**。

---

## 14. 亲手跑一遍

### 14.1 上游给的命令（**本文没有执行**）

```sh
uv venv && source .venv/bin/activate && uv pip install -e '.[dev]'
jevlike-data synthetic --output data/synthetic
jevlike-train data/synthetic/train.jsonl --validation data/synthetic/validation.jsonl \
  --output runs/synthetic.pt
jevlike-eval runs/synthetic.pt data/synthetic/test.jsonl
jevlike-predict runs/synthetic.pt \
  --context "Choose the exact badge amber badger. Badge: amber badger." \
  --option "azure crane" --option "amber badger" --option "gold heron"
```

（`README:54-69`；冻结编码器那条是 `--encoder hf --hf-model Qwen/Qwen2.5-0.5B --rank 256`，`README:87-96`。）

顺带一条不用跑就能看出的呼应：quickstart 里那句 `--context` 与 `synthetic_example`
的模板**同一句式**（`data.py:124` 是 `f"Choose the exact badge {target}. Notes: {notes}. Badge: {target}."`，
quickstart 省掉了中间那段 `Notes: …`）——两条的共同点是**答案字符串在 context 里出现两次**，
所以这个合成任务本质上是字符串匹配，`README:117` 的「约 98%」也就在这个意义上才成立。

**本文为什么没跑**：本机 Python 3.14.7 **没有 torch**（`import torch` 直接 `ModuleNotFoundError`），
没有 Stockfish（`which stockfish` 为空），也没有 GPU。加上本文遵守的一条自律：
**不运行上游程序**。所以下面三条是本文全部的「实测」，都是**不跑上游代码**的那一类。

### 14.2 实测 A：容器层算术（不解 pickle）

对三个 committed 检查点用 `unzip -l` 取逐条长度。以 `examples/checkpoints/joint-imitation.pt` 为例
（`unzip -l` 的条目长度 ÷ 4 = 该张量的 float 数）：

| zip 条目 | float 数 | 与哪一层的形状对得上 | 依据 |
|---|---|---|---|
| `data/0` | **2560** | `positions`：`position_2d(8,10,32)` = 8×10×32 | `vision.py:20-30`、`:63` |
| `data/1` | 1600 | `stem.0.weight`：16×4×5×5 | `vision.py:49` |
| `data/2` | 16 | `stem.0.bias` | 同上 |
| `data/3` / `data/4` | 16 / 16 | `stem.1.weight` / `.bias`：`GroupNorm(4,16)` | `vision.py:50` |
| `data/5` | 4608 | `stem.3.weight`：32×16×3×3 | `vision.py:52` |
| `data/6` | 32 | `stem.3.bias` | 同上 |
| `data/7` / `data/8` | 32 / 32 | `stem.4.weight` / `.bias`：`GroupNorm(4,32)` | `vision.py:53` |
| `data/9` | 16384 | `patch.weight`：32×32×4×4 | `vision.py:56` |
| `data/10` | 32 | `patch.bias` | 同上 |
| `data/11` | **384** | `options.weight`：**12 × 32** | `vision.py:57` |
| `data/12`–`data/15` | 32 ×4 | `head.context_norm` / `head.option_norm` 的 weight+bias | `model.py:17-18` |
| `data/16`–`data/18` | 1024 ×3 | `head.query` / `key` / `value`：32×32 且 `bias=False` | `model.py:19-21` |
| `data/19` / `data/20` | 32 / 32 | `value_head.0`：`LayerNorm(32)` | `vision.py:62` |
| `data/21` / `data/22` | 32 / 1 | `value_head.1`：`Linear(32,1)` | 同上 |

合计 **23 个张量、29 009 个 float = 116 036 字节**；zip 里所有条目的长度和是 **121 740**，
差额 **5 704** 恰好等于 `data.pkl`（5 653）+ `.format_version`（1）+ `.storage_alignment`（2）
+ `byteorder`（6）+ `version`（2）+ `.data/serialization_id`（40）。
也就是说**这个检查点里没有优化器状态、没有 EMA、没有任何额外键**——
`train.py:90` 只存 `{"config", "state_dict"}`，而 `trainable_state` 只挑 `requires_grad` 的参数
（`model.py:129-133`）；这里更进一步：`train_joint.py:26-35` 存的是 `model.state_dict()` 全量。

同一套算术用在另外两个检查点上，**只有 `options.weight` 一个数变了**：

| 检查点 | `data/11` float 数 | 推出选项行数 | 与哪段代码吻合 |
|---|---|---|---|
| `examples/checkpoints/joint-imitation.pt` | 384 | **12** | `actions=TOTAL_OPTIONS`（`train_joint.py:28-33` 的 `save`） |
| `examples/chess/checkpoints/chess-dagger1.pt` | 384 | **12** | `train.py:62` 用 `TOTAL_OPTIONS` 建模型 |
| `examples/doom/checkpoints/deadly-dagger.pt` | **224** | **7** | `DoomScorerV2(actions=7, …)`（`train_dagger.py:67`） |

这三个数正是 `load_expanded`（§11.2）与 `play.py`、`audit.py` 里
`payload["model"]["options.weight"].shape[0]` 那一行所依赖的前提（`play.py:40`、`audit.py:76`）。

### 14.3 实测 B：payload 字符串给出的谱系

`data.pkl` 里的**键名与字符串**可以用 `strings` 读出，不需要解 pickle。在 `joint-imitation.pt` 里：

- `actions` 是 12 个名字，顺序为 `turn left, turn right, move forward, move backward,
  strafe left, strafe right, attack, move up, move down, move left, move right, pick up / put down`
  ——与 `JOINT_ACTIONS`（`train_joint_ppo.py:24-25`）**逐字相同**；
- `scenario` = `deadly_corridor`；
- `source_doom_checkpoint` = `examples/doom/checkpoints/deadly-dagger.pt`；
- `training_config.method` = `alternating on-policy DAgger CE; no PPO`
  ——与 `train_joint.py:33` 的字面量相同，**因此这个检查点出自 `train_joint.py`，不是
  `train_joint_ppo.py`**（后者的 `save` 写的是 `joint_cycles` 而没有 `training_config`，`:229-238`）。

再叠上权重上的算术，谱系就闭合了。把三个检查点的 `data/11` 逐行读出来（标准库 `struct`，
`<f`，32 个 float 一行），算行间余弦与最大逐元素差：

| 比较 | 最大逐元素差 | 余弦 |
|---|---|---|
| `joint` 的第 0–6 行 vs `deadly-dagger` 的 7 行 | **0.00312** | **0.999998** |
| `joint` 的第 7–11 行 vs `chess-dagger1` 的第 7–11 行 | **0.00285** | **0.999997** |
| `chess-dagger1` 的第 0–6 行 vs `deadly-dagger` 的 7 行 | 3.89789 | 0.092946 |

三行的含义各不相同：前两行说明这个 12 行表**几乎就是两份源检查点各自那一段的拼接**——
每个元素的行范数在 4.56–5.84 之间，而改动幅度小于 0.004，即**相对改动 ≲ 0.06%**；
第三行是负对照，说明象棋那份检查点的 0–6 行与 Doom 的表**毫无关系**（它是一次象棋专训，
ids 0–6 没有被任何数据碰到，`examples/chess/train.py:30` 只传 `CHESS_OPTION_IDS`）。

这条实测支撑了两句上游自述：`examples/doom/README.md:29` 的 *The joint checkpoint has the same
weights and one 12-row option table for both games. It is an early checkpoint, not a strong
player.*，以及 `:70` 的 *The short released joint checkpoint lost much of the chess controller*
——**「early」在这里可以量化**：那个 20 更新量级的交替训练几乎没动选项表。

### 14.4 实测 C：文件哈希与一处设备标记

| 文件 | 字节 | sha256 |
|---|---|---|
| `examples/checkpoints/joint-imitation.pt` | 127 494 | `173ca4583c7a5d341898062d966043ff608cae4da10c34bcdac19926dd664f79` |
| `examples/doom/checkpoints/deadly-dagger.pt` | 125 964 | `96f0e78e4872eb70b5e12e322d09e4eae451011a6a3449781e50214365cb5970` |
| `examples/chess/checkpoints/chess-dagger1.pt` | 126 796 | `6d9a4828abb3106f838cd0d63d6a116607233e6e16420cb3025b3429f758b271` |

**这三个哈希没有任何清单可比对**——上游没有 `SHA256SUMS`、没有 `MANIFEST.json`
（Kev、Nimble、OpenJev(HF) 那几篇的哈希对照在这里没有对象）。它们的作用只是身份：
任何声称「用发布的检查点跑出来的数」都必须先对上这三个值。

一处顺带的观察：`deadly-dagger.pt` 与 `joint-imitation.pt` 的 storage 位置字符串是 `cpu`，
而 `chess-dagger1.pt` 是 **`mps`**（都写在 `data.pkl` 里）。这与两组上游命令的设备选择一致：
Doom 的例子命令用 `--device cpu`（`examples/doom/README.md:22-26`、`:36-45`），
象棋的命令用 `--device mps`（`examples/chess/README.md:34`、`:58`）。
三个读取点都带 `map_location`（`play.py:36`、`audit.py:71`、`chess/eval.py:252`、`chess/play.py:77`），
所以这个标记不影响加载——但**不带 `map_location` 加载 chess 那个会在无 MPS 的机器上失败**。

---

## 15. 实例走查：committed 三个检查点能说什么、不能说什么

**能说的**（上面三节已给）：容器结构、参数量、选项表行数、两个游戏的选项 id 划分、
这份 12 行表是从哪两份检查点拼出来的、以及那次联合训练对选项表的改动量级。

**不能说的**：任何准确率、任何概率、任何 ECE、任何对局结果、任何延迟。
原因是**上游没有提交任何原始记录**：`runs/` 在 `.gitignore:7`，`data/` 也在（`.gitignore:6`），
`examples/*/README.md` 里的每一个数（Doom 的 0.60 kills / −97.50 reward、象棋的
4/46/0 与 0/2/48、99.95% 读格子、94% 有标记走子、65% 下一步键、+7 点小方块、24%→4% wasted）
都**没有对应的 jsonl 或 json 随仓提交**。这与 CLM（提交了 `examples/t_rex/results/*.json`）、
Kev（提交了 `rows.json` / `report.json` 与发布清单）、decider（提交了 `tests/data/*pins.json`）
那三篇的处境不同——**这一篇的「实测」只能停在权重与容器这一层**。

若要复现那些数，上游给的路是：Doom 侧 `play.py --episodes 10 --game-seconds 60 --greedy`
（`examples/doom/README.md:22-27`，其中 60 秒 = 60×35/4 ≈ 525 步），
象棋侧 `eval.py --games 50 --sample`（`examples/chess/README.md:34`，KEY_BUDGET=40、
MAX_PLIES=200 在 `chess/eval.py:32-33`）。这两条都需要环境与权重，本文没有跑。
`examples/doom/README.md:17-27` 的注释还说明了一处口径：*Evaluation is paced at 8.75 decisions per
second*，即 `TICS_PER_ACTION / TICS_PER_SECOND = 4/35`（`environment.py:12-13`），
与 `examples/chess/play.py:30` 的 `DECISION_SECONDS = 4/35` 是同一个常数。

---

## 16. 本文不做什么

- **不跑上游代码**：没有 `import jevlike`、没有 `import torch`（本机也装不了）、没有跑训练、
  评测、游玩或审计，没有下载任何权重或分词器，没有 Stockfish。
  本文的「实测」只有三类：**zip 条目长度的算术**、**`data.pkl` 里的键名字符串**、
  **权重张量上按 32 float 一行的标准库算术**（全部不解 pickle）。
- **不给前向产生的数值**：概率、准确率、ECE、延迟、注意力熵的实际取值一律不填。
  凡是需要权重或环境才能得到的量，本文只说机制或给 `文件:行`。
- **未核实的四处**（明确列出，不推测填空）：
  ① `Qwen/Qwen2.5-0.5B` 的 `pad_token_id` 是否为 `None`，以及「EOS 被 `context_mask` 屏蔽」
  这条副作用在 hf 路径上的实际后果（其 §2.4）；
  ② `README:117` 的 26% 用的是哪一套 `--context-tokens`（字节还是 token 预算，其 §2.3）；
  ③ 冻结编码器路径上 `tokenizer.model_max_length` 与 `--context-tokens 192` 的相互作用
  （默认 192 远小于主流编码器的窗口，但这是上游的默认值，本文没有核对任何具体模型的
  `model_max_length` 会不会被 `truncation` 之外的机制影响）；
  ④ 视觉侧 `position_2d` 的宽高含义是否与 ViZDoom 的 `screen_buffer` 轴序在**所有**抓取分辨率下
  一致——本文只核对了 160×120（`examples/chess/data.py:192` 的断言）与
  `play.py:71-74` 那次显式 resize 之后的路径。
- **不评价选型**：本文只写这条读法怎么工作、代价落在哪里。
- **不复述收录约定**：约定在仓库 [`README.md`](../../README.md) 的「收录约定」一节。

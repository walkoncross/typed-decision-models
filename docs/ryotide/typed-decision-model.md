# RYOTIDE 是如何把本地 LLM“读”成类型化决策模型的

> 代码库：<https://github.com/csabag/ryotide>
> 对应 README 第 3 行：*Reading a local LLM as a typed decision model.*
> 本文按 `src/ryotide/` 的实际代码逐环节说明这句话的实现方式；文中出现的文件名与行号（`classify.py:42`、`bench/fit_temperature.py` 等）均相对该仓库根目录。
> 文中出现的模型简称（「Gemma 4 E4B」「Qwen3.5-9B」「Qwen3.5-4B」）对应 Hub 上的 `google/gemma-4-E4B-it`、`Qwen/Qwen3.5-9B` 与 `Qwen/Qwen3.5-4B`——就是 `server.py:57-62` 那三个 preset 各自钉住的模型（本仓 `docs/comparison.md` 的「基础模型」一行用这套全名）。

## 0. 这句话的四个词分别落在代码的哪里

| 词 | 含义 | 代码位置 |
|---|---|---|
| **Reading** | 只前向读一次，不写、不训练、不改权重 | `classify.py`、`torch_backend.py`（权重全程 frozen） |
| **a local LLM** | 现成的公开权重（Qwen3.5 / Gemma 4），跑在本机 MLX 或 torch 上 | `model.py`、`torch_backend.py`、`server.py:56` 的 `PRESETS` |
| **as** | 不是“训练成”，而是**重新解读**同一份权重：同一套 KV cache 既能生成，也能判类 | `classify.py`、`generate.py`（共享 weights 与 state） |
| **typed decision model** | 输出被**封闭**在一个已知标签集上：不是文本，而是一个概率分布 | `classify.py:42` `LabelSet`、`jevbench_adapter.py` 的 marker 掩码 |

关键立场：普通用法把 LLM 当生成器（自回归解码 → 文本 → 解析）；RYOTIDE 把**同一个模型**当分类器——**一次前向、只读最后一个位置的 logits、掩码到固定标签集、重归一化**。`classify.py` 的模块 docstring 把整个技术概括成三步：

1. 把共享 state 预填充进 KV cache（只做一次，所有问题复用）
2. 用一段短问题扩展 cache，**只保留最后一个位置**的 logits
3. 把这些 logits 掩码到固定标签 token 集合，再 softmax 重归一化

第 3 步就是“分类器而非生成器”的分界线：输出分布封闭在已知标签集上，**没有东西需要解析，也没有东西可以幻觉**。

## 1. 端到端数据流

```
                    ┌──────────────── server.py ────────────────┐
POST /v1/systemone  │ tasks_from_request → 每个 question 一个 task│
{state, questions}  │ 标签从问题本身重建（wire 不传 labels）      │
                    └───────────────┬───────────────────────────┘
                                    v
        ┌────────────────── jevbench_adapter.py ──────────────────┐
        │ _prompt()    state + 选项块(带标记) + 指令 (+ echo)      │
        │ _tokens()    chat template, enable_thinking=False,      │
        │              末尾拼上 answer prefix ("Answer: **")       │
        │ run()        共享前缀预填充 → 每个 order 分支 → 读 → 平均 │
        │              温度 softmax(z/T)  ◀── §9.2：唯一改发布数值的步骤│
        └───────────────┬─────────────────────────────────────────┘
                        v
        ┌────────── classify.py / torch_backend.py ───────────┐
        │ prefill()/extend()   分块预填充共享 state            │
        │ logits_at_end()/_prefill()  只取最后一位置的 logits   │
        │ _decide()/read()     token_ids 掩码 → softmax → 概率  │
        └───────────────┬──────────────────────────────────────┘
                        v
        probs over labels → answer_for() → 决策 JSON（无生成 token）
```

产出概率的完整链条（含名字撞车的三处 “calibration”）见 §9。

## 2. “typed”从哪来：标签集必须能落到单 token

真正让输出“类型化”的约束只有一个：**标签必须能在读位置被当作一个 token 打分**。

- 通用路径 `classify.py:42` `LabelSet.resolve()`：对每个标签依次尝试 `"{}" / " {}" / "{capitalized}" / " {capitalized}"`，要求 `tokenizer.encode(表面形式)` **恰好长度为 1**，且不同标签不能撞到同一个 token id。否则抛 `MultiTokenLabelError`，并在错误信息里直接建议换标签词（`yes`/`no`、`A`/`B`）。
- 单 token 不是风格选择，而是**硬约束**：决策在单个位置上读出，跨两个 token 的标签无法在该位置评分（要评分就得解码一步，而解码正是要避开的东西）。

### 但真实标签往往不是单 token

JevBench 的标签是 `change_address` 这类多 token 串，无法直接评分。适配器的做法（`jevbench_adapter.py:8-18`）是把标签映射成**字母标记 A、B、C…**，再对标记做 softmax：

```
state                    → KV-cache 预填充（在多个 option order 之间共享）
instructions + criteria  → 选项块，每个选项一个字母标记
labels                   → A, B, C ...（单 token）
answer                   → 在 ONE position 上对这些字母的 softmax
```

分布始终是模型自己在标记上的 softmax，**从不被 verbalize，也从不从 logprob 文本里解析**。

标记的 token id 用 `_token_after(context, piece)`（`jevbench_adapter.py:387`）在**上下文里**求：`encode("Answer: **" + "A")` 与 `encode("Answer: **")` 恰好差一个 token 才算数。原因是带 dummy prefix 的 SentencePiece 分词器（Llama-2 系）把裸 `A` 编成 `▁A`，而在 `Answer: **` 之后模型写的是裸 `A`——单独编码会读到一个模型永远不会发出的 token（marker mass 0.000）。

### 超过 26 个选项

- ≤ 26：单字母 A–Z，与 v0.1 逐位一致。
- \> 26（默认 `digits`，`_uses_codes`/`_read_codes`，`jevbench_adapter.py:288-338`）：两 token 代码 `A0…Z9`，上限 260（高于 TypeSafe API 的 255 上限）。一次 prompt，先在答案位置读字母，再把每个字母接到预填充 cache 的**一条分支**上读数字：

  `P(option) = P(letter) × P(digit | letter)`

  这是**精确**的，且和为 1；marker mass 变成“发出任一合法代码”的全词表概率。
- \> 26 的 `grouped` 读法（`_read_grouped`，`jevbench_adapter.py:343`）：同组选项共用一个字母（`A A A B B B…`，组大小 `⌈n/26⌉`），先选最可能的若干组（累计到 0.95 质量或 26 个选项为止），再用字母重问这些组；被丢弃的组按组内均分保留第一遍份额——**这是近似**，与 `digits` 不同。

## 3. 读位置：先标定，再读

“读”的成败几乎全在**读哪个位置**。适配器不逐个模型特判，而是**探测后择一**：

- 候选答案前缀 `SUFFIXES`（`jevbench_adapter.py:218`）：`Answer:`、`Answer: **`、`**`、`Answer:\n`、`The answer is`、`Answer: (`、空串，以及 harmony 格式的 `<|channel|>final<|message|>` 系列——gpt-oss 必须先打开输出通道，否则下一个 token 是 `<|channel|>` 而不是任何选项标记。
- 候选标记表面形式 `PATTERNS`：`" {}"` 与 `"{}"`。二者**耦合**（`Answer:` 之后下一个 token 是 ` A`；`Answer: **` 之后是 `A`），所以必须成对扫描。
- `_calibrate_read_position()`（`jevbench_adapter.py:233`）在**若干条真实任务**上求平均，选 marker mass 最高的组合；`CALIB_TASKS = 6`。注释记录了一条重要教训：**同一条 prompt 单独测得 0.9988、跨集合却是 0.0001**，所以必须在多条、且选项数不同的真实 prompt 上取平均才可信。
- 打平时**取最短前缀**：长前缀每次决策都要多预填充，却没有任何收益（否则某些模型会用 9 个 token 的 harmony 前缀，而 3 个 token 的读法完全相同）。
- 标定在**计时之外**完成：`calibrate()` 由 `bench/run_jevbench.py:86` 在跑分前调用；`server.py` 则在启动 warm-up 时完成。之后每次决策都复用同一读位置，因此结果可复现、可比较。
- `pin_prefix` / `pin_marker` 用于 A/B：把读位置钉死，保证实验只改一个变量。

本节说的就是 §9.1 那个运行期的 `calibrate()`——它**只选读位置**，与温度、与 bias 都无关。

## 4. 一次前向：只读最后一个位置

这是性能与正确性的核心，也是 README 里两个 bug 中“整序列 logits”那条的来源。

### MLX 路径（参考实现）

`classify.py:145` `logits_at_end(cache, tokens)`：

- `tokens[:-1]` 按 `PREFILL_CHUNK = 512` 分块喂进去，每块后 `mx.eval` 落地；
- 只有**最后一个 token 单独跑一次**，取 `[0, -1, :]`。

为什么要分块：`[1, n_tokens, vocab]` 在 Qwen3.5 的 248k 词表下，3.7k token 的 prompt 会产生约 3.7 GB 的 logits，而只需要读其中一行——长 state 上直接 OOM，一般长度上也是巨大的延迟尾部。

该方法**有意原地修改 cache**，因此判完可以直接继续生成；需要保持 state 干净的场景用 `branch.py`。

### PyTorch 路径（CUDA / MPS / CPU）

`torch_backend.py:164` `_prefill()` 同样分块（`PREFILL_CHUNK = 2048`），并用 `logits_to_keep=1` 让 transformers 只保留最后一行 logits；峰值显存 = 权重 + cache + 一个 chunk 的激活，而不是整序列前向（实测 Gemma 4 E4B 单次整段前向每 1k token 涨 0.24 GB，24 GB 卡在 32k token 处 OOM）。分块在算术顺序之外是**精确**的：模型看到的是相同位置上的相同 token。

`TorchClassifier.read()`（`torch_backend.py:277`）返回两样东西：

- `probs`：**只在标记 token 上**做 softmax 并重归一化；
- `mass`：掩码**之前**，全词表 softmax 落在这些标记上的份额。

## 5. 掩码 + 重归一化：决策本身

`classify.py:96` `_decide()` 就是全部：

```python
sel = logits_row[mx.array(labels.token_ids)].astype(mx.float32)   # 取标签位置的 logit
raw = [...sel.tolist()]                                          # 留原始 logit 备查
if bias is not None: sel = sel - bias                            # 上下文校准偏置
probs = mx.softmax(sel)                                          # 只在标签集上归一化
```

产出 `Decision`：`label`、`index`、`probs`、`margin`（top − runner-up，可用的置信信号；与 `confidence` 哪个排序更好见 §9.6）、`entropy`、`raw_logits`、`calibrated` 标志（只有减过 bias 才为 True，见 §9.3）。

> 注：上面 `bias` 那一支在 benchmark / server 路径上从不触发——适配器不走 `_decide()`，
> 而是自己做掩码 softmax。详见 §9.3。

适配器侧的 `_read()`（`jevbench_adapter.py:273`）做同一件事，只是返回 `(probs, mass)`；两个后端在这里分叉，其余所有决策塑造步骤（prompt、chat template、读位置、选项标记、标签折叠）都是共享的。

### marker_mass：让“读错位置”无法静默发生

`marker_mass` = 全词表分布落在标记上的份额，被写进每条结果的 `res.raw["runtime"]["marker_mass"]`（`jevbench_adapter.py:685`），`server.py:178` 又把它冒泡到响应的 `runtime` 里。

README 记录的 bug 正是被这个数字抓住的：Qwen3.5 的生成模板会打开 `<think>` 块，读位置落进推理过程，只有约 15% 的概率质量到达选项标记——**适配器照样返回了一个格式正确的分布，harness 也照样开心地打了分**。改成 `enable_thinking=False`（`_tokens()`，`jevbench_adapter.py:526`，模板不接受该参数时回退）后分数从 27.4 → 61.8。现在 `marker_mass` 让这类错误不可能再静默发生：接近 1.0 说明 prompt 把读位置落在了真实答案上，低值说明决策是从噪声里重归一化出来的。

## 6. 顺序去偏：在共享前缀上开分支

JevBench 自己的 README 记录过 open-alternative-jev 在 `A. yes, B. no` 上得 72%、顺序反过来只有 21%——**选项顺序偏置是这个小模型规模下的主要失效模式**。因此默认在多个 label order 下评估同一条决策，再对每个标签的概率求平均。

- `_orders(n, k)`（`jevbench_adapter.py:83`）：恒等、逆序，然后旋转，最多 k 个。
- 顺序平均**几乎免费**的原因在这里：所有 order 共享同一段 state 前缀，所以 state 只预填充一次，每个 order 只是从该 cache 上分出来的一支（`fork_cache`）。state 是昂贵的那部分，选项块只有几十个 token。
- 实现：`run()` 先求所有 order prompt 的**公共 token 前缀长度** `shared`，`clf.prefill(prompts[0][:shared])` 一次，然后每个 order `fork_cache(base)`；结束时把 slot 概率**折叠回真实标签下标**（`totals[li] += p[slot]`）再归一化——注意这是概率平均，且不同 order 的 slot 指向不同标签，所以必须折叠。
- `shared = max(0, shared - 1)`：留一个 token 让分支去扩展。

温度也在这里生效（`jevbench_adapter.py:625`）：在**已有的** softmax 概率上重做 `softmax(z/T)`，数学上等价于 `softmax(logit/T)`，且**不改变选项排序、只改变分布的陡峭程度**。T 是逐 preset 钉死的常量，由 `bench/fit_temperature.py` 离线拟合、并在留出集 ECE 变好时才采纳（详见 §9.2：Gemma 1.924、Qwen9 0.667 采纳；Qwen3.5-4B 拟合出 1.592 但留出集变差被拒，保持 1.0）。

## 7. 多问题共享一次 state

一个 TypeSafe 请求可以针对同一 state 问多个问题（Decision Index 的 ToolRet 平均每个请求约 192 个问题）。`run_many()`（`jevbench_adapter.py:472`）：

1. 对每个问题构造 prompt tokens；
2. 求所有 prompt 的公共前缀长度 `plen`（并 `min(plen, shortest - 1)`，保证每个问题至少留一个自己的 token）；
3. `plen < MIN_SHARED_PREFIX (32)` 就退回逐个独立决策——共享带来的收益不值得那点记账；
4. 共享头预填充一次（MLX `clf.prefill`，torch `clf.extend`），每个问题在**自己的分支**上只处理自己的 token；
5. 计费上把共享 state **只算一次**（除第一个问题外逐个扣掉 `plen`）。

torch 路径还有一层加速：`read_batch()`（`torch_backend.py:231`）把若干问题的后缀**右填充成一个 batch**，在共享前缀 cache 的副本上一次性前向。因为模型是因果的，某问题最后一个真实 token 永远看不到它之后的 padding，所以不加 attention mask 也是精确的；`logits_to_keep` 只取各问题自己的最后位置。分组同时受 `BATCH_TOKENS`（填充 token 预算）和 `BATCH_PREFIX_TOKENS`（cache 副本数）约束，放不进任何组的问题退回单条 `read()`。`_repeat_cache()` 处理了一个坑：transformers 的 `batch_repeat_interleave` 覆盖不到线性注意力层（Qwen3.5 的 gated delta 层保留 conv 与 recurrent state），两边都得手做。`RYOTIDE_NO_BATCH=1` 可关闭批量以做 A/B。

## 8. KV cache 的三种分支策略

`branch.py` 把“一次读 state、N 个问题复用”的三种内存/能力权衡写清楚了：

| 策略 | 做法 | 内存 | 能否继续生成 |
|---|---|---|---|
| `fork` | 每个问题 fork 一份 cache，全部保活 | N × prefix | 任意分支立即可续写 |
| `trim` | 一份 cache，每个问题读完把写头回退到共享 state 末尾 | 1 × prefix | 不能，分支不存活 |
| `lazy`（默认） | 先按 `trim` 判类，只对你真正想续写的那一支 fork + replay | 1 × prefix（按需 +1） | 需要时 replay 一次短问题 |

默认 `lazy`：一个问题只有 10–30 个 token，重跑一次远比常驻 N 份完整前缀便宜。

两个**正确性**细节，注释里都是踩过坑的结论：

- `fork_cache()`（`branch.py:33`）对 `ArraysCache`（Qwen3.5 的线性注意力层，固定大小 recurrent state、没有 offset 可回卷）只复制列表——其槽位是写时重绑而非就地修改，所以列表拷贝就足以让兄弟分支互不影响。
- 对 `RotatingKVCache`（Gemma 4 **E4B** 的 **42 层里有 35 层**滑动窗口，`google/gemma-4-E4B-it` 的 `config.json`；上游 `branch.py:53` 的注释写的是「20 of 24」，与配置不符）必须**给每个分支自己的 buffer**：环形缓冲一旦绕回，`state` 交回的就是 buffer 本身，单 token 写入会**就地**落进父级，copy-on-write 的 fork 会把父级写花。便宜是因为它被窗口（512 token）限界。
- `trim_to()` 在任何层不可回卷时**直接报错**，而不是静默 no-op：recurrent state 没有 offset 可退，静默什么都不做会把一个问题泄漏进下一个——正是这套设计要避免的交叉污染。

`generate.py` 则演示了模式 2：判类之后，分支 cache 里已经有 state + 问题，把结果注入（json / prose / bare 三种格式，`ab_injection()` 让格式选择成为一次测量而非争论）再解码，代价只有注入 + 解码。

## 9. 置信度从哪来，又怎么被度量

置信度是这个项目的一等交付物：README 每个 entry 一行 `hard ECE`，看板上 `calib` 是四个轴之一。
但**产生**它的机制在 `src/` 里散在四处，且四者共用 “calibration” 这个名字。这一节把它们串成
一条链，再补上最容易缺的一环——**测量**。

### 9.0 一条链

| 环节 | 代码 | 改发布概率的数值？ | 改 argmax？ | 已发布 preset 里生效？ |
|---|---|---|---|---|
| 读位置标定 | `jevbench_adapter.py:227-271` | 否（只改落在标记上的质量占比） | 否 | 是 |
| temperature | `server.py:56-63` + `jevbench_adapter.py:625` | **是**（唯一直接改数值的） | 否 | 是（Gemma 1.924 / Qwen9 0.667） |
| contextual calibration bias | `classify.py:181-201` | 是（log 空间减先验） | 理论上可以 | **否，未接入** |
| 第二模型一致性重加权 | `bench/agree_boost.py` | 是（缩放 top label 的 odds） | **否，按构造成立** | 否，仅离线测量 |
| 校准的度量 | `vendor/jevbench/` + `bench/fit_temperature.py` | — | — | 用于评分与选 T |

最该看的是第三列：**发布出去的概率数值只被 temperature 改变**。其余环节要么不改数值
（读位置标定只改“多大概率质量落在标记上”），要么根本没接进已发布的路径。

### 9.1 读位置标定：只决定“读哪里”

`jevbench_adapter.py:227-271` 是全仓唯一在运行期跑的 `calibrate()`——扫 9 个答案前缀 × 2 个
标记表面形式，取平均 marker mass 最高者（机制、教训与全部调用点见 §3）。它**不碰 temperature、
也不碰 bias**，只决定模型被问在哪个位置，因此改的是落在选项标记上的质量占比，不是概率分布的
**形状**。结果记忆化在 `self._suffix` / `self._pattern`，跑分与启动时各调用一次。

### 9.2 temperature：逐 preset 钉死的常量，不是运行期标定出来的

`server.py:56-63` 直接写死：

| preset | temperature | 来源 |
|---|---|---|
| `ryotide-gemma` | 1.924 | 拟合后**采纳** |
| `ryotide-qwen` | 1.0 | 拟合过，**被拒** |
| `ryotide-qwen9` | 0.667 | 拟合后**采纳** |
| 无 preset 的默认（`server.py:247`） | 1.0 | — |

传递链：`PRESETS` → `--temperature` 的 default → `Engine.__init__`（`server.py:148`）→
`MlxJevLocalAdapter(temperature=…)` → `self.temperature`（`jevbench_adapter.py:169`）→
在 `run()` 里生效（`jevbench_adapter.py:625-631`）。**运行期没有任何地方求解它**：适配器里
只有一处 `if self.temperature != 1.0` 的应用，没有拟合。

应用是在**已经 softmax 过的标记概率**上重做一遍：

```python
z = [math.log(max(x, 1e-300)) / self.temperature for x in p]   # 等价于 softmax(logit / T)
```

因此它**只能改置信度、不能改 argmax**：任何 T 下选项排序相同，准确率完全相同。这正是它能
被单独拟合、并被允许在事后报告的原因。

`/health` 的 `prompt_hash` 里包含 temperature（`server.py:158`），所以 T 属于配置指纹的一部分，
尽管它不改动 prompt 的任何一个字节。

**真正的拟合在 `bench/fit_temperature.py`，离线，且只用合成数据：**

| 步骤 | 做法 | 位置 |
|---|---|---|
| 数据 | 只用 `data/synthetic/*` 上 `--temperature 1` 的 run；JevBench 条目一次都不参与 | `:5-9` |
| 划分 | 按场景序号**偶数拟合、奇数检验** | `:94-97` |
| 目标 | gold 选项的负对数似然（NLL） | `:46-47` |
| 搜索 | 在 `log T ∈ [log 0.2, log 20]` 上做黄金分割 | `:59-71` |
| 采纳 | **规则在跑之前就固定**：只有留出（奇数）半边的 ECE 下降才采纳，否则保持 T = 1 | `:110-114` |

三份报告正好把三种结局都演了一遍（`results/calibration/`）：

| 模型 | 拟合出的 T | 留出半 ECE（T=1 → T） | 决定 |
|---|---|---|---|
| Gemma 4 E4B | 1.924 | 0.124 → 0.070 | 采纳 |
| Qwen3.5-9B | 0.667 | 0.074 → 0.047 | 采纳 |
| Qwen3.5-4B | **1.592** | 0.029 → **0.084**（变差） | **拒绝，保持 1.0** |

所以 README 表格里 Qwen3.5-4B 的 `1.0` 并不是“没拟合”，而是**拟合过并被拒了**。
三个 T 各不相同的机制就在这里：**逐模型离线拟合 + 预先注册的接受/拒绝对照**，
不是运行期动态决定，也从不使用被报告的那批 benchmark 条目来定。

### 9.3 contextual calibration bias：算得出来，没接进来，而且试过没赢

`classify.py:181-201` `Classifier.calibration_bias()` 是 `src/` 里唯一叫 bias 的东西，实现的是
Zhao et al. 2021 的上下文校准：

```python
for cf in ("N/A", "", "[MASK]"):          # 三个无内容输入
    cache = self.prefill(state_tokens)    # 同一个 state
    q = self.encode(question_template.format(cf))
    row = self.logits_at_end(cache, q)    # 同一个读位置
    sel = row[mx.array(labels.token_ids)].astype(mx.float32)
    rows.append(mx.log(mx.softmax(sel)))  # 标签上的 log 概率
return mx.mean(mx.stack(rows), axis=0)    # 三者取均值
```

- **算出来的是“标签上的平均 log 概率向量”**：模型对无内容输入也会偏好某个标签，那是**先验**
  而非信号。
- **应用方式是 log 空间相减**：`sel = sel - bias`，在最终 softmax **之前**（`classify.py:100-101`）。
  减去平均 log 概率，等价于在概率空间里除以先验。
- 减过之后 `Decision.calibrated = True`。

**但它在实际跑的路径上是死代码**，三条证据：

1. `calibration_bias()` **全仓零调用点**。
2. **JevBench 适配器从不调用 `Classifier.classify()`** —— 它直接走 `clf.logits_at_end()` /
   `clf.read()` 自己做掩码 softmax（`jevbench_adapter.py:273-286`），所以 `_decide()` 里那句
   `sel - bias` 被整个绕过。§5 代码块里那行“上下文校准偏置”在 benchmark / server 路径上不会执行。
3. `bias=` 参数存在于 `Classifier.classify` 与 `BranchedClassifier.classify_many`
   （`branch.py:156,162`），但所有调用点传的都是 `None`。唯一真正算出 bias 的，是**已被取代的
   GLUE harness** 里的一段内联复制品（`glue.py:129-138` → `:150`），且藏在 `calibrate` 开关后面。

#### 实测：cc 试过，没赢

这不是一段闲置代码——`bench/probes/marker_probe.py` 就是它的实验（产出 `results/probes/marker.log`），
而且用的正是 `lp - prior` 这个与 `calibration_bias()` 相同的公式：

```python
lp, mass = read(a, t)
prior = mx.mean(mx.stack([read(a, t, cf)[0] for cf in ("N/A", "", "[MASK]")]), axis=0)
for arm, v in ((mname, lp), (mname + "+cc", lp - prior)):
```

3 种标记 × 2 种 cc 臂，40 条 hard（seed 1），Gemma 4 E4B 8-bit，`Answer: **` / `{}`：

| 臂 | acc | 标记质量均值 | 标记质量最小 |
|---|---|---|---|
| **letters**（A–P） | **0.675** | **0.9998** | **0.9967** |
| letters + cc | 0.600 | 0.9998 | 0.9967 |
| korean（16 个随机单 token 谚文音节） | 0.600 | 0.9854 | 0.8579 |
| korean + cc | 0.600 | 0.9854 | 0.8579 |
| far（每样本挑与 prompt 嵌入最远的字符） | 0.600 | 0.9210 | 0.3802 |
| far + cc | 0.550 | 0.9210 | 0.3802 |

两条读数：

- **cc 一次都没赢**：letters −0.075、far −0.050、korean 持平。n=40，这些差值只相当于 2–3 条题，
  不足以当结论；但方向与“它没被接入”一致，所以 §9.3 的死代码状态是**试过之后的结果**，
  而不是忘了接线。
- **标记必须是模型熟悉的符号**。故意换成与 prompt 嵌入最远的字符（平均最大余弦仅 0.115），
  标记质量均值从 0.9998 掉到 0.9210、最小掉到 0.3802，准确率同时下降。这解释了
  为什么默认用普通字母、超过 26 个选项时用 `A0`–`Z9` 这种字母+数字组合——而不是任意单 token
  符号：读位置是**学出来的**位置，标记越偏分布，它越不成其为一个答案位置。

> 结论：已发布 preset 的置信度来自 **temperature（9.2）**，不是来自这个 bias。
> 把 bias 接进来是一件**做过、且没得到支持**的事，不应读成流水线的一环。

### 9.4 另一种重校准：用第二个模型的一致性缩放 odds

`bench/agree_boost.py` 是第四种机制，性质与前三者都不同：**预测永不改变，只有置信度被重写**。
第二个模型与主模型一致就把 top label 的 odds 乘 k，不一致就除以 k：

```python
o  = q / (1 - q) * k        # q = top label 的概率
nq = o / (1 + o)
return {kk: (nq if kk == t else v * ((1 - nq) / (1 - q))) for kk, v in p.items()}
```

在 odds 上做缩放，保证分布仍然合法且单调；因为 argmax 不动，**intelligence 按构造固定**，
能动的只有 calibration。实测（primary=Gemma 4 E4B 8-bit，hard 集）：

| k（odds 因子） | hard ECE | calib 轴 |
|---|---|---|
| 1.0（只用主模型） | 0.298 | 40.4 |
| 1.5 | 0.295 | 40.9 |
| 2 | 0.303 | 39.4 |
| 3 | 0.285 | 43.0 |
| **5** | **0.274** | **45.2** |
| 8 | 0.322 | 35.6 |

有效但幅度很小，且 k 过大（8）反而比不重加权更差——这与 §9.6 那条“低置信不等于可翻转”的立场
一致：一致性只是一个弱的证据源。`bench/ensemble.py` 用同一个 calib 轴比较集成策略
（如“最高置信者胜”），是同一类实验。

### 9.5 校准怎么被测量

> **以下定义来自 `vendor/jevbench/`（JevBench 本体，MIT，vendored 在 `vendor/jevbench`），
> 不是 `src/` 的实现。** 列在这里是因为 README 的 `hard ECE` 与看板的 `calib` 列由它定义；
> 不读这一节，就看不出前面几个机制里哪个是主要矛盾。

- **ECE 本体**（`vendor/jevbench/jevbench/metrics.py:55` `ece_top_label`）：对 (置信度, 是否正确)
  对子在 [0,1] 上做 **10 个等宽分箱**，每箱取 |平均置信度 − 实际准确率|，按箱内样本数加权求和。
  返回 `ece` 与逐箱统计（边界、n、平均置信度、准确率）。`bench/fit_temperature.py` 用的就是这个
  函数，它也正是采纳 T 的判据（§9.2）。
- **评分侧的 calibration 轴**（`vendor/jevbench/jevbench/composite_v12.py:89`）：

  ```python
  def calibration(ece, mean_tvd=None):
      a = max(0.0, 100 * (1 - ece / 0.5))
      return a if mean_tvd is None else (a + 100 * (1 - mean_tvd)) / 2
  ```

  即 `100 × (1 − ECE/0.5)` 截断于 0（ECE = 0 → 100，ECE = 0.5 → 0），在能拿到 gold 分布时
  再加入一项到精确分布的 TVD。它取自 **hard tier**；只返回标签、不返回分布的系统没有这一项，
  在 JevBench Score 里 **calibration 记 0**。
- **它有多大权重**：JevBench Score 是 intelligence / calibration / speed / cost 四轴的
  **几何平均**，各 0.25，所以一个弱轴会狠狠拉低总分（`composite_v12.py:96`，缺轴按 0 处理、
  下限截到 1）。看板上 `calib` 列 40.4 / 73 / 83 这类数字就是上面那个式子的输出——
  `40.4` 对应 ECE 0.298，与 §9.4 表里第一行完全对得上。
- **本地评分只用 ECE**（`bench/score_jevbench.py:52` 传 `mean_tvd=None`）：gold 分布只随留出集
  发布，所以公开子集上算不了 TVD 项。这就是 README caveat 里“Calibration scores on ECE alone”
  的来源——**看板上我们那一列的 calib 与官方列并非同一个式子**，比较时要注意。

### 9.6 选择性预测：用置信度，而不是产生置信度

`selective.py`：`curve()` 按置信度排序给出风险-覆盖曲线，`aurc()` 算风险-覆盖面积，
`coverage_for_accuracy()` 求“仍能满足目标准确率的最大覆盖率”。两种排序分数：`confidence`
（top 概率）与 `margin`（top − runner-up，忽略剩余质量如何分布），哪个排得更好是逐模型的实证问题。

这里有一条被反复确认的立场：低置信度**不**意味着答案可以被翻转——准确率在置信度上单调、
且从不低于随机，所以反转低置信答案严格亏；低置信标记的是“不值得据以行动的决策”，
延迟它们才是奏效的干预。（§9.4 里 k=8 反而变差，是同一立场的另一个侧面。）

## 10. 服务化：让标准 harness 能驱动它

`server.py` 实现 TypeSafe 的 `POST /v1/systemone` 与 `GET /health`，因此 JevBench 用**原装 `typesafe` 适配器**就能驱动它，不需要任何 RYOTIDE 专属代码。

- **标签从问题重建**（`task_from_question`，`server.py:106`）——wire 上从不传 labels：`noul → ["no","yes"]`；`score → "0".."n-1"`（criteria 是 list，顺序即作者顺序）；`choice → criteria 的键`。
- **选项顺序**：JSON 对象的键序没有意义，所以默认把 choice 选项排成规范**自然序**：数字段按数值比较（`natural_key`，`6_credits < 9_credits < 12_credits`），其余按字母。数值量表因此读起来像量表，决策不再取决于客户端碰巧怎么排键。`--option-order received` 保留请求顺序。
- **`prompt_hash`**：把决策配置（orders、repeat、answer_prefix、marker_pattern、instruction、layout、temperature、option_order、markers）序列化后取 sha256 前 12 位，`/health` 暴露。两个后端在配置相同时报同一个 hash，评估方因此能确认实际服务的是哪套配置；temperature 在列说明**它是配置指纹的一部分，即使它不改动 prompt 本身**（§9.2）。
- **`instructions` 可以是对象**（v0.3.1，`instructions_text()`）：如 Decision Index 的检索问题 `{"task":..., "candidate": <document>}`，渲染成 `key: value` 行；字符串原样透传，所以字符串 prompt 及其 hash 不变。
- **并发**：`Engine.decide()` 用一把锁包住决策——MLX 不是线程安全的；单问题走 `run()`，多问题走 `run_many()`。
- **启动时 warm-up**：`main()` 先 `load()` 再打一条不计时的请求，把权重加载与首次编译排除在计时之外。
- **MCP**（`mcp_server.py`）：把决策 API 挂到 Model Context Protocol 上，暴露 `decide_choice` / `decide_yes_no` / `decide_score` / `decide_many` / `engine_info`，返回带校准概率的答案；给 `review_threshold` 时还返回 `confidence` 与 `needs_review`，用于把不可靠的决策路由给人（这是 §9.6 选择性预测的**消费端**：阈值比较发生在 MCP 层，`conf < review_threshold` 即 `needs_review`）。它**不加载模型**，只是一个薄客户端。

## 11. 拼在一起：一次决策的完整时序

```
1. 标定（跑分前/启动时，不计时）
   calibrate(pool) → 扫 SUFFIXES × PATTERNS，选 marker mass 最高的读位置（§9.1；不改概率形状）

2. 构造 prompt（每个 order 一份）
   _prompt():  instructions / "Options:" + "A. label - criteria" 行 / 指令
   + echo:     整个 question body 重复两次（state 只出现一次）
   + _tokens(): chat template(enable_thinking=False) + answer prefix

3. 预填充与分支
   公共前缀 prefill 一次 → 每个 order fork 一支

4. 读
   logits_at_end / _prefill：分块喂入，只有最后一个 token 单独前向
   _decide / read：logits[token_ids] → (可选减 bias，§9.3：此路未接入) → softmax → (probs, mass)

5. 折叠与平均
   每个 order 的 slot 概率折叠回真实标签 → 求平均 → 归一化
   （可选）温度重标定（§9.2）、(可选) idk 选项剥离后重归一化

6. 产出
   DecisionResult{probs, label, latency_s, usage{output_tokens: 0}, raw{runtime{marker_mass,...}}}
   server.answer_for(): noul → P(yes)；choice → 标签 + 概率；score → 整数等级
```

**全程 `output_tokens = 0`**：没有任何 token 被生成，这正是“读”而非“写”的字面含义。

## 12. 实例走查：同一个数据集上的两条记录，一条不超 26 选项，一条超过

下面用 `data/synthetic/many-options-v1.jsonl` 里的两条真实记录各走一遍完整处理过程。分界取在
**26 个选项**——这正是 `_uses_codes()` 切换标记方案、`_read()` 切换成 `_read_codes()` 的地方：

| | 实例 ① | 实例 ② |
|---|---|---|
| 记录 | `many-k020-000` | `many-k027-000` |
| 选项数 | 20（≤ 26） | 27（> 26） |
| 标记方案 | 字母 `A`–`T` | 两 token 代码 `A0`–`C6` |
| 读法 | 一次读，softmax 直接落在字母上 | 两步读，`P(字母) × P(数字 \| 字母)` |
| 正确答案 | `increase_travel_notice` | `open_mortgage` |

实例 ① 走完整六步（输入 → 标记 id → prompt → 分词与读位置 → 前向与掩码 → 结果）；实例 ② 只走
**与 ① 不同的那几步**（提示词、两步读、结果），其余步骤逐字相同。

数据来源：

- **分词与 token id** 由 `Qwen/Qwen3.5-9B` 的分词器实跑得到（本地缓存 revision
  `c202236235762e1c871ad0ccb60c8ee5ba337b9a`，与 `server.py` 的 `ryotide-qwen9` preset 同一版；
  离线加载，`local_files_only=True`，无网络）。复现时需显式指定带 `transformers` 的解释器
  （例如 `/opt/anaconda3/bin/python3`），裸 `python3` 可能落到没有该库的环境。
- **概率** 读自这两条记录在 `results/jevbench/many-gemma-default/results.jsonl` 中的**已存盘结果**
  （Gemma 4 E4B MLX 8-bit），不是本次重新前向得到的。该 run 覆盖 157/180 条（`many_k151` 仅 7/30），
  且 results 文件不记录输入文件名，所以这里只声称「这两条记录的结果如此」，**不**声称该 run 跑的是
  `many-options-v1.jsonl`。这两个 `task_id` 也出现在 `many-options-short.jsonl` /
  `many-options-mid.jsonl`，但同 id 记录的内容逐字段相同，因此数值的归属没有歧义。

### 实例 ①：`many-k020-000`，20 个选项（≤ 26 → 字母路径）

#### (a) 输入记录

```json
{
  "id": "many-k020-000",
  "state": "Customer message: \"Increase a travel notice. I checked my mortgage too.\"",
  "labels": ["cancel_email", "cancel_mortgage", ..., "update_transfer"],   // 20 个
  "expected": "increase_travel_notice",
  "question": {
    "type": "choice",
    "instructions": "Which intent does the customer message express?",
    "criteria": { "cancel_email": "The customer wants to cancel their email.", ... }
  }
}
```

注意 state 里埋了两件事（`Increase a travel notice` 与 `I checked my mortgage`），正确答案是前者——
这是一个需要真正读文档、不能靠先验猜的样本。

#### (b) 标签 → 标记：实际 token id

用该分词器直接验证：

| 步骤 | 实测结果 |
|---|---|
| `encode("Answer: **")` | `[15666, 25, 2972]`，解码回 `'Answer: **'` |
| 紧跟其后的 `A`…`Z` | **单 token，id 恰好是 `32`…`51`**（`decode([32..51])` = `A`…`T`…`Z`） |
| `_uses_codes(20)` | `False` → 走**字母**路径，`_letters(20)` = `[32, 33, …, 51]` |

于是 20 个 intent 标签被映射成 `A`–`T` 这 20 个**单 token id** 32–51——多 token 的标签
（`increase_travel_notice` 本身是多个 token）从此不再需要被评分，被评分的只有字母。

#### (c) prompt 的真实形态

调用真实的 `_prompt()`（非复述，实跑输出，中间省略）：

```
Customer message: "Increase a travel notice. I checked my mortgage too."

Which intent does the customer message express?

Options:
A. cancel_email - The customer wants to cancel their email.
B. cancel_mortgage - The customer wants to cancel their mortgage.
...
K. increase_travel_notice - The customer wants to increase their travel notice.
...
T. update_transfer - The customer wants to update their transfer.

Reply with the letter of the single best option.

Which intent does the customer message express?

Options:
A. cancel_email - The customer wants to cancel their email.
...
T. update_transfer - The customer wants to update their transfer.

Reply with the letter of the single best option.
```

三件事在这里同时可见：

1. **state 只出现一次**（它是昂贵的部分），而 question body（问题 + 选项块 + 指令）**整体出现两次**——
   这就是 echo（`repeat=2`，`repeat_min_options=0` 表示总是开启）。布局是 `S/Q/Q`：因为
   `question_first=False`，state 在最前，而不是 `Q/S/Q`。
2. `criteria` 的每一条被拼成 `标记. 标签 - 说明`，说明来自 `_criterion_for()`（choice 按标签取键）。
3. 选项顺序按 `order` 排列，最后统一以 `self.instruction` 收尾。

#### (d) 分词与读位置

- token 数：**实测 666**（Qwen 分词器）。同一条记录在 Gemma 分词器下的结果是 **709**——
  两者不同很正常，读位置只取决于各自分词器下 `Answer: **` 的最后一个 token。
- 读位置 = `_tokens()` 里 chat template（`enable_thinking=False`）之后**追加的 `Answer: **` 的末尾**。
  换句话说，模型被问到"下一个 token 是什么"，而这个位置正好是它会写下答案字母的地方。
- 该模板**确实接受** `enable_thinking=False`，实测生成前缀为：

  ```
  <|im_start|>user\nX<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n
  ```

  即思维块被模板**预先闭合**（空 `<think></think>`），所以读位置落在 `Answer: **` 之后而不是推理过程中间。
  这就是 §5/§13 里 "reasoning 模板" 那个 bug 的修复点：不加这个参数，读位置会落进 `<think>`，只有约 15% 的
  概率质量到达选项标记。

#### (e) 一次前向 + 掩码

```
logits_at_end(cache, tokens)  →  只有最后一个 token 单独前向，取 [0, -1, :]   （3.7k token 也不会物化整序列 logits）
row[ids]      ids = [32..51]  →  20 个字母位置的 logit
softmax(sel)  →                →  20 个标签上的概率分布（和为 1，无解析、无生成）
```

#### (f) 真实结果

```
model        mlx-community/gemma-4-e4b-it-8bit      predicted  increase_travel_notice   ✓ correct
top3         increase_travel_notice 0.999917 | update_transfer 5.1e-05 | dispute_travel_notice 1.1e-05
usage        {input_tokens: 709, output_tokens: 0}   latency 1.18 s
runtime      answer_prefix "Answer: **"   marker_pattern "{}"   markers "letters"
             question_repeats 2   orders_averaged 1   shared_prefix_tokens 0   marker_mass 1.0
             temperature 1.0
```

四处值得对照代码读：

- **`marker_mass: 1.0`**：全词表 softmax 的质量全部落在 20 个字母上，读位置干净——不是从噪声里重归一化出来的。
- **`orders_averaged: 1` 与 `shared_prefix_tokens: 0` 是同一件事**：该 run 用了 `--orders 1`，
  只有一个 prompt，`run()` 里 `len(prompts) > 1` 为假，所以没有可共享/可分叉的前缀，`shared` 保持 0。
  这正是 §6 顺序去偏的机制在只有一个 order 时的退化形态。
- **`temperature: 1.0` —— 这不是已发布 preset 的值。** `ryotide-gemma` preset 用 1.924（§9.2），
  本条 run 是配置默认值 1.0，所以上面那些概率**未经温度重标定**。影响范围可以精确说清：
  argmax 与准确率不可能变（§9.2 解释为什么），只有陡峭程度变——把这条记录的 20 个概率按
  `softmax(log p / 1.924)` 重算，`increase_travel_notice` 从 **0.999917 降到 0.981007**，
  次高者从 0.0001 升到 0.0058，预测标签不变。拿本文的数字与看板上的 `hard ECE` 相比时，
  务必先确认各自记录的 `runtime.temperature`。
- **`output_tokens: 0`**：整个决策没有生成任何 token。

### 实例 ②：`many-k027-000`，27 个选项（> 26 → 代码路径）

#### (g) 提示词与两步读

该记录（`expected = open_mortgage`）有 27 个标签，于是 `_uses_codes(27) = True`，走 `_read_codes()`。
除标记方案与读法之外，prompt 骨架与实例 ① 完全一致（state 仍在最前、仍只出现一次，echo 仍为 2）。
实测的选项块变成两 token 代码，收尾指令也换成 `CODE_INSTRUCTION`：

```
Options:
A0. cancel_card - The customer wants to cancel their card.
A1. cancel_savings_pot - ...
...
B7. open_mortgage - The customer wants to open their mortgage.
...
C6. update_mortgage - The customer wants to update their mortgage.

Reply with the code (letter and digit) of the single best option.
```

两步读（`_code_ids(27)` 的实测 token id）：

| 步骤 | 内容 | 实测 id |
|---|---|---|
| 1 | 在答案位置读**组字母** | `A, B, C` → `[32, 33, 34]` |
| 2 | 把该组字母接到 cache 的**一条分支**上，读该组**数字** | 组 A/B：`0`–`9` → `[15..24]`；组 C 只有 `0`–`6` → `[15..21]` |

`P(option) = P(letter) × P(digit | letter)`，精确且和为 1；`marker_mass` 变成"发出任一合法代码"的全词表概率。

#### (h) 真实结果

```
predicted open_mortgage   p = 1.0（保留 6 位）   usage {input_tokens: 1027, output_tokens: 0}
latency 1.81 s   markers "codes A0-Z9 (two-step read)"   marker_mass 1.0   temperature 1.0
```

与实例 ① 一样，这条 run 也是 `temperature 1.0`（非 preset 的 1.924）；`marker_mass` 为 1.0 说明
两步读的代码方案没有漏掉任何质量。

### (i) 反例（回到实例 ①）：为什么前缀与标记表面形式必须成对扫

把同一条 20 选项任务的读位置换成 `--prefix "Answer:" --marker "{}"`，直接失败：

```
ValueError: no single-token marker set for 20 options under '{}' after 'Answer:'
```

原因可以用分词器摊开：`encode("Answer:A")` = `[15666, 53779]`，而 `decode([53779])` = **`:A`**
——冒号被并进了字母，token 序列**根本不以 `[15666, 25]` 开头**。实测在这一前缀下只有 **11/26** 个字母能通过
`_token_after()` 的"前缀 + 恰好一个 token"检查；而在 `Answer: **` 之后是 **26/26**。

这就是默认值 `--prefix "Answer: **" --marker "{}"`（且二者必须同时给）的实测依据，也是 §3 里
"标定必须成对扫、且要在多条真实 prompt 上取平均"的原因。

### (j) 一处元数据细节（实例 ②）

代码路径在 >26 选项时用 `CODE_INSTRUCTION` 作为收尾指令（见 (g) 的实测 prompt），但
`run()` 写进 `raw["runtime"]["instruction"]` 的始终是 `self.instruction`。所以上面那条 27 选项记录的
`runtime.instruction` 记的是**字母指令**，与实际 prompt 里的代码指令不一致——只影响元数据可读性，
不影响决策（同一结构里的 `markers` 字段正确记为 `codes A0-Z9 (two-step read)`）。

## 13. 四条不变量（违反时数字会看似合理却是错的）

1. **读位置必须落在标记上。** 诊断量是 `marker_mass`；低值意味着分布在噪声上重归一化。Qwen3.5 的 `<think>` 模板、gpt-oss 的 harmony 通道都是靠它发现的。
2. **前缀与标记表面形式是耦合的。** `Answer:` → `" A"`，`Answer: **` → `"A"`。必须成对扫描，并且只信任跨多条真实 prompt 的平均。
3. **只读最后一个位置。** 计算 `[1, n_tokens, vocab]` 只为读一行，在 248k 词表 + 长 state 下是 OOM；分块 + 单 token 是等价且必需的写法。
4. **比较概率或 ECE 之前，先确认 temperature。** 发布出去的概率数值只被 T 改变（§9.0 的表），
   而 T 只出现在 `/health` 的 `config` 与逐条记录的 `runtime.temperature` 里。把 T=1 的 run 与
   preset 的 T 混起来比校准，会得到看似合理的错数——§12(f) 里那条记录在 T=1 时顶部概率 0.999917、
   在 preset 的 1.924 下是 0.981007，两个数各自都对，放在一起比就是错的。

## 14. 文件对照

| 文件 | 职责 |
|---|---|
| `classify.py` | 标签集解析、掩码-logit 决策、上下文校准偏置 `calibration_bias()`（**已实现，未被调用**，见 §9.3） |
| `jevbench_adapter.py` | 决策适配器：prompt、读位置标定、标记、echo、order 平均、代码/分组读法、多问题共享 |
| `torch_backend.py` | PyTorch 后端（CUDA / MPS / CPU）：分块前向、批量读、量化与低显存选项 |
| `server.py` | TypeSafe 兼容 HTTP 服务（`/v1/systemone`、`/health`）与 pinned presets（含每 entry 的 temperature） |
| `branch.py` | fork / trim / lazy 三种 KV 分支策略与内存算术 |
| `generate.py` | 从已分类的分支继续生成；结果注入格式的 A/B |
| `selective.py` | 风险-覆盖曲线、AURC、达到目标准确率的最大覆盖率（§9.6） |
| `model.py` | 模型加载与结构自省（每 token KV 字节数） |
| `franken.py` | 注意力头复制手术（负结果实验，含 identity 对照） |
| `glue.py` | 已被取代的 GLUE harness，仅作参考 |
| `cli.py` | 两个模式的命令行入口 |
| `mcp_server.py` | 把决策 API 暴露为 MCP 工具（薄客户端，不加载模型） |
| `bench/fit_temperature.py` | temperature 的离线拟合与采纳判定；报告在 `results/calibration/`（见 §9.2、§9.5） |
| `bench/agree_boost.py` | 用第二模型的一致性缩放 top label 的 odds 做重校准；预测不变（§9.4） |
| `bench/probes/marker_probe.py` | 上下文校准 × 标记选择的六臂实验，是 §9.3「试过没赢」的证据 |

## 15. 亲手跑一遍

```bash
# 直接对一段 state 提两个问题（默认 yes/no 标签）
uv run python -m ryotide.cli classify state.txt -q "Q: ...? Answer yes or no.\nA:"

# 判类后从同一分支继续生成
uv run python -m ryotide.cli hybrid state.txt -q "..." -f "..."

# 只看 KV 预算，不加载模型
uv run python -m ryotide.cli memory --state-tokens 6000 --n 10

# 起一个 TypeSafe 服务，再用 JevBench 原装适配器驱动
uv run python -m ryotide.server --preset ryotide-qwen

# 复现 temperature 的拟合与采纳判定（只用合成数据；--report 只报告、不参与决定）
uv run python bench/fit_temperature.py --fit <synthetic-run-tag> --report <pub-run-tag>

# 算出一个 run 的 ECE 与看板上的 calib 轴（§9.5；本地只用 ECE，不含 TVD 项）
uv run python bench/score_jevbench.py myrun

# 风险-覆盖曲线：答多少、准确率多少（§9.6）
uv run python bench/selective_curve.py myrun
```

看结果里 `runtime.marker_mass` 是否接近 1.0，以及 `usage.output_tokens` 是否为 0——这两个数就能确认“读”确实发生了。

想确认某个服务实例用的是哪个 temperature 与哪套配置，读 `/health` 的 `config.temperature` 与
`prompt_hash`（T 参与 hash 计算，见 §9.2）；决策响应本身**不**回传 temperature，
逐条记录里的 `raw.runtime.temperature` 才带它。

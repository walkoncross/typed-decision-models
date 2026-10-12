# pplx-decider-v1.1-27b 是如何用一个独立的 255 行读出头、在掀掉因果掩码的那次前向上读出类型化决策的

> 代码库：<https://huggingface.co/perplexity-ai/pplx-decider-v1.1-27b>
> 对应 README 第 38–39 行：*This is the **native decision-checkpoint layout**: a `Qwen3_5Model` backbone
> plus `readout.safetensors` containing a BF16 `[255, 5120]` decision head.*；
> 第 44–46 行：*Full-attention layers use the saved **noncausal** attention mode. The model's
> linear-attention layers retain their native behavior. Default causal inference does not reproduce
> this checkpoint's evaluated setup.*；以及第 50 行：*This artifact has a separate readout,
> **not a full-vocabulary `lm_head`**.*
> 本文按 `source/src/autojev/`（model / server / evaluate / train / loss / optim / data / synthetic /
> sft\_\* / report）、`decision_config.json`、`config.json`、`chat_template.jinja` 与
> `evaluation/`、`training/` 下的 committed 产物的实际代码逐环节说明这三句话的实现方式；
> 文中出现的文件名与行号（`model.py:215`、`server.py:207`、`evaluate.py:19`、`train.py:976` 等）
> 均相对该仓库根目录。
>
> **同名配对文档**：RYOTIDE 的 [`docs/ryotide/typed-decision-model.md`](../ryotide/typed-decision-model.md)。
> §0 先讲清分歧，§11 把它的四条不变量逐条落位。
>
> **注意与 [`docs/decider/`](../decider/typed-decision-model.md) 是两个不同项目**：那一篇是
> `Mapika/decider`（自训 0.8B–35B 与 stock 指令模型两条权重来源），这一篇是 Perplexity AI 的
> `perplexity-ai/pplx-decider-v1.1-27b`。两者恰好都是「255 个标签 + 一次前向」，但码表内容、
> 读位置、读出头与这条链上的服务面全都不同（§2 会逐条点出差别）。
>
> **本文读的是哪一个 revision**：`14c9d90137962d300f3de95633f7b6535e80c5ce`
> （2026-10-12T07:26Z，48 个文件；HEAD 提交是 `14c9d90 Credit Bespoke Nimble in the model card (#2)`）。
> 上游是 **Hugging Face 上的模型仓**，`git clone` 得到的是「权重 + 一份完整的训练与数据构建代码」，
> 没有对应的走查稿。所以本文引用的一切都能按上面的 revision 逐字节比对（§15 给出复算）。

pplx-decider 与 RYOTIDE 同属「把 LLM 当分类器读」这一支：一次前向、取某个位置的读数、在已知标签集上取最大。
它在这一条线上往前推的是五件事：

1. **读出头不再借用 `lm_head`，而是一个独立的 255 行权重文件**。理由是「`lm_head` 根本没有随权重发布」——
   检查点索引里有 1,183 个张量（`language_model` 850 个 + `visual` 333 个），**一个 `lm_head` 键都没有**
   （`model.safetensors.index.json`）。所以词表维度（`config.json:118` 的 `vocab_size` 248,320）在一次决策里
   从头到尾不存在，`marker_mass` 在这里**概念不适用**（§2、§5）。从基座起训时它确实是**从基座 `lm_head`
   的那 255 行 gather 出来的**（`model.py:250`），训完再单独存成 `readout.safetensors`（§4）。
2. **全注意力层被显式掀掉了因果掩码**。`attention_mode: "noncausal_full_attention"` 写进检查点，
   加载期用一个前向预钩子把 `attention_mask` 换成「只有 padding、没有因果」的那一份；线性注意力层
   保持原生递推（`model.py:30-58`）。在因果 LM 上把**一份已经训好的**注意力掩码掀掉，十七条里没有第二条做过——Laya / Jevlike / Semantic Router 的 encoder 路径虽然也是双向的，但那是编码器从构造上就不带因果掩码，而不是把一份训好的因果读法改成非因果（§4.1）。
   代价立刻出现：**这条读法在构造上不支持任何 KV cache**（`model.py:38-39` 直接报错）。
3. **读位置就是池化位置**，而池化档写在检查点里（`decision_config.json:540` 的 `"pooling": "last"`）；
   运行期没有任何标定、扫描或诊断量。位置之所以存在，是因为 chat 模板在 `enable_thinking=False` 下
   **物化一个空的思考块**（`chat_template.jinja:165-166`），于是「模板尾巴的最后一位」是良定义的（§3）。
4. **图像直接进同一条决策**。`row["images"]` 被渲染成 user 轮开头的若干个 `{"type":"image"}`
   （`model.py:106`），与文本一起编码、一次前向读出——它不是像 H2O-Lightning 那样另立一条服务面，
   而是**同一个 `/v1/systemone` 里的同一条读法**（§8）。
5. **发布的不只是权重，而是一整条可回源的训练栈**：数据构建（`data.py` 五个 fold、六个 HF 源、三个归档、
   十个经 Nimble 转换的 tasksource 任务）、合成数据（`synthetic.py` / `sft_*.py`）、训练（`train.py`）、
   评测（`evaluate.py`）、仪表盘（`report.py`）连同 `training/` 下的三份记录一起发布。
   代价是**两处哈希在同一份发布里对不上**，而这恰好是可以机械核对出来的（§12、§15）。

先说三句结论，后面逐条给代码位置：

- 「typed」来自**分词器**，而且**发布配置里的码表是它的校验、不是它的来源**：加载时把 26 个大写字母与
  676 个双字母对拼成 702 个候选，筛掉不能单 token 化的，取**前 255 个**（`model.py:215-223`）。
  本文在随仓发布的 `tokenizer.json` 上重算过，`decision_config.json` 的 `codes` 与 `token_ids`
  **逐项重现**：702 → 588 个单 token → 截到 `JT` 为止，下一个可用的 `JU` 纯粹是因为列表已满而落选（§2、§15）。
- 读位置是**构造出来的**，运行期没有可标定的自由度：chat 模板的生成前缀在 `enable_thinking=False` 下
  以 `<\|im_start\|>assistant\n<think>\n\n</think>\n\n` 结束，`pooling: "last"` 取它的下一位；
  左填充保证批内的 `-1` 就是真末位（`model.py:210`、`:307`）（§3）。
- 概率是**读出来的**（零标定的那一步不存在），但发布值被**训练本身**与**一个全局温度**先后改写；
  而这个温度是**一个单一全局值**——不分题型、不分布局、不随选项数（`decision_config.json:538`）；
  与它同类的只有 Open-Jev（2B 发布值 1.7360720317205194）与 Nimble（2.179078721266035，但只有一个 revision 拟合过），
  其余各家都按题型 / 分布局 / 逐 workload / 逐 runtime / 逐尺寸拆开（§10.2 会把差别列清）。
  另有一处同名不同量：服务面报的 `confidence` 是**改写过**的 `(p_max − 1/k)/(1 − 1/k)`，
  而评测记录里的 `confidence` 是**未改写的 `p_max`**，服务面不直接给后者（§10）。

## 本文的事实来源

| 类别 | 来源 | 本文中的标记 |
|---|---|---|
| 代码引用 | `source/src/autojev/**`（`model.py`、`server.py`、`evaluate.py`、`train.py`、`loss.py`、`optim.py`、`distributed.py`、`events.py`、`types.py`、`data.py`、`synthetic.py`、`sft_pipeline.py`、`sft_synthetic.py`、`sft_numeric.py`、`report.py`）与 `config.json`、`decision_config.json`、`chat_template.jinja`、`processor_config.json`、`source/pyproject.toml` | `文件:行` |
| 可实跑得到的产物 | ①**用第三方 `tokenizers` 库在随仓发布的 `tokenizer.json` 上重算** `codes` / `token_ids`（不运行任何 `autojev` 代码）；②在 committed 的 `release-manifest.json`、`decision_config.json`、`model.safetensors.index.json`、`evaluation/checkpoint-verification.json`、`evaluation/decision-index.json`、`training/*.json` 上做**标准库算术** | 标注「实测」，并注明是在哪份 committed 产物上重算 |
| 公式与常量 | `README.md`（模型卡）、`release-manifest.json`、`training/config.json`、`training/data-manifest.json`、`training/train.sbatch` | `README:行`、`training/...:行` |
| 未跑 | **本机没有 GPU、没有 torch / transformers、没有下载 52.17 GB 的 11 个权重分片与 `readout.safetensors`（两者都是 LFS 指针），没有起过 `autojev.server`，没有跑过 `train.py` / `evaluate.py` / pytest** | 照例**不给任何前向产生的概率数值** |

## §0 与 RYOTIDE 的分野

| 维度 | RYOTIDE | pplx-decider-v1.1-27b |
|---|---|---|
| 权重来源 | 公开 LLM，全程 frozen | `Qwen/Qwen3.8-27B` **全量微调**（26,085,330,160 参数 + 1,305,600 的读出头），BF16 共 52.17 GB（§13、§15.5） |
| 是否训练 | 否，零新增参数 | 是：训练改动全部 26.09 B 参数，**因此改数值也改 argmax**（§10） |
| 决策机制 | 读标签 token 的 logits → 掩码 → softmax | 读**池化位置**的 hidden → **独立 255 行读出头** → 掩码 → `softmax(logits / T)`（§4、§5） |
| 「typed」从哪来 | 请求声明的标签集，靠分词器保证单 token | **分词器**：702 个候选里能单 token 化的取前 255（`model.py:215-223`）；发布配置里的码表是**校验对象**（§2） |
| 读位置怎么定 | 运行期扫前缀、按标记质量标定 | **构造出来的**：模板尾巴的最后一位，由 `pooling` 档决定；运行期无标定、无诊断量（§3） |
| 一次前向读几个位置 | 一个（末位） | 一个（池化位）；**但注意力是非因果的**，全注意力层能看见后面（§4.1） |
| 掩码与重归一化 | 全词表 softmax → 掩码到标记 → 重归一化 | **没有词表**：只有 255 个 slot，掩码只是「本题选项数之外的 slot 置 `-1e9`」（§5） |
| 置信度从哪来 | 读出来的 | **训练改数值也改 argmax**，之后一个**单一全局温度**再改一次数值（§10） |
| 顺序偏置 | 多 order 共享前缀 fork，概率平均 | **训练期每步打散**，运行期保序，**全仓没有任何顺序偏置的度量**（§6） |
| 多问题共享 state | 公共前缀预填充一次 | **不做，而且结构上不可能**：非因果前向禁止 KV cache（§7） |
| 图像 | — | 与文本同一条读法、同一次前向（§8） |
| `output_tokens` | 0 | 0，且是响应组装处的字面量（`server.py:213`） |

一句话总结立场：**它把 RYOTIDE 的「读 `lm_head` 的标签行」推进成「把一个 255 行的读出头单独训出来并单独发布」，
再把「因果」这个前提整个拆掉（掀掩码、换池化位）；代价是这条读法从此与 KV cache 不相容，
多问题共享 state 连做的余地都没有。**

## §1 端到端数据流

```
POST /v1/systemone  {"model": …, "state": …, "questions": {qid: {type, instructions, criteria}}, "images": [...]}
    │   （server.py:216；model 字段必须落在 ALIASES ∪ {service.name} 内，:35、:77-82）
    │
    ├─ EvaluationRequest 校验                                   server.py:50-115
    │    · choice.criteria 1–255 项 / score.criteria 2–10 项 / noul 的 true|false / extra="forbid"
    │    · images ≤4 张：必须是 data:image/{png,jpeg,webp};base64，解码后 ≤8 MB、≤16 M 像素，
    │      image.verify() 过一遍                                     :84-115
    │
    ├─ 鉴权（AUTOJEV_API_KEY 设了才查，hmac.compare_digest）          :118-121
    ├─ 忙锁：non-blocking acquire 失败 → 529 + Retry-After: 1        :221-222
    │
    └─ predict()                                                   server.py:190-213
         questions → 每个问题一行，同一个 state 复制 N 份               :199-201
         每 8 行一批：model.prepare(rows)                            model.py:269-294
           ├─ decision_messages(row, codes)                          :95-113
           │    system："Classify the supplied state … Reply with only the selected option code."
           │    user   = [{"type":"image"} × len(images)] + prompt 文本          :106
           │    prompt = "State:\n…" + "Question:\n…" +
           │             "Options:\n<code>: <description>"（逐行）+ "Return only the letter code…"
           │    choice 的第 i 行码就是 codes[i]；noul 是 ["false","true"]        :83-92
           ├─ apply_chat_template(…, add_generation_prompt=True, enable_thinking=False)  :279-284
           │    → 尾巴 = "<|im_start|>assistant\n<think>\n\n</think>\n\n"   chat_template.jinja:164-168
           ├─ processor(text=…, images=…, padding=True)  → input_ids / pixel_values       :287
           ├─ >8192 token 直接 raise（**不截断**）                             :288-290
           └─ input_tokens = attention_mask.sum()                     :291
         一次前向：backbone(**inputs, use_cache=False).last_hidden_state   :297
           · 非因果钩子把 attention_mask 换成 dict（全注意力层只看 padding）  :30-58
         读位置：pooling="last" → hidden[:, -1]（左填充，所以就是真末位）       :210、:307
         读出头：Linear(hidden_size → 255, bias=False) → logits（词表维度从不出现）  :248/:256、:308
         掩码：slot ≥ 本题选项数 → -1e9                                    :309-314
         概率：(logits / T).softmax(-1)，T = 1.0087417621345625             :329 / server.py:207
         answer()：按题型把概率改写成答案                                   :116-148
             choice → {probabilities, choice, confidence=(p−1/k)/(1−1/k) 截到 [0,1]}
             score  → {probabilities, legend, score=Σ i·pᵢ, confidence=1−d/baseline}
             noul   → {"noul": P(true)}（**不报 confidence**）
    ← {"model", "answers": {qid: …}, "usage": {"input_tokens": Σ, "output_tokens": 0}}   server.py:213
```

有一处细节值得先记下来：**服务面吐出的 `probabilities` 被归一化了两次**。`softmax(-1)` 已经在有效 slot 上
和为 1（`model.py:329`），`answer()` 又除以一次总和（`model.py:123-124`）。第二次是恒等操作，
但它解释了另一件事——`answer()` 只收 `values[:count]`（`server.py:211`），
所以**输出的向量长度等于本题选项数，不是 255**。

## §2 「typed」从哪来：255 个码是分词器筛出来的，配置里那份是校验

码表的构造只有七行，而且完全不看 `decision_config.json`：

```python
tokenizer = self.processor.tokenizer
candidates = list(string.ascii_uppercase) + [
    "".join(pair) for pair in itertools.product(string.ascii_uppercase, repeat=2)
]
self.codes = [code for code in candidates if len(tokenizer.encode(code, add_special_tokens=False)) == 1][
    :MAX_OPTIONS
]
self.token_ids = [tokenizer.encode(code, add_special_tokens=False)[0] for code in self.codes]
if len(set(self.token_ids)) != MAX_OPTIONS:
    raise ValueError("Tokenizer must provide 255 distinct single-token answer codes.")
```
（`model.py:214-223`，`MAX_OPTIONS = 255` 在 `model.py:25`）

702 个候选（26 + 676）在**这个分词器**上有 588 个能单 token 化，取前 255 个。本文用第三方 `tokenizers`
库在随仓的 `tokenizer.json`（19,989,492 字节）上重算，`decision_config.json` 的 `codes` 与 `token_ids`
**逐项相同**（§15）。算出来的边界很具体：

- 第 1–26 个是 `A`–`Z`（token id 32–57）；
- 之后是单 token 化的双字母对，**按字典序**取，最后一个落在 **`JT`**；
- 下一个可用的单字母对是 **`JU`**——它落选**不是因为不能单 token 化**，而是因为前 255 个名额已经满了；
- 在 `AA`–`JT` 这段里，有 **25 个**双字母对不是单 token（`BQ`、`BZ`、`CJ`、`CQ`、`CZ`、`DQ`、`DZ`、`EJ`、
  `EY`、`FJ`、`FQ`、`FV`、`FZ`、`GJ`、`GK`、`GQ`、`GZ`、`HJ`、`IY`、`JF`、`JG`、`JH`、`JL`、`JN`、`JQ`）
  ——它们被跳过的原因才是「不能单 token 化」；
- 所以 255 = **26 个字母 + 229 个双字母对**，而 `KA`–`ZZ` 之间还有 **333 个**同样合格的单 token 对
  **永远不会被用到**。

加载期还有**第二道**检查：这些码在**套上 chat 前缀之后**必须仍然是单 token，且 id 不变（`model.py:224-235`）：

```python
prefix = tokenizer.apply_chat_template(
    [{"role": "user", "content": "Choose an option."}],
    tokenize=False, add_generation_prompt=True, enable_thinking=False,
)
prefix_ids = tokenizer.encode(prefix, add_special_tokens=False)
if any(
    tokenizer.encode(prefix + code, add_special_tokens=False) != prefix_ids + [token_id]
    for code, token_id in zip(self.codes, self.token_ids, strict=False)
):
    raise ValueError("Answer codes must remain single tokens after the chat prefix.")
```

这一句与 SemIf-OpenJev 的 `direct.py:42`、decider 的 `prompt.py:229-233` 是**同一类做法：断言而不是扫描**
（comparison.md §3 的 #2）。一处必须说清的差别：它验的是**代理 prompt**（只有一条 user 消息），
**不是真实 prompt**（真实 prompt 带 system 消息、带图像占位、带 `Options:` 那一整段）。
码是字母、前缀以 `</think>\n\n` 结尾，所以这个代理大概率足够——但它是代理，不是真 prompt 的逐字节比对。
**这一点本文未复算**（复算它需要 transformers 的 chat 模板渲染器，本机没有）。

然后才是配置出场，而它只被用来**比对**（`model.py:236-237`）：

```python
if saved and (saved["codes"] != self.codes or saved["token_ids"] != self.token_ids):
    raise ValueError("Checkpoint answer vocabulary differs from its tokenizer.")
```

所以 `decision_config.json:24-537` 那 255 个 `codes` 与 `token_ids` 是**这份分词器在这个构造下的答案的存证**，
不是反过来驱动构造的配置。三者（分词器 → 运行期算出的码表 → 配置）必须一致，否则拒载。

**与 `docs/decider/` 的关系**：那一篇的 decider 同样把 255 个单 token 标签交给分词器，同样是「26 个字母 + 229 个
双字母」这个结构（comparison.md §1「typed 从哪来」那一列），但两个仓库的分词器不同、码表内容不同、
`codes` 的**角色**也不同——decider 的 `label_table` 是运行期从分词器现算并断言的表，这里则是
「现算 + 与发布配置比对」两道。两篇可以并读，但**不能互相借用码表**。

选项数上限在三个地方各出现一次，且互相自洽：`model.py:25` 的 `MAX_OPTIONS = 255`、
`model.py:98-99` 的 `1 <= len(descriptions) <= min(MAX_OPTIONS, len(codes))`、
以及服务面的 `criteria: … Field(min_length=1, max_length=255)`（`server.py:57`）。
`score` 另有自己的下限：`min_length=2, max_length=10`（`server.py:62`），而 `model.py:137-138`
也要求「打分题至少两级」。**1 个选项是允许的**，此时 `answer()` 直接给 `confidence = 1.0`
（`model.py:130`）。

## §3 读位置：模板尾巴的最后一位，由池化档选出来

位置这件事在这条链上有两半，缺一不可。

**第一半是模板。** `enable_thinking=False` 在这份 `chat_template.jinja` 里**不是「不加思考块」，
而是「加一个空的思考块」**：

```jinja
{%- if add_generation_prompt %}
    {{- '<|im_start|>assistant\n' }}
    {%- if enable_thinking is defined and enable_thinking is false %}
        {{- '<think>\n\n</think>\n\n' }}
    {%- else %}
        {{- '<think>\n' }}
    {%- endif %}
{%- endif %}
```
（`chat_template.jinja:163-170`）

不传这个开关时模板**默认开推理**、档位还是 `xhigh`（`chat_template.jinja:46-48`），
而 assistant 前缀会停在**没有闭合的** `<think>\n`（`:168`）。这与 OpenJev(HF) 那处的坑同源
（comparison.md §4.1）：**漏传一处的后果就是读位置落进思维块**。这条链上的接法是「每一处都传死」：
两处调用（`model.py:224-229` 的码表校验、`model.py:279-284` 的真实 prompt）都显式给
`enable_thinking=False`，`data.py` 里做 token 计数的两处也传（`data.py:503-505`、`:514-516`）。
没有回退分支——与 RYOTIDE / Quyet 那种「模板不认这个开关就回退」不同，这里模板认，代码也不准备退路。

**第二半是池化。** 档位写在检查点里，加载时读出来，而**运行期不允许改**：

```python
saved_pooling = saved.get("pooling", "last") if saved else "last"
self.pooling = pooling or saved_pooling
if self.pooling not in {"last", "mean"}:
    raise ValueError("Unknown pooling mode")
if saved and self.pooling != saved_pooling:
    raise ValueError("Checkpoint pooling differs from the requested mode")
```
（`model.py:194-199`；发布值是 `decision_config.json:540` 的 `"last"`）

前向里两档分别是（`model.py:296-307`）：

```python
hidden = self.backbone(**batch.inputs, use_cache=False).last_hidden_state
if self.pooling == "mean":
    valid = batch.inputs["attention_mask"].bool()
    counts = valid.sum(dim=1, keepdim=True)
    ...
else:
    hidden = hidden[:, -1]
```

`last` 这一档取批张量的 `-1`。**它是安全的，但安全性来自另一处设置**：tokenizer 的 `padding_side`
被显式设成 `"left"`（`model.py:210`），所以补齐的 pad 排在真实 token **之前**，`-1` 永远是这一行
真实的最后一个 token。这与 comparison.md §4.2 里 SemIf-OpenJev 的 `shared`、Nimble 的分支、
Kev 的显式下标是同一类纪律的不同解法——它们靠「读每条自己的末位」，这里靠**把填充挪到左边**。
（`tokenizer_config.json` 的 padding 块自己也写着 `"direction": "Left"`。）

于是「读位置怎么定」在这条链上的答案是：**训练期固定 + 构造出来 + 加载期校验**，
运行期没有扫描、没有候选位置、没有阈值、**也没有任何诊断量**（`model.py` / `server.py` / `evaluate.py`
里没有任何名字带 `position` / `slot` / `calibrat` 的运行期对象）。「位置读错了」这件事在这里不会报错，
只会安静地给出一个格式正确的分布——这一点与 OpenJev(HF) 同病（comparison.md §3 的 #3、§4.1）。

## §4 一次前向：掀掉因果掩码，读一个独立的 255 行头

### §4.1 非因果：这一路独有的那个动作

`attention_mode` 与 `pooling` 一样从检查点读出来并被强制比对（`model.py:188-193`），
发布值是 `"noncausal_full_attention"`（`decision_config.json:539`）。开启时挂一个前向预钩子
（`model.py:258-259`）：

```python
def enable_noncausal_full_attention(text_model: torch.nn.Module) -> None:
    """Allow future keys in SDPA layers; retain padding and the original recurrence."""
    if text_model.config._attn_implementation != "sdpa":
        raise ValueError("The attention ablation requires SDPA")

    def mask_inputs(module, args, kwargs):
        if args:
            raise ValueError("The attention ablation requires keyword inputs")
        if kwargs.get("past_key_values") is not None or kwargs.get("use_cache"):
            raise ValueError("Noncausal classification does not support a KV cache")
        ...
        kwargs["attention_mask"] = {
            "full_attention": padding[:, None, None, :].bool(),
            "linear_attention": create_recurrent_attention_mask(
                config=module.config, inputs_embeds=embeddings, attention_mask=padding,
            ),
        }
        return args, kwargs

    text_model.register_forward_pre_hook(mask_inputs, with_kwargs=True)
```
（`model.py:30-58`）

三件事同时发生：

1. **全注意力层拿到的是一个只表达 padding 的 4 维布尔掩码**（`padding[:, None, None, :]`），
   里面**没有下三角**——这就是「掀掉因果掩码」的字面含义。模型卡原文是
   *Full-attention layers use the saved noncausal attention mode*（`README:44-45`），
   而 `config.json:16` 的 `full_attention_interval: 4` 说明 64 层里有 16 层是这种
   （`config.json:22-87` 的 `layer_types` 里每第 4 个是 `full_attention`）。
2. **线性注意力层不吃这一套**：它们保留原生行为，掩码换成
   `create_recurrent_attention_mask(...)`——递推层的状态推进与掩码语义本来就不是同一回事
   （`config.json:93` 的 `mamba_ssm_dtype: "float32"` 说明这些层有自己的状态精度）。
   模型卡照写：*The model's linear-attention layers retain their native behavior*（`README:45-46`）。
3. **KV cache 被显式拒绝**：`past_key_values is not None or use_cache` 直接抛
   `"Noncausal classification does not support a KV cache"`（`model.py:38-39`）。
   而 `forward` 也确实每次都传 `use_cache=False`（`model.py:297`）。

第 3 条的后果比它看起来大：**这条读法在构造上与「一次 prefill、多题复用」不相容**。
comparison.md §4.3 里那些 cache 分支复制的坑（SemIf-OpenJev 的 `reorder_cache`、Kev / decider / Nimble
逐状态名深拷、Semantic Router 重跑后缀）在这里**没有对象**——不是因为选择了「不做」，
而是因为**没有 cache 可以复制**。§7 会把它的代价算清楚。

本文**不给出非因果与因果两条路径上的任何数值差**：那需要权重与 GPU。唯一能确定的是
`README:46-47` 的自陈——*Default causal inference does not reproduce this checkpoint's evaluated setup*，
以及 `training/train.sbatch:32` 里 `--attention-mode noncausal_full_attention` 是**训练时就用着的**，
所以非因果不是推理期的花招，而是这个检查点被训练与评测时的那套设置。

### §4.2 读出头：一个 255 行的权重文件，词表维度从不出现

读出头就是一个不带 bias 的线性层，输出宽度写死成 255（`model.py:248`、`:256`）：

```python
self.readout = torch.nn.Linear(config.hidden_size, MAX_OPTIONS, bias=False, dtype=dtype)
```

它有两种来源，分别对应「从基座起训」与「读已发布的检查点」：

```python
if checkpoint is None:
    original = Qwen3_5ForConditionalGeneration.from_pretrained(...)
    config = cast(Qwen3_5TextConfig, original.config.text_config)
    self.readout = torch.nn.Linear(config.hidden_size, MAX_OPTIONS, bias=False, dtype=dtype)
    with torch.no_grad():
        self.readout.weight.copy_(original.lm_head.weight[self.token_ids])
    self.backbone = original.model
    del original
else:
    self.backbone = Qwen3_5Model.from_pretrained(str(checkpoint), dtype=dtype, attn_implementation="sdpa")
    config = cast(Qwen3_5TextConfig, self.backbone.config.text_config)
    self.readout = torch.nn.Linear(config.hidden_size, MAX_OPTIONS, bias=False, dtype=dtype)
    self.readout.load_state_dict(load_file(str(Path(checkpoint) / "readout.safetensors")))
```
（`model.py:239-257`；`hidden_size` 是 `config.json:19` 的 5120）

从基座起训的那条路把**基座 `lm_head` 的那 255 行**搬进读出头——所以「typed」在**初始化时**是
「`lm_head` 的码行」这个构造，训练再把它改掉。读检查点的那条路则完全不碰 `lm_head`：
它加载的是 `Qwen3_5Model`（**基座类，本来就没有 `lm_head` 属性**，那是
`Qwen3_5ForConditionalGeneration` 才有的），再从 `readout.safetensors` 装这一个矩阵。

发布检查点里到底有没有 `lm_head`，可以机械地看一眼（§15 复算）：

- `model.safetensors.index.json` 的 `weight_map` 有 **1,183** 个张量，前缀分布是
  `language_model` 850 + `visual` 333；
- 含 `lm_head` 的键：**0 个**；含 `readout` 的键：**0 个**（读出头是另一个文件）；
- 元数据 `total_parameters: 26085330160`、`total_size: 52170660320`，两者之比是 **2.0** 字节/参数
  → 整套权重是 BF16。

`readout.safetensors` 自己有多大也能算：255 × 5120 × 2 字节 = **2,611,200**，
加上 safetensors 的 8 字节长度前缀与补到 8 字节对齐的头（73 + 7 = 80），合计 **2,611,288**——
与 `release-manifest.json` 里记的 `readout.safetensors` 字节数**逐字节相同**（§15）。
所以模型卡第 39 行那句 *BF16 `[255, 5120]`* 是**可以在不下载权重的前提下核对出来的**。

于是 RYOTIDE 的 `marker_mass`（「全词表质量有多少落在标记上」）在这里没有对象：
**255 就是这个读出头输出的全部宽度，标签之外的 logits 在数学上不存在**。
comparison.md §3 的 #1 把这种情形里的各家按**机制**分开记（Open-Jev 的头标量、Semantic Router 的
三套非词表读法、CLM 的没有词表、Kev 的 pointer head、decider 的 `F.linear(hs, lm_head[labels])`、
Nimble 的只投影候选 K 行、Jevlike 的每选项一个标量、H2O 的 `logprob_token_ids` 请求参数）——
pplx-decider 是**第九种**：不是运行期去 `lm_head` 里取行，而是**把那 255 行落成一个独立的权重文件、
单独发布、单独加载**。改写后的等价命题随之变成
「**读出头里装的必须恰好是那份码表对应的 255 行**」，而它的守卫有三道：
`Linear(hidden, 255)` 的**形状**（装错尺寸在 `load_state_dict` 就会失败）、
`len(set(self.token_ids)) != MAX_OPTIONS` 的**唯一性断言**（`model.py:222-223`）、
以及与发布配置的**逐项比对**（`model.py:236-237`）。**运行期一道都没有。**

## §5 掩码 + softmax：255 个 slot 上的计数掩码

掩码只有一件事可做：把本题用不到的 slot 压掉。`forward` 的收尾是（`model.py:308-314`）：

```python
logits: torch.Tensor = self.readout(hidden).float()
mask = (
    torch.arange(MAX_OPTIONS, device=logits.device)[None]
    >= torch.tensor(batch.counts, device=logits.device)[:, None]
)
return logits.masked_fill(mask, -1e9)
```

`batch.counts` 是每行的选项数（`model.py:276` 的 `len(options(row["question"])[0])`）。
`-1e9` 而不是 `-inf`，与 H2O-Lightning、SemIf-OpenJev 那类实现同一个习惯
（留着有限值，免得 NaN 传播进后续的 `float()` 与温度除法）。

温度在下一环，而且**在掩码之后**（`model.py:329`）：

```python
probabilities: list[list[float]] = (self(batch) / scale).softmax(-1).cpu().tolist()
```

服务面把同一句话又写了一遍（`server.py:207`）：

```python
distributions: list[list[float]] = (model(batch) / model.temperature).softmax(-1).cpu().tolist()
```

所以这条链上的 `softmax(logits / T)` 与 RYOTIDE、decider、Nimble、H2O 是同一个式子，
只是这里的「logits」只有 255 个分量、且**被掩掉的那些在除以 T 之前就已经是 `-1e9`**——
T ≈ 1.0087 时它们仍是约 −9.9e8，`exp` 之后是 0。

两处与别家不同、值得单独记下：

1. **`argmax` 是在概率上取的，不是在原始 logits 上。** `answer()` 里是
   `best = max(range(len(values)), key=values.__getitem__)`（`model.py:127`），
   而 `evaluate.py:128` 也一样（`labels[max(range(len(labels)), key=values.__getitem__)]`）。
   这与 Nimble / Kev 那两条**相反**——那两条刻意在**原始 logits** 上取 argmax，
   于是「温度不改选择」是**构造性质**（comparison.md §5 第 10 条、§2 的 Nimble 行）。
   在这里它只是**数学性质**（正标量除法单调）。差别在温度为正时看不出来，但**性质不同**：
   若 T 变成 0 或负数，这里会改选择，那里连路径都不经过。
2. **没有「重归一化」这一步可做，但做了两次归一化。** 见 §1 末尾：`softmax` 已归一，
   `answer()` 又除一次总和（`model.py:123-124`）。第二次是恒等操作。

`marker_mass` 在这里概念不适用（§4.2）。真正被计数的诊断量是 `zero_probability_count`——
`metrics()` 会数「概率恰好为 0 的项」（`evaluate.py:265`），committed 的验证记录里它是 **0**
（`evaluation/checkpoint-verification.json`）。这个量在 15 个等宽箱的 ECE 之外，
是这条链上唯一一个与「分布是否退化」有关的数。

## §6 顺序偏置：训练期每步打散，运行期没有任何度量

打散发生在**每个训练 step**，而不是每个 epoch（`train.py:976`）：

```python
group = augment(rows, random.Random(args.seed + 100003 * (step + 1)))
```

`augment` 只对 `choice` 题型动手，把 `criteria` 的顺序重排，**并把软目标按同一个置换跟着重排**
（`train.py:367-379`）：

```python
def augment(rows: Sequence[Example], rng: random.Random) -> list[Example]:
    result = copy.deepcopy(list(rows))
    for row in result:
        if row["question"]["type"] == "choice":
            criteria = row["question"]["criteria"]
            target = row["target"]
            weights = dict(zip(criteria, target, strict=True)) if isinstance(target, list) else None
            items = list(criteria.items())
            rng.shuffle(items)
            row["question"]["criteria"] = dict(items)
            if weights is not None:
                row["target"] = [weights[key] for key, _ in items]
    return result
```

注意 `weights` 只有在 `target` 是 list 时才有——即**只有带软目标的题**会连目标一起置换，
硬标签题只换选项顺序（硬标签是字符串，跟着 `criteria` 走，所以语义上也是对的）。
种子按 step 派生（`args.seed + 100003 * (step + 1)`），与 RNG 的可续跑状态一起存进 `resume.pt`
（`train.py:894-908`），所以打散是**可复现的**。

运行期**保序**：`decision_messages` 按 `zip(codes, descriptions)` 逐行渲染
（`model.py:102-104`），第 i 个选项永远是第 i 个码、第 i 个读出头 slot。

**度量不存在。** 在决策与服务的四份文件里（`model.py`、`server.py`、`evaluate.py`、`optim.py`、
`distributed.py`）对 `permut` / `shuffle` / `reorder` / `rotate` 的搜索是**零命中**；
评测侧也没有任何「反转选项重读」的开关（对比 decider 的 `second_reading`、
Kev 的 `RotationAveraged`、H2O-Lightning 那两条「工具与度量都不在」）。
所以「pplx-decider 对选项顺序敏感吗」这个问题，**这份发布里没有任何东西能回答**，
包括训练记录：`training/` 下只有 `config.json`、`data-manifest.json`、`train.sbatch`，
没有逐 step 的 `evaluations.jsonl`（那是 `train.py:875` 在运行目录里写的，未随仓发布）。
它与 Kev 的处境不同（Kev 有 `/v1/systemone/permute` 端点、有 committed 的 `permutation` 记录）、
与 Nimble 也不同（Nimble 工具写好了、实验没跑）——**它是工具与度量同时缺席**，
与 H2O-Lightning 同列（comparison.md §1「顺序偏置」那一列）。

## §7 多问题共享 state：不做，而且结构上做不到

服务面把「多个问题」展开成**多行**（`server.py:199-212`）：

```python
rows: list[DecisionInput] = [
    {"state": body.state, "question": question, "images": list(body.images)} for question in questions.values()
]
...
for start in range(0, len(rows), 8):
    batch = model.prepare(rows[start : start + 8])
    distributions: list[list[float]] = (model(batch) / model.temperature).softmax(-1).cpu().tolist()
    ...
    input_tokens += batch.input_tokens
```

同一个 state 被复制成 N 行，每行**各自完整编码一次**。而且这**不是选择，是约束**：
非因果钩子在任何带 cache 的调用上直接抛异常（`model.py:38-39`），`forward` 也恒传
`use_cache=False`（`model.py:297`）。所以 comparison.md §7 那一列里其他实现用的手段
——vLLM 前缀缓存（OpenJev(HF)、H2O-Lightning）、`reorder_cache` 复制分支（SemIf-OpenJev）、
共享前缀分块 fork（decider 的 `shared_prefix.py`）、`prefix_prefill_count`（Nimble）——
在这条链上**一条都接不进来**：没有 KV cache 就没有可复用的前缀状态。

代价落在账上，而且账是**如实**的：`input_tokens` 累加的是每批的
`attention_mask.sum()`（`model.py:291`、`server.py:212`），也就是**同一个 state 被算了 N 次、
也计了 N 次**。这与 CLM 那种「计费随缓存状态变」的账不同（comparison.md §5 第 8 条），
也与 H2O-Lightning / OpenJev(HF) 那种「共享头只计一次」相反——这里是**老老实实按重复编码收费**。
另一半账是 `output_tokens: 0`，它是响应组装处的**字面量**（`server.py:213`），
机制与字段两个层面都成立（这一条比 Kev 干净，与 CLM / Semantic Router 同类）。

`batch_size` 在这里有两个不同的值：`predict` 的默认是 8（`model.py:318`），
服务面**自己写死 8**（`server.py:205-206`），而 `evaluate.py` 的 CLI 默认也是 8（`evaluate.py:508`）。
训练侧的 batch 是另一套（32，`train.sbatch:30`）。

## §8 图像：同一条读法里的第二个输入，不是第二条服务面

图像进决策只有两行（`model.py:95-113` 里的收尾）：

```python
content = [{"type": "image"} for _ in row.get("images", [])] + [{"type": "text", "text": prompt}]
return [
    {"role": "system", "content": "Classify the supplied state using the question and option descriptions. …"},
    {"role": "user", "content": content},
]
```

注意顺序：**图像占位排在文本之前**，所以视觉 token 出现在这条序列的开头，prompt 文本跟在后面。
`processor(text=texts, images=images, padding=True, return_tensors="pt")` 一次把它们编码好
（`model.py:287`），`accepts` 的三种输入形式（PIL 对象 / 文件路径 / `data:image/...;base64` URL）
由 `open_image` 统一成 RGB（`model.py:151-158`）。

尺寸的处理在**两处**，而且都在加载期就定死：

```python
MIN_IMAGE_PIXELS = 256 * 256
MAX_IMAGE_PIXELS = 2048 * 2048
...
if max_image_pixels < MIN_IMAGE_PIXELS:
    raise ValueError(f"max_image_pixels must be at least {MIN_IMAGE_PIXELS}.")
self.processor.image_processor.size = {"shortest_edge": MIN_IMAGE_PIXELS, "longest_edge": max_image_pixels}
```
（`model.py:26-27`、`:211-213`）

模型卡把这条几何讲清了：每边圆整到 32 px 的整数倍，总面积夹在 256×256 与 `max_image_pixels` 之间，
**每个 32×32 的 patch 花一个视觉 token**，所以一张图大约用 `width * height / 1024` 个 token
（`README:91-93`）。这个 32 不是随便写的：`processor_config.json:18-19` 里
`patch_size: 16` 与 `merge_size: 2`，16 × 2 = 32 ✓。

**这里有一处必须记下的口径分歧，而且它是本文发现的最像 OpenJev(HF) 那一类的地方：**
发布代码的**默认上限是 2048×2048**，而训练与已报告的评测都把图限制在 **512×512**：

> *Training and the reported evaluation capped images at 512×512 (262,144 px, ≤256 tokens).
> Higher resolutions keep more detail but fall outside the training distribution … To reproduce
> the evaluated setup exactly, use `max_image_pixels=512 * 512`.*
> （`README:107-110`）

也就是说 `DecisionModel(checkpoint, device="cuda")` 用**默认参数**起服务时，
图像走的是训练分布的 **16 倍面积**那一档；而 `README:114-117` 自己承认发布代码与训练代码的差别正是这一处：

> *`source/src/autojev/model.py` is the evaluated training-run code, with one change: the image-size
> cap is now the configurable `max_image_pixels`. It was previously hardcoded at 512×512 pixels.*

这个差别可以在**发布记录里被机械地核出来**：`decision_config.json` 的 `provenance` 给 `model.py`
记的 sha256 是 `824721da…`（训练时那一版），而 `release-manifest.json` 与仓内实际文件都是
`02940963…`（发布版）；**其余八个代码文件的哈希两边完全相同**（§15）。所以这不是笔误，
是一次被记录下来的、只改了默认值的改动——代价是**默认参数不在发布数字的分布里**。

### 与 H2O-Lightning 的图像路径对比

| | H2O-Lightning | pplx-decider |
|---|---|---|
| 图像在哪 | **第二条服务面**：带图的请求走 `/v1/chat/completions` | **同一条读法**：同一路由、同一批、同一读出头（`model.py:106`） |
| 图像的读法 | assistant 轮用 `continue_final_message` 预填同一个 `Answer:`，仍是一次 1-token 读数 | 视觉 token 拼在 user 轮开头，与文本一起过一次（非因果）前向 |
| 温度 | 图像路径温度独立（0.65） | 一个全局 T，不分模态 |
| 取得图像的方式 | 一个自带 SSRF 边界的 URL 抓取器 | **只收 base64 data URL 或本地路径**；服务面对 URL 一律拒（`server.py:90-96` 只认 `data:image/{png,jpeg,webp};base64`） |

最后一行值得单独说：H2O-Lightning 那条路里最需要小心的一块（SSRF 边界）在这里**不存在**，
因为**根本不接受远程 URL**。代价是调用方必须自己把图编码进请求体，
而 `server.py:88-89` 又把「编码前」的字符串长度卡在 12 MB。

## §9 服务化与 fail-loud

服务面是 FastAPI 的一个进程、四个路由（`server.py:139`、`:160`、`:165`、`:176`、`:216`）：

| 路由 | 说明 |
|---|---|
| `POST /v1/systemone` | 唯一的决策入口；`model` 字段必须落在 `ALIASES ∪ {service.name}` 内（`:35`、`:77-82`） |
| `GET /health` | `{status, model, checkpoint, authentication, modalities}`；**`status` 是 `loading` 还是 `ready` 只看 `service.model is None`**（`:165-173`） |
| `GET /v1/models` | 把 `ALIASES` 逐个报出来，带 `release_date`——**它是 `decision_config.json` 的 mtime 的日期**，不是训练日期（`:131-132`） |
| `GET /` | 返回 `playground.html` |

`ALIASES` 里有四个兼容名，且**不区分大小写地都用同一份权重**（`server.py:35`）：

```python
DEFAULT_MODEL = "autojev-qwen3.8-27b"
ALIASES = {"autojev", "jev-latest", "jev-preview", "jev-1.13.0", DEFAULT_MODEL}
```

`jev-1.13.0` 这个别名值得记一笔：它是 TypeSafe Jev 的版本号，而 `evaluate.py:510` 的参考实现
默认打的正是 `typesafe/jev-1.13`（经 `https://openrouter.ai/api/alpha/decisions`，`evaluate.py:21`）。
所以这条链上的「对照臂」与「兼容别名」指的是同一个产品。

**fail-loud 的清单**（每一条都是加载期或请求期的硬报错，没有静默降级）：

| 检查 | 位置 | 触发时 |
|---|---|---|
| `format_version != 1` | `model.py:183-185` | 加载期拒载 |
| `attention_mode` 不在两档内、或与检查点不一致 | `model.py:188-193` | 加载期拒载 |
| `pooling` 不在两档内、或与检查点不一致 | `model.py:194-199` | 加载期拒载 |
| `revision` 必须是 40 位十六进制（*immutable*） | `model.py:200-201` | 加载期拒载 |
| `max_image_pixels < 256×256` | `model.py:211-212` | 加载期拒载 |
| 码表不是恰好 255 个互异 token id | `model.py:222-223` | 加载期拒载 |
| 码表在 chat 前缀后不再单 token | `model.py:231-235` | 加载期拒载 |
| 检查点码表 ≠ 分词器现算的码表 | `model.py:236-237` | 加载期拒载 |
| `temperature` 非正或非有限 | `model.py:265-266` | 加载期拒载 |
| 输入超过 8192 token | `model.py:288-290` | **拒绝而不是截断**：`"…; no input was truncated."` |
| 选项数不在 1–255 | `model.py:98-99` | 请求期 → 422 |
| 打分题不足两级 | `model.py:137-138` | 请求期 → 422 |
| 概率非有限 / 有负 / 总和非正 | `model.py:121-122` | 请求期 → 422 |
| 图像不是三种 data URL 之一 / 解码后超 8 MB / 超 16 M 像素 / 格式不对 / `verify()` 失败 | `server.py:90-114` | 请求期 → 422 |
| 模型未就绪 | `server.py:219-220` | **503** |
| 已有请求在跑（non-blocking 锁） | `server.py:221-222` | **529** + `Retry-After: 1` |

两点与别家并读才看得出来的：

1. **`/health` 只报「模型对象在不在」，不报任何自检结果。** 对比 vLLM Semantic Router 的
   golden 门控（跑不过就不 ready）、OpenJev(HF) 的 `/v1/version` 自证、
   H2O-Lightning 的 `/health` 报告温度与地板的**来源**——这里的 `/health` 是十七条里最薄的一档。
   加载期的那些断言做得很实，但**加载成功之后没有任何东西再验证一次读法**。
2. **`GET /` 会 500。** `playground.html` 被 `Path(__file__).with_name("playground.html").read_text()`
   读取（`server.py:160-162`），而**这个文件不在发布的 48 个文件里**——`release-manifest.json` 的
   `files` 逐项列了 46 个文件（只漏掉它自己和 `.gitattributes`），其中没有
   `source/src/autojev/playground.html`。
   所以这个路由是一个**引用了不存在文件的活路径**——它不影响决策面（`/v1/systemone` 不碰它），
   但它是这份发布里唯一一处「文档/路由承诺了、仓里没有」的东西。
   comparison.md §4.5 那一类「写了但不在生效路径上」在这里是**反向的**：
   生效路径指着的东西**没写**。

## §10 置信度从哪来，又怎么被改写

### §10.1 谁在改发布数值

这条链上的数值只有一个来源和一个改法，链条比大多数实现短：

| 环节 | 机制 | 改 argmax？ | 位置 |
|---|---|---|---|
| ① **训练本身** | 全部 26.09 B 参数被全量微调，损失是**选项 logits 上的交叉熵**（`loss.py:13-18`）——不是语言建模 | **是** | `train.py:989-991`、`train.py:716-720` |
| ② **一个全局温度** | `softmax(logits / T)`，T = **1.0087417621345625**，随权重发布 | 否（正标量单调）——但见 §10.3 关于「数学性质 vs 构造性质」 | `decision_config.json:538` → `model.py:264`、`:329`、`server.py:207` |
| ③ **`answer()` 的三个逐题型公式** | choice 报 `(p_max − 1/k)/(1 − 1/k)` 截到 `[0,1]`；score 报 `1 − d/baseline` 截到 `[0,1]`；noul 直接报 `P(true)` | 否（只改报出去的那个数） | `model.py:129-148` |
| ④ **`max_image_pixels`** | 默认 2048×2048，而训练与评测用 512×512 | **可以**（它改的是输入） | `model.py:27`、`:211-213`；`README:107-110` |

第 ④ 条是这条链上最容易被忽略的一环，也是与 OpenJev(HF) 的「代码默认值 ≠ 发布配方」性质最接近的一处
（comparison.md §5 第 6 条）——但**方向更重**：OpenJev(HF) 那三处默认值改的是**报出去的数**，
这里改的是**喂进去的图**。所以「跑了默认参数、看着正常、数字合理」在这里的含义是
「用了一个训练分布之外的输入尺度」——而报出来的概率仍然是一条形状正确的分布。

### §10.2 温度是怎么来的

发布值 1.0087417621345625 不是手填的，是 `fit_temperature` 在 **983 行的 temperature fold**
（`training/config.json:63`）上拟合出来的，而在选择检查点的那一次评测里拟合（`train.py:803-806`）：

```python
temperature_logits = infer(model, temperature_rows, args)
fitted = fit_temperature(
    temperature_logits, [label_index(options(row["question"]), hard_label(row)) for row in temperature_rows]
)
```

拟合本身是**黄金分割搜索最小化 mean NLL**，numpy only、不加载模型
（`evaluate.py:194-224`）。三处细节值得记：

```python
low, high = math.log(0.05), math.log(20.0)
...
candidates = [math.log(0.05), (low + high) / 2, math.log(20.0), 0.0]
return math.exp(min(candidates, key=loss))
```
（`evaluate.py:210`、`:223-224`）

1. 搜索区间是 `T ∈ [0.05, 20]`（在 log 域上搜），80 次迭代；
2. **`0.0` 是显式的候选**——也就是 `T = 1.0`（未校准）永远在候选集里，
   所以当拟合没赢过恒等映射时，返回的就是 1.0。这与 SemIf-OpenJev 的做法一致
   （comparison.md §2 的温度表那一行），与 Open-Jev / Laya 那种「拟合值无条件写入」不同；
3. **没有接受 / 拒绝判据**：`fit_temperature` 的返回值直接进 `save_selected`
   （`train.py:832-842`）、由 `model.save(temperature=scale)` 写进 `decision_config.json`
   （`model.py:354`）。这一点与 decider（只 print JSON，写回是人工步骤）不同，
   与 RYOTIDE 的预注册判据、Kev 的留出集闸门、Nimble 的预注册验收也都不同
   （comparison.md §4.4）——**这里的温度是自动落盘的、无门槛的**。
   唯一的间接约束来自温度 fold 与训练集**不交叠**：`train.py:611-620` 要求各 partition 的
   `id` 集合与 `(dataset, family)` 集合两两不交，违反即报错。

**温度是单一的、全局的，而且是自动落盘的。** 单值这一档并不只有它：Open-Jev 也是一个数
（2B 发布值 1.7360720317205194）——但那是**无条件发布**、没有候选集回退；
拆开的那一档则很多：decider 分 state-first / schema 缓存两套再乘逐题型
（comparison.md §2 的 decider 行）、H2O-Lightning 三个题型各一个、SemIf-OpenJev 逐 workload 三个、
Kev 逐尺寸四个、Semantic Router 逐 runtime 分开。这里 `decision_config.json` 只有一个 `temperature` 键
（`:538`），`model.py:264` 读它、`:329` 用它、`server.py:207` 也用它——**不分题型、不分选项数、
不分模态**。所以「这个概率是在哪个 T 下产生的」在这里有一个干净的回答：**就是那一个数，写在权重包里**。

### §10.3 两个都叫 `confidence` 的数

服务面报的 `confidence` 与评测记录里的 `confidence` **不是同一个量**：

```python
# model.py:129-136（服务面，choice）
confidence = 1.0 if len(values) == 1 else (values[best] - 1 / len(values)) / (1 - 1 / len(values))
return {
    "type": "choice",
    "probabilities": distribution,
    "choice": keys[best],
    "confidence": max(0.0, min(1.0, confidence)),
}
```

```python
# evaluate.py:144（评测记录）
"confidence": values[chosen_index],
```

前者是**机会校正后的归一化最大值** `(p_max − 1/k)/(1 − 1/k)`，后者是**未改写的 `p_max`**。
而 **ECE 算的是后者**（`evaluate.py:237` 用 `row["confidence"]` 分箱，`:144` 把它填成 `p_max`）。
所以：

- `evaluation/checkpoint-verification.json` 里那 15 个可靠性箱、`ece = 0.015921474840787767`
  描述的是**未经 `answer()` 改写的那个数**；
- 服务面**从来不吐** `p_max` 本身（要拿它只能从 `probabilities` 里自己取最大）。

这与 decider 的处境是同一个形状（那个包里 `system_one()` 报 TypeSafe 的 rescale、`decide()` 报 `p_max`，
comparison.md §5 第 11 条），也与 Kev 的 ③ 同一族公式（`(p_max − 1/K)/(1 − 1/K)`，comparison.md §2 的 Kev 行）。
差别在**分工**：decider 是两个 API 各报一个、都在同一个服务进程里；
这里是**服务面报改写过的、评测面报未改写的**，两边的消费者是不同的人。
所以「引用这个 `confidence`」之前要先问：**是调用方从 HTTP 拿到的，还是从评测记录里读的？**

三个题型的报法还要分开记（`model.py:125-148`）：

- **`choice`**：报 `probabilities`（长度 = 选项数）+ `choice` + 改写过的 `confidence`；
- **`score`**：报 `probabilities` + `legend` + **`score = Σ i·pᵢ`**（期望档位）+ `confidence = 1 − d/baseline`，
  其中 `d = Σ pᵢ·|i − argmax|`、`baseline = (Σ|i − midpoint|)/k`。**两个数都是导出的，都不是学出来的**；
- **`noul`**：只报 `{"noul": P(true)}`，**根本不报 `confidence`**（`model.py:125-126`）——
  与 H2O-Lightning 同一做法（comparison.md §1「置信度来源」那一列的末行）。

两处 clamping 会让服务面的数**比数学值更平**：`confidence` 被截到 `[0,1]`
（`model.py:135`、`:147`），而 `(p_max − 1/k)/(1 − 1/k)` 在 `p_max < 1/k` 时是负数——
也就是说**任何最大概率低于随机水平的题，服务面一律报 `confidence = 0`**。
这与 H2O-Lightning 的 yes/no 地板（把下界抬到 0.801，comparison.md §2）方向相反（一个抬、一个压），
但性质相同：**都不是精度问题，是把一个区间整个抹平**。
它是**只读代码就能确定的**（不需要任何前向数值）：对 `k ≥ 2`，`p_max < 1/k ⇔ 该公式为负`。

最后回到 §5 记的那条性质差别，它在 §3 的不变量 #4 上是有后果的：
`answer()` 与 `evaluate.py:128` 都在**概率**上取 argmax。所以「温度不改选择」在这里是
**数学性质（正标量单调）**，不是 Nimble / Kev 那种**构造性质（argmax 走原始 logits）**。
温度为正时两者等价；差别在于**这条链上没有任何东西阻止一个非正的 T 改掉选择**——
`model.py:265-266` 只要求 `T > 0` 且有限，而这个断言一旦失效（或有人直接把 `logits` 传进来做 argmax），
`answer()` 的路径本身并不设防。

## §11 四条不变量的落位

RYOTIDE §13 那四条在这条链上**两条要改写、一条概念不适用、一条要加内容**：

| 不变量 | 落位 | 说明 |
|---|---|---|
| **#1** 读位置必须落在标记上（`marker_mass`） | **概念不适用** | 读出头只有 255 个分量，词表维度（248,320）在推理期从不出现，标签外的质量在数学上不存在，`marker_mass` 恒等于 1。这是 comparison.md §3 #1 里**第九种机制**：不是运行期取 `lm_head` 的行（decider / Nimble），而是**把那 255 行落成一个单独发布的权重文件**（§4.2）。改写后的等价命题是「读出头里装的必须恰好是那份码表的 255 行」，守卫有三道且**全在加载期**：`Linear(hidden, 255)` 的形状、255 个 id 互异的断言（`model.py:222-223`）、与发布配置的逐项比对（`model.py:236-237`）。**运行期没有诊断量。** |
| **#2** 前缀与标记表面形式耦合，必须成对扫 | **成立，且被显式断言——但断言的是代理 prompt** | `model.py:224-235` 断言「每个码在 chat 前缀之后仍是单 token 且 id 不变」，这与 SemIf-OpenJev 的 `direct.py:42`、decider 的 `prompt.py:229-233` 是同一类做法（断言而非扫描）。一处必须说清：它用的前缀是 `[{"role":"user","content":"Choose an option."}]`，**不是真实 prompt**（真实 prompt 带 system 消息与图像占位）。所以「换模板就拒起」这句话**只对那个代理成立**——这一点本文未复算（需要 chat 模板渲染器）。 |
| **#3** 只读最后一个位置 | **改写为「读池化档选出的那一个位置」** | `pooling: "last"` 取 `hidden[:, -1]`（`model.py:307`），而 `-1` 之所以安全，是因为 tokenizer 被设成**左填充**（`model.py:210`）——不是因为它读的是「末位」。另一档 `"mean"` 在掩码上做平均（`model.py:298-305`），而加载器**拒绝与检查点不一致的档**（`model.py:198-199`）。「不物化词表维度」这半条成立。这里还多一个别家没有的形状：注意力是**非因果**的，「最后一个位置」不再是「唯一能看见全部前文的位置」（§4.1）。 |
| **#4** 比较概率前先确认 temperature | **成立，而且这条最好确认** | T 只有一个值、写在权重包里、加载期断言正有限（`decision_config.json:538` → `model.py:264-266`）。但要确认的**不止 T**：①`confidence` 是哪一层（服务面的机会校正值 vs 评测记录的 `p_max`，§10.3）；②跑的是哪个 `max_image_pixels`（默认 2048² 而发布数字是 512²，§8）；③`argmax` 是在概率上取的，所以「T 不改选择」是数学性质而非构造性质（§5、§10.3）。**ECE 是 15 个等宽箱**（`evaluate.py:19`）——与 Open-Jev、Laya 同口径，与 RYOTIDE / SemIf-OpenJev / Nimble / Jevlike 的 10 箱不同（comparison.md §5 第 2 条）。 |

## §12 数据与训练管线：发布的不只是权重

这一节是本文与另外十六篇最不同的一处：**上游把整条训练栈连同记录一起发布了**，所以训练配方是可以回源的
（但**不是**可以复算的——训练集本身没有随仓发布）。

### 训练配方的权威来源是 `train.sbatch`，不是代码默认值

`training/train.sbatch:24-32` 是那次运行真实用到的命令行，而它与 `train.py` 的参数默认值**有五处不同**：

| 参数 | `train.py` 默认 | `train.sbatch` 实传 |
|---|---|---|
| `--warmup-ratio` | **0.05**（`train.py:510`） | **0.15** |
| `--eval-every` | **20**（`:518`） | **50** |
| `--public-eval-every` | **100**（`:519`） | **50** |
| `--resume-every` | **100**（`:520`） | **50** |
| `--save-every` | 无默认（`None`） | **500** |
| `--attention-mode` | **`causal`**（`:527`） | **`noncausal_full_attention`** |
| `--pooling` | `last`（`:528`） | `last`（相同） |
| `--lr` / `--seed` / `--batch-size` / `--effective-batch-size` | 2e-6 / 20260920 / 32 / 256 | 相同 |

**`--attention-mode` 也在这一列里**——代码默认是因果，发布的那次是非因果。
这是 comparison.md §5 第 6 条那一类（「代码默认值 ≠ 发布配方」）的新一例，
而这一条的后果比「温度默认值不同」严重：默认跑起来**读法本身就不是被评测的那一套**，
且按模型卡自己的话（`README:46-47`）*Default causal inference does not reproduce this checkpoint's
evaluated setup*。`server.py` 不暴露这个开关（它从检查点读 `attention_mode`，`model.py:188-193`），
所以**服务面反而是安全的**；不安全的是直接用 `DecisionModel` 库入口的人——
那条路上 `attention_mode=None` 会回落到检查点里的值（`model.py:188-189`），
所以库入口也是对的。真正要小心的是**自己从基座起训**：`train.py` 的默认是因果。

### 数字

| 量 | 值 | 来源 |
|---|---|---|
| train rows | **626,033** = 95,930 original + 530,103 tasksource（task source 占 **84.68 %**） | `training/data-manifest.json` |
| steps | **2,446** = ceil(626,033 / 256) | `training/data-manifest.json`；本文复算（§15） |
| warmup steps | **366** = int(0.15 × 2,446) | `training/data-manifest.json`；本文复算（§15） |
| 开发折 / 温度折 | 1,497 / **983** | `training/config.json:62-63` |
| 世界大小 | 8（8×GPU，`train.sbatch:6`） | `training/config.json:68` |
| 发布步 | **2,446**（= 总步数） | `decision_config.json:2`、`release-manifest.json:5` |
| 参数量 | 26,085,330,160 + 读出头 1,305,600 | `model.safetensors.index.json` 的 metadata；本文复算（§15） |

### 选择策略在这次运行里**没有被用到**

`train.py` 的正常路径是**按开发折挑检查点**，而且有一道标定闸门
（`train.py:823-829`）：

```python
public_ok = calibration_ok(calibrated, reference)
eligible = public_ok and guard_ok
improved = (
    not diagnostic
    and eligible
    and (best is None or selection_key(calibrated, step) < selection_key(best, cast(int, best_step)))
)
```

其中 `calibration_ok` 要求本地 ECE 与 Brier 都不比参考实现差过 **0.010**
（`evaluate.py:20`、`:278-279`），`selection_key = (-accuracy, brier, step)`
（`evaluate.py:282-283`），参考实现是经 OpenRouter 打的 `typesafe/jev-1.13`（`evaluate.py:21`、`:510`）。

**但这次运行用的是 `--save-every 500`，于是上面那整段被替换掉了**（`train.py:830-831`）：

```python
if fixed_saving:
    improved = step == total_steps or (args.save_every is not None and step > 0 and step % args.save_every == 0)
```

`fixed_saving` 的定义在 `train.py:536`（`args.save_final_only or args.save_every is not None`）。
所以在这次运行里：

- 保存时机是「步数是 500 的倍数，或到了最后一步」——2446 命中后者；
- `calibration_ok` 闸门与 `selection_key` **都不参与选择**；
- `summary.json` 里 `best_step` 会被写成 `None`、`saved_step` 是 2446
  （`train.py:1060-1061`），`checkpoint_policy` 是 `"periodic_final"`（`:1062`）。

**结论要说准**：发布的就是**最后一步**的权重（`decision_config.json:2` 的 `step: 2446`
与 `release-manifest.json:5` 的 `step: 2446` 一致），而 `train.py` 里那套「按开发折挑 + 标定闸门」
在这条路径上**没有生效**。`evaluate()` 仍然每次跑、温度仍然被拟合（`:800-806`、`:832-842`），
只是**选择**这件事被换成了「取最后一步」。

### 五个 fold 与数据构建

`data.py` 从六个 HF 数据集（`data.py:35-42` 的 `SOURCES`，每个钉 revision）、三个归档
（`:43-55` 的 `ARCHIVES` + `ARCHIVE_SHA256`）与**十个经 Nimble 转换的 tasksource 任务**
（`:56-67` 的 `TASKS`）构建 fold，并且：

- **用 Nimble 里那一份已发布的 benchmark 子集清单来排除**（`data.py:537`）：
  `published = json.loads((dependency / f"docs/assets/public-benchmarks/subsets/{task}-manifest.json").read_text())`，
  然后按 `ids` 与 `family` 把已知 benchmark 内容连同**相关 family** 一起挡在训练外
  （`:569`、`:585`）。这正好接上 [`docs/nimble/`](../nimble/typed-decision-model.md) 那篇里
  逐文件核过的 `docs/assets/public-benchmarks/subsets/*-manifest.json`——**同一个制品，
  在这里是被当作训练集的排除清单消费的**。模型卡的致谢也照写这一点（`README:142-144`）。
- Nimble 本身按 `NIMBLE_REVISION = "d2387fc0b32d1173bfc995395c076a25a2a107c9"`（`data.py:24`）
  下载并写 `.revision` 文件，不符即报错（`:149`）。
- 图像侧的 1,000 行（`data-manifest.json` 的 `image_retention_rows`）是**程序化生成的**
  （`data.py:400-451`：两种颜色 + 三种形状画在 224×224 上，问「左边/右边物体的颜色/形状」），
  suite 名就叫 `image_retention`。所以视觉训练数据里有一部分是**几何题**，
  不是自然图像。

### 训练机制

- **损失**：`decision_loss`（`loss.py:13-18`），交叉熵 + 可选的 Brier 项，
  而 `--brier-weight` 发布值是 **0**（`train.sbatch:32`），所以实际只有交叉熵；
- **软目标**：`targets()` 支持三种（`train.py:346-364`）——`target` 是 list 时直接用、
  `noul` 转成 `[1 − p, p]`、其余是 hard one-hot；三种都要求非负、有限、和为 1（`:357-362`）；
- **优化器**：`CPUOffloadAdamW`（`optim.py:19-82`）——FP32 master 权重留在 **CPU**，
  `step()` 里把梯度拷下来、把参数拷回去（`:61-82`）。这是「27B 全量微调塞进 8 张卡」的做法；
- **学习率**：`learning_rate_factor`（`train.py:143-147`）——线性 warmup 后余弦退火到 **0.1**
  （不是退到 0）；
- **梯度裁剪**：`clip_grad_norm_(parameters, 1.0)`，而非有限即报错（`train.py:1004-1006`）；
- **可续跑**：`resume.pt` 存 optimizer 状态、RNG 状态、`best`/`best_step`/`selected_temperature`、
  以及五份数据的 sha256 与九份代码的 sha256（`train.py:890-916`）；续跑时**逐项比对**，
  任何一项不同即报错（`:741-753`）。代码哈希表是**手写的九项**（`train.py:663-678`），
  而且 `sft_pipeline.py` **只在走 `--schedule` 时**才被加进去（`:676-677`）——
  发布的 `training/config.json:47-57` 里确实只有九项、没有 `sft_pipeline.py`，
  与该次运行 `schedule: null`（`:6`）一致。

### 合成数据那半条流水线是**发布的、但不是这次跑的**

`synthetic.py`（用 `MODEL = "gpt-5.6-sol"` 调 LLM 造决策，`synthetic.py:16`）→
`sft_synthetic.py`（`COUNTS = {"nli": 18000, "intent": 10000, "numeric": 10000, "news": 8000,
"evidence": 4000}`，`sft_synthetic.py:24`）→ `sft_pipeline.py`（10 个 tranche、每批 5,000 新行 +
1,920 重放行，`sft_pipeline.py:19-22`）这条链**只挂在 `--schedule` 上**：
`train.py:39` import 了 `sft_pipeline`，但 `ScheduledRows` 只在 `args.schedule` 非空时构造
（`train.py:621-630`），而这次运行 `schedule: null`、sbatch 传的是 `--train …/train.jsonl`。
调用链是 `train.py → sft_pipeline.py → sft_synthetic.py → synthetic.py`，
所以**没有 `--schedule` 时这四个模块一行都不会执行**。

这条链**照发布**（`sft_synthetic.py` 1,141 行、`sft_pipeline.py` 737 行、`synthetic.py` 481 行、
`sft_numeric.py` 364 行，合计 2,723 行），并且带一整套内置守卫：tranche 的 readiness receipt
逐字段比对（`train.py:185-196`）、每份文件的 sha256 比对（`:199-206`）、
训练 ID 与留出 fold 不得重叠（`:163-164`、`:228-229`）。所以它是**一条完整但这次没走的路径**——
这与 comparison.md §4.5 里 Nimble 那条「整条旧读数连同它的诊断量一起被留在发布链外」
形状相似，但轻得多：这里没有「能力净损失」，只是这次训练没走那条路。

### 那个「验证记录」没有产生它的代码

`evaluation/checkpoint-verification.json` 是这份发布里最有分量的一件东西：

```json
{"passed": true, "step": 2446, "rows": 1497, "max_logit_delta": 0.0, "changed_answers": 0, ...}
```

它说的是「把发布检查点在 1,497 行开发折上重跑一遍，与训练时那一次**逐 logit 相同**
（`max_logit_delta` 恰好 **0.0**）、没有一个答案改变」。而 1,497 正好等于
`training/config.json:62` 的 `development_rows`。

两点必须说清：

1. **产生它的代码不在这个仓里。** 全仓对 `max_logit_delta` / `changed_answers` /
   `checkpoint_verification` 的搜索只命中 `release-manifest.json:12`（一个布尔）与它自己；
   `train.py` / `evaluate.py` 都不写这个文件。`train.sbatch:21` 提到过一个 `verify_inputs.py`，
   但它在仓库外（`/traindata/...`）。
2. **它跑的那份数据也不在仓里。** `dev.jsonl` 属于 gitignored 的训练数据，
   发布里没有任何 `.jsonl`。所以这条记录**无法被第三方复现**——它是**自陈**，
   而不是像 H2O-Lightning 的 `routing_reproducibility/` 或 Kev 的 `rows.json` 那样
   可以拿产物重算的东西。

本文对它的处理是：**只登记它说了什么，不把它当作证据**。「发布检查点与训练时逐 logit 相同」
这句话的可核对部分只剩一件事——`decision_config.json` 的 `step` 与 `release-manifest.json` 的 `step`
都是 2446（§15）。

## §13 文件对照表

| 文件 | 行数 | 职责 |
|---|---:|---|
| `source/src/autojev/model.py` | 359 | **读法本体**：码表构造与两道校验、`decision_messages` 的 prompt 构造、非因果钩子、`DecisionModel`（加载/池化/读出头/掩码/温度）、`answer()` 的三个题型公式、`save()` |
| `source/src/autojev/server.py` | 238 | FastAPI 服务面：四个路由、pydantic 契约（选项数 1–255 / 2–10）、图像校验、忙锁 529、`predict()`（8 行一批、`/model.temperature`、`output_tokens: 0` 字面量） |
| `source/src/autojev/evaluate.py` | 624 | 评测与标定：`evaluate_logits`（`softmax(logits/T)`）、`fit_temperature`（黄金分割 + 候选含 1.0）、`metrics`（15 箱 ECE / Brier / NLL / soft 指标）、`calibration_ok`、`selection_key`、`compare`（配对 bootstrap）、`remote_prediction`（打 OpenRouter 上的 Jev） |
| `source/src/autojev/train.py` | 1,084 | 训练主循环：fold 划分与不交叠校验、`augment`（每 step 打散选项）、软目标、`CPUOffloadAdamW`、续跑状态、评测/保存调度、`fixed_saving` 分支 |
| `source/src/autojev/loss.py` | 18 | `decision_loss`：选项 logits 上的交叉熵（+ 可选 Brier） |
| `source/src/autojev/optim.py` | 110 | `CPUOffloadAdamW`：FP32 master 留在 CPU、梯度下行 / 参数上行 |
| `source/src/autojev/distributed.py` | 85 | 单机 DDP 包装、三处 `all_reduce`、参数广播、按 rank 切 batch |
| `source/src/autojev/events.py` | 25 | `record()`：带 `fcntl` 排他锁的 JSONL 事件日志 |
| `source/src/autojev/types.py` | 83 | 全部 `TypedDict`：`Question` 三型、`DecisionInput`、`Example`、`Answer` 三型、`Usage`、`DecisionResponse` |
| `source/src/autojev/data.py` | 773 | 数据构建：六个 HF 源 + 三个归档 + 十个 tasksource 任务、Nimble 钉 revision 消费、按已发布 benchmark 清单排除、family 不交叠、程序化生成的 1,000 行图像题、token 计数 |
| `source/src/autojev/synthetic.py` | 481 | 用 `gpt-5.6-sol` 造合成决策（5 种 condition × 若干 task），带 usage 记账 |
| `source/src/autojev/sft_synthetic.py` | 1,141 | SFT 合成数据的槽位计划与生成（5 个任务族、共 50,000 行）、续跑与 provenance |
| `source/src/autojev/sft_pipeline.py` | 737 | 流式 tranche 流水线：10 个 tranche、readiness receipt、不可变写入、quick 子集 |
| `source/src/autojev/sft_numeric.py` | 364 | 数值题的表达式树（`Fraction` 精确求值、三种渲染） |
| `source/src/autojev/report.py` | 892 | matplotlib 仪表盘 / 影片：从 `events.jsonl` 画训练曲线与里程碑 |
| `source/src/autojev/__init__.py` | 0 | **空文件** |
| `source/pyproject.toml` | 65 | 包定义 |
| `config.json` | 146 | 骨干结构：64 层、每第 4 层 `full_attention`、`hidden_size` 5120、`vocab_size` 248,320、`mamba_ssm_dtype` float32 |
| `decision_config.json` | 541 | **发布契约**：255 个 `codes` + `token_ids`、`temperature` 1.0087417621345625、`attention_mode`、`pooling`、`step` 2446、以及训练期九份代码的 sha256 |
| `chat_template.jinja` | 169 | `enable_thinking=False` → 空思考块（`:165-166`）；不传时默认 xhigh 且前缀不闭合（`:46-48`、`:168`） |
| `processor_config.json` | 61 | 图像处理器：`patch_size` 16、`merge_size` 2（⇒ 32 px/token）、默认尺寸夹在 65,536–4,194,304 px |
| `tokenizer.json` | 19,989,492 B（LFS） | 码表的**真正来源**；本文在它上面重算过 `codes`/`token_ids`（§15） |
| `readout.safetensors` | 2,611,288 B（LFS） | BF16 `[255, 5120]` 读出头，不带 bias；尺寸算术见 §4.2、§15 |
| `model-0000{1..9},10,11-of-00011.safetensors` | 合计 52,170,802,640 B（LFS；逐文件相加，`release-manifest.json` 记的就是这组数；`model.safetensors.index.json` 的 `total_size` 52,170,660,320 是张量数据，差额是 11 个分片各自的头） | 骨干权重，**不含 `lm_head`** |
| `release-manifest.json` | 199 | 逐文件字节数 + sha256、`decision_index: 61.56`、`checkpoint_verification_passed: true` |
| `evaluation/checkpoint-verification.json` | — | 1,497 行开发折的复跑记录：`actual == expected`、`max_logit_delta` 0.0、`changed_answers` 0（产生它的代码与数据都不在仓里，§12） |
| `evaluation/decision-index.json` | — | Decision Index 0.2.1 的完整读数：5 个 area、44 个 benchmark（38 个计入）、`index` 61.56 |
| `training/config.json` | 69 | 那次运行的完整 `args` + 五份数据与九份代码的 sha256 + `train_rows` 626,033 |
| `training/data-manifest.json` | 382 | 数据侧记录：tasksource 530,103 / original 95,930、`steps` 2,446、`warmup_steps` 366、逐任务计数、`filter_complete: false` |
| `training/train.sbatch` | 32 | **真实用到的命令行**（与代码默认值有五处以上不同，§12） |
| `source/uv.lock` | 696,353 B | 锁文件（其 sha256 进了 `code_sha256`） |
| `requirements.txt` | 314 B | 运行时依赖（`transformers==5.17.0`、`torch==2.14.0`、`flash-linear-attention==0.5.2` 等） |

## §14 亲手跑一遍

本文**没有跑过**其中任何一条。下面这几条里，①–④ 是**不加载权重**就能做的核对（本文实际做的，
结果在 §15）；⑤–⑦ 需要 GPU 与 52 GB 权重，本文一条都没跑。

```bash
# ① 克隆（跳过 LFS；git-lfs 没装时也能拿到全部文本文件）
git clone https://huggingface.co/perplexity-ai/pplx-decider-v1.1-27b
cd pplx-decider

# ② 复合校验：manifest 里每个非 LFS 文件的 sha256 + 字节数
python3 - <<'PY'
import hashlib, json, pathlib
man = json.loads(pathlib.Path('release-manifest.json').read_text())
for name, meta in man['files'].items():
    p = pathlib.Path(name)
    if not p.exists(): print('missing', name); continue
    d = p.read_bytes()
    if d.startswith(b'version https://git-lfs'): print('lfs    ', name); continue
    print('ok ' if len(d)==meta['bytes'] and hashlib.sha256(d).hexdigest()==meta['sha256'] else 'DIFF', name)
PY

# ③ 在随仓的 tokenizer.json 上重算 255 个码（本文用的就是这一段）
python3 -m venv /tmp/tk && /tmp/tk/bin/pip install tokenizers
/tmp/tk/bin/python - <<'PY'
import itertools, json, string, pathlib
from tokenizers import Tokenizer
tk = Tokenizer.from_file('tokenizer.json')
single = [c for c in list(string.ascii_uppercase) +
          ["".join(p) for p in itertools.product(string.ascii_uppercase, repeat=2)]
          if len(tk.encode(c, add_special_tokens=False).ids) == 1]
codes, ids = single[:255], [tk.encode(c, add_special_tokens=False).ids[0] for c in single[:255]]
cfg = json.loads(pathlib.Path('decision_config.json').read_text())
print(len(single), codes[-1], codes == cfg['codes'], ids == cfg['token_ids'])
PY

# ④ 读出头与检查点的算术（不需要权重）
python3 - <<'PY'
import json, pathlib
man = json.loads(pathlib.Path('release-manifest.json').read_text())
hdr = json.dumps({"weight":{"dtype":"BF16","shape":[255,5120],"data_offsets":[0,255*5120]}},
                 separators=(',',':')).encode()
print("readout 推算字节:", 8 + len(hdr) + 7 + 255*5120*2,
      "| manifest:", man['files']['readout.safetensors']['bytes'])
idx = json.loads(pathlib.Path('model.safetensors.index.json').read_text())
print("张量数:", len(idx['weight_map']), "| 有 lm_head 吗:",
      any('lm_head' in k for k in idx['weight_map']), "|", idx['metadata'])
PY

# ⑤ 起服务（需 CUDA + 约 49 GiB 显存；服务面自己会读检查点里的 attention_mode/pooling）
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
AUTOJEV_CHECKPOINT=. python -m autojev.server          # PYTHONPATH=source/src

# ⑥ README 里那条最小调用（注意：这样跑用的是默认 2048×2048 图像上限，不是被评测的 512×512）
PYTHONPATH=source/src python - <<'PY'
from autojev.model import DecisionModel, answer
m = DecisionModel('.', device='cuda')          # 想复现评测口径：max_image_pixels=512*512
q = {"type":"choice","instructions":"Which team should handle this request?",
     "criteria":{"billing":"Charges and refunds","support":"Technical integration errors",
                 "sales":"Questions about buying a product"}}
p = m.predict([{"state":"My integration keeps failing. Please help.","question":q}])[0]
print(answer(q, p))
PY

# ⑦ 在推出的检查点上重跑开发折 —— 需要 dev.jsonl（未随仓发布），所以 §12 那条验证记录不能这样复现
PYTHONPATH=source/src python -m autojev.evaluate --data dev.jsonl --local --checkpoint . --output /tmp/dev.json
```

三处「跑之前先知道」：

- **`--attention-mode` 与 `--pooling` 的库默认不是发布值**（`causal` / `last` 对
  `noncausal_full_attention` / `last`，`train.py:527-528`）。读已发布检查点时加载器会拒绝不一致
  （`model.py:192-193`），但**自己起训**时默认跑的是因果版本，不产出被评测的那套读法。
- **`max_image_pixels` 默认 2048×2048，而被训练与被评测的是 512×512**（`README:107-110`）。
- `transformers==5.17.0` 与 `torch==2.14.0` 是钉死的（`requirements.txt`），
  而 `model.py:15-19` 依赖 `Qwen3_5ForConditionalGeneration` / `Qwen3_5Model` /
  `Qwen3VLProcessor` / `create_recurrent_attention_mask` 这四个来自很新版本 transformers 的符号。

## §15 实例走查：在 committed 产物上的复算

本节全部是**不运行任何 `autojev` 代码**的核对——标准库算术，加上一次用第三方 `tokenizers` 库
对随仓 `tokenizer.json` 的重算。**没有任何前向产生的数值。**

### 15.1 `release-manifest.json` 的覆盖范围

这个 revision 的树里有 **48 个文件**，而清单收录 **46 个**——漏掉的正是它自己
（一个文件装不下自己的哈希）与 `.gitattributes`。所以清单覆盖 46/48，下面按这 46 条核：

| 类别 | 数量 | 结果 |
|---|---:|---|
| 非 LFS 文件，逐字节核过 | **32** | sha256 与字节数**全部相符** |
| LFS 指针（11 个权重分片 + `readout.safetensors` + `tokenizer.json`） | 13 | 只核到指针；其中 `tokenizer.json` **另行下载后核对通过** |
| 与清单**不符** | **1** | `README.md`（见 15.2） |

`tokenizer.json` 单独说：按 revision `14c9d901…` 下载后是 **19,989,492 字节 /
`6f32ce20dc35f57a7f9ad1eac03525bd7d30f9df8cea6507e958279cc3657706`**，
与清单记的**逐字节相符**，也与 LFS 指针里的 `oid sha256` 相符。所以分词器本体是可核的；
**11 个权重分片与 `readout.safetensors` 的内容本文未核**（52 GB）。

### 15.2 清单过期的那一条：`README.md`

```
release-manifest.json:22-25   README.md   5696 B  265e16155cbc9c65bcf2a797521b8214eb659cf327f7bbed19ddcc298491eae4
仓内 HEAD 的 README.md        6712 B  1cc6f24b69b187db2d8de8e5a39e8193afbfa0356afcd02c820cf1240d428c0a
git show 6195aa5:README.md    5696 B  265e16155cbc9c65bcf2a797521b8214eb659cf327f7bbed19ddcc298491eae4   ← 与清单相同
```

也就是说：清单里那一行对应的是**倒数第二个提交 `6195aa5`（Raise image cap from 512x512 to
configurable max_image_pixels）时的 README**，而 HEAD 是 `14c9d90`（Credit Bespoke Nimble in the
model card）——那次提交改了 README，清单没有跟着重算。所以**清单与模型卡不是同一次生成的**，
这与 OpenJev(HF) 那处 `MANIFEST.json` / `SHA256SUMS` 里 `README.md` 过期是**同一类问题**
（comparison.md §5 末尾那一列）。判据是机械的：`git show <commit>:README.md | shasum -a 256`。

### 15.3 同一份发布里 `model.py` 有两个哈希

`decision_config.json:12` 的 `provenance` 与 `release-manifest.json` 各自给了一份 `model.py` 的 sha256，
而它们**不同**：

| 文件 | `decision_config.provenance`（训练期） | `release-manifest` / 仓内实际 | 相同？ |
|---|---|---|---|
| `train.py` | `33b7c1447abb…` | `33b7c1447abb…` | ✓ |
| **`model.py`** | **`824721da4d8f…`** | **`02940963e9fb…`**（仓内文件重算也是它） | **✗** |
| `optim.py` | `68b1b30c7fce…` | `68b1b30c7fce…` | ✓ |
| `evaluate.py` | `f293a50320ab…` | `f293a50320ab…` | ✓ |
| `types.py` | `250ba2eea1cd…` | `250ba2eea1cd…` | ✓ |
| `events.py` | `72315600ac07…` | `72315600ac07…` | ✓ |
| `distributed.py` | `d8e5bf5c0e97…` | `d8e5bf5c0e97…` | ✓ |
| `loss.py` | `2e4f4ae4b8aa…` | `2e4f4ae4b8aa…` | ✓ |
| `uv.lock` | `cba79e0f2caa…` | `cba79e0f2caa…` | ✓ |

**九个文件里八个两边一致，只有 `model.py` 不一致**——而这正是模型卡第 114–117 行承认的那一处改动
（图像上限从硬编码 512×512 变成可配置的 `max_image_pixels`）。所以这不是笔误：
`decision_config.provenance` 记的是**训练时那一版** `model.py`，`release-manifest` 记的是**发布的那一版**，
同一份发布里两个哈希各自都是对的，**它们描述的是不同时刻的同一个文件**。
换句话说：发布代码**不是**训练代码，差别被记录下来了，而且**能这样被机械地验出来**。

### 15.4 在随仓 `tokenizer.json` 上重算那 255 个码

用第三方 `tokenizers` 0.23.3（**不 import `autojev`**），按 `model.py:214-221` 的三行逻辑重算：

| 量 | 重算结果 | 与 `decision_config.json` |
|---|---:|---|
| 候选数（26 + 676） | 702 | — |
| 能单 token 化的候选 | **588** | — |
| 取前 255 个 → `codes` | `A`…`JT` | **逐项相同** ✓ |
| `token_ids` | 32–57 起，全部互异 | **逐项相同** ✓（255 个 id 无一重复） |

边界与「落选原因」也一并算出来了：

- 最后一个码是 **`JT`**；下一个合格的单 token 对是 **`JU`**，它落选**仅仅因为名额满了**；
- 在 `AA`–`JT` 之间有 **25 个**双字母对不是单 token：
  `BQ BZ CJ CQ CZ DQ DZ EJ EY FJ FQ FV FZ GJ GK GQ GZ HJ IY JF JG JH JL JN JQ`；
- `KA`–`ZZ` 之间还有 **333 个**合格的单 token 对**从未被使用**；
- 所以码表的构成是 **26 个字母 + 229 个双字母对 = 255**。

**未复算**：`model.py:224-235` 的第二道检查（码在 chat 前缀之后仍单 token）。
它需要 transformers 的 chat 模板渲染器，本机没有——而且它验的是代理 prompt 而不是真实 prompt（§2）。

### 15.5 读出头与检查点的算术

| 量 | 算术 | 与记录 |
|---|---|---|
| `readout.safetensors` 字节数 | 8（长度前缀）+ 73（最小头 JSON）+ 7（补到 8 字节对齐）+ 255×5120×2 = **2,611,288** | `release-manifest.json` 记 **2,611,288** ✓ |
| 该张量 | `BF16`、`shape [255, 5120]`、无 bias | 与 `README:39` 的 *BF16 `[255, 5120]`* 一致 ✓ |
| 检查点张量数 | 1,183 = `language_model` 850 + `visual` 333 | — |
| 含 `lm_head` 的键 | **0** | 与 `README:50` 的 *not a full-vocabulary `lm_head`* 一致 ✓ |
| 参数量 | 26,085,330,160 + 读出头 255×5120 = 1,305,600 → **26,086,635,760** | `model.safetensors.index.json` 的 metadata ✓ |
| 每参数字节 | 52,170,660,320 / 26,085,330,160 = **2.0** | ⇒ BF16 ✓ |

所以模型卡第 39 行与第 50 行那两句 —— **发布的是一个独立读出头、词表维度不参与** ——
是**可以在不下载任何权重的前提下**核对出来的。

### 15.6 评测与索引记录上的标准库算术

| 复算 | 做法 | 结果 |
|---|---|---|
| 15 箱 ECE | 用 `evaluation/checkpoint-verification.json` 的 15 个 `reliability` 箱按 `evaluate.py:244-248` 的式子重算 | **0.015921474840787767**，与记录**逐位相同** ✓ |
| accuracy | 用同一组箱的 `count × accuracy` 加权和 | **0.9044756179024717**，与记录**逐位相同** ✓ |
| 行数 | 15 个箱的 `count` 相加 | **1,497** = 记录的 `count` = `training/config.json:62` 的 `development_rows` ✓ |
| Decision Index | 用 `evaluation/decision-index.json` 的五个 area `skill` 按 `note` 里写明的权重（arts 10 %，其余四个按 benchmark 数的平方根分 90 %：知识 25.849 % / 语言 25.849 % / 检索 20.023 % / 工具 18.278 %）重算 | **61.5566 → 61.56**，与 `index` / `scores.balanced_skill` **相同** ✓ |
| raw index | 同一组权重的 `raw` 列 | **70.682 → 70.68**，与 `raw_index` / `scores.balanced_raw` **相同** ✓ |
| benchmark 计数 | `benchmarks` 字典里 `in_index == true` 的条目 | **38**，与 note 的「38 benchmarks in five areas」一致；字典共 44 个 ✓ |
| 模型卡五个类目 | `README:25-29` 的 48.18 / 69.45 / 61.26 / 78.88 / 44.66 | 与 `areas[].skill`（0.4818 / 0.6945 / 0.6126 / 0.7888 / 0.4466）**相同** ✓ |
| 步数与 warmup | ceil(626,033 / 256) = **2,446**；int(0.15 × 2,446) = **366** | 与 `data-manifest.json` 的 `steps` / `warmup_steps` 和 `decision_config.json:2` 一致 ✓ |
| 五份数据哈希 | `decision_config.json` 的 `provenance` 里 `development` / `temperature` / `reference` / `train` / `public` 五个 | 与 `training/config.json:40-46` 的 `data_sha256` **五项全同** ✓ |

两处与「验证记录」有关、但**不能**这样核的：

- `max_logit_delta: 0.0` 与 `changed_answers: 0` **本文无法核实**：产生这份记录的代码不在仓里，
  它跑的那 1,497 行 `dev.jsonl` 也不在仓里（§12）。本文只登记它自陈了什么。
- `evaluation/decision-index.json` 里 44 个 benchmark 只有**分数与计数**，没有逐题记录，
  所以「模型在 JevBench 上到底答了什么」**无法回源**——这与 H2O-Lightning 有 3,474 行逐题读数、
  Kev 有 `rows.json` 的处境不同。

### 15.7 本文没有核实的东西

一律不给数值：

- **一切前向产生的量**：任何概率、任何准确率、任何 `marker_mass` 的类比量、非因果与因果两条路径的差异。
  需要 GPU 与 52 GB 权重。
- **那 11 个权重分片与 `readout.safetensors` 的内容**（只有指针）。
- **`model.py:224-235` 的第二道单 token 检查**（需要 chat 模板渲染器）。
- **`checkpoint-verification.json` 那两条自陈**（产生它的代码与数据都不在仓里）。
- **`training/data-manifest.json` 的 `filter_complete: false`**：它自陈过滤没跑完
  （`filtered_rejected_rows: 11629`、`filtered_unreviewed_rows_excluded: 1`、
  `retained_partial_groups: 2322`），而训练数据本身没发布，所以那些计数**无法复核**。
- **顺序偏置**：这条链上没有任何度量（§6），所以「它对选项顺序敏不敏感」连一个可引用的数都没有。
- **`README:19` 那句「Decision Index 从 56.4 提到 61.56、超过 Jev 3.5 分以上」的前半段**：
  61.56 在 `evaluation/decision-index.json` 里可核（上文），但 56.4 与前作
  `perplexity-ai/pplx-decider-v1-27b` 的读数**不在这个仓里**，未核。

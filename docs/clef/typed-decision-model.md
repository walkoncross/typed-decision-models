# Clef 是如何把 schema 文本同时当作标签集与读位置的

> 代码库：<https://huggingface.co/Cloudflare/clef>
> 对应 README 第 26–29 行：*Clef is a 27B multimodal model that turns a state and a schema of typed
> questions into decisions. It reads the state as text, JSON, images, or video, and returns a
> probability for every allowed option of every question in a single forward pass. There is no
> free-form text generation and no output parsing.*
> 以及 README 第 31 行：*The Clef API is fully compatible with Jev and SystemOne.*
> 本文按 `joint_schema_model.py`（唯一的实现文件）、`joint_head_config.json`、`config.json`、
> `chat_template.jinja` 与发布的 `joint_head.safetensors` 的实际内容逐环节说明这两句话的实现方式；
> 文中出现的文件名与行号（`joint_schema_model.py:381`、`joint_head_config.json:2`、`chat_template.jinja:166`
> 等）均相对该仓库根目录。
>
> **同名配对文档**：RYOTIDE 的 [`docs/ryotide/typed-decision-model.md`](../ryotide/typed-decision-model.md)。
> §0 先讲清分歧，§12 把它的四条不变量逐条落位。
>
> **本文读的是哪一个 revision**：`Cloudflare/clef` 的 `0b331204bb13fbd2ca93a64df1956f5b55478ce5`
> （2026-10-09T18:46Z，25 个文件）。该仓**没有 Git 之外的代码来源**：这个 HF 模型仓就是上游本体，
> `joint_schema_model.py` 只在这里发布过一份。**54.7 GB 的骨干权重未下载**；本文引用的权重数字只有两类——
> `joint_head.safetensors`（256 MB，已完整下载）与 `model.safetensors.index.json`（111 KB）的内容，
> 外加一个 19 MB 的 `tokenizer.json`。§15 给出复算，§16 列出未核项。

Clef 与 RYOTIDE 同属「把 LLM 当分类器读」这一支：一次前向、不生成任何 token、输出封闭在已知选项集上。
它往前推的是另外四件事：

1. **标签集与读位置是同一段文本**。请求里的 `criteria` 被渲染成 prompt 中段的 `SCHEMA FIELDS:` 段
   （`joint_schema_model.py:110`、`:116`、`:130`），头再按这段文本**自己的 token 下标区间**去池化——
   `question_span` 与 `option_spans` 在拼 prompt 时算出来（`:119`、`:136`、`:171-187`）。
   于是「这个选项是什么」与「去哪里读它」不是两件事，而是**同一个下标的两种说法**（§2、§3）。
2. **「typed」不要求单 token**。选项 id 是任意字符串，渲染进 JSON 后可以占多个 token——
   本文实测 `overdue` 是 **2 个 token**（§2）。所以 255 / 52 / 26 这些上限在这条链上**没有对象**。
3. **置信度是两项之和**：`logit = prior + σ(g)·joint`（`:455-457`），其中 `prior` 是**只有 1 个可训练参数**
   的余弦项——拿**骨干自己的输出嵌入矩阵**取选项文本的均值，与「问题向量 + 全序列末位」算余弦（`:433-439`）。
   RYOTIDE 那条零样本读法没有被丢掉，它以**加性先验**的形式留在头里（§4、§9）。
4. **发布物里没有任何身份指纹，也没有服务代码**。`systemone()` 回声的是**调用方自己发的 `model` 字符串**
   （`:573`），仓里没有 `/health`、没有 `/v1/version`、没有 prompt 哈希、也没有 HTTP 服务器——
   HTTP 层在 SGLang 的 `dev-clef` 镜像里，不在本仓（§8、§12）。

## §0 与 RYOTIDE 的分野

| 维度 | RYOTIDE | Clef |
|---|---|---|
| 权重来源 | 公开 LLM，全程 frozen | `Qwen/Qwen3.8-27B` **post-train 后的骨干**（README 第 33 行）+ 一个 **1.28 亿参数**的 `joint_head.safetensors`（§13） |
| 是否训练 | 否，零新增参数 | 是：训一个头；骨干也被 post-train（训练代码不在发布仓里）（§16） |
| 「typed」从哪来 | 分词器：标签必须单 token | **请求自己给的字符串**，渲染成 schema 文本；**不要求单 token**（§2） |
| 读位置怎么定 | 运行期标定（扫前缀、取 marker mass 最高） | **不标定**：`encode_record` 拼 prompt 时算出的下标区间（§3） |
| 一次前向读什么 | 末位词表 logits，按标记 id 取出 | 五路：全序列当 memory 做注意力、每题的 `question_span` 均值、每选项的 `option_span` 均值、每选项 span 的**词嵌入**均值、以及末位全局向量（§4） |
| 掩码 + 重归一化 | 全词表 softmax → 掩码到标记 → 重归一化 | **没有掩码这一步**：头输出恰好 K 个标量，再在这 K 个上 softmax（§5） |
| 词表维度 | 每次都物化（这就是 `marker_mass` 的对象） | **从不作为 logits 物化**，但**作为嵌入被用到**：`output_embedding_weight[token_ids].mean(0)`（`:381-382`）（§4、§12） |
| 置信度从哪来 | 读出来的 | **两项之和**：一个 1 参数的词嵌入余弦先验 + 一个训练出来的联合项；两个尺度都 clamp 到 ≤100（§9） |
| 温度 | 唯一的数值旋钮 | **不存在**：全文件 `temperature` 零命中；尺度是权重的一部分（§9） |
| 顺序偏置 | 多 order 共享前缀 fork，概率平均 | `choice` 的选项在**渲染期按 id 排序**，调用方的顺序进不了 prompt；`score` 保序（§6） |
| 多问题 | 公共前缀预填充一次 | 所有问题在**一次前向**里，而且**题与题互相可见**（fields 组成一条序列做自注意力）（§7） |
| `output_tokens` | 0 | 0，且是响应组装处的字面量（`:575`） |

一句话总结立场：**它把「标签集从哪来」与「读位置在哪」合并成同一段自造文本，再用一个训练出来的头去读它；
代价是这段文本的排版从此是权重契约的一部分，而发布物里没有任何东西把这份契约记下来。**

## §1 端到端数据流

```
systemone(model, processor, request)                    joint_schema_model.py:547
    request = {"model", "state", "questions"{qid:{type,instructions?,criteria?}},
               "images"?, "videos"?, "media_kwargs"?}
    │
    ├─ 校验  :553-561        model 必须是 str 且有 state；questions 非空 dict；
    │                        每题 type ∈ {noul, choice, score}；非 noul 必须给 criteria
    │
    ├─ encode_record(tok, request, max_length=16384, processor)   :103
    │   ├ question_options(q)  :46-57
    │   │    noul  → [("true", 默认真描述), ("false", 默认假描述)]，请求可覆盖描述
    │   │    choice→ sorted(criteria.items())        ← 按 option_id 排序,不看请求顺序
    │   │    score → [(str(i), v) for i,v in enumerate(criteria)]
    │   ├ schema_ids  :110-139   "\n\nSCHEMA FIELDS:\n"
    │   │    每题: "\nFIELD k\nID: {qid}\nTYPE: {type}\nINSTRUCTION: " + render(instructions)
    │   │          "\nALLOWED OPTIONS:\n"
    │   │          每选项: "OPTION k: " + render({"option_id":…,"description":…}) + "\n"
    │   │          "END FIELD\n"
    │   │    ＊ question_span / option_spans 在这里被记下（:119、:136）——这就是读位置
    │   ├ prefix_ids  :150-153  "<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n<|im_start|>user\nSTATE:\n"
    │   ├ media_ids   :158-161  若带图/视频，插在 STATE:\n 之后、state 文本之前
    │   ├ suffix_ids  :154-157  "\n<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\nJOINT SCHEMA DECISIONS:"
    │   ├ state_ids   :162        render(record["state"])
    │   ├ 长度        :163-170   fixed_length = |prefix|+|schema|+|suffix|;
    │   │                        fixed_length > max_length → 抛错（响）;
    │   │                        state 超长 → 从右截断（静默）
    │   ├ 平移        :171-187   全部 span += |prefix|+|state|
    │   └ input_ids = prefix + state + schema + suffix        :188
    │
    ├─ collate_records([encoded], pad_id, device)  :199
    │     右填充 + attention_mask（1/0）；media 拼接；mm_token_type_ids 放到 token_offset
    │
    ├─ model(batch)  :468        ← 一次前向，use_cache=False（:481）
    │   ├ text_model = base_model.model（纯文本时再降到 .language_model，:476-477）
    │   └ head(last_hidden_state, input_ids, attention_mask, records,
    │           base_model.get_output_embeddings().weight)      :485-491
    │
    └─ 每题 softmax(-1)  :568  →  systemone_answer(q, probs)  :523
         noul   → {"type":"noul","noul": round(P(true),4)}          :526
         choice → argmax + confidence=round(p,4) + probabilities    :527-535
         score  → score=round(Σ i·pᵢ,4) + confidence=round(max p,4) :536-543
       → {"model": request["model"], "answers": {...},
          "usage": {"input_tokens": len(encoded.input_ids), "output_tokens": 0}}   :572-576
```

两处值得先记下：**state 在 schema 之前**（`:188`），而所有读位置是 state 之后的 schema 段；
**`usage.input_tokens` 数的是整条 prompt 一次**（`:575`），多问题不重复计 state。

## §2 「typed」从哪来：选项是请求给的字符串，不要求单 token

RYOTIDE 的「typed」靠分词器：标签必须恰好是一个 token，否则得退化成字母标记。
Clef 这条链上**没有这个约束**，因为它根本不从词表里挑标签——**标签是请求给的字符串，被写进 prompt**。

`question_options()`（`:46-57`）是唯一的「标签集构造处」，它按题型给出 `(option_id, description)` 列表：

| 题型 | option_id 从哪来 | 顺序 |
|---|---|---|
| `noul` | 固定 `("true","false")`；`criteria` 若给了则**覆盖描述**（`:49-54`） | 固定 |
| `choice` | `criteria` 这个 dict 的 **key**（`:56`） | **按 id 排序**（见 §6） |
| `score` | `criteria` 这个 list 的**下标字符串** `"0"`,`"1"`,…（`:57`） | 保序 |

然后每个选项被渲染成一行 JSON，写进 schema 段（`:129-138`）：

```python
schema_ids.extend(_tokens(tokenizer, f"OPTION {option_index + 1}: "))   # :130
option_start = len(schema_ids)                                          # :131
semantics = {"option_id": option_id}                                    # :132
if description is not None:
    semantics["description"] = description                              # :134
schema_ids.extend(_tokens(tokenizer, render(semantics)))                 # :135
option_spans.append((option_start, len(schema_ids)))                     # :136
```

`render()`（`:35-43`）对非字符串走 `json.dumps(..., ensure_ascii=False, separators=(",",":"), sort_keys=True)`——
紧凑、键排序、不转义非 ASCII。所以 README 那个例子里的选项行**逐字节确定**。本文按这段代码自行重拼了一遍
（不 import 上游、不加载权重），得到的 schema 段解码如下：

```
\nFIELD 1\nID: status\nTYPE: choice\nINSTRUCTION: What is the invoice status?\nALLOWED OPTIONS:\n
OPTION 1: {"description":"Not sent.","option_id":"draft"}\n
OPTION 2: {"description":"Invoice is past due.","option_id":"overdue"}\n
OPTION 3: {"description":"Invoice is paid.","option_id":"paid"}\n
END FIELD\n
\nFIELD 2\nID: large\nTYPE: noul\nINSTRUCTION: Is the total above 1000 USD?\nALLOWED OPTIONS:\n
OPTION 1: {"description":"The proposition is true or the answer is yes.","option_id":"true"}\n
OPTION 2: {"description":"The proposition is false or the answer is no.","option_id":"false"}\n
END FIELD\n
```

三件事在这里可见，都是「typed」这条链的性质，而不是文档里的说法：

1. **同一段文本里，标签是 `option_id`，被读的位置是整行 JSON**。`option_spans` 圈的是
   `{"description":…,"option_id":…}` 整段（`:136`），不是 id 那三个字符。所以**描述文本也进读数**
   ——`lexical` 那一项取的就是这一段所有 token 的嵌入均值（`:379-383`）。改描述就是改分数。
2. **选项 id 可以是多 token**。本文用发布的 `tokenizer.json`（`transformers` 未参与，只用 `tokenizers`
   加载这一个文件）实测：`paid`→`[33530]`、`draft`→`[43494]`、`true`→`[1802]` 是单 token，
   而 **`overdue`→`[1908, 22906]` 是 2 个 token**。这条链上没有任何地方要求它单 token。
   所以 decider 的 255、OpenJev(HF) 的 52、SemIf-OpenJev 的 16、Nimble 的 26、H2O-Lightning 的 255
   这些上限——**它们量的是「从词表里挑标签」这件事的容量，而这里没有这件事**。
3. **special token 是构造出来的，不是拼出来的**。`<|im_start|>` / `<|im_end|>` / `<|endoftext|>` 在
   `tokenizer.json` 里各是**一个** added token（实测 id 248045 / 248046 / 248044），
   与 `tokenizer_config.json` 的 `eos_token` / `pad_token` 逐一对上（§15）。

**代价**：「typed」这一层没有任何守卫。选项数与 `option_spans` 的个数恒等（构造使然），
但**选项文本能不能被正确读到**完全依赖那段 schema 的排版——而排版是手写在 Python 字符串里的。

## §3 读位置：`encode_record` 拼 prompt 时算出来的下标区间

RYOTIDE 要在运行期**找**读位置：扫 9 个前缀 × 2 种表面形式，取 marker mass 最高的那个。
Clef **没有「找」这个动作**，因为位置不是被找出来的，是**拼 prompt 的时候顺手记下来的**：

```python
question_start = len(schema_ids)                 # :119
...
schema_ids.extend(_tokens(tokenizer, render(instructions)))   # :123
question_end = len(schema_ids)                   # :124
...
option_start = len(schema_ids)                   # :131
schema_ids.extend(_tokens(tokenizer, render(semantics)))
option_spans.append((option_start, len(schema_ids)))          # :136
```

这些下标是**相对 schema 段**的，拼完之后统一平移（`:171-187`）：

```python
schema_offset = len(prefix_ids) + len(state_ids)                 # :171
option_spans = tuple((s + schema_offset, e + schema_offset) ...)  # :180-183
```

于是最终落在 `input_ids` 上的区间是绝对下标。头拿到的就是这个（`:344-351`）：

| 读什么 | 怎么算出来 | 行 |
|---|---|---|
| memory（全序列） | `sequence_hidden[:sequence_length]`，`sequence_length = attention_mask.sum()` | `:355-357` |
| 全局向量 | `sequence_hidden[-1]` —— **真正的末位** | `:358` |
| 每题的问题向量 | `question_span` 上的**均值池化** | `:359-364` |
| 每选项的上下文向量 | `option_span` 上的**均值池化** | `:369-378` |
| 每选项的词法向量 | `output_embedding_weight[input_ids[start:end]].mean(0)` | `:379-383` |

三点必须说清楚：

1. **这不是「读一个位置」，是读五类位置，而且其中一类是整条序列**。落在 RYOTIDE 第 13 节 #3 上时，
   这条要改写（§12）。它与 Kev 的「位置是 `encode` 算出来的」同源，但 Kev 读的是 K 个 `</opt>` 端点，
   这里读的是**每题一段 + 每选项一段 + 一个末位 + 一次全序列注意力**。
2. **「不物化词表维度」这条在这里半成立，而且方式与另外几条都不同**。头确实从不把 hidden state 投影到
   248,320 维的词表上（所以 `marker_mass` 没有对象），但**词表矩阵本身是被用的**：
   `base_model.get_output_embeddings().weight`（`:490`）——`tie_word_embeddings: false`
   （`config.json:116`），所以这是一张独立的 `[248320, 5120]` 矩阵（按 config 推算 ≈ 2.54 GB BF16），
   代号取它的**若干行**求均值，当**嵌入**用，从不当 logits 用。这是这条路线最容易被误读的一处：
   仓库里出现 `output_embedding_weight` 不等于「读了词表分布」。
3. **位置正确性没有任何守卫**。位置是绝对下标，所以**填充只能在右端**——而 `collate_records`
   （`:204-219`）确实是右填充，`sequence_hidden[:sequence_length]`（`:355-356`）也确实剥的是右端。
   但这个耦合**只体现在两处代码恰好配合**：`encode_record` 产出的 span 是绝对下标，
   而「绝对下标 ⇒ 只能右填充」这句话在仓里**没有断言、没有注释、也没有诊断量**。
   对照 Nimble 把 prompt 源码的 sha256 钉进权重契约、decider 把布局写进 `decider_config.json`：
   这里连一句「别改排版」都没有。

## §4 一次前向：头把五路信息拼成一个 logit

一次前向（`:468-491`）之后，全部工作在一个 1.28 亿参数的头上（§13）。头的公式可以完整写成五行：

```
# 1. 全序列当 memory，供后面的注意力用                                  :357
memory   = memory_projection(sequence_hidden)                        # [1, L, 1024]

# 2. 五路特征                                                            :358-398
global   = sequence_hidden[-1]                                       # 末位
qvec[q]  = mean(sequence_hidden[question_span_q])                    # 每题一段
ctx[q,o] = mean(sequence_hidden[option_span_qo])                     # 每选项一段
lex[q,o] = mean(W_out[token_ids_of_option_span_qo])                  # 词嵌入均值
opt[q,o] = ctx_proj(ctx) + lex_proj(lex) + q_proj(qvec[q])           # 选项 query

# 3. 证据路由 + 选项汇总                                                  :399-420
opt      = evidence_layers(opt, memory)            # 2 层,跨注意力读全序列
field[q] = q_proj(qvec[q]) + option_summary_norm(Σ_o softmax(opt·field/√d)·opt)
           + global_proj(global) + type_embedding(type_id[q])
field    = decoder_layers(field, memory)           # 4 层;fields 之间互相自注意力

# 4. 两个打分项                                                           :433-454
anchor        = normalize(qvec[q] + global)
prior_scale   = exp(clamp(prior_logit_scale, max=log(100)))          # 训练出来的
prior[q,o]    = prior_scale · cos(normalize(lex[q,o]), anchor)
joint_scale   = exp(clamp(joint_logit_scale,  max=log(100)))
joint[q,o]    = joint_scale · cos(field[q], option_norm(opt[q,o]))
                + residual_scorer([field, opt, field*opt, |field−opt|])

# 5. 合成                                                                 :455-457
logit[q,o] = prior[q,o] + sigmoid(residual_gate) · joint[q,o]
```

几处只有这条路线才有的东西：

- **第 4 步的 `prior` 项只有 1 个可训练参数**（`prior_logit_scale`，`:335`）。
  其余部分——取哪几行词嵌入、怎么平均、跟谁算余弦——**没有任何可训练参数**。
  所以 RYOTIDE 那条零样本读法在这里没有被废掉，它变成了头内的一条**加性先验**：
  「把选项文本的词嵌入平均一下，跟『问题 + 整条 state 的末位』算余弦」。
  Jeff 的做法是把训练起点对到零样本读法（`slice_init_`），Clef 的做法是**让零样本那一项一直留在和里**。
  这两条是同一个念头的两种落法，是并读第十四、第十五篇才看得见的对照。
- **两个尺度用的是同一个 idiom，而且都在权重里**：`exp(clamp(·, max=log(100)))`（`:438`、`:453`）。
  本文实测发布权重里 `prior_logit_scale == joint_logit_scale == 2.296875`（bf16），
  于是两个尺度都是 **9.94306**（§15）。CLM 那条路线用的是同一个 `clamp(max=100)` 写法
  （`heads.py:95`），但那是它自己的头；这里是 Clef 的头。
- **`sigmoid(residual_gate)` 是合成时的混比**（`:337`、`:456`）。本文实测发布权重里它是
  **0.119203**（`residual_gate = −2.0`，§15.2），也就是说**联合项的那条余弦被压到先验项余弦的 11.9 %**。
  唯一可能改变这个结论的是 `residual`——它加在门**里面**（`:454`、`:456`），量纲无界，
  相对大小取决于训练后那两层 MLP 的实际输出，**需要一次前向才能定**，本文没跑（§16）。
- **头一次前向把全部问题一起算完**，所以延迟与题数无关的部分很大（一个 27B 骨干 + 一个 1.28 亿的头），
  而 `usage.input_tokens` 只数一次 prompt（`:575`）。

## §5 掩码 + softmax：没有掩码这一步，只有 K 个标量

RYOTIDE 的读法是「全词表 softmax → 掩码到标记 → 重归一化」。Clef 上**这三步一步都不存在**，
理由比 decider / Nimble / Kev / H2O-Lightning 更靠前：**头的输出个数就是 `len(option_spans)`**。

全流程只有两次 softmax，都很小：

| 位置 | 在哪 | 维度 | 做什么 |
|---|---|---|---|
| 选项汇总 | `:408-411` | **选项维**（每题 K 个） | `softmax(opt·field/√d)` 当权重，把 K 个选项压成一个摘要向量 |
| 最终读数 | `:568` | **选项维**（每题 K 个） | `softmax(question_logits)` ——这就是发布出去的概率 |

```python
answers = {
    question.question_id: systemone_answer(
        questions[question.question_id],
        dict(zip(question.option_ids, question_logits.float().softmax(-1).tolist())),
    )
    for question, question_logits in zip(encoded.questions, logits)
}                                                                    # :565-571
```

`.float()` 是唯一一处精度处理：头在 BF16 下算，softmax 在 FP32 里做。**没有温度、没有 bias、
没有地板、没有重加权**——全文件对 `temperature` / `calibrat` / `top_k` / `top_p` / `sample` / `argmax`
零命中（`mask` 只出现在 `attention_mask` 里）。

所以 §12 的 #1 在这条链上是**概念不适用**，而且理由与 Kev / decider / Nimble / H2O-Lightning 都不同——
它们的词表维度「没有出现」，这里的词表矩阵**出现了但只当嵌入用**（§3）。

## §6 顺序偏置：渲染期把选项排序

这是本条路线最干净的一处，也是最容易被忽略的一处：**调用方的选项顺序进不了 prompt**。

```python
if question_type == "choice":
    return sorted((str(key), value) for key, value in question["criteria"].items())   # :56
```

`choice` 的选项**按 `option_id` 字符串排序**后才渲染。本文用 README 那个例子验证了这一点：
请求里 `criteria` 的插入顺序是 `paid, overdue, draft`，而 prompt 里是
`OPTION 1: draft / OPTION 2: overdue / OPTION 3: paid`（§2 的解码）。

三种题型在这个变量上的处境各不相同：

| 题型 | 顺序从哪来 | 调用方能否控制 |
|---|---|---|
| `choice` | `sorted()` 后的**字典序** | **不能**——请求顺序被丢掉 |
| `score` | `enumerate(criteria)`，即请求给的 list 顺序 | 能，而且**必须**能——下标就是档位 |
| `noul` | 固定 `("true","false")` | 不能 |

所以「顺序偏置」在这里不是被消除、也不是被度量，而是**被渲染期规范化**了：
同一个选项集无论以什么顺序传进来，prompt 逐字节相同（唯一的前提是 `option_id` 不变）。
这是第十五种处理方式，与已收录的十四种都不同——最接近的是 Laya 的「训练时用 `parallel` 布局让位置对
选项不可见」，但那是在**表示层**消解，Clef 是在**文本层**消解。

三点代价与边界要说清楚：

1. **字母序本身仍是一个顺序**。模型看到的永远是 id 字典序，所以「对排在前的选项有偏好」这类偏置
   在原理上仍可能存在——只是调用方没法把它当旋钮拧。**仓里没有任何度量**（对 `permut` / `shuffle` /
   `reorder` 零命中），也没有可以自己跑一次对照的入口。
2. **`score` 是例外**。`score` 的选项顺序**就是它的语义**（下标 = 档位），所以那一路必须保序，
   也必然带着位置偏置。README 第 179 行只说 `criteria` 对 `score` 是「indexed from 0」，没说顺序无关。
3. **`choice` 的字典序是大小写敏感的字符串序**（`sorted` 对 `str`）。`"Paid"` 与 `"paid"` 会排到
   两个不同的位置——这不是 bug，但意味着「改一个选项 id 的大小写」会改变 prompt 的排版，
   进而改变所有选项的读位置（§12 #2）。

## §7 多问题共享 state：不只是共享，是**联合解码**

已收录的十四篇里，「多问题共享 state」问的都是**前缀能不能复用、账单怎么算**：
RYOTIDE 预填充一次只计一次、Jeff 干脆不做、SemIf-OpenJev 用 `reorder_cache` 复制分支、
decider 把前缀缓存按 GB 预算分块 fork、Nimble 一次 prefill 后分支、H2O-Lightning 交给 vLLM 前缀缓存
再改账单。**Clef 换了问题**：它共享的不只是 state，而是**答案本身互相可见**。

三处代码：

```python
fields = fields.unsqueeze(0)                       # [1, Q, 1024]，Q = 题数
for layer in self.layers:                          # 4 层 TransformerDecoderLayer
    fields = layer(fields, memory)                 # 自注意力在 fields 之间，跨注意力对 memory
fields = self.field_norm(fields[0])                # :421-424
```

`nn.TransformerDecoderLayer.forward(tgt, memory)` 先在 `tgt` 内部做自注意力——而 `tgt` 就是
**全部问题的 field 向量**。所以 `logit[q,·]` 在算出之前，已经通过 4 层自注意力看过 `field[q']`（q′≠q）。
README 第 41 行把这个说成 *scores all options of all questions jointly*，
`SYSTEM_PROMPT` 更是把这件事写成给模型看的指令：*Decide every field jointly*（`:25`）。

**后果一：不能逐题单独跑。** 把 schema 里的某一题删掉，剩下题目的答案**可以变**——这不是近似误差，
是这条架构的定义性质。已收录的十四条路线里，只有 Kev / Jevlike / CLM 那种「选项之间零交互」的读数
是严格可分解的；Clef 恰好相反，**题与题之间也有交互**。仓里**没有度量这个效应**的工具。

**后果二：前缀缓存只能覆盖到 state 为止。** prompt 的排列是 `prefix + state + schema + suffix`
（`:188`），所以换一道题就要重算整个 schema 段与 suffix，以及头的那一次前向。能缓存的只有
`prefix + state`——而这两段在 README 那个例子里只占 62/253 个 token（§15）。头的前向是 O(题数 × 选项数)
的小计算，省不了多少；真正贵的是 27B 骨干的那一次。

**后果三：账单是干净的。** `usage.input_tokens = len(encoded.input_ids)`（`:575`），一条 prompt 数一次，
多问题不重复计 state、也不存在「共享头只计一次」这种需要修补的账（对照 H2O-Lightning 的
`common_prefix_len` 与 Nimble 的 `prefix_prefill_count`）。这条链上**没有缓存代码**，
所以也没有缓存带来的口径问题。

## §8 服务化 / fail-loud 风格

**上游没有发布服务代码。** 仓里 25 个文件（§13），唯一的 Python 是 `joint_schema_model.py`，
它导出的是库函数与一个**请求/响应适配器**：`systemone(model, processor, request)`
（`:547`）吃一个 Jev/SystemOne 的 `POST /v1/systemone` 请求体，吐一个同形状的响应体。
HTTP 层在别处——README 第 96–110 行给的是 SGLang 的 docker 命令（镜像 `lmsysorg/sglang:dev-clef`）。
所以「The Clef API is fully compatible with Jev and SystemOne」（README 第 31 行）这句话的实现方式，
在本仓里就是**这一个函数**；`/health`、`/v1/models`、`/v1/version`、鉴权、并发、超时、错误码
**一概不在本仓**（§16）。

在这个前提下，fail-loud 只有四处，全在库函数里：

| 位置 | 什么情况下 | 行为 |
|---|---|---|
| `:553-554` | `model` 不是字符串，或没有 `state` | `ValueError("model and state are required")` |
| `:555-556` | `questions` 不是非空 dict | `ValueError("at least one question is required")` |
| `:557-561` | 某题 `type` 不在 `{noul, choice, score}`；或非 noul 而 `criteria` 为空 | `ValueError` |
| `:166-169` | schema + prefix + suffix 本身就超过 `max_length` | `ValueError("schema requires N tokens before state; maximum is M")` |

`load_release_model` 另有一处：`head.load_state_dict(..., strict=True)`（`:517`）——
头权重与 `joint_head_config.json` 必须逐键对上，缺一个键或多一个键都拒载。

**该响的那一处确实响，不该静默的那一处确实静默**——这是本文单独拎出来的一节：

```python
if max_state_tokens is not None:
    state_ids = state_ids[:max_state_tokens]                    # :163-164  静默
...
state_ids = state_ids[: max_length - fixed_length]              # :170      静默
```

**state 超长是从右端静默截断的**，没有告警、没有返回值里的标记、没有计数。`render()` 产出的是
紧凑 JSON（键排序、无空格），所以截断点很可能落在**一个 JSON 值的中段**——模型拿到一段语法不完整的
state，而响应里没有任何字段说明这件事。对照同一批收录里做对了的几条：Kev 超量 state → **422** 不静默截断
（其 §8）；Nimble 留出集保序且截断有记录；Jevlike 把静默截断记成明确的负例（其 §2.3）。
Clef 这一处的形状与 Jevlike 那条同类：**截断发生了，但没有任何东西告诉你**。
`encode_record` 的 `max_state_tokens` 默认是 `None`（`:107`），所以自动截断只有在显式传参时发生；
但 `max_length` 那条（`:170`）**默认就在**（`:106` 的 16,384）。

另有一处口径值得记：`encode_record` 的 `max_length` 默认 **16,384**（`:106`），
而 `tokenizer_config.json` 的 `model_max_length` 与 `config.json` 的 `max_position_embeddings`
都是 **262,144**（§15）。也就是说**代码默认只用了模型上下文长度的 1/16**，
`systemone()` 的签名里也把这个 16,384 又写了一遍（`:547`）。这不是 bug，
但引用「Clef 支持多长输入」时要知道问的是哪一层。

## §9 置信度从哪来，又怎么被改写

这条链上的链条**只有三环**，而且第一环的形状与前面十四条都不同：

| 环节 | 做什么 | 改 argmax？ |
|---|---|---|
| ① **训练本身** | 整个 `joint_head`（1.28 亿参数）是训出来的；骨干也被 post-train（README 第 33 行）。训练代码**不在发布仓里**，所以「训练做了什么」在本文里是**未核实** | **是** |
| ② **两个可训练尺度 + 一个门** | `prior_scale = exp(clamp(prior_logit_scale, ≤log100))`、`joint_scale` 同理、`sigmoid(residual_gate)`（`:438`、`:453`、`:456`）。它们是**权重里的一部分**，不是运行期常数 | 是（改两项的相对权重就会改 argmax） |
| ③ **`round(·, 4)`** | `systemone_answer` 里每个报出去的数都过 `round(...,4)`（`:526`、`:533`、`:534`、`:539`、`:540`、`:542`） | 否（只在极近并列时可能） |

**没有第四环**：没有温度（全文件零命中）、没有校准 bias、没有地板、没有重加权、没有顺序平均、
没有 ECE、没有拟合产物、没有接受/拒绝判据、也没有每请求的入参。

这带来一个本文要单独说的分界问题。`docs/comparison.md` §4.4 那条「读出来的 vs 训出来的」，
Clef 的答案**不是二选一**，因为发布出去的 logit 显式分成两项：

```
logit = prior + σ(g) · joint
        └─ 1 个可训练参数:prior_logit_scale
           (余弦的两个操作数都由冻结的取法决定:选项文本的词嵌入均值、问题向量+末位)
```

`prior` 那一项里**唯一的可训练量是一个标量**——取哪些 token、怎么平均、跟谁算余弦，
全是**取法**而不是**参数**。所以这条路线同时具备两种性质：

- 它对 RYOTIDE 那条分界的回答是「**训的**」——`joint` 项（4 层 decoder + 2 层 evidence routing +
  一个 4 路特征 MLP）是训练出来的，而实测 `joint_scale = 9.94306` 与 `prior_scale` 逐位相同、
  门是 `sigmoid(−2.0) = 0.119203`（§15.2），所以「哪一项在主导」不能靠看尺度判断，
  得看三项一起算出来的实际取值——那要跑一次前向；
- 但它留了一条**可以在不训练头的情况下算出来**的通路：把词嵌入按选项文本平均一下算余弦。
  这条通路在别的实现里是整条读法（RYOTIDE / SemIf-OpenJev / H2O-Lightning），在这里是**头里的一个加数**。

所以引用 Clef 的概率时，要确认的不是「温度是多少」（没有温度），而是**「发布的是哪一份权重」**——
而这件事**在响应里查不到**：`"model": request["model"]`（`:573`）回声的是**调用方自己发的字符串**，
不是权重路径、不是 revision、不是任何哈希。对照已收录几条把身份做进响应里的实现：
OpenJev(HF) 的 `/v1/version` 报补丁自身 sha256（其 §12）、Nimble 把 prompt 源码 sha256 钉进配置并在三处重算
（其 §10）、H2O-Lightning 的 `/health` 连「温度来自配置还是环境变量」都回报（其 §9）。
Clef 这一层是空的。

**另一个只有它才有的口径**：`score` 题型的 `score` 与 `confidence` **不是同一个档位**。

```python
return {
    "type": "score",
    "score": round(sum(index * probabilities[level] for index, level in enumerate(levels)), 4),
    "confidence": round(max(probabilities[level] for level in levels), 4),
    ...
}                                                                    # :536-543
```

报出去的 `score` 是**期望档位**（`Σ i·pᵢ`），而 `confidence` 是**最大档位的概率**。
一个 `score = 1.4` 的答案可能来自 `p = (0.1, 0.5, 0.4)`（`confidence = 0.5`），
也可能来自 `p = (0, 1, 0)`（`confidence = 1.0`）——两者 `confidence` 差一倍。
这与 CLM 那条「`label_of` 取 argmax 档位而答案里的 `score` 是期望档位」（其 §10.3）是同一个形状的问题，
也与 decider 的 `decide()` 报 `p_max`（其 §10.3）同类：**同一个名字下的数与它描述的对象不是一回事**。

## §10 图像与视频：同一份代码里的第二条输入面

这条链与前十四条的另一处不同是**输入面**：`state` 之外还有 `images` 与 `videos`，
而且**都在同一次前向里**（README 第 27–28 行：*reads the state as text, JSON, images, or video*）。

```
_encode_media(processor, record)                                   :81-100
    text = IMAGE_PLACEHOLDER·len(images) + VIDEO_PLACEHOLDER·len(videos) + "\n"
    encoded = processor(text=[text], images=…, videos=…, return_tensors="pt",
                        **record["media_kwargs"])                  :89-95
    media = {pixel_values, image_grid_thw, pixel_values_videos, video_grid_thw}   :96
    media["mm_token_type_ids"] = encoded["mm_token_type_ids"][0].tolist()         :97-99
```

三个占位符常量（`:28-29`）就是 `config.json` 里那三个 vision special token 的拼法：
`<|vision_start|><|image_pad|><|vision_end|>`（实测 id 248053 / 248056 / 248054，§15）。

拼接位置在 `prefix_ids` 之后、state 文本之前（`:158-161`），并且**顺手记下自己在整条序列里的起点**：

```python
if media is not None:
    media["token_offset"] = len(prefix_ids)          # 记的是「还没接上去」时的长度
    prefix_ids = prefix_ids + media_ids
```

`collate_records` 再把这个偏移用上（`:225-233`）：`mm_token_type_ids` 按**记录**逐行写进一个
`[B, L]` 张量，起点就是 `token_offset`。这是这条链上唯一一处「位置」被显式传过层的量
——与 §3 的 `question_span` / `option_spans` 不同，那两个是每题的、内部用的；
`token_offset` 是**批级**的、要拼到 batch 张量上的。

几点：

- **纯文本路径不跑视觉塔**。`ClefModel.forward` 先取 `base_model.model`，再看有没有 media：
  `if not media and hasattr(text_model, "language_model")` → 降到 `text_model.language_model`（`:476-477`）。
  所以不带图/视频的记录**连视觉编码器的模块都不进**；而一旦批里有任何一条带媒体，`media` 非空，
  整个批都走带视觉的那条路径。
- **图文混批由 `collate_records` 负责**。`pixel_values` 这些按 `torch.cat(values, dim=0)` 拼（`:221-224`），
  只收**存在的**那些记录；`mm_token_type_ids` 则给每条记录补成全 `[B, L]`（`:225-233`）。
  README 第 165 行说 *Text-only and multimodal records can be mixed in the same batch*，指的就是这段。
- **图片处理参数在 `processor_config.json` 里，不在代码里**：`Qwen3VLProcessor` +
  `Qwen2VLImageProcessor`（patch 16 / merge 2 / `longest_edge` 16,777,216），视频侧 `fps: 2`、
  `max_frames: 768`。`media_kwargs`（`:94`）是留给调用方覆盖这些参数的逃生口。
- **视觉塔的规模在 `config.json:123-139`**：depth 27、hidden 1152、`out_hidden_size` 5120
  ——与文本侧的 5,120 对齐，这是它能进同一条序列的前提。

## §11 三种题型与两种答案形态

这条链的「typed output」不止一种形状，这是它与前十四条最实际的一处差别：
**同一个头既报分类标签，也报一个实数**。

| 题型 | prompt 里的选项 | 报出去的东西 | 行 |
|---|---|---|---|
| `noul` | 固定两项 `true` / `false` | `{"type":"noul","noul": round(P(true),4)}`——**只有这一个数，没有 `confidence`，也没有 `probabilities`** | `:525-526` |
| `choice` | `criteria` 的 key（排序后） | `{"type":"choice","choice": argmax, "confidence": p[argmax], "probabilities": {…}}` | `:527-535` |
| `score` | `criteria` 列表的下标 | `{"type":"score","score": Σ i·pᵢ, "confidence": max p, "legend": {…}, "probabilities": {…}}` | `:536-543` |

三处口径要注意：

1. **`noul` 报的是概率不是判断**。字段名就叫 `noul`，值是 `P(true)` 的四舍五入——没有阈值、
   没有布尔化。所以下游要不要按 0.5 切是**下游的事**，模型不替它决定。README 第 124 行把它写成
   *a `noul` answer has the probability of true*，与代码一致。
2. **`choice` 的 argmax 取的是概率不是 logits**：`max(options, key=probabilities.__getitem__)`（`:529`），
   而 `probabilities` 是 softmax 后的（`:568`）。所以与 Nimble 那条「argmax 取原始 logits，温度在构造上
   不可能改选择」（其 §5）相比，这里的选择**是**在概率上做的。没有温度，所以这条差别目前没有后果；
   但**一旦有人在上游加一个温度，它在这里会比在 Nimble 那里更容易改掉 argmax**。这是并读才看得见的一处。
3. **`score` 的 `confidence` 与 `score` 不同源**（§9 已展开）。另外 `score` 的 `legend`
   是 `dict(zip(levels, question["criteria"]))`——把下标映回描述文本，**只进响应、不进 prompt 之外的地方**。

`question_options` 对 `score` 用的是 `enumerate`（`:57`），所以档位数就是 `criteria` 的长度，
没有上限检查——这与 `choice` 的 `criteria` 是 dict（键唯一，天然无重复）不同：
`score` 的 `criteria` 是 list，**重复的档位描述会被当成两个不同的档位**（下标不同）。
这一点在 `choice` 上不存在（dict 的键不能重复），是两种题型不对称的地方。

## §12 四条不变量的落位

| RYOTIDE §13 不变量 | 在 Clef 上的落位 |
|---|---|
| **#1** 读位置必须落在标记上（诊断量 `marker_mass`） | **概念不适用**：头的输出个数就是选项数，**没有词表分布**，`marker_mass` 恒等于 1、无信息。**改写后的等价命题**是「softmax 的输入必须恰好是这一题的 K 个选项标量」，由构造保证（`:485-491` 返回 `list[list[Tensor]]`，每个内层张量的长度 = `len(option_spans)`）。守卫：**没有**。但与 Kev / decider / Nimble / Jevlike / H2O-Lightning 不同，这里的词表矩阵**确实被用了**（当嵌入，`:381-382`），所以「词表维度从不出现」这句话要说成「**从不作为 logits 出现**」 |
| **#2** 前缀与标记表面形式耦合，必须成对扫 | **不适用（没有标记字符串）+ 换成一个更硬也更没守卫的命题**：标签是请求给的字符串，没有「标记 token」这个东西。**改写后的等价命题**是「**schema 的渲染必须与头训练时逐字节相同**」——头是按 `question_span` / `option_spans` 的**绝对下标**池化的（§3），渲染改一个空格，池化的就是别的 token。而这套渲染是**手写在 Python 字符串里的六处字面量**（`:110`、`:116`、`:125`、`:130`、`:139`、`:150-157`）。**没有任何守卫**：没有 sha256、没有布局配置、没有加载期断言。对照 Nimble 把 prompt 源码的 sha256 钉进 `schema_config.json` 并在三处重算（其 §10）、decider 把布局写进 `decider_config.json` 并在遇到未知布局时报错（其 §11）——**这是本条路线最薄的一处** |
| **#3** 只读最后一个位置 | **要改写，而且是十五条里最复杂的一次**：头读的不是一个位置，而是**五路**——末位 `sequence_hidden[-1]`（`:358`）、每题的 `question_span` 均值（`:359-364`）、每选项的 `option_span` 均值（`:369-378`）、每选项 span 的词嵌入均值（`:379-383`）、以及**全序列当 memory** 做两层跨注意力（`:399-401`）。其中「末位」只是**一路**，它进的是 `anchor = normalize(qvec + global)`（`:433-436`）与 `field` 的构造（`:418`）。位置全部由 `encode_record` 拼 prompt 时算出（§3），运行期无标定。**「不物化词表维度」这半条成立**（见 #1） |
| **#4** 比较概率 / ECE 前先确认 temperature | **没有 temperature → 要确认的是「哪份权重」，而这条链上没有可确认的东西**。全文件 `temperature` 零命中；数值上的温度被**两个可训练尺度**代替（`prior_logit_scale` / `joint_logit_scale`，都 clamp 到 ≤100，§15 给出实测值），而它们**在权重里**，不在任何配置或响应里。响应唯一的身份字段是 `"model": request["model"]`（`:573`）——**回声调用方的字符串**。所以 #4 在这里要改写成「**去 HF 上按 revision 确认权重**」：本文钉在 `0b331204bb13fbd2ca93a64df1956f5b55478ce5` |

**这条路线自己新增的、值得进总表的三条**：

1. **「schema 的渲染是权重契约的一部分，而发布物里没有这份契约」**——最薄的一处（见 #2），
   与 Nimble 的「prompt 源码 sha256 是权重的一部分」恰成对照：**同一种依赖，一边钉了哈希、一边什么都没有**。
2. **「题与题之间互相可见，所以不能逐题单独跑」**——4 层 decoder 在 `fields` 之间做自注意力（`:421-424`）。
   这与已收录各篇的「共享前缀、各题独立」是**相反**的做法，是第十五种「多问题共享 state」的回答。
3. **「先验项只有 1 个参数，所以读法与训练法可以共存于一个 logit 里」**——`logit = prior + σ(g)·joint`（`:455-457`）。
   已经收录的路线在这一层都是二选一（要么零训练读、要么训一个头）；Clef 把它做成了**加性分解**，
   于是「这个数有多少是零样本读出来的」在原理上**可以算**——只要知道 `σ(g)` 与两项的实际取值。

## §13 文件对照表

HF 仓 25 个文件中的全部非权重文件（权重是 12 个 `model-*.safetensors` 分片，合计 ≈ 54.7 GB，未下载）：

| 文件 | 大小 | 职责 |
|---|---|---|
| `joint_schema_model.py` | 23,263 B / **577 行** | 唯一的实现文件：`render` / `question_options` / `encode_record` / `collate_records` / `EvidenceRoutingLayer` / `JointSchemaHead` / `ClefModel` / `load_release_model` / `systemone_answer` / `systemone` |
| `joint_head.safetensors` | 256,125,024 B | 头权重，**122 个张量、全 BF16、128,056,324 个元素**（本文实测，§15） |
| `joint_head_config.json` | 119 B | 头超参：`hidden_size 5120` / `width 1024` / `routing_layers 2` / `layers 4` / `heads 16` / `feedforward 4096` |
| `config.json` | 3,688 B | 骨干：Qwen3.5 文本侧 64 层（**48 层 linear_attention + 16 层 full_attention**，每 4 层一个全注意力）+ vision depth 27；`tie_word_embeddings: false` |
| `model.safetensors.index.json` | 111,110 B | 1,184 个张量 → 12 个分片；`total_parameters: 27,356,728,560`、`total_size: 54,713,457,120` |
| `tokenizer.json` | 19,989,325 B | BPE，vocab 248,044 + 33 个 added token（21 个 special） |
| `tokenizer_config.json` | 1,075 B | `eos_token: <\|im_end\|>`、`pad_token: <\|endoftext\|>`、`model_max_length: 262144` |
| `chat_template.jinja` | 8,952 B | **发布仓里有，但 `joint_schema_model.py` 从不使用**（`apply_chat_template` 零命中，§14） |
| `processor_config.json` | 1,191 B | `Qwen3VLProcessor` + `Qwen2VLImageProcessor`（patch 16 / merge 2）+ `Qwen3VLVideoProcessor`（fps 2 / max_frames 768） |
| `generation_config.json` | 214 B | `temperature: 1.0` / `top_k: 20` / `top_p: 0.95` / `do_sample: true`——**与这条读法无关**（不生成） |
| `README.md` | 11,105 B | 模型卡：两句话的立场、43 行 Decision Index 表、5 行 workflow 表 |
| `LICENSE` / `.gitattributes` | 11,544 B / 1,570 B | Apache-2.0；LFS 规则 |

`generation_config.json` 那一行值得单列：它是**唯一一个带 `temperature` 的文件**，
值是 1.0，而它属于采样路径。本文的「没有温度」指的始终是 `joint_schema_model.py`——
引用时不要拿这个 1.0 去当「Clef 的温度」。

## §14 亲手跑一遍

**先读代码顺序**（按一次请求走）：

```bash
# 1. 入口与校验
sed -n '546,576p' joint_schema_model.py         # systemone()
# 2. 标签集与 prompt 拼装（读位置在这里被算出来）
sed -n '35,57p'   joint_schema_model.py         # render / question_options
sed -n '103,196p' joint_schema_model.py         # encode_record
# 3. 一次前向
sed -n '462,491p' joint_schema_model.py         # ClefModel.forward
# 4. 头与公式
sed -n '281,337p' joint_schema_model.py         # JointSchemaHead.__init__
sed -n '344,459p' joint_schema_model.py         # forward():五路特征 + prior/joint 合成
# 5. 答案成形
sed -n '523,543p' joint_schema_model.py         # systemone_answer
```

**跑上游本身**（需要 H200，本文没跑）：

```bash
pip install 'torch==2.11' 'transformers==5.10.2' pillow   # README 第 57 行的版本组合
python - <<'PY'
import sys, torch
from huggingface_hub import snapshot_download
path = snapshot_download("Cloudflare/clef", revision="0b331204bb13fbd2ca93a64df1956f5b55478ce5")
sys.path.insert(0, path)
from joint_schema_model import systemone, load_release_model
model, processor = load_release_model(path, device="cuda")
print(systemone(model, processor, {
    "model": "clef",
    "state": "Our checkout started returning errors and orders are blocked.",
    "questions": {
        "department": {"type": "choice",
                       "instructions": "Which team should handle the message?",
                       "criteria": {"billing": "Payments or invoices",
                                    "technical": "Bugs or outages"}},
        "urgency": {"type": "score", "criteria": ["Can wait", "This week", "Today"]},
        "outage": {"type": "noul", "instructions": "Is a service down?"},
    },
}))
PY
```

**本文实际跑过的那一类**（不加载权重、不 import 上游、不执行上游代码）：

```bash
# A. 头权重的容器算术（纯 stdlib 解析 safetensors 头）
#    → 122 个张量 / 全 BF16 / 128,056,324 元素 → ×2 = 256,112,648 B
#    → 8 + 12,368 + 256,112,648 = 256,125,024 = 文件实际大小（精确闭合）
# B. 三个标量的实际值（bf16 解码,只用 struct）
# C. 用发布的 tokenizer.json 数 token（tokenizers 库加载一个文件,不跑上游）
# D. README 两张结果表的加粗规则与夺魁计数（只用 re + 算术）
# E. config / tokenizer_config / generation_config / tokenizer.json 的 special token 口径对照
```

## §15 实例走查：在发布产物上的复算

本文没有跑任何上游程序：**没有下载 54.7 GB 权重、没有 `import joint_schema_model`、没有 GPU**。
下面五类数字全部来自发布产物本身，用**本文自己的脚本**算出。

### 15.1 头权重的容器算术（精确闭合）

| 量 | 值 |
|---|---|
| `joint_head.safetensors` 文件大小 | **256,125,024 B** |
| safetensors header 长度 | 12,368 B（JSON，`__metadata__ = {"format": "pt"}`） |
| 张量数 / dtype | **122 个，全部 BF16** |
| 元素总数 | **128,056,324** |
| 8 + 12,368 + 128,056,324×2 | **= 256,125,024** ✓ 与文件大小逐字节相等 |

按名字前缀分组的参数量（同一份脚本算出）：

| 组 | 参数量 | 占比 |
|---|---|---|
| `layers`（4 层 `TransformerDecoderLayer`，每层含 `self_attn` + `multihead_attn` + 两个 FFN） | 67,186,688 | 52.47 % |
| `evidence_layers`（2 层 `EvidenceRoutingLayer`） | 25,196,544 | 19.68 % |
| 6 个 `*_projection`（各 `[1024, 5120]`） | 5,242,880 × 6 = 31,457,280 | 24.57 % |
| `residual_scorer`（`[1024,4096]`+bias、`[1,1024]`+bias） | 4,196,353 | 3.28 % |
| 4 个 LayerNorm + `type_embedding` | 19,456 | 0.02 % |
| 3 个标量（`prior_logit_scale` / `joint_logit_scale` / `residual_gate`） | 3 | 0.00 % |
| **合计** | **128,056,324** | 100 % |

对照：同一份仓里 `model.safetensors.index.json` 声明的骨干是 **27,356,728,560** 个参数
（54,713,457,120 B）。所以头是骨干的 **0.47 %**。这解释了 §4 里那句「一个 27B 骨干 + 一个 1.28 亿的头」。

### 15.2 三个标量的实际值（发布权重里的「温度」）

`prior_logit_scale` / `joint_logit_scale` / `residual_gate` 在 `__init__` 里都初始化成 `torch.zeros(())`
（`:335-337`），所以**发布值直接告诉你训练把它们推到了哪里**。用纯 `struct` 解码 BF16：

| 张量 | 发布值 | 换算 | 代码里的界 |
|---|---|---|---|
| `prior_logit_scale` | **2.296875** | `exp(·)` = **9.943061786014486** | `clamp(max=log(100))` = 4.60517 |
| `joint_logit_scale` | **2.296875** | `exp(·)` = **9.943061786014486** | 同上 |
| `residual_gate` | **−2.0** | `sigmoid(·)` = **0.11920292202211755** | 无界 |

**两个尺度在发布权重里逐位相同**（同一个 BF16 位型）。代入三项，发布出去的 logit 是

```
logit = 9.94306 · cos(lexical_option, anchor)          ← 先验项：1 个可训练参数
      + 0.119203 · ( 9.94306 · cos(field, option)      ← 联合项的余弦分量
                     + residual )                      ← 4 路特征 MLP,量纲无界
```

三处必须说清楚边界的观察：

1. **两个尺度相等是可核的，为什么相等不可核。** 读的是权重里的两个 2 字节；而「是训练时绑定了、
   还是碰巧收敛到同一个 BF16 值」要看训练代码，**不在发布仓**（§16）。引用时只能说
   「发布权重里两个尺度相等」。
2. **三个标量的形状不像独立收敛的结果，但这一点未核实。** `2.296875 / 2.296875 / −2.0`——
   前两个逐位相同，第三个是**精确整数**，而且恰好是 `sigmoid` 好算的 −2（σ = 0.11920）。
   三者都初始化为 0（`:335-337`）。本文只能说**发布权重里它们是这三个值**，
   至于是训练推出来的还是发布时定的，**未核实**。
3. **`clamp` 在发布权重下不起作用。** 两个尺度都离上限（`log(100)` = 4.60517 → 100）很远
   （2.2969 << 4.6052），所以那个 `clamp` 是**防御性上界**，不是当前生效的约束。
   这与 CLM 的 `clamp(max=100)`（`heads.py:95`）是同一个 idiom，但 CLM 那条路线的尺度是否贴到上限
   是另一篇的事。

这三项里**哪一项主导**不能只看尺度：先验项的余弦范围是 ±9.943，联合项的余弦分量被压到 ±1.185
（= 9.94306 × 0.119203），但 `residual` 无界、且先乘门再相加。要定这件事只能跑一次前向，
**本文没有跑**（§16）。

### 15.3 prompt 的 token 账（用发布的 tokenizer.json 数）

按 §1 的布局自行重拼 README 那个两题例子（不 import 上游），交给发布的 `tokenizer.json` 数 token：

| 段 | token 数 |
|---|---|
| `prefix`（system + `<\|im_start\|>user\nSTATE:\n`） | **36** |
| `state`（紧凑 JSON，键排序） | **26** |
| `schema`（两个 FIELD，共 5 个选项） | **173** |
| `suffix`（含空 think 块与 `JOINT SCHEMA DECISIONS:`） | **18** |
| **合计** | **253** |

所以在这个最小例子里，**固定开销（36 + 173 + 18 = 227）是 state 的 8.7 倍**。
多问一题的边际成本不是「state 再算一遍」而是**再加一整段 FIELD**——这与 §7 的「联合解码」是同一件事的两面。

单段实测：`suffix` 整体 18 token，其中空 think 块 4 token、`JOINT SCHEMA DECISIONS:` 8 token；
`"\n\nSCHEMA FIELDS:\n"` 7 token；`"\nFIELD 1\nID: …\nTYPE: …\nINSTRUCTION: "` 17 token；
`"\nALLOWED OPTIONS:\n"` 6 token；`"OPTION 1: "` 5 token；`"END FIELD\n"` 3 token。

### 15.4 README 两张结果表的加粗规则（43/43 + 5/5）

README 第 189 行自陈 *The best value in each row is in bold*，并声明两处例外
（第 189 行末：*ForecastBench is a Brier score, where lower is better. The last two rows are request
latency in milliseconds, where lower is better.*）。本文把 43 行全部解出来逐行验：

| 检查 | 结果 |
|---|---|
| 数据行数 | 43（41 个基准 + 2 个延迟行），列数恒为 7 |
| 无加粗的行 | **0** |
| 加粗的不是该行最优（按「越小越好」的三行做例外） | **0** |
| 行内并列最优 | **0** |
| Workflow 表（5 行）同样规则 | **0 例外** |

按同一条规则数各列夺魁（43 行）：

| 列 | 夺魁行数 |
|---|---|
| Clef-flash | **17** |
| Clef | **11** |
| Jev | **11** |
| Kev 9B | 2（`RouterBench` 80.0、`SGD/SGD-X` 64.0） |
| DiffusionGemma Jev | 1 |
| Laya | 1（**只有中位延迟** 5.8 ms） |

这张表本身就是一条要小心引用的东西：**它是 Clef 团队自己跑 Decision Index 0.2.1 的结果**
（README 第 189 行 *from our internal run*），不是被比较的那些项目自己发布的数字。
其中 `Kev 9B` 与 `Laya` 两列对应本文库里已经收录的两条路线——引用它们时要知道这是**第三方复跑列**。
同一张表里还有两处方向相反的例子，说明「小一号的变体更快也更差」不是安全假设：
`CLINC150+OOS` 上 Clef 97.4 对 flash **66.8**（差 30.6），而 `Home appliance simulator` 上
Clef 83.0 对 flash **97.7**（反过来差 14.7）。

### 15.5 配置文件之间的 special token 与长度口径

| 量 | 出处 | 值 |
|---|---|---|
| `eos_token` | `tokenizer_config.json` | `<\|im_end\|>` = **248046**（与 `tokenizer.json` 的 added token 对上） |
| `pad_token` | `tokenizer_config.json` | `<\|endoftext\|>` = **248044** |
| `eos_token_id`（生成） | `generation_config.json` | `[248046, 248044]`（两个都收） |
| `eos_token_id`（文本配置） | `config.json:15` | **248044** —— 与 `tokenizer_config.json` 的 eos **不是同一个**。因为这条链不生成，这一处**无后果**，但它是一处口径不一致，引用 `config.json` 的 eos 时要知道 |
| `image_token_id` / `video_token_id` | `config.json:6,122` | 248056 / 248057 —— 与 `tokenizer.json` 实测一致 |
| `vision_start_token_id` / `vision_end_token_id` | `config.json:140-141` | 248053 / 248054 —— 一致 |
| `vocab_size` | `config.json:118` | **248,320**，而 `tokenizer.json` 是 248,044 + 33 = **248,077** —— 差的 243 个槽位是嵌入表的对齐余量。本文 §3 里那张 `[248320, 5120]` 的 lm_head 用的是配置值 |
| `max_position_embeddings` / `model_max_length` | `config.json:94` / `tokenizer_config.json` | 都是 **262,144**，而代码默认 `max_length=16384`（`:106`、`:547`） |

### 15.6 两处「同一个坑的两种落法」：`<think>` 与 chat template

`chat_template.jinja` 在发布仓里，但 `joint_schema_model.py` **从不使用它**——
全文件对 `apply_chat_template` / `enable_thinking` / `reasoning` 零命中。prompt 是**手写的**：
六处字面量（`:110`、`:116`、`:125`、`:130`、`:139`、`:150-157`）拼出全部结构。

而 suffix 那一段（`:156`）里有一个空 think 块：

```python
suffix_ids = _tokens(
    tokenizer,
    "\n<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\nJOINT SCHEMA DECISIONS:",
)                                                                    # :154-157
```

对照同一份仓里的 `chat_template.jinja:163-169`：

```
{%- if add_generation_prompt %}
    {{- '<|im_start|>assistant\n' }}
    {%- if enable_thinking is defined and enable_thinking is false %}
        {{- '<think>\n\n</think>\n\n' }}
    {%- else %}
        {{- '<think>\n' }}
    {%- endif %}
{%- endif %}
```

两处字面量**逐字节相同**（`'<think>\n\n</think>\n\n'`）。也就是说：Clef 手工写死了模板在
`enable_thinking=False` 下的那一支，再在后面接上自己的 `JOINT SCHEMA DECISIONS:`。
这解释了 §12 #3 里「读位置不需要标定」的来由——**要读的东西排在 `JOINT SCHEMA DECISIONS:` 之后，
而它前面那段是闭合的**。

它与已收录几条同类做法的区别在于**同步方式**：
H2O-Lightning 也是把空 think 块硬编码进 prompt 字符串（其 §3），但它**在图像路径上有一条断言**
保证模板渲染与文本渲染逐 token 相同（`:1004-1021`）；SemIf-OpenJev 的 reranker 路径同样手写这个字符串
（其 §8）；decider 则是**有模板**但显式补闭合串并加断言（其 §11）。
Clef 属于**手写、且没有断言**的那一类——`chat_template.jinja` 与 `:156` 那一行必须手工保持同步，
而仓里**没有东西检查这一点**。

## §16 未核实清单

按「不给看起来合理的值」的规矩，下面这些**本文没有算，也不猜**：

- **任何由前向产生的概率、准确率、延迟。** 54.7 GB 权重未下载，`import joint_schema_model` 未执行
  （本机也没有 `torch` / `transformers==5.10.2` / GPU），所以 `noul` 的概率、`choice` 的分布、
  `score` 的期望值**一个都没有**。§11 里那些答案形状是**代码的形状**，不是任何一次运行的输出。
- **README 两张结果表的数字本身。** §15.4 只核了**加粗规则**（哪一格是行内最优），
  没有核任何一格的**取值是否正确**——那需要复跑 Decision Index 0.2.1，本文没有跑。表里 `Kev 9B`
  与 `Laya` 两列同样是 Clef 团队的内部复跑，不是那两条路线自己发布的数字。
- **三项里哪一项主导。** §15.2 给出了三个标量的发布值，但 `prior` 与 `joint` 的实际大小还要看
  `residual`（无界）与两个余弦的取值，那需要一次前向。**本文没有跑**，所以「Clef 的置信度主要由
  词嵌入先验决定」这句话本文**不说**——只能说三个旋钮的发布值摆在那里。
- **训练做了什么。** 训练代码不在发布仓（25 个文件里没有 `train`、没有 `data`、没有配置）。
  所以「骨干被 post-train 成什么」「两个尺度为什么相等」「`clamp(max=log 100)` 是训练时就在还是事后加的」
  **全部未核实**。README 第 33 行只说 *post-trained from Qwen/Qwen3.8-27B*。
- **`joint_head` 与骨干的对应关系。** 发布物里没有任何哈希把 `joint_head.safetensors`
  与那 12 个骨干分片绑在一起；两者只是同一个 revision 下的同名目录内容。§12 #4 那处「没有可确认的东西」
  指的就是这个。
- **HTTP 面的一切。** 鉴权、错误码、并发、超时、限流、`/health`、日志：
  `joint_schema_model.py` 里都没有，SGLang 的 `dev-clef` 镜像不在本仓，未核。
- **`Clef-Flash`。** README 第 34 行指向 `Cloudflare/clef-flash`，它是另一个仓，本文一行都没读，
  §15.4 里它的夺魁次数只是**别人表里的一列**。
- **视觉分支的数值行为。** 三个占位符常量、`token_offset` 的写法、`collate_records` 的拼接（§10）
  都是代码事实；但「图像 token 展开后有多少个」「视觉塔输出怎么进序列」需要 processor + 权重，未核。
- **`score` 题型里重复档位的后果。** §11 指出 `criteria` 是 list、重复描述会变成两个档位，
  但「这会不会改变 argmax / 期望值」需要跑模型，未核。
- **`choice` 的字典序偏置有多大。** §6 证明调用方的顺序进不了 prompt，
  但「字母序本身带来多少偏置」在仓里**没有任何度量**，本文也没有能跑它的入口。

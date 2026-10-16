# StartLux-Decision 是如何只用 26 个字母 token、在一个模板钉死的读位置上读出类型化决策的

> 代码库：<https://github.com/StartLuxLabs/StartLux-Decision>
> 对应 README 第 21–24 行：*StartLux-Decision is a family of open decision models. Given a state (text,
> JSON or images) and a set of typed questions, a model returns a probability for every option of every
> question: a choice among listed options (`choice`), a yes or no judgement (`noul`), or a level on a
> scale (`score`). Instant models answer in a single forward pass with bidirectional attention over the
> prompt. AutoThink models answer in one pass when the top probability reaches a gate, 0.6 by default,
> and otherwise generate a short reasoning trace and read the answer again.*
> 以及第 25–27 行：*This repository contains the inference engine and HTTP server for both variants.
> The API is compatible with System One `/v1/systemone`.*
> 本文按 `startlux_decide/`（`prompt` / `readout` / `pipeline` / `modelcfg` / `controls` / `server` /
> `serving` / `images` / `loops` / `vllm_plugin` / `launch` / `quantize` / `backends/`）、`v1/finetune/`、
> `v1/eval/` 与 `patches/llama.cpp/` 的实际代码逐环节说明这两句话的实现方式；文中出现的文件名与行号
> （`readout.py:16`、`pipeline.py:326`、`modelcfg.py:240` 等）均相对该仓库根目录。

## 本文的事实来源

| 类别 | 内容 | 文中标记 |
|---|---|---|
| 代码引用 | `startlux_decide/**`（6,620 行里的约 4,300 行）、`v1/finetune/`、`v1/eval/`、`patches/llama.cpp/` | `文件:行` |
| 在 committed 产物上重算 | 27B 示例响应的两条置信度公式、`v1/docs/results.md` 六行两张汇总列、JevBench 题数↔百分比、README 的增量断言、`patches/llama.cpp/README.md` 的两组分差 | 「实测」 |
| 公式与常量 | README、`docs/*.md`、`decision_config.json`、`config.json`、发布的 `tokenizer.json` | `文件:行` 或「Hub 核对」 |

**本文不给出任何由前向产生的数值**。26 个字母 token 的 id、逐题型温度与发布清单都能在 Hub 的
小文件上逐项核准（§13），但任何概率、准确率、延迟都需要把 9 GB 以上权重放进 GPU —— 那部分本文
只读代码推断机制，不给数值，并在 §12 给出复现命令。四个后端里的两个（vLLM / llama.cpp）需要各自
的推理服务器进程，本文没有启动过它们。

---

## 0. 与 RYOTIDE 的分野

RYOTIDE 那条读法把冻结的 LLM 当分类器读，代价是**读位置与标记表面形式都得在运行期标定**，诊断量
`marker_mass` 就是为这件事生的。StartLux-Decision 做的选择是把这三件事全部前移到训练与发布物里：

| 维度 | RYOTIDE | StartLux-Decision |
|---|---|---|
| 权重 | 公开 LLM，全程 frozen | **微调过的整份权重**（Instant / AutoThink 两种训练目标），`v1/` 里留下的是 LoRA 拟合脚本 |
| 「typed」怎么来 | 运行期掩码到固定标签集 | **26 个单 token 字母**，id 写进 `decision_config.json` 并在加载期与分词器逐项对齐（`pipeline.py:154-156`） |
| 读位置 | 运行期标定 | **由 chat 模板在 `enable_thinking=False` 下物化出的空思考块**钉死（`prompt.py:34`），运行期没有自由度 |
| 掩码在哪一步 | 读出全词表分布再掩码 | **不物化词表维度**：vLLM 交给 `allowed_token_ids`、transformers 用 `index_select`、llama.cpp 用 `logit_bias`+`top_k` |
| 置信度 | 读出来的（只有 temperature 能改数值） | 读出来的，**且只有一个可改数值的旋钮**：逐题型温度，拟合后随权重发布 |
| 一次决策读几次 | 一次 | 一次或**两次**（AutoThink）——本族新增的维度，见 §9 |
| prompt 源码的地位 | 可替换的读取配置 | **权重的一部分**：`prompt.py:5-6` 明说改任何字符串就改答案 |

一句话立场：**它把 RYOTIDE 那层运行期的自由度全部换成了发布物里的常量**。好处是服务端没有标定
代码、没有扫描、没有 `marker_mass`；代价是 §2 的字母表与 §3 的模板尾巴变成了不可协商的前提，而
§13 会指出这个前提在发布物里挂着一个更早的项目名。

---

## 1. 端到端数据流

一次 `/v1/systemone` 请求走完的全部环节：

```
POST /v1/systemone  {state, questions, images?, ...controls}
  │
  ├─ server.decisions                    server.py:175-209
  │    parse_controls(body, ...)         controls.py:66-106   -> 422 带 loc
  │
  ├─ Decider.decide                      pipeline.py:227-252
  │    images -> state + Media           pipeline.py:238-240 / images.py:37-64
  │    prepare(): 每题渲染               pipeline.py:185-224
  │      choice 选项数 > 26 -> knock-out pipeline.py:216-222
  │      choice 只有 1 个选项 -> 不调模型 pipeline.py:214-215 / 302-307
  │    _share(): 公共前缀入 prefix cache pipeline.py:270-287
  │    asyncio.gather 每题一条独立任务    pipeline.py:243
  │
  └─ _run()  每个问题                     pipeline.py:299-346
       ① plain = plain_prompt(msgs)      prompt.py:222-224     <- 尾部带空思考块
          z = readout(plain, letter_ids[:K])   backends/base.py:86-87
              vLLM: max_tokens=1 + allowed_token_ids  vllm.py:269-282
              transformers: index_select + 一次 matmul  transformers.py:108-109
          p0 = softmax(z / T[type])      readout.py:16-21  /  pipeline.py:289-291
       ② 门控: max(p0) < gate            pipeline.py:326
          _think() -> thought            pipeline.py:389-403
          z1 = readout(think 后的 prompt) pipeline.py:329
          p1 = retemper(p1, T[type], T_think)  readout.py:24-29  / pipeline.py:331-332
          p  = (1-mix)*p0 + mix*p1       readout.py:32-34  /  pipeline.py:333
       ③ answer() / choice_answer()      readout.py:62-84
          confidence = 由 p 直接导出的量   readout.py:37-51
  │
  └─ {"answers", "usage", "model" (+ "timing_ms")}   server.py:203-209
```

三处与 RYOTIDE 篇显著不同的形状：

1. **`output_tokens` 在默认路径上是 0**。Instant 与 v1 文件夹一次决策只做一次 prefill 读，不生成
   任何 token（`pipeline.py:338` 把 `thought.tokens if thought else 0` 计入）。只有 AutoThink 且真的
   触发思考时才有输出 token。
2. **每题一条独立任务，题与题之间不共享 thought**（`pipeline.py:243` 的 `_gather`）。这与「多问题共享
   state」是两件事：state 可以共享**前缀**，但推理过程不共享。
3. **一个问题可能同时有两次读数的中间量**，都被 `return_details` 暴露出来（`pipeline.py:373-387`），
   而不是只报最终值。

---

## 2. 「typed」从哪来：26 个单 token 字母

标签集就是 **A–Z 这 26 个字母**，且必须在分词器里各占**一个 token**。

**声明在哪。** `decision_config.json` 的 `letter_token_ids`（`modelcfg.py:218-221` 校验：必须正好 26 个
整数）。加载时 `pipeline.py:154-156` 拿分词器现算一遍，与文件夹声明的逐项比对，不同就抛
`ValueError("the tokenizer's letter ids differ from letter_token_ids in decision_config.json")`。
现算的那份在 `prompt.py:138-156` `check_tokenizer`：

```
假设模板在 thinking 关掉时以 "<think>\n\n</think>\n\n" 结尾        prompt.py:143-146
对 A..Z 每个字母:
    len(tokenizer.encode(letter)) == 1                             prompt.py:150
    且 tokenizer.encode(text + letter) == ids + [letter_id]         prompt.py:151
26 个 id 互不相同                                                   prompt.py:154
```

第二条是关键：它断言的是**「在完整 prompt 之后，这个字母是一个 token」**，而不是「这个字母在词表里
是单 token」。同一个字符在别的上下文里可能被合并，`prompt.py:151` 把这条路堵住了。

**Hub 核对（§13.1）**：v1.1 的 4B-AutoThink 发布 `letter_token_ids = [32, 33, …, 57]`，而这 26 个 id 在
随仓 `tokenizer.json` 的 `model.vocab` 里正好是 `'A'` 到 `'Z'`、按序、连续 —— `check_tokenizer` 的
六条断言在这份分词器上全部成立。

**三种题型的选项数上限不同**（`prompt.py:75-104`）：

| 题型 | 选项来源 | 上限 | 越界时 |
|---|---|---|---|
| `choice` | `criteria` 的键（或选项列表） | 26（`MAX_OPTIONS`，`prompt.py:28,85`） | **不报错**：先做 knock-out（§2.2） |
| `score` | `criteria` 的层级，总是最低在前 | 10（`MAX_LEVELS`，`prompt.py:29,102-103`） | 抛 `RequestError` |
| `noul` | 固定 `"true"` / `"false"` | 2（`prompt.py:100-101` 断言集合相等） | 抛 `RequestError` |

`bool` 是 `noul` 的别名（`prompt.py:173` 的 `t = "noul" if t == "bool" else t`）。

### 2.1 渲染：字母怎么写到 prompt 里

`option_lines`（`prompt.py:107-126`）把第 k 个选项写成 `"{LETTERS[k]}) {body}"`（`prompt.py:125`）。`body`
的取法有三条分支，其中一条专门消除歧义：

```
若 choice 且**所有列出的选项 id 都是裸字母/数字** 且**每个都有描述**:
    隐藏 id，只写描述        hide = t == "choice" and all(_BARE.match(i)) and all(crit[i])  prompt.py:114
否则若描述为空或与 id 同义:
    只写 id                  prompt.py:121-122
否则:
    写 "id: 描述"            prompt.py:123-124
```

`_BARE` 是 `^(?:\(?[A-Za-z][\).]?|\(?\d{1,3}[\).]?|option[_ ]?\d{1,3}|opt[_ ]?\d{1,3})$`（`prompt.py:35`）。
这条规则要挡的是 `"A) C: Paris"` 这种**答案字母有歧义**的行（`prompt.py:12-13`）。它按文档只对
`choice` 生效，且**只看列出的选项**——`allow_none` 加进来的那个额外选项不参与 `hide` 判定
（`prompt.py:116` 的 `listed`，`prompt.py:16`）。

### 2.2 超过 26 个选项：两轮 knock-out

`choice` 的选项数超过字母表时**不截断也不报错**，而是分两轮读（`readout.py:99-132`）：

1. `knockout_groups(keys, group)` 把选项切成**尽量等长的连续组**，每组至多 `group` 个
   （`readout.py:99-108`；`group` 来自 `decision_config.json` 的 `wide_choice.group`，默认 25，
   `modelcfg.py:223`）。注意组内是 `[start:end]` 的**连续切片**，不是按相似度分组。
2. 每组自己的 prompt 各读一次，组内取概率最高的 `keep` 个（默认 3）进决赛；若决赛人数超过
   `max_options`（= 26 − 是否有 none），**改为全按第一轮概率取前 26**（`readout.py:111-121`）。
3. 决赛轮再读一次这 ≤26 个；未进决赛的选项按第一轮概率**分摊一个 `residual` 质量**（默认 `1e-3`，
   `readout.py:124-132`），最后整体重归一化。

代价写在 `pipeline.py:338`：`input_tokens` 把**每一组**的 prompt token 都加进去
（`q.group_tokens`），所以一个 255 选项的问题的 `usage.input_tokens` 是 11 组 prompt 之和，不是一
个 prompt。`v1/docs/results.md` 里那条「usage.input_tokens of knock-out lists」被 `test_golden.py`
单独列为「deliberate difference」正是这件事。

### 2.3 `allow_none`：多一个「None of the above」

两个开关，行为不同（`controls.py:7-10`）：

- `none_check`（serve 默认**开**）：题目内部多一个 `"None of the above"` 选项（`prompt.py:32`
  `NONE_OPTION`，`prompt.py:186` 追加），读数里它**参与** softmax，答案报 `none_probability`，但
  `choice` **仍然是列出的选项里最大的那个**（`pipeline.py:355-360` 的 `null_ok`）。
- `allow_none`（默认关，`serve --allow-none` 可开）：在上面基础上让 `choice` **可以是 null**
  （`readout.py:80`）。

两者的关系在 `split_none`（`readout.py:87-94`）里：返回的 `probabilities` 是**列出的选项重归一化到
和为 1**，`none_probability` 是**重归一化之前**那个额外选项的概率，`abstain` 是「额外选项严格大于
每一个列出选项」（平局判给列出的选项）。加 `allow_none` 时若 26 个选项已经用满，先 knock-out，
额外选项**只进决赛轮**（`pipeline.py:216` 的 `len(crit) > MAX_OPTIONS - none`，以及 `pipeline.py:24-27`）。

---

## 3. 读位置：模板尾巴物化出来的空思考块

读位置不是标定出来的，是**被 chat 模板写出来的**。常量在 `prompt.py:33-34`：

```python
THINK_OPEN       = "<think>"
THINK_OFF_SUFFIX = "<think>\n\n</think>\n\n"   # 每个 one-pass prompt 的结尾
```

`plain_prompt`（`prompt.py:222-224`）用 `apply_chat_template(..., add_generation_prompt=True,
enable_thinking=False)` 渲染，于是 prompt 的最后四个 token 是**打开又立刻关闭的空思考块**，答案字母
就读在这个之后。`check_tokenizer` 第一步（`prompt.py:143-146`）断言模板确实以这个后缀结尾，否则抛
`"...this is not a StartLux-Decision tokenizer"`。**运行期没有任何代码去搜这个后缀**——它只在加载期
被断言一次。

这不是唯一的读位置。AutoThink 模型还会读**第二次**，位置在 thought 之后（`prompt.py:227-234`
`think_prompt` 用 `enable_thinking=True` 渲染并保证以 `<think>` 结尾，`pipeline.py:402` 再拼上截断或
闭合标记）：

```
read_prompt = base + seen + (CUT if ended == "budget" else CLOSE)     pipeline.py:402
CUT   = "\n\nI have to decide now with what I have worked out so far.\n</think>\n\n"   pipeline.py:55
CLOSE = "\n</think>\n\n"                                              pipeline.py:56
```

`seen` 是 thought **最后 `tail_chars` 个字符**，超过时前缀 `"[...]\n"`（`pipeline.py:399-401`，默认
4000 字符，`modelcfg.py:66`）。也就是说：**第二次读数的读位置依赖 thought 的长度，但 read_prompt 的
构造是确定性的**，没有标定自由度。

---

## 4. 一次前向：只读最后一个位置

读出的契约只有一个方法（`backends/base.py:86-87`）：

```python
async def readout(self, prompt: str, letter_ids: Sequence[int]) -> list[float]:
    """Next-token log-probabilities of the letters after the prompt, in the order of letter_ids."""
```

**它只接收 `letter_ids`，收不到词表大小**——这是四个后端「词表维度从不出现」的接口级原因。注意返回
的是**log-probabilities 或 logits 都行**（`backends/base.py:4-7`「only their differences are used」）；
这是成立的，因为 `probabilities` 是 `softmax(z/T)`，而 log-softmax 与 logits 只差一个常数，softmax
对常数平移不变。

四个后端各自怎么读：

| 后端 | 读法 | 位置 |
|---|---|---|
| vLLM | `/v1/completions`，`max_tokens=1`、`logprobs=k`、`temperature=1.0`、`allowed_token_ids=letter_ids`、`return_tokens_as_token_ids`，取 `top_logprobs[0]` | `vllm.py:269-276` |
| transformers | `hidden[0, -1].float() @ head.index_select(0, letter_ids).float().T` | `transformers.py:108-109` |
| llama.cpp | `n_predict=1`、`n_probs=k`、`post_sampling_probs`、`top_k=k`、对每个字母 `logit_bias +100` | `llamacpp.py:185-186` |
| fake（测试） | 由 prompt 的 hash 生成的确定性「log-prob」 | `fake.py:118-123` |

**没有 token 被生成。** vLLM 那条 `max_tokens=1` 是协议要求（要有一次采样），不是生成一个 token 再
丢弃；`transformers.py:108-109` 直接取最后一层的 hidden state 投影。

**与 Instant 文件夹的双向前向不冲突。** Instant 模型把全注意力层的因果掩码掀掉（见 §10.2），但读的
**仍然是最后一个位置**——区别在于那个位置的 hidden state 现在依赖整个 prompt，而不是只依赖它前面。
`transformers.py:105`（`bidirectional.text_forward`）与 `transformers.py:107`（因果前向）都返回
`last_hidden_state`，`transformers.py:109` 统一取 `[0, -1]`。

---

## 5. 掩码 + softmax

RYOTIDE 那一步是「读出整个词表分布，掩码到标签集，重归一化」。这条链上**掩码发生在请求里**，所以
没有「读出全词表再掩码」这个中间态：

- **vLLM**：`allowed_token_ids=list(letter_ids)`（`vllm.py:274`）交给采样器；配合启动参数
  `--logprobs-mode processed_logprobs`，返回的 log-prob **是掩码之后**取的（`launch.py:75-76`，
  理由写在 `launch.py:5-6`）。若某个字母没回来，`vllm.py:282` 给 `-1e9` 而不是 `-inf`。
- **transformers**：`index_select` 只取 K 行权重再乘（`transformers.py:108`），**词表矩阵被读，但只
  读 K 行**；词表维度的中间张量从不出现。
- **llama.cpp**：给 K 个字母各加同一个 `+100` 偏置（`llamacpp.py:75` `BIAS = 100.0`）并把 `top_k` 设为
  K，于是采样器只留这 K 个；软化的相对概率不变（理由写在 `llamacpp.py:6-9`）。缺字母直接抛
  `BackendError`（`llamacpp.py:207-210`），**不像 vLLM 那样补地板**。

**加载期守卫**：`VLLMBackend._probe`（`vllm.py:205-218`）在 `start()` 里发一次 noul 探针，要求 **26 个
字母全部返回且 log-prob > −9999**，否则报 `BackendError` 并打印该开的三个 flag。这把「掩码把字母吃掉」
从运行期故障变成了**起服务失败**——注意它检查的正是 `-1e9` 地板（`vllm.py:217` 的 `> -9999.0`），所以
那条地板不是容错，而是**探针的检测对象**。

**重归一化**发生在 `readout.probabilities`（`readout.py:16-21`）：

```python
m = max(z); e = [exp((x - m) / temperature) for x in z]; return [x / sum(e) for x in e]
```

减去 max 是数值稳定，不是掩码。掩码之后的归一化就在这一步完成——**K 个字母上的 softmax 天然和
为 1**；`score` 的期望值 `Σ i·p_i` 与 `choice` 的 argmax 都直接吃这个分布。

---

## 6. 顺序偏置

**这条链上不存在这个变量：既不处置，也不度量。**

三处核查：

- 引擎里**一次 `shuffle` / `permutation` 都没有**（`startlux_decide/**` 全目录零命中；唯一的
  `shuffle` 字样在 `loops.py:18` 的注释里，说的是模型自己把几行文字换序造成的假死循环）。
- 渲染按**请求列出的顺序**（`prompt.py:14` 的模块说明，`prompt.py:116` 的 `for k, i in enumerate(order)`）。
  唯一动过顺序的是 `score` 的层级：**总是最低在前**（`prompt.py:14`、`prompt.py:188-189` 用
  `score_keys` 排序，`prompt.py:161-166`：数字键按数值升序，否则按给定顺序）。
- 训练侧对齐同一个顺序：目标向量按**渲染顺序**构造
  （`v1/finetune/finetune_lora.py:48` `order = [o["id"] for o in row["options"]]`，
  `:52` `vec = [float(target.get(o, 0.0)) for o in order]`）。训练里唯一的 shuffle 是
  **bucket 顺序**（`v1/finetune/finetune_lora.py:78-79`），不是选项顺序。

也就是说：请求里换一次选项顺序就是换一次 prompt，答案可能变，而链上没有任何东西会告诉你变了多少。
**上游的语料构造是否打散过选项顺序，本文未核实**——数据构建不在本仓（§11 会说明 `v1/` 那棵树的处境）。

这一点与本仓另外三篇恰好形成光谱：Laya 在训练时消解、NanoJev 用折回实测证明 max logit 差 0.0、
decisio 是「无标定的第三种处置」。StartLux-Decision 是**第四种：不做，也不上报**。

---

## 7. 多问题共享 state

**做，但只共享前缀，不共享推理。**

`_share`（`pipeline.py:270-287`）在一次请求的所有问题都渲染完之后跑：

```
1. 收集本请求所有问题的 one-pass prompt（含 knock-out 的每组 prompt）   pipeline.py:275-276
2. 少于 2 条就返回                                                      pipeline.py:277-278
3. prefix = os.path.commonprefix(prompts)                              pipeline.py:279
4. 有图片且前缀里的占位符数 != 图片数 -> 放弃共享                        pipeline.py:280-281
5. n = 前缀 token 数 + 图片的额外 token 数                              pipeline.py:282
6. n < SHARE_MIN (4096) -> 放弃                                         pipeline.py:283-284
7. req.lane = backend.lane();  lane.prefill(prefix)                    pipeline.py:285-286
```

七道门里有三道是**拒绝共享**的理由，值得单独看：

- **`os.path.commonprefix` 是字符级前缀**，不是 token 级（`pipeline.py:279`）。它可能切在多字节
  字符或 token 边界中间 —— 这里不构成正确性问题，因为 `prefill` 收到的就是一个字符串前缀，各后端
  自己 tokenize；但如果字符前缀落在一个 token 内部，那条 token 在 prefill 与正式读里的切法必须
  一致才命中缓存，这一层对齐**代码未做保证**，本文未核实其命中率。
- **图片请求要求前缀里占位符数正好等于图片数**（`pipeline.py:280-281`），否则放弃。
- **双向文件夹从不共享**（`pipeline.py:273` 的 `self.folder.bidirectional`）：Instant 的 prompt 注意力
  是非因果的，**任何前缀的缓存状态都依赖于它后面的 token**，共享在构造上无效（同样理由写在
  `vllm_plugin.py:186-187` 的强制检查和 `patches/llama.cpp/0001` 的注释里）。

`SHARE_MIN = 4096`（`pipeline.py:61`）、`prefill` 的实现是「读一遍不回答」（`vllm.py:304-307`：一次
`max_tokens=1, temperature=0.0` 的调用，生成的 token 丢掉）。

**但不共享推理。** 每个问题是自己的 asyncio 任务（`pipeline.py:243`），各自可能走自己的 AutoThink
分支、各自动自己的 thought；`_gather`（`pipeline.py:133-142`）在任一题失败时取消其余题。代价是
N 个问题的 thought 之间**互相不可见**——它们碰巧都在同一个 state 上做决策，但推理过程不互通。
这与 Open-Jev 那类「N 个 pointwise 是非题」的形态相反：那是把一道题拆开，这是把多道题合在一个请求里。

---

## 8. 服务化与 fail-loud

### 8.1 HTTP 面

`server.py:1-22` 的模块说明就是权威的接口清单：

| 路由 | 行为 | 位置 |
|---|---|---|
| `POST /v1/systemone` | 决策 | `server.py:175-209` |
| `POST /v1/decisions` | 同一条路径的别名（OpenRouter 的命名），**请求里的 `model` 字段被忽略** | `server.py:6, 262-263` |
| `GET /v1/models` | `{name, release_date, description}`，`description` 里的 kind 由 `folder.kind` 得出 | `server.py:212-216` |
| `GET /health` | 状态与模型 kind；**有 API key 时对无 key 请求隐藏引擎地址与设置** | `server.py:224-237` |
| `GET /openrouter/models` | 运维自备的模型文档，原样回吐 | `server.py:219-221` |

错误码的形状（`server.py:20-21` + 实现）：**401** 无 key、**422** 服务器拒绝的请求（坏 JSON / 坏字段 /
模型答不了的问题）、**404/405** 未知路由、**429** 在飞请求超上限、**502** 引擎失败。响应始终带
`x-request-id`（客户端给了合法值就沿用）与同值的 `x-typesafe-request-id`（`server.py:88-89`）。

### 8.2 守卫都做成「拒绝启动」或「明确拒绝请求」

这是本条路线最值得对照 RYOTIDE 的一节 —— 它的 fail-loud 集中在**加载期**：

| 守卫 | 触发条件 | 位置 |
|---|---|---|
| 分词器不是 StartLux-Decision 的 | 模板不以 `THINK_OFF_SUFFIX` 结尾 | `prompt.py:143-146` |
| 字母不是 prompt 之后的单 token | 26 个字母逐个试 | `prompt.py:149-153` |
| 字母 id 与声明不符 | 现算 vs `letter_token_ids` | `pipeline.py:155-156` |
| vLLM 没开 `processed_logprobs` | 26 字母探针有一个没回来 | `vllm.py:205-218` |
| vLLM 版本低于 0.31 却要 stream thought | 起服务时 | `pipeline.py:159-161`；请求时 `controls.py:103-104` |
| Instant 模型被要求思考 | 请求里 `think_mode != never` | `pipeline.py:233-234`、`controls.py:76-79` |
| Instant 文件夹被当因果读 | vLLM 插件未激活 | `launch.py:293-302`（要求每个 server 日志里有那句 `ACTIVE_LINE`） |
| vLLM 跑双向文件夹时开了前缀缓存 | 插件自己的检查 | `vllm_plugin.py:186-187` |
| 引擎 refusals 归因到具体问题 | `BackendRequestError` -> `RequestError` 带 `("body","questions",key)` | `pipeline.py:294-297` |

注意最后一条：引擎拒绝（422）与引擎故障（502）是**分开的**（`server.py:196-200`），而且拒绝会被挂
上出问题的那道题的 key，不是笼统的 500。

### 8.3 公开服务面

`ServeOptions`（`serving.py:232-240`）默认**全关**。打开后：

- API key 存的是 **SHA-256 摘要**，比较用 `hmac.compare_digest` 遍历全部 key（无短路、无长度泄漏，
  `serving.py:87-97`）；日志里出现的是 **key id**（sha256 前 16 个十六进制位，`serving.py:100-107`）。
- key 文件、模型文档都是 `WatchedFile`：改了才重读，**读坏了保留上一版并告警**（`serving.py:25-55`）。
- 在飞超限**立刻 429 + `Retry-After: 1`，从不排队**（`server.py:160-167`）。
- 访问日志**不含任何内容**（`server.py:98-110`），请求日志（可选）才存内容，目录 `0o700`、文件 `0o600`
  （`serving.py:169-211`）。

---

## 9. 置信度从哪来，又怎么被改写

### 9.1 三个题型三个公式，全是从概率直接导出的

| 题型 | 报什么 | 公式 | 位置 |
|---|---|---|---|
| `choice` | `confidence` = 最大概率相对均匀分布高出多少 | `(n·p_max − 1) / (n − 1)`，夹到 `[0,1]` | `readout.py:37-41` |
| `score` | `score` = 期望档位；`confidence` = 1 − 概率加权的档位距离 / 均匀分布的同距离 | `1 − Σp_i·\|i−m\| / (Σ\|i−(n−1)/2\|/n)`，下限 0 | `readout.py:44-51`、`readout.py:70` |
| `noul` | `noul` = **答 yes 的概率**（不是 confidence） | `p[ids.index("true")]` | `readout.py:64-66` |

`choice_confidence` 定义在 **n−1 格**上：均匀分布给 0，全部质量在一项给 1（`readout.py:38-39`）。
`n < 2` 直接返回 1.0（`readout.py:41`）—— 但单选项的问题根本不会走到这里（§2 的 `only` 分支直接报
`confidence: 1.0`，`pipeline.py:302-303`）。

**置信度是在哪份分布上算的，有个容易读错的地方。** 有 `none_check` 时，`choice` 的 `confidence` 算的
是**列出选项重归一化之后**的分布，不是含 none 的原始读数：`_none_answer`（`pipeline.py:355-360`）先
`split_none`（`readout.py:87-94`）拿到重归一化的 `listed`，再把它交给 `choice_answer`
（`readout.py:80-81`）。knock-out 之后同理，`knockout_combine` 最后整体重归一化（`readout.py:131-132`）。

### 9.2 谁在改发布数值：一条链表

这条链上**唯一一个能改数值的标量是逐题型温度**，而它随权重发布：

```
① 拟合     v1/finetune/calibrate.py
   网格     GRID = 801 点，对数间隔，0.2 → 5.0（含两端）            calibrate.py:22
   目标     NLL 最小（不是 ECE；ECE 只报不优化）                    calibrate.py:25-30, 75
   每题型   样本 >= 200 才单独拟合，否则用**合并温度**              calibrate.py:79
   落盘     就地改写 decision_config.json 的 temperature_by_type
            值 round 到 4 位，整个文件按 indent=2 重写               calibrate.py:86-89
           （只有 --dry-run 能拦；没有备份，也不检查新值是否更好）    calibrate.py:85

② 发布     decision_config.json: temperature_by_type
   例（4B-AutoThink）  noul 1.0054 / choice 0.9276 / score 0.9503    Hub 核对 §13.2
   例（4B-Instant）    noul 1.2662 / choice 1.1103 / score 1.1344
   例（v1-27B）        noul 1.421  / choice 1.2098 / score 1.2294

③ 读取     modelcfg.py:213-215 校验为正数；modelcfg.py:162-163 缺省 1.0

④ 施加     probabilities(z, T[type]) = softmax(z / T[type])          readout.py:16-21
           pipeline.py:289-291 每条读数都过一次

⑤ 二次读数**换一个温度**  retemper(p1, T[type], T_think)              readout.py:24-29
   AutoThink.think_temperature 默认 {choice:1.0, noul:1.0, score:1.0}  modelcfg.py:68
   注意这不是恒等：p1 是 T[type] 下的分布，retemper 把它换成 T_think=1.0 下的分布
   （retemper(p, T, 1.0) == probabilities(z, 1.0)，断言在根目录 tests/test_readout.py:20-23）

⑥ 混合     p = (1 − mix)·p0 + mix·p1，mix 默认 0.75                    readout.py:32-34
           `mix` 是**每请求可覆盖**的（controls.py:82-83），服务默认来自文件夹   pipeline.py:333

⑦ 读数 0/1 地板   retemper 里 max(v, 1e-300)                          readout.py:27
⑧ 缺字母地板      vLLM 后端给 −1e9                                    vllm.py:282
⑨ knock-out 残差  未进决赛的选项分摊 wide_residual（默认 1e-3）        readout.py:124-132
```

**发布出去的数值在这一步之后不再被改。** 没有 `round`、没有 clamp、没有「托管面另算一个置信度」。
响应里的 `confidence` 是 `readout.py:37-51` 那两个函数的原值，`score` 是 `Σ i·p_i` 的原值
（`readout.py:70`）。整条链上被 `round` 的只有计时常量（`pipeline.py:129-130` 的毫秒）与
`usage` 里的整数 token 数；`v1/finetune/calibrate.py:81-83` 打印的 `*_before` 数值是**在 T=1.0 下**算
的，不是当前发布温度下的值 —— 引用那些字段时要注意。

### 9.3 AutoThink：本族新增的第二个维度

RYOTIDE 那条读法一次决策读一次。这条链上**可能读两次**，判据与合成方式都是常量：

```
触发     think_mode == "always"，或 "auto" 且 max(p0) < gate，且 room > MIN_ROOM
         gate 默认 0.6（modelcfg.py:62）——但**发布数字用的是 0.9**（modelcfg.py:59-60 明说）
         room = context_length − n_in − CONTEXT_MARGIN(64)，MIN_ROOM = 32             pipeline.py:59-60, 325-326
thought  AutoThink 默认：prefix "Let me work through this carefully, step by step.\n"
         （modelcfg.py:43 RELEASE_PREFIX）、budget 4096、tail_chars 4000、chunk_tokens 1024
         采样 temperature 0.6 / top_p 0.95 / top_k 20（modelcfg.py:69）
         seed 由 prompt 的 sha1 定死（pipeline.py:395），chunked 路径每轮 +7919×round
二次读数 读 read_prompt，retemper 到 think_temperature，再按 mix 混合
```

**`gate` 的默认值与发布值不一致，这件事上游自己写在了代码注释里**（`modelcfg.py:58-60`：发布数字用
0.9，默认 0.6「answers more requests in one pass」），README 的 Controls 表也重复了一遍
（README.md:150）。所以「用这条链复现发布分数」必须显式设 `--gate 0.9`。

**thought 的两条生成路径**（`pipeline.py:405-427` chunked / `pipeline.py:429-458` stream）在默认
`thought_generation: "auto"` 下由**引擎会不会安全地流式**决定（`pipeline.py:178-182`）：vLLM ≥ 0.31
走 stream，否则走 chunked。两条路径的 seed、采样、stop、budget 与循环检测**都相同**（`pipeline.py:12-16`），
差别只在传输方式与「循环检测每 `chunk_tokens` 个 token 跑一次」落在哪一层。

---

## 10. 四条不变量的落位

RYOTIDE §13 那四条在这条链上的状态：

| RYOTIDE §13 | 本实现 |
|---|---|
| **#1** 读位置必须落在标记上（诊断量 `marker_mass`） | **改写**：读的不是一个标记，而是 26 个单 token 字母；`marker_mass` 概念不适用（没有「标记」这个东西）。等价命题是「读位置必须落在 26 个**互不相同、且在完整 prompt 之后仍为单 token**的字母上」，判据在 `prompt.py:149-155`；诊断量也存在，是 `vllm.py:205-218` 的加载期 26 字母探针（它检查的正是 `-1e9` 地板） |
| **#2** 前缀与标记表面形式耦合，必须成对扫 | **改写**：不扫表面形式，而是**在加载期断言掉这个自由度**（`prompt.py:143-146` 断言模板尾巴、`prompt.py:149-153` 断言字母单 token 性），不合规的分词器**直接拒起**。与 SemIf-OpenJev 那条「断言掉自由度」同型，但断言的是一段**模板后缀**而不是一个编码等式 |
| **#3** 只读最后一个位置 | **成立**。四个后端都只取末位（`transformers.py:109` 的 `hidden[0,-1]`；vLLM 是 `max_tokens=1` 的第 1 个输出位，`vllm.py:273`）。Instant 文件夹掀掉因果掩码后**仍然只读末位**——变的是那个位置的 hidden 依赖整段 prompt，不是读几个位置 |
| **#4** 比较概率 / ECE 前先确认 temperature | **成立，且是这条链上唯一一个能改数值的标量**。逐题型温度随权重发布（§9.2），而且**要求确认的不止一个数**：还有 `think_temperature`（默认 1.0，只作用于二次读数）与 `mix`（默认 0.75）。三者在 `decision_config.json` 与 `auto_think` 块里分成两处 |

各自新增的：

- **渲染器是权重的一部分。** `prompt.py:5-6` 把这句话写成了模块约束：*Every string in this module is part
  of the trained format: changing any of them changes the answers.* 所以它不是「配置」，不能改。
- **发布物里的 `format` 字段挂着一个更早的项目名。** 见 §13.2。
- **`v1/` 那棵树引用一个本仓不存在的包。** 见 §11。

---

## 11. 文件对照表

`startlux_decide/`（6,620 行）：

| 文件 | 行数 | 职责 |
|---|---|---|
| `vllm_plugin.py` | 861 | vLLM 的 prompt-双向注意力插件：monkeypatch FlashAttention 与 runner，把 prompt 行用非因果 varlen 重算覆盖 |
| `pipeline.py` | 458 | 决策主循环：knock-out、门控、thought、两次读数、混合、共享前缀 |
| `backends/mlx.py` | 431 | MLX / mlx-vlm 后端 |
| `cli.py` | 379 | `serve` / `launch-vllm` / `bench` / `eval` / `quantize` 五个子命令 |
| `backends/vllm.py` | 360 | vLLM 客户端：读法探针、wave 调度、流式 thought、前缀缓存 |
| `launch.py` | 342 | vLLM 启动参数与「插件真的生效了吗」的检查 |
| `backends/llamacpp.py` | 302 | llama-server 客户端：logit_bias 读法、图片、双向模式协商 |
| `server.py` | 269 | aiohttp 服务、错误码、请求 id、访问/请求日志挂钩 |
| `modelcfg.py` | 257 | `decision_config.json` / `config.json` 的读取与校验、`AutoThink` 默认值 |
| `serving.py` | 240 | API key、WatchedFile、请求日志、TLS |
| `prompt.py` | 239 | 模板、选项渲染、`check_tokenizer`、`/v1/systemone` 题面 |
| `backends/bidirectional.py` | 202 | transformers 路径上掀掉全注意力层因果掩码 + 三项自检 |
| `evaluate.py` | 181 | 准确率 / ECE / Brier / 配对 bootstrap 比较 |
| `backends/transformers.py` | 176 | 进程内参考引擎 |
| `backends/fake.py` | 173 | 确定性假模型（测试与无 GPU 试客户端） |
| `images.py` | 167 | 图片占位、缩放边界、pass-through 模板 |
| `readout.py` | 132 | 概率、重定温、混合、三个置信度公式、knock-out |
| `bench.py` | 117 | 延迟压测与 CSV |
| `controls.py` | 106 | 每请求开关的解析与校验 |
| `backends/base.py` | 104 | Backend 抽象：`readout` + `generate` 两个原语 |
| `quantize.py` | 100 | FP8 / GGUF / MLX 三条量化路径的说明与执行 |
| `loops.py` | 86 | 死循环检测（v3 规则集）与切割点二分 |
| `backends/http.py` | 85 | JSON-over-HTTP 与 SSE |

**`v1/` 这一棵树不是可运行的。** 它引用的包叫 `startlux_decision`（**带 n**），而本仓的包是
`startlux_decide`；`ls -d startlux_decision` 报不存在，`pyproject.toml:48-49` 也只打包
`startlux_decide*`。具体地，`v1/` 下 9 个文件 import 了那个包
（`v1/finetune/finetune_lora.py:31-32`、`v1/finetune/calibrate.py:19-20`、`v1/eval/typed_decisions.py:90`、
`v1/eval/suites.py:129-130`、`v1/eval/di/startlux_decision_engine.py:26` 等），并且用到的符号
`StartLuxDecision`、`jevfmt`（`J`）、`padded_length`、`render_ids`、`model._logits`
**在本仓任何地方都没有定义**（全仓 grep 只命中 `v1/` 内部）。也就是说 `v1/` 是一份**更早的、
项目改名前**的代码副本，README:213-215 说「Its inference code, evaluation scripts, results, game logs
and demo harnesses are in v1/」——那些脚本在这里**跑不起来**。本文引用 `v1/` 的 `文件:行` 时，锚的是
**文本**，不是可执行语义。

**本文没有读的**：`backends/mlx.py`（431 行）只从模块说明与 `cli.py` 的接线判断其存在与形态，未逐行走查；
`v1/demos/` 下的两个演示夹具（chess / computer_use）未读。

---

## 12. 亲手跑一遍

以下命令全部来自 `README.md:96-107`、`cli.py` 与 `launch.py`，**本文没有执行过其中任何一条**：前两条
需要 GPU 与 9 GB 以上权重，`launch-vllm` 需要启动 vLLM 服务器进程。

```bash
# 装包（vLLM 0.19.1 是发布数字的那一版；0.31 也支持）
pip install -e '.[vllm]'

# 起一个 vLLM 服务器 + 决策服务（会打印实际命令行，含 processed_logprobs 等 flag）
startlux-decide serve --model models/StartLux-Decision-v1.1-4B-AutoThink \
    --launch-vllm --gpus 0,1,2,3 --port 8090 --dry-run     # 先看要跑什么
startlux-decide serve --model models/StartLux-Decision-v1.1-4B-AutoThink \
    --launch-vllm --gpus 0,1,2,3 --port 8090 --gate 0.9    # 复现发布数字要用 0.9

# README.md:111-122 的那三问（choice / noul / score）
curl -s localhost:8090/v1/systemone -H 'Content-Type: application/json' -d @- <<'JSON'
{"state": {"ticket": "I was charged twice for order #4411 and the app still shows it as unpaid."},
 "questions": {
   "team":     {"type": "choice", "instructions": "Which team should handle this ticket?",
                "criteria": {"billing": "Payments and refunds", "shipping": "Delivery", "technical": "App and login"}},
   "urgent":   {"type": "noul", "instructions": "Should this ticket be answered today?"},
   "severity": {"type": "score", "instructions": "How severe is the impact?",
                "criteria": ["cosmetic", "annoying", "blocks the customer"]},
   "who":      {"type": "choice", "instructions": "Who owns it?",
                "criteria": {"a": null, "b": null, "c": null}}}}}
JSON
# 加 return_details 会带出每一次读数的 probabilities、thought 的 tokens/ended/ms、gate 与 mix

# 把每一步的读数摊开：回看 thought 文本
curl -s localhost:8090/v1/systemone -H 'Content-Type: application/json' \
  -d '{"state": "s", "return_details": true, "return_thought": true,
       "questions": {"q": {"type": "noul", "instructions": "Is it?"}}}'

# 延迟与校准
startlux-decide bench --url http://127.0.0.1:8090 --requests reqs.jsonl --concurrency 8
startlux-decide eval  --url http://127.0.0.1:8090 --data dev.jsonl --out eval.json

# 只用标准库跑单元测试（不需要 aiohttp / torch / transformers）
python -m unittest -v
python -m unittest -v tests.test_golden      # 包内 pipeline 对照 reference_impl，1500 个随机请求逐位比
```

**`tests/test_golden.py` 是这条链上最值得读的一处对照物**：它拿本包的 `Decider + VLLMBackend` 与
`tests/reference_impl.py` 里那份「发布代理」的逐题流水线，**接同一个确定性假 vLLM**，跑 1500 个随机
请求，要求**发给引擎的 body 逐字节相同**（含每一段 thought chunk、seed、每次读数 prompt）且
**概率 / choice / score / thought token 数逐位相同**（`tests/test_golden.py:1-8`）。置信度公式、score
图例顺序、knock-out 的 `input_tokens`、`mix=0` 与单选项列表这几处「刻意不同」被单独列出来检查。这条
测试的形态说明**上游知道哪些行为是契约、哪些是实现细节**——这是本仓二十二条里少见的写法。本文没有
运行它（它不加载权重，但属于上游程序，按本仓约定不执行）。

---

## 13. 模型描述：以 Hub 为准

### 13.1 发布清单（枚举而非抽查）

`HfApi().list_models(author='startlux-models')` 共 65 个仓，其中 `Decision` 系列 **53 个**：

| 代 | 基座仓 | 衍生仓 | 衍生形态 |
|---|---|---|---|
| v1 | 6：`0.8B` / `2B` / `4B` / `9B` / `27B` / `35B-A3B` | 15 | 前五个各带 `BF16` / `Q4_K_M` / `Q8_0` GGUF（35B-A3B 无 GGUF） |
| v1.1 | 8：`27B-AutoThink` / `27B-Instant` / `9B-AutoThink` / `9B-Instant` / `4B-AutoThink` / `4B-Instant` / `2B-Instant` / `0.8B-Instant` | 24 | 每个各带 GGUF / MLX-8bit / MLX-4bit |

合计 6 + 15 + 8 + 24 = 53 ✓，且 README:64-71 那张 8 行表 × 4 列（本体 + GGUF + MLX 8bit + MLX 4bit）
正好等于 v1.1 的 32 个仓 —— **README 的发布清单是完整的，没有漏掉的尺寸或量化档**。

其余 12 个仓是 `gdn-*` 研究模型（`text-generation`），不属于 Decision 系列。

### 13.2 基座、参数量、体积与那处命名考古

| 事实 | 值 | 来源 |
|---|---|---|
| **基座声明** | **14 个基座仓全部 `base_model = None`**；39 个衍生仓**全部**声明 `base_model = <对应的基座仓>` + `base_model_relation = quantized` | Hub API，全区枚举 |
| 基座身份的唯一线索 | `config.json` 的 `architectures` = `Qwen3_5ForConditionalGeneration`（35B-A3B 是 `Qwen3_5MoeForConditionalGeneration`），`model_type = qwen3_5`，`text_config.vocab_size = 248320` | Hub `config.json` |
| 4B-AutoThink 参数量 | `4,539,265,536`（safetensors 元数据 `total`）；权重文件合计 `9,078,619,848 B` = 9.08 GB / 8.46 GiB | `hf-model-facts.py facts` |
| 4B-Instant 参数量 | `4,659,865,088`；权重合计 `9,319,828,240 B` = 9.32 GB / 8.68 GiB | 同上 |
| BF16 反证 | 参数量 ×2 与权重字节的差是各分片的 safetensors 头（4B-AutoThink：9,078,619,848 − 2×4,539,265,536 = **88,776 B**，跨 3 个分片；4B-Instant 差 98,064 B，跨 2 个分片）⇒ 权重确是 BF16 | 实测 |
| 上下文 | `text_config.max_position_embeddings = 262144`（= 256 K），与 `modelcfg.py:16` 的「262,144 tokens in these folders」一致 | Hub `config.json` |
| 层型 | `layer_types` 是 3 层 `linear_attention` + 1 层 `full_attention` 的循环，`num_hidden_layers = 32` —— 对应 vLLM 插件的「只改全注意力层」与 0.19.1 的混合模型 cache 坑 | Hub `config.json` |
| 双向标记 | `jev_prompt_attention: "bidirectional"` **只出现在 Instant 仓**（AutoThink 仓没有这个键） | Hub `config.json` |
| **`format` 的值** | v1.1 的四个仓写的都是 **`"lodestar-letter-readout-v1"`**；v1 的仓写 `"StartLux-Decision-v1"` | Hub `decision_config.json` |
| 代码里唯一比较的值 | `V1_FORMAT = "StartLux-Decision-v1"`（`modelcfg.py:44`），只在 `modelcfg.py:240` 的 `v1 = mode == "causal" and fmt == V1_FORMAT and "auto_think" not in cfg` 里用一次 | 代码 |

两处值得单独说：

**① 上游没有声明基座。** 模型卡里连 `base_model` 标签都没有，`config.json` 只说架构是 Qwen3.5 系。
README.md:99 把起点写成 `pip install -e '.[vllm]'` 并让读者直接 `hf download` 发布仓，没有提到任何上游
基座仓。所以「这四个尺寸是从哪个 Qwen3.5 检查点继续训出来的」在公开物里**无法确认**，本文不填。

**② `format` 挂着一个更早的项目名。** `lodestar-letter-readout-v1` 在代码里**从不被比较**：唯一被比较
的字符串是 `StartLux-Decision-v1`。后果是**有利的**（v1.1 的因果仓因此不会被误判成 v1，`kind` 正确
地报 `auto_think`），但这说明 `format` 这个字段在 v1.1 的发布物里是**改名前的遗留值**，任何按
`format` 分派的外部工具都会认错代际。这是本仓记录的又一处「命名考古」，与 decisio 那篇里线格式仍读
旧拼写 `rlcd-*` 同型 —— 但那里是**向下兼容地读旧名**，这里是**新发布物反过来写着旧名**。

### 13.3 在 committed 产物上重算（「实测」）

以下四类**都不运行上游程序**，全部是在提交进仓的文本上重算。

**① 27B 示例响应的两条置信度公式（README.md:126-136）。** README 给了一口 27B AutoThink 的示例响应
（自述「numbers rounded」），三道题的读数都三位小数：

| 题 | 发布值 | 本文重算 | 差 |
|---|---|---|---|
| `team` 的 `confidence` | 0.992 | `(3 × 0.995 − 1) / (3 − 1)` = **0.9925** | 0.0005 |
| `team` 的 `probabilities` 和 | 1.000 | 0.995 + 0.000 + 0.005 = **1.000** | 0 |
| `severity` 的 `score` | 1.858 | `0×0.005 + 1×0.133 + 2×0.863` = **1.859** | 0.001 |
| `severity` 的 `confidence` | 0.787 | `1 − 0.143 / 0.666667` = **0.7855** | 0.0015 |
| `severity` 的 `probabilities` 和 | — | 0.005 + 0.133 + 0.863 = **1.001** | 0.001 |

四处的偏差都在**三个概率各自被舍到三位小数**带来的上界之内（每项 ≤ 0.0005，`score` 的加权上界
0.001，`confidence` 分母 2/3 放大后 0.0015）。结论是[**公式对得上、数值差在舍入**]：`confidence` 算的
确实是**列出选项重归一化后**那份分布（§9.1），`score` 确实是 `Σ i·p_i`。这条重算的价值在于它**能区分
两种可能**——如果 `confidence` 是含 `none_probability` 的原始分布上算的，`(3×0.995−1)/2` 就不再是那
个数了。

**② `v1/docs/results.md` 的两张汇总列。** 该文件第 30-37 行给六个模型的七套件准确率与两列汇总。
用七个数重算 `Average`、用去掉 ToolACE 的六个数重算 `Average without ToolACE`：

| 模型 | Average 重算 | 发布 | 差 | 去 ToolACE 重算 | 发布 | 差 |
|---|---|---|---|---|---|---|
| 0.8B | 85.0286 | 85.03 | −0.001 | 83.8767 | 83.88 | −0.003 |
| 2B | 88.4643 | 88.46 | +0.004 | 87.6700 | 87.67 | 0.000 |
| 4B | 91.1714 | 91.17 | +0.001 | 90.6683 | 90.67 | −0.002 |
| 9B | 91.0829 | 91.08 | +0.003 | 90.4567 | 90.46 | −0.003 |
| 27B | 91.8229 | 91.82 | +0.003 | 91.3733 | 91.37 | +0.003 |
| 35B-A3B | 92.2886 | 92.29 | −0.001 | 91.7567 | 91.76 | −0.003 |

六行十二列的偏差全部 ≤ 0.004，量级等于七个分量各自被舍到两位小数（每个 ≤ 0.005/7 ≈ 0.0007，六项
平均后 ≤ 0.0008；实测最大 0.004 略大，来源是分量本身已是舍入值）。**两列汇总的定义因此可以确定**：
`Average` 是七套件的**等权算术平均**，`Average without ToolACE` 是去掉 ToolACE 后的六项等权平均
—— 不是按题数加权。

**③ JevBench 题数 ↔ 百分比。** 同文件第 49-65 行的「Counted in items」表与上表的 Easy / Original /
Hard 三列必须互相自洽（分母 48 / 72 / 111，合 231）：

| 系统 | Easy | Original | Hard | 合计 |
|---|---|---|---|---|
| 35B-A3B | 48/48 = 100.00 ✓ | 71/72 = 98.61 ✓ | 91/111 = 81.98 ✓ | 210 |
| 27B | 48/48 ✓ | 72/72 = 100.00 ✓ | 88/111 = 79.28 ✓ | 208 |
| 4B | 48/48 ✓ | 72/72 ✓ | 84/111 = 75.68 ✓ | 204 |
| 9B | 48/48 ✓ | 70/72 = 97.22 ✓ | 83/111 = 74.77 ✓ | 201 |

**12 格逐格吻合，0 处不一致**，且 `48 + 72 + 111 = 231` 与文中「231 public JevBench items」一致。

**④ README 的两组增量断言。** README.md:77-78 说 AutoThink 相对同尺寸 Instant 在 DecisionPivot 上
「adds 11.5 to 18.4 points」，按表内数字重算三个可比对：27B `86.9 − 75.4 = 11.5`、9B `59.9 − 47.5 = 12.4`、
4B `59.9 − 41.5 = 18.4` —— **端点恰好就是 11.5 与 18.4**，断言的区间就是三个差的最小/最大值。
另一条「Instant models exceed the strongest evaluated decision model of comparable size by 7.4 to 16.9
points」**无法在仓内重算**（对照模型的数字不在本仓），本文不核。

**⑤ 同 README 的两处自述互相打架（本文照实报，不解释）。**

- 模型表（README:64）给 27B AutoThink **Decision Index 0.3 = 73.37**、**DecisionPivot = 86.9**，并声明
  「The AutoThink results use a gate of 0.9」（README.md:75）。
- Controls 一节（README.md:158）说同一个模型、同一个 gate：「moving the gate from 0.9 to 0.6 lowers
  Decision Index 0.3 from **73.22** to 72.85 and DecisionPivot pair accuracy from **88.1%** to 68.4%」。

两处都自称是 gate 0.9 下的 Decision Index 0.3 / DecisionPivot，数值却是 73.37 vs 73.22 与 86.9 vs 88.1。
本文**不选一个当真值**：仓内没有任何一份 committed 读数能判定哪一组对（§11 已说明 `v1/results/` 与
`v1/docs/results.md` 提到的 `summary.json` / `summary.csv` 都不在仓里 —— 见下），所以这里只能记「上游
两处自述不一致，差额 +0.15 与 −1.2」。

**⑥ 一处被文档引用但不存在的产物。** `v1/docs/results.md:5-6` 写 *A machine-readable copy of our numbers
is in [results/summary.json](../results/summary.json) and [results/summary.csv](../results/summary.csv)*。
实测：`ls v1/results` 报 `No such file or directory`；`find . -name 'summary*'` 零命中；`git ls-files v1`
里没有任何 results 文件。**那两份机器可读副本不在仓里**（无论工作区还是 git 历史）。所以上面 ② ③
两类的重算只能用 **markdown 表格里的数字**，不能用「原始读数」。这也是本文不给任何由前向产生的数值
的一个具体后果：连上游自己声称存在的读数文件都没有随仓发布。

### 13.4 与 `patches/llama.cpp/README.md` 的两组分差

该 README 给了两组可交叉核对的分差，本文只做算术，不核对它们背后的测量：

- `README.md:19-20`：4B Instant Q8_0 在 32 张图片题上，stock 服务器（图片前文本单独一个 micro-batch）
  与 BF16 的平均差 **0.022**、最大 0.22；打上 patch 0003 后 **0.0055**、最大 0.06。同一段话在
  `llamacpp.py:53-54` 复述为「约 4 倍」，而 0.022 / 0.0055 = **4.0** ✓、0.22 / 0.06 ≈ **3.67**。
- `README.md:26-27`：3 个分差被并置 —— 用默认设置（批处理 prompt、切掉最后几个 token）代价 **7.7 分**，
  以因果方式读 prompt 代价 **26 分**。这两者**没有可重算的来源**（DecideBench 的逐题读数不在仓里），
  本文只记数值与出处。

---

## 14. 小结：这条路线把自由度放在了哪

如果只看一页，这是分界线：

| 环节 | 这条链上在哪定 | 能不能在服务端改 |
|---|---|---|
| 标签集（26 个字母） | 分词器 + `decision_config.json`，加载期逐项核对 | **不能**（改了拒起） |
| 读位置 | `THINK_OFF_SUFFIX`，加载期断言模板尾巴 | **不能** |
| 选项渲染 | `prompt.py` 的每个字符串 | **不能**（改了就是换模型） |
| 温度 | 逐题型，拟合后**随权重发布** | **不能**（只能换文件夹） |
| 要不要思考 | `gate` / `think_mode` | **能**，每请求可覆盖；默认 0.6 ≠ 发布值 0.9 |
| 二次读数怎么合 | `mix` / `think_temperature` | **能**改 `mix`（每请求），`think_temperature` 只随文件夹 |
| 置信度 | 由概率直接导出 | **不能**，链上没有任何改写步骤 |

一句话：**它能改的只有「想不想、想多久、两次读数怎么称重」，读法本身是冻结的。** 这与 RYOTIDE 恰好
互补 —— 那边读法可配、权重冻结；这边权重是训出来的、读法被训进去，换来的是一个没有任何标定代码、
没有任何扫描、也没有 `marker_mass` 的服务端。

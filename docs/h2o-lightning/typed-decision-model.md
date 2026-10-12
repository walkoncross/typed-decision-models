# H2O-Lightning-4B 是如何把「一次决策」做成一次 1-token 的候选标签读数的

> 代码库：<https://huggingface.co/h2oai/h2o-lightning-4b>
> 对应 README 第 24–25 行：*It runs on **unmodified vLLM 0.30.0** with a small standard-library shim in front
> (`h2o_lightning_shim.py`). Each decision is one forward pass and one output token, so the cost is input tokens only.*；
> 以及 README 第 82 行：*Calibration as a first-class goal: one forward pass, one output token, and probabilities that
> mean what they say.*
> 本文按 `h2o_lightning_shim.py`（唯一的实现文件）、`serve_config.json`、`serve.sh`、`chat_template.jinja` 与
> `routing_reproducibility/` 的实际代码逐环节说明这两句话的实现方式；文中出现的文件名与行号
> （`h2o_lightning_shim.py:837`、`serve_config.json:16`、`routing_thresholds.py:38` 等）均相对该仓库根目录。
>
> **同名配对文档**：RYOTIDE 的 [`docs/ryotide/typed-decision-model.md`](../ryotide/typed-decision-model.md)。
> §0 先讲清分歧，§13 把它的四条不变量逐条落位。
>
> **本文读的是哪一个 revision**：`h2oai/h2o-lightning-4b` 的 `b43f759b6385d21182f87c3829931ec3d4dfa98a`
> （2026-10-12T01:39Z，45 个文件）。该仓**没有 Git 之外的代码来源**：这个 HF 模型仓就是上游本体，`h2o_lightning_shim.py`
> 只在这里发布过一份。所以本文引用的一切都能按上面的 revision 逐字节比对（§16 给出复算）。

H2O-Lightning-4B 与 RYOTIDE 同属「把 LLM 当分类器读」这一支：一次前向、只取读位置的 logits、在已知标签集上取最大。
它往前推的是另外四件事：

1. **标签集不靠分词器构造，靠加载期逐一验证**。`serve_config.json` 里的 255 个标签**不是**「单 token 字母表」——
   其中 178 个是 BPE 片段（`in`、`er`、`the`、`ing`…，`serve_config.json:16`），凭构造根本不能保证是单 token。
   shim 起服务后第一次被调用时，把每个标签写成 `" " + 标签` 接到探针 prompt 后面，要求**恰好 +1 个 token 且互不重复**，
   否则 503 拒服务（`h2o_lightning_shim.py:936-944`）。「typed」因此是一个**运行期被证明的性质**（§2）。
2. **读位置是 prompt 的一部分，不是标定出来的**。prompt 末尾固定为 `<|im_start|>assistant\n<think>\n\n</think>\n\n`
   加上字面量 `Answer:`（`h2o_lightning_shim.py:75`、`serve_config.json:13`），答案就是紧跟其后的那个 token。
   运行期没有扫描、没有阈值、没有候选位置——标定被换成了**训练期钉死 + 加载期校验**（§3）。
3. **一次前向只读 K 个 token 的 logprob，词表维度从不出现**。它不取 `top_logprobs` 的前 k 个再筛，
   而是把该题的 K 个标签 token id 作为 `logprob_token_ids` 直接发给 vLLM（`h2o_lightning_shim.py:842-845`），
   所以回包里只有这 K 个数。于是 RYOTIDE 的 `marker_mass` 在这里**没有对象**（§5）。
4. **发布的不只是权重，还有一张可复算的升级表**。`routing_reproducibility/` 里有 3,474 行逐题读数、
   两份第三方题集、一份 sha256 清单和一个重算脚本（§12）；本文 §16 在自己的代码上把那张表重算了一遍。

## §0 与 RYOTIDE 的分野

| 维度 | RYOTIDE | H2O-Lightning-4B |
|---|---|---|
| 标签集从哪来 | 标签集合在请求里声明，靠分词器保证单 token | `serve_config.json:16` 的 255 个标签，**加载期逐标签探针证明**单 token 且互异（§2） |
| 读位置怎么定 | 运行期扫前缀、按标记质量标定 | prompt 末尾的字面量 `Answer:` 之后那一位，训练期钉死（§3） |
| 一次前向读几个位置 | 一个（末位） | 一个；但**每题各一次前向**，多题不做 K 槽位（§4） |
| 掩码与重归一化 | 全词表 softmax → 掩码到标记 → 重归一化 | **没有掩码**：只请求候选的 K 个 token id，再在这 K 个上 softmax（§5） |
| 置信度从哪来 | 读出来的 | 读出来的；但发布值被**温度**与**yes/no 地板**先后改写（§9） |
| 顺序偏置 | 有诊断量 | **不存在该变量**（§6） |
| 多问题共享 state | — | vLLM 前缀缓存共享，`input_tokens` 把共享头只计一次（§7） |
| `output_tokens` | 0 | 0，且是响应组装处的字面量（`h2o_lightning_shim.py:1067`） |

一句话总结立场：**它把 RYOTIDE 的运行期标定整条换成了「训练期钉死 + 加载期证明」，再把「发布什么」从权重扩展到
一张可复算的升级表；代价是标签集一旦不是 255 个以内的单 token，服务直接拒起。**

## §1 端到端数据流

```
POST /v1/systemone  {"state": …, "questions": {qid: {type, instructions, criteria}}}
    │   （别名 /api/alpha/decisions，h2o_lightning_shim.py:1118）
    │
    ├─ Contract(cfg)                     serve_config.json → 温度/标签/布局/prefill/地板（:697-752）
    │
    ├─ Shim.check_backend()  首次即用      GET /v1/models → 不是配置里的 model 就 503（:922-930）
    │                                   逐标签探针：tokenize(text+" "+L) == tokenize(text)+1 且 id 互异，
    │                                   否则 503（:936-944）——「typed」在这里被证明
    │
    ├─ 无图 → decide()  :966                有图 → decide_images()  :1028
    │   ├ render_state(state)  :98-104        compact JSON（state_compact: true）
    │   ├ truncate_state()     :950-959       超 max_state_tokens → 头 half + "\n...\n" + 尾 half
    │   ├ 每个问题 question_prompt()  :763-766
    │   │     option_lines()  :145-149   → "A) name: description" 逐行
    │   │     user_turn()     :152-159   → "record: …\nquestion: …\noptions:\n…"
    │   │     render_chat()   :162-168   → <|im_start|>system\n…<|im_end|>\n
    │   │                                  <|im_start|>user\n…<|im_end|>\n
    │   │                                  <|im_start|>assistant\n<think>\n\n</think>\n\n  + "Answer:"
    │   └ 每题一次 VLLM.label_logprobs()  :837-856
    │         POST /v1/completions {max_tokens: 1, temperature: 0.0,
    │              logprobs: len(chunk), logprob_token_ids: chunk(≤128), return_tokens_as_token_ids: true}
    │         ← choices[0].logprobs.top_logprobs[0] = {"token_id:114": -0.31, …}   只含这 K 个 id
    │
    └─ respond()  :1052-1070
         每题：probabilities() = softmax(logprob / T_type)     :638-649
               commit_noul()  = max(P,1−P) 低于 0.801 就抬到 0.801（不换答案）  :652-659
               confidence()   = (p_max − 1/k) / (1 − 1/k)     :630-635
        → {"model", "answers", "probability_source": "native", "effort_used",
           "usage": {"input_tokens", "output_tokens": 0, "latency_ms"}}
```

## §2 「typed」从哪来：255 个标签，靠加载期逐一证明

标签表在 `serve_config.json:16`，共 255 个（`max_options` 也是 255，:17）。它的构成值得抄下来，因为它解释了
为什么必须有校验：

- 前 52 个是 `A`–`Z` 与 `a`–`z`；
- 接着 25 个希腊字母 `α`–`ω`（**没有 `ρ`**——本文不推测原因，只记录该表如此）；
- 剩下 178 个是 BPE 片段：`in`、`er`、`on`、`re`、`at`、…、`the`、`ing`、`ion`，末尾是 `div`、`ell`。

`Contract.__init__` 只做三件与标签有关的事：必须是 list（:713-714）、截到 `max_options`（:715）、**互不重复**
（:716-717）。这三件都只是形状检查——**「每个标签在该读位置上恰好是一个 token」不在配置里，也不在分词器里，
而是在服务起来之后才被证明**：

```python
# h2o_lightning_shim.py:931-944（节选）
n = len(self.c.labels)
probe = {"type": "choice", "instructions": "Which?", "criteria": {f"o{i}": f"d{i}" for i in range(n)}}
text = self.c.question_prompt(render_state({"x": 1}, self.c.state_compact), probe)[0]
base = self.v.tokenize(text)
ids, bad = {}, []
for label in self.c.labels:
    full = self.v.tokenize(text + " " + label)
    if len(full) != len(base) + 1 or full[:len(base)] != base:
        bad.append(label)
    else:
        ids[label] = full[-1]
if bad or len(set(ids.values())) != len(ids):
    raise UpstreamError(... "is this serve_config.json the one for this model?" , 503)
```

四个细节决定了这个设计的性质：

1. **探测走的是真的那条 prompt**。探针题带着 255 个选项被 `question_prompt()` 渲染成完整 prompt（含 system、
   `record:`、`options:`、`Answer:`），再用 vLLM 的 `/tokenize` 分词。所以被证明的是「在**这个布局、这个
   tokenizer** 下是单 token」，不是抽象的单 token。
2. **判据是「恰好 +1」而不是「词表里以某字符开头」**。`len(full) == len(base) + 1 and full[:len(base)] == base`
   同时排除了两件事：标签被切成多 token，以及标签与前缀发生 BPE 合并（那样 `full[:len(base)]` 就不等于 `base`）。
3. **标签带前导空格**。探针拼的是 `text + " " + label`，所以**被读的是 `" A"`、`" in"` 这种空格开头的 token id**，
   而不是 `"A"`。这一点在推理期**原样保留**：`logprob_token_ids` 用的就是这里记下的 `label_ids`
   （`one()` 取 `self.c.label_ids[l]`，`h2o_lightning_shim.py:962`），因为 prompt 同样以 `Answer:` 结尾、
   模型续写的下一段同样以空格开始。
   （旁证：`ggml-org/llama.cpp` 的 PR #30361 为同一个模型加类型时，正文写的就是 *Our model answers with " A"
   rather than "A"*。该 PR 是第三方消费者，只作旁证，本文的结论不依赖它。）
4. **失败即 503**，不是 422、不是跳过这个标签。`check_backend` 抛的 `UpstreamError` 带 status 503
   （`h2o_lightning_shim.py:89-94`、:926-927、:943-944），`/health` 与 `/v1/systemone` 都把它原样转出去
   （:1108-1110、:1138-1139）。README 第 275–278 行把这件事写成了运维口径：*if not, it answers 503, so a
   misconfigured server stops a run instead of scoring it.*

于是与 decider／Nimble 两篇可以并读成一个谱：

| 实现 | 标签集怎么保证「单 token」 | 失败时 |
|---|---|---|
| decider | 训练期就在分词器里选定的 255 个标签行 | 概念上不成立的情形不存在（读出即 255 行） |
| Nimble | 训练期按位置分配，码表随 checkpoint 发布 | 加载期重算 `prompt_code_sha256` 不符即拒 |
| H2O-Lightning | **服务起来后用真实 prompt 逐标签探针** | 503，服务拒答 |

三者都把「封闭性」当成一个**必须被核对的断言**而不是一个假设；区别只在核对发生在训练、加载、还是首次请求。

## §3 读位置：`Answer:` 之后的那个位置

prompt 的构造是三段拼接，全在 `render_chat`（:162-168）：

```python
# h2o_lightning_shim.py:73-75
CHATML = {"system": "<|im_start|>system\n{content}<|im_end|>\n",
          "user": "<|im_start|>user\n{content}<|im_end|>\n",
          "generation": "<|im_start|>assistant\n<think>\n\n</think>\n\n"}

# h2o_lightning_shim.py:167-168
out = "" if system is None else CHATML["system"].replace("{content}", system.strip())
return out + CHATML["user"].replace("{content}", user.strip()) + CHATML["generation"] + prefill
```

`prefill` 来自 `serve_config.json:13`，值就是 `Answer:`。所以一条请求的 prompt 结尾恒为：

```
<|im_start|>assistant\n<think>\n\n</think>\n\nAnswer:
```

有三个点是要紧的：

1. **`<think>\n\n</think>\n\n` 是硬编码的、空的思考块**。它不是「让模型别思考」的参数，而是把闭区间**写进字符串**：
   模板里 assistant 轮的生成提示本来就长这样（`chat_template.jinja:149-150`，`enable_thinking is false` 分支）。
   `Contract.__init__` 直接拒绝反过来的配置：`prompt.enable_thinking` 为真就抛错（:705-706），
   理由是注释写的那句 *the answer is read after a closed think block*。
2. **读位置不是被算出来的，也不是被找出来的**。它等于「渲染完的 prompt 长度」——因为接着这一串再解码的**第一个**
   token 就是答案。代码里没有任何「第 N 个位置」的算术，`max_tokens: 1` 本身就把它钉住了（§4）。
3. **layout 是二选一的字符串开关**，不是模板：`plain`（默认）与 `markdown`（`user_turn`，:152-159）。
   `serve_config.json:11` 选了 `plain`，`Contract` 把别的值留给 `user_turn` 抛 `ValueError`。

这一句与 **decider** 那篇可以对照着读：decider 一次前向读 K 个 `Answer k: (` 槽位，读位置是「K 个已知子串的左括号」；
这里一次前向只读 **1 个**位置，而「多问题」被推到了**多次前向 + 前缀缓存**上（§7）。两者都在回答同一个问题
（「怎么让读位置可预测」），但一个用时序换空间、另一个用空间换时序。

## §4 一次前向：每题一次 1-token completion，只读 K 个 token id

`label_logprobs` 是唯一的读数入口（:837-856）：

```python
# h2o_lightning_shim.py:840-848（节选）
for k in range(0, len(ids), LOGPROB_CHUNK):
    chunk = ids[k:k + LOGPROB_CHUNK]
    r = self.post("/v1/completions", {
        "model": self.model, "prompt": prompt, "max_tokens": 1, "temperature": 0.0,
        "logprobs": len(chunk), "logprob_token_ids": chunk, "return_tokens_as_token_ids": True,
        "add_special_tokens": False})
    top = r["choices"][0]["logprobs"]["top_logprobs"][0]
    got = {int(key.split(":", 1)[1]): float(v) for key, v in top.items()}
```

六个参数各管一件事：

- `max_tokens: 1` —— 只要一个位置（§3 的读位置）。
- `temperature: 0.0` —— 采样温度**恒为 0**，与发布温度无关（发布温度在 shim 里施加，§9）。README `.114` 那句
  *the decisions read log-probabilities at temperature 0 and never sample* 与 `serve.sh:17-18` 的注释都指向同一件事：
  采样器根本不用，所以 FlashInfer 的采样 kernel 编不过也不影响服务。
- `logprob_token_ids: chunk` —— **这是本实现最关键的一步**：把候选的 token id 直接告诉 vLLM，
  回包里的 `top_logprobs[0]` 就只有这些 id。所以不存在「读出来再筛」这个环节，也就没有「候选落在 top-k 之外」这种失败模式
  （:10-11 的注释：*so every option is read at the answer slot and none can fall out of a top-k*）。
- `logprobs: len(chunk)` —— 要与请求的 id 数一致。
- `return_tokens_as_token_ids: True` —— 键是 `"token_id:114"` 而不是解码后的字符串，
  于是 `int(key.split(":", 1)[1])` 能无歧义地还原 id（:848）。
- `add_special_tokens: False` —— 与前面 `tokenize()` 探针的分词口径一致（:832）。

**分块**：vLLM 0.30 单请求最多接受 128 个 `logprob_token_ids`，所以 `LOGPROB_CHUNK` 默认 128
（:65-66，可用 `SHIM_LOGPROB_CHUNK` 改），255 个标签要发**两次** completion。`test_shim.py:241-246` 对这条路径有断言
（200 个选项 → 恰好 2 次 completion，且第 150 号选项被选中）。分块带来一个**交付性断言**：每个 chunk 的返回里
如果少了任何一个 id，就抛 `UpstreamError`（:851-853）——不允许「少读几个候选」。

**并发**：多问题时每题一个 `one()`，提交给 `ThreadPoolExecutor`（默认 64 线程，:912、:996-998）。
`test_shim.py:208` 断言 3 个问题 → 恰好 3 次 completion。注意这里**不是**「一次前向读 K 个槽位」：
题与题之间是各自独立的一次 1-token 前向，共享的只是 prompt 的前缀（§7）。

## §5 掩码 + softmax：没有掩码，只有 K 个 logprob 与一个温度

`probabilities`（:638-649）就是全部：

```python
# h2o_lightning_shim.py:643-649（节选）
v = [x / max(float(temperature), 1e-3) for x in logprobs]
m = max(v)
e = [math.exp(x - m) for x in v]
s = sum(e)
return [x / s for x in e]
```

三处与 RYOTIDE 不同：

1. **没有掩码这一步，因为词表从未出现在读数里**。RYOTIDE 的链条是「全词表 logits → softmax → 把标记外的质量掩掉 →
   重归一化」，`marker_mass`（落在标记上的份额）是它的核心诊断量。这里 `logprob_token_ids` 决定了回包**只有 K 个数**，
   所以「标签之外的质量」不是一个被丢弃的量，而是一个**从不被计算的量**：`marker_mass` **概念不适用**
   （与 decider、Nimble 同属一类，但机制不同——那两篇是「只投影候选行」，这里是「只请求候选 id」）。
   README 第 507 行仍把话说成概率口径：*The probabilities are calibrated for decision questions of this kind*。
2. **温度是显式的一次除法**，不是 vLLM 的采样温度。vLLM 收到的是 `temperature: 0.0`（§4），回的是原始 logprob；
   shim 再按题型的 T 除一遍（`format_answer` → `probabilities`，:670-672）。`temperature` 是**按题型**取的
   （`respond`，:1057：`t = temperature[typ] if isinstance(temperature, dict) else temperature`），
   值来自 `serve_config.json:5`：`choice 0.75 / noul 0.8 / score 0.65`。
3. **输入是 logprob 而非 logits，所以要防的不是溢出而是非法值**。`probabilities` 先校验整条向量是有限数
   （:640-642），归一化分母非正或非有限也抛（:647-648）——两处都是 `UpstreamError`（502 侧），不是 422。

**三种题型的答案成形**在 `format_answer`（:670-694）：

| 题型 | 选项数 | 返回字段 |
|---|---|---|
| `choice` | 请求的 `criteria` 键数 | `choice`（第一个最大值的名字，:688）、`probabilities`（按名字）、`confidence` |
| `noul` | 恒为 2，`["true","false"]`（`options_of`，:129） | 只有 `noul`（**没有 `probabilities`，也没有 `confidence`**）；带 `threshold` 时另给 `decision`（:676-679） |
| `score` | `criteria` 数组长度 | `score` = Σ i·pᵢ（:683）、`legend`、`probabilities`（按序号字符串）、`confidence` |

`noul` 那一行是三者里唯一**不出 `confidence`** 的：yes/no 的置信度就是它自己报的那个数（§9）。

`argmax` 取**第一个**最大值（:688 的注释 *the first maximum*）。这在 255 个标签下不是小事：`test_shim.py` 的
`MOCK_LP` 全是相同值时，谁赢取决于顺序。

## §6 顺序偏置：不存在该变量

选项的标签是**按位置分配**的：`self.labels[:len(names)]`（:759），第 i 个选项拿标签表里的第 i 个。
整份实现里没有任何置换、打散、逐选项对照的代码——本文核过全仓（`*.py` / `*.json` / `*.md` / `*.sh`，
排除题集本身）对 `permut` / `shuffle` / `order bias` / `reorder` 的搜索，**零命中**。

所以这里的结论不是「有偏置但没度量」，而是**「顺序偏置」这个变量在本实现里没有对象**：

- 它没有像 Kev、Nimble 那样在训练期打散选项（那是训练侧的事，本仓只有推理侧和一份发布配置，看不到训练代码）；
- 它没有像 SemIf-OpenJev 那样把选项反转一遍来度量；
- 它也没有像 Kev 那样留一个开关。

本文能核实的是**推理侧的行为**：给定选项顺序，标签就是它的下标。顺序偏置若存在，本仓既不测量也不补偿。
这是与 **Kev**（训练期打散 + 运行期度量工具在）和 **SemIf-OpenJev**（不消除只度量）都不同的第三种状态：
**工具与度量都不在**。代价是明确的——`serve_config.json` 里没有可以让部署者自己跑一次反转对照的入口
（对比 Kev 的 `/v1/systemone/permute`）。

## §7 多问题共享 state：前缀缓存 + 只计一次共享头

一次请求可以带多个问题（README 第 298 行：*One request can carry several questions about the same state*）。
实现上这是**同一个 state 被渲染进 N 条不同 prompt**，共享的部分交给 vLLM 的前缀缓存：

```python
# h2o_lightning_shim.py:995-1001（节选）
futs = [self.pool.submit(self.one, w[4], w[5]) for w in work]
toks = [self.pool.submit(self.v.tokenize, w[4]) for w in work]
reads = [f.result() for f in futs]
# COUNT THE SHARED HEAD ONCE: the system prompt and the record are computed once by vLLM's prefix
# cache, so N whole prompts would bill the record N times
shared = common_prefix_len([f.result() for f in toks])
```

`common_prefix_len`（:891-899）取若干 token id 列表的最长公共前缀；`respond` 再把它减回去：

```python
# h2o_lightning_shim.py:1059-1060
ntok += int(usage.get("prompt_tokens") or 0)
ntok -= shared * (len(work) - 1)   # = the shared head + each question's tail
```

三点值得记下：

1. **这是记账，不是省算**。前缀缓存本来就会让 vLLM 只算一次共享头；这段代码改的是 `usage.input_tokens`
   这个**上报数字**——否则 N 个问题会把同一份 record 计 N 次。README 第 323 行把口径写死在 API 上：
   *`usage.input_tokens` counts the prompt head that the questions share once.*
2. **答案不受影响**。`test_shim.py:214-233` 是这条不变量的直接断言：同一个 state 下，
   三个问题一起问 vs 逐个问，`answers` 的 JSON 必须逐字节相同（`sort_keys=True` 比较），
   同时 `input_tokens < Σ 单问`。
3. **图像路径要额外把图像 token 加回共享头**：`decide_images` 用返回的 `prompt_tokens` 减去文本 prompt 的长度，
   差额当作图像 token 并进 `shared`（:1047-1049）。因为图像被 vLLM 展开成多少个 token，是服务端决定的，
   客户端算不出来。

对比 **decider** 与 **SemIf-OpenJev** 那两篇的「共享 state 的 cache 复制策略」：那两篇是在**同一个引擎内**
决定 KV 怎么复用；这里把复用完全外包给 vLLM 的前缀缓存，shim 只负责**把账单改对**。

## §8 服务化 / fail-loud 风格

服务面是一个纯标准库的 `ThreadingHTTPServer`（:1148-1150，`request_queue_size = 512`，注释说明标准库默认的 5
会让并发客户端被 reset）。路由只有三条：

| 端点 | 方法 | 说明 |
|---|---|---|
| `/v1/systemone`（别名 `/api/alpha/decisions`） | POST | 主入口（:1118） |
| `/health` | GET | 会**先跑 `check_backend()`**，后端不健康时回 503 并把 `error` 塞进 body（:1106-1111） |
| `/v1/models` | GET | 回一个 id + `"readout": "native label log-probabilities over vLLM"`（:1112-1114） |

状态码的分工是这台实现最锋利的地方，因为它直接决定**评测跑不跑得下去**：

| 码 | 触发 | 代码 |
|---|---|---|
| 400 | 空问题集、未知 effort 档 | :973、:980-981 |
| 422 | 超过 255 个选项、未知题型、`criteria` 形状不对、超过上下文、状态里 16 进制解不开的 data URI | :122、:139-140、:826、:1129、:1131 |
| 502 | vLLM 挂了或连不上 | :829、:814 |
| 503 | **vLLM 服务的不是配置里的那个 model，或任一标签不是单 token** | :926-927、:943-944 |
| 401/403/429 | 原样透传 vLLM 的状态码 | :829 |
| 其他异常 | 兜底也是 422，**永不 500** | :1140-1143 |

`422` 与 `502/503` 的区分是有意的：README 第 374–381 行说明评测 runner 把 422 记成「答错一题」继续跑，
而 502/503 会让 runner 在连续三次失败后停下。所以「配置错了」必须是 503——**宁可让整轮跑不动，也不让它带着错的读数跑完**。
这正是 RYOTIDE 那条底线的运维版本。

`effort` 字段是一个**诚实的空实现**：`EFFORT_TIERS = ("fast", "balanced", "thorough")` 被接受并校验
（:64、:976-981），但只有 `fast` 真被实现；请求了别的档位不会报错，而是在响应里加一段自陈：

```python
# h2o_lightning_shim.py:1061-1065
used = {"tier": "fast", "passes_per_decision": 1}
if asked and asked != "fast":
    used["requested"] = asked
    used["note"] = (f"tier {asked!r} is accepted for API compatibility but not implemented; this response "
                    f"was produced at tier 'fast' with one forward pass per decision")
```

`passes_per_decision: 1` 与 `output_tokens: 0` 一起，把「一次前向」这条底线**写进了响应体**而不是留在文档里。

## §9 置信度从哪来，又怎么被改写

置信度是**读出来的**，不是训出来的；但读出来的值在到达调用方之前被改写了**两次**，而这两次都在 shim 里。

**第一次改写：温度。** vLLM 回的是原始 logprob，shim 除以题型的 T 再 softmax（§5）。所以同一个 logprob 向量，
`T=0.65` 与 `T=0.8` 给出不同的概率分布，也就可能给出不同的 argmax。发布温度在 `serve_config.json:5`，
可被 `SHIM_TEMPERATURE` 一次性覆盖所有题型（:720、:724-725）。

**第二次改写：yes/no 地板。** `commit_noul`（:652-659）：

```python
def commit_noul(p_yes, floor):
    """A yes/no probability with max(P, 1-P) under the floor moves to the floor on its own side (a tie goes to yes).
    The answer never changes; 0 turns the floor off."""
    if not floor:
        return p_yes
    if max(p_yes, 1.0 - p_yes) >= floor:
        return p_yes
    return floor if p_yes >= 0.5 else 1.0 - floor
```

`serve_config.json:6` 的值是 `0.801`。所以**发布出去的 yes/no 数字，下界是 0.801**——一个「几乎不确定」的
yes/no 问题会报 0.801 或 0.199，而不是 0.51。README 第 389–400 行把这件事说成部署选择：
基准跑用出厂配置（带地板），接 fallback 的路由场景才用 `SHIM_NOUL_FLOOR=0` 关掉。

**第三次改写：`confidence` 自己的公式。** `choice` 与 `score` 报的 `confidence` 不是 `p_max`，而是归一化最大值
（:630-635）：`(p_max − 1/k) / (1 − 1/k)`，k 是选项数。选项越多，同样的 `p_max` 给出越高的 `confidence`
（k=2 时 0.801 → 0.602；k=3 时 0.801 → 0.701）。注意它与 `probabilities` 是**两个不同的量**，
且 `noul` 的答案里没有这个字段。

### 「谁在改发布数值」

| 环节 | 位置 | 改了什么 | 会不会改 argmax |
|---|---|---|---|
| prompt 布局 + `Answer:` | `h2o_lightning_shim.py:162-168`、`serve_config.json:11-13` | 决定读哪个位置（读位置的**定义**） | 会（换了 prompt 就是另一个模型行为） |
| `max_tokens: 1` + `temperature: 0.0` | :843-844 | 只取一个位置的原始 logprob | 不会（不采样） |
| **`logprob_token_ids` 的分块** | :840-845 | 只把 K 个候选**带回来**（不改数值） | 不会 |
| **题型温度 T** | :643、`serve_config.json:5` | **改数值**（除 T 再 softmax） | **会**（多选项时可能翻） |
| **yes/no 地板 0.801** | :652-659、`serve_config.json:6` | **改数值**（抬到下界） | **不会**（:654 注释明写 *The answer never changes*；本文 §16 在 1,367 行上复算，翻转 0 次） |
| `confidence` 公式 | :630-635 | 只改 `confidence` 字段，不改 `probabilities` | 不会 |
| `argmax` 取第一个最大 | :688 | 平局时的确定性 | 会（平局时） |

与 **OpenJev(HF)** 那篇对照着看最清楚：那篇的 (1.829074, 0) 是**训练之后施加的常数**，而这里的
`{"choice": 0.75, "noul": 0.8, "score": 0.65}` 与 `noul_floor: 0.801` 是**一份随服务发布的配置**，
并且 `SHIM_TEMPERATURE` / `SHIM_NOUL_FLOOR` 两个环境变量可以在**不改配置**的前提下把它们换掉——
`/health` 会如实回报哪一个来源在生效（:1077-1078 的 `noul_floor.source`、`temperature_source`）。
这一点比多数实现干净：**改了数值的东西，自己会说是谁改的。**

## §10 第二条服务面：图像路径

`serve_config.json:18-22` 有一个 `image` 块（`max_images 4`、`max_pixels 1048576`、`temperature 0.65`）。
它一旦存在，`Contract.images` 为真，**带图的请求改走另一条路径**；不带图的请求**逐字节走原路径**
（`test_shim.py:381-396` 就是这条断言：同一个请求，图像部署与纯文本部署产生**同一份 prompt**）。

图像路径（`decide_images`，:1028-1050）与文本路径的差别只有四处，且每一处都是为了「让 prompt 仍是训练时的那个」：

1. **走 `/v1/chat/completions` 而不是 `/v1/completions`**，因为图像要走 chat 模板。
2. **assistant 轮用 `continue_final_message` 预填同一个 `Answer:`**（:180、:870-873）。于是答案槽位与文本路径
   **是同一个位置**，而不是「对话生成后的第一个 token」这种近似。
3. **首次图像请求前先做一次模板校验**（`check_chat`，:1004-1021）：用 vLLM 的 chat 模板渲染**不带图**的同一组
   messages，要求它与文本路径的 prompt **逐 token 相同**；不同就 503 并报出第一个分歧位置。
   这等于把「图像路径与文本路径只差图像 token」这件事也变成了加载期断言。
4. **图像温度独立**：图像路径整条用 `self.c.image_temperature`（:1050），`serve_config.json:21` 是 `0.65`，
   与文本三个题型都不同。README 第 335–336 行把它写成口径：*image answers use their own temperature, 0.65*。

图像从 `state` 里被「拎出来」时留下 `[image N]` 占位（`lift_images`，:547-569），所以 record 里可以引用它
（:28-29）。区分「是图像」与「只是长得像 base64 的普通文本」的判据在 `image_from_string`（:477-503）：
`state` 内部必须**至少 64 字符、无空白、且以已知图像头开头**（`_B64_HEADS`，:190）才当图像——
这条规则的存在是为了让一段普通文本不会被误当作图像而 422 掉（`test_shim.py:498-510`）。

## §11 图像 URL 抓取：一个自带的 SSRF 边界

这是本实现里最不「决策模型」、也最值得单独记一笔的部分：**vLLM 从不替调用方发请求**。
请求里的 `https` 图像链接由 shim 自己下载、自己转成 data URI 再交给 vLLM（:231-233 的注释，
以及 `_fetch_bytes` / `fetch_image_url`，:388-460）。规则写在 :234-251 的注释里，实现一一对应：

- **只允许 https**（`http` 要 `LIGHTNING_IMAGE_URL_ALLOW_HTTP=true` 显式打开），URL 里不许有 user:password，
  端口必须是 443（:331-335、:341-342）；
- **解析出的每一个地址都必须是公网**：loopback、private、link-local（含云元数据 169.254.0.0/16）、CGNAT 100.64.0.0/10、
  multicast、reserved、IPv6 ULA/link-local/site-local，以及各种 IPv4-mapped / -compatible / -translated 形式，
  任一命中即拒（:258-263、:272-282、:343-345）；
- **连接到被检查过的那个 IP**（`_PinnedHTTPS`，:352-363），同时 TLS 仍按原主机名验证书——这样不存在
  「检查时是公网、连接时被重绑」的 DNS rebinding 窗口；
- 最多 3 次跳转，每次重新走全部检查；最终必须 200 且 `Content-Type: image/*`，字节还得真的是 PNG/JPEG/GIF/WebP/BMP/TIFF
  （`sniff_image`，:193-207）；
- 整个抓取（含 DNS、连接、TLS、头、体、跳转）受一个硬截止时间约束，超时由一个 `threading.Timer` 直接
  shutdown socket（`_cut`，:377-385）；
- 请求里的链接**先计数再抓取**（:572-597 的 `_COLLECTING` + `_Link`），所以链接数超限的请求**一个字节都不抓**；
- 允许名单/拒绝名单/总开关（`LIGHTNING_IMAGE_URL_HOSTS`、`..._DENY_HOSTS`、`LIGHTNING_IMAGE_URLS=off`，:248-249）。

`test_security.py` 是为这套规则单独开的一份测试（10 个用例，覆盖内网/重绑拒绝、跳转、慢速滴流、
先计数后抓取、并发与总字节上限、IDN 与端口、关开关、以及「record 里的普通链接仍是文本」）。

**为什么值得记**：这是十三篇里第一条「模型服务自带的网络出口安全边界」。它不是决策读法的一部分，
但它改变了「一次决策要花多少钱」的口径——一个 URL 图像请求里，**shim 自己要去公网取一次图**，
而 README 第 515 行把 net-access 的自陈写成 *no network calls at serving time (except fetching an image URL a request supplies)*。

## §12 路由阈值：发布一张可复算的升级表

`routing_reproducibility/` 是随仓提交的一整套产物，它把「不确定的决策交给更强的模型或人」这件事变成一个
**可复算的表**：

| 文件 | 是什么 | 大小 |
|---|---|---|
| `predictions.jsonl` | 3,474 行逐题读数 | 1,666,994 B |
| `questions/plumb_s2k.jsonl` | 2,000 题（crh225/plumb-decisions） | 4,058,920 B |
| `questions/eikos.jsonl` | 1,474 题（caiovicentino1/eikos-decisions） | 7,433,871 B |
| `routing_thresholds.py` | 重算脚本（纯标准库，130 行） | 6,649 B |
| `SHA256SUMS` | 上面每个文件的 sha256 | 624 B |
| `ATTRIBUTION.md` | 两份题集的来源、许可与转换改动 | 1,728 B |

每行 `predictions.jsonl` 带 `raw_logprobs_t1`（该题标签的 logprob，T=1）、`served_probabilities`（出厂温度下
shim 的答案，地板关）、`gold` / `gold_indices`、`top_answer` / `top_probability` / `correct`，
以及 `dataset` / `license` / `source_id`。`routing_thresholds.py:50` 给出的重算公式与 shim 的
`probabilities()` **是同一个式子**：`softmax(raw_logprobs_t1 / T)`。

这张表的口径有两处必须写清楚，否则数字会被误用：

1. **它是「地板关」的数字**（README 第 234 行明说）。出厂配置下 yes/no 全被抬到 ≥0.801，
   于是 t=0.8 一档**一个都升不上去**——升级信号被地板抹平了。本文 §16 在 1,367 行 noul 上复算过：
   地板关时 t=0.8 升级 41.0 %，地板开时 **0.0 %**。
2. **两次独立 pass 的读数**。`raw_logprobs_t1` 与 `served_probabilities` 来自同一台服务器上的**两个不同 shim 进程**
   （一个 T 随出厂，一个 T=1；`routing_reproducibility/README.md:40-54`）。所以两者**不逐位相等**，
   差额就是「同一次前向的可复现性」的度量（§16 给出复算）。

## §13 四条不变量的落位

RYOTIDE 第 13 节那四条，在这条链上的状态：

| 不变量 | 状态 | 依据 |
|---|---|---|
| **一次前向** | **成立** | 每条 `max_tokens: 1` 的 completion 就是一次 1-token 前向；`passes_per_decision: 1` 写进响应体（:1061）。多问题=多次前向，不是一次多槽位 |
| **只读最后一个位置** | **需改写** | 读的不是「序列末位」而是「**渲染后 prompt 的下一位**」——读位置由 prompt 的最后一段字符串 `Answer:` 定义（:167-168）。在 vLLM 那一侧它就是末位，但把它叫「最后一个位置」会丢掉「位置由 prompt 决定」这件事 |
| **掩码到已知标签集** | **概念不适用（改写后成立）** | 没有掩码这一步：K 个标签 id 是**请求参数**（`logprob_token_ids`），只有它们被返回（§5）。等价的命题是「读数向量恰好是 K 个指定 token 的 logprob，长度恒等于选项数」 |
| **重归一化** | **成立** | `probabilities()` 在 K 个数上做 softmax，和为 1（:643-649）。但注意它同时**施加了温度**，所以它不是「把掩掉的质量补回来」的那种重归一化 |
| **`output_tokens = 0`** | **成立，且在响应组装处是字面量** | `h2o_lightning_shim.py:1067` 直接写 `"output_tokens": 0`；`test_shim.py:186` 有断言。图像路径也走同一个 `respond()`（:1050），所以两条路径都是 0 |

四条里两条要改写、一条概念不适用——这与 **CLM**（`output_tokens: 0` 是字面量、但没有前向也没有掩码）、
**decider**（`marker_mass` 概念不适用）是同一类：当实现把「读法」整体替换掉时，四条不变量里总有一半
需要翻译成等价命题才有意义。

## §14 文件对照表

| 文件 | 行数 | 职责 |
|---|---:|---|
| `h2o_lightning_shim.py` | 1,176 | 全部实现：prompt 构造、标签探针、vLLM 客户端、温度、地板、置信度、HTTP 服务、图像路径、URL 抓取 |
| `serve_config.json` | 23 | 发布配置：模型名、三个题型温度、`noul_floor`、prompt 布局、255 个标签、`max_options`、图像块 |
| `chat_template.jinja` | 153 | Qwen3.5 的 chat 模板（随权重仓发布；shim 只用来比对，不直接渲染——文本路径渲染的是 :73-75 的硬编码等价物） |
| `serve.sh` | 30 | `vllm serve` + shim 一起拉起；`VLLM_USE_FLASHINFER_SAMPLER=0` 的原因写在 :17-18 |
| `test_shim.py` | 514 | 无 GPU、无网络的测试：prompt 形状、温度、地板、分数、多问题共享、分块、截断、标签校验、状态码、图像路径 |
| `test_security.py` | 206 | 图像 URL 抓取的 SSRF 边界测试（10 例） |
| `config.json` | 109 | 模型配置；`head_dtype: "float32"` 出现在 :86 与 :109 两处（README 第 253 行说明它让 vLLM 用 fp32 算 logits） |
| `adapter/adapter_config.json` | 25 | 混合模式用的 LoRA：`r: 64`、`lora_alpha: 128`、12 个 target 模块（含 `in_proj_qkv` 等线性注意力层） |
| `routing_reproducibility/` | — | §12：3,474 行逐题读数 + 题集 + 重算脚本 + `SHA256SUMS` |

一份必须点明的**口径差**：README 第 104 行把发布温度概括成 *text temperature 0.8, image temperature 0.65*，
但 `serve_config.json:5` 里文本温度是**三个**（`choice 0.75 / noul 0.8 / score 0.65`）。
同一份 README 第 212 行在讲路由表时写对了（*text temperatures as shipped (choice 0.75, yes/no 0.8, score 0.65)*）。
所以「文本温度 0.8」只对 `noul` 成立——引用这句概括会得到错的常数。

## §15 亲手跑一遍

本机可复现的部分（不需要 GPU、不需要权重）：

```bash
# 只取代码与配置（不拉 9.1 GB 权重）
hf download h2oai/h2o-lightning-4b --include "*.py" "*.json" "*.sh" "*.jinja" "routing_reproducibility/*" --local-dir hl4b

# 1. 验证随仓发布的那份哈希清单（7 个文件）
cd hl4b/routing_reproducibility && shasum -a 256 -c SHA256SUMS

# 2. 用上游自己的脚本重算路由表（本文 §16 用的是自己的实现，不跑这个）
python3 routing_thresholds.py

# 3. 上游自带的测试（无 GPU、无网络）
python3 hl4b/test_shim.py && python3 hl4b/test_security.py
```

需要 GPU 与权重才能得到的东西（**本文不给任何自己的数值**）：

```bash
bash hl4b/serve.sh          # vLLM 0.30.0 + shim，端口 8000 / 8741
curl -s http://127.0.0.1:8741/v1/systemone -H 'Content-Type: application/json' -d '{…}'   # README 第 300-307 行
```

本文**没有跑**第 3 步（不会执行上游程序），也没有加载权重。所以要权重/GPU 才能得到的一切——
JevBench 205/231、ImageJevBench 8/8、图像验证集 87.4 %、以及 README 第 121-134 行那两张延迟表——
**本文一概不核**，只登记它们各自声明的条件（RTX PRO 4500 Blackwell 32 GB、vLLM 0.30.0、
`raw LoRA` 那几行是另一套权重表示与另一组超参）。

## §16 实例走查：在 committed 产物上的复算

四类检查，全部在 committed 产物上用本文自己的代码做，**不运行上游程序**。

### 16.1 随仓哈希清单（`routing_reproducibility/SHA256SUMS`）

```
ATTRIBUTION.md: OK                     questions/LICENSE-plumb-decisions-Apache-2.0.txt: OK
README.md: OK                          questions/eikos.jsonl: OK
predictions.jsonl: OK                  questions/plumb_s2k.jsonl: OK
routing_thresholds.py: OK
```

7/7 通过，说明这一整套产物是自洽的：题集、读数、重算脚本对得上清单里的 sha256。

### 16.2 内部计数

| 检查 | 结果 |
|---|---|
| `predictions.jsonl` 行数 | 3,474 |
| 按题型 | choice 1,854 · noul 1,367 · score 253，合计 3,474 |
| 两个题集文件的行数 | `plumb_s2k.jsonl` 2,000 · `eikos.jsonl` 1,474，合计 3,474 |
| `dataset` 字段 | 2,000 + 1,474，与上两项一致 |
| `served_temperature` 字段 | choice 全 0.75、noul 全 0.8、score 全 0.65——与 `serve_config.json:5` 一致 |
| `serve_config.json` 标签 | 255 个，`len(set(...)) = 255`（互异）；按 128 分块为 128 + 127 = 2 次 completion |

### 16.3 把路由表重算一遍（本文自己的脚本，不跑 `routing_thresholds.py`）

对每行取 `p = softmax(raw_logprobs_t1 / T_type)`，`top = argmax`，`correct = top ∈ gold_indices`：

| 题型 (n) | 阈值 | 升级 | 其余上的准确率 | 自信地错 |
|---|---|---|---|---|
| choice (1,854) | 0.8 | 34.1 % | 95.4 % | 3.0 % |
| choice | 0.9 | 47.5 % | 98.2 % | 1.0 % |
| yes/no (1,367) | 0.8 | 41.0 % | 96.0 % | 2.3 % |
| yes/no | 0.9 | 59.5 % | 99.1 % | 0.4 % |
| score (253) | 0.8 | 46.6 % | 96.3 % | 2.0 % |
| score | 0.9 | 60.1 % | 99.0 % | 0.4 % |
| all (3,474) | 0.8 | 37.7 % | 95.7 % | 2.7 % |
| all | 0.9 | 53.2 % | 98.5 % | 0.7 % |

不带阈值时：choice 82.7 % · yes/no 84.7 % · score 80.6 %。

**结论**：这张表与 `routing_reproducibility/README.md:62-73` 的 8 行**逐格相同**，
且与模型卡（README 第 215-226 行）在每一个格子上的差都不超过 0.5 个百分点；唯一 1 个百分点以上的格子是
`score @ 0.9`（卡片 98.0 % / 0.8 %，重算 99.0 % / 0.4 %），而这件事**上游自己已经写明并给出了原因**：
那一档只留下 101 题，其中恰好 1 题答错、且它的 top 概率是 **0.896**，卡片的读数里这一题落在 0.9 之上，
重读时落在 0.9 之下，于是 101 题里多/少一个正确答案就动 1 个百分点。
本文在自己的复算里**找到了那一行**：`top_p = 0.896`、`correct = false`、`train-23-000186-q1`（plumb）、
score 题型。上游的解释不是事后合理化，是可以在数据上指到具体某行的。

### 16.4 复算上游自陈的三个量

| 上游自陈 | 出处 | 本文复算 |
|---|---|---|
| `top_answer` 列 | `predictions.jsonl` | 与我的 argmax **0 处不符**（3,474 行） |
| `correct` 列 | 同上 | 与我的 `argmax ∈ gold_indices` **0 处不符** |
| `top_probability` 列 | 同上 | 149 行在 5e-7 容差外，最大差 **7.3e-07**——即该列是四舍五入到 6 位的同一个公式，不是另一个公式 |
| 「两次 pass 之间，逐题概率差额中位数 0.0005、99 分位 0.016」 | `routing_reproducibility/README.md:52-54` | 把口径读成**逐题取最大绝对差、再对 3,474 题取中位数/分位数**：中位数 **0.000500**、99 分位 **0.015646**——吻合。若改成把所有 (题, 选项) 对汇总再取分位数，会得到 0.000137 / 0.012249，与自陈不符；所以自陈用的是前者 |
| 「9 of 3,474 top answers differ」 | 同上 | **9 of 3,474**，完全吻合 |
| 「出厂配置下没有 yes/no 答案会被升级到 0.8 以下」 | README 第 234-236 行 | 对 1,367 行 noul 施加 `commit_noul(..., 0.801)` 后取 `max(P,1−P)`，最小值 **0.8010**，t=0.8 升级率 **0.0 %**（地板关时是 41.0 %）。且地板**从不改变 argmax**（复算翻转 0 次） |
| 卡片两个响应例里的 `confidence` | README 第 314-318、362-367 行 | 按 `(p_max − 1/k)/(1 − 1/k)` 在卡片给的（已四舍五入的）概率上复算：choice (0.956, k=3) → 0.934 ✓、score (0.880, k=3) → 0.820 ✓、图像 choice (0.958, k=4) → 0.944 ≈ 0.943 ✓、图像 score (0.937, k=3) → 0.906 ✓ |
| 卡片两个响应例里的 `score` | 同上 | `Σ i·pᵢ`：urgency 0·0.016+1·0.104+2·0.880 = **1.864** ✓、legible 0·0.011+1·0.052+2·0.937 = **1.926** ✓ |

**本文没有核、也无从核的**：`model.safetensors` 的 sha256（`routing_reproducibility/README.md:31-32` 声称是
`2d51561f…`）——该文件 9.1 GB，本文不下载；一切需要 GPU 的读数（§15 已列）。

## 与另外十二篇并读时的三个落点

1. **「typed」的证明位置**：decider 在训练期选定 255 个标签行，Nimble 在加载期重算 prompt 源码哈希，
   H2O-Lightning 把核对推到了**首次请求**（用真实 prompt 逐标签探针）——同一件事的三个时间点。
2. **`marker_mass` 的三次消失**：CLM（没有词表）、decider（读出头就是 255 行标签）、Nimble（只投影候选 K 行）、
   Kev（骨干丢掉 vocab head），现在是 H2O-Lightning（只请求候选 id）。四条路线、四种机制、同一个结论：
   **当读数向量一开始就只有 K 个数时，`marker_mass` 没有对象。**
3. **「谁在改发布数值」的第四种形态**：Ryotide 无、Jeff 训在权重里、OpenJev(HF) 是训练后的常数、
   Semantic Router 分档发布——H2O-Lightning 是**一份可被环境变量整体替换、且自己在 `/health` 里报告来源**的配置
   （§9）。区别不在于改不改，而在于**改了之后调用方能不能问出来**。

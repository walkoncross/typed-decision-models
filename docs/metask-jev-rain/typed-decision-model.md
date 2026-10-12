# metask-jev-rain-31B 是如何把一次决策做成一次 1-token 探针、再在「六种写法」上取边缘的

> 代码库：<https://huggingface.co/MeTask-Rain/metask-jev-rain-31B-trained>
> 对应 `service/README.md` 第 3–7 行：*A self-hosted decision service for JevBench. It serves `POST /v1/systemone`
> for the unchanged `typesafe` adapter of the official harness, and `GET /health`. The model has two parts that are
> loaded together and never merged …*；以及 `service/DISCLOSURE.md` 第 29–30 行：*Each decision is read in one
> prompt pass: the answer distribution comes from the next-token log-probabilities of the answer forms; the drafting
> path of the service is not used.*
> 本文按仓库根目录下的 `README.md`（= `service/MODEL_CARD.md`）、根目录的 `config.json`／`chat_template.jinja`／
> `tokenizer.json`，以及 `service/` 的十个 Python 文件与十来个 manifest／config 的实际内容逐环节说明这两句话的
> 实现方式。文中出现的文件名与行号（`core.py:266`、`backends.py:500`、`service/config.json:226` 等）均相对该仓库根目录；
> 不带 `service/` 前缀的 Python 文件名都在 `service/` 下。
>
> **同名配对文档**：RYOTIDE 的 [`docs/ryotide/typed-decision-model.md`](../ryotide/typed-decision-model.md)。
> §0 先讲清分歧，§10 把它的四条不变量逐条落位。
>
> **本文读的是哪一个 revision**：`da56acc0bb607cc214c184ad9f1e9231a0d63dee`（2026-10-11T17:19:06Z，43 个文件）。
> 这个 HF 模型仓**就是上游本体**，`service/` 只在这里发布过一份。注意 `service/README.md:53` 让读者按
> `--revision 9392335f5a17395ecd70049f85b1909d53500144` 下载**模型目录**：那个 revision 里只有 11 个模型文件、
> 没有 `service/`。本文核过（§13.2）：两个 revision 共有的 11 个模型文件的 blob oid 与 size 逐字节相同，
> 差别只在 `service/` 整棵树。

## 本文的事实来源

| 类别 | 具体是什么 | 在文中怎么标 |
|---|---|---|
| 代码引用 | `service/` 下 `core.py`(1197 行) / `backends.py`(691) / `bench.py`(795) / `launch.py`(456) / `quota.py`(454) / `pkg_check.py`(342) / `server.py`(218) / `selftest.py`(198) / `reconcile_usage.py`(156) / `attention.py`(37) / `model_setup.py`(14)，以及根目录的 `chat_template.jinja`(390) | `文件:行` |
| 随包发布的记录 | `service/` 下的 `MANIFEST.sha256`、`FREEZE.json`、`MODEL_MANIFEST.json`、`ADAPTER_MANIFEST.json`、`PIPELINE_PROVENANCE.json`、`SERVICE_LOCK.json`、`MEASURED_EVIDENCE.json`、`fit_report.json`、`calibration.json`、`config.json`、`requirements*.lock` | `文件:行`，并在 §13 上机复算 |
| 公式与常量 | 根目录 `README.md`、`service/DISCLOSURE.md`、`service/README.md` 里自陈的数字与条件 | `README:行` |
| **「实测」的边界** | 本文**不运行上游程序**：没起过服务、没加载过权重、没跑过 `pkg_check.py` / `selftest.py`。§13 的全部复算都只做两类事——(a) 对随包发布的文本文件取 sha256 并与包自己的记录对照，(b) 用本文自己的代码在**已发布产物**（`config.json`、`calibration.json`、`tokenizer.json`、`MEASURED_EVIDENCE.json`）上重算公式与算术。 | §13 标「本文复算」 |

**本文不给出任何前向产生的数值。** 概率、准确率、token 数、延迟全都只从包自己的记录里引，并标出处；§9 的
`p_cal` 与温度虽是发布值，但本文只做「它等于哪个式子算出来的数」这一步，不给任何一次真实前向的结果。

这个实现与前面几篇最大的不同，一眼看去是四条：

1. **它一次决策要花 **至少一个** 输出 token**，而且**如实上报**。服务把候选标签的 token id 当作
   `logprob_token_ids` 发给 vLLM（`backends.py:500-530`），引擎为此必须生成 1 个探针 token；候选多于 128 个时
   按 128 一档再发一次（`backends.py:506`）。`usage.output_tokens` 就是这个探针数，既不硬编码 0，也不是
   「题数」这类重新定义——它就是引擎真的生成了几个 token（§4）。
2. **「typed」不是一张标签表，也不是单个 token，而是「每个标签的一组写法」**。`choice` 的每个标签有 **6** 种
   文本写法（`A`、` A`、`a`、` a`、`A.`、`(A`，`core.py:258`），只留其中**恰好单 token** 的那些，再把同一标签的
   所有写法用 logsumexp 合成**一个**标签分数（`core.py:317-332`）。所以标签集仍是请求声明的字符串集合，
   但「答案落在哪几个 token 上」是运行期从分词器现取的（§2、§5）。
3. **掩码在这条链上不存在，而这不是省略，是被配置校验钉住的**。`validate_config` 有一张「neutral 白名单」，
   `suppress_tokens`／`bad_words_ids`／`logits_processors` 之类任何会改 logits 的开关都必须是中性值
   （`core.py:133-137`），vLLM 的 `logprobs_mode="raw_logprobs"` 与 `max_logprobs=-1` 又写死在引擎构建处且
   列入「不可被 `vllm_options` 覆盖」的保留名（`backends.py:387-396`、`422-428`）。于是候选 token 的 logprob 是
   **未掩码分布**上的真值，标签外的质量在数学上从来没有被减掉过（§5）。
4. **noul 的置信度不是 softmax，是一个常数**。`probabilities()` 对 `noul` 走一条完全不同的分支
   （`core.py:351-355`）：先取原始分数的 argmax，再把 `p_cal` 给它、`1 - p_cal` 给另一个。`p_cal` 在发布的
   `calibration.json:15` 里是 `0.9459459459459459`，而这个数正好是 `70/74`（§13.8）——noul 的置信度是一个
   **在 231 条公开题上拟合出来的单点常数，不带任何逐题信息**（§9）。

## §0 与 RYOTIDE 的分野

| 维度 | RYOTIDE | metask-jev-rain-31B |
|---|---|---|
| 权重从哪来 | 冻结的指令模型，零训练 | 官方 `gemma-4-31B-it-qat-w4a16-ct`（W4A16 compressed-tensors；模型卡的 `base_model` 字段声明的是家族基座 `google/gemma-4-31B-it`，权重这一档由 `DISCLOSURE.md:5-13` 与 `ADAPTER_MANIFEST.json` 的 `served_base` 钉住）**不合并**地挂一个自训 LoRA r16/α32（`DISCLOSURE.md:14-17`） |
| 标签集从哪来 | 请求里声明的标签，靠分词器保证单 token | 请求声明标签；每个标签展开成 2–12 种**写法**，运行期只留单 token 的那些（§2） |
| 读位置怎么定 | 运行期扫前缀、按标记质量标定 | prompt 的末位是 chat 模板吐出的「空思考通道」，读它后面那一位；运行期没有任何扫描与阈值（§3） |
| 一次前向读哪里 | 只读最后一个位置的一次 logits | 一次 prompt 前向 + **按候选数分档的 1-token 探针**；每次探针只取显式列出的 token id（§4） |
| 掩码 + 重归一化 | 掩码到标签集再在标签上重归一化，`marker_mass` 是它的诊断量 | **没有掩码**（配置层禁止）；重归一化发生在**标签**上：先按写法 logsumexp，再按温度 softmax 并加 1e-6 地板（§5） |
| 置信度从哪来 | 读出来的：分布是模型自己的 softmax，训练不参与 | **训出来的**：LoRA 参与产生 logits，之后还有一个逐题型温度与一个 noul 常数（§9） |
| 顺序偏置 | 一节专讲 | 服务侧**不存在该变量**；唯一的对照臂是维护者工具里的一个开关，且不随包发布结果（§6） |
| `output_tokens` | 0 | **真实探针数 ≥ 1**，字段与机制同时不成立（§10） |

一句话立场：**RYOTIDE 把「一次前向」压成一次调用；这一路把同一次决策拆成「一次 prompt 前向 + N 次 1-token
探针」，用多花的输出 token 换「标签集与写法的自由度」**——候选不必是单 token 字母，可以是任意字符串，
只要它至少有一种写法落在单个 token 上。

**同一个作者还发布了 12B 那一档**（本文走查的是 31B）：`MeTask-Rain/metask-jev-rain-12B-trained` 是同一套结构
换基座——根目录是 `google/gemma-4-12B-it-qat-q4_0-unquantized`（官方 QAT checkpoint，revision
`b6ed86275a6a5735884e208bfed95b445a684ca2`）的 **int4 W4A16 compressed-tensors（AWQ，group size 32）**，
`adapter/` 里是**不合并**的 LoRA（PEFT，rank 16 / alpha 32，q/k/v/o/gate/up/down，`adapter/adapter_model.safetensors`
262,373,216 B），模型卡 `base_model_relation: adapter`；`model.safetensors` 8,421,320,792 B，参数量 11,959,730,224。
两档的差别只在基座与量化档位，读法本身不变。

## §1 端到端数据流

```
POST /v1/systemone   {"state":…, "questions":{"decision":{"type":…,"instructions":…,"criteria":…}}}
       │
       │  server.py:77-130   只认这一个路由；先校验再推理；失败不产生替代答案
       ▼
normalize_request  (core.py:243-255)
       │   每题一个 task；payload["state"] 原样复制给每一题（§7）
       ▼
Service._one → _one_impl  (core.py:1003-1163)
       │
       ├── render(task)                                    core.py:266-281
       │       noul   → labels ["no","yes"]，12 条 yes 写法 / 12 条 no 写法   core.py:19-20
       │       score  → labels ["0"…"n-1"]，每条 3 种数字写法                  core.py:262-263
       │       choice → labels 来自请求（或 criteria 的键序），码 A..Z 再 AA..ZZ core.py:206-209
       │
       ├── prepare_task                                    core.py:551-609
       │       state 超过 6653 token 时按规则压缩（§7）
       │       渲染两份 prompt：
       │         fast_ids = prompt(system, text)                     ← 本次服务真正读的那份
       │         slow_ids = prompt(system, text + "\n" + cot 指令)    ← 本题型下永不使用
       │       prompt_ids：system 文本并进 **user 轮**，enable_thinking=False  backends.py:139-153
       │       chat 模板尾巴 = "<|turn>model\n<|channel>thought\n<|channel|>"   chat_template.jinja:381-386
       │
       ├── fast_readout                                    core.py:767-788
       │       candidate_ids：每种写法 encode 一次，只留单 token，保序保留重数  core.py:317-325
       │       backend.next_logits(ids, wanted)                              backends.py:500-530
       │             max_tokens=1 + logprob_token_ids=wanted + allowed_token_ids=wanted
       │             raw_logprobs（引擎构建时设死，config 不可覆盖）
       │             每 128 个候选 id 一档 = 一次引擎调用 = 恰好 1 个输出 token
       │       aggregate：每个标签 = 它全部写法 logprob 的 logsumexp          core.py:328-332
       │
       └── probabilities                                   core.py:350-362
               choice/score：softmax(标签分数 / T) → max(1e-6, ·) → 重归一化
                             T 来自 calibration.json 的 temperature_by_kind
               noul       ：不是 softmax——常量 p_cal 给原始 argmax，1-p_cal 给另一个
       ▼
{"answers":{"decision":{"type":…,"probabilities":{…},"noul":…|"choice":…}},
 "model":…, "usage":{input_tokens, output_tokens, …, alternatives:{engine,each_once,submitted},
                     questions:{…每题完整 detail，含 raw_label_scores…}}}
                                     core.py:1152-1197
```

## §2 「typed」从哪来：请求声明的标签 × 一组写法 × 单 token 过滤器

标签集**不在模型里也不在配置里**，它是每题从请求现取的（`core.py:212-240`）：

- `noul`：`labels = ["no","yes"]`（`core.py:272`）；
- `score`：`labels = [str(i) for i in range(len(criteria))]`（`core.py:275`）；
- `choice`：`labels = question["labels"]`；没有 `labels` 字段时退到 `list(criteria)`（字典取键序）
  （`core.py:231-237`）。码表是 A..Z 再 AA..ZZ，上限 **676**（`core.py:206-209`），超了报
  `UnsupportedRequest`（HTTP 422）。

真正定义「typed」的是**写法的展开与过滤**。`render` 给每个标签配一组写法：

```python
# core.py:258-263
def letter_forms(c):
    return [c, " " + c, c.lower(), " " + c.lower(), c + ".", "(" + c]

def digit_forms(d):
    return [d, " " + d, d + "."]
```

`noul` 更宽：`YES_FORMS` / `NO_FORMS` 各 12 条，把 `yes/Yes/YES/true/True/Y` 连前导空格与否都算上
（`core.py:19-20`）。然后：

```python
# core.py:317-325
def candidate_ids(backend, groups):
    # Keep multiplicity even when two textual forms map to the same token id.
    result = {}
    for label, forms in groups.items():
        tokens = [backend.encode(form) for form in forms]
        result[label] = [ids[0] for ids in tokens if len(ids) == 1]
    if not any(result.values()):
        raise ValueError("Tokenizer has no single-token candidate forms")
    return result
```

三件事值得逐条说清：

1. **过滤器是 `len(ids) == 1`，而且是逐写法的**。同一个标签的 6 种写法里，哪些是单 token 完全取决于分词器，
   代码不做假设也不在加载期探针——它是**请求期**过滤。
2. **重数是保留的**（第 318 行的注释）。两种不同写法若编码到同一个 token id，那个 id 会在这个标签的列表里
   出现两次，`logsumexp` 也就把它算两遍。这是「保真抄写上游 renderer」的一部分：`render` 的第一行注释写着
   *Source-faithful copy of cache_hidden_v2.render; do not deduplicate forms.*（`core.py:267`）。
3. **一个标签只要有一种单 token 写法就够了**；若**全部**标签都没有单 token 写法，`candidate_ids` 抛
   `ValueError`，`fast_readout` 接住它并改走逐 token 链式打分（`core.py:773-780`）——那条路把写法当
   token 序列走完，用 KV 分支只算增量（`core.py:612-688`）。

**写法在 `score` 题型上还有一处例外**：`is_prefix` 分支。当标签是数字且是另一个标签的前缀时（例如候选是
`1` 与 `10`），读 `1` 之后还要乘上「下一位不是数字」的概率，否则 `1` 会把 `10` 的质量也吃掉：

```python
# core.py:679-685
is_prefix=label.isdigit() and any(other!=label and other.startswith(label) for other in groups)
...
value=score(ids)
if is_prefix and form.strip().isdigit(): value+=termination(ids)
```

`termination` 就是 `non_digit_logprob`（`backends.py:626-634`、`677-681`）：把**全部数字起始 token** 的
概率求和再从 1 里减掉，取对数。这是本文在整条链上找到的**唯一一处真的要「看词表」的地方**——它必须在
未归一化的全词表分布上求补集质量，所以走的不是 `logprob_token_ids` 的显式取数，而是
`logprobs=-1` 的全词表探针（`backends.py:524`、`442-447`）。

**本文复算**（§13.10）：在随仓发布的 `tokenizer.json`（262144 个 entry，model.type 是 **BPE**，
normalizer 把空格换成 `▁`，pre_tokenizer 按空格切且与前一个合并）上做静态判定，字母 A–Z 的 6 种写法里
**恰好 4 种**（`A`、` A`、`a`、` a`）有对应的 vocab 条目、且构成它的 merge 对存在；`A.` 与 `(A` 在词表里
**根本不存在**，必拆成多 token。两位数码 `AA`…`ZZ` 同样是 4 种。数字则相反：只有裸数字 `0`…`9` 在词表里
（` 0`、`0.` 都不在），所以 `score` 的每个标签只有 **1** 个候选 id。`noul` 的 12+12 条写法**全部**是
非空 vocab 条目。

这直接决定了一次决策要花几个输出 token（§4）：26 个选项的 `choice` 是 26×4 = 104 个候选 id，落在 128 这一档里，
**一次引擎调用**；676 个选项是 2704 个候选 id，要 **22** 次。

## §3 读位置：由 chat 模板的尾巴定义，而 `enable_thinking=False` 是「开一个空的思考通道」

这条链上没有标定、没有扫描、没有候选位置，也**没有 `Answer:` 这类字面量标记**。读位置由两步决定。

**第一步：`system` 文本不是 system 消息，它被并进 user 轮。**

```python
# backends.py:139-153
def prompt_ids(self, system: str, text: str, enable_thinking: bool = False) -> list[int]:
    # The instructions and the context share one user turn, as in all measured runs.
    messages = [{"role": "user", "content": system + "\n\n" + text}]
    try:
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )
    except TypeError as exc:
        if "enable_thinking" not in str(exc):
            raise
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
    return self.encode(prompt)
```

`config.json` 的 `prompt.system` 是 `"You are a careful, decisive judge. Read the context, apply every rule,
definition, threshold, exception and date it contains, and answer the question. Reply with exactly one word and
nothing else."`（`service/config.json:233`）。它拼在 user 内容的最前面。因为 `messages[0]['role']` 是 `user`
而不是 `system`，模板第 190 行那个「先开一个 system 轮」的条件不成立，于是**模板里连 `<|turn>system` 都不会
出现**。

**第二步：`enable_thinking=False` 在这里的意思是「把思考通道开出来立刻关掉」。** 这不是「不加 think 块」，
模板在生成提示处显式吐一对开合标记：

```jinja
{# chat_template.jinja:381-386 #}
{%- if add_generation_prompt -%}
    {%- if ns.prev_message_type != 'tool_response' and ns.prev_message_type != 'tool_call' -%}
        {{- '<|turn>model\n' -}}
        {%- if not enable_thinking -%}
            {{- '<|channel>thought\n<channel|>' -}}
```

所以本次服务真正送进引擎的 prompt 尾巴是 `<|turn>model\n<|channel>thought\n<channel|>`，而**读的就是紧跟
`<channel|>` 之后的那一位**。模板第 156 行还有一个 `strip_thinking` 宏与第 239-242 行的「把历史
`reasoning` 渲染成 `<|channel>thought…<channel|>`」逻辑，都是多轮场景用的；本服务的请求只有一轮，用不上。

三条后果：

- **`fast_ids` 里既没有 chain-of-thought 指令，也没有 `ANSWER:` 标记。** 两份 prompt 是分开渲染的
  （`core.py:557`）：

  ```python
  return labels,groups,backend.prompt_ids(system,text,think),backend.prompt_ids(system,text+'\n'+instruction,think)
  ```

  第 3 个（`fast_ids`）只包含渲染出来的 `Context/Question/Options`；第 4 个（`slow_ids`）才多出
  `cot_instructions` 那段「先一步一步推理…最后输出一行 `ANSWER: <letter>`」。而 `_one_impl` 把 `fast_ids`
  交给快路径（`core.py:1040`、`1053`），`slow_ids` 只在 `decision.path == 'slow'` 时才会用
  （`core.py:1085`）——本题型的 `routing.mode` 是 `fast_only`（`service/config.json:262`），这个分支
  **永不进入**。

- **`answer_marker` 在快路径上完全不参与读数。** 它只出现在 draft 路径的 stop 串与解析里
  （`backends.py:541`、`547`）。`core.py` 里那两个「答案标记」工具函数——`parse_answer`
  （`core.py:284-301`，注释写 *Audit only*）与 `readout_prefix`（`core.py:304-308`）——**在整包里没有任何
  调用点**（本文 grep 过：只有定义，没有引用）。同理 `cot_instructions` 的 `.replace('ANSWER:', …)`
  （`core.py:555`）在 `answer_marker` 就是 `"ANSWER:"` 时是个恒等变换。这些是随包带过来的上游 draft
  侧零件，不参与本次服务的读数。

- **`prompt_ids` 有一个 `TypeError` 兜底**：分词器的 chat 模板不接受 `enable_thinking` 关键字时，
  退回到不传它（`backends.py:147-152`）。Gemma 4 的模板接受这个关键字，所以走的是第一条路。

## §4 一次前向：一次 prompt 前向 + 按候选数分档的 1-token 探针

vLLM 后端的读数入口是 `next_logits`，它是整条链上最该逐行看的一段：

```python
# backends.py:500-530（节选）
def next_logits(self, ids, candidate_ids, meter=None):
    candidates = _candidate_ids(candidate_ids)
    values = list(ids)
    scores = {}
    # v0.31 supports explicit gather from RAW logprobs (before masks).
    # Chunking preserves the same full-vocabulary normalizer for all IDs.
    for start in range(0, len(candidates), 128):
        wanted = candidates[start:start + 128]
        kwargs = dict(max_tokens=1, temperature=1.0, top_p=1.0, top_k=-1,
                      repetition_penalty=1.0, detokenize=False,
                      seed=self.generation['seed'])
        try:
            params = self.SamplingParams(**kwargs, allowed_token_ids=wanted,
                logprob_token_ids=wanted, logprobs=len(wanted))
        except TypeError:
            params = self.SamplingParams(**kwargs, logprobs=min(50, max(len(wanted), 1)))
        output = self._completion(values, params, meter)
        distribution = output.logprobs[0] if output.logprobs else None
        if len(output.token_ids) != 1:
            raise BackendError('Probe must produce exactly one token')
```

逐条读出来：

1. **`max_tokens=1`**：引擎必须走一次完整的前向并采一个 token，才可能返回那个位置的 logprobs。这就是
   「会生成 1 个 token」的字面来源，与服务文档里的 *read in one prompt pass (no drafting)*
   （`service/README.md:108`）不冲突——「一次 prompt pass」说的是不做草稿/推理，探针 token 是另一件事，
   而且被如实计费。
2. **`logprob_token_ids` + `allowed_token_ids` 都是显式候选表**。前者决定「要哪个 token 的 logprob」，
   后者只是采样约束（采出来的那个 token 从不被读）。所以回包里**只有 K 个 logprob**，词表维度从不出现。
3. **每 128 个候选一档，每档一次独立引擎调用**。`_candidate_ids` 去重（`backends.py:26-30`），但
   `candidate_ids` 已经把「同一 token 的多重写法」算成重复项，去重后实际档数由**不同 token id 个数**决定。
   分档的理由写在注释里：分块保证所有 id 用**同一个全词表归一化常数**——因为 raw logprob 是
   `logits - logsumexp(全词表)`，分块不影响这个常数。
4. **`len(output.token_ids) != 1` 直接抛 `BackendError`**。这不是「大概一行」，是硬断言。
5. **有一处「全词表抢救探针」**：若回包缺了任何一个被请求的 id，会再发一次 `probe_params`
   （`logprobs=-1` 的全词表探针），并在该题的 detail 里留一条 `candidate_full_vocab_recovery` 警告
   （`backends.py:520-528`）。这条路径是唯一会拿到全词表分布的地方。
6. **句柄级的计量**。`_completion` 每次调用都记账（`backends.py:488-496`）：

   ```python
   n=len(result[0].outputs[0].token_ids)
   meter['output']+=n;meter[kind]+=n
   cached=getattr(result[0],'num_cached_tokens',None)
   if cached is None: meter['cached_unknown']=True
   else: meter['cached']+=int(cached)
   meter['engine_calls']=meter.get('engine_calls',0)+1
   meter['engine_generated']=meter.get('engine_generated',0)+n
   meter['engine_input']=meter.get('engine_input',0)+len(values)-int(cached or 0)
   ```

   也就是说 `output_tokens` 是**实测的生成数**（探针数），`input_tokens` 是**提交数减前缀缓存命中数**。

**同一条读法在 HF 后端上是 0 个 token。** `TransformersBackend.readout_output_tokens = 0`
（`backends.py:159`）而 `VLLMBackend.readout_output_tokens = 1`（`:361`）：HF 后端用一次
`model(..., use_cache=False, logits_to_keep=1)` 直接取末位 logits（`backends.py:304-319`），不生成任何
token，`fast_readout` 于是加 0（`core.py:785-787`）。本次服务跑的是 vLLM（`service/config.json:90`），
所以走 1-token 那一支。

**启动热身会碰慢路径一次。** `Service.warmup`（`core.py:982-1001`）先做一次探针、一次
`generate(ids, 1)`，然后**如果后端有 `slow_session` 就再跑一次 draft**，而且用 `force_answer` 策略保证
它一定产生 `ANSWER:`（`max_new_tokens = server.warmup_slow_tokens`，配的是 2，`service/config.json:294`）。
这不是为了答案，是为了让 `/health` 的 `supports_slow_session` 为真——而 `launch.validate_health` **要求它
为真**（`launch.py:281`）。也就是说：**一个永不使用 draft 路径的部署，仍然必须能把 draft 路径建起来**。

## §5 掩码 + softmax：没有掩码，重归一化的对象是标签

### 5.1 掩码不存在，而且是双层钉住的

第一层在配置校验：任何会改 logits 的解码开关都必须是中性值。

```python
# core.py:133-137
if m['backend']=='vllm':
    neutral={'suppress_tokens':None,'begin_suppress_tokens':None,'bad_words_ids':None,
        'forced_bos_token_id':None,'forced_eos_token_id':None,'forced_decoder_ids':None,
        'no_repeat_ngram_size':0,'min_new_tokens':0,'min_length':0,'length_penalty':1.0,'renormalize_logits':False}
    if any(g.get(k,v)!=v for k,v in neutral.items()):raise ValueError('Unsupported vLLM decode override')
```

第二层在引擎构建处：读数契约相关的参数被列入保留名，`vllm_options` 里出现即拒。

```python
# backends.py:387-396（节选）
reserved = {
    "model", "tokenizer", "dtype", "quantization", "quantization_config",
    "max_logprobs", "logprobs_mode", "seed", "enable_lora",
    "generation_config", "override_generation_config", "logits_processors",
    ...
}
overlap = sorted(reserved.intersection(options))
if overlap:
    raise ValueError("model.vllm_options cannot override readout contract: " + ", ".join(overlap))
```

而 `LLM(...)` 里写死的是 `max_logprobs=-1, logprobs_mode="raw_logprobs"`（`backends.py:425`）。这三件事合起来
的含义是：**读的是采样前的原始分布，而不是被 `allowed_token_ids` 处理过的分布**。模块开头的注释把不变量
说得很清楚（`backends.py:1-6`）：

```
Both backends return unmasked next-token scores. vLLM returns raw log
probabilities, which differ from logits by one common additive constant; the
form-set logsumexp and subsequent label temperature are invariant to it.
```

于是 RYOTIDE 的 `marker_mass` 在这里**没有对象**：没有「被掩掉的质量」，因为从来没有掩过；也没有
「标签外质量的占比」这个量，因为压根没有把词表归一化到标签上。唯一一处真的用了补集质量的地方是 §2 末尾
那个数字前缀终止项（`backends.py:626-634`），它算的是「下一位不是数字」的概率，不是「不是标签」的概率。

### 5.2 标签分数 = 写法上的 logsumexp

```python
# core.py:328-332
def aggregate(logits, groups):
    for token in {t for ids in groups.values() for t in ids}:
        if token not in logits or not math.isfinite(logits[token]):
            raise RuntimeError("Backend omitted a candidate or returned a non-finite logit")
    return {label: logsumexp([logits[t] for t in ids]) if ids else -1e9 for label, ids in groups.items()}
```

两点：**候选缺一个就抛**（不静默当 0），以及**没有候选写法的标签得到 -1e9 而不是 -inf**
（`-1e9` 在后面的温度除法里仍可参与，`-inf` 会污染 `exp` 的差）。

### 5.3 标签上的 softmax：逐题型温度 + 1e-6 地板 + 重归一化

```python
# core.py:350-362
def probabilities(labels, logits, qtype, block):
    if qtype == "noul":
        # Deterministic tie: earliest in render's label order, namely no.
        winner = max(labels, key=logits.__getitem__)
        p_cal = min(1 - 1e-6, max(0.5, block["noul_calibration"]["p_cal"]))
        return {label: p_cal if label == winner else 1 - p_cal for label in labels}
    temperature = block["temperature_by_kind"][qtype]
    # Subtract before dividing to avoid overflow from small positive T.
    top = max(logits.values())
    weights = {label: math.exp((logits[label] - top) / temperature) for label in labels}
    total = sum(weights.values())
    floored = {label: max(1e-6, weights[label] / total) for label in labels}
    return {label: value / sum(floored.values()) for label, value in floored.items()}
```

把这个函数与 §5.1 的「无掩码」合起来看，就是「掩码 + softmax」这四个字在它身上的实际形态：

- **掩码 → 换成 gather**：不是把标签外的 logits 置 `-inf`，而是只问引擎要标签那几个 token 的 logprob；
- **softmax → 分两层**：内层是同一标签多种写法的 `logsumexp`，外层才是标签之间的 softmax；
- **重归一化 → 两次**：一次是温度 softmax 本身（对标签归一），一次是 1e-6 地板之后的重归一化
  （`core.py:362`）。地板意味着**任何标签的概率都严格大于 0**，且地板之后总和重新拉回 1。

两个边界条件写得很克制：`top` 先减去再除温度（避免小温度上溢），温度为正数由
`validate_calibration` 保证（`core.py:343`）。`noul` 那一条则完全不走 softmax——见 §9。

**`allow_nan=False` 是最后一道闸**：响应在序列化时禁止非有限数（`server.py:51`、`:117`），
`_one` 里还专门先 `json.dumps` 探一次，失败就把它算成一次 **item failure**（`core.py:1013-1016`）。

## §6 顺序偏置：服务里不存在该变量

**服务侧一整套都不存在。** 全仓对 `shuffl`／`permut`／`reorder` 零命中（本文 grep 过）。
`render` 按请求给的顺序打印选项：

```python
# core.py:278-280
labels = list(task["labels"]); rubric = crit if isinstance(crit, dict) else {}; codes = choice_codes(len(labels))
groups = {l: letter_forms(c) for c, l in zip(codes, labels)}
parts = [cq, "Options:\n" + "\n".join(c + ". " + str(l) + ((": " + str(rubric[l])) if rubric.get(l) else "") for c, l in zip(codes, labels)), "Answer with the letter of the best option."]
```

选项顺序就是请求里 `labels` 的顺序（没有 `labels` 时是 `criteria` 的键序）。码 A、B、C… 按位置分配，不是按
标签内容排序。

**唯一一个对照臂在维护者工具里，而且分两步才看得出它做了什么。**

```python
# bench.py:151-157
def request_payload(task, original_types=False, model=None, label_order=False):
    question=task['question']
    q={'type':question['type'],'instructions':question['instructions']}
    if question.get('criteria') is not None: q['criteria']=question['criteria']
    if original_types:q['type']=question.get('request_type',question['type'])
    if label_order and task['type']=='choice': q['labels']=list(task['labels'])
    return {'state':task['state'],'model':model,'questions':{QUESTION_NAME:q}}
```

默认（也就是**被测量的那一档**）payload 里**不带 `labels`**，服务于是退到 `criteria` 的键序；
`--compare-label-order` 打开时（`bench.py:695`、`756-759`）才显式带上 `labels`，服务于是用 `labels` 的顺序
（`core.py:231-233` 的优先级）。比较的是两者的准确率差：

```python
# bench.py:547-548
a=sum(left[i]['correct'] for i in ids)/len(ids);b=sum(right[i]['correct'] for i in ids)/len(ids)
return {'count':len(ids),'criteria_accuracy':a,'labels_accuracy':b,'labels_minus_criteria':b-a}
```

所以它**不是**把选项反转或打乱，而是「两种由题集自己提供的顺序，各跑一遍，看准确率差多少」。它需要把整轮
评测再跑一次，产物写在 `--output-prefix` 下（`bench.py:760-761`），**不随包发布**——包里没有任何一份
顺序对照的结果。

**而且发布的那份校准只能来自 criteria 顺序的那一轮。** 拟合温度的入口有一条硬门：

```python
# bench.py:551-555
def calibrate_temperature(rows):
    # Only replay raw scores from the official request order; never calibrate
    # already floored/temperature-scaled probabilities.
    if not rows or any(not r['ok'] or r.get('request_order')!='criteria' or not isinstance(r.get('raw_label_scores'),dict) or r.get('expected') is None for r in rows):
        raise ValueError('Temperature fit needs successful labeled criteria-order raw scores')
```

每行都记了 `request_order`（`bench.py:280`），`fit_calibration` 也再查一遍（`bench.py:582`）。

**顺带一处两条规则的 argmax。** 服务报的 `detail['prediction']` 是请求顺序里的第一个最大值
（`core.py:1152`，`max(labels, key=probs.get)`，`labels` 是请求顺序）；而 bench 的评分函数**不读**
服务报的 `choice`，它从 `probabilities` 重算，用的规则是字典序最小者胜：

```python
# bench.py:201-202
    # Match official score_task: lexicographically smallest label wins ties.
    return max(sorted(task["labels"]), key=lambda label: probabilities[label]), probabilities
```

同一个包里，同一件事有两个 tie-break 规则。只有在概率严格相等时才会分叉，且分叉只影响 bench 记的
`pred`（不读服务那一侧），但并读时值得知道。

## §7 多问题共享 state：不做显式共享，靠前缀缓存与一条压缩规则

`normalize_request` 把请求里的 `state` **原样复制**给每一题（`core.py:254`），然后逐题处理：

```python
# core.py:250-255（节选）
for name, question in questions.items():
    ...
    tasks.append((name, {"state": payload["state"], "question": q, "labels": labels}))
```

**没有跨题的 KV 复用、没有前缀切分、没有块边界对齐、没有 state 边界注册**——这些在别的实现里是显式的
机件，在这里全都不存在。N 题的请求就是 N 次独立的 `_one`（`core.py:1170-1171`），每次各自走一次完整
读数。唯一让第 2..N 题便宜的机制是 vLLM 的自动前缀缓存（`enable_prefix_caching: true`，
`service/config.json:94`），而它能不能命中取决于渲染出来的 prompt 前缀是否逐 token 相同：
`Context:\n<state>\n\nQuestion: ` 这一段在题与题之间是共享的，`Question:` 之后才分叉。计量上这件事是
看得见的——`input_tokens` 取的是「提交数减缓存命中数」（`backends.py:496`），
`prompt_tokens_details.cached_tokens` 单独上报（`core.py:1064`），而 `usage.alternatives` 里
`submitted` 那一档就是**不扣缓存**的提交数（`core.py:1143`）。也就是说「共享了多少」这件事在这条链上
是**由读者自己从两个口径相减得到的**，服务不额外做任何 bookkeeping。

`reconcile_usage.py` 把这件事做成了一个可执行恒等式（E1/E2）：

```python
# reconcile_usage.py:101-102
    if prefill is not None and not close(S['input'], prefill, tol):            # identity E1
        fails.append('E1: sum(usage.input_tokens) != engine prefill')
...
# reconcile_usage.py:122-125
    diff = sum(off[i]['input'] - on[i]['input'] for i in matched)
    cached = sum(on[i]['cached'] for i in matched)
    if not close(diff, cached, tol):                                             # identity E2
        fails.append('E2: sum(input_off) - sum(input_on) != sum(cached_on)')
```

E2 是「开缓存与关缓存两轮，输入差额应当等于缓存命中量」——这是一条**为共享前缀写的审计规则**，
而它要跑两次评测才成立（`--run` 与 `--run-off`），并且只有 `--run-off` 存在时报告里的
`submission_ready` 才为真（`reconcile_usage.py:148`）。

**长输入不拒答，只压缩。** 输入超过 `max_prompt_tokens = 6653` 时走 `shrink_state`
（`core.py:388-548`）：按字符数预压缩（`precompression_trigger_chars_per_token = 8`）、按 token 预算抽块、
**保护显式规则子树**（`_RULE` / `_RULE_KEY` / `contains_rule_subtree`，`core.py:367-375`）、用查询词的
词面重叠给块排序、保留首尾、用 `[... state middle omitted ...]` 标出省略（`core.py:378`）。最后还有一道
物理上限：`physical_max_prompt_tokens` 也是 6653，超出时从首尾截断并记 `physical_truncated`
（`core.py:596-609`）。`selftest.py:142-162` 用合成分词器验证了 80,000 词与 7,000 词两种输入都能压到
6653 以内、且 `physical_truncated` 与 `protected_content_exceeds_budget` 都是 False。

## §8 服务化 / fail-loud 风格

**路由只有两条半**（`server.py:58-130`）：`GET /health`、`GET /engine_stats`（注释写明
*not used by evaluation*，`server.py:71`）、`POST /v1/systemone`。其余 404。

**错误码分三档，而且都不产生替代答案**：

| 情况 | 码 | 出处 |
|---|---|---|
| 结构性问题（非法 JSON、超长 body、`Transfer-Encoding`） | 400 | `server.py:110-113` |
| 结构合法但超出支持域（题目类型不支持、选项数超 676、分数档位为空） | 422 | `server.py:106-109` |
| 单题推理失败 | 422 | `server.py:118-121` |
| 引擎级失败、或连续第 3 题失败 | 503 | `server.py:122-129`，`core.py:34`、`945-950` |

`ITEM_FAILURE_LIMIT = 3` 的注释把理由写清了：*The third consecutive item failure is treated as service-level
(HTTP 503, service marked failed), so the harness stop rule for consecutive infrastructure errors applies*
（`core.py:32-34`）。一旦判死，`/v1/systemone` 与 `Expect: 100-continue` 都不再读请求体，直接 503
（`server.py:26-29`、`80-83`）。引擎死没死的判定靠**类名匹配**而不是 import vLLM：
`unrecoverable_engine_error`（`core.py:37-57`）在 `__cause__`/`__context__` 图里找 `EngineDeadError`，
或者任何非 `EngineGenerateError` 的 `vllm.*` 的 `VLLMServerError` 子类。

**启动阶段的 fail-loud 更硬**（`launch.py`）：

- `check_preparation` 要求 `package.nonformal_fitted` 为真、`package.finalized` 为真、`FREEZE.json` 在、
  `MODEL_MANIFEST.revision` 是完整的 40 位十六进制、`repo_id` 不是 `OWNER/`
  （`launch.py:62-78`）；
- `check_versions` 逐包比对 `vllm 0.31.0 / torch 2.13.0 / transformers 5.17.0 / tokenizers 0.23.2 /
  triton 3.7.1 / compressed-tensors 0.17.0`，不符就拒（`launch.py:33-34`、`140-149`）；
- `check_model` 对根目录文件逐个 sha256、对权重 shard 只比 size（`full=True` 时才比全哈希），
  并**拒收任何加载器可能读到的多余文件**（`launch.py:192-213`）；
- `check_adapter` 要求 `adapter/` 里的文件名集合**恰好**是两个 pin 过的文件、且 size 与 sha256 都对
  （`launch.py:81-97`）；
- `check_environment` 拒绝任何落在 `--state-dir` 之外的继承缓存路径变量（`XDG_*`、`HF_HOME`、`TRITON_*`、
  `TORCH_HOME` 等 20 个，`launch.py:55-58`、`246-257`）；
- **`validate_health` 要求 `/health` 的 `warnings` 为空**才算就绪

  ```python
  # launch.py:279-282
  def validate_health(health, no_cache=False):
      bad=health.get('warnings',[])
      if bad or health.get('status','ok')!='ok' or health.get('prefix_caching') is not (not no_cache) or health.get('supports_slow_session') is not True:
          raise RuntimeError('Service readiness rejected: health warnings or cache/session capability; health='+json.dumps(health)[:1500])
  ```

- 注意力后端两段式启动：先让 vLLM 自选，只有引擎确实起不来才第二次强制 `JEV_ATTENTION_BACKEND=TRITON_ATTN`
  （`launch.py:51`、`317-382`）。注释解释了原因：这个 checkpoint 混用 256 与 512 两种 head size，
  FlashAttention 4 不可用的卡上 vLLM 0.31 会把 auto 解析成 TRITON_ATTN（`launch.py:48-50`）。

**离线与只写 state 目录**：`child_environment` 设 `HF_HUB_OFFLINE=1`、`TRANSFORMERS_OFFLINE=1`、
`VLLM_BATCH_INVARIANT=1`、`VLLM_WORKER_MULTIPROC_METHOD=spawn`、`VLLM_USE_FLASHINFER_SAMPLER=0`，
并把 `HOME`／`TMPDIR` 改到 state 目录下（`launch.py:260-276`）。`pkg_check.py:286-289` 会拒绝运行期代码里
出现 `rmtree(` 或 `snapshot_download(`/`hf_hub_download(`，`pkg_check.py:298-300` 拒绝 shell 里的递归删除。
`prepare_model.sh` 是**唯一**会联网的东西，且文件头写死 *NETWORKED maintainer preparation, never called by
serve.sh*（`prepare_model.sh:2`）。

**「冻结」与「发布授权」是两件事，而且被做成互斥**。包内各组件的状态是：

| 记录 | 状态字段 | 值 |
|---|---|---|
| `FREEZE.json:66` | `status` | `FROZEN` |
| `MEASURED_EVIDENCE.json:28,31,45` | `frozen` / `publication_authorized` / `signed` | `false` / `false` / `false` |
| `ADAPTER_MANIFEST.json:112` | `publication_authorized` | `false` |
| `PIPELINE_PROVENANCE.json:22,26,49` | `official_submission_authorized` / `release_ready` / `status` | `false` / `false` / `PREPARATION_NOT_SIGNED` |
| `SERVICE_LOCK.json:24` | `status` | `prepared_not_frozen` |

而 `pkg_check.py:282-283` 把这件事写成了一条断言——**冻结成立时，那些授权标志必须仍然是 false**：

```python
if (c['package']['finalized'] and frozen is None) or prov.get('release_ready') or prov.get('official_submission_authorized'):
    raise ValueError('Only the recorded freeze may finalize; the builder never authorizes a release')
```

`pkg_check.py --release` 永远退出 2（`pkg_check.py:340`），`release_problems()` 的最后两条恒为「冷启动确认」
和「发布与提交授权（由仓库所有者决定）」（`pkg_check.py:326-329`）。换句话说：**这个工具能证明「包是自洽
且被冻结的」，在设计上不能证明「它可以被发布」。**

## §9 置信度从哪来，又怎么被改写

### 9.1 链路上的每一环

```
LoRA r16/α32（训练）                            DISCLOSURE.md:14-15
   └─ 其中 10 个 full-attention 层的 v_proj LoRA = k_proj LoRA 的逐字节副本（KVFILL）  ADAPTER_MANIFEST.json:24-62
        ↓
W4A16 官方基座 + 不合并适配器，一次 prompt 前向 + 1-token 探针
        ↓
候选 token 的 raw logprob（未掩码）              backends.py:425、500-530
        ↓
同一标签的多种写法 logsumexp → 标签分数          core.py:328-332
        ↓
choice/score：softmax(·/ T)，T = temperature_by_kind[题型]     core.py:356-362
noul       ：不是 softmax——常量 p_cal 给了原始 argmax          core.py:351-355
        ↓
每标签 max(1e-6, ·) → 重归一化                  core.py:361-362
```

### 9.2 三个旋钮里有两个随包发布，第三个藏在代码里

`calibration.json` 给 `fast` 与 `fallback` 两块同一组数（本文 §13.12 核过两块逐字段相同），
`slow` 块另有一组。发布值是：

| 量 | 值 | 出处 | 本文复算 |
|---|---|---|---|
| `choice` 温度 | `1.2968395546510096` | `calibration.json:10`、`:41` | `= 2**(3/8)`（§13.7） |
| `score` 温度 | `1.189207115002721` | `calibration.json:21`、`:52` | `= 2**(2/8)`（§13.7） |
| `noul` `p_cal` | `0.9459459459459459` | `calibration.json:15`、`:56` | `= 70/74`，74 正是 noul 的题数（§13.8） |
| 每标签地板 | `1e-6` | `core.py:361` | **硬编码，不在 calibration 里** |

两个温度落在 `bench.py:557` 那条网格上：

```python
grid=[2**(i/8) for i in range(-32,41)]
```

也就是 2⁻⁴ 到 2⁵、每 1/8 倍频一个格点；拟合就是在这条网格上取使 NLL 最小的点（`bench.py:564-570`）。
`p_cal` 的公式是 argmax 正确率（`bench.py:561-563`），并把结果夹到 `[0.5, 1-1e-6]`。

### 9.3 noul 的置信度是一个常数，而且发布文档里只有一句「拟合」

对 `noul` 题型，`probabilities()` 完全不看分数的**间隔**，只看谁最大（`core.py:352-355`）。发布值是
`p_cal = 0.9459459459459459`，`1 - p_cal = 0.05405405405405406`。所以：

- 一条 noul 题的响应里，`probabilities` **只可能是这两对数中的一个**（`{"no":p,"yes":1-p}` 或反过来）；
- 它对「这题有多难」零信息——`p_cal` 是按题型给的常数，不是按题给的；
- 平票时归 `no`，注释把这条规则写明了：*Deterministic tie: earliest in render's label order, namely no*
  （`core.py:352`），而 `render` 里 `labels = ["no","yes"]`（`core.py:272`）。

`validate_calibration` 用一条断言把这个形状钉死——`noul` 的温度字段**必须是字符串 `"one_bin"`**
（`core.py:344-345`），不接受数值温度。也就是说「noul 走的是分箱常数而不是温度缩放」是校验器层面的
不变量，不是实现的偶然。

### 9.4 谁在改发布数值

| 谁 | 改什么 | 改的是数值还是答案 | 在哪 |
|---|---|---|---|
| LoRA 训练 | 全部 logits | 两者都可能 | 不在包内，只由 `ADAPTER_MANIFEST.json:120-127` 记录来源哈希与 `checkpoint_step: 150` |
| KVFILL（10 层 v_proj = k_proj 副本） | 这 10 层的 value 通路 | 两者都可能 | `ADAPTER_MANIFEST.json:24-62`，`added_count: 20` |
| 逐题型温度 `choice` / `score` | 概率分布 | 只改数值（softmax 单调） | `calibration.json:29-33` |
| noul 的 `p_cal` | 概率分布 | **只改数值**：argmax 取自原始分数，`p_cal` 只是给它一个大小 | `core.py:353` |
| 每标签 1e-6 地板 + 重归一化 | 概率分布 | 只改数值 | `core.py:361-362` |
| `generation.temperature = 1.0` | **不参与读数** | 无 | `service/config.json:40`，只被 draft 路径读（`core.py:721`、`backends.py:538`） |

**最后一行的同名不同物值得单独说**：`generation.temperature`（1.0）是**采样**温度，属于那条永不执行的
draft 路径；`temperature_by_kind`（1.2968… / 1.1892…）是**读数**温度。两个都叫 temperature，作用对象
完全不同。而 noul 那一路连读数温度都没有——它是 `"one_bin"`。

**进不进身份哈希也值得单说。** `threshold_identity`（`core.py:873-884`）把下面这些折进一个 sha256：

```python
canonical=json.dumps({'prompt':{k:config['prompt'][k] for k in ('system','enable_thinking')},'render':inspect.getsource(render),'choice_codes':inspect.getsource(choice_codes),
    'generation':{k:v for k,v in config['generation'].items() if k not in ('max_new_tokens','reasoning_words','wall_timeout_seconds')}},sort_keys=True,ensure_ascii=False)
```

也就是：**渲染器与码表的源码文本**、`prompt.system`、`prompt.enable_thinking`，以及去掉三个键的
`generation`。**没有进去的**有三类，值得记住：

- 校准常量（温度、`p_cal`）——它们由 `calibration.json` 自己携带，由 `SERVICE_LOCK` / `FREEZE` 绑定；
- **读数算术本身**（`probabilities`、`aggregate`）——改这两个函数不会改变 `prompt_sha`；
- `limits` 那一整块（含两个预算与两个准入常数）。

本文 §13.3 把这个哈希**从源码文本重算了一遍**，与 `service/config.json:196` 和
`calibration.json:73` 两处**逐位相同**。

## §10 四条不变量的落位

| 不变量 | 在本实现的落位 |
|---|---|
| **一次前向** | **机制层不成立，改写为**：一次决策 = 一次 prompt 前向 + ⌈候选 id 数 / 128⌉ 次 1-token 探针（`backends.py:506`、`518-519`）。26 个选项的 `choice` 是 104 个候选 id → **1** 次探针；676 个选项 → **22** 次。HF 后端是另一支：真的 0 个 token（`backends.py:159`、`304-319`） |
| **只读最后一个位置** | **成立，但「最后」由 chat 模板定义**：读的是 `<\|turn>model\n<\|channel>thought\n<\|channel\|>` 之后的那个位置（`chat_template.jinja:381-386`、`backends.py:139-153`）。运行期没有扫描、没有候选位置、没有阈值、没有标定——比「训练期钉死」更彻底，因为钉住它的是**模板文本**而不是权重 |
| **掩码到已知标签集** | **掩码概念不适用**：配置层禁止任何 logits 处理器（`core.py:133-137`），引擎层 `raw_logprobs` 写死且不可覆盖（`backends.py:387-396`、`425`）。改写后的等价命题是「**输出封闭在显式列出的 K 个 token id 上**」——不是把词表裁到标签，而是**只取标签那几个 id 的数**（`backends.py:512-513`）。因此 `marker_mass` 没有对象 |
| **重归一化** | **成立，但归一化的对象是标签不是词表**：内层按写法 logsumexp（`core.py:328-332`），外层按温度 softmax + 1e-6 地板后再归一（`core.py:356-362`）。和 RYOTIDE 的差别在于：RYOTIDE 要先减掉标签外质量，这里从来没有那个质量 |
| **`output_tokens = 0`** | **两层同时不成立**：机制上真的生成了探针 token，字段上就是这些 token 的**实测计数**（`backends.py:488-496`、`core.py:1131`、`1142`）。与 Kev（字段被重新定义成序列化答案的 token 数）、OpenJev(HF)（硬编码 0 而实际采 1 个）、decisio（定义成题数、而 packed 模式下并不生成）都不同——这一条是三者里唯一「生成即计费即上报」的 |

`selftest.py` 把其中两条做成了回归测试：`test_served_mode_is_fast_only`（`:75-77`）与
`test_input_budget_6653_everywhere`（`:131-140`，里面还断言了 `6653 + 3 <= 6656`，正是
`max_model_len` 与输入预算之间那 3 个 token 的余量）。

## §11 文件对照表

| 文件 | 行数 | 职责 |
|---|---|---|
| `service/core.py` | 1197 | 渲染、写法展开与单 token 过滤、标签聚合、温度与 `p_cal`、state 压缩、路由与预算分层、服务类与 fail-loud 状态机。**没有模型 import** |
| `service/backends.py` | 691 | 两个后端（vLLM / transformers）、探针式 `next_logits`、KV 分支会话、计量器、生成参数对照审计。**import 它不会加载模型** |
| `service/bench.py` | 795 | 维护者评测工具：跑题集、算成本与延迟、拟合温度与阈值、顺序对照臂。本服务运行期不 import 它 |
| `service/launch.py` | 456 | 前台督导：版本/身份/适配器/环境检查、两段式启动、就绪判定、信号与回收 |
| `service/quota.py` | 454 | 慢路径配额与成本保险丝：跨进程文件锁、原子写、`Decimal` 比较、内存退化模式 |
| `service/pkg_check.py` | 342 | 离线包自检：清单成员与哈希、冻结记录与前一份 config 的一致性、发布文本的内部标记扫描、披露完整性 |
| `service/server.py` | 218 | 标准库 HTTP 适配器，三条路由，三档错误码 |
| `service/selftest.py` | 198 | 纯软件自测（合成分词器 + 合成题），11 个用例，禁网 |
| `service/reconcile_usage.py` | 156 | 用量对账：E1/E2 两条恒等式，退出码 0/1/2 |
| `service/attention.py` | 37 | 从引擎日志里读**实际**注意力后端名，与「请求的」分开上报 |
| `service/model_setup.py` | 14 | 报告模型目录里缺哪个 processor config，不下载不复制 |
| `service/config.json` | 299 | 唯一配置：模型/量化/预算/路由/提示/服务面 |
| `service/calibration.json` | 127 | 三个块（`fast`/`slow`/`fallback`）的温度与 `p_cal` |
| `service/MODEL_MANIFEST.json` | 118 | 基座 + 适配器的逐文件哈希，以及 `identity_sha256` |
| `service/ADAPTER_MANIFEST.json` | 135 | 适配器来源、KVFILL 的 20 个张量、`adapter_config` 的两处归一化 |
| `service/FREEZE.json` | 67 | 冻结记录：冻结前后的逐文件哈希、唯一允许的文本改动 |
| `service/PIPELINE_PROVENANCE.json` | 55 | 构建来源：`copied_request_code_unchanged: true` 与逐文件哈希 |
| `service/SERVICE_LOCK.json` | 26 | 拟合时的身份锁：配置/校准/拟合报告的哈希 |
| `service/MEASURED_EVIDENCE.json` | 47 | 成本、延迟、可重复性的测量记录 |
| `service/fit_report.json` | 127 | 拟合过程记录：数值环境、数据哈希、`q=0` 阈值 |
| `service/MANIFEST.sha256` | 31 | 包成员清单（自身除外） |
| `service/requirements.lock` / `.measured.lock` | 204 / 201 | 版本锁各 201 条；后者是测量环境，两者只差三行注释与 `--extra-index-url`（§13.9） |
| `service/DISCLOSURE.md` / `README.md` / `MODEL_CARD.md` | 49 / 131 / 47 | 披露、运行说明、模型卡（模型卡与仓库根 `README.md` 逐字节相同，§13.1） |
| `chat_template.jinja`（根目录） | 390 | Gemma 4 的对话模板；读位置的尾巴在这里 |
| `tokenizer.json`（根目录） | — | 262144 个 entry 的 BPE 词表；单 token 写法的判定依据 |

## §12 亲手跑一遍

本文**没有运行**上游的任何程序（见「本文的事实来源」表）。下面是上游自己给的路径，照抄即可。

**离线、无 GPU、无网络的三条自检**（`service/README.md:120-126`）：

```bash
cd /abs/path/to/service
python3 -B pkg_check.py             # 包完整性、配置与适配器契约；FROZEN_PACKAGE_CHECK_PASS
python3 -B selftest.py --offline    # 纯软件测试，含长输入压缩；model_called: false, network: false
python3 -B pkg_check.py --release   # 列出还缺什么，永远退出 2
```

**完整的服务路径**（`service/README.md:34-95`）：先按
`--revision 9392335f5a17395ecd70049f85b1909d53500144 --exclude 'service/*'` 下模型目录（§13.2 说明它与
HEAD 的模型文件逐字节相同），再用固定 sha 的 `vllm/vllm-openai` 镜像起容器（`--network none --read-only`，
只挂一个可写 `WORK`），`serve.sh` 交给 `launch.py`，就绪判据是日志里出现 `{"status": "ok"` 那一行；
然后在同一个容器里跑官方 harness 的 `typesafe` 适配器。

**只取文本、不拉权重的取法**（本文用的就是这一条）：对上面那个 revision 的每个非 `*.safetensors` 文件，
按 `https://huggingface.co/<repo>/resolve/<sha>/<path>` 逐文件取。两个 `safetensors`
（23.3 GB 基座 + 495 MB 适配器）不必下载——它们的身份由 `MODEL_MANIFEST.json` 的 size 与 sha256 记录，
`launch.py` 默认也只比 size（`launch.py:209-212`）。

**对账工具**（需要一次完整的 `--engine-stats` 评测，本文没跑）：

```bash
python3 -B reconcile_usage.py --run /abs/out/fast --run-off /abs/out/routed_off --tol 0.001
```

## §13 在 committed 产物上的复算

这一节是本文的「实测」全部——十四项，全部只读随包发布的文本，不运行上游程序、不加载权重。

### 13.1 包内哈希清单：31/31 全过

`MANIFEST.sha256` 列 31 个文件（自身除外）。本文对下载到的文件逐个取 sha256：**31 个全部相同**，
且目录里的文件集合与清单成员集合相等——这正是 `pkg_check.py:84-99` 的检查内容。

顺带核到：仓库根的 `README.md` 与 `service/MODEL_CARD.md` **逐字节相同**（都是
`b274d2276d916a513c09cc2b6c362a8a74a69c6e0726b2c8f43241dfadd852e4`），而 `MODEL_MANIFEST.json:66-69`
把根目录的 `README.md` 与 `.gitattributes` 列为 `excluded_repository_documents`（被这张卡和仓库默认的
`.gitattributes` 替换掉的那两个）。

### 13.2 两个 revision 的模型文件逐字节相同

`service/README.md:53` 让读者按 `9392335f…` 下模型目录。本文按文件比对了 `9392335f…`（11 个模型文件）
与 HEAD `da56acc0…`（43 个文件）共有的 11 个：**blob oid 与 size 全部相同，0 处不同**；
`9392335f…` 里没有任何 HEAD 没有的文件，两个 revision 的差别就是 `service/` 整棵树。
所以「按 README 的 revision 下模型目录」与「按 HEAD 下模型目录」得到的是同一份基座与适配器。

### 13.3 重算 `prompt_sha`：与发布值逐位相同

按 `threshold_identity`（`core.py:873-884`）的定义重算：取
`render`（`core.py:266-281`）与 `choice_codes`（`core.py:206-209`）的**源码文本**、`prompt.system`、
`prompt.enable_thinking`，以及去掉 `max_new_tokens`/`reasoning_words`/`wall_timeout_seconds` 的
`generation`，用 `json.dumps(…, sort_keys=True, ensure_ascii=False)` 规范化后取 sha256。结果：

```
本文重算     e18ed7a06eb52f1b351458f384710a2c240aaf2b4f64d916b776c6d18ca3f938
config.json:196  e18ed7a06eb52f1b351458f384710a2c240aaf2b4f64d916b776c6d18ca3f938
calibration.json:73（fitted_for）  同上
```

三处逐位相同。这意味着：**改一个字符的 `render` 或 `choice_codes`，或者改 `prompt.system`、
`enable_thinking`、任何一个采样设置，这份校准就自动失效**——`validate_threshold`（`core.py:887-897`）
在 `mode == "routed"` 时会直接拒绝启动，在本配置（`fast_only`）下降级为一条 warning。
反过来也说明：改 `probabilities` 的算术**不会**让校准失效（§9.4）。

### 13.4 `identity_sha256` 复算一致，且五处互证

按 `pkg_check.py:76-82` 的 `canonical()` 重算 `{format, files, adapter:{format,subdir,files}}`：

```
本文重算     0513d121db967ec1a6aaccf627d3c42149d092dd027298da5b53684b030821ea
```

与下面五处**全部相同**：`MODEL_MANIFEST.json:101`、`service/config.json:118`（`pipeline.model_identity_sha256`）、
`FREEZE.json:22`、`MEASURED_EVIDENCE.json:29`、`PIPELINE_PROVENANCE.json`（`serving_identity_sha256` 经
`SERVICE_LOCK.json:23`）。`launch.check_preparation`（`launch.py:68-69`）要求配置与 manifest 的这两个
字段相等，正是这个链条。

### 13.5 `SERVICE_LOCK` 的两个文件哈希与实际文件一致

`SERVICE_LOCK.json:13`（`calibration_sha256`）与 `:17`（`fit_report_sha256`）与本文对
`calibration.json`、`fit_report.json` 取到的值逐位相同。这两条正是 `pkg_check.py:274-277` 检查的内容
（冻结包用 `FREEZE.json` 记的**冻结前** config 哈希，非冻结包用当前 config 哈希）。

### 13.6 延迟换算逐位吻合

`MEASURED_EVIDENCE.json` 里 `raw_p50_seconds = 0.24976357631385326`、
`adjusted_p50_seconds = 0.6495271526277065`。本文代入 `service/config.json:81-82`
（`latency_scale: 2.0`、`latency_offset_seconds: 0.15`）与 `DISCLOSURE.md:43` 自陈的换算律：

```
2 × 0.24976357631385326 + 0.15 = 0.6495271526277065   ← 与发布值逐位相同
```

`bench.py:348-350` 就是这条律的实现（`scaled = raw × latency_scale + latency_offset_seconds`）。

**但 `config.json` 里另有一组更松的准入常数**：`limits.median_latency_seconds = 1.0` 与
`limits.raw_median_latency_seconds = 0.425`，同样满足 `2 × 0.425 + 0.15 = 1.0`（逐位相同），而这组数
是 `bench.py:356`、`:430` 用来判「过不过门」的。同理 `limits.cost_per_1000_decisions_usd = 0.0646`
对发布值 `0.058646753246753246`（比值 0.9078），也是 `bench.py:353` 的门限。两组数各自自洽，但**不是同一
次测量**；`pkg_check.check_serving_policy`（`pkg_check.py:135-153`）并没有 pin 这三个字段（它 pin 的是
输入预算、上下文长度、显存比例、模式、配额与保险丝），所以它们是配置里的自由参数。引用「这一路的成本/
延迟」时要认准是 `DISCLOSURE.md` / `MEASURED_EVIDENCE.json` 那一对。

### 13.7 两个温度都是网格点

`1.2968395546510096 == 2**(3/8)` **为真**；`1.189207115002721 == 2**(2/8)` **为真**。
`bench.py:557` 的网格是 `[2**(i/8) for i in range(-32,41)]`，指数 −32…40 覆盖 2⁻⁴…2⁵。
所以这两个温度是**逐位落在拟合网格上的点**，不是任意浮点数。

### 13.8 `p_cal = 70/74`，而 74 正是 noul 的题数

`calibration.json:15` 的 `0.9459459459459459` 与 `70/74` 逐位相同；`calibration.json:43-47` 记的 noul
`count` 是 **74**，`bench.py:563` 的公式正是「argmax 正确率」。三个题型的 count 加起来
**139 + 74 + 18 = 231**，与 `service/config.json:6`（`expected_public_count`）和
`fit_report.json:26`（`n: 231`）相同。

### 13.9 两个 lockfile 只差三行

`requirements.lock`（204 行）与 `requirements.measured.lock`（201 行）逐行 diff 的结果是**只有三行**
（两行注释与一行 `--extra-index-url https://download.pytorch.org/whl/cu130`）；**201 条 pin 全部相同**。
而 `PIPELINE_PROVENANCE.json:12` 写着 `equal_to_source_package_measured_lock: false`——差异指的就是
那三行，不是版本。`pkg_check.py:306-308` 要求 `requirements.lock` 的 pin 与 `launch.VERSIONS` 六个版本一致。

### 13.10 单 token 写法的静态判定（§2 那张表的来源）

在随仓的 `tokenizer.json` 上做**纯数据**判定：`model.type` 是 BPE，normalizer 是
`Replace(" " → "▁")`，pre_tokenizer 是按空格切且 `MergedWithPrevious`。因此一条写法是单 token 的
**必要条件**是它规范化后的字符串在 `model.vocab` 里；对 BPE 再加一条充分性检查——构成它的 merge 对
存在。结果：

| 写法 | `A` | ` A` | `a` | ` a` | `A.` | `(A` |
|---|---|---|---|---|---|---|
| 规范化后 | `A` | `▁A` | `a` | `▁a` | `A.` | `(A` |
| 在词表里 | 是(236776) | 是(562) | 是(236746) | 是(496) | **否** | **否** |
| merge 对存在 | — | 是(`▁`+`A`) | — | 是(`▁`+`a`) | 否 | 否 |

`AA`/`▁AA` 同样都在（8686 / 23952）；数字只有裸 `0`…`9` 在词表里（`▁0` 的 merge 对不存在）；
`noul` 的 12 条 yes 写法与 12 条 no 写法全部在词表里。**注意本文这一项只做词表与 merge 的静态判定，
没有调用分词器**——`core.py:322` 真正的判据是 `len(encode(form)) == 1`，两者在 BPE 上应当一致，
但本文没有实际执行过 `encode`。

### 13.11 3 个 token 的余量

`limits.physical_max_prompt_tokens = 6653`（`service/config.json:79`）与
`model.vllm_options.max_model_len = 6656`（`:106`）相差 **3**；`selftest.py:140` 把这条写成了断言
（`assertLessEqual(6653 + 3, …max_model_len)`）。三个 token 的余量供给 chat 模板尾巴与生成提示。

### 13.12 `slow` 块不是本仓代码的产物

`calibration.json` 的 `fast` 与 `fallback` 两块**逐字段相同**（本文做的是 JSON 对象相等比较，结果 True）。
`fast.by_type` 的每个条目 `{count, fitted, nll, temperature}` 与 `bench.py:570` 的输出形状一致；
而 `slow.by_type` 的条目多出 **五个本仓任何代码都不写**的键——`inherited`、`inherited_from`、
`fast_count`、`applied_rule`、以及 `status: "inherited"`（对比 `bench.py:609` 只会写
`'insufficient'` 或 `'candidate'`）。`slow` 块的 `p_cal` 是 `0.9`，而 `bench.py` 的默认值是 `0.5`
（`bench.py:606`），`fit_calibration` 里那条继承分支只对 `fallback` 生效（`bench.py:610`）。
三条证据合起来说明：**`slow` 块来自本仓之外的一套拟合流程**，它连同 `applied_rule` 里那句
*"q0 has no slow decisions; inherit fast; noul clamp [.81,.90]"* 一起被发布出来。
这在本次服务里不影响任何数值——`mode: "fast_only"` 下 `block` 只可能是 `'fast'`（`core.py:1080`、
`1087`），`slow` 与 `fallback` 两块只被 `validate_calibration` 校验、从不被读（`core.py:911-921`）。

### 13.13 FREEZE 记录的「冻结前」config 可以逐字节还原

`MANIFEST.sha256:17` 记 `config.json` 是 `89bb9dce…`，而 `FREEZE.json` 的 `prepared_files.config.json`
记的是 `98b9630c…`——两个哈希不同，看起来矛盾。本文把发布出来的 `config.json` 里**唯一一处**
`"package": {"finalized": true` 的 `true` 改成 `false`（一个字符），重取哈希：

```
发布文件的哈希                      89bb9dce235caa8e165418330b32e9de5d1e32a702162f1a721c3715d667fccf  = FREEZE.config.frozen_sha256
把 finalized 改成 false 之后的哈希   98b9630c0fdeb663c03b7351263171f88afc9e5e4d2353467764bfe37b1003ee  = FREEZE.config.prepared_sha256
```

与 `FREEZE.json:5-6` 两个字段、以及 `:39` 的 `prepared_files.config.json` 逐位相同。这不是矛盾：
`FREEZE.json` 是**两段记录**——`prepared_files` 是冻结**之前**那一份包（当时 `finalized: false`），
`frozen_sha256` 与 `MANIFEST.sha256` 描述的是**最终**那一份。代码里对应的是
`pkg_check.py:155-159` 的 `prepared_config_bytes`（要求全文件恰好一处 `"finalized": true`、且不出现
`"finalized": false`，再替换），以及 `pkg_check.py:181-191` 那条「除 `config.json` 与被替换的三份文档之外，
任何文件都不得与 `prepared_files` 不同」的断言。`DISCLOSURE.md` 是同一形状的第二个例子：
`MANIFEST.sha256:4` = `FREEZE.replaced_documents.DISCLOSURE.md.frozen_sha256` = `6b2ba5c1…`（发布的那一份），
而 `prepared_sha256` = `0f31493a…`（冻结前的那一份）。

### 13.14 臂名只在机器记录里，不进发布文档

`pkg_check.py:50-52` 的 `DOCUMENT_PROCESS_MARKERS` 明确禁止三份发布文档出现 `FINALIST`、`KVFILL`
这类内部臂名；本文 grep 的结果与之一致：`FINALIST` 与 `KVFILL` 只出现在 **JSON 记录**里
（`ADAPTER_MANIFEST`、`config.json`、`fit_report`、`MEASURED_EVIDENCE`、`MODEL_MANIFEST`、
`PIPELINE_PROVENANCE`、`SERVICE_LOCK`）以及 `pkg_check.py` 自己的标记表里，
`README.md`／`DISCLOSURE.md`／`MODEL_CARD.md` 三份**零命中**。同一张表里还有
`(?<![A-Za-z])Rain(?![A-Za-z])`——发布文本里不许出现裸的 `Rain`；`internal_markers()` 会先把
`repo_id` 从文本里抹掉再扫（`pkg_check.py:196-207`），所以 `MeTask-Rain/…` 这种自引不算命中。

## §14 与另外几篇并读时的落点

**没有实例走查，因为包里没有逐题跑盘记录。** 这一路提交的是**包自证材料**（清单、冻结记录、身份哈希、
测量摘要、数值环境），而不是别的实现那种 `runs/` 或 `results.jsonl`。所以本文的「实测」只能落在
§13 那十四个静态复算上：哈希、算术、词表判定、以及一处可以逐字节还原的文本改动的验证。
**没有前向、没有概率、没有延迟的独立复算**——`MEASURED_EVIDENCE.json` 里那三个数（成本、两个 p50、
可重复性）只有上游自己一次测量的记录，本文无法重跑，也不能核。

并读时最值得与别篇对上的三处：

1. **「会生成 1 个 token」这一侧的第三个例子**，而且是与前两个都不同的形态：SemIf-OpenJev 的浏览器路径
   生成的 token 是**接口要求**（为了拿到 `top_logprobs`），OpenJev(HF) 生成 1 个 token 但把
   `output_tokens` **硬编码成 0**，而这一路是**生成数即上报数**（§4、§10）。
2. **「词的读法」的第五种机制**：decider、Nimble、Kev、H2O-Lightning 之后，这一路是又一条
   `marker_mass` 概念不适用、但机制完全不同的路线——不是「头只输出 K 个标量」，也不是「只请求 K 个
   token id 的 logprob」这一条本身，而是**在这之上再加一层「同一标签多种写法的 logsumexp」**。
   写法的重数还刻意保留（`core.py:318`），所以「一个标签对应几个 token」不是一对一，而是多对一。
3. **`prompt_sha` 的第二例**：Nimble 让发布出去的 prompt 源码哈希变成加载期硬校验；这一路的
   `threshold_identity` 更进一步——它把**渲染器的源码文本与码表的源码文本**一起折进哈希，并且这个哈希
   同时写在 `config.json`、`calibration.json` 与 `/health` 的 `threshold_identity` 里（§13.3）。
   两处都能从源码文本逐位重算，这是两条路线共同的、可被机器验证的性质。

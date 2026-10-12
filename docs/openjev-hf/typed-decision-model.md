# OpenJev 是如何把微调过的 Qwen 读成类型化决策模型的

> 代码库：<https://huggingface.co/openjev/openjev>
> 对应 README 第 52 行：*The answer is read from the scores at the first output position: one score
> per option, converted to probabilities using fixed calibration settings.*
> 本文按 `helper/` 与 `serve/` 的实际代码逐环节说明这句话的实现方式；文中出现的文件名与行号（`shim.py:148`、
> `chat_template.jinja:165`、`config.json:16` 等）均相对该仓库根目录。
>
> **同名配对文档**：RYOTIDE 的 [`docs/ryotide/typed-decision-model.md`](../ryotide/typed-decision-model.md)。
> §0 先讲清分歧，§14 把它的四条不变量逐条落位。
>
> **与另外两份「OpenJev」的区别**：本仓库里已有两篇名字里带 OpenJev 的走查——
> [`docs/open-jev/`](../open-jev/typed-decision-model.md)（[Zefan-Cai/Open-Jev](https://github.com/Zefan-Cai/Open-Jev)，
> `jev/` 包、标量决策头）与 [`docs/semif-openjev/`](../semif-openjev/typed-decision-model.md)
> （[TheoLeeCJ/SemIf-OpenJev](https://github.com/TheoLeeCJ/SemIf-OpenJev)，零训练、冻结 4B）。
> 本篇的上游是 Hugging Face 上的 `openjev/openjev`，**第三套独立代码**：微调过的 Qwen3.8-27B，加上一个挡在
> vLLM 前面的 HTTP 补丁。三套代码各自独立（可按文件核对）；它们的历史关系本文**未核实**，不推测。

## 0. 与 RYOTIDE 的分野

两边都读「选项字母 token 在某个位置的 logits，掩码后 softmax」，也都承诺不解析自由文本。
但这一条路线把 RYOTIDE 的每一个自由度都换成了**训练期决定 + 事后校准常数**。

| | RYOTIDE | OpenJev |
|---|---|---|
| 决策机制 | 读标签 token 的末位 logits，掩码 + softmax | 读**字母 token 在第一个输出位置**的 logits，`allowed_token_ids` 掩码 + 除以 T 的 softmax |
| 需要训练 | 否，权重全程 frozen | **是**：在 Qwen3.8-27B 上微调（*trained to select the correct option and stay consistent when the options are reordered*，`README.md:191`） |
| 「typed」从哪来 | 分词器：标签必须单 token，否则映射成字母标记 | 请求里的 `criteria` + 固定的 52 个字母（`A`–`Z`、`a`–`z`，`shim.py:16`）；服务端用 `allowed_token_ids` 硬约束 |
| 候选数上限 | 260（`A0`–`Z9` 两步读；>26 另有 `grouped` 近似读法） | **52**（一步）；>52 走**分块两级近似**（§9） |
| 读位置怎么定 | **运行期标定**：扫 9 前缀 × 2 表面形式，取 marker mass 最高 | **训练期固定**：微调让第一个输出位置携带决定；运行期只有 chat 模板这一道防线（§3） |
| 顺序偏置 | 多 order 共享前缀 fork，折叠回标签后概率平均 | 有开关（`READOUT_PERMS`），**但发布配置里是 1、不平均**；训练把翻转率从 18.5% 压到 2.3%（§7） |
| 置信度来源 | **读出来的**：只有 temperature 能改数值 | **训出来的**：训练本身在改数值、也在改 argmax（18.5% → 2.3% 就是训练改出来的）；T=0.85 与 noul 的 (1.829074, 0) 是**训练之后**施加的常数（§13） |
| 诊断量 | `marker_mass` | `GET /v1/version`：served model dir、全部标定常数、全部 flag、**补丁文件自己的 sha256**（`shim.py:339-348`） |
| 多问题共享 state | `run_many()`：公共前缀 prefill 一次 | 每题各跑一次 prefill，靠 vLLM 前缀缓存 + `prewarm` + `SHIM_STAGGER` 复用（§8） |
| 服务形态 | FastAPI `/v1/systemone` + `/health` | 标准库 `http.server` 起的一个补丁进程：`/v1/systemone`、`/v1/version`、`/v1/prewarm`、`/v1/chat/completions`（§12） |
| 共同底线 | **`output_tokens = 0`** | 响应里**硬编码** `"output_tokens": 0`（`shim.py:325`），但对 vLLM 发的是 `max_tokens: 1`——确实采了一个 token，只是那个 token 的值不被读（§4） |

一句话：RYOTIDE 是**「不改权重，只换读法」**，OpenJev 是**「把读法训进权重，再用一组常数把分数掰成概率」**——
它省掉了 RYOTIDE 那套运行期标定，代价是每一个常数都变成发布配方的一部分，且**代码里的默认值与发布值不是同一套**（§13）。

## 1. 端到端数据流

```
        ┌──────────── helper/shim.py（标准库 http.server 补丁）────────────┐
POST    │ H.do_POST                                    shim.py:287        │
/v1/    │  ① SHIM_TOKEN 校验（设了才校验）                shim.py:288-289    │
systemone│  ② state + questions 非空校验 → 422            shim.py:304-306    │
        │  ③ 每个 question.type ∈ {choice,score,noul}    shim.py:314-316    │
        │  ④ state → with_image()：拆出 screenshot 通道   shim.py:69-80     │
        │  ⑤ ThreadPoolExecutor(16) 并发跑每题            shim.py:283,320    │
        └───────────────────────────┬─────────────────────────────────────┘
                                    v
        ┌─────────── _readout_once()：一次补丁内的“一次前向” ────────────┐
        │ 拼 prompt：State / Question / [A] opt …        shim.py:132-146     │
        │ letter_ids()：52 个字母各取一个 token id        shim.py:25-30      │
        │ allowed = [该题各选项的字母 id]                 shim.py:148        │
        └───────────────────────────┬─────────────────────────────────────┘
                                    v  OpenAI 兼容调用 → vLLM
                  ┌─────────────────┴──────────────────┐
                  │ READOUT_TARGETED=1（发布配置）      │ 未设（文件默认）
                  │ logprob_token_ids + 按 id 匹配      │ top_logprobs + 按字符串匹配
                  │ 缺一个就 ReadoutIncomplete 报错     │ 掉出 top-K 的给 -30 兜底
                  │ shim.py:149-155                    │ shim.py:156-164
                  └─────────────────┬──────────────────┘
                                    v
                   z = logprob / T ；softmax(z)          shim.py:155,164-166
                                    │
        ┌───────────────────────────┴─────────────────────────────────────┐
        │ choice → choice_confidence()        shim.py:192-194,220-226      │
        │ score  → 期望档位 + score_confidence() shim.py:196-201,228-235   │
        │ noul   → logit(p_yes)/NOUL_T+NOUL_BIAS → sigmoid  shim.py:237-242 │
        └───────────────────────────┬─────────────────────────────────────┘
                                    v
        {"answers": {…}, "usage": {"input_tokens": Σ, "output_tokens": 0}}   shim.py:324-325
```

三条题型的公共部分是同一件事：`_distribution()`（`shim.py:203-218`）→ `readout()`（`shim.py:168-178`）
→ `_readout_once()`（`shim.py:125-166`）。`choice` 与 `noul` 的差别只是候选集不同
（`noul` 就是两个候选 `yes`/`no`，`shim.py:240`），`score` 多一层「把概率当有序档位求期望」（`shim.py:233`）。

## 2. 「typed」从哪来：请求里的 `criteria` + 固定的 52 个字母

标签集**不随模型固定，随请求固定**。请求的每个 question 自带候选：

- `choice`：`criteria` 是「选项名 → 描述或 null」的**非空 map**（`shim.py:221-223`）；
- `score`：`criteria` 是**有序数组**、至少两档、从低到高（`shim.py:229-231`）；
- `noul`：`criteria` 可选，给 `{"true": …, "false": …}` 的自定义描述，缺省时用「The statement is true./false.」（`shim.py:238-239`）。

这些候选按**送出顺序**映射到字母（`shim.py:132`）：

```python
lines = "\n".join(f"[{LETTERS[i]}] {k}: {d}" for i, (k, d) in enumerate(options))
```

字母表是**26 个大写 + 26 个小写**（`shim.py:16`）：

```python
LETTERS = [chr(65 + i) for i in range(26)] + [chr(97 + i) for i in range(26)]
```

52 这个数是「一次前向能读多少」的上限，不是模型的词汇限制：一个字母一个 token，52 个字母就够读到 52 个选项；
再多就分成多趟（§9）。

**字母要成为可读的槽位，靠的是两件事**：

1. `letter_ids()` 用分词器把每个字母换成 token id（`shim.py:25-30`）：

   ```python
   _ids.update({L: t.encode(L, add_special_tokens=False)[0] for L in LETTERS})
   ```

   注意它取的是 `[0]`——**只假设第一个 token 就是那个字母，没有验证这个字母恰好是一个 token**，
   也没有验证往返解码。这是本篇与 `docs/semif-openjev/` 那篇最直接的一处强弱差：那边用
   「单 token + 往返一致 + 边界不变」三重断言把这条路堵死（其 §2），这里靠的是**基座模型的分词器事实**
   加上 `allowed_token_ids` 的服务端强制。
2. 请求里带 `allowed_token_ids=allowed`（`shim.py:152,159`），把 vLLM 的采样集合直接限成这 52 个
   （本题实际用到的那些）。**这不是 prompt 层的技巧，是服务端的 grammar 级约束**——所以模型没有机会
   输出一个不在候选里的字母。

同时 `letter_ids()` 和 `_header()` 都从 `TOKENIZER` 指向的目录加载分词器（`shim.py:28,88`），
默认值是 `Qwen/Qwen3.8-27B`；`serve/SERVE.md:61` 把这个默认值单列出来，提醒发布时必须改成模型目录，
因为**它同时被当成「served 模型名」写进 `/v1/version`**（`shim.py:343`）。

## 3. 读位置：第一个输出位置，由微调固定；chat 模板是唯一防线

`README.md:191` 把这件事写得很直白：*The model is tuned so that this single position carries the decision.*
所以运行期没有任何位置标定——没有 RYOTIDE 的 9 前缀 × 2 表面形式扫描，也没有 `marker_mass` 这类诊断量。
位置是权重属性。

**这让 chat 模板的唯一防线变得很关键。** 这个模型的 `chat_template.jinja` 有一个不常见的默认：

```
46  {%- if enable_thinking is undefined or enable_thinking is true %}
47      {%- set resolved_reasoning_effort = reasoning_effort|default('xhigh') %}
...
52          {%- set reasoning_instructions = 'Reasoning effort is set to xhigh. Please think carefully …' %}
```

**不传 `enable_thinking` 就等于打开 thinking，而且默认档位是 `xhigh`**，会在 system 位置塞进一段
「请仔细思考、验证关键假设」的指令，并且 assistant 的前缀写成 `<think>\n`（`chat_template.jinja:168`）
——**一个没有闭合的思维块**。读位置落进去，读到的就是思维块里的第一个 token，而不是选项字母。
这就是 RYOTIDE 不变量 #1 的失败模式，在这里它不是靠诊断量发现，而是靠**每一处调用都显式传
`enable_thinking: False`** 来避免：

- 三处补丁内的调用：`shim.py:111`（prewarm）、`shim.py:152`（TARGETED）、`shim.py:159`（旧协议）；
- 一处透传：`passthrough_chat()` 把请求体里的 `chat_template_kwargs` 直接**覆盖**成
  `{"enable_thinking": False}`（`shim.py:275`），并顺手丢掉 `reasoning` / `thinking` 两个字段（`shim.py:274`）。

传 `false` 之后，模板在 assistant 位置写的是**一个空思维块**（`chat_template.jinja:165-166`）：

```
165      {%- if enable_thinking is defined and enable_thinking is false %}
166          {{- '<think>\n\n</think>\n\n' }}
```

于是「第一个输出位置」落在 `</think>\n\n` 之后——这正是训练时读的那个位置。MLX 那条替代路径也照抄了这套
（`shim_mlx.py:31` 用 `enable_thinking=False` 调同一个模板）。

> 与另外几条路线的接法对比：这里既不是「传个 kwarg 就完事」（模板一改就失效），也不是 reranker 那种
> 手写死整个 prompt；是**模板本身**把 thinking 的两种形态都写好了，补丁负责**每一处都不漏传**。

## 4. 一次前向：`max_tokens=1`，但答案不从生成结果读

对 vLLM 的两次调用都带 `max_tokens=1`、`temperature=0`、`logprobs=True`（`shim.py:151,157`）。
所以严格说，**这条路线确实采了一个 token**——`output_tokens = 0` 在这条路径上不成立。

但被读的不是那个 token：

- TARGETED 分支拿的是 `r.choices[0].logprobs.content[0].top_logprobs`，按 `token_id:<id>` 取
  （`shim.py:153`）；
- 旧协议分支拿的是同样的 `top_logprobs`，按 token 字符串取（`shim.py:160-163`）。

`temperature=0` 在这里只是把采样变成确定性，**它不影响读出来的 logprob**——logprob 是那一步的
log-softmax 结果，不是采样结果。所以「采样了一个 token」是**接口的代价，不是读法的一部分**：
vLLM 的 OpenAI 兼容接口没有「只给 logits 不要 token」的模式，`max_tokens=1` 是能拿到该位置分布的最短路径。

响应里 `"output_tokens": 0` 是**写死的**（`shim.py:325`）：

```python
out = {"id": f"shim-{int(time.time() * 1000)}", "model": MODEL_STRING, "answers": answers,
       "usage": {"input_tokens": toks, "output_tokens": 0}}
```

`input_tokens` 是各趟 `r.usage.prompt_tokens` 的真实累加（`shim.py:166,323`），`output_tokens` 不是实测值。
`serve/SERVE.md:159` 在响应示例里也照这么写。引用这条路线时应当写成
**「响应自报 output_tokens = 0」**，而不是「不生成 token」。

## 5. 两条读数协议：TARGETED 与旧的 top-K 兜底

`READOUT_TARGETED` 决定走哪一条（`shim.py:21`），**两条路读的是同一个位置，但取分方式不同**。

**TARGETED（发布配置，`serve/SERVE.md:65`）**——`shim.py:149-155`：

```python
assert len(set(allowed)) == len(allowed), "candidate token ids are not unique"
r = client.chat.completions.create(..., max_tokens=1, temperature=0, logprobs=True, ...,
    extra_body={"allowed_token_ids": allowed, "logprob_token_ids": allowed,
                "return_tokens_as_token_ids": True, "chat_template_kwargs": {"enable_thinking": False}})
got = {t.token: t.logprob for t in r.choices[0].logprobs.content[0].top_logprobs}
raw = [got.get(f"token_id:{i}") for i in allowed]
if any(v is None or not math.isfinite(v) for v in raw):
    raise ReadoutIncomplete(f"{...} of {len(allowed)} candidate scores missing or not finite")
```

要点有三个。第一，它按 **token id 匹配**，不是按字符串——注释写明了理由（`shim.py:149`）：

> *top_logprobs is the raw top-K over the whole vocabulary, taken before the allowed-token mask, and drops labels on many-option heads*

第二，缺分**直接报错**，注释写着 *never floored, never a whitespace look-alike*（`shim.py:154`）——
不允许用一个兜底值把缺口糊过去。

第三，`assert len(set(allowed)) == len(allowed)` 是个**同字母碰撞的自检**：如果两个选项拿到同一个 token id，
说明分词器把两个字母映射到了一起，直接崩掉而不是给出一个静默错位的分布。

**旧协议（未设 `READOUT_TARGETED` 时的默认）**——`shim.py:156-164`：

```python
r = client.chat.completions.create(..., max_tokens=1, temperature=0, logprobs=True,
                                   top_logprobs=max(20, len(options)), ...)
lp = {}
for t in r.choices[0].logprobs.content[0].top_logprobs:   # exact letter token wins; ' U' must not overwrite 'U'
    k = t.token if t.token in LETTERS else t.token.strip()
    if k not in lp or t.token in LETTERS: lp[k] = t.logprob
z = [lp.get(LETTERS[i], -30.0) / TEMP for i in range(len(options))]   # OLD PROTOCOL: a label outside the raw top-K gets the -30 floor
```

这里有三处会改变数值的东西，值得逐条记：

1. `top_logprobs=max(20, len(options))`——旧协议要求把 top-K 开到至少和选项数一样大。
   如果某选项字母没挤进 top-K（很多选项时会发生），就拿不到分。
2. **字符串匹配 + 大小写保护**：`' U'`（带前导空格）与 `'U'` 是两个不同的 token 字符串，
   代码用 `t.token in LETTERS` 判断哪个是「真正的字母 token」，保证 `' U'` 不会覆盖 `'U'` 的分。
3. **掉出 top-K 给 `-30.0` 兜底**。代入 T=0.85，这个兜底值在 softmax 里的相对权重是
   e^(−30/0.85) = e^(−35.3) 量级——也就是被压到几乎为零，但仍**不是零**，分布仍然归一。
   这是与 TARGETED 的实质差别：TARGETED 路径宁可失败，旧协议路径宁可给一个极小的假值。

`serve/SERVE.md:65` 把这条关系说得很清楚：**T 和 noul 的斜率是在 TARGETED 打开的情况下拟合的**
（*The temperature and the yes/no slope above were fitted with this ON, so keep the three together*）。
所以「用发布值的 T、却跑旧协议」不是「差一点」，是**把两个不同读数的常数混用**。

所有日志都走两条分支之后同一段（`shim.py:165-166`）：

```python
m = max(z); e = [math.exp(v - m) for v in z]; s = sum(e)
return [v / s for v in e], r.usage.prompt_tokens
```

即先除以 T、再减最大值、再指数、再归一化。

## 6. 掩码 + softmax + 两个 `confidence` 公式

这里的「掩码」有两层，而且**方向相反**：

- **服务端掩码**（真掩码）：`allowed_token_ids=allowed` 让 vLLM 只在候选字母里采样（`shim.py:152,159`）；
- **客户端取值**（不是掩码）：`z` 只包含候选字母的分数，然后在这 ≤52 个数上 softmax。

两层叠起来的效果是：分布是**严格条件在请求给的候选集上**的。这正是 README 里
*conditional* 那层意思没有明说、但代码里硬保证的东西。

**两个 `confidence` 不是模型给的，是按公式算的。** `shim.py:191` 标明它们来自
`typesafe-ai/system-one-adapter-python confidence_metrics.py`——也就是说，它们是 **Jev API 的官方定义**，
不是这条路线自己的发明：

```python
def choice_confidence(p):
    if len(p) == 1: return 1.0
    u = 1.0 / len(p); return max(0.0, (max(p) - u) / (1.0 - u))

def score_confidence(p):
    if len(p) == 1: return 1.0
    mode = max(range(len(p)), key=p.__getitem__)
    dist = sum(pi * abs(i - mode) for i, pi in enumerate(p))
    c = (len(p) - 1) / 2; umad = sum(abs(i - c) for i in range(len(p))) / len(p)
    return max(0.0, 1.0 - dist / umad)
```

- `choice_confidence`：把「比均匀分布高多少」按均匀分布到 1 的剩余区间归一。N=2 时
  `(p_max − 0.5) / 0.5`，即 2·p_max − 1；N=3 时 `(p_max − 1/3) / (2/3)`。**N 不同，同一个 p_max
  给出不同的 confidence**——跨选项数比较 confidence 之前必须先对齐 N。
- `score_confidence`：算「到最可能档位的期望距离」，除以均匀分布下的平均绝对距离 `umad`，再取补。
  它用的是**档位序号**（`i`）本身的度量，所以**假设档位是等距的**：`["低","中","高"]` 与
  `["不可能","也许","一定会"]` 被判成同样的距离尺度。代码里没有任何地方表达档位间距。

这两个公式与 RYOTIDE 的 `marker_mass` 是两种东西：`marker_mass` 诊断「读得对不对」，
这两个公式只是把已经读出来的分布**换个刻度**给调用方看。它们**不参与** `choice` / `score` 的选择
——`shim.py:225` 明写 *winner picked before rounding*，argmax 取在四舍五入之前。

## 7. 顺序偏置：README 说 2.3%，而代码里默认根本不平均

这条路线对顺序偏置的处理是**训练**，不是推理期的技巧。`README.md:104-109` 的对照表把它写成了
一条主结果：

| test | steps | before tuning | OpenJev |
|---|---:|---:|---:|
| answer flips when the options are shuffled | 2,000 | **18.5%** | **2.3%** |

也就是说：**同一个基座、同一套读数，微调把「换一下选项顺序就改答案」的比例从 18.5% 压到 2.3%**。
这正好落在 `docs/comparison.md` §4.4 那条分界上——它是「训出来的」，训练本身在改 argmax。

代码里同时保留了一个**推理期的平均开关**，但发布配置是关的：

```python
19  PERMS = int(os.environ.get("READOUT_PERMS", "1"))   # >1 = average over that many letterings (4 ≈ +16 pts acc, 4× cost)
...
174      for j in range(PERMS):
175          order = list(range(len(options))); random.Random(j).shuffle(order)
176          p, t = _readout_once(state_text, instructions, [options[i] for i in order]); tokens += t
177          for pos, i in enumerate(order): acc[i] += p[pos] / PERMS
```

三点值得记：

1. **默认 1，即不平均**。`serve/SERVE.md:69` 明确把它列进「发布数字没有动过的默认值」。
2. 置换用的是 `random.Random(j)`——**每个 j 一个固定种子**，不是全局随机。所以同一个请求重复跑，
   取的是同一组 lettering，PERMS>1 也是可复现的。
3. 注释给的代价是 *4 ≈ +16 pts acc, 4× cost*：4 种排列大约涨 16 个点准确率，代价是 4 倍算力。
   这条注释没有任何文件路径或 run 标签可追（§17）。

所以引用「2.3%」时必须说清楚：**那是训练后的模型在默认配置（不平均）下的实测**，
不是「这套代码消除了顺序偏置」。代码把它量化在 2.3%，同时留下一个能把剩下的压下去的开关，
且**默认不开**。

## 8. 多问题共享 state：vLLM 前缀缓存 + prewarm + stagger

一次请求带多个问题时，每题各跑一次 prefill（`shim.py:320` 用 `ThreadPoolExecutor.map` 并发），
**共享 state 的复用交给 vLLM 的前缀缓存**，补丁只做三件辅助的事。

### 8.1 `prewarm`：把 page 先读一次

`POST /v1/prewarm`（`shim.py:298-303`）调 `prewarm()`（`shim.py:108-112`）：

```python
r = client.chat.completions.create(model="qwen", max_tokens=1, temperature=0,
                                   messages=[{"role": "user", "content": prefix_text(state_text)}],
                                   extra_body={"chat_template_kwargs": {"enable_thinking": False}})
return r.usage.prompt_tokens
```

它发的是**只有 state、没有问题**的那一段（`prefix_text`，`shim.py:104-106`），
让 vLLM 把 page 的 KV 缓存下来。后续每个问题的 prompt 都以这一段开头，于是只付问题尾巴的钱。

注意这个「前缀」是**文本层切出来的**（`f"State:\n{state_text}"`，`shim.py:106`），
不是 RYOTIDE 那种从完整 prompt 里反推 token 前缀的做法。它能不能命中缓存，取决于
**`State:\n` 之后的文本逐字节相同**——所以 state 里任何一个每次请求都变的字段都会让缓存全废。
`DROP_KEYS`（`shim.py:38-39`）注释里点了名：*the telemetry changes every send, which defeats prefix reuse across steps*。

### 8.2 `SHIM_PAD`：把前缀对齐到 784 token 的块边界

`shim.py:82` 与 `92-102`：

```python
PAD = int(os.environ.get("SHIM_PAD", "0"))  # 784 = the cache block of this model: pad the shared page prefix to a block boundary so every question reuses all page blocks

def pad_prefix(prefix):
    """Append filler so the chat header + prefix is 2 tokens past a multiple of PAD tokens (the boundary stays inside the shared text)."""
    if not PAD: return prefix
    t, h = _header(); n = len(t.encode(h + prefix, add_special_tokens=False)); k = (2 - n) % PAD
    if k < 2: k += PAD
    out = prefix + "\n" + " pad" * k
    for _ in range(3):  # BPE can merge fillers; correct the count
        m = len(t.encode(h + out, add_special_tokens=False)); d = (m - 2) % PAD
        if d == 0: break
        out = out + " pad" * (PAD - d) if d > PAD // 2 else out[:len(out) - 4 * d]
    return out
```

这段是**为了对齐缓存块而往 prompt 里塞填充**，而且承认填充本身会改变分词：`" pad"` 重复若干次之后，
相邻的填充可能被 BPE 合并成一个 token，所以它迭代三次、每次重算 token 数并修正。
`_header()`（`shim.py:84-90`）先单独渲染一次模板、切出用户内容之前的部分，是为了让**token 计数和 vLLM
看到的一致**——因为补丁要的是「header + prefix」的总长，不是 prefix 自己的长度。

**发布配置里 `SHIM_PAD` 是关的**（默认 0；`serve/SERVE.md:69` 把它列进未设置的默认值）。
784 这个数在 `config.json` 里没有对应字段，源码注释只说它是 *the cache block of this model*——
**它的推导本文未核实**。

### 8.3 `SHIM_STAGGER`：长 state 先让第一题独占一拍

`shim.py:284-285` 与 `319-320`：

```python
STAGGER = os.environ.get("SHIM_STAGGER") == "1"
STAGGER_MIN_CHARS = int(os.environ.get("SHIM_STAGGER_MIN_CHARS", "16000"))  # ~4k tokens: below that all questions fit one scheduler step and share blocks in flight anyway
...
stagger = self.STAGGER and len(state_text) >= self.STAGGER_MIN_CHARS  # first question alone so its state blocks are cached before the rest run
results = ([f(items[0])] + list(self.pool.map(f, items[1:]))) if stagger else list(self.pool.map(f, items))
```

逻辑是：长 state 下如果所有问题同时发出去，vLLM 的调度器可能还没把 page 的块算出来，
每个问题就各自算一遍前缀。先跑一题、等缓存落地，其余再并发。阈值按**字符数** 16000 判
（注释说约 4k token），不是按 token 数——一个近似的启发式。
发布配置里 `SHIM_STAGGER=1`（`serve/SERVE.md:67`）。

### 8.4 线程池与并发

`H.pool = ThreadPoolExecutor(max_workers=int(os.environ.get("SHIM_POOL", "16")))`（`shim.py:283`），
`protocol_version = "HTTP/1.1"` 开 keep-alive、并关掉 Nagle（`shim.py:281-282`）：

> *keep-alive: far-away clients otherwise pay a full round trip per call* /
> *headers and body go out immediately instead of waiting for the ACK (one round trip less)*

`README.md:141` 那个「100 个问题 3.9 秒、平均每题 39 ms」的数字，就是在这个 `concurrency 16` 下测的，
**不是单请求延迟**——README 自己把这句话写出来了。

## 9. 超过 52 个选项：分块两级读法，以及它成立的前提

52 是「一次前向一个字母能读多少」的硬上限（§2）。超过之后 `_distribution()` 改成两趟
（`shim.py:203-218`）：

```python
if len(opts) <= 52: return readout(state_text, instr, opts)
k = -(-len(opts) // 52); size = -(-len(opts) // k); chunks = [opts[i:i + size] for i in range(0, len(opts), size)]
parts, tokens = [], 0
for chunk in chunks:
    p, t = readout(state_text, instr, chunk); tokens += t; parts.append((p, max(range(len(p)), key=p.__getitem__)))
pf, t = readout(state_text, instr, [chunk[w] for chunk, (_, w) in zip(chunks, parts)]); tokens += t
raw = [pf[c] * pi / p[w] for c, (p, w) in enumerate(parts) for pi in p]; s = sum(raw)
return [v / s for v in raw], tokens
```

先把选项切成**近似等长**的块（每块 ≤52，块数取最小值），每块单独读一次得到一个块内分布，
再拿每块的**胜者**组成一个只有 k 个候选的 prompt 读一次，最后按

```
p(i) ∝ p_final(第 c 块) · p_chunk(i) / p_chunk(第 c 块的胜者)
```

把两趟拼起来。docstring（`shim.py:203-210`）把它的前提和保证都写清楚了：

> *Approximation: an option's letter logit is taken to be the same in its chunk prompt and in the winners prompt,
> so each winner anchors its chunk's mass on the winners' scale; when that holds exactly, p equals the softmax
> over all options. Every option gets a non-zero probability, the distribution sums to 1, it does not depend on
> option order beyond float noise, and confidence is computed on it (not on the winners alone).*

三个性质它明确保证：**每个选项概率非零**、**和为 1**、**（在近似成立时）与选项顺序无关**。
一个性质它明确承认是近似的：**同一个选项的字母 logit 在「块内 prompt」与「胜者 prompt」里被当成同一个值**
——但这两次 prompt 的候选集不同，模型对同一个字母的打分未必一样。所以 >52 走的是近似，
`serve/SERVE.md:171` 也把它列进 *Known helper behaviour*：*Several passes, approximate, and less stable
between runs than a single pass.*

注意这里**字母是复用的**：第 c 块内部的选项用的是 `A`、`B`、`C`…（每个块各自从 A 开始），
而不是全局编号。所以「选项 → 字母」这个映射在 >52 时是**逐块重置**的，
只有块内位置才有意义——这和 §2 那套「按请求顺序映射到字母」的规则并不冲突，但引用时不能混。

## 10. 三种题型与 noul 的校准公式

`ANSWER = {"choice": answer_choice, "score": answer_score, "noul": answer_noul}`（`shim.py:244`），
三种题型共用 `_distribution()`，差别在候选集与最后的成形：

| 题型 | 候选集 | 成形 |
|---|---|---|
| `choice` | `criteria` map 的键，按插入序 | argmax 选项名 + 每个选项的概率 + `choice_confidence`（`shim.py:220-226`） |
| `score` | 档位下标 `0..N-1`（描述可空） | 期望档位 `Σ i·p_i` + `legend` + 概率 + `score_confidence`（`shim.py:228-235`） |
| `noul` | 固定两项 `yes` / `no` | 一个标量：校准后的「是」的概率（`shim.py:237-242`） |

`noul` 是三者里唯一**再做一步校准**的，也是唯一改动了分布的题型（`shim.py:241`）：

```python
py = min(max(p[0], 1e-4), 1 - 1e-4); z = math.log(py / (1 - py)) / NOUL_T + NOUL_BIAS  # the raw readout over-says yes on skewed sets
return {"type": "noul", "noul": round(1 / (1 + math.exp(-z)), 4)}, tokens
```

也就是：先把 `p_yes` 夹到 `[1e-4, 1−1e-4]`（防止 logit 发散），转成 log-odds，
**除以斜率 `NOUL_T`、加上偏置 `NOUL_BIAS`**，再 sigmoid 回概率。注释给了理由：
*the raw readout over-says yes on skewed sets*——原始读法在「是/否 严重不平衡」的数据上高报「是」。

`shim.py:23` 的注释还记了拟合口径：*yes/no calibration fit on 19 boolean sets, checked on 18
(results/noul_calibration.json)*。那份 `results/noul_calibration.json` **不在本仓库里**——
这个仓库只有模型、`helper/`、`serve/` 和 `assets/`，没有 `results/` 目录。

`score` 的 `expected value` 也是同一类「档位序号有意义」的假设（§6）。

注意 `choice` 与 `score` 的 `criteria` 在**结构上就是不同的**（一个 map、一个数组），
传到 `_instr`/`_desc` 之后才变成文本；`serve/SERVE.md:173` 明确列了一条已知行为：
*noul criteria sent as a list*……会「关掉连接而不返回 JSON 错误体」。也就是说
**结构错误的 `criteria` 不保证有 422**——只有代码显式检查的那几类才有（§12）。

## 11. agent 形态：state 压缩、图片通道与 `loop_break`

这条路线有一块别家没有的东西：它假定 state 是**浏览器的 DOM / 操作序列**，并为此写了三段预处理。

### 11.1 `compact()`：丢几何与遥测字段

`shim.py:36-62`。`DROP_KEYS` 是几何 / 标识 / 遥测字段的白名单式黑名单（`shim.py:38-39`），
注释给的第二个理由同样指向缓存：*the telemetry changes every send, which defeats prefix reuse across steps*。

去重的规则比看上去保守（`shim.py:56-58`）：

```python
ident = next((str(e[k]) for k in ID_KEYS if e.get(k) is not None), None)
key = (e.get("tag"), (e.get("text") or "").strip(), ident)
if ident is not None and key in seen: continue
```

**只有在元素带 id 字段时才去重**（*an element without any id field is never deduped, so distinct
same-label controls survive*，`shim.py:45-46`）——两个都叫「Delete」但没有 id 的按钮是两个选项，
不能合并。然后只保留交互元素或有文本/标签/aria/role 的元素（`shim.py:59`）。

`SHIM_COMPACT_CAP`（`shim.py:37`）是唯一**有损**的一档，注释直接说明它是 off：
*a lossy cap, off unless a held-out check on real states proves it neutral*。

### 11.2 `page_first()`：把不变的字段放前面

`shim.py:64-67`：`STABLE_FIRST = ("page", "elements", "goal", "task", "plan")` 提到最前，
其余（历史、遥测）放最后。注释写明动机是 *so consecutive steps of a task share the longest prefix*，
并且提醒 **这是一次对训练布局的改动**，上线前要在 web-300 上查（*a change of the trained layout:
web-300 check before go-live*）。

### 11.3 图片通道：`State` 是个 `str` 子类

`shim.py:32-34` 与 `69-80`：

```python
class State(str):
    """The state as text, plus an optional screenshot (data URL) that rides along to the model as an image."""
    image = None; fields = None
```

`with_image()` 从 dict 里摘出 `screenshot` / `image`（认 `data:image` 前缀，或长度 >2000 的裸 base64），
把**剩下的字段**序列化成文本，把图片挂在属性上。于是后面所有函数都照常把 state 当字符串用
（`prefix_text(f"State:\n{state_text}")`），只有 `_readout_once` 会 `getattr(state_text, "image", None)`（`shim.py:133`）
来决定走哪套 prompt。

**有三套 prompt**（`shim.py:125-146`），而它们的可靠性并不一样：

1. **截图 + `state.task`**：走「训练过的截图布局」——任务、历史动作、**被红字母标记的候选**、
   问题（`shim.py:139-143`）；
2. **截图 + harness 形状**（`page`/`elements`/`recent_actions`）：同一布局但没有标记，
   docstring 明说 *the image lane never trained on this shape … so it needs the matched check before
   it is relied on*（`shim.py:129-131`）；
3. **无图，或其它形状**：文本布局 `State: … / Question: … / Options: …`（`shim.py:145-146`）。

这条「画面上用红字母标出候选元素」的截图布局，是这条路线里唯一**把选项和像素对齐**的机制——
文字通道靠 `[A] …` 标号，图片通道靠画在画面上的红字母。`serve/SERVE.md:175` 把第 2 种情况
也列进了 *Known helper behaviour*。

### 11.4 `loop_break()`：一个默认关闭的、项目自己承认是「benchmark 专用」的补丁

`shim.py:246-268`。当最近三次动作的 `(action, kind)` 完全相同且**页面没有变化**时，
把那个动作从候选里剔掉（`shim.py:264`），免得模型在同一页上无限重复同一个点击。
两条底线写在代码里：连续重复 ≥4 次才连 `kind` 也剔（`shim.py:263`），
**且剔完必须还剩至少两个候选**（`shim.py:264`）。

它默认关闭，理由是所有者自己的判断（`shim.py:246`）：

> *OFF unless set: the owner classed it a benchmark-specific patch, not model behaviour.*

这是一个很少见的东西：**一个为了刷分而写的补丁被显式标注成「不是模型行为」并默认关掉**。
与 `docs/comparison.md` §4.5 里 Laya 那条「接了没收益」是同一类诚实记录，但方向相反——
这里是有收益、但项目自己不认。

## 12. 服务化 / fail-loud 风格

服务是**一个标准库 `ThreadingHTTPServer`**（`shim.py:280-354`），不是 FastAPI。
`serve/SERVE.md` 是它的完整说明书，而这份说明书本身值得单独看：它把每一次「不安全默认」都摊开写。

### 12.1 四条路由

| 路由 | 干什么 |
|---|---|
| `POST /v1/systemone` | 主接口：一个 state + 一组问题（`shim.py:304-327`） |
| `GET`/`POST` `/v1/version` | 返回 served model dir、`T`、noul 两个常数、**全部 flag**、补丁文件自己的 sha256（`shim.py:292,332,339-346`） |
| `POST /v1/prewarm` | 只读 state，预热缓存（`shim.py:298-303`） |
| `POST /v1/chat/completions` | 透传给 vLLM，强制关 thinking（`shim.py:293-297,271-278`） |

`/v1/version` 是这条路线最值得抄的一处。它把「我到底跑的是什么」变成一个**可远程读取的对象**：

```python
return {"model_dir": model, "T": TEMP, "noul_t": NOUL_T, "noul_bias": NOUL_BIAS, "flags": flags,
        "shim_file": os.path.basename(__file__), "shim_sha256": hashlib.sha256(open(__file__, "rb").read()).hexdigest()}
```

而且同一个摘要串会被写进**每一条回答**的 `model` 字段（`shim.py:347-348`）：

```
"<model dir name> T=0.85 noul=1.829074,0.0 flags={…} shim=shim.py@81a22f1b1b89"
```

所以一份保存下来的响应**自带它的出处**：模型目录名、两个标定常数、全部 flag、补丁文件的哈希前 12 位。
配合 `serve/SERVE.md:54` 给出的补丁文件完整 sha256，事后可以判断「这个数字是不是这份补丁跑的」。

### 12.2 fail-loud 的边界在哪

**显式报错的部分**：

- `ReadoutIncomplete`：TARGETED 下任何一个候选分数缺失或非有限，直接抛（`shim.py:22,154`）。
  **但它没有对应的错误响应**——`/v1/systemone` 的分发只包了 `except BadQuestion`（`shim.py:321-322`），
  而 `ReadoutIncomplete` 是 `RuntimeError` 的子类（`shim.py:22`），会一路穿出 `do_POST`、连接被关掉。
  也就是说：**TARGETED 的「fail loud」在代码层是真的（缺分绝不兜底），但 loud 的形式是断连接而不是 502。**
  `serve/SERVE.md:173` 把这一类统一成一句 *closes the connection without a JSON error body.
  Treat a dropped connection as a failed request*；
- `BadQuestion` → `422`：`instructions` 缺失（`shim.py:184`）、`choice.criteria` 不是非空 map（`shim.py:222`）、
  `score.criteria` 不是长度 ≥2 的数组（`shim.py:230`）、`type` 不在三种里（`shim.py:315-316`）、
  `state` 或 `questions` 缺失（`shim.py:305-306`）；
- `/v1/version` 的 `shim_sha256` 在**进程启动时**算一次（`shim.py:346`）。

**明确不报错的部分**（`serve/SERVE.md:169-175` 自己列的）：

- *Malformed JSON, a body that is not a JSON object … or a failure of the model server closes the
  connection without a JSON error body. Treat a dropped connection as a failed request.*
  ——**连 JSON 解析失败都没有错误响应体**；
- *No caps are enforced on option count or score levels* ——52 与 16,384 是**测过的界限，不是拒收的界限**。
  `/v1/systemone` 不会因为给了 200 个选项而拒绝你，它只会去走 §9 那套近似；
- 四舍五入到 4 位之后**概率可能不等于 1**（`shim.py:225,233-235`），`serve/SERVE.md:172` 提醒
  *with a near-tie, check `probabilities[choice]` against the maximum with a tolerance*。

### 12.3 认证：一个有默认值的开关，以及说明书对它的诚实

`SHIM_TOKEN`（`shim.py:20`）设了才校验，四处路由**都要** `Authorization: Bearer <token>`
（`shim.py:288-289,330-331`）。空 / 未设时**完全没有检查**。

`serve/SERVE.md:94-100` 这一节把话说透了，值得整段引用要点：

- *When it is empty or unset there is no check at all. The helper does not refuse to start on `0.0.0.0`
  without a token: it starts normally and serves every request from anyone who can reach the port,
  including the chat pass-through, which is open access to the model. The requirement is yours to enforce*；
- 检查是 *a plain string comparison with no rate limiting: treat the token as a gate, not as hardened
  authentication*；
- 明文 HTTP，*the token and every state you send travel unencrypted*；
- **认证补丁不等于保护后端**：*The token guards the helper's port only. The vLLM backend on port 8000
  has no authentication in this setup*（`serve/SERVE.md:100`）。

命令里那一行 `SHIM_TOKEN="${SHIM_TOKEN:?…}"` 的写法也有解释（`serve/SERVE.md:90`）：
它同时做两件事——token 没设时在 Python 启动前就中止，以及把 token 真的交给补丁进程。
理由是 *a token that is set in your shell but not exported passes such a guard, while the helper sees
no token and serves unauthenticated*。这不只是文档风格，是被测试过的（*Tested with a stub in bash and zsh*）。

## 13. 置信度从哪来，又怎么被改写

### 13.1 链条：训练 + 三个常数

与另外四条路线不同，这条链上**第一环是训练本身**。`README.md:191` 说模型被训成「这个位置携带决定」，
`README.md:104-109` 给出训练改 argmax 的直接证据（shuffle 翻转率 18.5% → 2.3%）。

之后在**发布路径上**改数值的环节，按代码顺序：

| 环节 | 在哪 | 改 argmax？ |
|---|---|---|
| ① 训练本身（微调 Qwen3.8-27B） | 不在本仓库（只有权重） | **是** |
| ② `logprob / TEMP` | `shim.py:155`（TARGETED）/ `shim.py:164`（旧协议） | 否（单调） |
| ③ `softmax(z)` | `shim.py:165-166` | 否 |
| ④ `noul`：`logit(p_yes)/NOUL_T + NOUL_BIAS` → sigmoid | `shim.py:241` | 否（单调），但**只对 noul 生效** |
| ⑤ `round(·, 4)` | `shim.py:225,233-235,242` | 理论上可以在极近并列时改（`serve/SERVE.md:172` 提醒过） |
| ⑥ `READOUT_PERMS > 1` 时对多个 lettering 取平均 | `shim.py:174-177` | **是**（会改 argmax），**发布配置为 1、未启用** |

②③④ 都是单调变换，所以**能改 argmax 的只有 ①、⑤、⑥**；其中 ⑥ 没开、⑤ 只在四舍五入的边界上有影响。

### 13.2 发布值 ≠ 文件默认值

这是本篇最实用的一张表。`serve/SERVE.md:58-69` 逐行列出「这里用的值 / 文件里的默认值」，
每一个都对应 `/v1/version` 会报出来的东西：

| 变量 | 发布值 | **文件默认值** | 作用 |
|---|---|---|---|
| `READOUT_T` | **0.85** | **1.1** | choice / score 的温度 |
| `READOUT_NOUL_T` | **1.829074** | **3.0** | noul 的斜率 |
| `READOUT_NOUL_BIAS` | **0** | **−0.4** | noul 的偏置 |
| `READOUT_TARGETED` | **1（开）** | **未设（=关）** | 按 id 取精确分数 vs top-K 兜底 |
| `READOUT_INSTR_STYLE` | **pyrepr** | **未设（=JSON）** | 对象型 `instructions` 的写法 |
| `SHIM_STAGGER` | **1（开）** | **未设（=关）** | 长 state 先让第一题独占一拍 |
| `READOUT_PERMS` | 1 | 1 | 同上 |
| `SHIM_POOL` | 16 | 16 | 并发题数 |
| `SHIM_COMPACT` / `_CAP` / `SHIM_LAYOUT` / `SHIM_PAD` / `SHIM_LOOP_BREAK` / `SHIM_MODEL` | 全部未设 | 全部关 | §8、§11 |

**直接 `python helper/shim.py` 跑起来的是右列，不是左列。** 三个数值常数全都不同，
而且 `READOUT_TARGETED` 一关就换了读数协议（§5）。`serve/SERVE.md:4` 把这件事说在最前面：

> *Every number in the model card was measured with exactly the flags and settings below. Change one and
> the numbers no longer apply.*

`serve/SERVE.md:62` 还解释了 `TOKENIZER` 的双重身份：它既是查字母 token id 的分词器，
**它的目录名又被当成「served 模型名」写进 `/v1/version`**（`shim.py:343`）——
所以本地目录叫 `openjev` 还是 `openjev-fp8`，会直接出现在每条回答的 `model` 串里。

### 13.3 一处自相矛盾，以及一处过期

**矛盾**：`READOUT_INSTR_STYLE=pyrepr` 的说明是「instructions 是对象或数组时，按模型训练时的
Python 字面量写成文本」（`shim.py:186`、`serve/SERVE.md:66`）。但 `_instr()` 的实现是
`str(i)`（`shim.py:186`）——对 dict 来说 `str()` 恰好等于 Python 字面量文本，对含非字面量对象的
输入则不然；而 `_task()` 那一侧反过来用 `ast.literal_eval` 去**解析**它（`shim.py:118-123`，
注释：*literal_eval: literals only, never code*）。两边是同一套约定的读与写，代码里没有 enforce。

**过期**：`MANIFEST.json` 里 `README.md` 记的是 12158 字节 / `3753084b…`，
而仓库当前的 `README.md` 是 **13007 字节 / `bff7f997…`**。`SHA256SUMS:4` 记的也是旧的
`3753084b…`。（实测：对仓库内 30 个文件逐个算 sha256。除 `README.md` 与 17 个 LFS 指针文件外，
其余 **12 个**——含 `helper/shim.py`、`serve/SERVE.md`、`chat_template.jinja`、`config.json`、
`generation_config.json`、`processor_config.json`、`tokenizer_config.json`、`NOTICE` 与
两份 LICENSE，以及唯一不进 LFS 的图片 `assets/hero_stats.png`——与 `MANIFEST.json` **逐字节相符**。）所以这是**卡片被改过、清单没跟着重算**，
不是内容被篡改：`git cat-file -s HEAD:README.md` 也是 13007。

## 14. 四条不变量的落位

| RYOTIDE §13 不变量 | OpenJev 的落位 |
|---|---|
| **#1** 读位置必须落在标记上（诊断量 `marker_mass`） | **结构性成立，但没有诊断量**。位置由微调固定（`README.md:191`），运行期不扫、也不上报落在哪。代码里唯一相关的东西是 `assert len(set(allowed)) == len(allowed)`（`shim.py:150`）——那是**槽位唯一性**自检，不是位置诊断。防线是每一处调用都传 `enable_thinking: False`（§3），而模型的 chat 模板在不传时默认开 `xhigh` 推理 |
| **#2** 前缀与标记表面形式耦合，必须成对扫 | **不适用**：没有 `SUFFIXES`/`PATTERNS` 这个概念，标记是 token id（`letter_ids()`，`shim.py:25-30`）不是字符串表面形式。但它有一个同类的耦合：**字母 token 与分词器绑定**，所以 `TOKENIZER` 既决定读哪个 id、又决定 `/v1/version` 报什么名字（§13.2），换成另一个分词器就要整条重来 |
| **#3** 只读最后一个位置 | **改写**：读的是**第一个输出位置**（`max_tokens=1`，`shim.py:151,157`），不是输入的末位。这是这条路线与 RYOTIDE 最本质的一处不同——RYOTIDE 读 prompt 末尾那一步的分布，这里读「生成第一个 token 时」的分布，两者在因果 LM 里是同一个位置，但**这条路线是靠微调把语义搬到那个位置上的**。同时「不物化 `[1, n_tokens, vocab]`」不适用：词表维度由 vLLM 内部处理，补丁只拿 top-K 或指定 id 的分 |

| **#4** 比较概率 / ECE 前先确认 temperature | **成立，且比别家更硬**：T、noul 两个常数、**全部 flag**、补丁文件的 sha256 都在 `GET /v1/version` 里（`shim.py:339-346`），并且**每一条回答的 `model` 字段自带同一串摘要**（`shim.py:347-348`）。前提是调用方真的去读它——`serve/SERVE.md:167` 提示 *use it to pin what you measured* |
| （新增）**`output_tokens = 0`** | **不成立**：对 vLLM 发的是 `max_tokens: 1`（`shim.py:151,157`），确实采了一个 token；响应里的 `"output_tokens": 0` 是**硬编码的**（`shim.py:325`）。读的不是那个 token，但「零输出 token」这句话在这条路径上是**自报值，不是事实** |
| （新增）**发布的常数必须与发布的读数配对** | **代码默认值不等于发布值**：三个数值常数（T、noul 斜率、noul 偏置）与两个行为开关（`READOUT_TARGETED`、`READOUT_INSTR_STYLE`）的默认值与发布值全都不同（§13.2）。而且 T 与 noul 斜率是**在 TARGETED 打开时拟合的**，混用即为错配（`serve/SERVE.md:65`） |
| （新增）**仓库自带的清单要与仓库当前内容一致** | **不成立**：`MANIFEST.json` 与 `SHA256SUMS` 里的 `README.md` 条目是过期的（12158B/`3753084b…` vs 实际 13007B/`bff7f997…`）；其余 12 个非 LFS 文件相符（§13.3）。**引用卡片上的数字前，先确认清单是否覆盖到它** |

## 15. 文件对照

| 文件 | 行数 | 职责 |
|---|---:|---|
| `helper/shim.py` | 354 | 全部决策逻辑：52 字母读法、两条读数协议、三种题型、两个 confidence 公式、>52 分块、state 压缩、prewarm/stagger、HTTP 路由与鉴权 |
| `helper/shim_mlx.py` | 51 | **不改 `shim.py` 一个字节**，只替换 `shim.client`，把 TARGETED 那一次调用接到 MLX 上（`shim_mlx.py:39-41`） |
| `serve/SERVE.md` | 175 | 发布配方：锁定的包版本、vLLM 命令、补丁环境变量表、鉴权与暴露端口的说明、已知行为清单 |
| `chat_template.jinja` | 169 | 模型的 chat 模板。`enable_thinking` 的三态与两个 `<think>` 形态都在这里（`:46-56`、`:163-169`） |
| `config.json` | 146 | 架构：`Qwen3_5ForConditionalGeneration`、64 层、`full_attention_interval: 4`、`mamba_ssm_dtype: float32`、词表 248320、`max_position_embeddings: 262144`（`:16,22,88,93-99,118`） |
| `generation_config.json` | 13 | 采样默认（`temperature 1.0` / `top_k 20` / `top_p 0.95`）——**补丁完全不使用它**，每次都显式传 `temperature=0` |
| `README.md` | 199 | 模型卡：结果表、格式表、请求上限、"How it works, in one paragraph"（`:191`）。**本文只当线索用** |
| `MANIFEST.json` / `SHA256SUMS` | 122 / 30 | 逐文件 sha256。两者对 `README.md` 的条目都过期（§13.3） |
| `model.safetensors.index.json` + 12 个分片 | — | 权重（约 54 GB，BF16）。本仓库只留 LFS 指针，未下载 |
| （无 `results/`、无训练代码、无 tokenizer 训练脚本） | — | `noul` 的拟合结果 `results/noul_calibration.json` 只被引用（`shim.py:23`），**不在这个仓库**；微调过程本身也不在这里 |

## 16. 亲手跑一遍

### 16.1 官方路径（需要一张 80 GB 卡）

```bash
pip install "vllm==0.29.0" "openai==3.16.2" "httpx==0.28.1"      # README.md:147；版本锁定见 serve/SERVE.md:8-15
hf download openjev/openjev --local-dir openjev                  # README.md:148

# 1. 模型
vllm serve ./openjev --host 127.0.0.1 --served-model-name qwen --port 8000 \
  --enable-prefix-caching --max-model-len 16384 --gpu-memory-utilization 0.90 \
  --limit-mm-per-prompt '{"image":1}' --trust-remote-code --max-num-seqs 256 \
  --max-logprobs 64 --gdn-prefill-backend triton --quantization fp8   # README.md:151-154

# 2. 决策 API
VLLM=http://localhost:8000/v1 TOKENIZER=./openjev \
READOUT_T=0.85 READOUT_NOUL_T=1.829074 READOUT_NOUL_BIAS=0 \
READOUT_TARGETED=1 READOUT_INSTR_STYLE=pyrepr SHIM_STAGGER=1 \
python openjev/helper/shim.py --host 127.0.0.1 --port 3000        # README.md:157-160
```

跑完先读一次版本，确认自己跑的确实是这套常数与 flag：

```bash
curl -s http://localhost:3000/v1/version        # shim.py:332 → {model_dir, T, noul_t, noul_bias, flags, shim_sha256}
```

三个 flag 值得单独核对——`serve/SERVE.md:35-38` 说明其中三个是**补丁的硬依赖，不是调参**：
`--served-model-name qwen`（补丁按这个名字调模型）、`--max-logprobs 64`（一次请求要拿最多 52 个字母的精确分）、
`--max-model-len` 与 `--limit-mm-per-prompt`（就是卡片里「16,384 token / 一张图」这两个上限的来源）。

### 16.2 本机实测得了的部分

本文只跑过一件与这个仓库有关的事，而且它不执行上游代码：**文件哈希核对**。

```bash
cd openjev
python3 - <<'PY'                     # 只算 sha256，不 import 任何上游模块
import hashlib, json, pathlib
m = json.load(open("MANIFEST.json"))
for n, v in sorted(m.items()):
    p = pathlib.Path(n)
    print("ok " if p.exists() and hashlib.sha256(p.read_bytes()).hexdigest() == v["sha256"] else "不一致", n)
PY
```

结果就是 §13.3 那一段：12 个文件相符；`README.md` 与 17 个 LFS 文件不一致——
其中 17 个是**本机没拉 LFS**造成的，不是上游问题（`model-00001-of-00012.safetensors` 在本机是 135 字节的指针文件），
只有 `README.md` 是上游自己的清单过期。

**本机跑不了的部分**：`helper/shim.py` 需要 vLLM + 54 GB 权重 + 一张 80 GB 卡；
`shim_mlx.py` 需要 MLX 与 MLX 权重。二者本机都不可得。所以**本文不给出任何由这条路线前向产生的
概率、准确率或 token 数**，卡片上的数字一律标注来源行号。

`docs/jeff-serve/` 那篇先例里的边界也适用于此：**本文没有执行这个仓库的任何代码**——
导入 `helper/shim.py` 就会在模块层实例化 `OpenAI(...)` 并读环境变量，那是运行上游程序，不是读它。

### 16.3 一处无法独立复算的东西

`shim.py:19` 的注释 *4 ≈ +16 pts acc, 4× cost* 与 `shim.py:23` 的 *fit on 19 boolean sets,
checked on 18 (results/noul_calibration.json)* 都指向**不在本仓库**的产物。前者没有任何路径可追，
后者指向的 `results/` 目录在这个仓库里不存在。**本文按原样引用注释，不把它当成可核对的实测。**

## 17. 本文的事实来源，与本文不做什么

| 类别 | 判断方法 | 本文标记 |
|---|---|---|
| 代码引用 | 直接读 `helper/**`、`serve/**`、`chat_template.jinja`、`config.json` | `文件:行` |
| 可实跑得到的产物 | 不执行上游代码即可完成的核对：文件 sha256 与 `MANIFEST.json` / `SHA256SUMS` 对照 | 标「实测」 |
| 公式与常量 | `README.md`、`serve/SERVE.md`、`MANIFEST.json` | `文件:行` |
| 前向产生的数值 | 本机无 vLLM、无 MLX、无权重 | 一律只标卡片或 README 的行号，不给自己的数值 |

**本文不做的四件事**：

1. **不执行这个仓库的任何代码**。`helper/shim.py` 在模块层就实例化客户端、读环境变量；
   `shim_mlx.py` 会加载 MLX 权重。两者都不是「读代码」。因此 §5 里 `e^(−30/0.85)` 那类量
   是**把源码里的常数代进公式的手算**，不是实测；凡是手算的地方都写明了。
2. **不给任何前向产生的数值**。§16.2 的命令本机跑不了，卡片上的 84.0% / 2.3% / 210 ms
   等全部标注 `README.md` 的行号，按行回源。
3. **不把上游文档当结论**。`README.md` 与 `serve/SERVE.md` 是本篇最重要的**线索**
   （它们指了该去看哪些代码行），但凡本文写下的机制结论都回到了 `helper/shim.py`、
   `chat_template.jinja`、`config.json` 的行号。两者冲突处以代码为准——§13.3 那两处就是例子。
4. **不推断三套同名代码的历史关系**。`docs/open-jev/`（Zefan-Cai）、`docs/semif-openjev/`
   （TheoLeeCJ）与本篇（HF `openjev/openjev`）是**可核对的三套不同代码**；
   至于它们之间是否有 fork、改名或人员重叠，**本文不推测**。

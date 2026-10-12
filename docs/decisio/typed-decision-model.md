# decisio 是如何把冻结的基座读成字母分类器的

> 代码库：<https://github.com/aminry/decisio>
> 对应 `README.md` 第 12 行：*Decisio is an open-source server that answers typed questions about any text,
> with a calibrated probability for every option, from one forward pass and no generated text.*
> 本文按 `src/decisio/`（读数、引擎、服务、基座画像、vLLM 插件）的实际代码逐环节说明这句话的实现方式；
> 文中出现的文件名与行号（`letters.py:73`、`vllm_engine.py:539`、`systemone.py:600`、`families.py:153` 等）
> 均相对该仓库根目录。
>
> **同名配对文档**：RYOTIDE 的 [`docs/ryotide/typed-decision-model.md`](../ryotide/typed-decision-model.md)。
> §0 先讲清分歧，§11 把它的四条不变量逐条落位。

decisio 是这条线上的**零训练**一侧（与 RYOTIDE、SemIf-OpenJev 同类，但比它们都更「服务化」）：
读法与 RYOTIDE 同源——把基座的输出层当分类器，一次前向、取答案位的 logits、只在 K 个标签行上
softmax。它往前推的是四件事：

1. **标签集不是一张固定表，而是请求时从分词器推出来的**：`letter_labels(tok, K)` 取前 K 个
   「加前导空格后恰好单 token」的字母码（`A`…`Z`、`AA`…`ZZ`），所以 K 是**每题一个数**
   （committed 记录里 K 从 3 到 151 都出现过），上限 255 是**线格式的 8 位选项下标**而不是分词器的
   上限（§2）；
2. **同一条读法服务三个基座与两套被服务的布局**：布局是 `PromptFormat` 四个字段（`tail` / `slot` /
   `variants` / `system_prompt`）加逐 base 画像——Gemma 两代共用一套（**系统提示 + 模板自带的答案位 +
   同一标签的多种单 token 形式求和 + 不填充**），Qwen 一套（**`Answer:` 预填 + 一标签一 token +
   块对齐填充**）；另有一个**历史布局** `compact` 保留给在它下面拟合过的任务（§7）；
3. **多问题共享 state 做成了三件可观测的机件**：前缀按 token 空间切、按块边界填充、答案发出后再把
   state 的边界注册进缓存（§8）；
4. **报出去的 `confidence` 是读出来的**，但链条上有五个先后改写它的东西，其中**四个是可选、默认关的**
   （三个要显式注册、一个要显式开 flag），默认生效的只有逐题型温度（§10）。

先说三句结论，后面逐条给代码位置：

- 「typed」来自**分词器加一条运行期断言**：`_labels_for` 对每个候选码断言
  `len(tok.encode(" " + code)) == 1` 且 token id 不与更早的标签相撞，凑不够 K 个就**直接抛**
  （`letters.py:57-70`）——标签集不写死在文件里，也不来自请求，而是**同一次请求里现算**（§2）；
- 读位置是**构造出来的答案位**：Qwen 基座上 prompt 以字面量 `Answer:` 结尾（`letters.py:149`
  有断言），Gemma 基座上读的是**聊天模板自己的第一个助手位置**（`answer_slot="template"`，
  `families.py:141`）；运行期**没有任何位置标定**，也没有可调的候选位（§3）；
- 掩码**不由代码做**，而是交给采样器的 `allowed_token_ids`（`vllm_engine.py:543-544`）：返回的
  K 个 logprob 在构造上就是这 K 个标签行上的 log-softmax，所以 `marker_mass` 在这条链上
  **没有对象**（§5）。

## 本文的事实来源

| 类别 | 来源 | 本文中的标记 |
|---|---|---|
| 代码引用 | `src/decisio/**`（`readout/`、`serve/`、`vllm_plugin/`、`bench/`、`families.py`、`hub*.py`、`names.py`）与 `benchmarks/` | `文件:行` |
| 可实跑得到的产物 | `runs/**` 里 **committed 的存盘记录**：Decision Index 的逐题 `items.jsonl.gz`、三道一致性门的输出、三道 README 实例走查的响应、两份 JevBench 规模的分析清单、`manifest.json` 的 sha256 清单；`tests/data/intent_readouts/*.npz` | 标注「实测」，并注明「在 committed 产物上重算」 |
| 公式与常量 | `README.md`、`docs/api.md`、`docs/running.md`、`docs/design/*.md`、`EVAL_CARD.md`、committed 的 `runs/*/manifest.json` | `README:行`、`docs/...:行` |
| 未跑 | **本机没有 GPU、没有下载任何 `Qwen/Qwen3.6-*` 或 `google/gemma-4-*` 权重、没有装过 `decisio`、没有起过服务、没有跑过 pytest、也没有那两份分词器** | 照例不给任何前向产生的概率数值，也不给「255 个标签里到底有几个能单 token 化」这类需要分词器的计数 |

本文所有「实测」只有三种动作：**读 committed 的 JSON/JSONL，用标准库（`json` + `math`）算术重算**、
**对 committed 文件做 sha256 对照**、**读 committed 的 `.npy` 头部**。没有运行上游的任何程序——
没有 `import decisio`、没有 `python -m decisio.serve.vllm_engine`、没有跑 `benchmarks/*.sh`。
命令在 §13、§14 原样列出，并标明哪些**本文没有执行**。

---

## 0. 与 RYOTIDE 的分野

| 维度 | RYOTIDE | decisio |
|---|---|---|
| **权重来源** | 公开 LLM，全程 frozen | 公开基座，**全程 frozen**，且**逐 base 钉住 revision**（`families.py:101`、`:120`、`:157`）；31B 在加载期量化到 FP8（`families.py:177`），也可换成一份已存 FP8 的 decisio 仓库（§15） |
| **是否训练** | 否，零新增参数 | **否，零新增参数**——主路径一个字都不训。仓库里有三处**可选、逐任务、按客户标注样例拟合**的东西：per-task 先验 bias（`readout/calibration.py`）、intent head（`readout/intent_head.py`）、abstention 阈值（`serve/abstention.py`），三者都要 `POST /v1/tasks` 或 `POST /v1/abstention/tasks` 显式注册才存在（§10） |
| **决策机制** | 读**标签 token** 的 logits → 掩码 → softmax | 同源：读**答案位**的词表 logits，但**只把 K 个标签 id 交给采样器**（`allowed_token_ids`），返回的就是 K 个 logprob → softmax（`vllm_engine.py:539-568`）（§4、§5） |
| **「typed」从哪来** | 分词器：标签必须单 token，否则映射成字母标记 | 分词器，但**不是一张固定表**：`letter_labels(tok, K)` 每次现取前 K 个能加空格单 token 的字母码，`MAX_LABELS = 255`（`letters.py:28`）；凑不够 K 个**抛异常**而不是降级（§2） |
| **读位置怎么定** | **运行期标定**：扫 9 前缀 × 2 表面形式，取 marker mass 最高 | **构造期固定**：Qwen 基座是字面量 `Answer:` 之后的那个位置（`letters.py:149` 断言 prompt 以它结尾），Gemma 基座是聊天模板自己的第一个助手位置（`answer_slot="template"`，`letters.py:285`）。运行期**没有标定、没有扫描、没有候选位**（§3） |
| **读出来的是什么** | 末位**词表 logits**，再按标记 id 取出 | **只有 K 个 logprob**：`allowed_token_ids` 是请求参数，`logprobs_mode="processed_logprobs"`（`vllm_engine.py:277`），所以词表其余部分的概率**从不进入任何读数**（§4） |
| **候选数上限** | 260（`A0`–`Z9` 两 token 两步读；>26 另有 `grouped` 近似） | **255**，但性质不同：它是 `MAX_LABELS`（`letters.py:28`，注释写明「the wire format's cap, an 8-bit option index」），而**真正会先撞上的是分词器**——同一个 tokenizer 里前 K 个单 token 字母码到哪为止，是运行期算出来的（§2）。packed 模式另有 77（`vllm_engine.py:235`） |
| **顺序偏置** | 多 order 共享前缀 fork，折叠回真实标签后**概率平均** | **默认不处理**（`--orders 1`）。可选的 `--orders 2`（after Reflex）把每题读**一个**不同的第二序再平均， disagreement 只进分支日志（§6）。另有一条**与顺序无关**的先验修正：把标签池按无内容 prompt 下的先验排序，使任意前 K 个尽量接近均匀（`letters.py:102-114`）——但它在本仓**零调用点**（§6 末） |
| **置信度来源** | **读出来的**：只有 temperature 能改数值 | **读出来的**，且因为零训练，温度在构造上不可能改 argmax（§14 在 20,810 条 committed 读数上复算翻转 **0 次**）。链条上的五个改写者见 §10 |
| **诊断量 / 守卫** | `marker_mass`（全词表落在标记上的份额） | **没有等价量**（§5）：返回的向量在构造上就是重归一化的。守卫搬到了**加载期与一致性门**上：保留 id 不许撞标签（`vllm_plugin/hidden.py:59-67`）、adapter 张量名必须能被映射（`vllm_engine.py:138-156`）、三道一致性门（C2/C3/C4，`serve/systemone_conformance.py`）、仓库画像逐键比对（`hub.py:194-197`）（§9） |
| **多问题共享 state** | `run_many()`：公共前缀预填充一次，计费只算一次 | **做，而且拆成三件可观测的机件**：前缀在 token 空间切（`state_prefix`）、按块边界填充（`--pad-to`/`--pad-policy`）、答案发出后把 state 边界注册进缓存（`--register-boundary`）。另有 packed 模式走**另一条**共享路径（一次 prefill 里读每个轮次的答案位）（§8） |
| **后端** | MLX（参考）+ torch | **vLLM 0.30.0（默认）+ MLX + 一个纯 HF 的 CPU 冒烟替身**；vLLM 路径还分 separate / packed 两种模式（§8、§9） |
| **服务接口** | `/v1/systemone` + `/health`（含 `prompt_hash`），另有 MCP | `/v1/answer`（原生）+ TypeSafe 的 `/v1/systemone` + `/v1/models` + `/health` + 六个任务端点（`POST`/`GET` `/v1/tasks`、`POST /v1/tasks/import`、`DELETE /v1/tasks/{id}`、`POST`/`GET` `/v1/abstention/tasks`，`systemone.py:838-1094`）。**两条路由对同一个问题给的是不同的数**——`/v1/answer` 是**未加温度**的原生读数，`/v1/systemone` 是加过温度的（§9.3、§15.2） |

一句话总结立场：**decisio 把「读位置」和「标签集」都从「训练期固定」推到了「构造期算出来 + 加载期断言」，
然后把省下的自由度全部投到服务化上**——前缀缓存、填充策略、边界注册、三道一致性门、一个把「当前
服务的是什么」整体打印出来的 `/health`。

---

## 1. 端到端数据流

```
POST /v1/systemone                       systemone.py:906-944
  ├─ 读 body：JSON 或 multipart；state 里内嵌的 data:image 被抽成图片并换成 "[image N]"   :861-904
  ├─ pydantic 校验 SystemOneRequest（state + questions，题目按 type 判别联合）            :75-81
  └─ run_in_threadpool(SystemOne.answer)                                                  :929
       │
       ├─ 每题 to_engine_question / noul_as_letters → 引擎题目 dict + keys                :129-181 / :190-208
       │    · choice：选项按请求 key 顺序，渲染规则三层（hide_index_keys → describe_options → key: desc）
       │    · noul：默认 words（"Yes means:" / "No means:" 两行），letters 则变成一个两选项 choice
       │    · score：档位当选项列出，字母做图例
       │
       ├─ 每题 readout_task(q)：按 task_key 查已注册任务（先验 bias / intent head）        :658-663
       │
       ├─ engine.answer(state, questions)  ← 见下                                        vllm_engine.py:392
       │
       ├─ 逐题：交换回 (yes,no) → 两序平均 → 任务修正 → 温度 → 拒答 → commit 带 → to_answer  :515-587
       │
       ├─ usage：input_tokens = state + 每题自己的文本；output_tokens = 题目数             :598-601
       └─ _timing 里的 stages 进 `x-decisio-stages` 头                                    :606-621 / :939-941

engine.answer → answer_many                                                              vllm_engine.py:401-452
  ├─ health.check()：引擎死了直接拒，不碰锁                                              :409-410
  └─ 拿锁 → registrar.run_due()（把到期的边界注册先发掉）→ 分派                        :423-435
       │
       ├─ separate（默认）：每题一行 token，一行一次引擎请求                                :635-692
       │    · _prepare_separate：state 只分词一次，题文本单独分词后拼                           :472-515
       │    · pad_prompt：把共享前缀顶到块边界（Qwen）或什么都不做（Gemma）                    :118-131
       │    · score_prompts：SamplingParams(max_tokens=1, temperature=0.0,
       │                      logprobs=K, allowed_token_ids=<K 个标签 id>, detokenize=False)  :539-549
       │    · 返回的 K 个 logprob → softmax → 概率                                             :562-568
       │
       └─ packed：把多题拼成一条 prompt 的连续聊天轮次，用 vLLM 的 pooling runner            :719-746
            一次前向读每个轮次答案位上的分类器列                                             :735-744
```

`engine.answer` 的两种模式共用一个不变式：**「读出来的东西」永远是 K 个标签上的一个分布**，
差别只在这 K 个数是怎么来的——一个是采样器掩码后的 logprob，一个是池化 runner 上的分类器列。

---

## 2. 「typed」从哪来：标签集是运行期算出来的

标签不是一张写在文件里的表。`letter_labels(tok, k, form)`（`letters.py:73-87`）每次调用都走
`_labels_for`：按 `A`…`Z`、然后 `AA`…`ZZ` 的顺序遍历候选，只保留「编码后恰好一个 token」的那些，
凑够 k 个为止。

```python
# letters.py:58-70（节选）
ids = tok.encode((" " + code) if form == "spaced" else code, add_special_tokens=False)
if len(ids) != 1:
    continue
if ids[0] in seen:
    raise AssertionError(f"label {code!r} collides with an earlier label's token {ids[0]}")
...
if len(out) == k:
    return tuple(out)
raise AssertionError(f"only {len(out)} single-token letter labels exist in this tokenizer, {k} were asked for")
```

三件事值得单独记下来：

1. **上限有两个，性质不同。** `MAX_LABELS = 255`（`letters.py:28`）是线格式的约束（注释：
   *the wire format's cap, an 8-bit option index*）；而**先撞上的是分词器**——`_labels_for` 找不到
   第 k 个单 token 字母码时抛的是断言，不是退回多 token 标签。代码注释给了 Qwen3.6 分词器上的
   计数（*all 26 single letters and 544 of the 676 two-letter codes are single tokens*，`letters.py:15-17`），
   但那需要那份分词器，**本文未核**。
2. **标签形式有两种，由布局决定。** `form="spaced"` 要求 `" A"` 单 token（`Answer:` 之后的位置读它），
   `form="bare"` 要求 `"A"` 单 token（聊天模板自己的答案位读它）。`PromptFormat.label_form()`
   （`letters.py:231-233`）在 `(variants, slot) == ("single", "template")` 时选 `bare`，否则 `spaced`。
   `letter_labels` 的注释写明两套列表**在 Qwen3.6 上到第 85 个选项还一致**，再往后 bare 会跳过
   `CJ`/`CZ`/`EJ`/`GK`/`JL`（`letters.py:76-79`）——同样地，**本文未核**。
3. **请求侧的选项数没有第二条限制。** `to_engine_question`（`systemone.py:129-181`）只要求
   `criteria` 至少 1 项（`systemone.py:63`、`:69`），选项数超过本 tokenizer 能提供的单 token 字母码
   时由 `_labels_for` 的断言兜住——报出来的是一条 500 之下的 400（`systemone.py:930-931` 把
   `AssertionError` 映射成 400）。

标签 id 最后经 `label_token_ids`（`letters.py:117-127`）再验一次：每个候选串必须恰好一个 token，
且 id 互不相同。这两条断言是**整个「typed」性质的守门人**——它们一旦成立，softmax 的支撑集就恰好
是选项集，没有第二个出口。

`pad_prompt`、`state_prefix` 这些名字里都带「prompt」，而**每一条 prompt 的标签都是本次请求
现算的**：`_labels` 有一个按 `(cands, fmt)` 的缓存（`vllm_engine.py:465-470`），键是标签串而非
选项数，所以同一份布局下同一组字母码只算一次分词。

---

## 3. 读位置：构造出来的答案位，运行期无标定

prompt 的骨架在 `chat_wrap` / `chat_turn`（`letters.py:133-156`、`:272-285`）：

```python
# letters.py:145-156（节选）
if mode == "base":
    return prompt
assert prompt.endswith("\nAnswer:"), prompt[-40:]
kw = {"enable_thinking": False} if mode == "chat" else {}
out = tok.apply_chat_template([{"role": "user", "content": prompt[: -len("\nAnswer:")]}],
                              add_generation_prompt=True, tokenize=False, **kw)
if mode == "chat" and out.count("<think>") != out.count("</think>"):
    raise AssertionError(f"thinking left open in chat mode: {out[-60:]!r}")
return out + "Answer:"
```

- **Qwen 基座**（`answer_slot="prefill"`，`families.py:70`）：用户轮之后是模板的生成提示，再**手工拼上**
  `Answer:`。答案位就是这一段之后的那个位置——与 Jeff / decider 的「槽位」是同一类构造。
- **Gemma 基座**（`answer_slot="template"`，`families.py:141`、`:172`）：模板自己的第一个助手位置就是
  答案位，不再拼 `Answer:`（`letters.py:285` 的三元表达式）。同一段代码在 `chat` 模式下**断言思维块
  已闭合**（`letters.py:154-155`），并在 `PromptFormat.system_prompt` 打开时插一个系统轮
  （`letters.py:279-283`，文本见 `readout/system_prompt.py:7-11`；两个文本都是 MIT 许可的
  `THIRD-PARTY.md` 条目，`spaced.py:1-7` 同样）。

`enable_thinking=False` 的接法在这里只有**一条**：`kw = {"enable_thinking": False} if mode == "chat" else {}`
（`letters.py:150`）——注意 `chat-turn` 对应的 `chat-think` 模式**故意开着思维块**
（`letters.py:141` 的注释：*the template's default, which opens "<think>" and leaves it open*），
它是评测对照臂，不是服务路径。服务路径上 `question_body` / `chat_turn` 都走 `chat`
（`letters.py:277` 与 `:283` 都显式传 `enable_thinking=False`）。

**运行期没有任何标定**。全仓与「扫位置」有关的代码只有一处，且不在服务路径上：

- `state_prefix(tok, body)`（`vllm_engine.py:159-174`）**扫的是共享前缀的长度**，不是读位置：它把
  state 后面分别接一个 `"A"` 与一个 `"B"` 的假问题各编码一遍，取两者第一个不同的下标 n。注释写明
  为什么必须只看 state：*a split computed from the request's questions ... would move with the question
  mix, and so would any padding inserted there*。
- 真正会在运行期「找位置」的是 `--register-boundary` 那条路（§8.3）——它找的是**state 的边界**，
  同样不是读位置。

于是「读位置对不对」这件事在这条链上**没有被测量过**：没有 `marker_mass`（§5），也没有任何
「答案位是否落在预期 token 上」的断言。替代它的是一条**结构性**的保证：位置是 prompt 构造时写进去的，
而 prompt 的构造又被三处断言钉住（`letters.py:149` 的 `endswith("\nAnswer:")`、
`letters.py:154-155` 的思维块闭合、`vllm_engine.py:500-504` 的「题文本不得与 state 末 token 合并」）。

---

## 4. 一次前向：只读一个位置，只取 K 行

separate 模式的核心就是一次 `llm.generate`，每题一行（`vllm_engine.py:539-558`）：

```python
sps = [
    SamplingParams(
        max_tokens=1,
        temperature=0.0,
        logprobs=len(allowed_ids(lab)),
        allowed_token_ids=allowed_ids(lab),
        skip_reading_prefix_cache=skip_cache or None,
        detokenize=False,
    )
    for _, lab in rows
]
outs = guarded(self, self.llm.generate,
               [TokensPrompt(prompt_token_ids=ids, **extra) for ids, _ in rows], sps, ...)
```

三处细节决定了这是「读」而不是「生成」：

- `allowed_token_ids=allowed_ids(lab)` 把采样器的支撑集压到这一题的 K 个标签 token 上；
- `logprobs=len(allowed_ids(lab))` 只要这 K 个；`logprobs_mode="processed_logprobs"` 在引擎构建时
  设死（`vllm_engine.py:277`），于是返回的 K 个数**就是这 K 个标签行上的 log-softmax**；
- `detokenize=False`（`vllm_engine.py:546`）。注释给了理由：*labels are read by token id and no text is
  used; the per-request detokenizer was 73% of the frontend's CPU at an 8,000-token state (profiled)*。

**一次前向是真的只有一次**，但**确实采样了一个 token**：`max_tokens=1`。这一点与 decider 的
`serve_vllm`、H2O-Lightning 的 shim 是同一个手法——向采样器要一个 token 只为拿回那个位置的
logprobs，采到的 token 本身（`o.outputs[0].logprobs[0]` 的键）从不被读。区别在**记账**上：
decisio 的 `usage.output_tokens` 干脆就是**题目数**（`systemone.py:600`），而 `docs/api.md:77`
把这条写成了契约：*`usage.output_tokens` is one per question*（§11 #4、§15.2）。

一道题一行、一行一次引擎请求，是 `--multi-question sequential` 的行为（`vllm_engine.py:700-715`
把每题的 `score_prompts` 单独调一次）。同一段落里另有一句注释把「为什么默认是 sequential 而不是批量」
说成了一条可观测的性质：*so no question shares a forward pass with another and its answer cannot depend
on what else the request asked*（`vllm_engine.py:701-702`）。

---

## 5. 掩码 + 重归一化：掩码由采样器做，重归一化在构造上已经发生

RYOTIDE 的 `marker_mass` 量的是「全词表概率里有多少落在标记上」。在这条链上**这个量没有对象**：

- separate 模式返回的 K 个数，是 `processed_logprobs` 在 `allowed_token_ids` 上的 log-softmax，
  **和恒为 1**——标签外的质量在返回向量里恒等于 0，无论它在模型内部是多少；
- packed 模式更彻底：分类器只由标签行的 token id 建起来（`vllm_engine.py:285-301`），
  `score_packed` 直接在这 K 列上做 `softmax`（`vllm_engine.py:741-743`），**词表维度根本没有被物化**；
- `--model-class hidden-readout` 那条路也一样：`W[labels]` 是从 checkpoint 里挑出的 K 行
  （`hidden_engine.py:295-296`、`vllm_plugin/hidden.py` 与 `SingleEngineHidden.label_logits`），
  其余行不参与。

所以**这里没有「掩码 + 重归一化」这一步可以观测**：重归一化发生在采样器内部，本仓看不到；本仓能看到的
是一个**已经被采样器重归一化过**的向量。与 decider / Nimble / Kev / H2O-Lightning 是同一个结论、
第五种机制（§11 #1）。

本仓**唯一一处**会显式处理「一个标签的多种 token 形式」的地方是 `--label-variants summed`
（Gemma 基座的默认，`families.py:142`、`:173`）：一个标签的 `" A"`、`"A"`、字节回退 token
（`letters.py:288-296`）**全部**作为 allowed id 送进去，再按标签求和：

```python
# letters.py:348-358（节选）
if not is_grouped(lab):
    return np.array([lp_of(t) for t in lab], dtype=np.float64)
out = []
for g in lab:
    v = np.array([lp_of(t) for t in g], dtype=np.float64)
    m = v.max()
    out.append(m + np.log(np.exp(v - m).sum()))   # 同一标签的多个单 token 形式的 log-sum-exp
```

这是一步**标签内部的边缘化**，不是词表掩码——它的输入仍然只是那些标签形式。它同时把 allowed id
的个数从 K 抬到最多 4K（`vllm_engine.py:275-276`：`max_lp = 256 if single else 1024`），
这是 `--label-variants summed` 唯一可见的代价。

---

## 6. 顺序偏置与标签先验：两件事，一件默认关、一件零调用点

### 6.1 两序平均（`--orders 2`，默认关）

`--orders` 只有 1 和 2 两档（`vllm_engine.py:1292-1299`），默认 1。打开后每题再读一个**不同的第二序**，
再把两个分布折回真实选项平均；agree 与 disagree 都只进分支日志，**不进响应**：

```python
# systemone.py:252-266（节选）
if q["kind"] == "noul":
    return dict(q, noul_order="no_yes"), [0, 1]     # yes/no：不换位，改问法
n = len(q["options"])
rng = random.Random(f"0:{name}")                    # Reflex 的播种：一题的序不依赖别的题
perm = list(range(n))
if n == 2:
    perm = [1, 0]
else:
    for _ in range(50):
        rng.shuffle(perm)
        if perm != list(range(n)):
            break
```

第二支的折叠与平均在 `systemone.py:529-534`（`p2[orig] = probs[j][pos]`，然后
`p = (p + p2) / 2` 在 `:567-568`），disagreement 是两支到其平均的**平均全变差距离**
（`systemone.py:318-320`），只有 `--branch-log` 给了才落盘（`systemone.py:588-597`，
每条记录带 `request_sha256`）。

两处硬边界：yes/no 的第二支是**改问法**而不是改顺序（`vllm_engine.py:190-198` 把
`"Answer yes or no."` 换成 `"Answer no or yes."`，并在找不到时抛
`AssertionError("the yes/no prompt changed; update the second branch")`）；而**intent head 与两序平均
不许同时用**，直接抛（`systemone.py:488-493`，理由是 *the head is bound to one option order*）。
`orders == 2` 时 noul 也强制走 words 形式（`systemone.py:463`）。

### 6.2 标签先验排序（本仓零调用点）

`letters.py` 里另有一整套与选项顺序无关、但**改标签先验**的机件：

- `consonant_pair_pool(tok)`（`:90-99`）：只取两个辅音组成的码（`BCDFGHJKLMNPQRSTVWXZ`，`:36`）。
  理由写在注释里——`" A"`…`" J"` 本身是词或词首（`" I"`、`" A"`），基座在 MMLU-Pro 上**过度偏好
  `" I"` 与 `" J"`**，而 `" BK"` 这样的码没有这层含义（`letters.py:33-35`）；
- `pool_prior_prompt()`（`:102-105`）：一条**没有 state、也没有选项列表**的 prompt，
  它的分数就是每个码作为答案的先验；
- `order_pool(pool, logprobs)`（`:108-114`）：按「内容无关概率离池中位数多远」**最平的在前**排序，
  这样**任意前 K 个**都尽量接近均匀（K 就是提问的选项数）。

注释说这是 `--labels consonant-pairs` 的行为（`letters.py:33`）——**本仓没有这个 flag**。
`grep` 的结果：`consonant_pair_pool` / `order_pool` / `pool_prior_prompt` / `LABEL_SCHEMES` 全部
只有定义、没有调用点，`letters_prompt` 的 `pool` 参数（`:159`）也从未被传过；
而**服务路径根本不走 `letters_prompt`**——它走 `question_body`（`letters.py:248-263`），
后者没有 `pool` 这个形参（`vllm_engine.py:189` 是它唯一的服务侧调用点）。
所以这是一条**写了、注释指着一个不存在的 flag、且没有任何调用点**的路径（§11 #6 那条）。

### 6.3 与顺序有关的第三处：并列时的取首

`top_index(keys, p, among=None)`（`systemone.py:287-294`）在概率**精确相等**时取 key 按 Unicode
码点排序最靠前的那个，理由写得很直接：*JSON objects are unordered, and clients that serialise with
sorted keys reorder them*。`docs/api.md` 的字段表里也有这一条。这与 Kev 的 `option_isolation`、
Jevlike 的索引取首是同一个问题的第三种处理：**承认并列存在，把它交给一个与请求顺序无关的规则**。

---

## 7. 布局是配置：`PromptFormat`、逐 base 画像与任务指纹

§3 已经看到 `Answer:`，但 `Answer:` 只是四个互相独立的开关之一。`PromptFormat` 是一个 frozen
dataclass（`letters.py:195-242`），字段与取值：

| 字段 | 取值 | Qwen 基座 | Gemma 两代 |
|---|---|---|---|
| `tail` | `compact` / `spaced` | `spaced`（`families.py:69`） | `spaced`（`:140`、`:171`） |
| `slot` | `prefill` / `template` | `prefill`（`:70`） | `template`（`:141`、`:172`） |
| `variants` | `single` / `summed` | `single`（`:71`） | `summed`（`:142`、`:173`） |
| `system_prompt` | 布尔 | `False`（`:68` 的默认值） | `True`（`:139`、`:170`） |

**Qwen 那一列不是显式设置，而是 dataclass 的默认值**（`families.py:68-71`）——`QWEN` 画像
（`:96-114`）四个字段一个都没写，所以「Qwen 的布局」就是「没有基座时的布局」；Gemma 两代四个
字段全部显式写了，而 12B 与 31B 在这四个字段上完全一致（`:139-143` 与 `:170-174`）。

`__post_init__`（`letters.py:219-226`）对前三个字段逐个校验取值；`is_default()`（`:228-229`）用来判
「是不是那个逐字节不变的默认布局」；`facts()`（`:235-242`）把开着的 `system_prompt` 单独放进去，
理由写在注释里：*so the facts (and task fingerprints) of every Qwen format are unchanged*。

三个可见后果：

1. **布局进了任务的指纹。** `task_fingerprint(...)`（`names.py:70-101`）只在 `prompt_format` 不是
   默认时才把它写进指纹，其余键按「不是默认就不写」的同一原则处理，于是**默认布局下注册的任务在
   换布局之后一律失效**（`names.py:84-87` 的注释，`systemone.py:1079-1081` 的
   `same_fingerprint` 拒绝，`vllm_engine.py:1676-1678` 启动时把失效任务打成
   `WARNING: tasks fitted under another model or rendering are not applied`）；
2. **换基座会换掉一整组默认值。** `resolve_base`（`vllm_engine.py:1087-1166`）在 `--base`/`--model`
   都没给时用 `DEFAULT_BASE = "gemma-4-31b"`（`families.py:194`），并把 `pad_to`、`served_name`、
   `noul_rendering`、`temperature`、`temperature_choice` 全部回落到画像的值（`:1156-1165`）；
   被服务的那个名字也随之变（`decisio-gemma-4-31b-it-letters` / `-12b-it-` / `-qwen3.6-35b-a3b-`，
   `names.py:20` 与 `families.py:103`、`:121`、`:158`）；
3. **两代 Gemma 在同一份代码里是两套默认。** 12B 与 31B 共享 prompt 布局与 `head_rows`
   （`families.py:163` 直接引 `GEMMA4.head_rows`）、但温度不同（3.592 vs 5.252 / 4.672）、
   `register_boundary` 不同、`multi_question` 不同（§8.4）——`/health` 的 `profile` 段把它们逐项打印
   （`vllm_engine.py:1061-1084`）。

`--model-class` 是另一个开关，但**它不改布局，改的是权重怎么装**
（`MODEL_CLASSES = ("hidden-readout", "text-only", "view")`，`vllm_engine.py:983`）：
`view` 让 vLLM 把 `--model` 当目录原样加载（不装插件），另两个把 checkpoint 装到 decisio 注册的
类上（`vllm_plugin/models.py:29-40`、`vllm_plugin/gemma.py:34-47`）。`view` 这条路的用处是
**不装包也能服务**——`serve/make_text_only.py` 造的那个纯文本视图就是给它的（其模块头 `:13-14`）。

---

## 8. 多问题共享 state：三件套与另一套

### 8.1 前缀怎么切：只看 state，不看请求

`state_prefix` 已经在 §3 引过。这里补一条它的**用法上的**细节：切点是「state + 空行」的最后一个
token，而**题文本必须不与它合并**——`_prepare_separate` 在拼完行之后逐行验一次：

```python
# vllm_engine.py:500-504
for ids, _ in rows:
    if ids[:n] != prefix:
        raise ValueError(
            "a question's text merges with the state's last token; questions must not start with whitespace"
        )
```

同一段代码还有一条更便宜的路径：先用**第 0 题**的完整 prompt 减去它自己的后缀得到 n，如果这个
切法能对上就用它，对不上才退回上面的双假问题切法（`vllm_engine.py:488-493`）。两条路都只分词一次
state——注释给了代价数字：*Full-prompt tokenization per question cost 0.6 ms (500-token state) to 5.7 ms
(8,000) of CPU per question*、*one tokenization of the state (three cost 87 ms at 8,000 tokens)*（`:476-477`、`:485-486`）。

### 8.2 填充：四个策略、三个插入位置

`pad_prompt(ids, n, k, token, where, head_len)`（`vllm_engine.py:118-131`）在三个位置之一插入 k 个
填充 token —— `between`（state 空行之后）、`front`（位置 0）、`user`（用户轮之内、state 之前）。
`k` 的算法在 `_prepare_separate:505-514`：默认 `k = (-n) % pad_unit`（**让 state 的边界落在块边界
上**），`row` 策略则改成让**整行**落在块边界上（`:508-512`），`none` 与「共享前缀只有一道题可用时的
`shared`」都置 0（`:506-507`）。

四种策略的语义与默认值（`vllm_engine.py:1015-1022`、`:1205-1215`）：vLLM 默认 `always`、MLX 默认
`none`。MLX 的 `none` 有理由：*MLX's cache needs no padding, so its prompts differ from the vLLM path's by
the padding only, a question asked alone and inside a request is one prompt, and a single question costs about
half the time*。

**这里有一个必须原样记住的实测数字**：`between` 这个位置是**第一个**做法，而它
*「moved 9.4% of top answers against the unpadded prompt」*（`vllm_engine.py:122-124` 的注释）。
三种位置的**前缀长度与缓存命中完全相同**，差别只在模型看到了什么——这正是「填充方向/填充位置」
在本仓是一个**被测量过**的自由度、而不是一个风格选择（对照 decider 的 `--pad-policy row` 与
`runs/2026-10-02_pad-policy-row/conformance_l1.json.gz`：**committed 的 C4 在那次试验里是 FAIL**，
max |Δp| = 0.014737647390295483，§14.4）。

### 8.3 边界注册：`--register-boundary` 的三个取值与一个后台线程

Gemma 基座与 Qwen 基座在这一步也不一样。`register_state_boundary` 在 Gemma 两代是 `True`、
Qwen 是 `False`（`families.py:113`、`:147`、`:180`），理由是 Qwen 的 state 本来就填到 1,056 的块边界、
而一道题比一个块短，所以 state 末尾**已经是**那道题结束时的最新检查点（`families.py:111-113` 的注释）；
Gemma 不填充，最近检查点在题文本内部，于是第二道**不同的**题会把整个 state 再读一遍——注释给了
两处代价数字（`:144-146`：*531 ms instead of 47 at 3,000 tokens*）。

`--register-boundary` 的三档（`vllm_engine.py:1227-1236`、`serve/boundary.py:1-23`）：

- `before`：先发一个 P+1 token 的预热，再发问题（0.8.1 的行为，两个 Gemma 基座现在的默认）；
- `after`：**先把问题答完**，把预热挂起（`Registrar.defer`），等响应的最后一个字节交给服务器之后
  再发（`AfterResponse` 中间件，`boundary.py:81-106`）——`runs/2026-10-06_latency-585w` 的结果是
  它在 31B 上把吞吐吃掉 39% / 26%（`families.py:181-184` 的注释），于是两个 Gemma 基座都回到
  `before`；
- `off`：从不注册。

`Registrar`（`boundary.py:109-253`）是一个带条件变量的后台线程 + 一个 `OrderedDict`：
`PENDING_KEPT = 256` 条之后丢最旧的并记 `dropped`（`:112`、`:140-143`），发送失败只计数不失败
调用方的请求（`:190-203`），退出时等一个进行中的预热最多 60 秒（`CLOSE_WAIT_S`，`:113`、`:224-227`）。
`/health` 会把 `pending` / `running` 与四个计数器一起报出来（`boundary.py:125-129`、
`vllm_engine.py:852-854`）。

### 8.4 一次多题：三种策略

`--multi-question`（`vllm_engine.py:1216-1226`、`:962-972`）：

| 取值 | 做法 | 谁在用 |
|---|---|---|
| `sequential`（默认） | 预热一次，然后**每题各自一次引擎调用** | 除 31B 外的全部；MLX 强制（`:967-968`） |
| `warm` | 预热一次，然后**所有题一批**发出 | gemma-4-31b 的默认，**只在引擎跑在服务器进程里时生效**（`engine_process == "in"`） |
| `batch` | 不预热，一次调用把每题作为独立一行发出（每题自己预填充 state） | 可选 |

`warm` 之所以绑在 `engine_process in` 上，注释给了一条 vLLM 的行为：*`in` runs it in the server's
process ... where every question of a request is added before the first engine step, so --multi-question warm
repeats exactly (vllm-project/vllm#59764)*（`vllm_engine.py:919-923`）。而 `resolve_multi_question`
（`:962-972`）只为 31B 保留 `warm`，且注明它过了一道门：*no choice changed against sequential on the
suite, JevBench and travel, the largest probability difference 0.0073*（`:965-966`）。

### 8.5 packed 模式：另一条完全不同的共享路径

`--mode packed`（`vllm_engine.py:719-773`）把多题拼成**一条 prompt 的连续聊天轮次**，每个轮次的
助手回复停在 `Answer:`，然后用池化 runner 在一次 prefill 里读**每个轮次最后位置**上的分类器列。
两处关键差异：

- 分类器**在加载时**由标签行的 token id 建好：`classifier_from_token` 用 `[" yes", " no"]` 加
  最多 `max_labels = 77` 个字母码（`vllm_engine.py:285-301`）——所以 packed 的 K **上限 77**，
  与 separate 的 255 不是同一个数；
- **第一条题与它单独服务时的 prompt 逐字节相同**，第二条起会看到前面的题（不是答案），
  所以隔离**不是结构性的**：*later ones see the earlier questions (not their answers), so isolation is
  not structural and must be measured*（`vllm_engine.py:29-32`）。

`usage.output_tokens` 的语义在这里第一次出现裂缝：packed 模式下**一个 token 都不生成**
（`llm.encode` + pooling runner），而 `output_tokens` 仍然报题目数（§11 #4）。

---

## 9. 服务化与 fail-loud

### 9.1 引擎死了怎么办：一个共享的 `EngineHealth`

`serve/engine_health.py` 的模块头把契约写死了三条（`:10-14`）：引擎一死，之后的每个请求立刻 503、
`/health` 报 `"engine": "dead"`、**进程以 70 号（EX_SOFTWARE）退出**让重启策略拉起新进程。

判定链是**两段式**的，而且第二段很讲究：vLLM 自己说死了（`EngineDeadError`、引擎核进程没了、
`errored` 标志）就立即判死（`engine_health.py:88-107`）；否则**罚一次探针**——一个一 token 的前向
（`vllm_engine.py:343-358`、`hidden_engine.py:188-192`），超时是 `probe_timeout_s`（5 秒）与
「最近成功调用的最慢者的 3 倍」里较大的那个（`engine_health.py:198-200`）：

```python
# engine_health.py:253-270（节选）
def call(self, engine, fn, *args, **kwargs):
    self.check()
    t0 = time.perf_counter()
    try:
        out = fn(*args, **kwargs)
    except Exception as e:
        if is_request_error(e):          # ValueError / TypeError / KeyError / VLLMClientError：请求的错
            raise
        if not self.dead:
            why = self.confirm(engine, e)
            if why is None:
                raise                    # 500：引擎活着，是 decisio 自己的 bug
            self.mark_dead(why)
        raise EngineDead(self.reason) from e
```

退出也不是立刻：`mark_dead` 记一次原因、起一个后台线程，等 `grace_s = 2.0` 秒、再等在飞请求清零
（最多 `DRAIN_S = 8.0` 秒），最后才 `os._exit(70)`；`os._exit` 之前先 kill 掉所有子进程，注释写明
理由（`:138-141`：*`os._exit` alone orphans the engine core with the GPU held*）。
`vllm_engine.run_or_die`（`:88-111`）是同一个教训的另一半：启动阶段任何异常都打印、杀子进程、
`os._exit(1)`——注释里的现场是 *a crashed gate hung for ten minutes at 88 GB until killed by PID*。

### 9.2 启动期的四道闸

| 闸 | 位置 | 拒绝什么 |
|---|---|---|
| `deep_gemm_guard` | `vllm_engine.py:898-912` | `VLLM_USE_DEEP_GEMM` 被设成 0 以外的值（默认替服务方设成 0），除非显式 `--allow-deep-gemm`——理由是 FP8 MoE 的 DeepGEMM 路径在 Blackwell 卡上给错结果 |
| `engine_process_guard` | `:918-959` | 有第二引擎时 `--engine-process in`（两个 vLLM 引擎同进程在卡上每个请求都失败）；环境变量说 0 时显式的 `separate`（否则 `/health` 会谎报） |
| `check_adapter_names` | `:138-156` | adapter 的张量名与将要加载它的类对不上——不查的话 vLLM 会把它当**静默 no-op** 装进去 |
| `check_reserved` | `vllm_plugin/hidden.py:59-67` | 保留 logit 列（给 intent head 传 hidden state 用）越出词表，或**与任何标签 token 相撞** |

还有一条不在启动期但同类的：vLLM 把「prompt 超上下文」报成 `VLLMValidationError`
（**不是** `ValueError`），所以路由的 `except ValueError` 抓不到——`client_error`
（`engine_health.py:68-85`）把它翻成 422，其余客户端错翻成 400，并在消息里点明
*the rendered prompt (the state with one question) is longer than the context*。

### 9.3 三道一致性门（C2 / C3 / C4）

`serve/systemone_conformance.py` 定义了三道门，模块头写明「在每个 runs/ benchmark 测量前运行」
（`:3-14`）。三道门各对一个具体的不一致来源：

- **C2**（`:170-225`）：用 **TypeSafe 官方 SDK** 打 12 条覆盖请求，断言响应能被 SDK 的
  `SystemOneResponse` 解析、答案名与问题名一致、`usage.output_tokens == len(questions)`、
  `models.list()` 可解析、非法请求回 422 且 detail 的形状是 `HTTPValidationError`；
  然后**按适配器的规则重算每一题**（`check_answer`，`:148-167`）：概率键顺序、和为 1、
  `choice == top_index(keys, p)`、`confidence == choice_confidence(p)`、
  `score == Σ i·pᵢ`、`confidence == score_confidence(p)`；
- **C3**（`:289-310`）：**只用分词器、不起服务**。同一个 item 的 `/v1/systemone` 形态与
  `/v1/answer` 形态，经引擎自己的 `_prepare_separate` 必须给出**相同的 token 行与相同的标签**；
- **C4**（`:313-357`）：在 HTTP 上比两条路由的概率，而且**先把温度补回去**——
  `/v1/systemone` 服务的是加过温度的读数、`/v1/answer` 是原生的，所以门用 `/health` 报出来的
  温度对 `/v1/answer` 施加同一个函数再比（`:318-335`，`served_temperature(health, type)`）。
  判据是 `max_abs_delta_p == 0.0` 且 `top_answer_flips == 0`（`:350`）。

C4 那一步是这条链上**唯一把「两条路由给的不是同一个数」写成代码的地方**——
它不是 bug，是被 `/health` 明确回声、被门显式补偿的设计。

### 9.4 三个后端

`--backend` 有 `vllm` / `hf` / `mlx` 三档（`vllm_engine.py:1266-1273`）：

- **hf**：`serve/hf_letters.py` 的 CPU 替身，`score_prompts` 里 `logits_to_keep=1` 只取末位
  （`hf_letters.py:97`），自己报 `engine: "hf_letters (CPU stand-in, not for measurement)"`（`:60`）；
- **mlx**：`serve/mlx_engine.py`，Apple silicon。它自己实现了一份**跨请求**的前缀缓存
  （`PrefixCache`，LRU + 字节预算 + 内容全量比对，`:147-187`），并且每个问题从共享前缀的**一份拷贝**
  上续跑（`copy_cache`，`:89-113`）；`--prefix-cache-mb` 的默认值逐 base 不同
  （`families.py:90`：Qwen 2048、gemma-4-12b 7400）。MLX 路径**拒绝**图像、packed、adapter 与
  非 sequential 的多题（`vllm_engine.py:1471-1485`）；
- **vllm**：默认，见上。

---

## 10. 置信度从哪来，又怎么被改写

### 10.1 谁在改发布出去的数值

按**执行顺序**（`systemone.py:515-587` 的逐题循环）：

| # | 环节 | 位置 | 改 argmax？ |
|---|---|---|---|
| ① | **训练本身** | **不存在**——主路径零训练。逐任务的可选修正见 ②③ | — |
| ② | per-task 先验 bias：`softmax(lp − b)`，`b` 由客户标注样例拟合、要过 5 折交叉验证 + 单边 95% 检验 | `readout/calibration.py:67-130`，应用在 `systemone.py:552-566` | 可以 |
| ③ | intent head：`softmax(lp + h @ A + c)`，`h` 是答案位的 final-norm 隐状态，`A`、`c` 由标注样例拟合 | `readout/intent_head.py:139-191`，应用在 `systemone.py:517-523` | 可以 |
| ④ | **逐题型温度**：`softmax(log p / T)`（float64；`T == 1` 原样返回，逐位不变） | `serve/temperature.py:22-32`，应用在 `systemone.py:569-575` | **否**（单调） |
| ⑤ | `--noul-commit` 带内推移：`0.20 < P(yes) < 0.80` 时按自己那一侧推到 0.80 / 0.20 | `systemone.py:211-221`，应用在 `:583-584` | **否**（保侧，恰好 0.5 归 no） |
| ⑥ | 拒答阈值：过阈值就拒答，否则取其余选项里最大的 | `serve/abstention.py:125-144` | **是**（阈值就是 argmax 的替代品） |
| ⑦ | 报告用的公式：`choice_confidence = (p_max − 1/k)/(1 − 1/k)`、`score_confidence = 1 − MAD/uniform MAD`、`score = Σ i·pᵢ` | `systemone.py:269-284`、`:311` | 否（只改报出去的数） |
| ⑧ | README 里四舍五入到三位小数 | `README.md:195` | 否 |

四条必须一起读的约束：

- **②③ 与 ④ 互斥**：`systemone.py:571-572` 的 `if not corrected and t_q != 1.0` ——
  任务修正生效时温度**不叠加**；任务的 head 生效时连先验 bias 也不叠（`serve/tasks.py:15-17`
  的注释：*Where a head applies, calibration is not stacked on it (its c covers what the prior does)*）；
- **④ 是逐题型的**：`temperature_of(qtype)` 先查 `temperatures[qtype]`，没有才用全局
  （`systemone.py:629-631`）。committed 的 `/health` 里能直接读到这一层
  （`runs/2026-10-09_readme-pip-path/g31_responses.json` 的 `systemone.temperature = 5.252`、
  `systemone.temperatures.choice = 4.672`）；
- **⑤ 只作用于 noul**，而且**默认关**——committed 的 manifest 里记着它的代价：
  同一批 1,474 题，"准确率" 两边都是 0.8385（题目本身没动），而 harness 记的那个口径从
  0.6716 抬到 0.8385、log loss 从 0.3977 升到 0.4145、ECE 从 0.044 升到 0.078
  （`runs/2026-10-03_noul-commit/manifest.json` 的 `results.served_default`）。**改的是报出去的数，
  一条答案都没改**——这是本仓最干净的一处「改数值不改答案」；
- **⑦ 只改 `confidence`，不改 `probabilities`**，且 ①—⑥ 全都不改 `probabilities` 与 `confidence`
  的对应关系：`confidence` 永远是从**已经改过的** `probabilities` 现算的
  （`to_answer`，`systemone.py:297-315`）。这与 Nimble 的托管面同类，但这里**三条公式全在本仓**，
  所以 §14.2 可以逐位复算（对照 vLLM Semantic Router：同仓三套公式、且消费者自己另算一套）。

### 10.2 温度的实际值

| 基座 | 全局 T | choice 的 T | 来源 |
|---|---|---|---|
| `qwen3.6-35b-a3b` | **1.506** | **1.370** | `families.py:109-110`、`temperature.py:18-19` |
| `gemma-4-12b` | **3.592** | 同全局（`choice_temperature=None`） | `families.py:137-138` |
| `gemma-4-31b`（默认基座） | **5.252** | **4.672** | `families.py:168-169` |

`serve/temperature.py` 的模块头把拟合口径写清楚了：*fitted by minimum log loss on the served prompt's
plain readouts of a private 1,400-item decision suite, never on a benchmark's items*（`:5-7`），
并记了 Qwen 的旧值：*1.307 before*（`:18`）。**三个基座的温度都远大于 1**（Gemma 31B 的 5.252 尤其
压平），这与 H2O-Lightning 的 0.65–0.8 是相反方向的选择——本仓要的是**校准**而不是置信度。

`apply_temperature` 的一句不变式是可以机械验证的：*The most probable option stays the most probable,
and exactly tied options stay tied*（`temperature.py:23-24`）。§14 在 committed 的 20,810 条
读数上把它验了一遍。

### 10.3 逐任务拟合的三样东西与它们的接受判据

三个可选的拟合环节共用同一种写法：**从客户自己的标注样例拟合，用一个写死的接受判据决定要不要真的启用，
不通过就存一个「关」并写明理由**。

- **先验 bias**（`readout/calibration.py`）：`MIN_GAIN = 0.005` nats/item 的交叉验证 log loss 改善、
  单边 95% 检验（`T_ACCEPT = 1.645`）、交叉验证准确率不得下降（`:110-111`）；L2 惩罚**固定**为
  `LAMBDA = 0.1`，注释给了为什么不让交叉验证自己选：*with ~20 examples a cross-validated choice often
  picked no penalty and overfit (in development, 12 of 16 tasks passed with it, 16 of 16 at 0.1)*（`:35-36`）。
  少于 `MIN_EXAMPLES = 10` 条样例直接返回「关」（`:86-87`）。
- **intent head**（`readout/intent_head.py`）：只在**选项数 ≥ 10**（`tasks.py:36` 的
  `MIN_OPTIONS_FOR_HEAD`，注释：*an intent set, not a yes/no or multiple-choice question*）且**每个选项
  至少 5 条样例**（`MIN_PER_OPTION`，`:34`）时才拟合；惩罚从 `LAMS` 里按 5 折分层交叉验证挑，
  **备选里包含「不要头」**，它赢了就存关（`:101-119`）。这个头**绑死在它拟合时的那个选项列表和顺序上**
  （`options_sha256`，`:134-136`、`:188`），注释写明原因（*it learns which answer slot to trust*），
  并由 `serve/tasks.py:15-17` 再次强调不许与两序平均叠。
- **拒答阈值**（`serve/abstention.py`）：`MIN_EXAMPLES = 10` 且至少 2 条可答 + 2 条不可答；
  判据是**留一法**上阈值规则至少多对 1 条、误拒不多于原行为（`:108`）。阈值候选是相邻两个不同
  概率的**对数几率中点**，加「低于最小者」与 1.0 两个端点（`candidates`，`:42-50`）。

三条路都**不改权重**：拟合产物是一个 K 维 bias、一个 (d × K) 矩阵加一个 K 向量、一个标量阈值。

### 10.4 intent head 的隐状态从哪来：把 logits 的保留列当旁路

这是本仓最特别的一处工程做法，值得单独写一段。vLLM 的 generate 路径**不返回隐状态**，而头需要它。
decisio 的做法是注册一个自己的模型类，在 `compute_logits` 里**覆盖 d+1 个保留 token id 的 logits 列**
（`vllm_plugin/hidden.py:70-79`）：

```python
logits[..., start] = 0
logits[..., start + 1 : start + 1 + d] = hidden_states.to(logits.dtype)
```

一个全零的参考列，然后 h 本身。请求时只允许这些 id 并要它们的 logprob，返回的是这些列上的
log-softmax，于是 `h_j = lp_j − lp_0`（`:82-87`）。三条由卡上实测换来的细节：

1. vLLM 0.30.0 的单请求 allowed id 上限是 1024（`MAX_ALLOWED`，`:38`，注释引到了
   `vllm/v1/worker/gpu/sample/logit_bias.py` 的常量名），所以 2048 维要分 **3** 次请求
   （`reserved_chunks`，`:42-48`）；
2. **一批里发出去的多行不是同一个前向**——注释记的数是 *the three chunk requests sent together gave
   label probabilities that differed across the batch's rows by up to 0.59*，所以分块请求**一次一条、
   串行发出**（`hidden_engine.py:326-350`）；
3. 引擎的 logits 是 bf16 的，所以 `label_logits` 刻意**先在 bf16 里走一遍**
   （`hidden_engine.py:298-305`，理由：float64 重算与引擎最多差 0.019，bf16 之后 3.1e-8）。

启动时 `check_reserved` 断言保留区间不撞任何标签 token（`vllm_plugin/hidden.py:59-67`），
而 `hidden_start` 是**逐基座**给的（Qwen 走默认 100,000、12B 170,000、31B 210,000；`families.py:63`、
`:136`、`:167`）——最后一个的注释还解释了为什么 31B 不能用 12B 那个数：*the 12B's 170,000 would put a
label form (174,960) inside 5,376 dimensions*。

`--head-engine` 是替代路径：**再起一份权重**用 pooling runner 拿 h（`hidden_engine.py:77-204`），
代价是多一份显存、`--gpu-memory-utilization + --head-gpu-memory-utilization` 合计不得超过 0.95
（`vllm_engine.py:1616-1621`），而且**不能与图像引擎共存**（`:1502-1506`）。
默认走的是单引擎那条：`--model-class hidden-readout`（`resolve_head_mode`，`:986-1012`）。

---

## 11. 四条不变量的落位

| RYOTIDE §13 不变量 | decisio 的落位 |
|---|---|
| **#1** 读位置必须落在标记上（诊断量 `marker_mass`） | **概念不适用**：返回的向量在构造上就是 K 个标签行上的 log-softmax，词表其余部分从不进入任何读数（§5）。**改写后的等价命题**是「这 K 个 token id 必须恰好是这道题的标签，且互不相同」——它由两条运行期断言保证（`letters.py:62-66`、`:122-126`），凑不够就抛，**没有运行期诊断量** |
| **#2** 前缀与标记表面形式耦合，必须成对扫 | **改写为「把自由度删掉，而不是扫掉」**：`allowed_token_ids` 要的是 **token id**，所以不存在「表面形式」这个自由度（§4）。剩下的耦合换了个方向——`letter_labels` 的两种 `form`（`spaced` / `bare`）与 `slot` 绑定（`letters.py:231-233`），**换布局可能换掉标签列表**；这个耦合的守卫是**任务指纹**（`names.py:70-101`）与启动时的 `WARNING`（`vllm_engine.py:1676-1678`） |
| **#3** 只读最后一个位置 | **成立**，但「最后一个位置」在两种 slot 下是两个不同的东西：`prefill` 是手工拼的 `Answer:` 之后（`letters.py:156`），`template` 是聊天模板自己的第一个助手位置（`letters.py:285`）。多题时是**每题一行、各自一次前向**（`multi_question sequential`），packed 模式下则是**一次前向读每个轮次的答案位**（§8.5）——后者仍然是「只读构造出来的那个位置」，只是位置有 K 个 |
| **#4** 比较概率 / ECE 前先确认 temperature | **成立，而且要确认的是四件事**：①全局 T 与逐题型 T 哪一层生效（`systemone.py:629-631`）；②这个任务上温度**是否被跳过**（任务修正生效时温度不叠加，`:571-572`）；③**是哪条路由**——`/v1/answer` 是未加温度的原生读数、`/v1/systemone` 是加过的（C4 门就是干这个的，§9.3）；④`noul` 的带内推移（`--noul-commit`）有没有开。①②的值都能从 `/health` 读到，③在 `x-decisio-route` 头里，④在 `/health.systemone.noul_commit` |
| （各自新增的） | ①**标签集是运行期算出来的**——不是一张表、也不来自请求，而是「这个 tokenizer 里前 K 个能单 token 化的字母码」，所以**换分词器就是换标签空间**，且会以 400（断言）而不是静默降级的形式出现（§2）；②**同一个读法有两条给不同数的路由**，而且是**被 `x-decisio-route` 头与 C4 门显式承认**的（§9.3）；③**`output_tokens` 的定义是「题目数」而不是「生成的 token 数」**，且这是 `docs/api.md:77` 写下来的契约、被 C2 门断言（§11 #4 见下，`systemone.py:600`）；④**布局进了任务的指纹**——换布局之后注册过的任务**静默不生效**（只在启动日志里 WARNING）（§7） |

**#4 的数值链条**并排放进 `docs/comparison.md` 的 §2。这里只留一句：因为①不存在，
温度在这条链上是**单调的**——§14.1 在 20,810 条 committed 读数上复算，翻转 **0 次**。

---

## 12. 文件对照表

| 文件 | 行数 | 职责 |
|---|---:|---|
| `serve/vllm_engine.py` | 1717 | 引擎主体：两种模式、前缀切分、填充、边界注册、`/health` 与 `/v1/answer`、全部 CLI 与启动守卫 |
| `serve/systemone.py` | 1094 | TypeSafe 线格式、渲染规则、两序平均、拒答规划、三条 reporting 公式、四个任务端点 |
| `serve/mlx_engine.py` | 408 | Apple silicon 后端：自建跨请求前缀缓存 + 分支拷贝 |
| `serve/hidden_engine.py` | 402 | intent head 的隐状态三种取法（单引擎保留列 / 第二引擎池化 / CPU 替身） |
| `serve/systemone_conformance.py` | 392 | C2 / C3 / C4 三道一致性门 |
| `readout/letters.py` | 368 | 标签推导、prompt 布局（`PromptFormat`）、标签的 log-softmax 与多形式求和 |
| `hub_fp8.py` | 365 | 把 31B 的在线 FP8 量化离线复刻成一份 checkpoint，并给出逐层指纹 |
| `serve/engine_health.py` | 327 | 引擎存活判定、探针、503、退出码 70、在飞请求计数 |
| `hub.py` | 308 | decisio 仓库（权重 + `decision_config.json`）：打开、画像比对、拒绝 |
| `families.py` | 253 | 三个基座的画像（revision、布局、温度、填充、边界注册、多题策略） |
| `serve/boundary.py` | 252 | `--register-boundary after` 的挂起-到期-后台发送机制 |
| `serve/tasks.py` | 215 | 逐任务先验与头的注册、存储、指纹校验、导入导出 |
| `serve/hf_letters.py` | 194 | CPU 冒烟替身（只换模型加载与前向） |
| `readout/intent_head.py` | 191 | 线性头的拟合（含「不要头」这一档）与服务 |
| `serve/image_engine.py` | 148 | 图像路径：第二个多模态引擎、图片内容 id、token 展开算术 |
| `readout/abstention.py` | 144 | 拒答阈值的拟合（留一法）与判定 |
| `readout/calibration.py` | 130 | 逐任务先验 bias 的拟合与接受判据 |
| `vllm_plugin/__init__.py` | 117 | vLLM 插件注册（只对一个版本生效，且不改 vLLM 源码） |
| `names.py` | 105 | 线上的名字、旧拼写（`rlcd`）的兼容读取、任务指纹 |
| `vllm_plugin/hidden.py` | 87 | 保留 logit 列的写入与恢复（无 vLLM / torch 依赖） |
| `serve/temperature.py` | 32 | 温度的两个发布常数与 `softmax(log p / T)` |
| `readout/spaced.py` / `readout/system_prompt.py` | 7 / 11 | 两段 MIT 许可的提示文本（`THIRD-PARTY.md`） |
| `bench/di_cal.py` / `di_rows.py` / `di_report.py` | 87 / 48 / 68 | Decision Index 的 ECE（**10 个等质量箱**）、行集重建、看板口径复算 |
| `bench/latency.py` / `fresh_state.py` | 275 / 191 | 延迟与吞吐网格；「全新 state」的分阶段延迟 |
| `benchmarks/*.sh` | — | 调外部 harness（JevBench 与 Decision Index 都按 commit 拉取，`fetch_harnesses.sh:4-19`） |

---

## 13. 亲手跑一遍

**本文没有执行以下任何一条**（本机无 GPU、无权重、未安装 `decisio`）。列出它们是为了标明
「哪些数字是跑出来的、要什么条件」：

```bash
# CPU 冒烟（作者的验证路径之一）
python -m decisio.serve.vllm_engine --backend hf --model <small-model> --port 8000

# 默认基座（gemma-4-31b，FP8 加载，单引擎 hidden-readout 类）
python -m decisio.serve.vllm_engine --port 8000

# Qwen 基座 + 图像引擎
python -m decisio.serve.vllm_engine --base qwen3.6-35b-a3b --image-model Qwen/Qwen3.6-35B-A3B

# 三道一致性门（需要一份由 benchmarks/make_conformance_items.py 生成的 items 文件与那份分词器）
python -m decisio.serve.systemone_conformance --url http://127.0.0.1:8000 \
  --items items.json --tokenizer <tokenizer> --block-size <server's pad unit> --out conformance.json

# Decision Index（拉外部 kit，起服务，然后四步）
DECISION_INDEX=<kit checkout> benchmarks/decision_index.sh
```

---

## 14. 实例走查：在 committed 产物上复算

三处「实测」，动作都只有一种：读 committed 文件 + 标准库算术。**没有运行上游程序**。
另外，凡需要权重或 GPU 的读数（JevBench 236 项、ImageJevBench、三张延迟表、`/health` 里的
`requests_per_question`）**一概未核**，只登记它们各自声明的条件。

### 14.1 温度律：`softmax(log p / T)` 逐位复现，argmax 翻转 0 次

产物：`runs/2026-09-30_served-default/decision_index/{temperature_1,default}/items.jsonl.gz`
（两份文件各 20,810 行，`run_id` 集合完全相同），温度取同目录 `health_default.json` 里
`systemone.temperature` = **1.307**。

```python
# 标准库算术，逐题比较
z = [math.log(x) / 1.307 for x in p_temperature_1]   # p 来自 temperature_1 那一列
# softmax(z) 与 default 那一列逐项比较
```

结果（本文复算）：

- 20,810 条 choice 答案，`max |softmax(log p / 1.307) − 服务读数|` = **7.77e-16**；
- 同一批上 `choice` 字段**翻转 0 次**——与 `temperature.py:23-24` 那句不变式一致；
- K 的取值集合是 `{3, 4, 5, 6, 7, 8, 9, 10, 77, 151}`，**同一份产物里 K 逐题不同**，
  这本身就是 §2「标签集是运行期算出来的」的一个直接旁证。

必须说清楚的三点：①这个 run 的 `model` 字段写的是 `rlcd-qwen3.6-35b-a3b-letters`
（`names.py:21` 的旧名），`/health` 里还没有 `profile` 段，**所以它不是当前 HEAD 的产物**，
它证明的是温度那一行算术，不是当前代码的完整行为；②`1.307` 是 **Qwen 基座的旧温度**
（`temperature.py:18` 的注释：*1.307 before*），当前值是 1.506——**换基座换版本都换 T**；
③两份文件里的概率是 JSON 里的十进制字面量，复现到 1e-16 说明这条链没有别的隐藏环节。

### 14.2 SystemOne 的三条公式：在 committed 响应上逐位吻合

产物：`runs/2026-10-09_readme-pip-path/{g31,g12,g31_restart}_responses.json`
（`manifest.json` 记 `decisio 0.11.0` / `vllm 0.30.0` / RTX PRO 6000 600 W；`/health` 段里
`systemone.temperature = 5.252`、`systemone.temperatures.choice = 4.672`）。

| 公式 | 复算对象 | 最大偏差 |
|---|---|---|
| `choice_confidence = (max(p) − 1/k)/(1 − 1/k)` | 10 条 choice 答案 | **0.0** |
| `score_confidence = 1 − Σpᵢ·\|i − mode\| / uniform_mad` | 6 条 score 答案 | **0.0** |
| `score = Σ i·pᵢ` | 同上 | **0.0** |

同一组公式在 §14.1 那批 **20,810** 条上再算一遍 `choice_confidence`，最大偏差同样是 **0.0**。
所以「`confidence` 是从 `probabilities` 现算的、公式就是这三行」在这条链上是**可以逐位验证**的
（`systemone.py:269-284`、`:311`）。

一个**未复算**的邻项：`noul` 的答案对象里**只有 `noul` 一个字段**（没有 `confidence`，
`to_answer`，`systemone.py:299-300`）——本文只核了这一点（三个文件里每一处 noul 的键集都恰好是
`{type, noul}`），没有别的数可对。

### 14.3 发布清单的 sha256：8 / 8 全过

`runs/2026-10-09_readme-pip-path/manifest.json` 的 `files` 段给了 8 个文件的 sha256。
本文对 committed 的文件逐个重算：**8 个全部相同**（`g31_responses.json`、
`g12_responses.json`、`g31_restart_responses.json`、`job.log`、`pip_freeze.txt`、三份
`start_seconds_*.txt`、`versions.txt`）。`manifest.json` 自己不在那份清单里（它没有给自己一条），
所以「清单覆盖到哪」要按 `files` 的键去数，不能按目录里有几个文件去数。

### 14.4 引擎读数的两个旁证

- **C4 的一致性在大多数 run 上是 0.0**：`runs/*/conformance*.json.gz` 里，`C4.detail.max_abs_delta_p`
  与 `top_answer_flips` 在 1,400 条 items 上大多报 `0.0` / `0`；
  **唯一一条 FAIL 是自陈的**——`runs/2026-10-02_pad-policy-row/conformance_l1.json.gz`，
  `max_abs_delta_p = 0.014737647390295483`、翻转仍是 0。它对应 §8.2 那个被测量过的填充自由度：
  **同一个读法换一种填充位置，答案不变而概率会动**。
- **intent 读数的形状**：`tests/data/intent_readouts/{banking77,clinc150}_examples.npz` 的三份数组
  是 `lp [770, 77]` / `[1500, 150]`、`h [770, 2048]` / `[1500, 2048]`、`y` 一维
  （读 `.npy` 头部即得，未解数据）。770 = 77×10、1500 = 150×10，正是
  `intent_head.MIN_PER_OPTION = 5`（`intent_head.py:34`）与 `serve/tasks.py:36` 那条
  「≥10 个选项才拟合头」的样例规模；`h` 的 2048 维与 `families.py` 里 Qwen 基座的
  `hidden_start = 100_000` 一起，给定 3 次保留列请求（`reserved_chunks`，1024 上限）。

---

## 15. 版本、命名与三处「自己报告来源」

### 15.1 这条路线改过名字

代码里留着**上一版的名字**，而且是逐处兼容读的：

- 线格式前缀：`PREFIX, LEGACY_PREFIX = "decisio", "rlcd"`（`names.py:19`），请求头两种都认
  （`names.py:30-33`），记录格式 `decisio-<kind>/1` 与 `rlcd-<kind>/1` 都收（`names.py:36-51`）；
- 被服务的默认名：`SERVED_NAME = "decisio-qwen3.6-35b-a3b-letters"`，
  `LEGACY_SERVED_NAME = "rlcd-qwen3.6-35b-a3b-letters"`，前者在**任务指纹**里被规范化成后者
  （`names.py:20-21`、`:54-67`），所以**改名之前注册的任务在改名之后照旧生效**；
- 注释里还散着更早的实验目录名（`families.py:89`、`:144`、`:179` 的 `RLCD experiments/...`）。

committed 的产物也带着这个印记：`runs/2026-09-30_served-default/decision_index/*/items.jsonl.gz`
里 `response.model` 是 `rlcd-qwen3.6-35b-a3b-letters`，而同一天的 `/health` 里
`systemone` 段就已经有了当天的字段集合。**所以引用一份 committed 记录之前，先看它的 `model`
字段与 `/health` 的形状**——这一条与「同一个字段名在不同代际是两个量」是同一类坑。

### 15.2 三处「改了数值的东西自己会说」

1. **`usage.output_tokens` 是题目数，不是生成的 token 数**（`systemone.py:600`）。
   `docs/api.md:77` 把它写成契约、`systemone.py:17` 的模块头把映射表整个列出来、
   C2 门断言 `r.usage.output_tokens == len(questions)`（`systemone_conformance.py:182`）。
   **separate 模式下它恰好等于真实采样出的 token 数**（每题 `max_tokens=1`，§4）；
   但 **packed 模式一次 token 都不生成**（池化 runner），**CPU 替身与 MLX 也不生成**——
   同一份代码在这三种情况下仍然报题目数。这不是口径不一，是**这个字段被定义成了「答案的个数」**，
   而它的名字看起来像「生成量」。引用它之前先问是哪种模式（§11 #4）。
2. **两条路由给的不是同一个数，而且头里写着**。`x-decisio-route` 报 `text` / `image`
   （`vllm_engine.py:936`），而 `/v1/answer` 与 `/v1/systemone` 的差别不在这个头里——
   它写在 C4 门的注释与实现里（`systemone_conformance.py:318-321`）：一个是原生读数，
   一个是按 `/health` 报出的温度加过权的。
3. **`/health` 把「当前服务的是什么」整体打印出来**：`profile.temperatures` 逐题型给实际生效的温度，
   `profile.prompt` 给四个布局字段与 `noul_rendering` / `describe_options` / `multi_question` /
   `register_boundary` / `pad_policy` / `pad_unit` 六项，另有 top-level 的 `base`、`block_size`、
   `cache_hit_unit`、`pad_unit` 与 `repository` 段（仓库名 + revision）（`vllm_engine.py:1061-1084`、
   `:360-388`）。
   committed 的 `runs/2026-10-09_readme-pip-path/g31_responses.json` 里这一段是完整的，可以直接
   当「这一次服务用的到底是什么」的存证读。与 H2O-Lightning 的 `/health` 是同一类做法，
   差别是本仓把**全部**策略一起报，而不只是被环境变量覆盖的那两个旋钮。

### 15.3 一条没有被接进来的路径

`consonant_pair_pool` / `order_pool` / `pool_prior_prompt` / `LABEL_SCHEMES`
（`letters.py:37`、`:90-114`）与 `letters_prompt` 的 `pool` 形参（`:159`）在本仓**零调用点**，
而 `letters.py:33` 的注释指着一个**本仓不存在的 CLI flag** `--labels consonant-pairs`。
服务路径走的是 `question_body`（`letters.py:248-263`），它没有 `pool`。
这一条的性质是「**写了、注释还指着它、但没有任何调用点**」——比 Laya 的「接了没收益」更早一步，
与 Jevlike 的 `PlainConvPolicy`（读者在、写者从没有过）方向相反：**这里是作者写全了、注释也写了，
读者从来没有过**。

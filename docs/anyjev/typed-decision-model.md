# AnyJev：同一条读法读两份权重——冻结的任何 LLM，与自蒸馏出来的 Tacit

> 代码库：<https://github.com/nokia-applied-research/AnyJev>
> 对应 README「关于各篇文档」一节

本文按 `anyjev/`（读法、校准、后端、服务）、`scripts/`、`examples/decision_index/` 与 committed 的 `bench/`、`tests/`、`CHANGELOG.md` 的实际内容逐环节说明这条实现的方式；文中出现的文件名与行号（`anyjev/decider.py:557`、`anyjev/readout.py:98`、`anyjev/tacit.py:216` 等）均相对该仓库根目录。与 [RYOTIDE](../ryotide/typed-decision-model.md) 的对照是逐节的。

---

## §0 与 RYOTIDE 的分野

RYOTIDE 是**零训练把一份冻结 LLM 读成分类器**：运行期扫 9 前缀 × 2 表面形式去**标定读位置**，掩码到标签行，重归一化，`marker_mass` 是它的守卫。AnyJev 在这条链上换掉的不止一处，而换得最狠的一处**不是读法**——是**这个仓里同时住着两份权重**。

| | RYOTIDE | AnyJev |
|---|---|---|
| 权重 | 一份冻结的公开 LLM | **两份**：任何开放因果 LM（冻结，库入口 `Decider`），以及自己发布的五档 Tacit（`morriszjm/Tacit-1.7B/2B/4B/8B/9B`，库入口 `Tacit`）——而**读法只有一条**（`Decider(level="raw")` 与 `Tacit` 的 `one_forward` 发出的 prompt 逐字符相同，`tests/test_tacit.py:60-77` 钉住） |
| 「typed」从哪来 | 分词器里的单 token 标签表 | 同样是分词器，但**每个 (题型, K) 现解一次**：choice `A..Z`、score `1..10`、noul `Yes`/`No`；score 的数字不是单 token 时**回落到字母**（`readout.py:35-48`） |
| 读位置 | **运行期标定**（扫前缀 × 表面形式，取 marker mass 最高者） | **没有标定**：chat 模板在 `add_generation_prompt=True` 下的最后一位；没有模板时是字面量 `\nAnswer:`（`readout.py:130-139`） |
| 一次读什么 | 末位词表 logits，掩码到标签，重归一化 | 一样——但 **L0 把这一次读重复 K 次**（选项的 K 个循环移位），log 空间平均，再除以一个无标签估计的先验 |
| 词表维度 | 出现（掩码） | 出现（`log_softmax` 后按 id 取出，`backends/hf.py:385-388`） |
| 诊断量 | `marker_mass` | `answer_mass`——**名字像、用途不同**：它量的是「标签 token 的总质量」，不是「读位置对不对」，因为位置不是找出来的（§5.1） |
| 置信度 | 读出来的 | **两条都有**：读法上**没有温度这个变量**；Tacit 那条是训出来的，而发布路径上**一个数都没被二次改写**（§9） |
| 会生成 token 吗 | 不会 | 读法不会；**升级到模型自己的推理**会，而且每条升级决策都把生成量报出来（`reasoning_tokens`，§4.3） |

一句话立场：**AnyJev 把 RYOTIDE 那条读法做成了一个包，然后在同一个包里放了「谁被读」的两种答案**——读一份冻结的 stock 模型，或者读一份用**它自己的推理**蒸馏出来的检查点。前者是零训练的（`L0` 的排列与先验都不拟合任何参数），后者是全量微调的；而两份权重走的是**同一条 prompt、同一个读位置、同一次重归一化**。所以本文里「读法」与「权重」这两件事要分开看：`§2`–`§8` 讲读法（两份权重共用），`§9`、`§11` 讲权重那一侧的差别。

还有一层不属于读法、但决定了这个仓长什么样：**它是一个库 + 一个模型族 + 一个看板提交，而不是一份服务**。`anyjev/` 3,096 行、`tests/` 1,875 行、`scripts/` 765 行（全仓 6,088 行 Python，实测 `wc -l`）；发布面有四个（库、`python -m anyjev.serve` 的 HTTP 网关、一个 gradio Space、一个提交给 Decision Index 的引擎），另有五个模型仓与一份第三方看板的提交记录。所以本文的「实测」分三类：**在 committed 结果 JSON 上的标准库重算**、**在 committed 实例读数文件上的重算**、以及**借助 Hub API 与 HTTP Range 做的容器算术**——本机没有权重、没有 GPU，也没有跑过上游的 pytest。

---

## 本文的事实来源

| 类别 | 来源 | 本文中的标记 |
|---|---|---|
| 代码引用 | 直接读 `anyjev/**`、`scripts/**`、`space/app.py`、`examples/decision_index/anyjev_engine.py`、`tests/**` | `文件:行` |
| 在 committed 产物上重算 | `bench/results_v01/2026-09-22/**`（11 个 checkpoint × 3 题型的消融）、`bench/results_tacit/2026-10-06/**`（10 个格子）、`bench/results_layout/2026-09-27/**`、`bench/results_decision_index/2026-10-12/**`、`assets/flip_examples.json` | 「实测（committed 产物）」 |
| 公式与常量 | `README.md`、`README.zh-CN.md`、`docs/levels.md`、`docs/rotation_budget.md`、`docs/limitations.md`、`CHANGELOG.md`、`ROADMAP.md`、`pyproject.toml` | `文件:行` |
| 模型事实 | Hugging Face API（`model_info` 的 `safetensors.total` / `card_data` / 逐文件 size / `sha`）、五份模型卡的 `README.md`、`config.json` 与 `model.safetensors.index.json`，以及对单分片 `model.safetensors` 的 **HTTP Range 头部**（§11.2、§14.5） | 「实测（Hub）」 |
| 第三方看板 | <https://benchmarkheaven.com/jev-models/all>（本文核时 board **v1.7.46**，页面自陈 *192 ranked systems*；经代理抓取） | 「实测（看板）」 |

**本文不给出任何由本机前向产生的概率数值**——那需要权重与 GPU，本机都没有。凡引用小数，都是**上游自己在 committed 产物里记下的数**，逐条标出来源；凡`§14` 里的复算，都只用标准库在那些产物上做算术。

---

## §1 端到端数据流

库有两条入口，共用中间那一段。

```
┌─ 入口 A：任意开放 LLM，零训练 ────────────────────────────────────────────────
│ from anyjev import Decider, Question
│ d = Decider(HFBackend("Qwen/Qwen3-8B"))            anyjev/decider.py:37（默认 level="L0"）
│ r = d.decide(state, [q1, q2, …], level="L0")       anyjev/decider.py:153
│     └─ require="L0" → 低于该 level 直接抛 LevelError  anyjev/result.py:87-95
│ m = d.decide_batch(states, q)                      anyjev/decider.py:163（批量先验的路径）
│ d.calibrate_adaptive(q, unlabelled_states, target=0.01)   anyjev/decider.py:261（§6.2）
│
└─ 入口 B：Tacit 检查点，一次前向 + 可选升级 ──────────────────────────────────
  from anyjev import Tacit
  t = Tacit.from_pretrained("morriszjm/Tacit-9B", engine="transformers")   anyjev/tacit.py:313
  d = t.decide(state=…, question=…, options=[…], kind="choice")            anyjev/tacit.py:358
      └─ 批：t.decide_batch([{state, question, options, kind}, …])          anyjev/tacit.py:366
                    │
                    ▼
  ① render_state(state) → 一段文本                              anyjev/state.py:8-20
      str 原样；list[dict(role,content)] → "role: content" 逐行；其余 json.dumps(indent=2)
                    │
  ② 每题：resolve_labels(tokenizer, q) → (标签串, 标签 id)      anyjev/readout.py:35-48
      单 token 校验 + 碰撞校验，不满足即抛 LabelTokenError      anyjev/readout.py:63-85
                    │
  ③ 每次读：build_prompt(state, q, perm, system, labels) → PromptSpec   anyjev/readout.py:98-127
      prefix = ["State:", state, "", "Question: …"] + 题型专属行
      suffix = 每选项一行 "A. <选项文本>"（按 perm 重排）+ "Answer with the letter only."
      split = len(user_prefix)  ← 前 K 次读共享的那一段      anyjev/readout.py:127
                    │
  ④ render_chat_parts(tok, spec) → (prefix_text, suffix_text)   anyjev/readout.py:148-160
      用一个不可见哨兵 U+2063ANYJEV_SPLIT U+2063 在 user 中间切断，再走
      tokenizer.apply_chat_template(add_generation_prompt=True, enable_thinking=False)
      模板不认识 enable_thinking → TypeError → 不带这个开关重来   anyjev/readout.py:130-139
      没有 chat_template → f"{system}\n\n{user}\nAnswer:"        anyjev/readout.py:133-134
                    │
  ⑤ backend.next_token_logprobs(prompts, label_ids) → 每条 [K] 个标签 logprob
      一次前向、只取末位（logits_to_keep=1）、全词表 log_softmax 后按 id 取出
      anyjev/backends/hf.py:57-65、:367-389
      K 个移位共享前缀时走 backend.score_shared（§7）          anyjev/backends/hf.py:93-168
                    │
  ⑥ 每题：_softmax(标签 logprob) → 位置空间的 [P, K]            anyjev/decider.py:30-34
      L0：apply_contextual(÷ 先验^strength) → marginalize(logmean)   anyjev/calibrate/contextual.py:43-47
                                                                    anyjev/calibrate/permute.py:48-66
      raw：只取第 0 个移位                                      anyjev/decider.py:557
                    │
  ⑦ Decision(q, probs, level, diagnostics)                    anyjev/result.py:12-17
      diagnostics 含 answer_mass / raw_probs / l0_probs / permutations / p_pos_raw /
      order_flip_raw / order_flip_l0 / prior_method / prior_strength   anyjev/decider.py:558-578
                    ▼
  ⑧ Tacit 那侧多一步：margin < tau 的决策升级到模型自己的推理   anyjev/tacit.py:379-395
      生成 → 截到 </think> → 追加 "\n\nAnswer:" → 同一套标签读法再读一次
      返回 answer / index / probs / margin / route / first_pass? / reasoning_tokens?
```

### §1.1 四个发布面

| 面 | 入口 | 谁在读 |
|---|---|---|
| 库 | `from anyjev import Decider, Tacit` | 进程内 |
| HTTP 网关 | `python -m anyjev.serve --model … --upstream http://127.0.0.1:8000` | `POST /v1/decide`（单条或 `{"items": […]}`）、`GET /v1/stats`、`GET /health`；`anyjev/serve.py:24-59` |
| gradio Space | `space/app.py`（`huggingface.co/spaces/morriszjm/anyjev`） | 同一条 `Decider`，左边 raw、右边 L0，把选项**反转**再点一次 |
| Decision Index 引擎 | `examples/decision_index/anyjev_engine.py` | 第三方 harness 调 `engine(state, questions)`；**不进 wheel**（`pyproject.toml` 的 `packages.find` 只含 `anyjev*`） |

四个面里只有第二个有 HTTP 形状，而它**没有鉴权**：`serve.py:12` 的模块 docstring 自己写着 *it listens on 127.0.0.1 unless told otherwise and has no authentication of its own*，默认 `host="127.0.0.1"`（`serve.py:68`）。日志被 `log_message` 吞掉（`serve.py:56-57`）。它的价值在 `serve.py:9-11` 那一句：**一个 `Tacit` 服务所有客户端，所以升级上限是共享的**——这是本仓唯一一处「多客户端共享一个状态」的设计（§4.3）。

---

## §2 「typed」从哪来

答案：**分词器，但不是一个固定的标签表，而是每个 (题型, K) 现解一次**。这是它与 decider 的 255 个标签表、Nimble 的 26 个位置分配字母码都不同的地方——这里没有表，只有三个构造函数加一个解析函数。

| 题型 | 标签集 | 上限 | 谁决定 |
|---|---|---|---|
| `choice` | `A`, `B`, `C`, … | **26**（`question.py:9`、`:34-35`，超了抛 `QuestionError`） | `answer_labels` → `LETTERS[:k]`（`readout.py:24-32`） |
| `score` | `1`, `2`, … 或 `A`, `B`, … | 2–10（`question.py:50-56`：bins 或 levels） | 先试数字，**数字不是单 token 就整体回落到字母**（`readout.py:42-48`） |
| `noul` | `Yes`, `No` | 2 | 常量（`readout.py:28-29`），且这两个是**选项名**不是位置名（§2.2） |

解析的落点是 `map_label_tokens`（`readout.py:63-85`）：每个标签先试裸串、再试 `" " + lab`，**两者都必须是恰好一个 token**，否则抛 `LabelTokenError` 并打印实际切分结果；全部标签解析完后还要查一次 id 是否有碰撞（`:83-84`）。这意味着「标签必须单 token」这条约束在这条链上是**fail-loud 的**，不是靠一张预先验过的大表——这一点与 H2O-Lightning 的加载期逐标签探针同类，但它发生在**每次读的时候**（结果按 `(kind, k)` 缓存在 `Decider._label_ids`，`decider.py:189-194`）。

`score` 的回落是整族里少见的形状：**同一个题型在同一个模型上可以有两种标签集**，取决于分词器。`tests/test_question.py:43-64` 用一个 sentencepiece 形状的假分词器（数字切成两块、字母和 Yes/No 是一块）把这条钉住，并在最后断言「什么都切不出单 token 的分词器」直接抛错。后果是一条口径差异：**报出去的 `score` 标签值不能跨模型比较**（Mistral 那条链给的是字母，Qwen 给的是数字），而 `Decision.value` 是标签中心点的加权和（`result.py:49-53`、`question.py:82-90`）——它本身不受标签形式影响，但那列原始分布是。

### §2.1 上限只有 26，而两位数的读法只活在 examples 里

`Question.choice` 硬性拒绝 27 个以上选项（`question.py:34-35`），注释写着 *letter-readout limit; span readout will lift this*，而 `ROADMAP.md` 的「Next」一节也把 span readout 列为未做项。但**同一条路线在提交给 Decision Index 的引擎里已经越过这条线**：`examples/decision_index/anyjev_engine.py:120-126` 在选项超过 26 时走 `_two_level`——每个选项拿一个「字母 + 数字」码（`A0`…`A9`、`B0`…），

```
p(option) = p(letter) · p(digit | letter)          examples/decision_index/anyjev_engine.py:196-222
```

先读一次「用到的那些字母」，再对每个字母各读一次「十个数字」（两次读都限制在各自集合上重归一化），上限因此变成 `26 × 10 = 260`（`anyjev_engine.py:41`），超过 260 就报 `Unsupported` 而不是截断（`:121`）。这个能力**不在 wheel 里、不在库的 API 上**，只在那个提交物里——也就是 `docs/comparison.md` §4.5 里 AnyJev 那一行的第③条。

### §2.2 `noul` 的标签是选项名，所以要反着映射

choice 与 score 的标签是**位置名**（`A` 是第一个位置），noul 的标签是**选项名**（`Yes` 永远是 Yes）。于是 `label_ids_for_perm` 分了两支（`readout.py:51-60`）：

```python
def label_ids_for_perm(q, ids, perm):
    if labels_are_positional(q):        # choice / score
        return list(ids)                # 按位置读 → id 序不变
    return [ids[i] for i in perm]       # noul：perm[j] 是第 j 个位置上写的是哪个选项
```

而 noul 的 perm 只交换**措辞顺序**：`Answer Yes or No.` → `Answer No or Yes.`（`readout.py:108-111`）。这样做的后果是后端返回的下标 `j` 永远意为「第 j 个位置上显示的那个选项」，与 choice 的语义一致。`tests/test_decider_fake.py:97-108` 用一个没有任何偏置的假模型断言：两种措辞下 Yes 的概率逐位相同（差 < 1e-9），`order_flip_raw == 0.0`。上游在 `CHANGELOG.md` 的 0.0.2 节把这条当一个 bug 记着：*a position-bound readout silently swapped Yes/No in the second phrasing*。

### §2.3 自蒸馏那一侧的「typed」是同一套，但训练把答案的形状钉死了

Tacit 的三类题型与库里的 `Question` 是同一套（`tacit.py:68-81` 的 `make_question` 就是构造 `Question`），训练用的 prompt 也由库自己的 `build_prompt` 生成——`tests/test_tacit.py:46-57` 把三个题型的完整 prompt 字符串逐字符钉在测试里，注释写着 *the prompt the models were trained on*。所以这一节对两份权重都成立，差别只在**权重是照着这条 prompt 训出来的**：库改了措辞，Tacit 的读数就失效，而冻结模型那条不受影响（只改分布）。`AGENTS.md` 的「Ask a human about」一节把这一点写成了协作规则：*Any change to … Tacit's one-forward prompt* 要先问人。

---

## §3 读位置

**读位置不是找出来的，是构造出来的：chat 模板在 `add_generation_prompt=True` 下的最后一位。**

```python
# anyjev/readout.py:130-139
def _render(tokenizer, system, user):
    template = getattr(tokenizer, "chat_template", None)
    if not template:
        return f"{system}\n\n{user}\nAnswer:"          # 没有模板 → 字面量 Answer:
    kwargs = dict(tokenize=False, add_generation_prompt=True)
    try:
        return tokenizer.apply_chat_template(messages, enable_thinking=False, **kwargs)
    except TypeError:                                  # 模板不认识这个开关
        return tokenizer.apply_chat_template(messages, **kwargs)
```

而真正被读的位置在后端里，是两个字符的事：

```python
# anyjev/backends/hf.py:57-65（Tacit 的 TransformersEngine 同构，tacit.py:143-146）
out = self.model(**enc, position_ids=pos, logits_to_keep=1)   # 只算末位的 logits
return out.logits[:, -1, :].float()
```

于是这条链上：

- **没有「扫位置」这一步**，因此 RYOTIDE 的 `marker_mass` 守卫没有对象——`answer_mass` 虽然也量「全词表里落在标签上的质量」，但它不参与选位置（§5.1）。
- **「位置对不对」在原理上不可诊断**，只能靠「prompt 是不是它训练时见到的那个形状」。上游付的代价与 metask-jev-rain 同类（读位置由模板文本钉住），但这里连一条身份哈希都没有：**没有 prompt 哈希、没有版本端点、`/health` 只回 `{"ok": true}`**（`serve.py:37-38`），库入口也不带任何身份字段。
- 两个读法的读位置**不是同一个字符串位置**：一次前向那条读完的是模板尾巴（`<|im_start|>assistant\n` 之类），而升级后的那条读的是**手工追加的** `"\n\nAnswer:"` 之后（`tacit.py:164`、`:197`、`:252`）。这不是同一个前缀，也正是下一节那个表面形式分叉的成因。

### §3.1 同一个标签在两个读法上取的是不同的 token

标签 id 的解析有**两个方向**，分属两条读法：

| 读法 | 函数 | 优先试哪个表面形式 | 位置 |
|---|---|---|---|
| 一次前向 | `map_label_tokens` | **裸串** `A` → `" A"` | `readout.py:72` |
| 升级后重读 | `_spaced_label_ids` | **带空格** `" A"` → `A` | `tacit.py:111-124` |

理由写在代码的形状里：一次前向的 prompt 由模板收尾（末 token 后面没有空格），升级那条的 prompt 结尾是字面量 `Answer:`（模型要输出的是一个前导空格）。两条都保留回退，两条都查碰撞。所以第 #2 条不变量（前缀与标记表面形式耦合、必须成对扫）在这条链上被**改写**成：**不是扫，而是两条读法各自声明自己那侧的形式并优先取它，取不到（不是单 token）就抛错**。

### §3.2 随权重发布的那份副本多一个模板尾巴补丁

每个 Tacit 模型仓里都躺着一份 `tacit_decision.py`（27,868 B，五个仓**逐字节相同**，sha256 见 §14.4），它是这条读法的第二份副本。两份在渲染那一步已经分叉：副本里有这一段，包里**没有**：

```python
# tacit_decision.py:409-410（各 Tacit 模型仓，不在 anyjev/ 里）
if text.endswith("<|start|>assistant"):          # harmony-format models: read the final channel
    text += "<|channel|>final<|message|>"
```

也就是说：**模板尾巴如果不是答案位置，这份副本会把它手工拨到答案通道上**，而包里那版会照原样读——读到的就是通道标记后面的位置。同一个坑，两个副本两种命运（§4.1 表）。

---

## §4 一次前向

「一次前向」在这条链上有两处**字面**成立：`logits_to_keep=1`（`backends/hf.py:60-63`，注释解释了为什么——600 token 的 prompt 在 batch 32 下会物化约 10 GB 的 logits），以及 `next_token_logprobs` 只取 `logits[:, -1, :]`。没有任何位置被采样、被解析、被生成。

### §4.1 六个后端，同一件事的六种接法

`backends/base.py:11-31` 把协议压到**一个方法**（外加两个可选方法）：

```python
def next_token_logprobs(self, prompts, token_ids) -> List[np.ndarray]:
    """… taken from the full-vocabulary log-softmax at the last position.
    Decision mode never samples; a backend that cannot expose restricted
    next-token logits is not a supported backend."""
```

| 后端 | 怎么拿到标签的 logprob | 文件 |
|---|---|---|
| transformers | 左填充 + **显式 `position_ids`**（`cumsum(mask)-1`）+ 传 mask；全词表 `log_softmax` 后按 id gather | `backends/hf.py:367-389` |
| vLLM（OpenAI 服务） | `max_tokens=1, temperature=0.0, logprobs=K, allowed_token_ids=标签`；**要求服务端 `--logprobs-mode processed_logprobs`**，否则缺标签即抛 `LabelTokenError` | `backends/vllm.py:58-83` |
| SGLang（原生 `/generate`） | `max_new_tokens=0` + `return_logprob=True` + `token_ids_logprob=标签`；按 id 从 `meta_info["output_token_ids_logprobs"][0]` 映射，缺一个即抛 | `backends/sglang.py:44-70` |
| llama.cpp（GGUF） | 清空 KV → 只给最后一个位置要 logits（`b.logits[len-1] = 1`）→ 自己算全词表 log-softmax | `backends/llamacpp.py:173-215` |
| 进程内 vLLM | `SamplingParams(max_tokens=1, temperature=0.0, logprobs=20)`，标签从 top-20 里取，**不在 top-20 计 0** | `tacit.py:168-189` |
| `vllm serve` 客户端 | 同上，但走 `/v1/completions` 且 `return_tokens_as_token_ids=True`，按 token id 映射 | `tacit.py:235-246` |

三种「标签缺失」策略在同一个包里并存，而且都写在注释里：库的 `VLLMBackend` **抛错**（0.17.1 上遇到过 Nemotron-H 的 top token 开始写思维链，`CHANGELOG.md` 记在 `#13`）、`Tacit` 的 vLLM 引擎**计 0**（`restricted_logprobs` 把 `-inf` 变成概率 0，`tacit.py:59-65`）、提交给看板的引擎**计 0 但只在「一个标签都没有」时抛**（`anyjev_engine.py:45-62`，注释写明理由：`processed_logprobs` 下缺失只可能是 `-inf` 与掩码 token 并列）。同一个失败模式，三个面三种答案——这与 Nimble 那三个服务面是同类麻烦，只是这里分歧在**读数的容错**上。

`FakeBackend`（`backends/fake.py`，266 行）是第七个、也是唯一能上 CPU 的：它**解析 `readout.build_prompt` 打出来的真 prompt**，把 `logit(position j, option o) = content(state,o) + position_bias[j] + label_prior[label] + logit_noise·n(state,o,j)` 当模型，并把 `logits` 之外还塞了一个 `-5.0` 的幽灵条目再算全词表 log-softmax（`:262-264`），好让 `answer_mass` 在测试里不是恰好 1。它还植入了隐藏状态（dims 26:38 是 state 的哈希、38:50 是最优选项的哈希，`:137-149`），供被删掉的 L2 头与本文 §10 那条早退路径用。

### §4.2 L0 = 同一条读法读 K 次

`Decider` 的默认 level 是 `L0`。它做的事全部发生在读数之后：把选项按 K 个循环移位各渲染一遍（`calibrate/permute.py:24-27`），每次读出一份位置空间的分布，然后

```python
# anyjev/calibrate/permute.py:48-66（"logmean"）
per_option[s, i] = p_by_perm[s, j]                  # 位置 → 选项
z = np.log(np.clip(per_option, EPS, None)).mean(0)  # 几何平均
out = np.exp(z - z.max()); out /= out.sum()
```

docstring 把为什么是它讲清楚了：如果偏置在 logit 空间是可加的（`logit(i at pos j) = c_i + b_j`），那么逐选项取平均后 `softmax(c)` 精确成立——**位置偏置被消掉，且结果与选项当初怎么列无关**。这也是为什么 `assets/flip_examples.json` 里 `l0_argmax` 是**一个**值而不是一对：反转选项不产生第二条读数，因为 L0 的答案本来就与列法无关（§14.1）。

先验那一侧同样是位置索引的：`_running[q.key]` 存的是 `[P, K]` 的位置空间均值（`decider.py:149`、`:546-547`），而修正发生在**marginalize 之前**（`decider.py:565-567`）——所以这个「标签先验」实际上同时携带着位置先验的成分。committed 的消融能把这两者分开（§6.2）。

### §4.3 第二条路：把不确定的题升级到模型自己的推理

这是整条链上**唯一会生成 token 的地方**，而且是明说、计量、并且被限流的：

```
margin = logp[top1] - logp[top2]                       anyjev/tacit.py:439-448
margin < tau（默认 0.5）→ 候选，按 margin 从小到大排序      anyjev/tacit.py:382
按 max_cot_share（0.2）在最近 cot_window（1000）条里取整数个 → chosen   anyjev/tacit.py:407-420
chosen：COT_SYSTEM + cot_user(...) → 开思考生成 → 截到 </think> → 追加 "\n\nAnswer:" → 再读一次
                                                        anyjev/tacit.py:390-395、:127-131
返回 route="cot"，带 first_pass（原来那次一次前向的结果）与 reasoning_tokens（生成了几个 token）
                                                        anyjev/tacit.py:395
```

四处细节值得单列：

- **上限是「窗口内取整」，不是「累积配额」**：`_budget` 用 `math.ceil`（`tacit.py:415`、`:418`、`:420`），所以一个安静的时期不会攒下预算；`tests/test_tacit.py:137-150` 用 10,000 条自信 + 2,000 条不确定的两段流把两种语义分开测：窗口语义下最后 2,000 条里最多升级 400 条，`cot_window=None` 的累积语义下**2,000 条全升级**。默认是窗口，理由写在 docstring 里（`tacit.py:408-411`）：*unused budget does not pile up over a long run*。
- **配额在锁里结算、推理在锁外跑**（`tacit.py:379-395`）：所以并发调用共享一个上限，而推理不互相阻塞。`tests/test_tacit.py:160-173` 用 8 个线程 × 100 条断言 `escalated <= ceil(0.2·800)+1`。
- **升级后仍然是分布，不是文本**：`Answer:` 之后读的还是那套标签 id，所以 `probs` 照样加起来是 1，`tests/test_tacit.py:120-125` 断言升级决策也带 `first_pass` 与 `margin`。这是「不生成文本就没有解析失败」这条设计在**生成路径上**的延续。
- **推理的采样参数是另一套旋钮**（`temperature=0.6, top_p=0.95, top_k=20`，`tacit.py:293-304`），而读数那条请求一律 `temperature=0.0`。全仓 `temperature` 的出现位置只有这两类（§9.2）。

---

## §5 掩码 + softmax

**这条链上没有掩码这一步**——但这不是因为它不读词表（它读），而是因为**它不做「把越界的质量挡回去」这件事，它只重归一化**：

```python
# anyjev/decider.py:30-34
def _softmax(lp):
    z = np.asarray(lp, dtype=np.float64); z = z - z.max()
    p = np.exp(z); return p / p.sum()
```

`lp` 是后端给出的**标签 token 的 logprob**（来自全词表 log-softmax 的 gather，`backends/hf.py:385-388`），所以「越界质量」并没有被掩掉，它只是没被读：`answer_mass` 会把它量出来（下一节）。这与 Kev / decider / Nimble / pplx-decider / RSI-Jev / JevAny 那一类「头只输出 K 个标量、词表维度从不出现」不同：**这里的词表维度是出现过的，只是被 gather 掉了**——与 JevAny 的 `choice` 读法、metask-jev-rain 的 gather 同属一类，本族里已经收过三次。

第二个重归一化实现在 Tacit 里，语义多一条：

```python
# anyjev/tacit.py:59-65
z = np.asarray(values, dtype=np.float64)
if not np.isfinite(z).any():     # 一个标签都没回来
    z = np.zeros_like(z)         # → 均匀分布，而不是 NaN
m = np.max(z[np.isfinite(z)])
return z - (m + np.log(np.exp(z - m).sum()))   # -inf 留在原地 → 概率恰为 0
```

所以「标签缺失」在库路径上是**抛错**，在 Tacit 路径上是**概率 0**，在「全缺」这一档上是**均匀分布**（`tests/test_tacit.py:107-110` 把后两条钉住）。三档行为都写在测试里，但没有一条会静默给一个看似合理的错数——这是它与 `VLLMBackend` 那处 `-30.0` 兜底（`CHANGELOG.md` 的 `#13`，已修）的区别。

### §5.1 `answer_mass`：诊断量在，但它量的不是位置

```python
answer_mass = float(np.exp(lp_real).sum(axis=1).mean())      anyjev/decider.py:556
```

它是**标签 token 在全词表里的总质量**，按移位平均。用途与 RYOTIDE 的 `marker_mass` 不同：那条链用它**选位置**，这条链用它回答「prompt 有没有落地」——`docs/levels.md:27-29` 把它列在诊断量第一条，Space 界面直接把它印出来（`space/app.py:76`：`label-token mass {…:.3f}`）。它的取值范围与含义可以在 committed 产物上量：假后端里那个 `-5.0` 幽灵条目的意义正在于此（§4.1）。

### §5.2 三条「不适用」在同一个仓里各有各的理由

| 族的常见约束 | 在这条链上 |
|---|---|
| 标签必须单 token | **成立，而且是 fail-loud 的**（`readout.py:63-85`；两条读法各查一次，§3.1） |
| 字母表 / 码表上限 | **26 / 260 两个数**：库 26（`question.py:9`），提交给看板的引擎 260（`anyjev_engine.py:41`），后者不在 wheel 里 |
| 选项文本必须单 token | **不适用**：选项文本只进 prompt，从不进读法（唯一被读的字符串是标签） |
| 掩码到已知标签集 | **不适用**（不是掩码，是 gather + 重归一化） |
| 重归一化 | **成立，对象是标签**（两处实现，§5 开头） |

---

## §6 顺序偏置

**训练期不做，运行期用读 K 次来消。** 这是这条路线与「训练时打散」那一组（Laya / JevAny / metask-jev-rain / RSI-Jev）最直接的分别：读法本身零训练，所以它只能靠**多读几遍**。

默认就是全周期：`max_permutations=None` → `cyclic_shifts(k)` 给 K 个移位（`permute.py:24-27`），每个选项在每个位置上恰好坐一次。`combine="logmean"` 把加法偏置精确消掉（§4.2）。可选的三个开关都在 `Decider.__init__` 上，而且**默认全关**：

| 开关 | 默认 | 做什么 | 代价 |
|---|---|---|---|
| `max_permutations=n` | `None`（=K） | 只读前 n 个移位 | 部分预算下位置偏置不再被抵消，答案可能依赖调用方的列法（`tests/test_stopping.py:128-143` 是这条的对照实验） |
| `adaptive_shifts=True` | `False` | 按 `spread_order`（van der Corput）逐个读，够自信就停 | 需要一条阈值；未标定时用 `DEFAULT_LOG_MARGIN=8.5`，那是个在**真实模型 K=18–20** 上证过的数，小模型上等于「读满」（`tests/test_stopping.py:146-170`） |
| `canonical_order=True` | `False` | 把移位应用在**按选项文本排序**的规范列表上，而不是调用方的列表 | 让 prompt 成为选项**集合**的函数——重列同一组选项在任何预算下给出同一组概率（`decider.py:436-453`；`tests/test_stopping.py:107-125`） |

`canonical_order` 的理由值得照抄一遍（`decider.py:444-451`）：读满 K 个移位确实让每个选项走遍每个位置，但**哪些选项挨着**仍然跟着调用方的顺序，而选项之间会互相注意；预算不满时就更明显。这一条是本仓自己提出的、RYOTIDE 四条不变量之外的**新不变量**：「答案与选项的列法无关」在部分预算下要靠「旋转规范列表」来保住。

### §6.1 旋转预算：一条不需要标签的停止规则

这是这条路线里工程含量最高的一段（`docs/rotation_budget.md` 是它自己的专题页，代码在 `anyjev/calibrate/stopping.py`）：

1. **统计量换了**。概率间隙在 1 处饱和，恰好在规则最需要判断的区域失去分辨力；改成 **log-odds margin**（`log_margin`，`stopping.py:49-57`，对归一化不变，所以可以直接喂 marginalize 的输出）。
2. **目标换了**。不去认证准确率（那要标签），而是认证「与**我们自己的全 K 答案**一致」——所以可在任何未标注的状态集上测（`stopping.py:1-32`）。
3. **判据是置信上界，不是点估计**。`choose_threshold` 在 `np.arange(0, 20.01, 0.25)` 上找最便宜的阈值，要求 **Clopper-Pearson 95 % 上界** ≤ 目标（`stopping.py:106-135`，`cp_upper` 在 `:78-92` 用二项尾概率二分）。300 条状态在 1 % 目标下允许 3 次不一致，而恰好拟合到 3 次就是过拟合——`tests/test_stopping.py:47-52` 把这个差写成了断言：`3/300 == 0.01` 但 `cp_upper(3,300) > 0.02`。
4. **标不出来就不标**。`threshold=None` 时读满全周期（`tests/test_stopping.py:191-199`：24 条状态哪怕零不一致也只能把速率界到 12 %，因为下界是 `1 - delta^(1/n)`，所以它如实返回「没有阈值」）。
5. **一次读多个**。`adaptive_wave`（默认 2）一轮要 w 个移位；上游为这个默认值给了两组实测（vLLM 上 wave 1 与 2 发出同样多的请求、墙钟差 1.7×；本地两者打平，更宽的波在两边都更差），写在 `decider.py:95-103`。

### §6.2 committed 产物里能把这套消融拆开读

`bench/results_v01/2026-09-22/` 下 11 个 checkpoint × 3 个题型，每个都记了**七个 level**：`raw`、`L0-perm`（只旋转）、`L0-bc`（只批先验）、`L0-cf`（只内容无关先验）、`L0-perm+bc`、`L0-perm+cf`，以及 `L0`（= `L0-perm+bc`）。以 Qwen3-8B / banking20（K=20，n=300）为例，「实测（committed 产物）」：

| level | acc | flip（反转选项） | ECE | cov@5% |
|---|---|---|---|---|
| `raw` | 0.7467 | 0.2300 | 0.2397 | 0.0767 |
| `L0-perm` | 0.8000 | 0.0767 | 0.1890 | 0.3367 |
| `L0-bc` | 0.7500 | 0.2233 | 0.2375 | 0.4800 |
| `L0`（= perm + bc） | 0.8033 | 0.0733 | 0.1839 | 0.4633 |
| `L0-perm+cf` | 0.8033 | 0.0800 | 0.1851 | 0.1633 |
| `L0-cf` | 0.7333 | 0.2433 | 0.2527 | 0.0133 |

这张表说的是两件事，而且与 README 那张「raw vs L0」的两行并不重复：**翻转率几乎全是 perm 带来的**（0.2300 → 0.0767；只加批先验只到 0.2233），**覆盖率与 acc 则要先验**（`L0-bc` 单独把 cov@5 % 抬到 0.48，而 `L0` 是 0.4633）；`L0-perm` 单独也已经把 cov 抬到 0.3367。另一头，**内容无关先验在 noul 的提示注入题上是另一个故事**：同一个 checkpoint 的 `injection`（K=2）上 `L0-perm+cf` 的 acc 是 **0.8067**，而 `L0`（批先验）是 **0.7000**、`raw` 是 0.6933——`calibrate/contextual.py:12-15` 的 docstring 把这个方向写成了它自己的说明：*+8 to +12 points on a prompt-injection noul, but -3 on ordinal scores*。而 README 的「raw vs L0」两行只引 `raw` 与 `L0` 两个数（`tests/test_readme_numbers.py:41-48` 就是查这两个数）。

同一批 JSON 里还有一列**现在已经不存在的 level**：`L1`（Qwen3-8B / banking20 上 acc 0.8067、ECE 0.0947）。它在这份产物里是温度缩放的读数，而 `LEVELS` 从 0.3.0 起是 `("raw", "L0")`（`decider.py:26`），`tests/test_require.py:34-39` 把 `L1`/`L2` 当成未知 level 断言报错（§9.3）。

---

## §7 多问题共享 state

**答案：同一次调用里的多道题共享一次「服务端往返」，但不共享计算，也不共享 state 的账。**

`Decider.decide(state, [q1, q2, …])` 把一个 state 上的所有题放进同一次 `_run`（`decider.py:153-161`）。`_run` 的实现是：

1. 先把**每道题 × 每个移位 × （可选）每个内容无关探针**的 prompt 全部登记进两个注册表（真实 prompt 与探针 prompt 各一个，`decider.py:482-514`）——注册表按**渲染后的文本**去重（`:490-495`），所以同一段 state 在批里出现两次不会重复算；
2. 再一次性打分（`:524-525`），走共享前缀那条路（见下）；
3. 最后逐题组装。

三处代价与三处省法：

- **不省账**：state 的 token 在每个 prompt 里各出现一次（K 个移位 × 每移位一条 prompt），没有「计费只算一次」这回事——与 RYOTIDE 的 `run_many()` 是一个对比。
- **题与题之间不共享**：每道题有自己的 prompt 前缀（`Question: …` 那一段不同），所以两道题的前缀在两个不同的组里，各自算一次。这与 decider 的「K 个槽位落在同一行、多问一题不改任何已有答案」是相反的取舍。
- **移位之间才共享**：`render_chat_parts` 用一个不可见哨兵把 user 内容切成 `(prefix, suffix)`（`readout.py:88`、`:148-160`），同一 state 的 K 个移位共享同一个 prefix；后端实现了 `score_shared` 时，前缀的 KV 算一次，K 个短后缀批着算（`backends/hf.py:93-168`：左填充前缀 → 显式位置 → `_repeat_cache` 复制 K 份 → 右填充后缀 → 每行取自己的 `last`）。

分组有三条判据，都在 `decider.py:201-210`：同一 `(prefix, suffix 的标签 id 序列)`、组内至少 **3** 个后缀（`shared_prefix=True` 时放宽到 2）、以及 `prefix` 至少 **256** token（`"auto"` 时；docstring 给的理由是低于这个长度，一次批着的 plain forward 更便宜）。**而且它有精确性判据**：

```python
# anyjev/backends/hf.py:108-114
exact = all(tok.encode(prefix + sfx, add_special_tokens=False) == p_ids + s_ids …)
if exact and len(suffixes) > 0: plan.append(...)
else: self.shared_fallbacks += 1; fallback.extend(...)   # 不精确 → 走普通路径，并计数
```

`shared_fallbacks` 是公开计数（对拍脚本 `scripts/shared_prefix_parity.py:64` 会打印它）；`scripts/shared_prefix_parity.py` 是它的对拍脚本，打印 `max|dlogp|`、`max|dlogsoftmax|`、argmax 一致数与回退组数。一处细节值得注意：**noul 的 K 个措辞永远不会被分到同一组**，因为每个措辞的标签 id 序列不同（`tests/test_shared_prefix.py:69-72` 断言这一条）。

---

## §8 服务化 / fail-loud 风格

| 面 | 形状 | 错误面 |
|---|---|---|
| 库 | `decide` / `decide_batch` / `calibrate_adaptive` | `QuestionError`（选项数、重复、题型）、`LabelTokenError`（多 token / 碰撞 / 一个标签都取不到）、`LevelError`（`require=` 低于实现 level）、`ValueError`（配置项）、`TokenMismatchError`（GGUF 与 HF 分词器不一致） |
| HTTP 网关 | 标准库 `ThreadingHTTPServer`，零依赖 | 400（`ValueError`/`KeyError`/`TypeError`）、502（上游 `RuntimeError`）、404（其它路径）、`/health` 只回 `{"ok": true}`；`serve.py:43-54` |
| 看板引擎 | 第三方 harness 的 `Engine` | 超窗 → `Unsupported`（**拒绝，不截断**，`anyjev_engine.py:160-165`、`:173-174`） |
| Space | gradio，ZeroGPU 感知 | 选项少于两个时返回提示串（`space/app.py:62-63`） |

四处 fail-loud 的判据值得单列：

1. **prune 掉「看起来对」的数**：`serve.py:53-54` 只在 `RuntimeError` 上回 502，而 vLLM 服务不可达正是这条（`tacit.py:231-232` 把 HTTPError 包成 RuntimeError 并带上上游正文）；
2. **不静默截断**：看板引擎在发请求前量一次渲染长度，超窗直接 `Unsupported`（`anyjev_engine.py:160-165`），HTTP 400/413 且正文含 `context` 时也转成 `Unsupported`（`:184-188`）；
3. **同一段文本两种分词器必须逐 token 相同**：`LlamaCppBackend` 在传了 HF 分词器时，每个 prompt 都比对两侧 token 序列、每个标签 id 都比对两侧解码文本（`backends/llamacpp.py:134-164`），不一致即 `TokenMismatchError` 并打印第一处差异；
4. **`anyjev.truncate` 会写一份来历**：截断后的检查点旁边落一个 `anyjev_truncation.json`，记 `anyjev_truncated_from` / `kept_blocks` / `original_blocks` / `dropped_tensors`（`truncate.py:79-83`），因为「一个截断模型不是它名字所指的那个模型」。

它**没有**的东西同样成体系：没有鉴权、没有版本端点、没有 prompt 哈希、没有 `usage` 字段、没有 `/metrics`。能自证身份的只有三处——响应里的 `route`、诊断里的 `level`，以及（看板那条链上）服务端本地路径形式的 `checkpoint.directory` 之类字段。这与 NanoJev 那篇记的「一份权重可以彻底没有身份指纹」是同一个观察，只是这里多一层：**库连一个 `env`/版本块都不往响应里放**（`Decision.to_dict` 只有 kind / level / answer / confidence / distribution / value，`result.py:60-74`）。

---

## §9 置信度从哪来，又怎么被改写

### §9.1 「谁在改发布数值」的链表

这条链上**没有任何拟合出来的标量**参与读数。按顺序数：

| 环节 | 改数值？ | 改 argmax？ | 判据 / 来源 |
|---|---|---|---|
| ① 训练本身（**只有 Tacit 那侧**） | 是 | 是 | 自蒸馏；训练代码不在任何发布物里，只有模型卡的叙述（§11.1、§15） |
| ② `prior_strength`（`batch` 默认 **0.75**，`content_free` **1.0**） | 是 | **可以** | 离线在 230 个 (model, question) 单元上回放选出来的（`decider.py:49-52`、`CHANGELOG.md` 的 0.1.0 节），不是在线拟合 |
| ③ `prior`（batch / content_free / none） | 是 | 可以 | 无标签估计：批内均值（≥ `min_prior_n=8` 条）或三个内容无关探针（`calibrate/contextual.py:23`） |
| ④ `combine`（默认 `logmean`） | 是 | 可以 | 数学性质（加法偏置下精确），不是拟合 |
| ⑤ `adaptive_margin` / 标定出来的阈值 | 是 | **可以**（它决定读几个移位） | `calibrate_adaptive` 的 Clopper-Pearson 认证，或 `DEFAULT_LOG_MARGIN=8.5` |
| ⑥ `canonical_order` | 是（改 prompt） | 可以 | 构造性（让答案与列法无关） |
| ⑦ Tacit 的 `tau` + `max_cot_share` | 是 | **可以**（决定哪几题被重答） | 固定默认（0.5 / 0.2 / 1000） |
| ⑧ 升级推理的采样参数（T 0.6 / top-p 0.95 / top-k 20） | 是 | 可以 | 请求侧默认 |
| ⑨ **温度缩放** | **不存在** | — | 0.3.0 删掉了 L1（§9.3） |

也就是说：**读数本身（②③④）全是无标签估计，⑦⑧ 只作用在「哪些题被重新回答」上，而 ① 只存在于自蒸馏那份权重里**。这与 RYOTIDE「读出来的 + 只有一个温度能动数值」是同族的结论，但形状不同：这里一个拟合出来的标量都没有，代价是**多了一整个「读几遍」的预算维度**（⑤⑥），而那个维度是能被认证的（§6.1）。

### §9.2 温度这个变量在本仓存在，但不在读法上

`grep -rn temperature anyjev/` 的结果只有两类（实测）：

- **读数请求**：一律 `temperature=0.0`——`tacit.py:180`（vLLM 进程内）、`:236`（`vllm serve` 客户端）、`backends/vllm.py:60`、`backends/sglang.py:47`；SGLang 那条连 `max_new_tokens` 都是 0。
- **升级推理的采样**：`temperature=0.6, top_p=0.95, top_k=20`（`tacit.py:293-304`）。

再加一处**合成模型上的测试旋钮**：`FakeBackend(temperature=…)` 把 logits 除以它（`backends/fake.py:87`、`:144`、`:261`），`tests/test_require.py:26` 用它造一个「概率更尖」的假模型。所以：

- 引用 AnyJev 的概率时**不存在「先去确认 T」这个动作**——不是「T 恰好是 1」，而是**读法上没有这个变量**（与 Jevlike / CLM / Clef 同类，但理由各不相同：那三条是链上没有那个位置，这条是有温度、只是它管采样不管读数）。
- 反过来，**⑦⑧ 那两个旋钮是真的会改答案的**，而它们既不随权重发布、也不在任何 committed 产物里记录（`bench/results_tacit/2026-10-06/*.json` 的 `settings` 记的是 `adaptive/tau/max_cot_share/cot_window/cot_max_tokens/block/order`，没有采样参数）。所以「这条 `cot` 决策的答案是哪个温度下生成的」在这条链上**无从核对**——这是 §5 里该记的一条口径。

### §9.3 已删除的 L1 仍留在 committed 产物里

`bench/results_v01/2026-09-22/**` 的每个任务都有一列 `L1`（温度缩放：Qwen3-8B / banking20 上 ECE 0.1839 → **0.0947**、cov@5 % 0.4633 → 0.5200），而 0.3.0 的 `LEVELS` 是 `("raw", "L0")`，`require="L1"` 会抛 `ValueError`（`tests/test_require.py:34-39`）。`README.md:158-160` 只在 `anyjev==0.2.0` 那一行提它。所以：**这份仓里跑不出 `L1` 那一列**，引用它会得到「AnyJev 的 ECE 是 0.09」这样的错数——而同一份 JSON 里 `L0` 的 ECE 是 0.1839。这与 RSI-Jev 那篇记的「盘上有、装进去了、但没接上」是相反的形状：**这里是算得出来过、代码里已经没有了，而产物还在**。

---

## §10 四条不变量的落位

RYOTIDE §13 那四条在这条链上落得比较整齐，但第 #2 与第 #4 要改写：

| 不变量 | 落位 |
|---|---|
| **#1 读位置必须落在标记上（诊断量 `marker_mass`）** | #1 的**前半（诊断量存在）成立，后半（用它选位置）不成立**。`answer_mass` 在（`decider.py:556`），但位置不是选出来的，是模板尾巴定的（§3）。所以这条要改写成：**「读位置由构造决定，诊断量改为量「prompt 有没有落地」」** |
| **#2 前缀与标记表面形式耦合，必须成对扫** | **改写**：不扫。两条读法各自声明自己那侧的表面形式并优先取它（一次前向取裸串、升级后取带空格，`readout.py:72` / `tacit.py:111-124`），取不到就抛 `LabelTokenError`。而前缀那一侧彻底不动：**没有 `SUFFIXES`/`PATTERNS` 这类可扫的自由度** |
| **#3 只读最后一个位置** | **字面成立**（`logits[:, -1, :]`、`logits_to_keep=1`）。L0 不是「读多个位置」，而是**同一条读法读 K 次**——这与 decider 的「一次前向读 K 个槽位」、RSI-Jev 的「1 + K 个位置」是第三种改写 |
| **#4 比较概率 / ECE 前先确认 temperature** | **改写为**：①确认这条分布是 `raw` 还是 `L0`（同一 prompt 的两个 level，中间差着 K 倍的计算）；②确认它是**哪份权重**读出来的（冻结基座还是 Tacit 检查点——同一个 prompt、同一条读法，两个分布）；③如果它来自 `route="cot"`，那么它还带着一条**无从核对的采样温度**（§9.2） |
| （各自新增的） | ①**答案与选项列法无关**：读满 K 个移位时是数学性质，部分预算下要靠 `canonical_order`（§6）；②**「读几次」可以被认证**：停止阈值以「与自家全 K 答案一致」为目标，用 Clopper-Pearson 上界，不用标签（§6.1）；③**「标签集」是每题现解的产物**，`score` 的标签甚至可以整档变成字母（§2） |

---

## §11 模型描述核对（以 Hub 为准）

### §11.1 五个 Tacit 与它们的基座

五个仓的 `sha`、参数量、权重体积与逐文件体积，「实测（Hub）」：

| 仓 | sha | 模型卡 `base_model` | 参数量 | 权重文件（十进制） | 张量数 |
|---|---|---|---|---|---|
| `morriszjm/Tacit-1.7B` | `1cb46fff…` | `Qwen/Qwen3-1.7B` | 1,720,574,976 | 3.44 GB | 310 |
| `morriszjm/Tacit-2B` | `30d2b954…` | `Qwen/Qwen3.5-2B` | 2,213,241,664 | 4.43 GB | 617 |
| `morriszjm/Tacit-4B` | `af3fa1dc…` | `Qwen/Qwen3-4B` | 4,022,468,096 | 8.04 GB（2 片） | 398 |
| `morriszjm/Tacit-8B` | `3e4df896…` | `Qwen/Qwen3-8B` | 8,190,735,360 | 16.38 GB（4 片） | 399 |
| `morriszjm/Tacit-9B` | `884838f1…` | `Qwen/Qwen3.5-9B` | 9,409,813,744 | 18.82 GB（4 片） | 760 |

核对下来有四件事与仓内文档一致、两件事值得单列：

- **`base_model` 与 README 表里那列「base」逐档相同**（1.7B / 2B / 4B / 8B / 9B ↔ `Qwen3-1.7B` / `Qwen3.5-2B` / `Qwen3-4B` / `Qwen3-8B` / `Qwen3.5-9B`），五份模型卡也都写着 `license: apache-2.0` + `library_name: transformers`。
- **两档是混合架构**：Tacit-2B / 9B 的 `config.json` 是 `qwen3_5` / `Qwen3_5ForConditionalGeneration`，`layer_types` 里每 4 层一个 `full_attention`、其余 `linear_attention`（每档 15 个 `mtp.*` 张量也是这个系列的标志）。这直接决定了服务侧的处理：`tacit.py:257-272` 的 `_needs_native_gdn` 读 `config.json` 判出 `qwen3_5`，`_patch_native_gdn` 把 vLLM 的 `ChunkGatedDeltaRule` 换成 `forward_native`（Hopper 上没有 `nvcc` 时），`docs/limitations.md:15-17` 把这条写成了部署注意事项。
- **两档还带着视觉塔**：Tacit-9B 的 760 个张量里 **333 个是 `model.visual.*`**（另有 426 个 `model.language_model.*` 加一个 `lm_head.weight`）。也就是说这份检查点仍然是一个多模态检查点，而**这条读法从不碰视觉那一路**——它是本仓里唯一一个「发布物带着一整条不使用的通道」的例子（对照 RSI-Jev 那篇：那边图像走同一条读法）。
- **权重是纯 BF16**：两个单分片仓的头部字节给出 `dtypes {"BF16": 参数量}`（§14.5）。

### §11.2 容器算术：这条读法不给自己加头

把每个 Tacit 与它基座的张量名集合对着看，「实测（Hub）」：

| | 只在基座里 | 只在 Tacit 里 | 参数量差 |
|---|---|---|---|
| Tacit-1.7B ← Qwen3-1.7B | `lm_head.weight` | — | 311,164,928 = 151,936 × 2048 |
| Tacit-2B ← Qwen3.5-2B | 15 个 `mtp.*` | — | 60,828,160 |
| Tacit-4B ← Qwen3-4B | —（398 = 398） | — | 0 |
| Tacit-8B ← Qwen3-8B | —（399 = 399） | — | 0 |
| Tacit-9B ← Qwen3.5-9B | 15 个 `mtp.*` | — | 243,290,624 |

而后两个数**正好等于**基座那 15 个 `mtp.*` 张量的元素个数之和——用 HTTP Range 取基座分片的头部，逐个张量累加形状（§14.5 是同一段算术的两个方向）：Qwen3.5-2B 的 `mtp.fc.weight [2048, 4096]` 加 14 个兄弟 = **60,828,160**；Qwen3.5-9B 的是 **243,290,624**。两处逐位相等。

所以这条路的自蒸馏**没有给自己加任何头、也没有留下任何 LoRA 张量**（名字集合里一个 adapter 名都没有）：4B / 8B 与基座逐名相同，2B / 9B 少的正是基座的**多 token 预测块**，1.7B 少的是那一档的 `lm_head.weight`——两个 config 都写着 `tie_word_embeddings: true`，基座仍然把那个冗余张量存了一份，而 Tacit 那份没有存（`transformers` 会从词嵌入重建它）。这与本仓已收各条「训一个头」的路线（Jeff / Kev / Clef / pplx-decider / RSI-Jev / NanoJev）形成对照：**它的训练改的是同一批张量，不是新加一批**。而发布物里看不出是合并过的 LoRA 还是全量微调——这一点本文**未核实**（§15）。

### §11.3 一份没有权重、只有一个引擎的模型卡

`morriszjm/AnyJev-L0-Gemma-4-31B` 是 Training-free 那侧的发布物，而它**只有两个文件**：`README.md`（3,602 B）与 `.gitattributes`。「实测（Hub）」+ 模型卡自陈：

```
README.md:11   **This repository holds no weights.** It is the card of a training-free decision engine:
README.md:2-3  the unchanged `google/gemma-4-31B-it` read through AnyJev's L0 readout.
```

README 里那张表（Full score **60.22**、public 59.79、同技能私有 59.13、新域私有 52.98、延迟 98/260/1,080 ms、2026-10-12 上 149 条里的第 5）与 `bench/results_decision_index/2026-10-12/anyjev-l0-gemma-4-31b.json` 的 `board` 块逐项相同，而 `tests/test_readme_numbers.py:51-62` 把这张表钉在测试里（连徽章里的百分号转义都查）。该 JSON 还记着看板维护者的三条附注，其中一条与本文 §9.1 ②直接相关：

```
"Running prior is per process and history-dependent; submitted environment provenance
 does not match the published engine."        bench/results_decision_index/2026-10-12/…json 的 board.maintainer_notes
```

——**批先验是进程内累积的，所以同一个问题在不同请求历史下会给出略微不同的分布**。这条不是别人猜的，是上游自己写进 README 与模型卡的（`README.md` 的 How-it-works 之外，模型卡那一节也有一句 *depends slightly on the requests that came before it*）。所以 AnyJev 的「可复现」是有条件的：**同一进程、同一请求序列内逐位可复现；跨进程不可复现**。

---

## §12 文件对照表

| 文件 | 行数 | 职责 |
|---|---|---|
| `anyjev/decider.py` | 580 | 读法总装：level（raw / L0）、prior、旋转预算、共享前缀分组、诊断量 |
| `anyjev/tacit.py` | 448 | Tacit：三引擎（transformers / vLLM / server）、升级与配额、HTTP 网关的 `serve_gateway` |
| `anyjev/backends/hf.py` | 389 | transformers 后端；共享前缀快路径；隐藏状态与按层块循环（0.3.1 用） |
| `anyjev/backends/fake.py` | 266 | 合成偏置模型（位置偏置 / 标签先验 / logit 噪声 / 措辞仿射 / 植入隐藏状态） |
| `anyjev/backends/llamacpp.py` | 231 | GGUF 后端 + GGUF 侧分词器 + 双分词器逐 token 校验 |
| `anyjev/readout.py` | 221 | prompt 构造、标签解析、哨兵切分、选项跨度与 token 下标（供特征缓存用） |
| `anyjev/backends/vllm.py` | 133 | vLLM 生成服务（`allowed_token_ids` + `processed_logprobs`）与 embed 服务的隐藏状态 |
| `anyjev/calibrate/stopping.py` | 135 | 旋转预算：log-odds margin、Clopper-Pearson 上界、阈值选择 |
| `anyjev/result.py` | 127 | `Decision` / `DecisionSet` / level 顺序与 `require` |
| `anyjev/truncate.py` | 107 | 把一个模型的前 n 层写成独立检查点（含来历文件） |
| `anyjev/question.py` | 90 | 三个题型的构造与校验、`key`（spec 的 sha256 前 16 位）与 `id` |
| `anyjev/serve.py` | 86 | 标准库 HTTP 网关（`/v1/decide`、`/v1/stats`、`/health`） |
| `anyjev/calibrate/permute.py` | 76 | 循环移位、spread 顺序、logmean / mean 合并、跨移位翻转率 |
| `anyjev/backends/sglang.py` | 75 | SGLang 原生 `/generate`（`max_new_tokens=0` + `token_ids_logprob`） |
| `anyjev/calibrate/contextual.py` | 47 | 批先验 / 内容无关先验 / 相除重归一 |
| `anyjev/backends/base.py` | 31 | 后端协议（一个必需方法 + 两个可选方法的说明） |
| `anyjev/__init__.py` | 24 | 导出与版本（回落到 0.3.0） |
| `anyjev/state.py` | 23 | state 渲染（str / 对话列表 / JSON） |
| `scripts/eval_tacit.py` | 224 | JevBench 231 与 bev-decision 46,320 的评测 harness（分块、可续跑、写 summary） |
| `examples/decision_index/anyjev_engine.py` | 225 | 提交给 Decision Index 的引擎（两位读法、超窗拒绝、provenance） |
| `space/app.py` | 127 | gradio 演示（选项反转对照） |
| `scripts/llamacpp_parity.py` / `sglang_parity.py` / `vllm_parity.py` / `shared_prefix_parity.py` | 98 / 66 / 61 / 69 | 四份对拍脚本 |
| `scripts/make_flip_gif.py` / `make_banner.py` / `smoke.py` | 95 / 126 / 26 | 素材与冒烟脚本 |
| `tests/`（16 个文件） | 1,875 | 其中 `test_readme_numbers.py` 反查两份 README 的表，`test_tacit.py` 钉住训练 prompt 与配额语义，`test_stopping.py` 钉住认证与规范列法 |

---

## §13 亲手跑一遍

本文**没有跑过任何一条**（本机无 torch / 无 GPU / 无权重）。下面是上游给出的、可以复现出本文所引数字的命令，按「不需要权重 → 需要权重」排列：

```bash
pip install -e ".[dev]"                                  # 核心 + pytest + ruff，纯 numpy 即可
ruff check anyjev scripts space tests && pytest -q        # CI 的两条命令（ci.yml）

# 库路径（README 的例子，需 GPU + 权重）
python - <<'PY'
from anyjev import Decider, Question
from anyjev.backends.hf import HFBackend
d = Decider(HFBackend("Qwen/Qwen3-8B"))
q = Question.choice("Which team should handle this?", ["billing","technical","sales","other"], name="route")
r = d.decide("My card was charged twice for one order.", [q])["route"]
print(r.distribution, r.level)                            # level 默认 "L0"
PY

# 旋转预算：先认证阈值，再决定（不需要标签）
python - <<'PY'
from anyjev import Decider, Question
from anyjev.backends.hf import HFBackend
d = Decider(HFBackend("Qwen/Qwen2.5-7B-Instruct"), adaptive_shifts=True, canonical_order=True)
q = Question.choice("Which team should handle this?", [f"team_{i}" for i in range(18)], name="route")
print(d.calibrate_adaptive(q, unlabelled_states, target=0.01))   # 需几百条未标注状态
PY

# 共享前缀对拍（打印 max|dlogp| / max|dlogsoftmax| / 回退组数 / 加速比）
python scripts/shared_prefix_parity.py --model Qwen/Qwen3-1.7B --dtype float32

# Tacit：一个 vLLM 服务 + 四副本评测（本文引用的那批数就是这条命令产出的）
vllm serve morriszjm/Tacit-9B --host 127.0.0.1 --port 8000 --max-model-len 32768
python scripts/eval_tacit.py --model morriszjm/Tacit-9B \
    --servers http://127.0.0.1:8020,http://127.0.0.1:8021 \
    --task jevbench --jevbench ./jevbench --adaptive \
    --out runs/Tacit-9B_jevbench --result bench/results_tacit/<date>/Tacit-9B.jevbench.json

# HTTP 网关（一个 Tacit 服务所有客户端，配额共享）
python -m anyjev.serve --model morriszjm/Tacit-9B --upstream http://127.0.0.1:8000 --adaptive --port 8100
curl -s 127.0.0.1:8100/v1/decide -H 'Content-Type: application/json' \
  -d '{"state": "Customer: my package has not arrived.", "question": "What does the customer want?", "options": ["track_order", "refund"]}'

# 看板那条链（需要 gemma-4-31B 与第三方 kit）
vllm serve google/gemma-4-31B-it --quantization fp8 --logprobs-mode processed_logprobs \
    --enable-prefix-caching --host 127.0.0.1 --port 8000
python -m decision_index run --engine anyjev_engine:AnyJevEngine \
    --option base_url=http://127.0.0.1:8000 --option model=google/gemma-4-31B-it
```

一处必须提醒：**`--logprobs-mode processed_logprobs` 是 `VLLMBackend` 的前提**。默认的 raw 模式返回的是模型自己的 top-K，一个标签掉出去就直接抛 `LabelTokenError`（`backends/vllm.py:79-83`），这是 0.3.0 的 `#13` 修复（之前是静默按 `-30` 读）。

---

## §14 实例走查与机械核对

### §14.1 `assets/flip_examples.json`：10 条真实读数，四种读法

这份 committed 文件是 Qwen3-8B 在 banking20（K=20、20 个客服意图）上的 10 条实例，每条记四个分布：`raw_as_typed` / `raw_reversed` / `l0_as_typed` / `l0_reversed`，外加 `raw_argmax`（一对）、`l0_argmax`（一个）与 `l0_correct`。「实测（committed 产物）」：

| # | state | gold | raw 原序 → 反转 | L0 |
|---|---|---|---|---|
| 0 | Please stop my purchase. | request refund | request refund → declined cash withdrawal | request refund ✓ |
| 1 | I have a payment listed in error. | card payment not recognised | card payment not recognised → request refund | 同 gold ✓ |
| 2 | Cash I didn't get shows in my app. | cash withdrawal not recognised | balance not updated… → cash withdrawal not recognised | 同 gold ✓ |
| 3 | I can't see my latest bank transfer | balance not updated after bank transfer | transfer not received… → balance not updated… | 同 gold ✓ |
| 4 | I had a wrong fee charged at this ATM. | cash withdrawal charge | extra charge on statement → cash withdrawal charge | 同 gold ✓ |
| 5 | Why was a charged just for transferring? | transfer fee charged | transfer fee charged → cash withdrawal charge | 同 gold ✓ |
| 6 | I have multiple charges on the same transaction. | transaction charged twice | transaction charged twice → extra charge on statement | 同 gold ✓ |
| 7 | How long until my transfer will be available to me | balance not updated after bank transfer | transfer not received… → balance not updated… | 同 gold ✓ |
| 8 | Why were there additional charges when transferring? | transfer fee charged | transfer fee charged → extra charge on statement | 同 gold ✓ |
| 9 | I took out a foreign currency and the exchange rate is wrong. | wrong exchange rate for cash withdrawal | card payment wrong exchange rate → wrong exchange rate… | 同 gold ✓ |

**10/10 反转后 raw 翻转、10/10 L0 不变、10/10 L0 答对。** 这个 10/10 **是构造出来的，不是估计**：这份文件是挑出来做演示的（`CHANGELOG.md` 的 0.0.2 节记着选它的脚本 `scripts/find_flip_example.py`、渲染它的 `scripts/make_flip_gif.py`，前者**已经不在树里**）。它对应的整体翻转率在 `bench/results_v01/2026-09-22/Qwen__Qwen3-8B.json` 的 `raw.flip = 0.2300` —— 也就是说**大约每五条里有一条会翻**，README 顶部与 Space 首页那句 *one in five answers changes* 就是从这个数来的。

`l0_argmax` 只有一个值这件事本身是个证据：**L0 不需要「反转后再读一次」**，因为读满 K 个移位的 logmean 合并与选项的列法无关（§4.2）。所以这张表里 `l0_as_typed` / `l0_reversed` 的**数值并不构成两条独立证据**——它们是同一条读法在两种列法下的两次渲染，差别只在该状态下 bf16 的批组成（第 0 行：`l0_as_typed` 的 gold 是 **1.0**、`l0_reversed` 是 **0.855**，而两者的 argmax 相同）。真实模型上的这种差异落在哪个量级，committed 产物里没有配对的重复记录，本文**不量化**。

### §14.2 两份 README 的表被 CI 反查

`tests/test_readme_numbers.py` 是本文见过的这批实现里**唯一一处「文档里的数被测试反查」**：

| 断言 | 覆盖 |
|---|---|
| `test_tacit_table_matches_its_json`（`:28-37`） | 两份 README 里 5 行 `\| [Tacit-…]`，第 2/3 列与第 4/5 列分别对 `bench/results_tacit/2026-10-06/Tacit-X.{jevbench,bev}.json` 重算 `correct/n` 与 `escalated/n`（`expected()` 在 `:19-25`） |
| `test_raw_vs_l0_table_matches_its_json`（`:40-48`） | README 里 `\| raw \| L0 \|` 那三行（flip / acc / ece）与 `cov@5%` 那一行，逐字符串查在 JSON 里 |
| `test_decision_index_numbers_match_their_json`（`:51-62`） | 看板那行（四位小数 + 千分位延迟）与徽章里的 URL、`**60.22**` 出现两次、`5th of 149` 出现两次 |

也就是说**这个仓里「README 的数」与「JSON 的数」不允许分叉**，而 `AGENTS.md` 的 Ground rules 第 1 条把它写成了纪律：*If a run did not happen, the cell is empty.* 代价是它只覆盖**被引用的那几格**：`bench/results_v01/2026-09-22/` 的 33 个 task × 7 个 level 里，只有 Qwen3-8B / banking20 的 raw 与 L0 被 README 引用（其余靠 JSON 自己站着）。

### §14.3 模型卡与仓内 README 的三处差

同一份发布有两条叙述线：**模型卡**（每个 Tacit 仓的 `README.md`）与**仓内 README**（本仓的），两者的表结构不同（前者是「基座 vs Tacit」，后者是「一次前向 vs 升级」），而三格数字对不上，「实测（Hub + committed 产物）」：

| 仓 | 模型卡 JevBench | committed JSON | 模型卡 bev | committed JSON |
|---|---|---|---|---|
| Tacit-1.7B | 0.623 | **0.619**（143/231） | 0.598 | 0.598（27,696/46,320） |
| Tacit-8B | 0.749 | **0.736**（170/231） | 0.662 | 0.663（30,723/46,320） |
| Tacit-9B | 0.823 | 0.823（190/231） | 0.725 | **0.727**（33,652/46,320） |

这三处都不是事故，`CHANGELOG.md` 自己写明了成因：模型卡那批数是维护者用 transformers 跑的那一版，仓内 README 换成 `vllm serve` 复跑之后，*one forward stays within bf16 noise of that evaluation on bev-decision (−31 to +80 decisions of 46,320) and differs by at most three items of 231 on JevBench (Tacit-8B 0.749 → 0.736, Tacit-1.7B 0.623 → 0.619; the other three unchanged)*。逐条对上：8B 差 3 条、1.7B 差 1 条、9B 的 bev 差 **70** 条（落在 −31..+80 的带里）、2B 与 4B 的四个格子逐位相同。所以引用 Tacit 的准确率时，**第一件事是问「模型卡那一版还是仓内 README 那一版」**——这与 §5 里那类「同一个名字两个数」同族，只是这次分歧的两侧**都属于同一个作者**，而且被 CHANGELOG 交代得清清楚楚。

### §14.4 同一份读法的两份副本

五个模型仓各带一份 `tacit_decision.py`（27,868 B），「实测（Hub）」它们的 sha256 **逐字节相同**：

```
9ea93535f3dec625da8e80f412739ffeafa6218dd07a2de50fe9ffaa4e1cc4fa   Tacit-1.7B / 2B / 4B / 8B / 9B 的 tacit_decision.py
```

但这份副本与包里的那份**已经分叉**。逐项对照（左＝`GITHUB` 仓里的 `anyjev/**`，右＝权重仓里的副本）：

| 项 | 包里 | 副本里 |
|---|---|---|
| harmony 模板尾巴 | 无 | `text += "<\|channel\|>final<\|message\|>"`（副本 `:409-410`） |
| `score` 数字→字母回落 | 有（`readout.py:44-48`） | **无**——标签不是单 token 就抛 `ValueError`（副本 `:413-426`） |
| `first_level`（score 从 0 编号） | 有（`tacit.py:359-364`） | 无 |
| `system=` 覆盖 | 有（`tacit.py:294`） | 无（常量 `ONE_FORWARD_SYSTEM`） |
| transformers 5 的 `dtype`/`torch_dtype` 改名 | 处理了（`tacit.py:349-351`） | 没有 |
| 没有 chat 模板时 | 回落到 `f"{system}\n\n{user}\nAnswer:"` | 只捕 `TypeError`，模板缺失会抛 `AttributeError` |
| import 期 | 不导入 torch / vLLM（`tests/test_tacit.py:183-185` 断言 `"vllm" not in sys.modules`） | 模块级 `import torch` |
| 多了什么 | `Decider`、`Question`、L0、`anyjev.serve` | `serve_gateway` + `tacit_decision.py serve` 子命令 |

于是**「Tacit 的读数」这个名字底下有两个实现**：装包用 `from anyjev import Tacit`，用模型卡那三行 `get_class_from_dynamic_module("tacit_decision.Tacit", …)` 则是另一份。前者在后加的三处（`first_level` / 系统提示可换 / 数字回落）上更宽，后者在 harmony 那一处更宽。这与 NanoJev 那篇记的「训练器四个哈希互不相同」是同一种麻烦：**同一份发布里的两份副本，谁都不是错的那一份**。

### §14.5 单分片检查点的字节恒等式

两个只发单分片的仓可以直接用 HTTP Range 读 safetensors 头部（前 8 字节是头部长度，随后是 JSON），「实测（Hub）」：

| 仓 | 头部长度 | 张量数 | 元素数 | dtype 分布 | 文件字节 | 恒等式 |
|---|---|---|---|---|---|---|
| Tacit-1.7B | 35,648 B | 310 | 1,720,574,976 | BF16 全部 | 3,441,185,608 | 8 + 35,648 + 2×元素数 = **3,441,185,608** ✓ |
| Tacit-2B | 75,528 B | 617 | 2,213,241,664 | BF16 全部 | 4,426,558,864 | 8 + 75,528 + 2×元素数 = **4,426,558,864** ✓ |

两处**逐字节相等**，于是三件事同时被证：①权重是纯 BF16（不是量化档）；②文件里没有多余的张量（头部声明的就是全部）；③「参数量 ×2 = 权重字节」在这两档上精确成立。而多分片那三档也用同一个恒等式在 `model.safetensors.index.json` 的逐文件 size 上验证过（4B 两片、8B 四片、9B 四片，各自和 = 2×参数量 + 头部），本文不逐一列出。

同一段算术的另一头用在基座的 `mtp.*` 上（§11.2）：对一个**只读头部**的请求来说，拿到的是形状而不是权重，所以「基座参数量 − Tacit 参数量 = 那 15 个张量的元素数」是一个**两个方向都成立**的恒等式，而不是拟合。

---

## §15 一句话总结与未核实清单

**一句话**：AnyJev 把 RYOTIDE 那条读法做成了一个包，然后在这个包里放了「谁被读」的两个答案——一份任意冻结的开源 LLM（零训练，`L0` 用 K 个循环移位把位置偏置消掉，先验与停止阈值都无标签），和五档自己发布的 Tacit 检查点（用基座自己的推理蒸馏出来的，**一个头都没加**）；两份权重走同一条 prompt、同一个读位置、同一次重归一化，而这条链上**没有温度这个变量**，唯一的生成通道是「把不确定的题送去重新推理」并且把生成量逐条报出来。

**未核实清单**（本文明确不主张的）：

1. **Tacit 是怎么训的**。训练代码不在本仓、也不在任何模型仓（那里只有推理副本 `tacit_decision.py`）。能核到的只有：模型卡自陈「自蒸馏、基座自己出题、开思考作答、无人类标签/无外部数据集/无别的模型」，以及发布物里**没有新张量、没有 LoRA 名、权重是 BF16**。「合并过的 LoRA」与「全量微调」在发布物上不可区分。
2. **任何由本机前向产生的数**。本机无权重、无 torch、无 GPU；`§6.2`、`§14.1` 的两组数都是上游 committed 产物里的读数，`§11`、`§14.5` 是 Hub 元数据与头部字节。
3. **`bench/results_v01/**` 的评测口径**。算那批 ECE / flip / cov@5% 的运行代码**不在树里**（0.3.0 删掉了 `bench/` 的运行部分，`CHANGELOG.md` 说它留在 `anyjev==0.2.0`），所以**分箱数核不了**——引用那三列 ECE 与别家比较时，这一点必须先记住（本文因此不在这里做跨实现的 ECE 对比）。
4. **`cot` 决策的采样参数**（T 0.6 / top-p 0.95 / top-k 20）没有被任何 committed 产物记录，所以「这条升级后的概率是在哪个采样温度下生成的」不可核对。
5. **`docs/rotation_budget.md` 里引的 `bench/layout/`（tag `v0.2.0`）与 `CHANGELOG.md` 里引的 `docs/research_log.md`**：这份 clone 只有 **1 个 commit、0 个 tag**（`git log` / `git tag` 实测），这些路径在这里无从解析。这不是说上游没有——是**这份 clone 里核不了**。
6. **`shared_prefix` 快路径与普通路径在真实模型上的数值差**：只有一份对拍**脚本**（`scripts/shared_prefix_parity.py`），本文没有在树里找到 committed 的对拍结果文件，所以那对差只在上游各处被描述为「3e-4 nats 量级」，本文不给数。

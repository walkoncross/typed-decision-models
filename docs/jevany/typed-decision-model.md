# JevAny 是如何把两条读法（训练出来的头 / 零训练的选项 ID）装进同一个决策运行时的

> 代码库：<https://github.com/SimpleJev/JevAny>
> 对应 README 第 25-30 行：*JevAny is open infra for System 1 decision model training and deployment … One API takes
> the state, question and candidate options, then directly returns a choice and its probabilities.*
> 本文按 `jevany/` 的实际代码逐环节说明这句话的实现方式；文中出现的文件名与行号（`model.py:97`、`api.py:33`、
> `runtime.py:143`、`letter_predictor.py:512` 等）均相对该仓库根目录。
>
> **同名配对文档**：RYOTIDE 的 [`docs/ryotide/typed-decision-model.md`](../ryotide/typed-decision-model.md)。
> 本文凡称「RYOTIDE §N」均指该篇的节号；§0 先讲清分歧，§10 把它的四条不变量逐条落位。
>
> **与 `docs/kev/` 那篇的关系**：本仓每个改过的 Python 文件第二行都写着 `Derived from Kev by Jared Palmer under
> Apache-2.0. See NOTICE.`，`model.py` / `api.py` / `runtime.py` / `checkpoint.py` / `metrics.py` 的骨架与 Kev
> 同源。所以本文与 Kev 那篇**并读**才完整：Kev 讲那条读法本身，本文讲它被铺成了什么形状——两条读法、五套权重、
> 一个 agent 消费面，以及一份比 Kev 厚得多的 committed 产物。

一句话概括它与前十九篇的关系：**它不是又一条读法，而是把「读法」抽成了一层可插拔的东西**——`native`
（检查点自带一个头，pointer 或直接读 255 个词表行）与 `choice`（零训练，只读 52 个单字符 ID 的概率质量），
同一个 HTTP 形状、同一个 `DecisionRuntime` 契约、同一套限额与守卫，换的是 `--readout` 一个开关。

## 本文的事实来源

| 类别 | 来源 | 本文中的标记 |
|---|---|---|
| 代码引用 | 直接读 `jevany/**`、`scripts/**`、`recipes/**`、`tests/**` | `文件:行` |
| 可实跑得到的产物（纯标准库／纯算术） | `api.py` 的渲染与舍入、`letter_readout.py` 的 prompt 构造与 log-mass 聚合、`model.py` 的 `encode` 打包 | 「实测（纯 Python）」 |
| 在 committed 产物上重算 | `results/**` 里的 JSON：`letter-readout-v1.json`、`choice-readout-v2.json`、`decision-evaluation-v1/`、`model-support-v1.json`、`agent-harness-v1/` | 「实测（committed 产物）」 |
| 公式与常量 | README、`recipes/*.toml`、`pyproject.toml` | `README:行` |
| 模型事实 | Hugging Face API 与模型仓里的小文件（`release-manifest.json`、`tokenizer_config.json`、`config.json`） | 「实测（Hub）」 |

**本文不给任何由前向产生的概率数值。** 本机没有 GPU，也没有下载任何一份基座权重（最小的 `Qwen/Qwen3.5-4B` 也是
9.32 GB BF16），所以 §13 的「实测」全部落在三类不跑上游程序、不加载权重的核对上：在 committed 产物上重算公式、
用 Hub 给的**文件字节数**做容器算术、以及纯 Python 的字符串与算术复算。凡需要一次前向才能定的量，写「未核实」。

---

## 0. 与 RYOTIDE 的分野

| 维度 | RYOTIDE | JevAny |
|---|---|---|
| 权重 | 公开 LLM，全程 frozen | **两条**：检查点自带的头（LoRA + `PointerHead`，或 Direct-Token 的 `lm_head` 行），以及同一个运行时里**零训练**的 `choice` 读法 |
| 「typed」从哪来 | 分词器：标签必须单 token | **三套**：请求声明的 `criteria` 顺序 + 头给出的恰好 K 个标量（pointer）；分词器给的 255 个单 token 标签（direct-token，`model.py:39-62`）；52 个单字符 ID 在**全词表上按解码结果 gather**（choice，`letter_readout.py:122-178`） |
| 读位置 | **运行期标定**：扫 9 前缀 × 2 表面形式，取 `marker_mass` 最高 | **构造期算出来**：`encode` 在拼序列时写下 `decide_idx` / `opt_idx`（`model.py:129-133`）；choice 那条是 chat 模板在 `enable_thinking=False` 下的末位（`letter_predictor.py:497-504`）。运行期**无标定、无扫描、无诊断量** |
| 掩码 + 重归一化 | 必须（读出了整个词表分布） | **pointer 上不适用**（头只输出 K 个标量）；**direct-token 上也不适用**（只取 `lm_head` 的 K 行，`model.py:488-491`）；**choice 上是「按解码结果 gather + logsumexp」**（`letter_readout.py:202-244`）——与 metask-jev-rain 同族但机制不同 |
| 置信度 | **读出来的**：只有 temperature 能改数值 | **两条链各自不同**：native 是**训出来的**（头/LoRA，温度是加载时落到头上的单调缩放）；choice 是**读出来的**（零训练）。两条链的发布值都还要过一道 **`round(x, 2)`**（`api.py:233-234`） |
| `output_tokens` | 0 | **继承 Kev 的定义**：序列化后答案 JSON 的 token 数（`api.py:251-254`）——机制成立、字段不成立 |

一句话立场：**JevAny 把 RYOTIDE 那条读法变成了「一个运行时上的两种 payload」**，并且为此付出了三处别篇没有的
代价：①发布出去的概率被**四舍五入到 2 位小数**（这是本文 §9 里唯一一处「无条件、无判据、每一条响应都过」的改写）；
②`usage.output_tokens` 沿用 Kev 那个有自己定义的数字；③因为要同时服务五套不同骨干的发布检查点，标签集与读位置
**必须能在运行期被证明**，而不是被构造保证——它选择的证明方式是加载期与构造期的硬断言，而不是运行期诊断量。

---

## 1. 端到端数据流（一次请求）

```
POST /v1/systemone   api.SystemOneRequest {state, model, media[], questions{id → Noul|Choice|Score}}
  │
  ├─ serve.prepare()                      可选前缀处理，两步都在模型之前
  │    ├─ JEVANY_DATE_FACTS=1  → api.with_date_facts()  把 state 里两两日期的差写成句子塞回去
  │    └─ 有 media            → 路径必须落在 JEVANY_MEDIA_ROOT 内、不许是网络 URI、
  │                            体积 / 像素 / 帧数逐个校验（serve.py:73-113）
  │
  ├─ runtime.answer()                     runtime.py:143
  │    ├─ 身份校验：request.model ∈ {model_id, "jevany-latest"}（:144-145）
  │    ├─ api.to_record()                 state 渲染成文本、每题 options 渲染成文本（api.py:200-218）
  │    └─ runtime.probs()                 runtime.py:94
  │         ├─ model.encode()             打包成**一条** token 序列：
  │         │                             [<|fim_prefix|> state] 每题[<|fim_middle|> 指令
  │         │                              <|box_start|>选项<|box_end|>… <|fim_suffix|>]（model.py:97-135）
  │         ├─ 四道限额：state / branch / packed / 骨干上下文窗口
  │         │                             超限抛 ContextLengthError → HTTP 422（serve.py:132-137）
  │         ├─ 前缀缓存（仅 native，且骨干声明支持，且 state ≥ 384 token）
  │         └─ model.forward()            branch_mode=rows：每题一行因果行（model.py:499-518）
  │              └─ _question_readout()   只取两个位置：本行的 <|fim_suffix|> 当 query、
  │                                      每个选项的 <|box_end|> 当 key（model.py:483-493）
  │                   ├─ pointer  ：PointerHead(q·k)          → K 个标量
  │                   └─ lm_token ：lm_head 的那 255 行        → K 个标量
  │         └─ softmax(-1)                → 概率
  │
  ├─ api.to_answers()                     每题成形（choice / noul / score）
  │                                       **概率、confidence、score 全部 round(x, 2)**（api.py:237-248）
  └─ api.validate_response()              形状 / 有限性 / 概率和容差 / choice 必须是**舍入后**概率的 argmax
                                          （api.py:99-144）
  → {model, answers, usage:{input_tokens, output_tokens}, latency_ms}
```

`--readout choice` 时上半段换掉：`JevModel.from_pretrained` 不构造 `DecisionRuntime` 而构造
`LetterDecisionRuntime`（`runtime.py:244-246`），后者的 `answer()` 走
`letter_predictor.LetterReadoutPredictor`：每题**一条 chat prompt、一次 prefill**，读 `last_hidden_state[0, -1]`，
再投影到「能解码成某个选项 ID 的全部 token 行」上（`letter_predictor.py:512-540`）。这条路径**不接受 media**
（`letter_runtime.py:122-123`、`letter_predictor.py:570-571`），也没有前缀缓存（`:55-56`）。

---

## 2. 「typed」从哪来：三套标签集，各自封闭的方式不同

### 2.1 pointer（发布检查点的默认档）

选项就是请求里 `criteria` 的**值**，按插入顺序成为 `<|box_start|> … <|box_end|>` 跨度，长度不限、可多 token
（`model.py:118`）。词的闭合性来自**头的输出个数就是 K**：`PointerHead` 只有 q / k 两个线性层，
`forward` 返回 `[K]`（`model.py:281-286`）。所以这条链上**没有词表维度**，也就没有「越界」这件事可以发生——
这与 Kev 的结论相同，而 JevAny 多出来的部分是同一份代码里还有另外两套标签集。

上限：HTTP 层是 `MAX_OPTIONS = 4096`（`api.py:33`），注释写明了理由——*Pointer checkpoints are bounded by the
configured context window rather than by a fixed verbalizer vocabulary*（`api.py:30-32`）。

### 2.2 direct-token（`decision_mode == "lm_token"`）

标签集是**分词器给的 255 个单 token 符号**：`decision_verbalizers()` 先试
`BASE_VERBALIZERS = "A-Za-z0-9"`（62 个），不够就从码点 `0x00A1…0xA000` 里按序补，要求
`len(encode(sym)) == 1` **且** `decode([id]) == sym`（往返一致）且 id 互不相同（`model.py:39-62`）。
不够 255 个就直接抛：*tokenizer exposes only N usable single-token verbalizers; need 255*（`:60-61`）。

然后**标签被写进选项文本本身**：`DecisionModel.encode` 在 lm_token 模式下把每个选项改写成
`f"{verbalizers[index]}: {option}"`（`model.py:426-435`）。所以「typed」在这里的落点是 prompt 文本 + 词表的行号，
而不是任何运行期映射表。读出头就是 `lm_head` 的那 255 行（`:353-359`、`_question_readout` 里的
`weight.index_select(0, self._verbalizer_index[:len(options)])`，`:488-491`）。

`verbalizer_token_ids()` 在加载时对这张表再验一遍（单 token + 往返），不符即抛（`model.py:65-75`）。

### 2.3 choice（零训练那条）

标签集是 **52 个单字符 ID**：`CHOICE_SYMBOLS = "A-Z" + "a-z"`（`letter_readout.py:21-28`），
超过 26 个选项就从大写换到小写。与 metask-jev-rain / decider 的「多字符码」不同，这里的每个 ID
**必须是词表里真实存在的、能解码回该字符的 token**：`letter_token_ids()` 把**整个词表**逐 id 解码一遍，
把 `decode([id]) == 某个 ID` 的所有 id 收进该 ID 的别名元组（`letter_readout.py:122-178`）。
词表大小 248,320（Qwen3.5-4B / Qwen3.8-27B 都是这个数）时它按 4096 一批走 `batch_decode`，
注释写明理由：*A 248k-token vocabulary otherwise takes minutes when decoded one ID at a time*（`:151-155`）。

**这条链上的「多对一」是真存在的**：代码注释明说 *distinct token IDs may decode to identical text, and every such
ID contributes probability mass*，并且**刻意不按解码文本去重**（`:122-130`、`:207-212`）。所以 `" A"` 与 `"A."`
被排除，而两个都解码成 `A` 的 id 会被 `logsumexp` 合成一个质量（`:233-243`）。这一点与 metask-jev-rain 的
「一个标签一组写法」同构，但那里的重数是**代码里写死的字符串表**，这里的重数是**从分词器现扫出来的**。

> **未核实**：Qwen3.8-27B 的分词器上每个 ID 到底有几条 token id。判定它需要那份 `tokenizer.json`
> （19,989,325 B，随仓发布），本文没有下载。

### 2.4 分隔符不可伪造

与 RYOTIDE 那类「标记是字符串」的实现同处一个风险面：用户文本不能造出分隔符。JevAny 的处置是**改写**而不是扫描
——`user_tokens()` 在分词前把 `<|name|>` 替换成 `<¦name¦>`，并把本分词器实际的五个决策 token 也做同样处理
（`model.py:81-91`）。与 SemIf-OpenJev 的 `direct.py:42`（断言）、decider 的 `prompt.py:229-233`（断言）
相比，这是「把自由度删掉」那一类。

这五个分隔符**随检查点发布**：`tokenizer_config.json` 里有 `jevany_decision_tokens` 这个字段，
Qwen 两份检查点是 `["<|fim_prefix|>","<|fim_middle|>","<|box_start|>","<|box_end|>","<|fim_suffix|>"]`
（复用 Qwen 自带、原本极少用的特殊 token，所以**不需要新增嵌入行**，`model.py:12-14` 的注释写了这一点），
Gemma 那份是 `["<unused0>" … "<unused4>"]`。加载时 `prepare_tokenizer()` 先认这个字段，认到就要求五者在词表里
且 id 互不相同，否则抛（`backbones.py:69-79`）；发布仓里三份 `tokenizer_config.json` 都带着它（**实测（Hub）**）。

---

## 3. 读位置：构造出来的下标，运行期没有可标定的自由度

`encode()` 返回四个与本问题直接相关的量（`model.py:97-135`）：

| 字段 | 含义 |
|---|---|
| `seg` | 每个 token 属于 state（0）还是第 k 题（k） |
| `pos` | 位置 id；**每题的分支从 state 之后重新起算**（`p0 = len(S)`，`:122`） |
| `decide_idx[k]` | 第 k 题 `<\|fim_suffix\|>` 那个 token 的全局下标 |
| `opt_idx[k][j]` | 第 k 题第 j 个选项 `</opt>`（`<\|box_end\|>`）的全局下标 |

读数就是这两个下标的点积：query = `h[decide_idx]`，keys = `h[opt_idx]`（`model.py:483-493`）。所以
**位置不是「找到的」，是「算出来的」**——与 Kev 的 `decide_idx` / `opt_idx` 是同一个设计（Kev 那篇 §3），
只是这里的 `opt_idx` 取的是每个选项**自己的结束符**而不是整段选项的池化。

choice 那条则完全不看 `decide_idx`：它把选项渲染成一段人读的编号列表，交给聊天模板，读**末位**
（`letter_predictor.py:512-526`）：

```python
ids = self.tokenizer.apply_chat_template(
    [{"role": "system", "content": SYSTEM_PROMPT},
     {"role": "user",   "content": _safe_chat_text(self.tokenizer, prompt)}],
    tokenize=True, add_generation_prompt=True, return_tensors="pt",
    enable_thinking=False,
)
...
hidden = output.last_hidden_state[0, -1]
```

`enable_thinking=False` 是**显式传死**的，没有回退分支（对照 RYOTIDE / Quyet 会为「模板不认识这个关键字」准备
`TypeError` 回退）。而 `-1` 在这里安全，靠的是**这个前向不组批**（每题一条 prompt、`use_cache=False`），
不是靠填充方向。

> **一处与 `padding_side` 有关但本文未展开的细节**：发布的 Gemma 检查点 `tokenizer_config.json` 里
> `padding_side = left`（**实测（Hub）**）。native 路径自己用 `_pad_rows` 右填充 + 显式掩码
> （`model.py:460-471`），不读这个设置；choice 路径也不组批。所以这个字段在这两条链上都不被消费——
> 但它是发布物里唯一一处「填充方向」的声明，引用时不要把两者混为一谈。

---

## 4. 一次前向：native 读一处、choice 读一处，两条都不生成 token

### 4.1 native：每题一行，行内读两个位置

`DecisionModel` 按骨干能力二选一（`model.py:331-339`）：

- **packed**：所有题在同一序列里，用 `branch_mask_batch` 造一张 block-causal 加性掩码
  （`model.py:217-243`）——`attend(i, j)` 当且仅当 `j ≤ i` 且 `seg[j] == 0` 或 `seg[j] == seg[i]`。
  只有 `supports_packed(config)` 为真的骨干才允许：`config.model_type in {"llama","mistral"}` 且无滑窗
  （`backbones.py:205-211`）。
- **rows**：每题一行因果行 = state + 该题分支，右填充 + 掩码后一次前向（`model.py:499-518`）。
  `rows_of()` 把打包编码拆回行，并保持「位置已接续 state」这一性质（`:246-260`）。

**发布的那五套权重都落在 rows 一侧**：Qwen3.5/3.8 的 `model_type` 是 `qwen3_5_text`、Gemma 是 `gemma4_text`、
Muse 是 `muse_glimmer_text`（**实测（Hub）**，取各仓 `config.json` 的 `text_config`），都不在
`{"llama","mistral"}` 里，所以 `branch_mode="auto"` 解析成 `"rows"`。代价写在注释里：*the state is recomputed
per row (Q x state tokens), which training accepts; serving uses the prefix cache instead*（`model.py:500-503`）。

读出头有两个分支（`_question_readout`，`model.py:483-493`）：

```python
query = h[decide]
if self.decision_mode == "lm_token":
    weight = self.lm_head.weight
    if not self.training:
        weight = weight.index_select(0, self._verbalizer_index[:len(options)])   # 只取 K 行
    logits = F.linear(query.to(...), weight).float()
    return logits if self.training or self.temperature == 1.0 else logits / self.temperature
return self.head(query, h[torch.tensor(options, device=self.device)])
```

`PointerHead` 本身是 q/k 两个 `Linear(d, 256)` 加一个**可选的残差 MLP**（`residual_dim` 非零时
`Linear(256,256) → GELU → Linear(256,1)`，末层零初始化，`:263-286`）。它与 `temperature` 的关系写在注释里：
*calibration: logits are divided by this at inference (eval mode) only. 1.0 = raw. A checkpoint carries the value
fitted on its in-distribution development rows*（`:276-279`），并且**训练时永远看 T=1**，所以 argmax 在构造上不被它改。

### 4.2 choice：每题一条 chat prompt，一次 prefill

`_letter_question` 的关键几行（`letter_predictor.py:512-540`）：

```python
prompt = build_prompt(state, question.get("instructions") or "", descriptions)
ids = self._chat_ids(prompt)
output = self.language_model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False)
hidden = output.last_hidden_state[0, -1]
selected_logits = F.linear(hidden.to(...), self.letter_projection, self.letter_bias).float()
if self.output_softcap is not None:
    selected_logits = self.output_softcap * torch.tanh(selected_logits / self.output_softcap)
```

`build_prompt()` 的排版是纯字符串拼接，**实测（纯 Python）**——取一组两选项的输入直接跑出来是：

```
{
 "goal": "g"
}

Choose the next action.

Options:
A. Allow it
B. Deny it

Answer with the case-sensitive ID of exactly one option, and nothing else:
```

（state 是 dict 时走 `json.dumps(..., indent=1)`，选项是「标签 + `. ` + 描述」，描述为空则回落到标签本身，
`letter_readout.py:48-89`。）

三处需要点出的实现选择：

1. **`letter_projection` 不是完整的词表投影头**。它只装「能解码成某个 CHOICE_SYMBOL 的全部 token 行」：
   `tie_word_embeddings` 为真时直接从输入嵌入表 `index_select` 出来（Qwen3.5-4B **是** tied，
   **实测（Hub）**），为假时从 `lm_head.weight` 用 safetensors 的 `get_slice` 按行取（Qwen3.8-27B、
   Muse-Glimmer-30B **都是** untied）。后者的注释直接给了算术理由：
   *``get_tensor`` materializes Qwen3.8-27B's complete 2.4 GiB head. Safetensors slices perform indexed row reads*
   （`letter_predictor.py:211-217`）。按 `config.json` 复核：248,320 × 5,120 × 2 B = **2,542,796,800 B ≈ 2.37 GiB**
   （**实测（Hub）**），与那句注释相符。
2. **`output_softcap` 只在 choice 这条链上被套**。它取自 `config.final_logit_softcapping`
   （`letter_predictor.py:422-426`）——Gemma-4-E4B-it 是 **30.0**、Muse-Glimmer-30B 是 **20.0**、
   Qwen3.5/3.8 都是 `null`（**实测（Hub）**）。native 的 `_question_readout` **没有**对应的 tanh 一步
   （`model.py:483-493`）。也就是说：同一份 Gemma 权重，走 choice 读法与走 direct-token 读法时，
   「权重的最终 logit 软上限」只在前者上生效。**改没改结论本文未核实**（需要一次前向）。
3. **一次决策 = 每题一次 prefill**。`__call__` 对每题循环（`letter_predictor.py:574-577`），
   多问题共享 state 在这条链上**不做**；`input_tokens` 是各题 token 数之和加上 native 那一趟
   （`:612`）。所以 `choice-native-weight` 打开时是**两条链各跑一遍**，注释也这么写：
   *this costs one extra native-formatted prefill per request*（`:230-233`）。

### 4.3 两条链都不生成 token

native 路径从头到尾没有采样器、没有 `max_new_tokens`、也没有 `logprob_token_ids`：调用的是
`AutoModel`（不是 `AutoModelForCausalLM`，`backbones.py:192-197` 里取的是 `causal.base_model`），
返回 `last_hidden_state`。choice 路径多一层：`_release_unused_output_head()` 在确定不需要 native
lm_token 那一趟时，把 `decision_model.lm_head` 与 `adapter._output_embeddings` 两个别名**都置 None**
（`letter_predictor.py:66-83`）——所以纯 choice 部署的显存里连那个 2.4 GiB 的头都不在。

这与底线上「`output_tokens = 0`」的机制层完全吻合。字段层见 §9.4。

---

## 5. 掩码与 softmax：两条链上分别是「不适用」与「在标签上」

### 5.1 native：没有掩码，也不需要

`_question_readout` 的输出向量长度恒等于 `len(options)`：

- pointer：头的输出维度就是 K，**词表维度从不出现**；
- lm_token：`index_select` 之后是 K 行，`F.linear` 出来的也是 K 个数，训练时才用全长词表
  （`model.py:485-491`）。

所以 `marker_mass` 这个量在这里**概念不适用**——不是「等于 1、无信息」的问题（那是词表被投影但未掩码的情形），
而是分母根本不存在。与 Kev / decider / pplx-decider 同一结论，这也是 RYOTIDE §13 #1 在这条链上的落法（§10）。

### 5.2 choice：掩码是「按解码结果 gather」，重归一化落在**标签**上

这是本文最值得单独说的一步。`read_letter_distribution`（`letter_readout.py:320-344`）分三步：

```
letter_log_masses(vocab_logits, token_ids)      # 每个 ID 的别名 id 做 logsumexp → 一个 log 质量
raw = probabilities_from_log_masses(masses)     # 稳定 softmax（先减 pivot）→ T=1 的原始分布
calibrated = softmax(masses / T)                # = p ** (1/T)，在对数域做，避免极小温度下溢
```

三点必须分清：

1. **`vocab_logits` 不是完整词表向量**。它可以是完整向量，也可以是「索引已被 `token_ids` 重映射过的紧凑向量」
   ——docstring 明说（`:207-211`）。本仓走的是后者：投影矩阵只含别名行，所以**词表其余部分从不被投影**。
   归一化只在「这些别名行」上做，等价于 Cygnet 那条 *structured-output choice mask*（`:210-212`）。
2. **同义词不去重、也不允许跨 ID 复用**。同一个 token id 被分配给两个不同 ID 会直接抛
   （`:237-239`），而两个不同 id 解码成同一个字符则**都必须计入**（`:140-149`）。
3. **温度作用在标签质量上，不作用在词表 logits 上**。`temper_probabilities` 的注释写明这是代数上的
   `p ** (1/T)`，只是挪到对数域算（`:307-317`）。所以这条链上的温度**在构造上不可能改 argmax**，
   `readout.calibrated_probabilities` 与 `raw_probabilities` 的 argmax 恒相同（单调变换）。

「重归一化」这个词在这条链上要改写：重归一化的对象是**选项 ID 的质量**，而归一化之前的「未归一化质量」
已经不是词表概率了——它是别名行的 logsumexp。这与 metask-jev-rain 的「一个标签一组写法」形状最近，
差别在写法表是现扫的还是写死的、以及那边的探针会**真的生成 token**（这边不会）。

---

## 6. 顺序偏置：训练期打散、一个构造性开关、一个条件性的度量

三件事分开说，因为它们的**状态各不相同**。

### 6.1 训练期打散（默认生效）

所有 `choice` 类型的问题，在训练数据的 `augment()` 里**每题独立重排选项**：`rng.shuffle(keys)`
（`data.py:54-56` 的 `target` 分支、`:71-74` 的主分支），同一处还会按概率插入
none-of-the-above 与干扰项（`:59-70`）。种子由 `source_seed(seed, source)` 派生（`:28-29`），
所以同一份数据的打散可复现。

运行期**保序**：`to_record` 按 `criteria` 的插入顺序生成选项文本（`api.py:209-210`），`letter_predictor`
也是同一顺序（`letter_predictor.py:86-132`）。没有任何运行期的重排代码。

### 6.2 `option_isolation`：一个构造性不变的开关，在发布权重上不可达

`model.py:97-107` 的 docstring 写得很完整：`option_isolation=True` 时**每个选项跨度自成一个子分支**
（只看得到 state + 指令 + 自己），所有选项跨度**共享同一组位置 id**，`<decide>` 放在最长跨度之后的固定位置。
于是「逐选项表示」与「`<decide>` 对它们的注意力」**在构造上就是置换不变的**。

实现落在两处：`encode()` 里改 `br_pos`（`model.py:124-126`），`branch_mask_batch` 里加一层
`o` 掩码（`:234-241`）。

但它在发布权重上用不了，有三重原因叠加：

1. 训练默认关：`--option_isolation` 默认 **0**（`train.py:349`）；
2. 与 rows 模式互斥：`if self.branch_mode == "rows" and option_isolation: raise`
   （`model.py:338-339`）——发布骨干解析出的正是 rows（§4.1）；
3. 与媒体记录互斥：`encode` 里 `if self.option_isolation: raise ValueError(...)`
   （`model.py:439-440`）。

这是 Kev 那条（默认关 + 混合骨干不支持）的同一条路径、同一类状态。

### 6.3 度量：字段在，committed 产物里没有值

`benchmark.summarize()` 会算一个 `permutation` 块：对每个 `variant == "permuted"` 的 `choice` 行，
把它折算回原始选项顺序后与原始行比，报 `{n, mean_max_delta, flip_rate}`
（`benchmark.py:102-107`、`:115-116`）。另有 `paired_flip()` 对成对样本报 `flip_rate` 与
`both_correct_rate`（`:33-60`）。

**但 committed 的 `results/` 里，属于 JevAny 自己的记录没有一处 `permutation` 带值。** 唯二两处命中：

- `results/model-support-v1.json` 的 `protocol.evaluation` 散文里一次，原文是
  *Current media probes score each row in both choice orders (16 evaluations) to match training-time option
  permutation. Each check records its evaluation size and choice-order coverage.*——对应字段是
  `choice_orders_per_media_row`（文本行 1、媒体行 2），量的是**覆盖了几个顺序**，不是翻转了多少；
- `results/decision-evaluation-v1/official-jev-references.json` 里有 12 个带值的 `permutation` 块
  （`{"n": 72, "mean_max_delta": 0.020611672278338947, "flip_rate": 0.0}` 这一形状），但它们
  **不是 JevAny 的测量**：每一条的 `provenance` 都写着 `published_by_Kev_not_rerun`、
  `source_path` 指向 Kev 仓里的 `runs/jev-decision-v2/report.json`，是**把 Kev 已发布的报告逐字段搬进来当参照物**
  （**实测（committed 产物）**）。

所以这条的准确说法是：**训练期打散、运行期保序、代码里有一个度量字段，而 JevAny 自己的 committed 产物里
这一栏没有值**——它反而把 Kev 的翻转率搬了进来（那批是 0.0，样本量 72 / 60 / 36）。发布卡片在这一栏上不给数字。
这与 Kev（工具在 `/v1/systemone/permute`、发布卡片不给翻转率）、Nimble（工具写好、一次对照都没跑）
三者形态各不同——JevAny 是**报告字段存在、被测的题集里没有 permuted 变体**。

---

## 7. 多问题共享 state：三套机制，而发布权重用的是既不省算也不省账的那套

`DecisionRuntime` 的路径按骨干能力分三层（`runtime.py:94-141`）：

| 情况 | 机制 | `input_tokens` |
|---|---|---|
| packed 骨干 + 缓存命中 | `probs_with_prefix()`：只跑分支，state 的 KV 复用（`model.py:595-612`） | 整条编码长度（一次） |
| packed 骨干 + 缓存未命中 | `probs_and_prefix()`：一次前向同时返回概率与 state 前缀，`past_key_values.crop()` 之后留下 state（`:576-593`） | 同上 |
| rows 骨干 | `probs()`：每题一行，**state 每题各算一遍** | 同上 |

缓存键是 `(state 段那一段 token id, option_isolation)`（`runtime.py:114-115`），容量与门槛来自
`InferenceOptions`：`prefix_cache_size=4`、`prefix_min_tokens=384`（`inference.py:18-22`），
环境变量 `JEVANY_PREFIX_CACHE` / `JEVANY_PREFIX_MIN_TOKENS` 可覆盖。

**关键事实：发布的两个 pointer 检查点在 bf16 下拿不到这个缓存。** `inference_capabilities` 有一段注释与两行
硬关（`model.py:413-419`）：

```python
# Changing sequence shapes in a cached pass amplified BF16 rounding on
# real Llama/Qwen checkpoints. Keep prefix reuse on validated FP32 paths.
if self.lm.dtype != torch.float32:
    capabilities = replace(capabilities, prefix_cache=False)
```

而 `prefix_cache` 本身的允许名单只有 `{"llama", "mistral", "gpt2"}`（`backbones.py:268-276`）——
Qwen / Gemma / Muse 都不在里面。发布检查点的 `head.pt` 里 `weights_dtype` 是 bf16，默认加载路径就是 bf16
（`checkpoint.py:227-231`），所以两层关叠在一起。`describe()` 会把实际生效值报出来
（`runtime.py:79-85`，`enabled = prefix_cache_size > 0 and capabilities.prefix_cache`）。

两条结论：

- **多问题共享 state 在发布权重上是「不做」**：不是被配置关掉，而是被骨干能力关掉，而且运行期不会因此报错——
  它只体现在 `/v1/models` 的 `prefix_cache.enabled: false` 上。
- **`input_tokens` 的账是「编码长度」的账，不是计算量的账**。rows 模式下 state 被重复计算 Q 遍，
  而 `measurements["tokens"]` 仍然是 `len(encoding["ids"])`（`runtime.py:139`），
  也就是那条打包序列的长度——**一次不重复的账配一个重复了 Q 次的计算**。
  这与 SemIf-OpenJev 的 `shared` 模式（"计费只算一次"且确实只算一次）不同，也与 decisio 的
  「定义成题目数」不同：JevAny 的定义是诚实的，只是它量的东西和它干的事不是一件事。

---

## 8. 服务化与 fail-loud

三个路由，全部在 `serve.py` 里（`:116-156`）：

| 路由 | 行为 |
|---|---|
| `POST /v1/systemone` | `runtime.answer(prepare(req))`；`ContextLengthError` → **422** 且带 `X-JevAny-Error-Code: context_length_exceeded`；其余 `ValueError` → **422**（`:126-137`） |
| `GET /v1/models` | `runtime.describe()` + 四项媒体限额（`:140-150`） |
| `GET /health` | 未就绪时 **503**，`{"status": "loading"}`（`:153-156`） |

`describe()` 报的东西（`runtime.py:54-86`）：`id` / `aliases` / `run` / `base` / `lora` / `device` /
`device_map` / `devices` / **`temperature`** / `decision_mode` / `readout` / `backbone_adapter` /
`branch_mode` / `acceleration`（含 `compile_mode`、`lora_merged`、`approximate_bf16_merge`、线性注意力
Triton 内核是否可用、CUDA graphs 统计）/ `capabilities` / `limits` / `prefix_cache`（含 hits / misses）。
choice 那条的 `describe()` 换一整套，并额外报 `choice_readout` 这一整块 self-report
（`letter_runtime.py:58-115`），内容来自 `LetterReadoutPredictor.provenance`（`letter_predictor.py:427-470`）：
`method`、`constraint_emulation`、**`system_prompt_sha256`**、**`chat_template_sha256`**、
`prompt_format_version`、`canonical_base` / `canonical_revision`、`checkpoint`、`adapter_applied` /
`adapter_scale`、`tied_output_embeddings`、`unused_full_output_head_released`、`choice_token_rows`、
`output_softcap`、`temperature` / `native_weight` / `native_weight_argument`、`dtype`、`device`。

### 与 Kev 的守卫对比，JevAny 加厚的位置

- **请求 / 响应两侧都校验**。`validate_response` 在服务端（`runtime.py:152`）与客户端
  （`client.py:212`）各跑一次，内容包括：问题 id 集合相等、`type` 一致、概率键集合相等、
  每个概率有限且 ∈ [0,1]、概率和与 1 的偏差在 `max(1e-5, K × 0.005 + 1e-8)` 之内、
  `choice` 必须是**舍入后**概率里的 argmax、`score` 的 `legend` 必须覆盖每一档
  （`api.py:99-144`）。那个容差本身值得单独记住：**K = 255 时允许总和偏离 1 达 1.275**。
- **`X-JevAny-Error-Code` 这个头是唯一的机器可读错误分类**，客户端靠它把 422 还原成
  `ContextLengthError`（`predictors.py:113-115`）。
- **`/v1/systemone` 没有鉴权**，`JevClient` 默认带 `api_key="local"`；CORS 是 `allow_origins=["*"]`
  （`serve.py:225`）。`GET /v1/models` 的语义被写进了客户端 docstring：*One real round trip, so it also
  proves the endpoint is reachable and ready*（`client.py:194-200`）。
- **媒体面是本地文件、不是 URL**。`prepare()` 拒绝任何含 `://` 的 `uri`，要求解析后的真实路径落在
  `JEVANY_MEDIA_ROOT` 之内，并逐个校验字节数 / 总字节数 / 像素数 / 视频帧数；PIL / PyAV 缺失时
  报的是 422 `install the multimodal extra to serve media`（`serve.py:78-113`、`:44-70`）。
  这是与 H2O-Lightning 那条自带 SSRF 边界的抓取器**相反**的选择：那里只许公网 https、连到检查过的 IP；
  这里干脆不许网络位置。
- **没有的东西**：没有 `/v1/version`、没有请求 id、没有 `server-timing`、没有 prompt 哈希、
  没有 MCP、没有鉴权。描述身份的唯一入口是 `/v1/models`，而它服务的是**加载时认定的那个模型目录**
  ——`request.model` 只被允许等于 `model_id` 或其别名 `jevany-latest`（`runtime.py:144-145`），
  所以响应里的 `model` 不是请求回声，这一点与 Clef / Nimble 的处境不同。

### CUDA graphs：唯一一条会明说「结果不做位精确」的加速路径

`jevany/cudagraphs.py` 为 rows 模式按**填充长度桶**捕获图，桶表是写死的
`LENGTHS = (128, 192, …, 16384)`（`:23-24`）。它的头两段 docstring 把边界写得很清楚：
只对单题请求重放、只有整个模型在一块 CUDA 卡上时可用、**结果不与 eager 路径逐位相同**
（*Results are not bit-identical to the eager path: padded shapes select different kernels*）。
`LoadOptions` 的 `cuda_graph_max_tokens` 默认 2048（`checkpoint.py:166`），
`serve.py` 的 `--cuda-graphs` / `JEVANY_CUDA_GRAPHS=1` 打开。

---

## 9. 置信度从哪来，又怎么被改写

### 9.1 两条链的定性

- **`native` 是训出来的**。头（q/k + 可选的残差 MLP）与 LoRA 都是学出来的参数，训练本身就在改数值、也可能改
  argmax。温度退化成训练之后的单调缩放。
- **`choice` 是读出来的**。零训练、零新增参数；那条链上唯一能改数值的是温度，而温度在构造上改不了 argmax（§5.2）。

### 9.2 「谁在改发布出去的数值」——JevAny 的链表

| 环节 | 机制 | 改 argmax？ | 出处 |
|---|---|---|---|
| ① 训练本身 | pointer：头 + LoRA；direct-token：头 + LoRA 且标签集本身是优化对象；choice：**无此环** | **是** | `model.py:263-286`、`checkpoint.py:258-296` |
| ② 检查点自带温度 | `head.pt` 的 `Meta.temperature`，加载时同时落到 `model.temperature` 与 `head.temperature` | 否（正标量、单调） | `checkpoint.py:281-283`、`model.py:286`、`:492` |
| ③ 部署覆盖 | `JEVANY_TEMPERATURE` 或 `LoadOptions.temperature`；choice 那条拒绝这个变量并改要 `choice_temperature`（**部署参数、不是检查点元数据**，`readout.py:31-54`） | 否 | `checkpoint.py:191`、`runtime.py:219-220` |
| ④ choice 的温度 | `softmax(log_mass / T)` = `p ** (1/T)`，`letter_readout.py:320-344` | 否 | 同上 |
| ⑤ `choice_native_weight` 的 log-linear 混合 | `geometric_blend`：两项取对数加权后重新归一化。**这是第三种运算**——它既不是温度也不是重归一化，会改答案 | **是** | `letter_predictor.py:135-159`、`:598-603` |
| ⑥ 报出去的 `confidence` 公式 | `choice_confidence(p) = (p_max − 1/K)/(1 − 1/K)`；`score_confidence(p) = 1 − E\|level − mode\|/(L − 1)`（后者自陈 *Approximation of TypeSafe's "distance from the modal level" statistic (exact formula unpublished)*） | 否（都是 `p_max` 的单调函数） | `api.py:221-230` |
| ⑦ **`round(x, 2)`** | 概率、`confidence`、`score`、`noul` **全部**四舍五入到 2 位小数 | 否（舍入单调），但会**抹掉一切小于 0.005 的差** | `api.py:233-234`、`:237-248` |

**⑦是这份走查里最容易被漏掉的一环**：它是无条件、无判据、每一条响应都过的，而且它的粒度是 **2 位小数**——
比 Kev 的 `round(·,4)`、Clef 的 `round(·,4)` 都粗两个数量级。两个直接后果：

1. **调用方拿不到原始概率**。一条 3 选项的题，服务报出来的三个数加起来可能不等于 1——而这不是 bug，
   是舍入。`validate_distribution` 那个 `max(1e-5, K × 0.005 + 1e-8)` 的容差正是为它准备的（§8）。
2. **同一题在两次运行之间的差异，如果小于 0.005，在响应里不可见**。这条与 Kev 那篇 §13.3 记的
   「`logits` 已经被温度除过、要乘回去」是同一种性质的东西：**引用服务报出的数之前，先确认你在读哪一层**。

### 9.3 发布出去的 T 是多少

三处 committed 记录给出的是**三个不同的答案**，而且它们指的是不同的检查点，引用时必须连版本一起引：

| 来源 | 检查点 | 温度 |
|---|---|---|
| `results/decision-evaluation-v1/`（2026-09-27） | `tianxinwei/JevAny-27B-SFT@ad7b48b7056a9742f54aacc9b98b6b46dc2ce167` | 运行时记录 **1.8660659830736146** |
| `results/release-equivalence-v0.2.json`（v0.2） | 同一路径的发布仓 | `embedded_temperature` **1.8660659830736146**；同文件的 **source** 侧是 **1.0** |
| `results/choice-readout-v2.json`（2026-10-02） | `SimpleJev/JevAny-Qwen3.5-4B-LoRA@1c7aa9ba…` 与 `SimpleJev/JevAny-Qwen3.8-27B-LoRA@09c9e910…` | `predictor.native_temperature` / `pointer_temperature` 都是 **1.0** |

也就是说：**当前 Hub 上那两个 pointer 检查点，在最近一次 committed 评测里被加载器读出的自带温度是 1.0**
（**实测（committed 产物）**，`source_runs[*].predictor.native_temperature`）。而 `tianxinwei/*` 那一对
（SFT / RLCR）带着 1.86607…，它**今天在 Hub 上查不到**（**实测（Hub）**：不带鉴权时返回
`RepositoryNotFoundError`；是私有还是已删除，本文未核实）。

还有一处与 `native_weight` 有关、容易读错的地方：**v2 报告里那个「additional temperature」不是检查点自带的那个**。
`results/choice-readout-v2.json` 的 `selected` 段里，`blend_temperature` 是**在 Transfer-v9 development 上另拟合**
的一个温度（`ensemble_protocol.temperature_fit = "bounded scalar NLL minimization, T in [0.05, 20]"`），
三个 pointer / direct-token 跑分别是 1.0977784557618349 / 1.1240763000234277 / 0.8352476617411964——
上游 `docs/CHOICE_READOUT.md` 那张表里写的就是它们（1.10 / 1.12 / 0.84）。
同一个 `selected` 段里还有一个 `choice_temperature`（0.8505257300716687 / 1.246278988841639 /
0.5969253208700946），那是 choice 分支自己的那一个。**同一份产物、三个温度、三个不同的对象**。

---

## 10. 四条不变量的落位

| RYOTIDE §13 不变量 | JevAny 的落位 |
|---|---|
| **#1** 读位置必须落在标记上（诊断量 `marker_mass`） | **概念不适用，而且是在两个不同层次上不适用**。native：头的输出维度就是 K（pointer）或只取 K 行（lm_token），分母不存在。choice：质量是**按解码结果 gather** 出来的，归一化只在别名行上做——「落在标记上的份额」这个量在链上没有被定义过。**改写后的等价命题**是「读的那些行必须恰好是这道题的标签」，守卫有三条，全部在构造期与加载期：`decision_verbalizers` 的单 token + 往返（`model.py:45-49`）、`verbalizer_token_ids` 的加载期复验（`:65-75`）、`letter_token_ids` 的「每个用到的 ID 都至少有一条别名」（`letter_readout.py:175-177`）。**运行期一道诊断量都没有** |
| **#2** 前缀与标记表面形式耦合，必须成对扫 | **成立，而且这份耦合被写进了权重契约**。choice 的 prompt 是 `build_prompt` 拼出来的**字面排版**（`letter_readout.py:86-89`），system 提示词是另一段字面量（`:29-33`），两者各有一个 sha256 进 `provenance`（`letter_predictor.py:431-433`）。与 Nimble 的 `prompt_code_sha256` 是同一个做法。**代价要说准**：这两个哈希**只被报出来、不被校验**——`provenance` 进 `/v1/models`，没有任何一处拿它去比对，也没有任何一处据此拒绝加载。对照 Nimble 在加载、合并、起服务三处各重算一次并拒绝不符。native 那侧则**不适用**：标记是 token id（`SPECIAL = LEGACY_TOKENS`，`model.py:14`），不存在表面形式这个自由度；换来的同类耦合是「五个分隔符必须与分词器绑定」，由 `prepare_tokenizer` 的加载期断言兜住（`backbones.py:93-98`） |
| **#3** 只读最后一个位置 | **改写**。native 读的是**每题一行、行内两个构造出来的位置**（`<\|fim_suffix\|>` 当 query、各 `<\|box_end\|>` 当 key）；choice 读的确实是末位，但那个「末位」由 chat 模板在 `enable_thinking=False` 下的生成提示定义，且**每题各一次前向**。「不物化词表维度」这半条在两条链上都成立，机制不同（§5） |
| **#4** 比较概率 / ECE 前先确认 temperature | **成立，而且这条链上要确认的是四件事**：①检查点自带的 `Meta.temperature`（`/v1/models` 会报）；②有没有被 `JEVANY_TEMPERATURE` 覆盖；③choice 那侧的 `choice_temperature` 与 `native_weight` 是**部署参数**，不在检查点里；④**报出来的数已经被舍入到 2 位小数**（§9.2 ⑦）。第 ④ 条是与 RYOTIDE / Kev 都不同的一点：别家要确认的是「这个数在哪个 T 下产生」，这里还要多问一句「它是不是已经被舍到 2 位了」 |

**它自己新增的不变量**（本表最后一行）：**同一个响应里的每一个概率、每一个 `confidence`，都是舍入后的值，
而校验器只用一条 0.5% × K 的容差与「choice 必须是舍入后概率的 argmax」来兜底**（`api.py:90-96`、`:132-133`）。
这条不是缺陷声明，而是一条使用约定——下游若把 `probabilities` 当成精确分布去算熵、KL 或 ECE，
算出来的东西与模型实际输出的分布不是同一个对象。

---

## 11. 文件对照表

| 文件 | 行数 | 职责 |
|---|---:|---|
| `jevany/api.py` | 254 | SystemOne 请求 / 响应形状、`to_record`、`to_answers`、概率与 TypeSafe 公式、**`r2` 舍入**、`output_tokens` 的定义 |
| `jevany/model.py` | 615 | `encode()` 打包、`branch_mask(_batch)`、`rows_of`、`PointerHead`、`DecisionModel`、三种读数路径、前缀缓存前向 |
| `jevany/letter_readout.py` | 359 | choice 读法的**纯函数**部分：prompt 构造、词表扫描、log-mass 聚合、温度 |
| `jevany/letter_predictor.py` | 632 | choice 的**建模**部分：加载、投影行、chat 模板、softcap、native 混合、provenance |
| `jevany/letter_runtime.py` | 144 | 把 choice 预测器套进 SystemOne 契约 |
| `jevany/runtime.py` | 264 | `DecisionRuntime` / `JevModel`：限额、前缀缓存、`answer()` |
| `jevany/readout.py` | 169 | `--readout` 的解析与别名（`letter` → `choice`），**故意不含建模 import** |
| `jevany/backbones.py` | 523 | 五个分隔符、tokenizer 准备、七个 backbone adapter、媒体编码 |
| `jevany/checkpoint.py` | 342 | `head.pt` 的读写、`LoadOptions`、加载路径与温度落位 |
| `jevany/inference.py` | 58 | `InferenceOptions`：state / branch / packed / 前缀缓存四条限额 |
| `jevany/serve.py` | 275 | FastAPI 三路由、媒体校验、`create_app` |
| `jevany/client.py` | 212 | `JevClient`：同源重定向、错误体截断、响应校验 |
| `jevany/predictors.py` | 127 | `LocalPredictor` / `RemotePredictor`（评测侧） |
| `jevany/benchmark.py` | 259 | 打分与落盘：`prediction_rows`、`summarize`、`permutation` 块 |
| `jevany/metrics.py` | 287 | ECE（**10 个等宽箱**）、Brier、NLL、选择性预测、温度拟合、成对 bootstrap |
| `jevany/suite.py` | 154 | 冻结数据集的清单与 sha256 校验、语义哈希去重 |
| `jevany/data.py` | 202 | 训练数据加载、`augment()` 的选项打散、`materialize` |
| `jevany/agent.py` / `agent_harness.py` | 96 / 463 | 把决策嵌进 agent 循环（§15.1） |
| `jevany/symbolic.py` | 113 | 带验证的决策树（§15.3） |
| `jevany/cudagraphs.py` | 108 | rows 模式的 CUDA graph 捕获与重放 |
| `scripts/calibrate_checkpoint.py` | 39 | 在 calibration 分片上拟合温度并写回 `head.pt` |

---

## 12. 亲手跑一遍

上游 README 给的部署命令（`README:215-219`）与训练入口（`README:180-200`）：

```bash
python -m pip install -e '.[serve,multimodal]'
jevany serve --checkpoint SimpleJev/JevAny-Qwen3.5-4B-LoRA --device cuda --dtype bf16 --port 8008
jevany decide examples/request.json --checkpoint runs/my-jev
```

零训练那条读法要显式选：

```bash
jevany eval  --run SimpleJev/JevAny-Qwen3.5-4B-LoRA --suite /path/to/suite \
             --out runs/choice --device cuda --readout choice --choice-temperature 1.0
jevany decide request.json --checkpoint SimpleJev/JevAny-Qwen3.5-4B-LoRA \
             --device cuda --dtype bf16 --readout choice
jevany serve  --checkpoint SimpleJev/JevAny-Qwen3.5-4B-LoRA --device cuda --dtype bf16 --readout choice
```

三条 CLI 的护栏写在参数解析里，不是文档里：`--readout choice` 与 CUDA graph / native 限额类参数**互斥**
（`serve.py:253-258`、`cli.py:76-81`）；`choice_options` 与 `inference_options` 不能同时给
（`create_app` 里显式抛，`serve.py:184-185`）；`JEVANY_TEMPERATURE` 在 choice 模式下被拒
（`runtime.py:219-220`，理由是 *JEVANY_TEMPERATURE applies to the native head*）。

叠加 native 与 choice 要显式给权重（**这一项会改答案**）：

```bash
jevany eval --run SimpleJev/JevAny-Qwen3.5-4B-LoRA --suite /path/to/suite --out runs/stack \
            --device cuda --readout choice --choice-native-weight 0.78 --choice-temperature 0.85
```

**本文没有跑过上面任何一条。** 本机没有 GPU、没有装 `jevany[local]`（需要 `torch>=2.6,<2.15`
与 `transformers>=5.17,<6`，`pyproject.toml:38-39`），也没有下载任何一份基座权重。

---

## 13. 实例走查：在 committed 产物上能复算什么

这一节全部是**不跑上游程序、不加载权重**的核对。三类：纯 Python 重算、Hub 上的容器算术、
在 committed JSON 上重算公式。

### 13.1 令牌与 prompt 的字面量（实测（纯 Python））

`build_prompt` 的输出在 §4.2 已给。此外，**当前源码里那段 system 提示词的 sha256 是
`e2f0ac9f62eb446d7826b3abc3f7909287f4075175e9b58e4cd3ca72718671fe`（284 个字符）**，
而 committed 的 `results/letter-readout-v1.json` 记的是
`547f5cb476dc1d82a769f3f59e22fe0f216569064afb2545ed655b39dae97ca1`，
它自己那一栏写的 `prompt_format_version` 是 **1**，`letter_constraint` 是
*exact decoded uppercase option-letter tokens*（只有大写）。
现在代码里 `provenance["prompt_format_version"]` 是 **2**、`prompt` 是
*Cygnet-compatible A-Z; extended with a-z above 26 options*（`letter_predictor.py:430-433`）。
两者的措辞与数字互为佐证：**v1 那份读数用的不是现在这份提示词，也不是现在这套 52 个 ID 的字母表**。
上游 `docs/CHOICE_READOUT.md` 的 *Historical v1* 一节也这么写，但结论出自代码与产物的对照，不是转述。

### 13.2 head.pt 的容器算术（实测（Hub））

五份发布检查点的 `head.pt` 字节数（Hub API，**只读元数据、不下文件**）：

| 检查点 | 基座 | `hidden_size` | `decision_mode` | `head.pt` |
|---|---|---:|---|---:|
| `JevAny-Qwen3.5-4B-LoRA` | `Qwen/Qwen3.5-4B` | 2560 | pointer | 5,513,315 |
| `JevAny-Gemma-4B-LoRA` | `google/gemma-4-E4B-it` | 2560 | pointer | 5,513,315 |
| `JevAny-Qwen3.8-27B-LoRA` | `Qwen/Qwen3.8-27B` | 5120 | pointer | 10,756,195 |
| `JevAny-Muse-Glimmer-30B-LoRA` | `meta-models/Muse-Glimmer-30B` | 6656 | pointer | 13,901,923 |
| `JevAny-Qwen3.5-4B-Direct-Token-LoRA` | `Qwen/Qwen3.5-4B` | 2560 | **lm_token** | **4,699** |

`PointerHead` 带 `head_residual_dim=256` 时（`recipes/finetune.toml:16` 给的就是这个值，
`train.py:724` 把它映射成 `head_type="residual"`），张量负载是：

```
q.weight [256, d] + q.bias [256] + k.weight [256, d] + k.bias [256]      = 2048d + 2048
residual: Linear(256→256) 256×256×4 + 256×4 = 263,168
          Linear(256→1)   256×4     + 4     =   1,028                    = 264,196
                                                       合计 = 2048d + 266,244
```

四个 pointer 检查点上，`head.pt − (2048·d + 266,244)` **都等于 4,191**（逐仓相同）。而
`JevAny-Qwen3.5-4B-Direct-Token-LoRA`——那个 `decision_mode = lm_token`、`DecisionModel.head is None` 的仓
——它的 `head.pt` **只有 4,699 B**，也就是「同一套 `Meta`、一个张量都没有」的净元数据加容器开销。

**这两条合起来把「头里到底有没有那个残差 MLP」从一个假设变成了算术结果**：若不带残差，那 4.7 KB 的元数据
要变成 268,387 B，而那与同一基座、同一 `Meta` 结构的 lm_token 仓给出的 4,699 B 不相容；
4,191 B 与 4,699 B 才是同一个量级。所以**发布出去的四个 pointer 头都是「q/k + 256→256→1 残差 MLP」**，
全 fp32，且那个残差在初始化时被设成零（`model.py:274-275`），训练起点等价于纯点积。

**这一条是本文唯一一处「用文件大小反推结构」的算术**；要直接核实需要解 `head.pt` 的 pickle，本文没有做。

### 13.3 发布清单与两份产物互证（实测（Hub）+ 实测（committed 产物））

`SimpleJev/JevAny-Qwen3.8-27B-LoRA` 的 `release-manifest.json` 给的是**逐文件 sha256**，
`format` 写着 `JevAny checkpoint v2`。把它与早一天生成的 committed 产物对照：

| 文件 | `release-manifest.json` | `results/letter-readout-v1.json` |
|---|---|---|
| `adapter_model.safetensors` | `f4b835f4f7be78a819067dac226feaad2fc7b9eb0b968c5e6e665ad5328e6731` | 同 |
| `head.pt` | `f78fa283f7e979ce418c5b137d24459350fecaea73e5766894532c1cd99684c7` | 同 |
| `chat_template.jinja` | `c3cf9e34abf4f9e36c2d72165aa9c132d3e2a725b6c2586aaa3a8af9d7a81041` | 同 |

三处逐字节相同，而 `letter-readout-v1.json` 记的 revision 就是当前的 `09c9e910…`。
所以**那份 v1 读数确实是在现在这份发布的权重上产生的**（只是用了一套更早的 prompt 与字母表，见 §13.1）。

### 13.4 在 committed JSON 上重算（实测（committed 产物））

**(a) McNemar 精确检验。** `results/letter-readout-v1.json` 的 `paired_vs_native` 报
JevAny-Qwen3.5-4B 的 letter 读法相对 native：`candidate_wins = 21`、`native_wins = 18`、
`p_value = 0.7492586247608415`。按定义重算两尾精确二项概率

```
2 · Σ_{k≤18} C(39, k) / 2^39 = 0.7492586247608415
```

**逐位相同**；`fixed_blend_0_5` 那组 `9 / 5 → 0.4239501953125` 同样逐位相同。
顺带核 `accuracy_delta`：3/231 = 0.012987…、4/231 = 0.017316…，与记录一致。

**(b) 复合分是四轴的等权调和平均。** 同一文件的 `official_cygnet_reference` 给了
`score = 73.7013` 与四个轴值 71.09 / 87.01 / 90.97 / 56.43、`metric` 自陈是
*equal-weight harmonic mean of Intelligence, Calibration, Speed, and Cost*。
用**记录里那四个已经舍到 2 位的轴值**重算 `4 / Σ(1/aᵢ)` 得 **73.70104690…**，
与 73.7013 差 **2.5e-4**——差异来自轴值本身被舍入，量级吻合。这条同时说明了§5 第 1 条
那边要防的事：**这个 73.70 是复合分，不是准确率**，它自己的 `comparable_to_accuracy` 也写着 `false`。

**(c) 延迟比就是两个中位数相除。** 同一文件 `latency_diagnostic` 里 4B 的
`native.median_ms = 138.4339111391455`、`letter.median_ms = 131.85732485726476`，
`median_speedup_vs_native = 1.0498765335107463`——138.4339…/131.8573… = 1.0498765…，逐位相同。
`interpretation` 里那句「4B 中位延迟改善 1.05×」就是它。（同一段也自陈
`repeats: 1`、`concurrency: 1`、*not a saturated server throughput benchmark*。）

**(d) 计数自洽。** `results/choice-readout-v2.json` 的 `expected_counts` 给 transfer 记录 1,264、
`headline_n` 1,046，而 `datasets[*].runs[*].zero_shot.choice_t1` 里 `n = 1046`、
`excluded_non_headline_rows = 218`——1,264 − 218 = 1,046。同一份产物里
`ece_bins: 10`，与 `metrics.ece(conf, correct, bins=10)` 的默认值一致（`metrics.py:17`）。
JevBench 那一侧：231 = 48（easy）+ 72（original）+ 111（hard），三档相加逐项吻合。

**(e) `results/model-support-v1.json` 的规模与门槛。** `summary` 写 7 个 family / 16 个 series /
**37 个检查点 / 108 项必过检查 / 108 过 / 15 次失败尝试**；`protocol.reload_max_probability_delta_limit`
是 **1e-5**，而逐条记录里实际的最大概率差是 **3.46e-08** 与 **4.40e-08**（重载一致性），
媒体那条同名字段是 **0.197** 与 **0.262**（大两个数量级，但在同一份记录里被分开命名）。
这两组数**本文没有逐条遍历 108 项**，只核了 `summary` 与协议常数两处。

### 13.5 不能复算的部分

- **一切需要前向的量**：五份检查点的任何概率、JevBench / Transfer / Typed / JevJudge 的任何准确率、
  `results/decision-evaluation-v1/comparison.json.gz` 里 16 个模型 × 67 个 panel 的任何数。
  那 7.8 MB 里只有**聚合结果与 panel 的元数据**——逐题读数在 `panels/kev--*.jsonl` 这类文件里，
  而它们**不在仓库中**。
- **`results/decision-evaluation-v1` 里那两个 `tianxinwei/*` 检查点**：Hub 上不带鉴权查不到（§9.3），
  所以「它跑的是不是现在那份权重」无法核实。
- **`SCORE` 与 `accuracy` 两个 CSV**（`accuracy.csv` / `scores.csv`）本文只读了表头与首行，
  没有与 `comparison.json.gz` 做逐格比对。

---

## 14. 模型描述核对（以 Hub 为准）

`SimpleJev` 这个 author 下**恰好五个模型仓**（`HfApi().list_models(author="SimpleJev")`），
与那份 collection 的成员完全一致（**实测（Hub）**）：

| 仓 | 基座（模型卡 `base_model`） | 读法（`release-manifest.json`） | adapter | head.pt |
|---|---|---|---:|---:|
| `SimpleJev/JevAny-Gemma-4B-LoRA` | `google/gemma-4-E4B-it` | pointer | 69,828,016 B | 5,513,315 B |
| `SimpleJev/JevAny-Qwen3.5-4B-LoRA` | `Qwen/Qwen3.5-4B` | pointer | 64,993,432 B | 5,513,315 B |
| `SimpleJev/JevAny-Qwen3.5-4B-Direct-Token-LoRA` | `Qwen/Qwen3.5-4B` | **lm_token** | 64,993,432 B | 4,699 B |
| `SimpleJev/JevAny-Qwen3.8-27B-LoRA` | `Qwen/Qwen3.8-27B` | pointer | 233,584,648 B | 10,756,195 B |
| `SimpleJev/JevAny-Muse-Glimmer-30B-LoRA` | `meta-models/Muse-Glimmer-30B` | pointer | 209,690,512 B | 13,901,923 B |

基座逐项（**实测（Hub）**，取各仓 `config.json`）：

| 基座 | 参数量（safetensors `total`） | 权重文件合计 | 每参数字节 | 精度 |
|---|---:|---:|---:|---|
| `Qwen/Qwen3.5-4B` | 4,659,865,088 | 9,319,828,096 B = 8.68 GiB | 2.00 | BF16 |
| `Qwen/Qwen3.8-27B` | 27,781,427,952 | 55,563,006,776 B = 51.75 GiB | 2.00 | BF16 |
| `google/gemma-4-E4B-it` | 7,996,156,490 | 15,992,595,884 B = 14.89 GiB | 2.00 | BF16 |
| `meta-models/Muse-Glimmer-30B` | 29,776,626,688 | 59,553,435,272 B = 55.46 GiB | 2.00 | BF16 |

四点要写清楚：

1. **`-it` / `-Base` 是不同档。** Gemma 那一档的基座是 `google/gemma-4-E4B-it`（模型卡自陈，
   `tokenizer_config.json` 也随仓发布），而上游 `recipes/family-sft.toml:3` 里写的是
   `google/gemma-4-31B-it`——**配方里的例子与发布的那一档不是同一个尺寸**，引用时不能混。
2. **`JevAny-Qwen3.8-27B` 的默认检查点是 LoRA 目录，不是合并权重。** 上游 README 也这么写
   （*The repository contains LoRA adapter weights and JevAny readout metadata, not the base weights*），
   代码侧的对应事实是 `DEFAULT_CHECKPOINT = "SimpleJev/JevAny-Qwen3.8-27B-LoRA"`（`runtime.py:19`）
   与 `model_name` 的推导（`runtime.py:236-243`）。所以「27B 要 54 GB」是**基座**的体积，
   adapter 只有 233,584,648 B。
3. **`head.pt` 是发布物里唯一不在 HF 常规清单里的权重文件**，它既不是 safetensors 也不是 config，
   而是 `torch.save` 出来的一个 dict（`Meta.to_dict()`，`checkpoint.py:114-115`）。
   每个仓的 `release-manifest.json` 都把它和 adapter 一起列进 `files`，所以身份是可核的（§13.3）。
4. **`tianxinwei/JevAny-27B-SFT` / `-RLCR` 这两个被代码特判的名字（`runtime.py:240`）今天不在 Hub 上。**
   不带鉴权查询返回 `RepositoryNotFoundError`。**私有还是已删除，本文未核实**；
   引用 `results/decision-evaluation-v1` 与 `results/release-v0.2.json` 里那些数字时，
   它们指向的权重**今天无法从 Hub 取回**。
5. **五份清单各自钉了基座 revision，与 Hub 上这四个基座当前的 sha 逐个相同**（**实测（Hub）**）：
   `google/gemma-4-E4B-it` = `ee0ef6023621cff504d758262d4e04895a5af4a2`、`Qwen/Qwen3.5-4B` =
   `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`、`Qwen/Qwen3.8-27B` =
   `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`、`meta-models/Muse-Glimmer-30B` =
   `a4e59da52a7bc87ae7251dd5545c0dd437c44b68`。发布文件集本身也对过一遍：在
   `SimpleJev/JevAny-Qwen3.8-27B-LoRA@09c9e910…` 上取回 `chat_template.jinja`（8,952 B），
   sha256 = `c3cf9e34abf4f9e36c2d72165aa9c132d3e2a725b6c2586aaa3a8af9d7a81041`，与 §13.3 表里
   `release-manifest.json` 记的那个值一致；该清单自己记的 `adapter_config.json`（`31d8e262…`，
   取回后逐字节相符）、`adapter_model.safetensors`（`f4b835f4…`）与 `head.pt`（`f78fa283…`）
   也都在同一份发布里（只取小文件，**没有下载任何权重**）。

---

## 15. JevAny 引入的、别篇没有的三个维度

### 15.1 决策被嵌进一个 agent 循环，而请求是从环境里现拼的

`jevany/agent.py:44-53` 的 `action_request()` 就是那条缝：goal + 当前 observation + 最近 K 步动作当 `state`，
环境的**动作表**当 `criteria`：

```python
{"model": model,
 "state": {"goal": goal, "observation": observation, "recent_actions": history},
 "questions": {"action": {"type": "choice",
                          "instructions": "Choose the next action that makes the most progress toward the goal.",
                          "criteria": dict(actions)}}}
```

`agent_harness.py` 把它做成「前沿模型按需委派」：`BedrockJevAgent` 有三种模式
（`baseline` 只有一步直动作、`optional` 多一个 `delegate_to_jev` 工具、`jev_only` 是不要 Bedrock 的低成本对照，
`:99`），委派时环境会把 subgoal 与原 goal 拼在一起（`:264`）。**返回控制的条件是五个，而不是一个**：
完成、达到步数上限、负奖励、动作被判为无效、**以及 Jev 自己报的 `confidence` 低于阈值**
（`:285-298`）——这是本文库里少见的「把置信度接进控制流」的一处，而且它接的是 §9.2 那个**已经舍入到 2 位小数**
的 `confidence`。

committed 的 `results/agent-harness-v1/formal-matrix.json` 记的是**成对结果**（baseline 与 optional 各一遍、
同一批 seed），含 `success_rate_delta`、`bedrock_call_reduction`、`bedrock_token_reduction`、
`wall_time_reduction` 与**按 AWS 公开价目表估的成本**（价目表日期写在
`pricing_checked_at: 2026-09-30`，且对 `us.openai.gpt-5.6-sol` 明确标注
*No matching public Price List SKU; cached-token price unavailable.*）。
这些数字需要 Bedrock 与 GPU，本文未核。

### 15.2 确定性日期预处理，在模型之前改的是**输入**

`JEVANY_DATE_FACTS=1`（或 `jevany eval --date_facts`）会把 state 里**两两绝对日期**的差写成句子塞回去：
`"August 3, 2026 is 12 days after July 22, 2026."`（`api.py:164-179`），对象型 state 加一个 `date_facts` 字段、
字符串型 state 追加一段（`:182-189`）。它是**纯 Python、可复算**的：日期识别用两条正则
（`api.py:160-161`），少于两个日期就原样返回。

这件事在本文库里是独一份：别家改的是**报出去的数**或**读位置**，这里改的是**喂进去的 state**，
而且改法是确定性的（不涉及模型）。它与 pplx-decider 那条「发布的默认 `max_image_pixels` 落在训练分布之外」
是同一类影响的两种落法——那条是**默认值陷阱**，这条是**显式开关**，代价都要在引用数字前确认它开没开
（`benchmark` 会把 `date_facts` 与 `serve` 的 `DATE_FACTS` 分开记录，`benchmark.py:237`）。

### 15.3 符号决策树：把 Jev 当树的内节点

`jevany/symbolic.py` 的 `JevTree` 在**跑之前**先把树验一遍：根节点存在、节点名与 outcome 名不重叠、
每个节点必须为每一个选项定义一条分支、每个分支目标必须已声明、**无环**、**所有节点与 outcome 都可达**
（`:47-82`）。跑的时候每个节点就是一次 `SystemOneRequest`，答案里报的 `confidence` 会被记进 trace
（`:84-100`）。它自己带一个 `tree_prompt()` 用来让规划模型产这棵树（`:103-113`），并且明确要求
*Do not decide the case itself.*

这一节本身不改变读法，但它演示了 §15.1 之外的另一条消费方式：**同一个决策面既可以被 agent 逐步调用，
也可以被一棵预先验证过的树驱动**，而两者的差别只在「谁保证选项集是闭的」——前者靠环境，后者靠 `_validate`。

---

## 附：本文与上游文档的关系

上游 `docs/` 下有一批面向用户的文档（`ALGORITHM.md` / `API.md` / `CHOICE_READOUT.md` / `TRAINING.md` /
`DEPLOYMENT.md` 等，共 3,284 行），**没有一份是走查稿**。本文的结论全部来自 `jevany/**` 与 `results/**`，
上游文档只用于两处**指路**：§12 的三条命令抄自 README 第 209-224 行，§13.1 提到 `CHOICE_READOUT.md`
的 *Historical v1* 一节与本文的结论方向一致。反过来，本文有两处结论**只能在代码与产物的对照里看出来**，
而上游文档没有写：§6.3（`permutation` 字段存在但 committed 产物里没有值）与 §13.2（head.pt 的字节算术）。

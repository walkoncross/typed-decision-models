# Jeff 是如何把开放 LLM 变成类型化决策模型的

> 代码库：<https://github.com/j-geuter/jeff-serve>
> 对应 README 第 1-4 行：*Serving code for **Jeff-1.0-Large**, a Jev-style decision model: a state, a typed
> question, and its options go in, and one forward pass returns a probability for each option.*
> 本文按 `src/jeff/` 的实际代码逐环节说明这句话的实现方式；文中出现的文件名与行号（`head.py:160`、
> `prompting.py:35-43`、`serve.py:60` 等）均相对该仓库根目录。

本文与 [`docs/ryotide/typed-decision-model.md`](../ryotide/typed-decision-model.md) 同构：那篇讲的是**如何把冻结的
LLM「读」成分类器**，这篇讲的是**如何在那次「读」之上训出一个决策头**。两条路线在第 0 节对照。

## 本文的事实来源

| 类别 | 来源 | 本文中的标记 |
|---|---|---|
| 代码引用 | `src/jeff/**`（本仓） | `文件:行` |
| 由实跑得到的产物 | `prompting.py` 与 `typesafe.py` 是**纯 Python**（只依赖 `hashlib/json/os/string/weakref/math`），可用 `importlib` 按路径加载、**不加载 torch、不加载模型**即可执行 | 标注「实测」 |
| 公式与常量 | README、模型卡 | `README:行` |

**没有跑模型前向**（权重约 61 GB，本机未下载）。所以本文**不给出任何前向产生的概率数值**；
所有「实测」都是不依赖权重的部分：prompt 的字节内容、选项到标记的映射、标签表、serving rule 的算术。
模型侧的输出按第 15 节的方法自行复现。

---

## 0. 这句话的四个词分别落在代码的哪里

| 词 | 含义 | 代码位置 |
|---|---|---|
| **open LLMs** | 现成的公开权重：Gemma 4 31B-it + LoRA，**已合并**后发布 | `model.py:56-67`、`README:6` |
| **Jev-style** | 输入输出是 TypeSafe / JevBench 的线格式；字母读法继承自 SemIf | `typesafe.py`、`prompting.py`（`NOTICE`） |
| **decision model** | 输出**封闭**在一个已知标签集上：不是文本，而是 `K` 个选项上的概率分布 | `head.py:160` `option_logits`、`LABELS` |
| **one forward pass** | 一次前向、不生成任何 token | `model.py:93-100`，`usage.output_tokens = 0` |

关键立场与 RYOTIDE 相对：RYOTIDE 的 "as" 是**重新解读**同一份权重（零新增参数，掩码 LM 自己的
字母 logit）；Jeff 的 "as" 是**在一个位置上训练**——把 LM 隐状态送进一个决策头，头的输出是选项分布。

而这两个立场在代码里是**同一个起点**：

```python
z = c·tanh(W0 h / c) + A·std(h) + b          # README:15
#   └──────┬──────┘   └──┬──┘
#   W0 = LM 输出嵌入的字母行     A = 0, b = 0  ← slice_init_() 把修正项清零
```

`slice_init_()`（`head.py:107-118`）把 `A`–`P` 的 LM unembedding 行拷进 `base_rows`，`delta`/`temp`
权重零初始化；`head.py:15-18` 的 docstring 把它写成一条可检验的性质：*all equal the zero-shot scorer
exactly at initialization when slice-initialized*。所以：

```
Jeff 的头（t=0）  ≡  RYOTIDE 的读法
训练              =  在这条读法上学习一组修正 A, b
```

RYOTIDE 是 Jeff 的**初始化点**，不是它的替代方案。第 3 节展开，并给出上式每个符号在代码里的落点。

---

## 1. 端到端数据流

```
        ┌────────────────── serve.py ──────────────────┐
POST /v1/systemone   {state, model, questions:{name:{type,instructions,criteria}}}
        │  逐题循环（一题一次前向），bad input → HTTP 422 │
        └───────────────────────┬──────────────────────┘
                                v
        ┌────────────────── typesafe.py ───────────────────────┐
        │ question_row_mapped()   问题 → SemIf 格式的一行 row    │
        │   noul   -> options "true"/"false"（4 种顺序映射）     │
        │   choice -> 每个 criteria 键一个 option（请求序）      │
        │   score  -> options "0".."K-1"                        │
        │   description 一律加 "<id>: " 前缀                     │
        └───────────────────────┬──────────────────────────────┘
                                v
        ┌────────────────── prompting.py ──────────────────────┐
        │ messages()  direct-options-v1 / plain-v2 / compact-v3 │
        │ render_prompt()  chat template, add_generation_prompt  │
        │                  =True, enable_thinking=False          │
        │ encode_prompt()  -> input_ids, slot_token_ids, sha256  │
        └───────────────────────┬──────────────────────────────┘
                                v
        ┌───────────── model.py / head.py ─────────────────────┐
        │ base(input_ids, use_cache=False).last_hidden_state[:,-1] │
        │   ← 只取最后一个位置，[1, d_model]，不物化 [1, n, vocab]  │
        │ option_logits(h, k)  → c·tanh(W0 h/c) + A·std(h) + b     │
        │   ← 无效槽位 masked_fill(-inf)，只留 0..k-1              │
        │ softmax(z)           → K 个选项上的概率                   │
        └───────────────────────┬──────────────────────────────┘
                                v
        ┌────────────────── typesafe.apply_rule ───────────────┐
        │ 逐类型 temperature（softmax(log p / T)）              │
        │ noul: 0.2 < p < 0.8 且离 0.5 足够远 → 推到 0.801/0.199 │
        └───────────────────────┬──────────────────────────────┘
                                v
   {"answers": {...}, "usage": {input_tokens, output_tokens: 0}, "runtime": {...}}
```

与 RYOTIDE 的六步时序（其 §11）同构，差别集中在第 4 步：**RYOTIDE 读 logits 再做掩码 softmax，
Jeff 读 hidden state 再过一个头**。前者的"头"就是 LM 自己的 unembedding，后者是一个独立训练的模块。

---

## 2. "typed" 从哪来：单 token 标签同样是硬约束

约束与 RYOTIDE 完全相同：**标签必须能在读位置被当作一个 token 打分**。差异在满足方式。

`prompting.py:35-43` 构造了 255 个标签：

```python
LETTERS = "ABCDEFGHIJKLMNOP"          # SemIf 的宽度
MAX_OPTIONS = 16                      # 头的默认槽位数
_NOT_ONE_TOKEN = set("BQ BZ CJ CQ CZ DQ DZ EJ EY FJ FQ FV FZ "
                     "GJ GK GQ GZ HJ IY JF JG JH JL JN JQ".split())   # 25 个
LABELS = tuple(ascii_uppercase) + tuple(
    a+b for a in ascii_uppercase for b in ascii_uppercase if a+b not in _NOT_ONE_TOKEN)[:229]
MAX_LABELS = len(LABELS)              # 255
```

**实测**（直接跑这段代码）：

| 量 | 值 |
|---|---|
| `len(LABELS)` | **255** |
| 单字母部分 | `A`..`Z`（26 个） |
| 双字母部分 | `AA`..（扫到 `JT` 为止，共 **254** 个候选、其中 25 个在黑名单里、留下 **229** 个） |
| 最后一个标签 | `JT` |
| 黑名单是否全部落在扫描区间内 | 是（25/25） |

即 `26 + 229 = 255`。黑名单来自 `decider` 的 `prompt.label_table`，含义是「这些双字母串在
**Qwen3.5 分词器**里不是一个 token」。宽行实际用到的标记（实测）：

| 选项数 | 标记 |
|---|---|
| 2 | `A`, `B` |
| 16 | `A`..`P`（SemIf 宽度） |
| 20 | `A`..`T` |
| 26 | `A`..`Z` |
| 27 | `A`..`Z`, **`AA`** |
| 255 | `A`..`Z`, `AA`..**`JT`** |

### 三个检查点

1. **`slot_ids()`（`prompting.py:153-167`）做 round-trip 检查**：每个标签必须
   `len(encode(label)) == 1` **且** `decode(encode(label)) == label`，否则抛
   `ValueError: Answer slot 'XX' is not one exact round-trip token`；结果按 tokenizer 记忆化
   （`weakref.WeakKeyDictionary`），并检查 token id 不撞车。
2. **`verify_boundary()`（`prompting.py:178-200`）是 SemIf 的边界检查**：
   `encode(prompt + label) == ids + [label_token_id]`，逐个标签验。
3. **>16 选项时走「尾巴」捷径**（`prompting.py:187-197`）。理由写在注释里：分词器在 BPE
   **之前**就把 added token 切出来，所以对「最后一个 added token 之后的那段」（chat template 的
   `</think>\n\n`）做同样检查是等价的，代价从「255 次全 prompt 重分词（每条宽记录几秒）」降到
   1 次全量 + 若干次短串检查。缓存按 tail 记忆化，上限 1000 条。
   若 prompt 的 ids 不以 tail 的 ids 结尾，回退到全量检查。

**实测**（选项数 → 走哪条路）：3 选项走全量检查；27 选项满足 `len(slots) > len(LETTERS)` → 走 tail 捷径。

### 与 RYOTIDE §2 的直接对照：>26 个选项的两条路

| | RYOTIDE | Jeff |
|---|---|---|
| ≤26 | 字母 `A`–`Z` 一次读 | 字母一次读 |
| \>26 | **两 token 代码** `A0`…`Z9`，两步读 `P(字母) × P(数字\|字母)`，上限 260 | **扩表单 token 标签**到 255，仍然**一次读** |
| 依据 | 无法在单位置给多 token 标签打分 | 找一个「在目标分词器里恰好一个 token」的表面形式 |

两边都在绕「跨 token 的标签无法在单个位置评分」这条硬约束，选了相反的方向。RYOTIDE 的 §9.3 还记
下一条实测结论——**标记必须是模型熟悉的符号**（换成嵌入最远的字符，标记质量从 0.9998 掉到 0.9210）；
Jeff 的标签表同样是「普通大写字母 + 单 token 的双字母串」，符合这条。

### 头的槽位约束

`head.py:160-172` 把「有效槽位」做成张量操作：

```python
invalid = torch.arange(self.k_slots, device=h.device)[None, :] >= k[:, None]
options = z[:, : self.k_slots].masked_fill(invalid, float("-inf"))
```

`k_slots` 是头的槽位数（构造默认 `MAX_OPTIONS` = 16，上限 `MAX_LABELS` = 255，`head.py:64`），
`k` 是每个 row 的选项数。`check_fits()`（`head.py:153-158`）保证没有 row 比头还宽，否则报错并提示
「use a head with head.k_slots >= N」。

`validate_row()`（`prompting.py:59-81`）在 prompt 侧兜底。**实测**：

```
  1 options -> ValueError: options must contain 2-255 entries
  2 options -> accepted
255 options -> accepted
256 options -> ValueError: options must contain 2-255 entries
```

RYOTIDE 是运行期按标签集解析并给出换词建议，Jeff 是**编译期固定槽位 + 掩码**：更省事，代价是头必须
在训练时就预留足够槽位（`head.py:271-279` 的 `load_letter_rows` 甚至会检查 letter_rows 的宽度够不够）。

---

## 3. 决策头：把「读」变成可训练的参数

`head.py:1-39` 的模块 docstring 是一份完整规格。五种 kind（`head.py:52`）：

| kind | 公式 | 初始化 | weight decay 拉向 |
|---|---|---|---|
| `linear` | `z = W h + W_x·std(extra) + b` | `W ← LM 字母行`（slice） | 均匀分布 |
| `residual` | `z = W0 h + A·std([h, extra]) + b`，`W0` 冻结 | `A, b = 0` | **零样本读法** |
| `temperature` | `z = (W0 h) / T(u)`，`T = t_min + softplus(a·u + c)` | `T = 1` | 零样本读法 |
| `typed-residual` | 同 `residual`，但 `delta` 输出 `3 × k_slots`，按问题类型 `q` 取一份 | 同上 | — |
| `typed-regression` | `residual` + `score_regression`，对归一化 level 做高斯 `center/spread` | `spread ≈ 0.20` | — |

### `README:15` 的公式：逐符号对照

`README:15` 印的公式正是 `residual` 这一行——§4 讲 `h` 从哪来，§5 讲 `z` 怎么变成 `p`：

```python
z = c·tanh(W0 h / c) + A·std(h) + b,     p = softmax(z_1, …, z_K)      # README:15
```

| 符号 | 含义 | 代码变量 | 位置 |
|---|---|---|---|
| `z` | 本题的选项 logit 向量，长度 `K`：softmax 的输入 | `option_logits()` 返回的 `options`（已掩码） | `head.py:173-174`、`:182`、`:193` |
| `h` | 一次前向在**最后一个 prompt 位置**的隐状态（post-final-norm，`[1, d_model]`）——正是 LM 自己的 head 要读的那个位置 | `hidden` → `h` | `model.py:94-96`（bf16 往返见 §4） |
| `W0` | LM **输出嵌入**里「答案字母」那几行，`[k_slots, d_model]`；第 j 行 = `LABELS[j]`。**冻结**（`register_buffer` 不进优化器） | `base_rows` | `head.py:83`、`:107-118` |
| `W0 h` | LM 自己读字母得到的原始 logit——零样本读法本身 | `_cap(h @ self.base_rows.T)`，即 `raw` | `head.py:173` |
| `c` | **final-logit soft-capping 常量**（Gemma 4 为 30.0），把 logit 压进 `(−c, c)`；Qwen 系无此常量，退化为恒等 | `self.softcap`、`_cap()` | `head.py:69`、`:145-147`、`:282-287` |
| `A` | **学出来的修正权重矩阵**，`[k_slots, d_model]`，作用在标准化后的 `h` 上 | `delta.weight` | `head.py:86` |
| `b` | 修正偏置，`[k_slots]` | `delta.bias` | `head.py:86` |
| `std(h)` | **逐维 z-score，不是 h 的标准差**：`(h − mu)/sigma`；`mu`/`sigma` 在训练集上算好、存进 checkpoint | `_std()`、`set_standardization()` | `head.py:132-134`、`:120-125` |
| `A·std(h) + b` | 两项合起来就是一个线性层的仿射映射 | `self.delta(u)` | `head.py:176-182` |
| `A=0, b=0` | 初始化时修正项清零，于是 `t=0` 时 `z = c·tanh(W0 h / c)`，**逐位等于零样本读法** | 两次 `nn.init.zeros_` | `head.py:87-88` |
| `p` | `K` 个选项上的概率分布，和为 1；没有东西需要解析，也没有东西可以幻觉（§5） | `probs` | `model.py:99-100` |
| `K` | 本题的选项数；`z` 只有前 `K` 项有效，其余槽位掩成 `-inf` | `k`（张量 `[B]`）、`invalid` | `model.py:97`、`head.py:164` |

四点须知：

1. **softcap 只包住第一项**。代码是先 `_cap(W0 h)`、再 `+ delta`（`head.py:173` → `:182`），修正项不过
   tanh；README 的括号范围与之一致，下面的逐点说明里有它为什么不能漏。
2. **`A·std(h)` 是简写**。喂给 `delta` 的实际输入是 `u = _std(h, extra)`：若开了 `extra`（早期层隐状态，
   `extra_dim=0` 时关闭）还要拼上它；`temperature` kind 的 `u` 另加三个零样本统计量（最大概率、
   归一化熵、`log k`，`head.py:136-143`，且只在 temperature 下启用，`head.py:72`）。对发布用的
   `residual`，`u` 恰好就是 `std(h)`。
3. **`c` 在这一节的另一处公式里是另一个东西**：`temperature` kind 的 `T = t_min + softplus(a·u + c)`
   里那个 `c` 是 `temp.bias`（加性偏置），与 softcap 无关。
4. **公式里的 `A` 与标签 `A` 无关**。`LABELS[0]` 就是字母 `A`，它恰好是 `W0` 的第 0 行；矩阵 `A` 只是
   「修正」的记号。

以下逐点说明（含 `residual` 之外的 kind）：

- **`slice_init_` 是「训练起点 = 零样本」这条性质的实现**（`head.py:107-118`）。
  `collect_letter_rows()`（`head.py:251-262`）用 `model.get_output_embeddings()`（**正确处理 tied
  embedding**）取字母行的权重，存成 `.pt`，好让头训练**不必加载基座模型**。
- **softcap 必须跟着模型走**（`head.py:145-147`、`:282-287`）。Gemma 4 有
  `final_logit_softcapping = 30.0`，LM 自己的 logits 是 `c·tanh(z/c)`；头必须把同一步施加在字母
  logit 上（`raw = self._cap(h @ self.base_rows.T)`，`head.py:173`），否则初始化时就不等于零样本。
  Qwen 系无此常量，`softcap=None` 退化为恒等（`head.py:223` 连 config 都只在有 softcap 时才写）。
- **`_std()` 是逐维 z-score**（`head.py:132-134`），`mu`/`sigma` 在**训练集**上算好、存进
  checkpoint（`set_standardization`，`head.py:120-125`），所以头对 hidden state 的尺度不敏感。
- **`temperature` kind 按构造不改 argmax**（`head.py:191`）：`T(u) > 0` 恒成立，除以正数只改分布的
  陡峭程度。docstring 原话：*decisions (and accuracy) are always those of the zero-shot model;
  only the confidence is learned*。这是 RYOTIDE §9.2 那条立场的**可训练版本**：RYOTIDE 的 `T` 是离线
  拟合的**常量**，Jeff 的 `T(u)` 是**逐输入学出来的函数**，输入 `u` 还额外拼了三个零样本统计量
  （`logit_stats`：最大概率、归一化熵、`log k`，`head.py:136-143`，且只在 temperature kind 下启用，
  `head.py:72`）。
- **一个 NaN 梯度的坑被注释下来**（`head.py:190`）：temperature 分支必须**先除再掩码**，否则
  `-inf / T` 会产生 NaN 梯度。
- **`_inverse_softplus` 让初值精确**（`head.py:55`、`:97`）：
  `t_min + softplus(inverse_softplus(1 - t_min)) = 1`，即 T 从**恰好 1.0** 开始。同款技巧用于
  `typed-regression` 的 `spread` 初值（`_inverse_softplus(0.20)`，`head.py:93`）。
- **abstain 槽位不进发布分布**。`option_logits` 返回 `(options, abstain)`，其中
  `abstain = logsumexp(options) + v·u + b_a`，`b_a = -3`（约 5% 弃权率）。但 `Decider.decide`
  **丢掉第二个返回值**（`model.py:98` 的 `logits, _ =`）。docstring 说得很直白：*used only by the
  expected-utility objective and never part of the option distribution reported to the evaluator*。

### 一处死代码级观察（对应 RYOTIDE §9.3 的同类发现）

**`typed-*` 两种头在发布路径上不可达。** `head.py:179-180` 要求显式传入问题类型：

```python
if q is None:
    raise ValueError("typed heads require explicit question types")
```

而 `Decider.decide`（`model.py:98`）调用 `option_logits(h, k)` 时**从不传 `q`**。所以

- `Jeff-1.0-Large` 发布的头必然是 `linear` / `residual` / `temperature` 之一，与 `README:15` 的公式
  对应即 `residual`；
- `q` 的取值约定（代码里只有 `q == 2` 走 score 分支，`head.py:189`）属于**训练侧**，不在这份 serving
  代码里，本文不推测。

这与 RYOTIDE §9.3「`calibration_bias()` 全仓零调用点」是同一类事实：**算得出来 ≠ 接进来了**。

---

## 4. 一次前向：只读最后一个位置

`model.py:87-103` 是全部：

```python
hidden = self.base(input_ids=torch.tensor([ids], device=self.device), use_cache=False,
                   return_dict=True).last_hidden_state[:, -1]      # 只取最后一行
h = hidden.to(torch.bfloat16).float()                          # "as in our evaluations"
k = torch.tensor([len(row["options"])], device=self.device)
logits, _ = self.readout.option_logits(h, k)                   # 头；abstain 被丢弃
logits = logits[0, : len(row["options"])]
probs = torch.softmax(logits, dim=-1).tolist()
```

- 这是 RYOTIDE 不变量 #3 的同一条。RYOTIDE 靠分块前填 + `logits_to_keep=1` 避免物化
  `[1, n_tokens, vocab]`；Jeff 用的是 `last_hidden_state` 而不是 `logits`，取到的是 `[1, d_model]`，
  **天然不物化 vocab 维度**，也不需要分块。
- `use_cache=False`：**不建 KV cache**。因为每道题都是独立的一次前向（见 §7），没有可复用的共享
  前缀，RYOTIDE §8 的三种分支策略（fork/trim/lazy）在这里没有对应物。
- `bf16 → float` 的往返是**刻意保留**的（注释 "as in our evaluations"）：离线评测在 bf16 下做，
  serving 必须复现同一条数值路径。
- **没有运行期标定、没有 bias**：一切都在训练期学定。RYOTIDE 需要运行期标定读位置（其 §3）与上下文
  校准 bias（其 §9.3），Jeff 把读位置**固化**在 `add_generation_prompt=True, enable_thinking=False`
  的模板末尾（`prompting.py:138-141`），头就训在那个位置上。
- 代价同样清晰：**prompt 格式一旦固定就不能换**——`PROMPT_VERSION` 参与特征缓存命名
  （`prompting.py:19-24`、`:45`），因为头是在那个位置上训的。RYOTIDE 可以在多个模型的多个模板间
  探测读位置，Jeff 没有这个自由度。

---

## 5. 掩码 + softmax：决策本身

`head.py:160-193` `option_logits()` → `model.py:100` `softmax`。就这两步。

- 无效槽位被 `masked_fill(-inf)` 排除（`head.py:171` / `:174` / `:191`），`logits[0, :k]` 再取前 `k` 个，
  softmax 因此**恰好**在 `K` 个选项上归一化，和为 1；**没有东西需要解析，也没有东西可以幻觉**。
- 与 RYOTIDE 的差异：RYOTIDE 的 `TorchClassifier.read()`（其 `torch_backend.py:277`）返回
  **`(probs, mass)`** 两样东西，其中 `mass` = 掩码**之前**全词表分布落在标记上的份额，并写进每条结果
  的 `runtime.marker_mass`。**Jeff 没有这个量。**

  这不是小事。RYOTIDE 记录过：Qwen3.5 的生成模板会打开 `<think>` 块，读位置落进推理过程，只有约
  15% 概率质量到达选项标记，而**适配器照样返回一个格式正确的分布、harness 照样打分**——分数从
  27.4 变成 61.8，靠 `marker_mass` 才抓住。Jeff 用自己的模板把读位置钉死，从设计上消掉了这类失败，
  代价是**失去了对应的诊断量**：`runtime` 里只有
  `probability_origin: "softmax over the options + serving rule"`（`serve.py:83`），看不到「有多少质量
  落在选项上」。这是§13 不变量 #1 在 Jeff 里的状态：约束成立，但没有体检指标。

---

## 6. 顺序偏置：提供开关，而不是平均

JevBench 记录过 open-alternative-jev 在 `A. yes, B. no` 上得 72%、顺序反过来只有 21%——选项顺序
偏置是小模型规模下的主要失效模式。`typesafe.py:44-61` 提供了四种 yes/no 渲染：

```python
NOUL_MAPPINGS = ("tf-true-first", "tf-false-first", "yn-yes-first", "yn-no-first")
```

**实测**（`urgent` 一题，criteria `{"true": "A reply is needed today", "false": "It can wait"}`）：

| mapping | option id 顺序 | description |
|---|---|---|
| `tf-true-first`（默认，= semif_direct） | `true`, `false` | `true: A reply is needed today` / `false: It can wait` |
| `tf-false-first` | `false`, `true` | `false: It can wait` / `true: A reply is needed today` |
| `yn-yes-first` | `true`, `false` | `yes: A reply is needed today` / `no: It can wait` |
| `yn-no-first` | `false`, `true` | `no: It can wait` / `yes: A reply is needed today` |

设计要点：**option id 始终是 `"true"` / `"false"`**，换的只是「谁占 slot 0」和 description 的措辞。
于是 `answer()`（`typesafe.py:70-78`）读的 `probs["true"]` 与顺序无关——docstring 明说
*Option ids stay "true" / "false", so answer() is unchanged*。**实测**：

```
answer(noul)  : {'type': 'noul', 'noul': 0.62}                      # 输入 probs 按 id 给出
answer(score) : {'type': 'score', 'score': 2, 'probabilities': {...}}
answer(choice): {'type': 'choice', 'choice': 'billing', ...}
```

> **与 RYOTIDE §6 的分工差异（重要）**：RYOTIDE 用**概率平均**去偏——多个 order 各 fork 一条 KV
> 分支，把每个 order 的 slot 概率**折叠回真实标签下标**再平均（`totals[li] += p[slot]`）。这是一套
> 运行期机制，实现在 `run()` 里。Jeff 这里**只提供切换顺序的开关，本身不做任何平均**：去偏要么在
> 训练期被消化掉，要么留给评测方做 A/B。两条路线在这一点上不是实现差异，而是**职责分配**差异。

---

## 7. 多问题共享一次 state：Jeff 不做

一个 TypeSafe 请求可以针对同一 state 问多个问题（Decision Index 的 ToolRet 平均每请求约 192 个）。
RYOTIDE 为此有 `run_many()`：求所有问题的公共 token 前缀、共享头预填充一次、每问题在自己分支上只
处理自己的 token、计费上共享 state **只算一次**（其 §7）。

Jeff 的处理是**朴素的逐题循环**（`serve.py:51-56`）：

```python
for key, question in questions.items():
    row = model.row(body["state"], question, key)
    probs, _, n, seconds = model.decider.decide(row)
    answers[key] = model.answer(question["type"], [o["id"] for o in row["options"]], probs)
    tokens, forward = tokens + n, forward + seconds       # 每个问题全额计费
```

每个问题的 prompt 里**都完整包含 `state`**，所以同一段 state 被重复编码 `len(questions)` 次，
`usage.input_tokens` 是逐题累加。这是 Jeffrey 相对 RYOTIDE 明确放弃的一层优化；换来的好处是
`use_cache=False` 的简单性（无需 KV 分支、无内存算术、无并发下的 cache 共享问题）。

---

## 8. 从已分类的分支继续生成：没有

RYOTIDE 的 `generate.py` 演示「判类之后，分支 cache 里已经有 state + 问题，注入结果再解码」。
Jeff **不生成任何 token**：`usage.output_tokens` 在代码里是**硬编码的 0**（`serve.py:60`），而不是
统计出来的。这正是 "one forward pass" 的字面含义——与 RYOTIDE 的 `output_tokens = 0` 完全一致。

---

## 9. 置信度从哪来，又怎么被改写

### 9.0 一条链

| 环节 | 代码 | 改发布概率的数值？ | 改 argmax？ | 发布 preset 里生效？ |
|---|---|---|---|---|
| 训练决策头（含 `residual` 的 A、b） | `head.py`（训练侧） | **是** | **是** | 是（baked in 权重） |
| `temperature` kind：学出的逐输入 `T(u)` | `head.py:191` | 是 | **否，按构造** | 取决于发布头是否选它 |
| serving rule：逐类型 temperature | `typesafe.py:99-104`、`:88-89` | 是 | 否 | 是（若 `serving_rule.json` 配了） |
| serving rule：`noul_edge` 带内推移 | `typesafe.py:92-96` | 是 | 否 | 是（若配了） |

与 RYOTIDE §9.0 的表对照：RYOTIDE 那一列「发布出去的概率数值只被 temperature 改变」，因为它零参数、
纯读；Jeff 多了一行**训练本身**——头的 A、b 是学出来的，它既改数值也可能改 argmax（这是「训练」相对
「读」的必然代价，也是换来的精度）。**训练之后**，能再动发布数值的只有 serving rule 的两项。

### 9.1 serving rule 的 temperature 与 RYOTIDE 是同一个式子

```python
def score_temperature(probs, temp):                      # typesafe.py:99-104
    logs = [math.log(max(p, 1e-12)) / temp for p in probs]
    top = max(logs)
    exp = [math.exp(v - top) for v in logs]
    total = sum(exp)
    return [v / total for v in exp]
```

RYOTIDE 的 `jevbench_adapter.py:625`：

```python
z = [math.log(max(x, 1e-300)) / self.temperature for x in p]
```

**逐字同一招**（只差 epsilon 常量）。数学上等价于 `softmax(logit / T)`，因此**不改变选项排序、只改变
分布的陡峭程度**。yes/no 版是同式的二分类形式：`noul_temperature(p, T) = sigmoid(logit(p)/T)`
（`typesafe.py:88-89`）。

**实测**（argmax 不变性）：

```
raw = [0.02, 0.13, 0.55, 0.22, 0.08]        # 一条 5 级 severity
  T=0.5 -> 0.0011 0.0451 0.8075 0.1292 0.0171   argmax=2
  T=1.0 -> 0.0200 0.1300 0.5500 0.2200 0.0800   argmax=2
  T=2.0 -> 0.0709 0.1807 0.3716 0.2351 0.1417   argmax=2
  T=4.0 -> 0.1231 0.1966 0.2819 0.2242 0.1741   argmax=2
  argmax changed in 0 / 100000 random (p, T) pairs
```

T=0.5 时最高项从 0.55 升到 0.8075，T=4 时降到 0.2819；预测标签始终是 level 2。所以——与 RYOTIDE
§9.2 一样——**逐类型的 T 可以被单独拟合与事后报告，因为它不可能改变准确率**。

对照 RYOTIDE 的机制差异：RYOTIDE 的 T 是 **preset 里钉死的常量**（Gemma 1.924 / Qwen9 0.667 /
Qwen3.5-4B 拟合出 1.592 但留出集变差被拒，保持 1.0），由 `bench/fit_temperature.py` 离线拟合。
Jeff 的 T 是**配置**（`serving_rule.json`），且另有一条**逐输入学习**的版本（`temperature` kind）。

### 9.2 `noul_edge`：比 temperature 更硬的一处改写

```python
NOUL_YES, NOUL_NO = 0.80, 0.20      # JevBench 的 yes/no 带
BAND_YES, BAND_NO = 0.801, 0.199    # 一个「已下定决心」的答案被推到带外

def noul_edge(p, margin=0.0):
    if NOUL_NO < p < NOUL_YES and abs(p - 0.5) >= margin and p != 0.5:
        return BAND_YES if p > 0.5 else BAND_NO
    return p
```

`typesafe.py:8-9` 的 docstring 直说了动机：*JevBench counts answers inside that band as abstentions
(wrong)*。**这不是校准，是适配评分规则。实测**：

| raw P(yes) | `margin=0.0` | `margin=0.15` |
|---|---|---|
| 0.05 | 0.05 | 0.05 |
| 0.20 | 0.20 | 0.20 |
| 0.201 | **0.199** | **0.199** |
| 0.50 | 0.50 | 0.50 |
| 0.60 | **0.801** | 0.60 |
| 0.79 | **0.801** | **0.801** |
| 0.80 | 0.80 | 0.80 |
| 0.95 | 0.95 | 0.95 |

两条读数：

- `margin` 的含义是「离 0.5 至少这么远才算已下定决心」：`margin=0.15` 时 0.60 不动（|0.60−0.5| = 0.10），
  0.79 动。
- **`noul_edge` 会破坏单调性。** 0.7999 → 0.801，而 0.8001 → 0.8001：**raw 更大，报出去更小**。
  同理 0.201 → 0.199 < 0.20 → 0.20。所以跨样本比较**已被 rule 改写的** P(yes) 的大小是不可靠的；
  要比较就必须先确认两侧用的是同一套 `serving_rule`。这是 §13 不变量 #4 在 Jeff 里的加强版——
  RYOTIDE 只提醒「先确认 temperature」，Jeff 还要多确认一项 `noul_edge`。

`apply_rule` 的施加顺序是 **先 temperature、后 edge**（`typesafe.py:113-119`）。**实测**（`rule =
{noul_t:1.5, noul_edge:0.15, score_t:0.8, choice_t:1.2}`）：

```
choice  : {'billing': 0.5149, 'technical': 0.3107, 'shipping': 0.1744}
noul    : {'yes': 0.5809, 'no': 0.4191}        # P(yes)=0.62 → T=1.5 → 0.5809 → |−0.5|>0.15? 否，不动
score   : {'0': 0.0100, '1': 0.1037, '2': 0.6295, '3': 0.2002, '4': 0.0565}
no rule : {'yes': 0.62, 'no': 0.38}            # 每一项缺失即恒等
```

注意 score 分支会 `sorted(probs, key=int)` 后重排（`typesafe.py:121`），choice 分支则保持插入顺序
（`typesafe.py:111`）——两者都保证输出 dict 与输入的键集一致。

---

## 10. 服务化

`serve.py` 实现 TypeSafe 的 `POST /v1/systemone`，因此 JevBench 用**原装 `typesafe` 适配器**就能驱动
它，不需要任何 Jeff 专属代码——这一点与 RYOTIDE 的 `server.py` 完全一致。

- **健康检查是 `GET /`**，返回 `{"status": "ok", "model": ...}`（`serve.py:40-41`），不是 RYOTIDE 的
  `/health`。没有 `prompt_hash` 字段；**配置指纹以另一套形式暴露**：响应的 `runtime` 里带
  `readout`、`probability_origin`、`serving_rule`（`serve.py:82-84`）。
- **`serving_rule` 走 `runtime` 而非响应顶层**：`rule` 在启动时从 `serving_rule.json` 读入
  （`model.py:125`），也可由 `JeffModel(rule=...)` 覆盖。
- **`usage.input_tokens` 是逐题累加的和**，`output_tokens` 恒为 0（`serve.py:60`）；
  `runtime.forward_seconds` 是各题前向耗时之和。
- **错误 → HTTP 422**，只捕获 `KeyError/TypeError/ValueError`（`serve.py:57-58`），把异常类型与消息
  回给客户端。输入超过 8192 token 走的是 `ValueError`（`prompting.py:215`），因此也是 422。
- **输入上限 8192 token，超长拒绝而非截断**（`MAX_TOKENS`，`model.py:26`；`prompting.py:214-215`）。
  RYOTIDE 的做法也是拒绝（"no truncation allowed"），理由是 JevBench 把超长算错。
- **启动 warm-up 3 次**（`serve.py:77-78`）：不吃计时的前向，把权重加载、kernel 编译与 allocator
  预热排除在计时之外。RYOTIDE 的 `main()` 也是「load 后打一条不计时请求」。
- **并发：单线程串行**（`http.server.HTTPServer`，`serve.py:85`），请求一条一条处理。RYOTIDE 的
  `Engine.decide()` 用一把锁包住决策（因为 MLX 不是线程安全的）——两边都选择了串行化，理由不同。
- **cuDNN attention 默认关闭**（`model.py:30-40`）。注释给出了实测：cuDNN attention 为**每个新输入
  形状**重建执行计划，而 serving 请求长度几乎总在变——H200 上约 2×（0.123 s vs 0.061 s 中位数），
  H100 + torch 2.11 上高达 10×（0.72 s vs 0.07 s），**答案不变**。长输入是例外，所以
  `Decider` 从 `CUDNN_MIN_TOKENS = 8192` 起把 cuDNN 打开（`model.py:90-91`、`:133`）。
- **容器**：`Dockerfile:8` 特意装了 `gcc`——Triton（torch 给 Gemma 的 rotary embedding 等算子用）
  会在**首次请求**时用它编译一个驱动小工具，缺了会**在第一次前向就崩**。这条和 RYOTIDE 把首次编译
  排出计时的动机是同一件事的两面。

---

## 11. 拼在一起：一次决策的完整时序

```
1. 启动（不计时）
   JeffModel.__init__: 读 jeff_config.json / serving_rule.json
   加载基座模型（bf16, device_map={"": "cuda"}）+ head.safetensors
   Decider.__init__: head.to(device).eval()
   main(): 3 次 WARMUP 前向

2. 请求
   serve.py do_POST → 校验 questions 是非空 dict

3. 逐题（一题一次前向，串行）
   a. typesafe.question_row_mapped(state, question, key, noul_mapping)
      → row{id, state, question, options[{id, description}]}
   b. prompting.encode_prompt(tokenizer, row, max_tokens)
      messages() = [system("Apply the supplied criterion..."),
                    user(JSON{evidence, criterion, options[{letter, description}]})]
      render_prompt: chat template, add_generation_prompt=True, enable_thinking=False
      ids = encode(prompt);  len(ids) > 8192 → ValueError → 422
      slots = slot_ids(tokenizer, k)          # 单 token round-trip 检查
      （serving 路径 check_boundary=False，首尾检查在训练期做）
   c. Decider.decide(row)
      前向 1 次，取 last_hidden_state[:, -1] → h [1, d_model]
      h = bf16 → fp32
      option_logits(h, k): z = c·tanh(W0 h/c) + A·std(h) + b
                           无效槽位 -inf；abstain 被丢弃
      probs = softmax(z[0, :k])
   d. served_answer → typesafe.apply_rule
      逐类型 temperature；noul 再做带内推移
      → {"type", choice/score/noul, probabilities}

4. 响应
   {"answers": {...}, "model": name,
    "usage": {"input_tokens": Σn, "output_tokens": 0},
    "runtime": {model, gpu, dtype, readout, probability_origin, serving_rule,
                torch, cudnn_attention, forward_seconds, input_tokens}}
```

**全程 `output_tokens = 0`**：没有任何 token 被生成，这正是 "one forward pass" 的字面含义。

---

## 12. 实例走查：三条真实记录

数据来源：本节所有 prompt 文本、选项映射、标签与 rule 数值都是**实跑 `prompting.py` / `typesafe.py`
得到的**（纯 Python，无 torch、无权重）。**概率数值未跑模型前向，因此不给出**——第 15 节给出复现命令。

三条记录都取自 `scripts/smoke_test.py` 的同一个请求：

```python
state = {"ticket": "I was charged twice for order #4411 and want the second charge refunded.",
         "customer_tier": "gold"}
```

### 实例 ①：`route`（choice，3 个选项）

#### (a) wire → row

```json
{"type": "choice", "instructions": "Which team should handle this ticket?",
 "criteria": {"billing": "Payments, charges and refunds",
              "technical": "Bugs and outages", "shipping": "Delivery problems"}}
```

**实测** `question_row_mapped` 的输出：

```json
{"id": "route",
 "state": {"ticket": "I was charged twice for order #4411 and want the second charge refunded.",
           "customer_tier": "gold"},
 "question": "Which team should handle this ticket?",
 "options": [
   {"id": "billing",   "description": "billing: Payments, charges and refunds"},
   {"id": "technical", "description": "technical: Bugs and outages"},
   {"id": "shipping",  "description": "shipping: Delivery problems"}]}
```

注意 `description` 的 `"<id>: "` 前缀是 `question_options()` 加的（`typesafe.py:35`），为的是对齐
JevBench 的 `semif_direct` 适配器。criteria 的**键序**就是选项顺序（JSON 对象键序在此处被当作语义用）。

#### (b) prompt 的真实形态（`direct-options-v1`，默认格式）

**实测** `direct_messages(row)`：

```
[system] Apply the supplied criterion to the supplied evidence. Choose exactly one listed
         option. Respond with only its uppercase letter, with no explanation or reasoning.

[user]   {"evidence": {"ticket": "I was charged twice for order #4411 and want the second
         charge refunded.", "customer_tier": "gold"}, "criterion": "Which team should handle
         this ticket?", "options": [{"letter": "A", "description": "billing: Payments,
         charges and refunds"}, {"letter": "B", "description": "technical: Bugs and outages"},
         {"letter": "C", "description": "shipping": "Delivery problems"}]}
```

（上面为可读性加了换行，实际是**一行 JSON**，`ensure_ascii=False`。）三件事同时可见：

1. **state 直接作为 JSON 的 `evidence` 字段**，只出现一次——与 RYOTIDE 的 `repeat=2`（问题体重复两次、
   state 只出现一次）不同，Jeff 不做 echo。
2. `options` 的 `letter` 来自 `LABELS`（`prompting.py:90`），**与选项 id 无关**：`billing` 是 `A`，
   因为它是第 0 个。
3. 指令在 **system** 里（`DIRECT_SYSTEM`），不是收尾行。

同一条 row 在另外两种格式下（**实测**，为 opt-in）：

```
---- plain-v2 ----
[user] State:
{"ticket":"I was charged twice for order #4411 and want the second charge refunded.","customer_tier":"gold"}

Question: Which team should handle this ticket?
Choose the option that fits best.

Options:
A. billing: Payments, charges and refunds
B. technical: Bugs and outages
C. shipping: Delivery problems

---- compact-v3 ----          # plain-v2 去掉逐类型指令行
[user] {"ticket":"I was charged twice for order #4411 and want the second charge refunded.","customer_tier":"gold"}

Question: Which team should handle this ticket?
A. billing: Payments, charges and refunds
B. technical: Bugs and outages
C. shipping: Delivery problems
```

⚠️ **一处文档与代码不一致**：`prompting.py:16-19` 的 docstring 说格式由环境变量 `JEV_PROMPT_FORMAT`
选择、「read once at import」，但代码里 `PROMPT_VERSION = "direct-options-v1"` 是**硬编码**的
（`:45`），`os` 虽被 import（`:31`）却**从未使用**。当前版本**不能**用环境变量切格式。

#### (c) 标记 → 槽位

`LABELS[:3]` = `('A','B','C')`，`slot_ids()` 对三者各做一次 round-trip 检查。头用槽位 0..2，
`invalid = [False, False, False]`，其余 13 个槽位（`k_slots=16` 时）被 `-inf` 掩掉。

#### (d) 读与产出

`last_hidden_state[:, -1]` → `h` → `option_logits(h, k=3)` → 3 个 logit → softmax → 3 个概率。
`answer()` 取 argmax 对应 id，`served_answer` 再按 `choice_t` 做 temperature（若配了），
返回 `{"type":"choice","choice":best,"probabilities":{...}}`（**实测** shaping 见 §6）。

### 实例 ②：27 个选项（>26 → RYOTIDE 会切代码路径，Jeff 不会）

**实测**：一条 27 选项的 choice row，其选项块是

```
Options（实际是 JSON 的 options 数组，此处按渲染后的形态列出）:
  {'letter': 'A',  'description': 'intent_00: the customer wants intent 0'}
  {'letter': 'B',  'description': 'intent_01: the customer wants intent 1'}
  {'letter': 'C',  'description': 'intent_02: the customer wants intent 2'}
   ...
  {'letter': 'Y',  'description': 'intent_24: the customer wants intent 24'}
  {'letter': 'Z',  'description': 'intent_25: the customer wants intent 25'}
  {'letter': 'AA', 'description': 'intent_26: the customer wants intent 26'}
```

与实例 ① 的**唯一**差别是标记表向后延伸：`A`..`Z` 之后接 `AA`。**没有代码路径切换、没有两步读、
没有 `P(字母) × P(数字|字母)`**——因为它仍然是一次读、一个 softmax：

```
27 options -> 第 27 个标记 = 'AA'         （LABELS[26]）
27 > len(LETTERS)=16  ->  verify_boundary 走 tail 捷径
头的槽位：0..26 有效，其余 -inf
```

这条记录同时说明了 §2 那个取舍的代价与收益：Jeff 用**更大的单 token 标签表**换掉了 RYOTIDE 的两步读
（更简单、和为 1 且精确），代价是标签表与**特定分词器**绑定（`_NOT_ONE_TOKEN` 是按 Qwen3.5 定的），
换分词器就得重新标定哪些双字母串是单 token。

### 实例 ③：`urgent`（noul，yes/no）——顺序偏置与带内推移

**(a) 四种顺序映射的实测输出**见 §6 的表。注意 option **id 恒定**，只有 slot 位置与措辞变。

**(b) serving rule 的实测数值**见 §9.2：`P(yes) = 0.60` 在 `margin=0` 下被推到 `0.801`，
在 `margin=0.15` 下保持 `0.60`；而 `0.7999 → 0.801` 与 `0.8001 → 0.8001` 一起说明
**该 rule 破坏 P(yes) 的单调性**。

### (c) 反例：超过 255 个选项

**实测**（`validate_row`）：

```
  1 options -> ValueError: options must contain 2-255 entries
  2 options -> accepted
255 options -> accepted
256 options -> ValueError: options must contain 2-255 entries
```

头的槽位上限与 prompt 的上限是**同一个常量** `MAX_LABELS = 255`，两层同时兜底。

---

## 13. 四条不变量在 Jeff 里的状态

| RYOTIDE 的不变量 | Jeff 的对应物 | 状态 |
|---|---|---|
| **#1** 读位置必须落在标记上（诊断量 `marker_mass`） | 头的数据已指向标记槽位；其余槽位 `-inf` | 结构性成立，但**无诊断量**（§5）。`runtime` 里没有类似 `marker_mass` 的健康指标 |
| **#2** 前缀与标记表面形式耦合，必须成对扫 | prompt 格式与头**一起被冻结**，无运行期探测 | 成立，且被训练固化；代价是不能换格式（§4） |
| **#3** 只读最后一个位置 | `.last_hidden_state[:, -1]` | 成立，且更彻底（连 vocab 维度都不物化） |
| **#4** 比较概率或 ECE 之前，先确认 temperature | `runtime.serving_rule` 回传；且**多一项 `noul_edge`** | 成立，且**有两处**改写（§9）；`noul_edge` 还会破坏单调性 |
| （新增，训练侧） | **单 token 标签表与分词器绑定** | 换分词器必须重标 `_NOT_ONE_TOKEN` 与 `slot_ids` |

---

## 14. 文件对照

| 文件 | 行数 | 职责 |
|---|---|---|
| `src/jeff/prompting.py` | 220 | SemIf 格式 prompt 构造（字节级一致）、255 标签表、`slot_ids` round-trip 检查、`verify_boundary` 与 >16 选项的 tail 捷径、三种 prompt 格式 |
| `src/jeff/typesafe.py` | 123 | TypeSafe/JevBench 线格式 ↔ SemIf row；`answer()`；**serving rule**（逐类型 temperature + `noul_edge`）；4 种 yes/no 顺序映射 |
| `src/jeff/head.py` | 303 | `DecisionHead` 五种 kind、`slice_init_`（= 零样本读法）、softcap、`_std` 标准化、`option_logits` 掩码、abstain 单元、safetensors 存取 |
| `src/jeff/model.py` | 148 | 权重与头的加载、`Decider.decide`（一次前向 + 只读末位）、`served_answer`、`JeffModel.decide`、cuDNN attention 开关 |
| `src/jeff/serve.py` | 93 | TypeSafe HTTP 服务（`/v1/systemone`、`GET /`）、warm-up、`runtime` 元数据 |
| `src/jeff/__init__.py` | 5 | 导出 `JeffModel` |
| `scripts/smoke_test.py` | 36 | 三种题型（choice / noul / score）的端到端冒烟测试 |
| `Dockerfile` | 17 | 无网络容器；装 `gcc` 供 Triton 首次编译；读 `/model` |
| `requirements.txt` | 8 | 测试过的精确版本（torch 2.13.0 / transformers 5.17.0 / CUDA 13.0 轮子） |

---

## 15. 亲手跑一遍

**不加载模型即可复现本文的全部「实测」**（prompting 与 typesafe 是纯 Python）：

```bash
cd /path/to/jeff-serve

# 标签表与 >26 选项的标记方案
python3 -c "
import importlib.util,sys
s=importlib.util.spec_from_file_location('P','src/jeff/prompting.py')
P=importlib.util.module_from_spec(s); sys.modules['P']=P; s.loader.exec_module(P)
print('MAX_OPTIONS', P.MAX_OPTIONS)              # 16  (SemIf 宽度 / 头默认槽位)
print('MAX_LABELS ', P.MAX_LABELS)               # 255 (选项上限 / 头槽位上限)
print('16th..28th', P.LABELS[15], P.LABELS[25], P.LABELS[26], P.LABELS[27])
print('last      ', P.LABELS[-1])                # JT
"

# 同一个请求的 row / prompt / 顺序映射 / serving rule
python3 - <<'PY'
import importlib.util, sys, json
def load(n,p):
    s=importlib.util.spec_from_file_location(n,p); m=importlib.util.module_from_spec(s)
    sys.modules[n]=m; s.loader.exec_module(m); return m
P = load("P","src/jeff/prompting.py"); T = load("T","src/jeff/typesafe.py")

state = {"ticket": "I was charged twice for order #4411", "customer_tier": "gold"}
q = {"type":"choice","instructions":"Which team should handle this ticket?",
     "criteria":{"billing":"Payments and refunds","technical":"Bugs and outages"}}
row = T.question_row_mapped(state, q, "route", "tf-true-first")
print(json.dumps(P.direct_messages(row)[1]["content"], ensure_ascii=False, indent=1))
print(T.question_options_mapped({"type":"noul","criteria":{"true":"y","false":"n"}}, "yn-no-first"))
print(T.apply_rule("noul", {"yes":0.62,"no":0.38}, {"noul_edge":0.0}))
PY
```

**跑模型前向**（需要 80 GB GPU 与约 61 GB 权重）：

```bash
pip install -r requirements.txt && pip install .
jeff-serve jgeuter/Jeff-1.0-Large \
    --revision 275325a5eab52cce4f109a09ee3832f2eb89fe2f \
    --host 127.0.0.1 --port 8013

python scripts/smoke_test.py --endpoint http://127.0.0.1:8013
```

看结果里三件事：

- `usage.output_tokens` **必须等于 0**——这一条确认「读」确实发生了（`serve.py:60` 是硬编码，若不为 0
  则说明跑的代码不是这一版）；
- `runtime.serving_rule` 是**实际生效**的 rule（`{"noul_t","noul_edge","score_t","choice_t"}`，
  缺失即恒等）；
- `runtime.cudnn_attention` 在短请求上应为 `false`（`model.py:127-133`）。

想看某条记录的逐题耗时，读 `runtime.forward_seconds` 与 `runtime.input_tokens`（两者都是**逐题累加**
的和，见 §7）。

**想验证头确实等于零样本读法**（`head.py:15-18` 那条可检验性质）：构造一个 `head=None` 的 `Decider`
（`model.py:78-82` 会建 `linear` 头并用 `slice_init_` 填入字母行），比较它与直接取 LM 的字母 logits
——两者应当逐位相同。

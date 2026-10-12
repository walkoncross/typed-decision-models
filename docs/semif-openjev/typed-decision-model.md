# SemIf-OpenJev 是如何把冻结的 Qwen 读成类型化决策模型的

> 代码库：<https://github.com/TheoLeeCJ/SemIf-OpenJev>
> 对应 README 第 25 行：*This baseline reads typed option probabilities directly from a model.
> No answer sentence, JSON repair, or decoding loop.*
> 本文按 `src/semif_phase1/` 的实际代码逐环节说明这句话的实现方式；文中出现的文件名与行号（`direct.py:30`、
> `shared.py:128`、`calibrate.py:97` 等）均相对该仓库根目录。
>
> **同名配对文档**：另一个项目 RYOTIDE 的 [`docs/ryotide/typed-decision-model.md`](../ryotide/typed-decision-model.md)。
> 两者都是「零训练、把冻结权重读成分类器」，本篇刻意与它同名以便并读对照。
> §0 先讲清分歧，§13 把它的四条不变量逐条落位到 SemIf-OpenJev。
>
> **与 `docs/open-jev/` 那篇的区别**：那个 Open-Jev 是 [Zefan-Cai/Open-Jev](https://github.com/Zefan-Cai/Open-Jev)
> （27B、LoRA + 标量决策头、候选数 255）；本篇的 SemIf-OpenJev 是另一个仓库
> （[TheoLeeCJ/SemIf-OpenJev](https://github.com/TheoLeeCJ/SemIf-OpenJev)，README 自述 *formerly OpenJev*），
> **零训练、冻结 Qwen3.5-4B、候选数 16**。两者只在历史上共用过 OpenJev 这个名字，机制上没有继承关系。

## 0. 与 RYOTIDE 的分野

两套系统都承诺**输出是一个封闭的概率分布，全程不生成 token**，而且都**不训练任何东西**。
SemIf-OpenJev 甚至比 RYOTIDE 更极端：它连「标定读位置」这一步都没有。分歧集中在四处。

| | RYOTIDE | SemIf-OpenJev |
|---|---|---|
| 决策机制 | 读**标签 token** 在末位的 logits，掩码整个词表，再 softmax | 读**字母槽位**（`A`–`P`）在末位的 logits，**按 id 取下标**，再 softmax |
| 需要训练 | 否，权重全程 frozen | 否，权重全程 frozen（`core.py:132` 只 `model.eval()`） |
| 「typed」来自 | 分词器：标签必须单 token，否则映射成 `A/B/C` 标记 | 请求里的 `options` 数组 + `LETTERS` 常量；分词器**三重校验**后才接受（`direct.py:13-22,42-44`） |
| 候选数上限 | 260（`A0`–`Z9` 两步读；>26 另有 `grouped` 近似读法） | **16**（`LETTERS = "ABCDEFGHIJKLMNOP"`，`core.py:11`；越界直接报错，`core.py:32-33`） |
| 读位置怎么定 | **运行期标定**：扫 9 前缀 × 2 表面形式，取 `marker_mass` 最高的那对 | **无标定**：固定末位。`allowed_token_mass` 只**上报**，从不参与选择（§3） |
| 候选顺序偏置 | 多 order 共享前缀 fork，折叠回真实标签后**概率平均** | **不消除**：不重排、不平均，只做扰动度量并把 10/36 的 argmax 翻转公开（§6） |
| 置信度来源 | 读出来的；T 由离线拟合 + **预先注册的接受 / 拒绝判据**决定（可能被拒，保持 1.0） | 读出来的；T **逐 workload 拟合**，带 group-disjoint 折外 ECE 与 bootstrap 区间，无条件发布（§12） |
| 诊断量 | `marker_mass` | `allowed_token_mass` + `full_vocab_argmax_id`（但只在两个后端上） |
| 多问题共享 state | `run_many()`：公共前缀预填充一次 | `shared` 模式：一次 prefill + `reorder_cache` 复制 + 只算末位 logits（20.03 vs 2.33 dec/s，§7） |
| 服务化 | FastAPI `/v1/systemone` + `/health` | **CLI**（`semif-score`）+ 只增不改的 JSONL；另有浏览器 WASM 路径 |
| 共同底线 | **`output_tokens = 0`** | Torch / MLX / llama.cpp 三条路径都成立；**浏览器路径不成立**（`max_tokens: 1`，§9） |

一句话：RYOTIDE 是**「不改权重，只换读法」**；SemIf-OpenJev 是**「同一句话，把读法工程化到三个 runtime、
一个浏览器、外加一条可审计的校准链」**——而且它对「这条读法有多脆」的记录比另外几篇都更主动。

## 1. 端到端数据流

```
            ┌────────────── cli.py:main（semif-score）──────────────┐
JSONL 输入   │ --mode {direct|serial|shared|reranker}                │
+ --model    │ --backend {torch|mlx|llamacpp}  --revision <40 位>    │
+ --revision │ ① validate_row() 逐行                        core.py:18 │
+ 输出必须新  │ ② 装后端（torch / mlx / llamacpp）          cli.py:65-87 │
            └───────────────────────┬───────────────────────────────┘
                                    v
            ┌──────────── encode_prompt()（三条后端共用）────────────┐
            │ apply_chat_template(..., enable_thinking=False) direct.py:35 │
            │ validate_row → JSON payload {evidence, criterion, options}  │
            │ _slot_ids()：A..P 必须单 token 且往返一致      direct.py:13 │
            │ 边界校验：encode(prompt+letter) == ids+[slot]  direct.py:42 │
            │ → (ids, slots, prompt_sha256)                            │
            └───────────────────────┬───────────────────────────────┘
                                    v
   ┌──────────────┬────────────────┬──────────────────┬─────────────────┐
   │ direct       │ serial         │ shared           │ reranker        │
   │ 整段前向      │ state 前缀缓存  │ 一次 prefill      │ 原生 Qwen3-     │
   │ use_cache=F  │ + 逐分支深拷贝  │ + reorder_cache  │ Reranker 模板   │
   │ logits_keep=1│（命中判 state） │ 复制 N 份分支     │ 左填充 yes/no   │
   └──────┬───────┴────────┬───────┴────────┬─────────┴────────┬────────┘
          │                │                │                  │
          v                v                v                  v
   logits[:, -1, :]  →  vocabulary[slots]  →  softmax(≤16 个数)  →  logit(yes)−logit(no)
                                                                   → softmax(跨选项)
                                    │                                     │
                                    └──────────────┬──────────────────────┘
                                                   v
                          JSONL 一行：option_logits / probabilities /
                          prompt_sha256 / allowed_token_mass / model{revision,...}
                          probability_status: "uncalibrated as decision confidence"
                                                   │
                              （离线、独立进程、numpy only，不加载模型）
                                                   v
                          benchmarks/calibrate.py → *.calibrated.jsonl
                          pred["calibration"] = {method, temperature}   calibrate.py:346
```

## 2. 「typed」从哪来：请求声明的选项 + 字母槽位 + 三重校验

标签集**不是**模型训练出来的，也不是词表里的自由 token——它是每次请求带上来的：

```json
{"id":"route-1","state":"...","question":"...",
 "options":[{"id":"account_access","description":"..."},{"id":"billing","description":"..."}]}
```

`validate_row()`（`core.py:18-40`）先把它卡死：`id`/`question` 必须是非空字符串（`core.py:22-23`），
`state` 必须是非空 string / object / array 且能被 `json.dumps(..., allow_nan=False)` 序列化（`core.py:24-30`），
`options` 必须是 **2–16** 项（`core.py:32-33`），每项要有字符串 `id` 与 `description`（`core.py:36-37`），
且 `id` 互不重复（`core.py:39-40`）。

然后 prompt 里出现的不是 `id`，而是**按显示顺序分配的字母**（`core.py:45-52`）：

```python
payload = {"evidence": row["state"], "criterion": row["question"],
           "options": [{"letter": LETTERS[index], "description": option["description"]} ...]}
```
配一条 system 消息要求 *Respond with only its uppercase letter*（`core.py:12-15`）。

字母能不能当标签，取决于分词器。`direct.py:13-22` 的 `_slot_ids()` 逐个验证三件事：

1. `tokenizer.encode(letter, add_special_tokens=False)` 长度必须是 **1**；
2. `tokenizer.decode(encoded) == letter` —— **往返必须一字不差**（`direct.py:16-18`）；
3. 所有槽位 token id 互不碰撞（`direct.py:20-21`）。

还有第四件、也是最能说明问题的一件——**边界稳定性**（`direct.py:42-44`）：

```python
for letter, token in zip(LETTERS, slots):
    if tokenizer.encode(prompt + letter, add_special_tokens=False) != ids + [token]:
        raise ValueError(f"Answer boundary changes tokenization for slot {letter}")
```

它不满足于「`A` 单独是一个 token」，而是要证明「把 `A` 接在**这个 prompt 的末尾**，它仍然恰好是那一个
token」。这正是 RYOTIDE 不变量 #2（前缀与标记表面形式耦合）在同一件事上的另一种处理：RYOTIDE 靠**扫描**
发现耦合对，SemIf-OpenJev 靠**断言**排除掉耦合的可能，断言不过就拒绝服务。

实测（committed 产物上的读数）：`results/raw/predictions/direct-authored144.jsonl` 里
`answer_token_ids` 恒为 `[32, 33, 34]`——Qwen3.5-4B 的 `A`/`B`/`C` 正是 id 32/33/34。

**同一个「typed」在浏览器路径上是另一套**（`webgpu-demo/worker.js`）：那里不查分词器，而是把
token id **硬编码**成 `labelBase + index`（`worker.js:147`），而 `labelBase` 按模型写死
（Qwen3-0.6B 与 Qwen3.5-4B 是 `32`，MiniCPM5-2B 是 `54`，`worker.js:11,16,23`）。
上限也不一样：Python 侧是 16（`core.py:11`），浏览器侧 `MAX_OPTIONS = 20`（`worker.js:28`）。
**同一个产品的两条路径，标签集与上限都不是同一个**——引用「SemIf 支持几个选项」时必须指明是哪条路径。

## 3. 读位置：固定末位，而且**没有标定**

这是 SemIf-OpenJev 与 RYOTIDE 最干脆的一处分歧。

RYOTIDE 把「读哪里」当成一个待求的变量：扫 9 个前缀 × 2 种标记表面形式，用 `marker_mass` 挑。
SemIf-OpenJev 直接把它写死——`direct.py:30` 的 `return model(**kwargs).logits[:, -1, :]`，
`serial.py:102`、`shared.py:124`、`mlx_backend.py:117`、`llamacpp_backend.py:299` 全都读末位。
全仓没有 `SUFFIXES`、没有 `PATTERNS`、没有任何位置扫描。

但它**照样算了那个诊断量**，只是用途完全不同。`serial.py:120-122`：

```python
"allowed_token_mass": float((selected_tensor.logsumexp(-1) - vocabulary.logsumexp(-1)).exp()),
"full_vocab_argmax_id": int(vocabulary.argmax()),
```

前者是「全部词表质量里，落在被选中的那几个槽位上的份额」——与 `marker_mass` 是同一个量。
后者更强：**模型在不受限时的 argmax 是哪个 token**。两个都只是**上报**，没有任何一行代码根据它们改变读法。

在 committed 的 144 行上实测（`results/raw/predictions/direct-authored144.jsonl`，144 行全是 3 选项）：

| 量 | 值 |
|---|---|
| `allowed_token_mass` 中位数 | **0.999743** |
| `allowed_token_mass` 最小 / 最大 | 0.973303 / 0.999954 |
| `allowed_token_mass < 0.99` 的行数 | **1 / 144** |
| `full_vocab_argmax_id` 落在 `answer_token_ids` 内的行数 | **144 / 144** |

也就是说：在这批数据上，自由生成的 argmax 与受限读法的 argmax **毫无分歧**，
词表里 99.97% 的质量本来就在那三个字母上。RYOTIDE 用 `marker_mass` 找「读错了」的位置
（它记录的失败案例只有约 15% 质量到达标记）；SemIf-OpenJev 这里没有可找的东西——它的失败模式不是读错位置。

**但这两个量只在两条后端上**：`serial.py:120-123` 与 `llamacpp_backend.py:285-286`。
`direct.py:64-77`、`shared.py:149-164`、`mlx_backend.py:95-106` 都**不上报**。
于是出现一处「算得出来 ≠ 接进来了」：默认的 `--mode direct`（README 的 Quick start 用的就是它）
恰恰是不带诊断量的那条路径（详见 §14）。

> `enable_thinking=False`（`direct.py:36`）在这里的作用不是标定，而是**保证末位落在该落的地方**——
> 这是 §13 里 RYOTIDE 不变量 #1 的失败模式在 SemIf-OpenJev 上的唯一防线，而它没有配套诊断量。

## 4. 一次前向：只读最后一个位置

`direct.py:25-30`：

```python
def _forward(model, inputs):
    parameters = inspect.signature(model.forward).parameters
    kwargs = dict(inputs, use_cache=False, return_dict=True)
    if "logits_to_keep" in parameters:
        kwargs["logits_to_keep"] = 1
    return model(**kwargs).logits[:, -1, :]
```

三点值得记：

- `use_cache=False`——单次前向不需要 cache（cache 是 §7 那两条路径的事）；
- `logits_to_keep=1` **是条件传的**：先反射模型签名，支持才传（`direct.py:28-29`）。这与 RYOTIDE 不变量 #3
  的动机（`[1, n_tokens, vocab]` 只为读一行）是同一件事，但这里退化成「有就用、没有就算了」的优化，
  而不是正确性要求——因为 `.logits[:, -1, :]` 无论如何都只取末位。
  `serial.py:34-40` 和 `shared.py:78-79` 则把这件事升级成了**硬要求**（缺 `logits_to_keep` 直接 `RuntimeError`），
  因为它们必须靠它避免物化整段后缀的 logits。
- 整个前向包在 `torch.inference_mode()` 里（`direct.py:60`），并且在前后各 `synchronize()` 一次
  （`core.py:90-96`，CUDA 与 MPS 分别处理），因为计时是这套代码的产出之一，不能把异步 kernel 的时间漏掉。

结果行里有两句自我标签（`direct.py:75-76`）：

```python
"readout": "native full-vocabulary last-position logits restricted to declared answer slots",
"probability_status": "conditional option score; uncalibrated as decision confidence",
```

`probability_status` 这句话值得单独表扬：**它把「这个数还不是置信度」写进了每一条产物的行内**，
而不是写在文档里等人去读。llama.cpp 版还额外说明了权重量化（`llamacpp_backend.py:291`：
*over quantized weights*）。

## 5. 「掩码」其实是取下标：softmax 只作用在 ≤16 个数上

RYOTIDE 的做法是**掩码整张词表**（把非候选位置置 `-inf`）再 softmax；SemIf-OpenJev 的做法是
**先取下标，再 softmax**（`direct.py:63-67`）：

```python
vocabulary = _forward(model, inputs)[0].float()
selected = vocabulary[slots].cpu().tolist()
...
"probabilities": softmax(selected),
```

`softmax` 在 `core.py:59-65`，是纯 Python 实现：要求至少两个有限值（`core.py:60-61`），
先减最大值再取指数（`core.py:62-64`）。

这两条路**数学上等价**——对全词表做 `-inf` 掩码后的 softmax，与对取出的那 16 个数直接 softmax，
结果逐位相同（softmax 对常数平移不变，其余项的指数为 0）。差别在于：

- SemIf-OpenJev **没有物化掩码张量**，因此也没有「掩码写错了但看起来正常」这个失效模式；
- 代价是**它没有强制 `slots` 覆盖全部候选**这一步的显式记录。RYOTIDE 的 `marker_mass` 正是用来
  检查「掩码是否盖在了模型真正想看的地方」；SemIf-OpenJev 只在两条后端上事后补了这个数（§3）。

实测等价性：在 committed 的 144 行上，`probabilities` 与 `softmax(option_logits)` 的最大逐元素差
为 **9.73e-08**——量级正好是 `.float()`（`direct.py:61`）把 bf16 抬到 fp32 后剩下的舍入，
不是公式差异。

> 归一化被丢弃的那部分质量，正是 `1 − allowed_token_mass`；`allow_token_mass` 最小值那一行
> （0.973303）意味着有约 2.7% 的质量落在 16 个字母之外。这仍然是一个**条件分布**，
> 不是「模型认为这件事的概率」。

## 6. 顺序偏置：不消除，只度量，并把翻转数公开

字母是**按显示顺序**分配的（`core.py:45-52`），所以在这套实现里「选项顺序」与「字母」是同一个变量——
换一下显示顺序，`A/B/C` 也跟着换。任何字母偏置都会直接变成选项偏置。

SemIf-OpenJev 不修它。它做的事是**测**：`docs/METHOD.md` 记录了一批 output-blind 的扰动，
其中第一种就是 *reverse the displayed option order while preserving semantic IDs*，
比较时按语义 `option_id` 对齐概率。36 行上的实测（`results/raw/perturbation-comparison.json`）：

| 变体 | argmax 翻转 | 最大概率位移（平均 / 最大） | 平均 family 平衡准确率 |
|---|---|---|---|
| 原样（36 行子集） | — | — | 0.7228 |
| **选项顺序反转** | **10 / 36** | **0.2648 / 0.8695** | 0.8132 |
| 判据换一种等价说法 | 9 / 36 | 0.2150 / 0.8654 | 0.7058 |
| 追加无关上下文 | 4 / 36 | 0.1034 / 0.8347 | 0.8206 |

**反转选项顺序会翻转 10/36 的 argmax，平均最大概率位移 0.2648。** 这不是个小数字，
它说明「同一道题、同样的三句话，只是换个显示顺序」就有 28% 的决定会变。
（平衡准确率不是单调变差的——反转那一列还更高，所以这不是「扰动都让结果变坏」，
而是「结果本来就不稳」。）

三条路线对同一个问题的三种答法可以并读：RYOTIDE 在**推理时**fork 多个顺序再平均；
Laya 在**训练时**用 `parallel` 布局让位置对选项不可见；SemIf-OpenJev 则**承认它、量化它、不修它**，
把 10/36 写进 committed 的 JSON 里。第三种做法的价值在于：它没有把偏差藏起来，
任何一个声称「SemIf 的顺序偏置是 0」的说法都可以被这个文件立刻否证。

## 7. 多问题共享 state：前缀提取 + 后缀布局 + cache 复制

这是 SemIf-OpenJev 工程量最大的一节，也是它相对 RYOTIDE `run_many()` 多出来的东西。

### 7.1 要求：state 必须逐字节相同

`shared.py:58-59` 第一句就是 `if not rows or any(row["state"] != rows[0]["state"] for row in rows[1:])`。
共享的是**完全相等的 state**，不是「相似」的 state。这条约束决定了前缀切分必须精确到 token。

### 7.2 前缀怎么切：用占位行反推出 evidence 的边界

`shared.py:14-37`（`serial.py:15-31` 是同一份逻辑）：

```python
payload = turns[-1]["content"]                      # user 消息 = JSON 字符串
if prompt.count(payload) != 1:
    raise ValueError("Cannot locate the unmodified evidence payload in the chat template")
evidence = json.dumps({"evidence": state}, ensure_ascii=False)[:-1]
if not payload.startswith(evidence):
    raise ValueError("Evidence serialization changed")
text = prompt[: prompt.index(payload)] + evidence
return tokenizer.encode(text, add_special_tokens=False)[:-1]
```

两个值得注意的动作：

1. `evidence = json.dumps({"evidence": state})[:-1]` —— 砍掉收尾的 `}`，让前缀停在 `{"evidence": ...` 之后；
2. `encode(text)[:-1]` —— **再砍掉最后一个 token**。注释给的理由是
   *Appending JSON punctuation can merge with the final boundary token*（`shared.py:36`）：
   文本层面的 `{"evidence":"..."` 末尾那个字符，和后面真正会接上的 JSON 标点合起来，
   在 BPE 里可能是一个**不同的 token**。与其在文本层猜，不如在 token 层多丢一个，让后缀从一个安全位置开始。

这两步都没有「差不多就行」的余地：`shared.py:65-66` 会逐行验证**每一行完整 prompt 的前缀**
恰好等于这个 prefix，不等就报错。所以前缀提取要么精确，要么 fail loud。

### 7.3 后缀怎么排：右填充 + 显式 position_ids + 记录每条的真实末位

`_suffix_layout()`（`shared.py:40-51`）：

```python
padding = width - len(sequence)
ids.append(sequence + [pad_id] * padding)
masks.append([1] * (prefix_length + len(sequence)) + [0] * padding)
positions.append(list(range(prefix_length, prefix_length + len(sequence))) + [0] * padding)
ends.append(len(sequence) - 1)
```

- **右填充**（`shared.py:47`），padding 的 `attention_mask` 为 0（`shared.py:48`）；
- **显式传 `position_ids`**（`shared.py:49`），因为 padding 位置的默认位置会污染（这里给 0）；
- 关键：**记住每条序列自己的末位**（`ends`，`shared.py:50`）。批内宽度不同，读的位置就不是 `-1`。

### 7.4 cache 怎么复制：`reorder_cache` 传全零

`shared.py:84-98` 先预填充前缀并保留 `past_key_values`，然后三重守卫：

```python
if cache is None or cache.get_seq_length() != len(prefix):
    raise RuntimeError("Invalid native prefix cache")
if not callable(getattr(cache, "reorder_cache", None)):
    raise RuntimeError("Native cache does not support duplicate branch selection")
```

第二句是本篇最值钱的一处守卫。复制分支在 HF 里的标准做法是 `reorder_cache(beam_idx)`；
传入全零索引，就是**把第 0 行复制 N 份**（`shared.py:128`）：

```python
cache.reorder_cache(torch.zeros(len(rows), dtype=torch.long, device=device))
```

这比手写 `expand` / `batch_repeat_interleave`（`docs/open-jev/` 那篇记录的另一个 Open-Jev
正是在这里踩了递推状态**别名**的坑）更省事，但那句守卫也承认了它的边界：
**cache 必须支持 `reorder_cache`**。对纯注意力 cache 这是浅拷贝列表；
对带递推状态的混合架构 cache，`reorder_cache` 复制的就是状态本身。

- **CUDA**：批量路径，复制后一次前向算完所有带 padding 的后缀，并且只算需要的末位 logits：
  `logits_to_keep=torch.tensor(selected_positions, ...)`（`shared.py:138`），
  再按 `output.logits[index, selected_positions.index(ends[index]), :]` 取回（`shared.py:142-144`）。
- **MPS**：逐条 batch-1，每条 `copy.deepcopy(cache)`（`shared.py:110`），
  注释写明理由——*MPS is faster with independent batch-1 suffixes; CUDA stays batched*（`shared.py:99`）。
  两种执行方式的 `serving_config` 不同（`shared.py:158-160`），结果行里能分辨是哪一种跑出来的。

### 7.5 代价：速度是真的，数值漂移也是真的

`results/raw/shape777-direct.json`（37 state × 21 判据 = 777 次决定，一张 RTX 3090）：

| 模式 | decisions/s | 777 次的决定总耗时 | 峰值显存 |
|---|---:|---:|---:|
| `fresh_batch1` | 2.333 | 333.06 s | 8.89 GB |
| `serial_prefix` | 10.752 | 72.26 s | 8.99 GB |
| `parallel_suffix` | **20.030** | **38.79 s** | 11.61 GB |

8.6 倍。但同一个文件里的 `comparisons_to_fresh` 把代价也记了：

| 对比 | 最大概率差 | argmax 翻转 |
|---|---:|---:|
| serial vs fresh | 0.09339 | **5 / 777** |
| parallel vs fresh | 0.06242 | **6 / 777** |

并且 `comparisons_to_fresh.note` 明说 *Tolerance is deliberately non-gating; report exact probability drift
and all argmax flips.* 也就是：**这是一个 knowingly 有损的加速**，README 也照写
（*BF16 execution changed 5–6 of 777 argmaxes relative to fresh scoring*）。
parallel 路径还多算了 62937 − 59940 = 2997 个纯 padding token（`shape777-direct.json`）。
引用这两个速度数字时，5–6 次 argmax 漂移必须一起引——否则就变成了「同一件事快 8.6 倍」。

### 7.6 其余两条后端怎么做同一件事

- **MLX**（`mlx_backend.py:215-217`）：不用 `reorder_cache`，而是
  `entry.merge([entry] * len(rows))` 复制，再 `entry.prepare(lengths=lengths, right_padding=[...])`
  显式告诉每条分支它的真实长度与右填充量——因为 MLX 的递推 cache 需要知道 padding 在哪。
  读取位置同样是每条自己的 `lengths[i] - 1`（`mlx_backend.py:223`）。
- **llama.cpp**（`llamacpp_backend.py:143-158,380-391`）：没有可复制对象，只能
  `save_state()` 把整段序列化，每行 `restore_state()` 回来再跑分支；
  `branch_state_bytes` 直接写进结果（`llamacpp_backend.py:351,401`）。

三种复制策略、三条 read-position 规则，指向同一件事：**前缀复用的写法是被后端绑定的**，
不是可以照抄的算法。

## 8. 第二条读法：原生 reranker 的 yes/no log-odds

同一个仓库里还有一条**完全不同**的读法（`src/semif_phase1/reranker.py`），
用的是 Qwen3-Reranker-4B 而不是 causal LM。它把「N 选一」改造成 **N 个 pointwise 是非题**：

- prompt **不走 chat template**，而是把 Qwen3-Reranker 的官方模板硬编码成字符串（`reranker.py:10-15`），
  其中 `<think>\n\n</think>\n\n` 是手写在 `SUFFIX` 里的——这是 `enable_thinking=False` 在本仓库的**第二种接法**；
- 指令按 `row["provenance"]["experiment"]` 在「决策」与「检索」两条之间切换（`reranker.py:27-28`）；
- `_answer_ids()` 要求 `yes`/`no` 各是单 token，**并且** `convert_tokens_to_ids` 的结果与
  *official yes/no token contract* 一致（`reranker.py:41-48`）——又是一处「不信分词器，要往返验证」；
- **左填充**（`reranker.py:64-67`）后读 `logits[:, -1, :]`（`reranker.py:77`）。

最后一点让这个仓库同时存在**两个填充方向**：reranker 左填充读 `-1`，`shared` 右填充读记录下来的末位。
两者都精确，但依赖的性质不同（左填充靠 mask 屏蔽，右填充靠因果性）——与 `docs/comparison.md` §4.2
里 RYOTIDE 与 Quyet 的对照是同一类差异。

打分与归一化（`reranker.py:82-83,113`）：

```python
odds = selected[:, 1] - selected[:, 0]        # logit(yes) − logit(no)
relevance = selected.softmax(-1)[:, 1]        # 单题内部的 yes 概率
...
"probabilities": softmax(log_odds),           # 跨选项归一化
```

`docs/METHOD.md` 明确划了界：*That last normalization is our comparison rule;
it is not part of the upstream reranker's calibration contract.*
也就是说跨选项那层 softmax 是项目自己加的，不是 reranker 的性质。

结果（`results/phase1-summary.json`，144 / 256 / 102 行）：

| 工作负载 | direct logits（4B BF16） | 原生 reranker（4B） |
|---|---:|---:|
| authored 144，family 平均平衡准确率 | **0.8132** | 0.6251 |
| WANLI 256，平衡准确率 | **0.6365** | 0.5220 |
| TypeSafe 102，等 case 众数一致率 | **0.8453** | 0.5595 |

配对 bootstrap 的差值区间（`phase1-summary.json` 的 `paired_reranker_minus_direct`）：
authored −0.1882 [−0.2557, −0.1200]、WANLI −0.1145 [−0.1844, −0.0444]，都不跨 0。
项目的结论也写在 summary 里：*a native reranker remains a retrieval control and is not a drop-in
universal decision operator* —— reranker 只在检索排序上追平（code RAG Recall@1 两边都是 1.0，
company knowledge 都是 0.9286）。**「同样是 pointwise 是非题」，在别处是主路径，在这里是被证否的对照臂。**

## 9. 浏览器里的一次「生成」：语法约束 + 强制 1 token

`webgpu-demo/worker.js` 跑的是 wllama + 量化 GGUF，读法是这套代码里唯一**会生成 token** 的：

```js
const grammar = `root ::= ${labels.map((label) => `"${label}"`).join(" | ")}`;   // worker.js:138
...
max_tokens: 1, temperature: 1, top_k: 0, top_p: 1,
logprobs: true, top_logprobs: 20,
logit_bias: Object.fromEntries(labels.map((label, index) =>
  [String(MODELS[modelId].labelBase + index), 100])),                            // worker.js:141-147
grammar,
```

四件事同时发生：

1. **GBNF 语法**把可生成集合限成 `"A" | "B" | ...`，这是浏览器侧对「掩码」的替代品；
2. **`logit_bias` +100** 把 16 个标签 token 一起抬起来，保证它们挤进 `top_logprobs: 20`；
3. **`max_tokens: 1`** —— 它确实生成了一个 token。RYOTIDE 不变量里的 `output_tokens = 0`
   **在这条路径上不成立**；
4. 但读出来的不是那个被采样的 token，而是 `top_logprobs` 里**每个标签的 logprob**，
   再自己 softmax（`worker.js:57-64,152-154`）。

第 3 点与第 4 点合起来是这套做法的巧处：用「生成一个 token」换来了「一个能返回 logprobs 的接口」，
而采样的随机性被绕开了——`temperature: 1, top_k: 0, top_p: 1` 刻意不退化（`worker.js:142-144`），
因为反正不读采样结果。副作用是：**同一件事，服务端读末位 logits（0 token），浏览器端走一次
1-token 的解码循环**。如果拿浏览器路径的耗时去和 §4 的 `forward_seconds` 比，比的是两件事。

同页并排跑的还有生成基线（`worker.js:165-199`）：要求模型输出一个「键 = `<label>: <完整选项文本>`」的
JSON 对象，然后 `validateGeneration()`（`worker.js:70-95`）逐项校验——键必须完全匹配、
每个值必须是 0–1 的有限数、总和与 1 的偏差不超过 0.02，失败则返回 `{valid: false, validationError}`。
这正是 README 那句 *No answer sentence, JSON repair, or decoding loop* 的对照组。

另外两处细节：`n_ctx: 2048`、`cache_prompt: false`、`n_gpu_layers: 999`（`worker.js:111-114`）；
加载失败时会 `engine.exit()` 清掉引擎再报错（`worker.js:217-221`），避免一个坏掉的引擎留在页面上
继续接请求。`enable_thinking: false` 在这里是**第三种接法**——通过 wllama 的
`chat_template_kwargs`（`worker.js:129,150,177`）。

## 10. 三个后端：槽位相同，数值不同

`--backend`（`cli.py:19`）在 torch / mlx / llamacpp 之间切（`cli.py:65-87`）。
三者共享的是**同一个 prompt 与同一组槽位**：

- MLX 直接 import `direct.encode_prompt`（`mlx_backend.py:19`），
- llama.cpp 也用同一个 `encode_prompt`（`llamacpp_backend.py:186`），**并且**再把渲染出的文本
  扔进 GGUF 词表重新分词，**逐 token 比对**（`llamacpp_backend.py:187-188`），失败即报错。
  加载期还额外验证「GGUF 词表与参考分词器一致」以及「16 个字母是共享单 token」
  （`llamacpp_backend.py:192-207`）。

所以 `prompt_sha256` 能跨后端逐行对齐——这也是 README 敢说 *Prompt construction stays on the pinned
reference tokenizer, so `prompt_sha256` matches the Torch backend row for row* 的原因。

但**数值不保证**。README 自己写明：*Direct and prefix-cached execution can have small numerical
differences from different llama.cpp evaluation paths; compare decisions or probabilities with a
tolerance rather than raw logits bit for bit.* 加上 §7.5 那 5–6/777 的漂移，引用规则应当是：

> **`prompt_sha256` 可以逐行比，`option_logits` 不可以。** 概率要带容差，决定要带翻转计数。

三条后端各自的元数据也带着自己的指纹：torch 带 `torch_version` / `transformers_version`（`core.py:138-139`），
MLX 带 `mlx_version` / `mlx_lm_version` / **每个源文件的 sha256**（`mlx_backend.py:84-90`），
llama.cpp 带 GGUF 校验和。MLX 还有两条硬门槛：必须 macOS + arm64（`mlx_backend.py:34-35`），
且 `config.model_type` 必须是 `qwen3_5`、不允许自定义模型代码（`mlx_backend.py:64-65`）。

## 11. 服务化 / fail-loud 风格

SemIf-OpenJev **不是 HTTP 服务**——入口是 CLI（`pyproject.toml:24-25` 的 `semif-score = "semif_phase1.cli:main"`），
产物是 JSONL。所以「服务化」这一节的对照点不在并发与健康检查，而在**产物不可变**与**加载完整性**。

**只增不改**：输出文件必须不存在才能跑（`cli.py:36-37`），并且用 `open("x")` 打开（`cli.py:89`），
`calibrate.py` 的 `apply()` 与报告写出也同样 `open("x")`（`calibrate.py:334,351`）。
配合 committed 的 `results/raw/SHA256SUMS`，整个证据链是「一次跑一条、事后可校验」的，
不存在「覆盖了上次的结果」这个状态。

**参数组合的早退**：`cli.py:38-58` 逐个 `parser.error()`——`--mlx-bits` 只能在 mlx 上用、
`--gguf` 只能在 llamacpp 上用、MLX 与 llama.cpp 都不支持 `reranker`、
`--max-tokens` 必须为正。**没有一个参数会被静默忽略。**

**加载完整性**：`load_causal_model()`（`core.py:99-141`）里三条：

```python
if not local and not re.fullmatch(r"[0-9a-f]{40}", revision or ""):
    raise ValueError("Remote models require a pinned 40-character commit revision")   # core.py:105-106
...
device_map={"": str(target)}, low_cpu_mem_usage=True, output_loading_info=True        # core.py:126-127
...
if any(loading.get(key) for key in ("missing_keys", "mismatched_keys", "error_msgs")):
    raise RuntimeError(f"Checkpoint did not load completely: {loading}")             # core.py:130-131
```

第一句禁止不带 revision 的远程加载；第三句是**加载后自检**——这正是 `docs/comparison.md` §4.5
里另一个 Open-Jev 反例（`device_map="auto"` 让模块落到 disk、决策头留在 `meta`，服务照样报 ready）
的对症写法：`device_map` 只有一个目标（`core.py:125`），并且有 `output_loading_info` 兜底。
另外 `trust_remote_code=False` 在三条后端上都硬关（`core.py:112`、`mlx_backend.py:76`）。

**行级校验**：CLI 在跑之前先把每一行过一遍 `validate_row()`（`cli.py:62-63`）；
`calibrate.py` 在写输出之前先把整份输入读进内存校验完（`calibrate.py:349-351` 的注释：
*A malformed later row must not leave behind a plausible-looking partial predictions file*）。
**先全量校验、再创建输出**，比逐行写更保守——这与 RYOTIDE 那种「一个坏问题泄漏进下一个」的顾虑同源。

> 代价要一起说：SemIf-OpenJev 没有 `/health`、没有 prompt_hash 的服务侧暴露面，
> 所以「服务说自己 ready」这一整类坑在它这里无从谈起——**不是修好了，是没有这个面**。

## 12. 置信度从哪来，又怎么被改写

### 12.1 链条

**来源是「读出来的」**——零训练、零新增参数，分布就是那次前向的 softmax 结果。
所以它能做到 RYOTIDE 那样一件事：**温度在数学上不可能改 argmax**。`calibrate.py:1-8` 把这条写进模块 docstring：

> *Softmax(logits / T) is monotone, so the argmax -- and therefore every committed accuracy -- is
> unchanged; only confidence moves. numpy only; runs offline on CPU, no model load.*

发布路径上改数值的环节只有两处，而且**分属两个不同文件**：

| 环节 | 在哪 | 改 argmax？ |
|---|---|---|
| ① 原生 `softmax(option_logits)` | `direct.py:67` / `serial.py:108` / `shared.py:153` 等 | 否 |
| ② 温度重写 `softmax(logits / T)` | `calibrate.py:345` | 否（单调） |

与另外四条路线不同，**这里没有「训练本身」这一环**，所以 `docs/comparison.md` §4.4 那条
「读出来的 vs 训出来的」分界上，SemIf-OpenJev 与 RYOTIDE 站在同一边。

### 12.2 改写时留下了账

`calibrate.py:345-346` 在重写概率的同一行写了它的来源：

```python
pred["probabilities"] = softmax(logits / temperature).tolist()
pred["calibration"] = {"method": "temperature_scaling", "temperature": temperature}
```

而且**原文件不动**——校准产物是另开的 `results/raw/calibration/*.calibrated.jsonl`。
两步配合起来，「这个概率被动过」与「被哪个 T 动过」都**不用回文档查**：前者看路径，
后者读那一行自己的 `calibration` 字段。这比只在文档里交代 T 更硬——
拿到一行数据就能回答「这个 0.81 是 T 之后的哪一个 T」。

### 12.3 怎么拟合、怎么评估（都是可以离线跑的）

`fit_temperature()`（`calibrate.py:97-112`）在 `T ∈ [0.05, 20.0]` 上用**黄金分割搜索**最小化
平均 NLL，60 次迭代；`mean_nll` 就是 `softmax(logits/T)` 在真实标签上的负对数似然（`calibrate.py:89-94`）。
`check_temperature()`（`calibrate.py:76-80`）守着「有限且 > 0」，理由写在 docstring 里：
*otherwise dividing logits can flip the argmax*。

评估用的是三件保守的做法：

- **group-disjoint 5 折**（`calibrate.py:176-198`）：按 `group_id` 而不是按行切分，
  因为语义等价的变体共享同一个 group——按行切会让变体泄漏进训练折。折分配本身也随报告一起提交
  （`fold_map` 写进 `cv.fold_groups`，`calibrate.py:223`），可以独立复算而不是只信种子；
- **ECE 用 10 个等宽箱**（`calibrate.py:134-156`，与 `benchmarks/evaluate.py:135` 一致），
  口径与 RYOTIDE 的 10 箱相同，与另外两个实现的 15 箱**不可互比**；
- **95% 区间按 source group 重采样**（`calibrate.py:159-173`，seed 217），
  而不是按行 bootstrap。

### 12.4 三个数字与一个负对照

`results/raw/calibration/{authored144,wanli256,every154}.json`：

| 工作负载 | 行数 | 发布 T | ECE（T=1） | ECE（折外 T） | 区间分离？ |
|---|---:|---:|---:|---:|:--:|
| authored 144 | 144 | **1.2342236287881765** | 0.06777 | 0.03841 | 否 |
| WANLI 256 | 256 | **2.499030870405456** | 0.20841 | 0.06909 | **是** |
| Every 154 | 154 | **1.7086356884405622** | 0.05028 | 0.04697 | 否 |

负对照（`calibrate.py:269-327`：用池化温度 `1.9647556181545844`（554 行）对每一列也能拟合，
`pooled_temperature_significantly_worse` 三个全是 `false`）——也就是说这批数据**没有证明**
逐 workload 拟合比一个池化标量更好。项目自己把它记成对照而不是卖点。

caveats 三条也写在报告里（`calibrate.py:232-236`）：ECE 在这个 n 上方差很大、要看 bootstrap 区间；
单标量只能纠正整体过度 / 不足自信，纠正不了族内的形状错误；`shipped_temperature` 是**在全量标签上**
拟合的，折外 ECE 才是泛化的诚实估计。

### 12.5 发布值的标签

每一条原生结果都带 `probability_status: "conditional option score; uncalibrated as decision confidence"`
（`direct.py:76`），llama.cpp 版还说明是在量化权重上（`llamacpp_backend.py:291`），
reranker 版说的是 `relative option compatibility; uncalibrated as categorical probability`（`reranker.py:125`）。
**「这不是置信度」被写在数据里**，而不是写在 README 里等人去读——这是本篇最值得其他实现抄的一条约定。

## 13. 四条不变量的落位

| RYOTIDE §13 不变量 | SemIf-OpenJev 的落位 |
|---|---|
| **#1** 读位置必须落在标记上（诊断量 `marker_mass`） | **成立，且诊断量存在——但只在两条后端上**。`allowed_token_mass` + `full_vocab_argmax_id` 在 `serial.py:120-123` 与 `llamacpp_backend.py:285-286`；`direct` / `shared` / `mlx` 不上报（§3）。含义也与 RYOTIDE 不同：它**不参与选位置**，位置永远是末位。实测 144/144 的自由 argmax 与受限 argmax 一致、质量中位数 0.999743 |
| **#2** 前缀与标记表面形式耦合，必须成对扫 | **改写为「断言掉这个自由度」**。不扫，而是断言 `encode(prompt + letter) == ids + [slot]`（`direct.py:42-44`）；llama.cpp 再加一层 GGUF 逐 token 比对（`llamacpp_backend.py:187-188`）。代价是不能像 RYOTIDE 那样通过换标记表面形式救回一个读错的位置——不过它本来也不扫位置 |
| **#3** 只读最后一个位置 | **成立**，且 `logits_to_keep=1` 在 direct 里是可选优化（`direct.py:28-29`）、在 serial / shared 里是硬要求（`serial.py:38-39`、`shared.py:78-79`）。**例外**：`shared` 模式读的是「**每条序列自己的末位**」而不是批的 `-1`（`shared.py:50,142-144`）——批内有 padding 时这两者不同；MLX 同理（`mlx_backend.py:223`） |
| **#4** 比较概率 / ECE 前先确认 temperature | **成立，且比别家更容易确认**：T 逐 workload 不同（1.2342 / 2.4990 / 1.7086），但改写过的每一行都自带 `calibration: {method, temperature}`（`calibrate.py:346`），原文件与校准文件是两个路径。**引用前先看文件名里有没有 `.calibrated`** |
| （新增）**`output_tokens = 0`** | **Torch / MLX / llama.cpp 成立；浏览器路径不成立**：`worker.js:141` 是 `max_tokens: 1`，它走了一次 1-token 解码再读 `top_logprobs`（§9）。这是本篇唯一一处不满足共同底线的地方，而且它换来的是「浏览器里能拿到 logprobs」 |
| （新增）**发布的产物必须与发布的代码同版** | **不成立，且可证**：`results/raw/predictions/direct-authored144.jsonl` 每行带 `cache_class: "DynamicCache"` 与 `encode_seconds` 两个字段，**HEAD 的 `serial.py` 都不产出**；`model` 里另有 `adapter` / `adapter_revision` / `adapter_sha256` 三个槽位（实测取值全是 `null`），**HEAD 的 `load_causal_model()` 根本没有 adapter 支持**（`core.py:132-141`）。所以这份 committed 产物**无法用 HEAD 的代码逐字段复现**——它记录的是更早的一次运行。文件名里的 `direct` 也是误导：它的 `serving_config` 实测是 `native-state-prefix-cache-v1`，即 `serial` 路径（`serial.py:49`），`cache_hit` 分布为 False 108 / True 36 |

## 14. 文件对照

| 文件 | 行数 | 职责 |
|---|---:|---|
| `src/semif_phase1/core.py` | 141 | 输入校验（`validate_row`）、prompt 消息构造、纯 Python `softmax`、设备解析、模型加载与完整性自检 |
| `src/semif_phase1/direct.py` | 77 | 单条前向读末位 logits；`_slot_ids` 的三重槽位校验与边界校验；`PROMPT_VERSION` |
| `src/semif_phase1/serial.py` | 132 | 同一 state 前缀缓存 + `deepcopy` 分支；**唯一产出 `allowed_token_mass` 的 torch 路径** |
| `src/semif_phase1/shared.py` | 185 | 一次 prefill + `reorder_cache` 复制 + 右填充后缀布局 + 只算指定末位 logits |
| `src/semif_phase1/reranker.py` | 126 | 第二条读法：Qwen3-Reranker 硬编码模板、左填充、yes/no log-odds、跨选项 softmax |
| `src/semif_phase1/mlx_backend.py` | 237 | Apple Silicon 路径；`entry.merge` / `prepare` 复制递推 cache；源文件逐个 sha256 |
| `src/semif_phase1/llamacpp_backend.py` | 405 | CPU 路径；GGUF 词表逐 token 比对；状态序列化 / 恢复做分支 |
| `src/semif_phase1/cli.py` | 107 | 四个 mode × 三个 backend 的 CLI；只增不改的输出；参数组合早退 |
| `benchmarks/calibrate.py` | 416 | 独立的校准层：黄金分割拟合 T、group-disjoint 折外 ECE、bootstrap 区间、池化负对照、`apply()` 写 `calibration` 字段 |
| `benchmarks/evaluate.py` | 285 | 准确率 / 平衡准确率 / macro F1 / 10 箱可靠性图 / risk-coverage |
| `benchmarks/shape777.py` | 109 | 37×21 共享 state 的三种执行方式计时；产出 §7.5 的漂移对比 |
| `webgpu-demo/worker.js` | 225 | 浏览器侧：GBNF 语法 + `logit_bias` + `max_tokens: 1` + 读 `top_logprobs`；并排的生成基线 |
| `docs/METHOD.md` / `docs/CALIBRATION.md` / `docs/RESULTS.md` | — | 上游自己的方法、校准、结果说明。**本文只当线索用，所有结论回到代码核实** |

## 15. 亲手跑一遍

### 15.1 不需要权重、本机可跑的部分（本文的实测来源）

```bash
# 1. 校准层：numpy only，不加载模型。读 committed 的 option_logits 拟合 T
python benchmarks/calibrate.py \
  --gold benchmarks/data/authored144.jsonl \
  --predictions results/raw/predictions/direct-authored144.jsonl
# 实测输出（本机 /usr/bin/python3 + numpy 2.0.2）：
# {"rows": 144, "shipped_temperature": 1.2342236287881765,
#  "ece_uncalibrated": {"value": 0.06777040669529547, ...},
#  "ece_calibrated_out_of_fold": {"value": 0.03841296293519653, ...}}
# —— 与 README 第 164 行的 1.23 / 0.068 / 0.038 对得上

# 2. 自检：断言 T 不改 argmax，且折外 ECE 变好（calibrate.py:356-366）
python benchmarks/calibrate.py --help   # demo() 需手动调用；见文件末

# 3. 提交产物的完整性（实测 23 项全部 OK）
(cd results/raw && sha256sum -c SHA256SUMS)
```

本文 §3 / §5 / §6 / §12 里那些「实测」的数字，全部来自上面这类**在 committed 产物上重算**的动作
（逐行读 `option_logits` / `probabilities` / `allowed_token_mass` / 扰动报告）。
它们不需要权重，因此是可复现的。

**不能这样跑的两个 workload**：`results/raw/calibration/workloads.json` 指向的
`build/gold-wanli256.jsonl` 与 `build/every/gold154.jsonl` **不在仓库里**——它们要靠
`benchmarks/build_wanli.py` 之类从外部数据源重新构造（WANLI 需从 HF 拉取并 pin 到
`61c95318fd71c55b6ba355d76253254615f387ec`）。所以 WANLI 那一列的 ECE 只能引用
`results/raw/calibration/wanli256.json` 的已记录值，**不能在本机重算**。

### 15.2 需要权重 / GPU 的部分（本文一律不给数值）

```bash
python -m venv .venv && . .venv/bin/activate
pip install -e '.[test]'                 # torch==2.10.0 / transformers==5.17.0（pyproject.toml:13-14）

CUDA_VISIBLE_DEVICES=0 semif-score \
  --mode direct --model Qwen/Qwen3.5-4B \
  --revision 851bf6e806efd8d0a36b00bdd55e13ccb7b8cd0a \
  --input examples/decisions.jsonl --output results.jsonl

# 三条路径的对照
CUDA_VISIBLE_DEVICES=0 semif-score --mode serial  ...   # 前缀缓存，state 相同的行会命中
CUDA_VISIBLE_DEVICES=0 semif-score --mode shared  ...   # 要求所有行 state 完全相同（shared.py:58）
CUDA_VISIBLE_DEVICES=0 semif-score --mode reranker ...  # 需要 CUDA（cli.py:82-86），走 Qwen3-Reranker-4B

pytest -q                                 # tests/ 978 行
python benchmarks/verify_published.py
```

**本文不给出任何前向产生的概率、准确率或 token 数**，理由与 `docs/jeff-serve/` 那篇相同：
本机既没有下载权重，也没有可用的 CUDA。文中出现的每一个前向数字都标了来源
（`results/raw/*.json` 或 README 的具体行），可以按路径回源核对。

## 16. 实例走查

### 16.1 一条记录走完整条链，包括顺序反转把它救回来的那一下

取 `benchmarks/data/authored144.jsonl` 里 id `a3f18f3a63d45345942b` 那条
（family `evidence_interpretation`，group `2fdaa8a61e6e989dd866`）：

- **state**：*The optician ordered replacement lenses. The workshop confirms they have not yet been
  fitted to the customer's glasses.*
- **question**：*Assess the claim: the replacement lenses have been fitted.*
- **options**：`supported` / `insufficient` / `contradicted` → 字母 `A` / `B` / `C`
- **gold**：`contradicted`（`label = 2`）

committed 行（`results/raw/predictions/direct-authored144.jsonl` 第 1 行）：

| 字段 | 值 |
|---|---|
| `input_tokens` | 142 |
| `prefix_tokens` | 62 |
| `answer_token_ids` | `[32, 33, 34]`（= `A`/`B`/`C`） |
| `option_logits` | `[22.0, 26.375, 24.375]` |
| `probabilities` | `[0.010966, 0.871138, 0.117896]` |
| `allowed_token_mass` | `0.999802` |
| `full_vocab_argmax_id` | `33`（= `B`） |
| `cache_hit` | `false`（首行，无命中） |
| `serving_config` | `native-state-prefix-cache-v1`（**serial 路径**，不是 direct） |

一步步读：

1. **取值**：`vocabulary[slots]` 取出 `B` 槽位上的 26.375 等三个数（`direct.py:63`）。
2. **softmax**：22.0 / 26.375 / 24.375 之间差 4.375，softmax 之后 `B` 拿到 0.8711（`core.py:59-65`）。
3. **确诊**：argmax 是 `B` = `insufficient`，但 gold 是 `contradicted`。**它在 87% 的自信下答错了。**
4. **诊断量说话**：`allowed_token_mass = 0.999802`——99.98% 的词表质量就在这三个字母上；
   `full_vocab_argmax_id = 33` = `B`——**就算放开整张词表，它还是选 `B`**。
   所以这不是「掩码盖错了地方」，是模型在这道题上真的这么认为。RYOTIDE 的 `marker_mass`
   在这里会是满值，什么也诊断不出来；故障在别处。
5. **温度**：T = 1.2342236287881765 → `softmax(logits / T)` 给出
   `[0.023540, 0.815205, 0.161255]`（`results/raw/calibration/direct-authored144.calibrated.jsonl` 第 1 行，
   同行的 `calibration` 字段写着 T 是多少）。**argmax 没变，置信度从 0.8711 降到 0.8152。**
   §12.4 的实测：这个 T 在全部 144 行上翻转 0 个 argmax，平均 top-1 从 0.8628 降到 0.8333。
6. **顺序反转**：`results/raw/perturbation-comparison.json` 的 `option_reversal.flip_rows`
   第一条，`base_id` 正是 `a3f18f3a63d45345942b`，`from: "insufficient"`、`to: "contradicted"`。
   也就是说：**同一道题、同样三句选项，只是把显示顺序反过来，它就从错变对了。**
   这不是答对，这是不稳——而这一条正好说明了 §6 那 10/36 是什么意思。

### 16.2 同一批判据在三种执行方式下

`results/raw/shape777-direct.json`（37 state × 21 判据 = 777 次，RTX 3090，warm）：

| 模式 | decisions/s | 总耗时 | 峰值显存 | 相对 fresh 的 argmax 翻转 |
|---|---:|---:|---:|---:|
| fresh_batch1 | 2.333 | 333.06 s | 8.89 GB | — |
| serial_prefix | 10.752 | 72.26 s | 8.99 GB | 5 / 777 |
| parallel_suffix | 20.030 | 38.79 s | 11.61 GB | 6 / 777 |

三列必须一起读：**8.6 倍的加速换来 6 次决定改变，并多吃 2.7 GB 显存。**
`serial_prefix` 那行的 `cache_hits: 740`（共 777）说明缓存的实际命中形态——
37 个 state 各预填充一次，随后 740 次直接复用。

## 17. 本文的事实来源，与本文不做什么

| 类别 | 判断方法 | 本文标记 |
|---|---|---|
| 代码引用 | 直接读 `src/semif_phase1/**`、`benchmarks/**`、`webgpu-demo/worker.js` | `文件:行` |
| 可实跑得到的产物 | 纯 Python / numpy，不加载权重：校准拟合、ECE、在 committed 产物上重算诊断量 | 标「实测」 |
| 公式与常量 | README、模型清单 `manifests/models.json`、`docs/METHOD.md`、`docs/CALIBRATION.md` | `README:行` 或文件名 |
| 前向产生的数值 | 本机无权重、无 CUDA，**跑不了** | 一律标注来源文件路径，不给自己的数值 |

**本文不做的三件事**：

1. **不给任何前向产生的概率或准确率**。§16.1 那一行的 logits 是 committed 产物里的记录，
   §15.2 的命令本机跑不了。所有「实测」都限定在不需要权重的那一层。
2. **不把上游 `docs/` 当结论**。`docs/METHOD.md`、`docs/CALIBRATION.md`、`docs/RESULTS.md`
   是本篇最重要的**线索来源**（它们指了该去看哪些代码），但凡本文写下的机制结论都回到了
   `src/` 与 `benchmarks/` 的行号。上游说明与代码不一致的地方（§13 最后一行那处）以代码为准。
3. **不推断未核实的技术结论**。例如「committed 产物里那三个 `adapter_*` 槽位为什么存在」，
   代码只告诉我 HEAD 没有 adapter 支持、取值全是 `null`；至于它是某个中途放弃的设计还是历史遗留，
   **本文不推测**。

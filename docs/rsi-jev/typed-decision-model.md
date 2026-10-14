# RSI-Jev：请求里的选项即标签集，一次前向读 1 + K 个位置

> 对应 README「关于各篇文档」一节
> 代码库：<https://github.com/Shanghua-Gao/RSI-Jev>

本文按 `rsijev/` 与 `serve/` 的实际代码逐环节说明这条实现的方式；文中出现的文件名与行号（`wire.py:217`、`arch.py:779`、`encode.py:211` 等）均相对该仓库根目录。与 [RYOTIDE](../ryotide/typed-decision-model.md) 的对照是逐节的。

---

## §0 与 RYOTIDE 的分野

RYOTIDE 是**零训练**：冻结 LLM、运行期标定读位置、把词表掩码到标签行。RSI-Jev 把这两件事都换掉了，而且换的方向和本仓已收的另外十九条都不同——它**训练一个只输出 K 个标量的头**，并且把「读什么」从「一个位置」变成「一个位置 + K 段跨度」。

| | RYOTIDE | RSI-Jev |
|---|---|---|
| 「typed」从哪来 | 分词器里的单 token 标签 | **请求声明的选项键**（`criteria` 的键），头输出维度就是 K |
| 词表维度 | 出现（掩码到标签行） | **不出现**（`proj: hidden → 1` 逐选项打分） |
| 读位置 | 运行期标定 | `encode` 在构造序列时算出来，运行期无标定 |
| 读几个位置 | 1 个（末位） | **1 + K 个**（决策位 + 每选项自己的 token 跨度） |
| 权重 | 冻结 | 全量微调（0.8B / 2B / 3B / 4B / 27B） |
| 置信度 | 读出来的 | **训出来的**（一个逐输入温度，或一次训练时就学好的 argmax） |
| 部署面 | 单出口 | **多出口**（同一权重在 layer 16 / 20 / 32 各有一个头）+ 图像通道 |
| 标定 | 无 | `calibration.safetensors` + `calibration.json`，随权重发布 |

一句话立场：**RSI-Jev 把 RYOTIDE 那条零样本读法当作自己必须打败的对照臂，然后把对照臂之外的每一件可训练的东西都训了**——那个对照臂在代码里还留着，叫 `LogprobReadout`（`rsijev/arch.py:954-1027`，**零可训练参数**，注释写着 *This is the arm every trained readout has to beat, and until it is measured there is no scale on which a trained number means anything*）。这是本仓二十篇里唯一一份把零样本读法**当基线保留在同一个模块里**的实现。

还有一层不属于「读法」但属于这条路线本身的：上游是一个**递归自改进的研究循环**（README 首段），8 个版本、496 个实验都由循环自己跑完并写档（`EXPLORE.md`）。这决定了它的仓形态——`rsijev/` 每个模块第一行标 **EDITABLE** 或 **PROTECTED**（`rsijev/README.md:9-27`），循环可以改模型、数据、配方，不可以改「什么算分」。

---

## 本文的事实来源

| 类别 | 来源 | 用法 |
|---|---|---|
| 代码引用 | `rsijev/`（arch / encode / contract / calibrate / adaptive / exit_policy / metrics / vision）、`serve/`（wire / infer / plan / app / release / server / decider / effort / images / padded / batcher）、`scripts/package_multiexit.py`、`scripts/run_arm_lib.py` | 全文逐环节，带 `文件:行` |
| 发布配方（仓内） | `data/v4.0-vl_recipe/`、`data/v5.0-vl_recipe/`、`data/v6.0-vl_recipe/`、`data/v6.1-vl_recipe/` 下的 JSON 与 README | 温度、出口、阈值、spec 字段 |
| 版本记录（仓内） | `versions/*.md` | 各代权重改了什么，与「发布了什么」 |
| committed 产物（仓内） | `serve/README.md` 的两份 `<!-- response: vX.Y -->` 实例、`serve/examples.json`、`data/*_recipe/*.json` | 末节实例走查的全部数字 |
| Hub 核对 | `huggingface.co` 的 API、逐文件体积与 blob id、以及**权重仓自己的小文件**（`meta.json` / `calibration.json` / `SHA256SUMS` / `md5.txt`，含对 `calibration.safetensors` 用 HTTP Range 取头部与标量字节） | §13.2 的容器算术、§13.3 的四类机械核对、§11.4–11.6 的两处同名两值 |

权重文件（`tower.safetensors` 等）一个都没下；`calibration.safetensors` 也只用 HTTP Range 取了头部与 4 个字节，没有整份下载。

**本文不给任何由前向产生的数值。** 权重没有下载（4B 档 9.68 GB、27B 档 54.47 GB），13 个 RSI-Jev 仓一个都没拉。所有数字要么是随仓提交的产物里的，要么是在这些产物上重算出来的，要么是 Hub 给的文件字节数——三类都不需要跑上游程序。需要权重的部分（`serve/README.md` 那两份实例响应、`docs/inference.md` 的逐格延迟）本文只**引用**并标明它自陈的来源 run 与条件。

---

## §1 端到端数据流

一次 `POST /v1/systemone`：

```
HTTP body {state, model, questions, images?, effort?, confidence_threshold?}
   │
   ├─ pydantic StrictModel: extra="forbid", strict=True        app.py:55-110
   │     → 未知字段拒、类型不coerce
   │
   ├─ state_to_text(state)                                      wire.py:78-103
   │     chat 转写：按参考实现校验角色/文本，然后 **序列化成 JSON**（不过 chat 模板）
   │     其他：字符串原样；其余 json.dumps(separators=(",",":"))
   │
   ├─ parse_questions(questions)                                wire.py:161-166
   │     每题 → Question(mode, instructions, options, criteria)
   │       noul   options 恒为 ("false","true")                 wire.py:120-132
   │       choice options = tuple(criteria)  ← 插入顺序即选项顺序  wire.py:140
   │       score  options = ("0",..,"K-1")                      wire.py:147-156
   │
   ├─ parse_images(images, state)   [有图时]                    images.py:29-36
   │     1-4 张，data URL，20 MiB / 20 MPix / 单帧 / 长宽比 ≤ 200:1
   │
   ├─ plan_request(tok, state, questions, enc)    ← 模型外，纯 CPU  plan.py:109-163
   │     encode_question 逐题：render → tokenize → 记下
   │       decision_index（"Answer:" 的末 token）               encode.py:211
   │       option_span[k]（第 k 个选项块的 [start,end)）          encode.py:205-209
   │     _shared_prefix → 决定走 plain / cached / doc            plan.py:180-201
   │
   ├─ [排到 GPU 线程]  score_planned                            infer.py:662-704
   │     plain : run_rows(每题整条)                             infer.py:95-158
   │     cached: encode_prefix 一次 → 每题从缓存续跑
   │     doc   : DocCache 跨请求复用                              infer.py:253-374
   │     (多出口 release + effort≠high) → score_adaptive         infer.py:577-659
   │
   ├─ 模型内（每次前向）
   │     tower → hidden_states
   │     h = hs[readout_layer]
   │     decision_h = h[b, decision_index]                       arch.py:779
   │     option_h   = mean(h over option_span[k])                arch.py:781-789
   │     logits = scorer(decision_h, option_h, option_mask)      arch.py:809-812
   │       option_xattn: q=decision, k/v=options, ctx=attn(q,k,v)
   │         v1.0-v2.1  xattn_combine="sum" → logits=proj(v+ctx)  arch.py:339-340
   │         v3.0+      xattn_combine="mlp" → comb([v, v*c, v-c]) arch.py:341-343
   │     logits.masked_fill(~option_mask, -inf)                  arch.py:351-352
   │     校准：logits /= exp(cal_log_temperature(...))            arch.py:821-824
   │
   ├─ unpermute_logits(·) 回到 canonical 顺序                     infer.py:153-154
   ├─ / temperature  （服务面恒为 1.0）                            infer.py:154
   ├─ F.softmax(logits, -1)  → K 个数                             infer.py:155
   │
   ├─ Prediction(probs)  兜底重归一（和与 1 差 > 1e-4 才动）      contract.py:63-74
   │
   └─ to_answer(q, probs)                                        wire.py:190-210
        noul  → {"type":"noul","noul": p(true)}        ← 只有这一个字段
        choice→ + choice / probabilities / confidence
        score → + score=Σ i·p_i / legend / probabilities / confidence
      finish(): usage = {input_tokens, output_tokens: len(questions)}   app.py:152-163
      _with_depth(): usage.depth / .confidence / .effort（有出口的 release 才带）
```

三个和另十九篇不同的地方，在这张图里就能看出来：**模型内的一次前向读两类位置**（`decision_index` 与 K 段 `option_span`）；**归一化之前有一次逐输入的标量除法**（校准）；**`usage.output_tokens` 是题目数**。

---

## §2 「typed」从哪来：请求里的选项键，加一个只输出 K 个标量的头

标签集不是分词器给的，不是训练时固定的槽位，而是**这一条请求自己声明的选项**：

- `choice`：`options = tuple(criteria)`（`serve/wire.py:140`），键的插入顺序就是选项顺序，描述是值；
- `score`：`options = tuple(str(i) for i in range(len(criteria)))`（`wire.py:153`），档位是下标；
- `noul`：`options` 恒为 `("false","true")`（`wire.py:53`），线上键是 `"true"`/`"false"`，`p(true)` **按键读、不按位置读**（`wire.py:120-132` 的注释：*p(true) is read by key, never by position, so the two orders cannot drift apart*）。

于是封闭性由**头的形状**保证，不由分词器保证。冠军 readout 是 `option_xattn`（`data/v6.1-vl_recipe/1-member-b-domain-mix.json:2`、`rsijev/README.md:128` 的 v1.0 spec 同），它的输出就是一个选项一个标量：

```python
# rsijev/arch.py:279-284
elif cfg.readout == "option_xattn":
    self.q = nn.Linear(hidden, d)
    self.k = nn.Linear(hidden, d)
    self.v = nn.Linear(hidden, d)
    self.attn = nn.MultiheadAttention(d, cfg.xattn_heads, batch_first=True)
    self.proj = nn.Linear(d, 1)
```

`self.proj` 是 `Linear(d, 1)`，作用在 `(B, K, d)` 上——**词表维度从头到尾不出现**。所以本仓同族的那个诊断量 `marker_mass` 在这个实现上**概念不适用**：标签之外的概率质量在数学上不存在（与 Kev / decider / Nimble / pplx-decider / JevAny-native 同一个结论，见 [comparison.md](../comparison.md) §5）。四条读法里 `pointer` 那一条（`arch.py:322-325`，`proj: hidden → max_options`）也一样。

**选项键对模型可见**，这是与参考实现的一处显式分歧，而且被写进了 `/v1/limits`：

> `serve/README.md:196-201`：*The reference hides them, so renaming a key provably cannot change the answer. Here the key is part of the rendered block, so renaming one can move the answer.* 报告为 `option_keys_visible_to_model: true`（`serve/wire.py:217`）。

渲染形式在 `rsijev/encode.py:160-180`：每个选项一个块 `- <键>: <描述>`（`encode.py:165`），描述为空时退化成 `- <键>`（`:166`）。所以键确实进了 prompt，改名**可以**移动答案——这是这条路线自己承认的一处不封闭。

**唯一会碰词表的地方是零样本对照臂**：`LogprobReadout`（`arch.py:954-1027`）按每个选项 key 的第一个 token 取 logprob（`arch.py:976-1000`），并且专门处理了「加空格后首 token 撞车」这件事——`" 0"/" 1"/" 2"` 在这个分词器下都以空格 token 开头（`arch.py:981-994`）。**发布里没有一条用这个读法**（13 个 spec 全是 `option_xattn`、`residual: false`），它留在仓里是当基准的。

---

## §3 读位置：`encode` 在构造序列时算出来，运行期无标定

与 Nimble 同一条思路，机制不同。位置不是标定出来的，是**拼序列时顺手记下的两个下标**：

```python
# rsijev/encode.py:201-211
head, parts = render(state, q, cfg, order)
ids = tokenizer(head, add_special_tokens=False)["input_ids"]
spans = []
for block in parts[:-1]:                       # one per option
    start = len(ids)
    ids += tokenizer("\n" + block, add_special_tokens=False)["input_ids"]
    spans.append((start, len(ids)))
option_index = [e - 1 for _, e in spans]
ids += tokenizer(parts[-1], add_special_tokens=False)["input_ids"]
decision_index = len(ids) - 1
```

- **决策位**是 `parts[-1]` 的末 token，而 `parts[-1]` 就是 `"\n\n" + cfg.answer_cue`（`encode.py:179`，`answer_cue` 默认 `"Answer:"`，`:60`）。所以读的是 `Answer:` 那个冒号所在的位置——**由末尾那句字面量定义**，与 H2O-Lightning / metask-jev-rain 同一族（「最后一位由 prompt 的尾巴定义」），但这里有第二条腿（下面 §4）。
- **选项跨度**是每个选项块自己的 `[start, end)`（`encode.py:208`）。`option_index` 是块的**末 token**（`:209`），但发布配方用的是 `option_pool: "mean"`（所有 spec），所以实际读的是**整段跨度的均值**（`arch.py:781-789`）。
- v5.0-VL 之后还有 `option_pool_own_tokens`：把每段跨度的起点推过它自己的 `"\n- <键>:"` 前缀（`encode.py:243-249`，`own_token_spans`），理由是**因果塔里这段前缀紧跟在上一选项之后、带着上一选项的信息**。`encode.py:53-56` 记着这个 bug 的代价：*Whole-block pooling made the 4B exit line pick the option after the right one on long numbered lists (CLINC150: 45% gold+1)*。

**运行期没有标定、没有扫描、没有候选位置、没有阈值。** 位置是构造出来的常量。训练期的 `readout_layer` 才是那个被扫过的自由度，而且它的下标约定被显式写下来防错：

> `rsijev/arch.py:31-46`：`output_hidden_states` 给的是 `(embeddings, out_of_layer0, ..., out_of_layer_{L-2}, normed_final)`，所以 `readout_layer = k` 读的是**第 k-1 层**的输出。注释：*Reading `layer_types[k]` instead of `layer_types[k-1]` is off by one, and on a 3:1 hybrid that lands on the wrong ATTENTION TYPE, not merely a neighbouring depth.*

这一句和 RYOTIDE 那篇被改正的滑动窗口层数错误是同一类坑：**在 3:1 混合架构上，差一层等于差一个注意力类型**。

---

## §4 一次前向读什么：1 个决策位 + K 段选项跨度

RYOTIDE 只读最后一个位置；这里读了 **1 + K 个对象**，而且两者来自**同一次前向**的同一张 hidden states：

```python
# rsijev/arch.py:775-789
decision_h = h[b, decision_index]                        # (B, H)
if option_span_start is not None and self.cfg.option_pool == "mean":
    t = torch.arange(h.shape[1], device=h.device).view(1, 1, -1)
    inside = ((t >= option_span_start.unsqueeze(-1)) &
              (t < option_span_end.unsqueeze(-1))).to(h.dtype)   # (B,K,T)
    denom = inside.sum(-1, keepdim=True).clamp_min(1.0)
    option_h = torch.einsum("bkt,bth->bkh", inside, h) / denom
```

然后 `option_xattn` 的合并方式：

```python
# rsijev/arch.py:333-340
q = self.q(decision_h).unsqueeze(1)                  # (B, 1, D)
k, v = self.k(option_h), self.v(option_h)            # (B, K, D)
pad = None if option_mask is None else ~option_mask
ctx, _ = self.attn(q, k, v, key_padding_mask=pad, need_weights=False)
if cfg.xattn_combine == "sum":
    logits = self.proj(v + ctx).squeeze(-1)          # (B, K); ctx cancels (see ArchConfig)
```

**注释里那句 `ctx cancels` 是这一节最值得抄下来的一行。** `proj` 是线性的，`proj(ctx)` 对全部 K 个选项是**同一个常数**，在 softmax 里约掉——所以在默认的 `"sum"` 组合下，「决策位去 attend 选项」这件事**对答案零贡献**，起作用的只有 `proj(v_k)` 那一项。代码自己写着 `arch.py:79-81`：*the cross-attention and the decision query contribute nothing*。

这不是猜想，是一个能从代码读出来的恒等式：`softmax(proj(v + c)) = softmax(proj(v) + proj(c))`，而 `proj(c)` 与 k 无关。**代价是那个把 `decision_index` 读出来的位置在 `sum` 下是白读的。**

**而这条路线的换代史正好是「把这一项修活」。** 从发布出去的 `meta.json` 逐个读 `spec.arch_extra.xattn_combine`：

| 代 | `xattn_combine` | `max_options` | `option_pool_own_tokens` |
|---|---|---|---|
| v1.0-0.8B / v1.0-2B | **未设 → 默认 `sum`** | 80 | 无（False） |
| v2.0 / v2.1 | **未设 → 默认 `sum`** | 80 | 无（False） |
| v3.0 | **`mlp`** | 160 | 无（False） |
| v4.0-VL | `mlp` | 160 | 无（False） |
| v5.0-VL | `mlp` | 160 | **`true`** |
| v6.0-VL / v6.1-VL（含 27B 与两个 MLX 构建） | `mlp` | 160 | `true` |

v1.0 与 v2.x 四代用的是**那个把交叉注意力约掉的默认值**。`versions/v3.0.md:16` 把 v3.0 描述成 *v2.1's lineage with more training data, **a new readout combine**, and that RL stage on top*，`:255` 的臂表里那一行写着 `v3-stack-ds2-xmlp` / *the MLP combine on data-scale-2: suite 0.7589*。也就是说：**v3.0 的「新 readout combine」就是让 §4 上半段那个恒等式不再成立的那一步**，而它发生在 v1.0 发布之后。

`mlp` 的算式是（`arch.py:341-343`）：

```python
c = ctx.expand_as(v)
logits = self.comb(torch.cat([v, v * c, v - c], dim=-1)).squeeze(-1)
```

`comb` 是 `Sequential(Linear(3d, 512), GELU(), Linear(512, 1))`（`arch.py:286-287`）——**从这一代起，决策位与「整个选项集合」才真正进了 logit**。另外两种变体（`bilinear` / `bilinear_norm`，`arch.py:288-292`、`:344-350`）零初始化所以起点逐位等于 `sum`，但发布里没有一条用。

`option_h` 用的是 `"mean"`（所有 spec），`"last"` 那条退化路径（取块末 token，通常标点）在 `encode.py:49-52` 被注释为 *can carry almost no option identity*。

---

## §5 掩码与归一化：掩码在头里，重归一化的对象是标签

**掩码**是 `option_mask`：批里每个样本的 K 个槽位中哪几个是真的（`encode.py:399`、`:417`），在头的出口统一压成 `-inf`：

```python
# rsijev/arch.py:351-352
if option_mask is not None:
    logits = logits.masked_fill(~option_mask, float("-inf"))
```

掩的**不是词表**——是「这一题的第 k 个选项槽位存不存在」。批里不同题选项数不同，统一补到 `max_options`：服务面的下限是 `DEFAULT_MAX_OPTIONS = 80`（`serve/infer.py:36`），而**权重自带的那个值**在 `meta.json` 里——v1.0–v2.1 四代是 80，v3.0 起是 160（`data/*_recipe/*.json` 同）。服务面取 `max(spec["max_options"], 本题选项数)`（`serve/server.py:153`）。

**但 `max_options` 是补宽的槽位数，不是头的容量。** `option_xattn` 的 `proj` 是逐选项的 `Linear(d, 1)`，K 多大都算得出来——`serve/wire.py:46-47` 的注释：*The option_xattn head reads any number of options. v3.0 and v4.0-VL trained with up to 160; picking one title among 5,120 still answers (no failure, accuracy falls slowly)*。真正有固定容量的是 `pointer` 那条 `proj: Linear(hidden, cfg.max_options)`（`arch.py:277-278`），而发布里没有一条用它——所以 `RSIJEV_MAX_ANSWERS` 那个 5,120 的上限是**接口定的**，不是权重定的。

**归一化**就是一次 `softmax`，落在 K 个（被掩过的）标量上：

```python
# serve/infer.py:153-155
logits = unpermute_logits(model(**batch), batch["option_perm"],
                          batch["option_mask"]) / temperature
probs = F.softmax(logits, dim=-1).float().cpu()
```

三处细节值得单独说：

1. **`unpermute_logits` 是置换唯一的还原点**（`encode.py:449-469`）。训练期选项顺序会被打散（§6），logits 以**呈现顺序**返回，在任何人读之前散射回 canonical 顺序。注释：*If a permuted presentation reached them unmapped, `score` would average POSITIONS rather than levels and the gold would be compared against the wrong option.* 填充槽的 `option_perm` 是 0，会撞上真选项 0，所以散射时把它们送到一个额外的草稿列再丢掉（`encode.py:461-469`）。
2. **`/ temperature` 在服务面上恒为 1.0**。`run_rows` 的默认参数是 `temperature: float = 1.0`（`infer.py:97`），而从 `make_scorer`（`serve/server.py:131-184`）到 `score_planned` / `score_image_planned` 没有一处传过它。**改数值的只有权重自带的那条校准**（§11）。
3. **`Prediction` 还会兜底归一一次**，但只在偏离超过容差时（`rsijev/contract.py:69-74`）：和不为正的有限数就抛，与 1 差超过 `1e-4` 才重新除一遍。

`noul` 与另外两个模式共用同一次 softmax，只是答案对象不同：`Prediction.noul()` 返回 `probs[1]`（`contract.py:80-81`，合约固定 `("false","true")`），并且**只有这一个字段**：

> `serve/README.md:174-177`：*noul carries only `noul` (no probabilities, no confidence)*。

---

## §6 顺序偏置：训练期按题打散，运行期只留一条诊断臂

因果编码器给第 k 个选项一个「看得到 1..k-1、看不到 k+1..K」的表示，位置本身携带了任务没打算给的信息。`encode.py:39-47` 把这个现象写成了配置项的注释，而且给了一个**具体的观测**：

> *The trained readout collapses onto that off-distribution: last option on 800/800 typed-decisions score questions, first option on about half of MMLU-Pro.*

处置分三档，都在 `EncodeConfig.option_order`（`encode.py:48`）：

| 取值 | 做什么 | 谁用 |
|---|---|---|
| `canonical` | 按原顺序 | **服务面**（`serve/release.py:507` 写死 `option_order="canonical"`） |
| `reversed` | 整题反转 | 诊断：*if the head still answers by POSITION it is positional, if it follows content it is semantic* |
| `shuffled` | 每例从编码器 rng 打散 | **训练期**（v1.0 spec `"option_order": "shuffled"`，`rsijev/README.md:129`） |

另外有一条独立的评估口径 `eval_option_orders: ["canonical","reversed"]`（`rsijev/README.md:130`），说明反转臂是**随训练配置一起声明的**，不是事后补的。

**与 Kev / SemIf-OpenJev / H2O-Lightning / pplx-decider / JevAny 的差别**：这里既有训练期打散（像 Kev），也有运行期可跑的诊断臂（像 SemIf-OpenJev 的反转），而且**这两件事都在同一份发布的代码里**，不像 H2O-Lightning 那样工具与度量同时缺席。但发布卡片与随仓记录**没有给出翻转率数字**——本文不代填。

---

## §7 多问题共享 state：前缀缓存、文档缓存，以及「读一次」的判据

合约本身要求每题独立（`rsijev/contract.py:5`：*the questions may not read each other*），所以共享的只能是 **state 那一段前缀**。这里把它做成了三档可观测的路径，由 `plan_request` 在**模型外**决定（`serve/plan.py` 是 torch-free 的，MLX 后端共用同一份规划代码，`serve/plan.py:1-6`）：

```
plain   每题整条跑             — 前缀不共享、或共享了但省不下
cached  本请求内 state 读一次   — encode_prefix → 每题从缓存续跑
doc     跨请求复用 DocCache     — 键是 (权重指纹, 精确 token ids, 图像键)
```

判据是**省下的 token 数要越过一次第二遍顺序前向 + 复制缓存的固定成本**：

```python
# serve/infer.py:547-551
elif prefix is None or len(questions) < 2 or \
        (len(questions) - 1) * len(prefix) < min_saved_tokens:
    path = "plain"
```

而 `min_saved_tokens` **按机器选**（`infer.py:59-81`）：融合核 + 数据中心 GPU（A100/H100）取 2048，否则 480。理由写在注释里而且是有数字的：`infer.py:40-58` 记着「A100 + fla 下 98 token 是 0.55x，是纯亏」而「GB10 的 torch fallback 下 161 token 起 1.09x 到 1,052 token 的 3.10x」。480 是**实测过的最小获胜点**，不是向零的外推。

`cached` 的精确性不是信任来的：

> `rsijev/arch.py:622-627`：*Under a causal mask, running the state once and continuing each question on its cache computes the same activations as running each full sequence -- so this is an exact optimisation, not an approximation, and `tests/test_prefix_cache.py` checks that rather than trusting it.*

但**混合架构让 cache 的复制不平凡**，这是这一节最值得记的一处：

```python
# serve/infer.py:186-210  _replicate
for src, dst in zip(cache.layers, replica.layers):
    for attr in ("conv_states", "recurrent_states", "is_conv_states_initialized",
                 "is_recurrent_states_initialized", "has_previous_state",
                 "conv_kernel_size"):
        val = getattr(src, attr, None)
        # Lists up to transformers 5.16, dicts keyed by state index from 5.17.
        # Missing the dicts shared them with the caller's cache, so reordering the
        # replica rewrote the original: every batch after the first, and every
        # document-cache hit, continued from a state that already held another
        # question.
        if isinstance(val, (list, dict)):
            setattr(dst, attr, val.copy())
```

Qwen3.5 的文本塔是 `linear_attention ×3 + full_attention` 的重复，所以 cache 里**同时**有注意力的 K/V 和 DeltaNet 的卷积态与循环态；逐层浅拷贝时漏掉 dict 形式的那几个字段，会把副本和原件共享，于是第二批开始就在读别人的 state。**这个 bug 有第二处独立记录**：`docs/inference.md` 的末尾勘误写着 transformers 5.17 下 release-day 的代码「第 17–32 题从错误 state 回答」，A 配置 1,052 token / 32 题的格子里「32 个答案里 3 个与参考不同」。

`DocCache` 另有两件设计：**键是 token ids 而不是文本**（同一个文本在不同上下文里可以分词不同，`infer.py:256-260`），以及 **holdback**——每条 state 末尾 8 个 token 不入缓存、每题重读（`infer.py:269-275`），因为「一条在增长的 state 会改掉自己最后几个 token（一个收尾括号变成逗号）」，留出这 8 个才让下一条更长的 state 还能找到它作为前缀。

---

## §8 `effort`：一次前向跑几层（本族新增的维度）

v6.0-VL 之后，同一份权重在 **layer 16 / 20 / 32 各有一个头**（`spec.arch_extra.aux_exits`，`rsijev/arch.py:181-195`；27B 是 48 / 56 / 64）。`effort` 决定一次请求用几层，**不是生成多少 token**：

| effort | 层 | 行为 |
|---|---|---|
| `low` | 最浅的 aux exit | `Policy(..., tau=-1.0)`，每行都在第一档停（`serve/effort.py:145-147`） |
| `medium` | 最深的 aux exit | 出口表缩成 `[L, main]`（`effort.py:148-150`） |
| `high` | 主出口、全层 | 即 `--adaptive off` |
| `auto` | 16/20/32 三档 | 级联：**停在第一个校准后 top-1 达到该档 tau 的出口**，否则主出口 |

级联的实现是**分段跑同一个塔**：`rsijev/adaptive.py:300-310` 的 `cascade()` 逐档把未定的行留下；`staged_scores_fast` 把上一段的**未归一化** state 当 `inputs_embeds` 喂下一段（`adaptive.py:20-24`），所以每一段看到的输入与一次不中断的前向完全相同。段边界必须是**层型周期的整数倍**（Qwen3.5 是 4），因为文本模型按层在列表里的位置选 mask（`adaptive.py:24-26`）。

三处容易踩的：

1. **停止判据是「该出口自己的校准之后」的概率**。每个出口有自己的温度（§11），没有它各档不可比——`rsijev/adaptive.py:16-19`：*tau: one threshold on the calibrated top-1 probability, the same at every exit (calibration is what makes the exits comparable)*。
2. **回答的行会带着「回答它的那个出口」的校准回去**，不是主出口的。`serve/infer.py:557-574` 的 `_adaptive_finalizer` 直接照抄 `DecisionModel._readout` 在主出口的算法，aux 出口用该出口自己 DEV 拟合的那一份。
3. **图像请求永远跑满层**，因为 aux 头只在文本上训过（`serve/infer.py:912-914`、`serve/batcher.py:199-203`、`docs/mac.md` 的 Limitations 最后一条）。微批池化的行也是。

还有一个**构造性的不变开关**：`tau > 1` 时永不提前退出，于是整条路逐位等于固定主出口——5-exit-policy 那个 JSON 就是把 `1.0` 映射成 `NEVER = 2.0`（`serve/effort.py:43`、`:109`）。v6.1-VL 的随仓记录把这条写成了可检查的断言：

> `versions/v6.1-vl.md:159-162`：*The `low`, `medium` and `high` effort paths give bitwise the forced-exit results. The unset default is bitwise the package without the `auto` thresholds, and a threshold of 1.0 is bitwise `high`.*

**发布出去的策略值全在权重仓的 `meta.json` 里**，可以逐个读出来（这就是 `serve/release.py:140-142` 的 `adaptive_block` 与 `:230-247` 的 `auto_thresholds` 读的那两块）：

| 仓 | `adaptive.exits` | `adaptive.tau`（单阈值回退） | `adaptive.auto_thresholds` |
|---|---|---|---|
| v6.0-VL-4B | `[16, 20, 32]` | 0.59 | `{16: 0.95, 20: 0.59}` |
| v6.1-VL-4B | `[16, 20, 32]` | **0.95** | `{16: 0.85, 20: 0.5}` |
| v6.1-VL-27B | `[48, 56, 64]` | 0.8 | `{48: 0.8, 56: 0.7}` |

三处值得停一下：**同一个「0.95」在 v6.0 里是 `auto` 的 tau16、在 v6.1 里退成了那个单阈值**；**27B 的出口不是 16/20/32 的倍数关系，而是 48/56/64**，所以 `serve/effort.py` 里「最浅的 aux 出口」「最深的 aux 出口」这两个说法是相对出口表定义的，不是相对层号；**27B 的 `meta.json` 里没有 `release` 块**（只有 `release.vision`），所以 `release.cal_mode` 那个字段在它身上连「旧值」都没有。

最后一条**默认值不是选出来的**这件事，这个实现自己说破了。v6.1-VL 没设 effort 时多问题请求用**一个**共享 tau，规则挑中的 0.85 在确认半区上超了深度上限，于是规则**回退到它最保守的成员 0.95**，而记录写着：

> `versions/v6.1-vl.md:117-119`：*That value is the fallback of a rule written before the fit, and it is **not confirmed***；`data/v6.1-vl_recipe/5-exit-policy.json` 的 `single` 块 `CONFIRMED: false`，并附一句 *0.80 passed C1-C3 on both halves; the rule does not pick it*。

---

## §9 图像通道：同一条读法，另一条编码路径

v4.0-VL 起，请求可以带 1–4 张图（`serve/images.py:29-34`：20 MiB 解码后、2,000 万像素、单帧、长宽比 ≤ 200:1）。state 里用字面量 `<image>` 按顺序引用，没有标记就把图放在 state 前面（`serve/README.md:114-116`）；`serve/wire.py:97-99` 拒绝任何非文本的 chat content，所以 Qwen 的 `<|image_pad|>` 之类 token 进不了 state。

**图像与文本走同一个头、同一条读法**（不像 H2O-Lightning 那样另立一条服务面），差异在编码端：

- 视觉塔是**基座自带的那个、冻结的**（`rsijev/vision.py:167` `load_visual`）；4B 档给 0.33B、27B 档给 0.46B（§13.2）。
- 图像 token 拿的是 **M-RoPE 三元组位置**（`rsijev/vision.py:139-152`），而 `serve/infer.py:744-752` 记着这带来的坑：*the text after an image continues from the grid's extent, not from the token count, so the positions of a question's tail cannot be re-derived from where the prefix stopped*。做法是**每个位置都从整条序列的 `mrope_position_ids` 上切**，而不是重新推。
- 共享前缀这条优化在图像上照样走，但判据换了：切出来的前缀与后缀的 M-RoPE 位置都来自那张整序列张量（`infer.py:846-853`），`tests/test_image_cache.py` 在 CPU fp32 上核到 ~3e-8。
- 图像 token 预算 **1,024 每问**，按图数均分（一张最多 1,024，四张各最多 256），文本上限另加这个预算，所以「带了一张图」不会让 state 被砍得更短（`serve/server.py:119-122`）。

`docs/mac.md` 的 Limitations 里有一条诚实但扎眼的：*On image questions MLX and PyTorch differ a little more than on text (mean probability difference 0.008 against 0.004), in both builds and in the unquantized one: the vision tower's arithmetic differs between the two frameworks.*

---

## §10 服务化与 fail-loud 风格

**契约面**：一张路由表（`serve/README.md:94-99`）——`POST /v1/systemone`、`GET /v1/models`、`GET /v1/limits`、`GET /health`、`/health/live`。健康路由**永不鉴权**（`serve/app.py:271`，与参考实现一致），设了 `--api-key` 后其余全要 `Authorization: Bearer`。

**「借来的名字不回声」**：请求可以指名 `jev-1.13.0` 这类本服务器并不提供的版本（`--accept-model`），但响应里的 `model` 是**本服务器自己的名字**：

> `serve/README.md:104-107`：*The answer's `model` field then carries **this** server's model name, not the borrowed one, because echoing it would claim the answer came from Jev — so an app that checks the returned name against the one it sent will need that check relaxed.*；实现见 `serve/app.py:304`（`served_model_name if req.model in borrowed else req.model`）。

这是**第三条**处理「响应里的 model 字段」的路线：Clef 是无条件回声调用方的字符串（`joint_schema_model.py:573`），这里是「只在借名时替换成真名」，并配一个 `x-rsijev-model` 头。

**三类错误信封**（`serve/app.py:253-265`、`serve/wire.py:58-63`）：

| 情形 | 状态码 | 形状 |
|---|---|---|
| 领域错误（未知 model、题数/选项数越界、图不被支持） | 422 | `{"error":{"message": …}}` |
| schema 失败（pydantic） | 422 | FastAPI 的 `{"detail":[…]}` |
| 输入过长 | 422 | `{"error":{"message": "<exc>; the input is not truncated, shorten it or raise RSIJEV_MAX_INPUT_TOKENS"}}` |
| 429 / 503 / 529 | 同上 | 带 `Retry-After: 1` |

**输入永不截断**是这个实现的一条硬立场，而且是在**服务面**上改的，不是丢弃训练期的行为：

```python
# serve/server.py:88-95
explicit = (max_length is not None or truncate is not None
            or os.environ.get("RSIJEV_MAX_LENGTH", "").strip()
            or os.environ.get("RSIJEV_TRUNCATE", "").strip())
if not explicit:
    from serve.wire import MAX_INPUT_TOKENS
    enc = dataclasses.replace(enc, max_length=MAX_INPUT_TOKENS, truncate="none")
```

也就是说：**同一份代码里，服务面 `truncate="none"`（超长 422）、benchmark 脚本仍是 `"left"`**——`serve/README.md:218-219` 直说 *Earlier servers cut the start of an over-long state without a warning; the benchmark scripts still do, so the published numbers reproduce*。这是一条**为了保号而保留的静默截断**，只在非服务路径上活着。

**启动期自陈**：`serve/server.py:334-368` 在接第一个请求之前打印设备、精度、生效的 speed path、micro-batching、rows sorted、option slots trimmed、图像上限、adaptive 模式与 tau、effort 默认、输入上限与越界策略、base model、calibration。**这是一个「改了数值的东西自己会说」的实例**，而且比 H2O-Lightning 的 `/health` 更全——它在 stdout 上，也在 `/v1/limits` 的 `calibration` / `version` / `images` 字段里。

**两条服务线程模型**：一个线程独占模型（`serve/batcher.py:1-21`），请求在调用方线程里 plan（分词），再排队。开了 `--batch-window-ms` 时，**「plain」请求的题会被池到一次前向里**（`serve/batcher.py:184-211` 的 `_pooled`），每个请求的 `input_tokens` 仍按自己的行算（`:208`），但**批里别的行会移动它的概率**（bf16 形状依赖），所以默认关（`batcher.py:19-21`）。

**一条和本仓同族的「算得出来 ≠ 接进来了」在这条路线上也存在**：`serve/README.md` 与 `serve/wire.py:14` 引的 `reference/openjev-sglang` **不在仓里**——`git ls-files` 对 `reference/` 零命中，`tests/test_serve.py:3-4` 引的那份 `reference/openjev-sglang/tests/test_api.py` 无从对照。README 里那句「与参考实现逐条相同」的清单，**只能靠仓内自己的断言维持**。

**另一处上游自陈与发布物不符**：`serve/README.md:204-206` 解释为什么不过 chat 模板时写的是 *A base model has no chat template*，而权重仓的 `tokenizer_config.json` 里那个 `chat_template` 是 **7,756 字符**、与基座 `Qwen/Qwen3.5-4B-Base` **逐字节相同**。真正的理由是**训练语料从没经过模板**（state 本身就是 JSON 文档，见 §1 的 `state_to_text`），渲染它会把请求带出分布——**结论对、说法不对**。同一件事还有一半是反的：`state_to_text` **照样按参考实现逐条校验 chat 转写**（同样的角色集、只许文本内容、同样的 messages 信封），只是校验完**序列化成 JSON 而不是喂给模板**（`wire.py:78-103`）。所以「模板」在这条链上的位置是：**请求形状里认它、渲染路径里不用它**。

---

## §11 置信度从哪来，又怎么被改写

### 11.1 从哪来：训出来的

发布权重里**没有一条**的置信度是读出来的。原始 logits 由训练出来的头给出（§2），而**发布出去的数值**在那之后还要过一道**随权重发布的校准**：

```python
# rsijev/arch.py:821-824
if self.cal_mode != "none":
    with torch.autocast(decision_h.device.type, enabled=False):
        log_t = self.cal_log_temperature(logits.float(), decision_h.float(), mode_id)
        logits = logits.float() / torch.exp(log_t).unsqueeze(-1)
```

`cal_log_temperature` 是四条路（`arch.py:927-948`）：

| `cal_mode` | 逐行温度 | 发布过它的仓（读自各仓 `calibration.json` 的 `cal_mode`） |
|---|---|---|
| `none` | — | v1.0-2B、v1.0-0.8B（**仓里根本没有 `calibration.safetensors`**；`serve/README.md:61`：*v1.0, which ships no calibration*） |
| `temp` | 一个标量（`cal_logT.expand(B)`） | **v6.0-VL、v6.1-VL 4B、v6.1-VL 27B、两个 MLX 构建** |
| `temp_mode` | 逐模式一个标量（choice/noul/score） | 训练时总会拟合并记录（`arch.py:479-480`），未见发布 |
| `oof_head` / `oof_head_scorefloor` / `oof_head_joint` | **逐输入**：`log τ = w·f(x) + b`，线性头，特征取自同一次前向 | **v2.0、v2.1、v3.0、v4.0-VL、v5.0-VL**（全部是 `oof_head_scorefloor`） |

`oof_head` 那三个的特征在 `arch.py:440-456`：决策态在 DEV 上 PCA 出的 16 个方向，加上「p_top、top-2 logit 差、归一化熵、log K、模式 one-hot」。`log τ` 被钳在 ±3，即 **τ ∈ [0.0498, 20.09]**（`arch.py:437`）；`oof_head_scorefloor` 再多一条 `score` 行**只能软化不能锐化**（`arch.py:942-946`），理由是训练时的软金标 Brier 不该为 top-label ECE 让路（`rsijev/calibrate.py:18-21`）。

**这条链的每一步都是 argmax 保持的**，代码把它当定理写下来：

> `rsijev/arch.py:929-930`：*Argmax-preserving: every row is divided by one positive scalar, so top-1 is exactly the uncalibrated model's.*

也就是说：**校准改数值、不改答案**。这是与 H2O-Lightning 的 yes/no 地板（也只抬不动 argmax）同一类，但这里对**每一行**都成立，且不是地板而是一个正标量。

### 11.2 发布出去之前，还有一层改写

`Prediction` 只管三个标量（`contract.py:76-88`），发布层另写一个：

```python
# serve/wire.py:169-187
def confidence(probs: list[float]) -> float:
    k = len(probs)
    if k < 2:
        return 1.0
    return min(1.0, max(0.0, (k * max(probs) - 1) / (k - 1)))
```

**这是 peak 统计量，不是 `p_max` 本身**，而且是**按 K 归一的**：K=2 时它等于 `2p−1`，K 大时它趋近 `p_max`。注释把一次真实事故记下来了（`wire.py:175-182`）：上游先发的是 `1 − H(p)/ln(K)`（熵形式），两者**只在 uniform 与 one-hot 两点相等**，所以「只钉这两点的测试」没抓住它——四个选项、峰值 0.50 时，熵形式给 0.10 而正确的是 0.33。**任何拿 confidence 做阈值的东西当时都在看错的数。**

于是 `confidence` 与 `probabilities` 是**两个不同的数**，`versions/v6.1-vl.md:262-265` 为此专门留了一句给用户：*The API returns a probability **and** a `confidence`, and they are different numbers … Route on the probability of the option you intend to act on.*

`score` 也是导出量：`Prediction.score()` 是 `sum(i * p_i)`（`contract.py:83-84`），一个**零基下标**，不是分数区间的比例——`serve/README.md:44-45` 专门点了这句：*`score` is an index into the rubric, so 1.081 is "Urgent", not a fraction of the scale.*

### 11.3 「谁在改发布数值」的链表

```
训练出来的头产出 logits                       arch.py:809-812
   ↓  logit_cap = c·tanh(logits/c)            arch.py:813-819   （发布全为 null，未启用）
   ↓  ÷ exp(log τ(x))    ← cal_mode           arch.py:821-824   （v1.0 无此步）
   ↓  residual: base + correction             arch.py:826-842   （发布全为 residual:false，未启用）
   ↓  unpermute_logits（顺序，不是数值）        infer.py:153
   ↓  ÷ temperature = 1.0                     infer.py:154      （**服务面从没传过别的值**）
   ↓  softmax → K 个数                         infer.py:155
   ↓  Prediction 兜底重归一（>1e-4 才动）        contract.py:73-74
   ↓  confidence = (K·p_max − 1)/(K − 1)      wire.py:169-187   （**改写**）
   ↓  score = Σ i·p_i                          contract.py:83-84 （**改写**）
```

**发布配方里改数值的只有两处**：权重自带的逐出口温度，和发布层的 `confidence` / `score` 两个导出量。前者对每行除以一个**正标量**（不改 argmax），后者只改**怎么把分布讲给调用方**。

### 11.4 一处「同名两个值，只有一个活着」——以及一个连名字都是旧的字段

v6.0-VL 的包里，主出口的温度**在两个文件里写着两个不同的数**，而只有一个是读的。这一处不用推测，可以用 HTTP Range 从发布文件里把字节取出来对：

| | v6.0-VL-4B | v6.1-VL-4B | v6.1-VL-27B |
|---|---|---|---|
| `calibration.safetensors` 的 `cal_logT`（fp32 标量，**实际被读**） | **0.5** | **0.02** | **0.05** |
| `calibration.json` 的 `main_logT`（记录） | **0.8** | 0.02 | 0.05 |
| `calibration.json` 的 `T.32` / `T.64` | 2.2255 | 1.0202 | 1.0513 |
| 该文件里 `T` 与 `logT` 的关系 | 2.2255 = e^0.8 | 1.0202 = e^0.02 | 1.0513 = e^0.05 |
| 服务实际用的 T | e^0.5 = **1.6487** | e^0.02 = 1.0202 | e^0.05 = 1.0513 |

也就是说：**v6.0-VL 的 `calibration.json` 里 `main_logT`/`T` 那一对数是自洽的（e^0.8），但它和真正在用的 0.5 不是一个数**。这不是失误，打包脚本自己写着：

> `scripts/package_multiexit.py:8-11`：*calibration.safetensors gets `cal_logT` = the main exit's temperature fitted on the checkpoint's own DEV "cal" half … That buffer is the main exit's temperature when serving … `--cal-buffers` carries the other cal-4b buffers of an existing calibration file along; **under cal_mode "temp" they are not read**.*

同一份 `package_multiexit.py:12-14` 也说清了 `calibration.json` 的 `main_logT` 与 `T` 是 *the cascade selection's (**recorded**; serving reads the main exit's from the buffer above)*。**「记录值」与「活值」在同一个包里并存，而哪一个活只能从 `rsijev/calibrate.py:365-382` 读出来。**

### 11.5 更硬的一处：`meta.json` 里的 `release.cal_mode` 是上一代留下的

每个仓的 `meta.json` 都有一个 `release.cal_mode` 字段。把它与 `calibration.json` 的 `cal_mode` 并排：

| 仓 | `meta.json release.cal_mode` | `calibration.json cal_mode`（**实际被读**） |
|---|---|---|
| v2.0 / v2.1 / v3.0 / v4.0-VL / v5.0-VL | `oof_head_scorefloor` | `oof_head_scorefloor` |
| **v6.0-VL-4B** | `oof_head_scorefloor` | **`temp`** |
| **v6.1-VL-4B** | `oof_head_scorefloor` | **`temp`** |
| **v6.1-VL-4B 的两个 MLX 构建** | `oof_head_scorefloor` | **`temp`**（这两个仓的 `meta.json` 与 fp32 版**逐字节相同**，`583e873e…`） |
| v6.1-VL-27B | （无此字段） | `temp` |

**v6.x 的 `meta.json` 里那个字段是错的。** 它不是笔误：`scripts/package_multiexit.py` 在打包时改写 `calibration.json` 与 `meta["adaptive"]`，但**没有更新 `meta["release"]["cal_mode"]`**（该脚本里 `cal_mode` 只出现在写 `calibration.json` 的那一行），于是它继续保留父检查点的值。而全仓对 `meta["release"]["cal_mode"]` **零读点**——`serve/` 只从 `meta["release"]` 里读 `vision` / `max_length_text` / `adaptive`（`serve/release.py:142`、`rsijev/vision.py:82`）。

后果是具体的：一个按 `meta.json` 判断「这个包带不带拟合置信度头」的运维脚本会得到**相反**的结论。v6.x 的包确实**带着**一整个 cal-4b 头的张量（见 §11.6），但 `cal_mode` 是 `temp`，那个头**一个数都不参与**。这正是 [comparison.md](../comparison.md) §4.5「算得出来 ≠ 接进来了」的一个新形态：**接进来了，但被上一层开关关掉了，而描述那个开关的字段是旧的。**

### 11.6 4 个字节的活值与 174,928 字节的陪跑

`calibration.safetensors` 的张量清单（HTTP Range 取头部，逐仓核对）：

| 张量 | 形状 | 字节 | `cal_mode = "temp"` 下 |
|---|---|---|---|
| `cal_logT` | `[]` | **4** | **被读**（`CAL_MODE_BUFFERS["temp"] = ("cal_logT",)`，`calibrate.py:385`） |
| `cal_logT_mode` | `[3]` | 12 | 不读（只有 `temp_mode` 读） |
| `cal_w` | `[23]` | 92 | 不读（只有 `oof_head*` 读） |
| `cal_b` | `[]` | 4 | 不读 |
| `cal_feat_mu` / `cal_feat_sd` | `[23]` ×2 | 184 | 不读 |
| `cal_pca_mean` | `[2560]`（27B：`[5120]`） | 10,240 | 不读 |
| `cal_pca_W` | `[2560, 16]`（27B：`[5120, 16]`） | 163,840 | 不读 |

数据区 174,376 B + 544 B 头 + 8 B 长度 = **174,928 B**（27B：349,008 B），**其中 4 个字节是活的**。`load_calibration` 会把文件里存在的每个 `CAL_BUFFERS` 键都拷进模型（`calibrate.py:378-380`），所以那些张量**确实进了显存**；它们只是**从来不参与任何计算**——`cal_log_temperature` 在 `cal_mode="temp"` 上第一行就 `return self.cal_logT.expand(B)`（`arch.py:932-933`）。

**这就是「盘上有、装进去了、但没接上」的第三种形态**：不是文件没发（Nimble 的 manifest），不是路由指着不存在的文件（pplx-decider 的 `playground.html`），而是**发了一整套参数、加载器按名单收下了、前向里没有任何一条路径会读它**。

### 11.7 出口级联用的「置信度」是另一个字段

`usage.confidence`（`serve/app.py:189-191`）报的是**回答该题的那个出口的校准后 top-1**（`serve/plan.py:210-214`），也就是级联拿来和 tau 比的量。它**不是** §11.2 那个 `(K·p_max−1)/(K−1)`。同一个响应里因此可以同时出现两个都叫「confidence」的数，含义不同：`answers.<key>.confidence` 是 peak 统计量，`usage.confidence` 是未归一的 top-1。

---

## §12 四条不变量在 RSI-Jev 的落位

| RYOTIDE 的不变量 | 在这里 | 依据 |
|---|---|---|
| **一次前向** | **成立，但要按出口数分档** | 固定出口：一次完整前向；`effort=low/medium` 与级联：一次**跑到该出口为止**的前向（分段跑，段边界是层型周期整数倍，`adaptive.py:20-26`） |
| **只读最后一个位置** | **要改写** | 读 **1 + K 个位置**：决策位（`Answer:` 末 token，`encode.py:211`）+ K 段选项跨度均值（`arch.py:781-789`）。而默认 `xattn_combine="sum"` 下，那 1 个决策位**对答案零贡献**——`proj(ctx)` 在 softmax 里约掉（`arch.py:79-81`） |
| **掩码到已知标签集** | **概念不适用** | 头的输出维度就是 K（`proj: Linear(d,1)`），词表维度从不出现，`marker_mass` 没有对象。**另一种机制**：既不是「只请求 K 个 token id」，也不是「头只输出 K 个标量」那么笼统——这里是**逐选项对一段跨度做池化再打分**，选项之间唯一的交互是 softmax 的分母 |
| **重归一化** | **成立，对象是标签** | `F.softmax(logits, -1)` 落在 K 个标量上（`infer.py:155`），`Prediction` 再兜底一次（`contract.py:73-74`） |
| **`output_tokens = 0`** | **机制层成立、字段层不成立——第三种定义** | 这条路径确实不生成：`wire.py:33-35` 的注释写着 *This path generates nothing at all*。但字段被定义成**题目数**：`usage = {"input_tokens": prompt_tokens, "output_tokens": len(questions)}`（`app.py:162`）。参考实现报 N+1（每题一个 token 加一次前缀热身），这里报 N |

`output_tokens` 这一条要说清楚与已收两家的关系：Kev 把它定义成「序列化后答案 JSON 的 token 数」（机制成立、字段不成立），decisio 定义成「题目数」并写进契约让门去断言。**RSI-Jev 与 decisio 是同一个定义**（题目数），但**没有那条契约与门**——它只是 `finish()` 里的一个字面量。committed 的实例响应可以机械核到：`serve/README.md` 的两个 JSON 各有 3 个 `answers` 键、`"output_tokens": 3`（§14.1），`tests/test_documented_example.py:67-68` 把这条钉成断言：*one readout per question; nothing is generated*。

### 这条路线自己提出的一条不变量

**「一个问题看不见另一个问题」**。它写在 PROTECTED 的合约里而不是散文里：

> `rsijev/contract.py:3-6`：*One `Case` is one state plus several typed questions, exactly the body of a `POST /v1/systemone` request. A model answers every question of a case from one encoding of the state, and **the questions may not read each other**.*

这与 Clef 的联合解码（题与题在 4 层 decoder 里做自注意力、删掉一题剩下的答案可以变）**恰好相反**，也是这条实现敢做「多问题共享 state」的前提：共享的只能是 state 那一段，问题段互不可见。级联按**每行自己的**置信度独立停（`adaptive.py:300-310`），也是这条不变量的直接后果——**同一个请求里两题的 `usage.depth` 可以不同**。

---

## §13 文件对照表

### 13.1 代码

| 文件 | 行数 | 职责 |
|---|---|---|
| `rsijev/contract.py` | 130 | **PROTECTED**。`Question` / `Case` / `Prediction`；三个标量读数（argmax / p(true) / Σi·p_i）只在这里定义一次 |
| `rsijev/arch.py` | 1,027 | 模型轴：四种 readout、`ArchConfig` 的全部开关、校准、residual 对照、多出口与 `exit_tower`、零样本 `LogprobReadout` |
| `rsijev/encode.py` | 469 | **PROTECTED**。位置与填充记账：`decision_index` / `option_span` / `option_perm` / `collate` / `unpermute_logits`；截断三策略 |
| `rsijev/calibrate.py` | 385 | cal-1 / cal-4 的拟合与装载；DEV 分组与权重 |
| `rsijev/adaptive.py` | 696 | 分段早期退出：`Policy`、`staged_scores_fast`、每出口校准 |
| `rsijev/exit_policy.py` | 791 | 温度拟合与出口策略选择（`fit_temperatures` / `select_cascade` / `Thresholds`） |
| `rsijev/metrics.py` | 170 | **PROTECTED**。选择式指标（tie-aware coverage / AURC）、ECE、proper scores、决策分 |
| `rsijev/vision.py` | 285 | 图像：`ImagePrep`、`encode_vision_question`、`mrope_position_ids`、`VisionDecisionModel` |
| `serve/wire.py` | 217 | **契约面，纯 Python 无 torch 无 HTTP**：请求 → `Question`，分布 → 答案；极限；`confidence()` |
| `serve/infer.py` | 951 | 服务面前向：`run_rows`、`DocCache`、`VisionCache`、`plan_request`、`score_adaptive` |
| `serve/plan.py` | 245 | torch-free 的规划半边：一次分词、共享前缀判定、`record_confidence`、`adaptive_applies` |
| `serve/app.py` | 342 | 路由、schema、鉴权、错误信封、`finish()` 的 usage |
| `serve/release.py` | 532 | 找 checkpoint、建 `ArchConfig`、装校准、`TRAINING_ONLY_ARCH_KEYS` |
| `serve/server.py` | 435 | 起服务、启动期自陈、服务面 `truncate="none"` |
| `serve/effort.py` | 153 | `low/medium/high/auto` 的校验与 `Policy` 构造 |
| `serve/padded.py` | 368 | 短批右填充到 pad 倍数 + CUDA graph；**前缀路径永不填充** |
| `serve/kernels.py` | — | 视觉塔 RoPE 的单个融合 Triton 核（逐位复现 eager 算术） |
| `serve/batcher.py` | 317 | 一个线程独占模型；`--batch-window-ms` 时的跨请求行池化 |
| `serve/runtime.py` | 249 | 设备/精度、`--profile`、启动日志 |
| `serve/decider.py` | 108 | `Decider`：同一请求路径的进程内版本 |
| `serve/images.py` | 197 | data URL → 校验过的图；上限；`<image>` 标记 |

`serve/infer.py` 存在的理由本身是一句设计声明（`infer.py:1-13`）：`evaluate.predict` 吃 `Case`，而 `Case` 要求 gold，服务请求没有 gold，**与其伪造一个均匀的 gold 去满足一个 dataclass，不如直接调同一批原语**——然后用 `tests/test_serve_parity.py` 钉住两条路逐位一致。

### 13.2 发布物（Hub 核对）

作者名下的仓全部枚举过（`HfApi().list_models(author="shgao")` 返回 22 个，其中 12 个是 RSI-Jev），与 `serve/release.py:28-40` 的 `REPOS` **一一对应，没有漏掉的尺寸或量化档**。参数量与逐文件体积如下（Hub API，核于 2026-10-14）：

| 仓 | sha（前 8） | `base_model` | 权重文件合计 | 分解 |
|---|---|---|---|---|
| `shgao/rsi-jev-v6.1-vl-4b` | `7f93b192` | `Qwen/Qwen3.5-4B-Base` | 9,676,732,504 B = 9.68 GB (9.01 GiB) | tower 8,411,549,456 + visual 667,057,616 + aux 398,633,704 + scorer 199,316,800 |
| `shgao/rsi-jev-v6.1-vl-27b` | `d0dccc05` | `Qwen/Qwen3.8-27B` | 54,468,026,840 B = 54.47 GB (50.73 GiB) | tower 51,249,295,504 + aux 1,531,259,664 + visual 921,492,896 + scorer 765,629,768 |
| `shgao/rsi-jev-v6.0-vl-4b` | `a2534d8e` | `Qwen/Qwen3.5-4B-Base` | 9,676,732,504 B | 与 v6.1-VL 4B **尺寸逐个相同**（同架构），但 `tower`/`scorer`/`aux`/`calibration` 四个 blob 不同（§13.3） |
| `shgao/rsi-jev-v5.0-vl-3b` | `66eeaee4` | `Qwen/Qwen3.5-4B-Base` | 6,600,544,104 B = 6.60 GB | tower 5,733,994,760（4B 的前 20 层）+ visual + scorer |
| `shgao/rsi-jev-v4.0-vl-qwen3.5-2b` | `079ec5f4` | `Qwen/Qwen3.5-2B-Base` | 5,623,331,936 B | tower 5,493,097,440 + scorer 130,094,384 + calibration 140,112 |
| `shgao/rsi-jev-v3.0-qwen3.5-2b` | `c778b68d` | `Qwen/Qwen3.5-2B-Base` | 5,623,331,936 B | 同尺寸；含 `calibration.safetensors` |
| `shgao/rsi-jev-v2.1-qwen3.5-2b` | `e858e07a` | `Qwen/Qwen3.5-2B-Base` | 5,610,744,596 B | scorer 117,507,044 |
| `shgao/rsi-jev-v2.0-qwen3.5-2b` | `25a2417a` | `Qwen/Qwen3.5-2B-Base` | 5,610,744,596 B | 与 v2.1 **尺寸逐个相同、blob 全不同**（见下） |
| `shgao/rsi-jev-v1.0-qwen3.5-2b` | `8211ae3f` | `Qwen/Qwen3.5-2B-Base` | 5,610,604,484 B | **无 `calibration.safetensors`** |
| `shgao/rsi-jev-v1.0-qwen3.5-0.8b` | `f9248cac` | `Qwen/Qwen3.5-0.8B-Base` | 2,021,881,564 B | tower 1,992,487,680 + scorer 29,393,884 |
| `shgao/rsi-jev-v6.1-vl-4b-mlx-8bit` | `ec7d5e05` | `shgao/rsi-jev-v6.1-vl-4b`（relation=`quantized`） | 6,330,764,328 B | tower 5,065,581,280；**scorer / aux / visual / calibration 与父仓 blob 全同** |
| `shgao/rsi-jev-v6.1-vl-4b-mlx-4bit` | `73a6578d` | `shgao/rsi-jev-v6.1-vl-4b`（relation=`quantized`） | 3,874,346,292 B | tower 2,908,136,440；**scorer 与 aux 也减半**（199,318,248 / 99,659,060） |

基座（同为 Hub API）：

| 基座 | sha（前 8） | 参数量 | 体积 |
|---|---|---|---|
| `Qwen/Qwen3.5-4B-Base` | `1001bb4d` | 4,659,865,088（4.66B） | 9,319,828,056 B |
| `Qwen/Qwen3.8-27B` | `1d4bf0f2` | 27,781,427,952（27.78B） | 55,563,006,776 B |

**两处措辞要按 Hub 收口**：

1. **27B 的基座没有 `-Base` 后缀**，是 `Qwen/Qwen3.8-27B`；而 0.8B / 2B / 4B 三档都是 `-Base`。`versions/v6.1-vl-27b.md:35` 与 README 的措辞一致，但「基座都是 Qwen3.5 Base」这句话只在 4B 及以下成立。
2. **v5.0-VL-3B 的基座是 4B 的 Base**，不是某个 3B——它是「4B 的前 20 层」（`versions/v5.0-vl.md`、`README.md:131`）。Hub 上的 `base_model` 也确实写着 `Qwen/Qwen3.5-4B-Base`，而它的 `meta.json` 把参数写成 `decoder_layers_1_20: 2,231,283,520` + embeddings 635,699,200 + 头 49,828,866 + 视觉塔 333,514,240，`total: 3,250,325,826`。

**一处发布侧的指纹可以交叉验证**：每个 `meta.json` 里的 `base_revision` 就是基座在 Hub 上的 sha——v6.0-VL-4B 与 v6.1-VL-4B 都写 `1001bb4d826a52d1f399e183466143f4da7b741b`（= `Qwen/Qwen3.5-4B-Base`），27B 写 `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`（= `Qwen/Qwen3.8-27B`）。**两边独立给出同一个 sha。**

**容器算术：参数预算与 dtype 的独立闭合。** v6.1-VL 4B 的记录把参数拆成「32 层 3.57B + 词嵌入 0.64B + 视觉塔 0.33B + 主头 0.05B + 两个早退头 0.10B = **4.69B**」（`versions/v6.1-vl.md:17-20`）。这个拆分**逐项写在权重仓的 `meta.json` 里**，可以逐项相加：

```
v6.1-VL 4B  meta.json parameters:
  decoder_layers     3,570,052,096
  embeddings           635,699,200
  decision_head_main    49,828,866
  decision_heads_aux    99,657,732
  vision_tower         333,514,240
  total              4,688,752,134   ← 五项之和，逐位相等；≈ 记录的 4.69B
```

再用 Hub 的逐文件字节数**独立**验证一遍，顺手定出两个 dtype：

```
tower.safetensors  8,411,549,456 B ÷ 2 (bf16) = 4,205,774,728  ≈ 3,570,052,096 + 635,699,200 = 4,205,751,296
visual.safetensors   667,057,616 B ÷ 2 (bf16) =   333,528,808  ≈ 333,514,240
scorer.safetensors   199,316,800 B ÷ 4 (fp32) =    49,829,200  ≈ 49,828,866
aux_scorers.saf.     398,633,704 B ÷ 4 (fp32) =    99,658,426  ≈ 99,657,732
```

四个文件都只比 `meta.json` 的参数数多几百到一万字节（safetensors 的逐张量头与对齐），**÷2 或 ÷4 的选择没有第二个能同时对上四项**。同一套算术在 27B 上给出：tower 51,249,295,504 ÷ 2 = 25,624,647,752 ≈ 24,353,201,664 + 1,271,398,400 = 25,624,600,064；scorer 765,629,768 ÷ 4 = 191,407,442 = `decision_head_main`；aux ÷ 4 = 382,814,916 = `decision_heads_aux`；visual ÷ 2 = 460,746,448 ≈ 460,730,096。五项 `total` = **26,659,551,478**，与 `versions/v6.1-vl-27b.md:17` 的 *26.66B parameters, all of which the package ships and runs* 逐位相符。

反过来说：**塔是 bf16、头是 fp32** 这件事不是从文档读来的，是从字节数除出来的（`serve/release.py:503` 的 `model.scorer.to(torch.float32)  # never follows the tower down` 是它的代码侧对应）。

**一处没有核实**：`Qwen/Qwen3.5-4B-Base` 的 `params_total` 是 4,659,865,088，而 v6.1-VL 4B 的 tower + visual 只有 4,539,303,536——比基座少约 1.2 亿。差在哪一档（`lm_head` 是否 untied、基座仓是否把视觉塔算在别处）本文没有核出来：那要看基座的 `config.json` 与张量名，而权重文件一个都没下。这里只把差额记下来。

### 13.3 发布物里能机械核的五件东西

**(a) 逐仓的 `SHA256SUMS` 覆盖完整，而且非权重的那几行可以逐字节验。** v5.0-VL-3B、v6.0-VL-4B、v6.1-VL-4B 与两个 MLX 构建各带一份清单（三代 15 行、两个 MLX 构建各 16 行，多出的那一行也是它多出来的那个 `mlx.json`），27B 带的是一份同样形态的 `md5.txt`（14 行，BSD 风格带 `*`）。把清单里的文件名与 Hub 给出的仓内文件集合对起来：**每个仓都是「除 `.gitattributes` 与清单自身之外的全部文件」，一个不多一个不少**。

而且这份覆盖**不只是计数**：15 行里有 **10 行是非权重文件**（`README.md`、`calibration.json`、`config.json`、`merges.txt`、`meta.json`、`preprocessor_config.json`、`tokenizer.json`、`tokenizer_config.json`、`video_preprocessor_config.json`、`vocab.json`），它们全都小到可以直接取下来重算。本文对**五个带清单的仓全部做了这件事**：`shasum -a 256 -c SHA256SUMS` 在它们的全部非权重条目上**逐个 `OK`**——三个 fp32 仓各 10/15 行（差的 5 行恰好是那五个权重文件，本文没下），两个 MLX 构建各 11/16 行（多一个 `mlx.json`，差的 5 行同样是权重）。所以「这份清单记的是这些文件的这些字节」**不是从清单自述推的**，是在能取到的那一部分上验过的。

顺带一处旁证：清单里 `meta.json` 那一行的 sha256 与本文单独算的**逐位相同**（v6.0-VL-4B `ada8c6e4…`、v6.1-VL-4B `583e873e…`、v5.0-VL-3B `1a820f62…`），而 `583e873e…` 同时出现在 v6.1-VL-4B 与它的两个 MLX 构建上——**三个仓的 `meta.json` 是同一串字节**（§11.5 那张表就建立在这件事上）。

**(b) v6.0 与 v6.1 的 `SHA256SUMS` 逐行对照，正好划出「变了什么」。** 两份清单的**行集合完全相同**，而逐行比对给出干净的分界：

| 两代之间 hash **相同**（8 项） | 两代之间 hash **不同**（7 项） |
|---|---|
| `config.json`、`merges.txt`、`vocab.json`、`tokenizer.json`、`tokenizer_config.json`、`preprocessor_config.json`、`video_preprocessor_config.json`、**`visual.safetensors`** | `tower.safetensors`、`scorer.safetensors`、`aux_scorers.safetensors`、`calibration.safetensors`、`calibration.json`、`meta.json`、`README.md` |

这与「v6.1-VL = v6.0-VL 与第二个微调的平均」对得上：**视觉塔逐字节没动**（两个成员共用基座那个冻结的 ViT），塔、三个头、校准全部重算。同一份对照也解释了一个容易看错的点——**两代的四个权重文件尺寸逐个相同**（tower 8,411,549,456、scorer 199,316,800、aux 398,633,704、visual 667,057,616），所以**只比大小会把两代判成同一份权重**。

顺着这条线再多核一层：`visual.safetensors` 的 `82f775bb…` 在 **v5.0-VL-3B、v6.0-VL-4B、v6.1-VL-4B 与两个 MLX 构建里是同一个 hash**，`merges.txt` 的 `a9d356d7…` 与 `tokenizer.json` 的 `fe000e3e…` 也是。也就是说 4B 那一条线从 v5.0-VL 起用的**就是同一份冻结视觉塔的同一串字节**，`versions/v6.1-vl.md` 里「视觉塔 0.33B」那句话在这四代之间指的是同一批字节——而 `meta.json` 里每个包都单独记了一份 `parameters.vision_tower`（v5.0-VL-3B 与 v6.x-4B 都写 333,514,240）。

**(c) 各代 `calibration.json` 里的 `T` 就是 `exp(logT)`。** 三个发布了 `temp` 的仓、九个出口温度全部逐位复现：

| 仓 | 出口 | `logT` | `exp(logT)` | 文件里的 `T` |
|---|---|---|---|---|
| v6.1-VL-4B | 16 | -0.14 | 0.869358 | 0.8694 |
| v6.1-VL-4B | 20 | -0.47 | 0.625002 | 0.6250 |
| v6.1-VL-4B | 32（主） | 0.02 | 1.020201 | 1.0202 |
| v6.0-VL-4B | 16 | 0.63 | 1.877611 | 1.8776 |
| v6.0-VL-4B | 20 | 0.68 | 1.973877 | 1.9739 |
| v6.0-VL-4B | 32（主） | 0.80 | 2.225541 | 2.2255 |
| v6.1-VL-27B | 48 | 0.01 | 1.010050 | 1.0101 |
| v6.1-VL-27B | 56 | 0.02 | 1.020201 | 1.0202 |
| v6.1-VL-27B | 64（主） | 0.05 | 1.051271 | 1.0513 |

（文件里的 `T` 是四舍五入到四位，所以最后一位有时差 1。**这个 `T` 块是给人看的**——服务读的是 `cal_logT` 那个 fp32 标量本身。）

**(d) 服务代码是随权重一起发过的，只是后来停了。** v1.0 到 v4.0-VL 的仓里都有一个 `code/` 目录，但它**逐代变厚，而且只有最后一代装了 HTTP 层**：

| 仓 | `code/` 文件数 | 内容 |
|---|---|---|
| v1.0-0.8B / v1.0-2B | 11 | 只有 `rsijev/`（arch / encode / contract / data / evaluate / fit / metrics / targets / train）+ `load_release.py` |
| v2.0 / v2.1 | 14 | 加上 `calibrate.py` 与 `targets_suite.py` |
| v3.0 | 15 | 再加上 `rl2.py` |
| v4.0-VL | **35** | 再加上 `vision.py` / `vision_fit.py`，**以及一个完整的 `serve/`（19 个文件：`app.py` / `infer.py` / `wire.py` / `decider.py` / `batcher.py` / `runtime.py` / `ui.html` / `examples.json` …）** |

**v5.0-VL 起 `code/` 归零**，改成一份 `SHA256SUMS`。所以「这份权重当时是被哪一份服务代码读的」这件事，**只有 v4.0-VL 那一代能整包取出来**；v1.0–v3.0 三代只有离线评估那一半（`evaluate.py` / `metrics.py` / `targets*.py`），没有请求解析、没有路由、没有那个 `output_tokens = len(questions)`。

**(e) 一处随代际换过名字的护栏。** v1.0–v2.1 的 `meta.json` 里，`release.truncation` 那个字段写的是「服务时怎么截」；v5.0-VL 写 *none when served: requests past 32,768 tokens are refused (422)*（与 `serve/server.py:88-95` 一致），而 v6.0/v6.1 写的是 *a state over max_length is cut from the LEFT (rsijev/encode.py), so a leading query is lost*——**同一个字段名，v6.x 记的是训练/评测口径，v5.0 记的是服务口径**。两者都对，但**读这个字段的人会得到关于服务行为的相反结论**。`meta.json` 里另一处同类：`tapped_layer` 由 `scripts/run_arm_lib.py:346` 在**训练**时写入，v5.0-VL-3B 写着 `31`（全深 32 层的那次训练），而服务实际只跑前 20 层（`spec.arch_extra.exit_layer: 20`）——**`serve/` 全仓对 `tapped_layer` 零读点**。

---

## §14 亲手跑一遍

三条都不需要 GPU（前两条连权重都不需要）：

```bash
# 1. 契约层：无 torch、无 GPU、无权重
pytest tests/ -m "not slow"          # wire contract + precision guards, <1s

# 2. 服务层：一个假 scorer 跑完整路由与错误信封
pytest tests/test_serve.py -q

# 3. 有权重时（本文没跑）：服务面前向与评估前向逐位一致
CUDA_VISIBLE_DEVICES="" pytest tests/test_serve_parity.py -q
```

第三条是这条路线最该跑的一条：`tests/test_serve_parity.py` 是**保证服务前向与被打分的前向逐位相同**的那道闸（`rsijev/README.md:155-159`）。它对这条路线特别重要，因为「服务面 `truncate="none"`、benchmark 脚本 `"left"`」意味着两条路**本来就是两个编码器配置**，能对齐的只有同一配置下的那一段。

上服务：

```bash
pip install "rsi-jev[fast,vision] @ git+https://github.com/Shanghua-Gao/RSI-Jev"
rsi-jev serve v6.1-vl-4b --effort auto --port 8000
curl localhost:8000/v1/systemone -H 'Content-Type: application/json' -d '{
  "model": "jev-latest",
  "state": "I was charged twice. Please refund.",
  "questions": {"refund": {"type": "noul", "instructions": "Does the user request a refund?"}}
}'
```

`--effort auto` 只对有多出口的 release 有效；单出口的 release 上 `low/medium/auto` 会被 **422** 拒掉（`serve/effort.py:120-127`），只有 `high` 与不设可用。

---

## 实例走查：`serve/README.md` 的三题请求，在两代权重上

来源：`serve/README.md` 的两段 `<!-- response: vX.Y -->`，标注为 **RSI-Jev-v3.0-2B**（`README:42-45`：*bf16 on a GB10 with fla, probabilities rounded to three places*）与 **RSI-Jev-v1.0-2B**（`README:81`）。同一个请求（3 题：`refund` noul、`department` choice 2 选项、`urgency` score 3 档），`usage.input_tokens` 都是 136、`output_tokens` 都是 3。这些是**上游自己在真权重上跑出来并提交进仓的**数字，本文只在其上重算导出量——**一个前向都没有跑**。

### v1.0-2B（无校准）

```json
"refund":     {"noul": 0.983}
"department": {"choice":"billing","probabilities":{"billing":0.979,"technical":0.021},"confidence":0.958}
"urgency":    {"score":1.197,"probabilities":{"0":0.068,"1":0.667,"2":0.265},"confidence":0.501}
```

逐项重算：

| 导出量 | 公式（出处） | 用提交的概率算 | 文档值 | |
|---|---|---|---|---|
| `department.confidence` | `(K·p_max−1)/(K−1)`，K=2（`wire.py:187`） | (2×0.979−1)/1 = **0.958** | 0.958 | 逐位相同 |
| `urgency.score` | `Σ i·p_i`（`contract.py:84`） | 0×0.068+1×0.667+2×0.265 = **1.197** | 1.197 | 逐位相同 |
| `urgency.confidence` | 同上，K=3 | (3×0.667−1)/2 = 0.5005 | 0.501 | 差半个末位 |

最后一行那 0.0005 的差**不是矛盾**：文档里的概率是**舍到三位**之后再印的，而 confidence 是用**未舍**的 p_max 算的。要让 `round(·,3)` 落在 0.501 上，真实 p_max 需 ≥ 0.6674 左右——而 0.6674 舍到三位仍然印成 0.667。**这一格量的是「文档自身三位小数不足以反推 confidence」，不是实现不一致。**

### v3.0-2B（`oof_head_scorefloor` 校准）

```json
"department": {"choice":"billing","probabilities":{"billing":1.000,"technical":0.000},"confidence":0.999}
"urgency":    {"score":1.081,"probabilities":{"0":0.178,"1":0.563,"2":0.259},"confidence":0.345}
```

| 导出量 | 用提交的概率算 | 文档值 | |
|---|---|---|---|
| `urgency.score` | 0×0.178+1×0.563+2×0.259 = **1.081** | 1.081 | 逐位相同 |
| `department.confidence` | (2×1.000−1)/1 = 1.0（用舍后值） | 0.999 | 需真实 p_max ∈ [0.9995, 0.99995) |
| `urgency.confidence` | (3×0.563−1)/2 = 0.3445 | 0.345 | 需真实 p_max ≥ ~0.5634 |

**两代之间可以直接读出来的东西**，只用这两段 committed 文本：

- `confidence` **不是 `p_max`**。v3.0 的 `department` 是最干净的一例：概率印成 `1.000 / 0.000`，而 `confidence` 是 `0.999`——若它等于 p_max，两者会是同一个数。K=2 时 `(2p−1)` 恰好把「与 1 的距离」放大一倍。
- `score` 是**零基下标的期望**，不是档位。1.081 落在 1（"Urgent"）附近而不是 1 与 2 的中点附近——所以它是「期望下标」，`legend` 才说得出人话。
- **校准确实改了数值**：同一题同一权重族，v1.0 的 `urgency` 给 (0.068, 0.667, 0.265)、v3.0 给 (0.178, 0.563, 0.259)——峰值从 0.667 压到 0.563，argmax 不变（都是下标 1）。这与 `arch.py:929-930` 那句 *Argmax-preserving* 一致，也解释了为什么 `serve/README.md:86-92` 要专门提醒：*calibration divides each question's logits by one positive number … a threshold tuned against one is not the same threshold against the other.*
- 两份响应的 `output_tokens` 都是 3，而 `answers` 各有 3 个键——`usage.output_tokens == len(questions)`（`app.py:162`），`tests/test_documented_example.py:67-68` 把它钉成断言。

### 另两处可以机械核的 committed 产物

- `serve/examples.json`（113,904 B）是 playground 的 11 条示例，`serve/demo.py` 直接喂进服务路径。它随 `code/serve/` 发布过**一次**，在 **v4.0-vl-qwen3.5-2b** 那个仓里（`code/` 共 35 个文件，是唯一一个带了完整 `serve/` 包的：`app.py`、`infer.py`、`wire.py`、`decider.py`、`ui.html`、`examples.json` … 19 个）。前一代 v3.0 的 `code/` 只有 15 个文件、**全是 `rsijev/` 加一个 `load_release.py`，没有任何 HTTP 层**。所以「这份权重当时是被哪一份服务代码读的」这件事，在 v4.0-VL 那一代是**可以逐文件取出来的**，在 v1.0–v3.0 三代只有离线评估代码（`evaluate.py` / `metrics.py` / `targets*.py`），在 v5.0-VL 之后连 `code/` 都没有了（§13.3d）。
- `data/v6.1-vl_recipe/5-exit-policy.json` 的 `auto.result` 里，`A/B` 两半的分层数字、6 条约束（C1–C3 × 两半）与最终 `CONFIRMED: true` 是可对读的；同一个文件的 `single` 块写着 `CONFIRMED: false` 与 `fallback: true`——**同一个脚本在同一次运行里给出一个已确认与一个未确认的阈值**，而发布出去的是那个未确认的（§8）。

---

## 未核实清单

- **所有前向数值**。权重一个都没下（4B 档 9.68 GB、27B 档 54.47 GB），所以任何概率、准确率、ECE 都不给；`docs/inference.md` 那张延迟表的每一格与 `versions/*.md` 的每个分数本文只引用、不复现。
- **`reference/openjev-sglang` 不在仓里**。README 里「与参考实现逐条相同」的清单（`serve/README.md:165-184`）没有可对照的第二份代码，`tests/test_serve.py:3-4` 引的路径解析不了。本文只能记下这份清单**由仓内断言自持**。
- **`Qwen/Qwen3.5-4B-Base` 与 v6.1-VL 4B 的 tower+visual 参数差（约 1.2 亿）**，见 §13.2 末。
- **权重仓里那份 `code/` 与仓内 HEAD 的差异**。v1.0–v4.0-VL 的包里各带一份当时的代码（只有 v4.0-VL 那个 35 文件的包含 `serve/`），本文核了**每个包里有哪些文件名**，没有逐个与当前 HEAD 做内容比对，所以「当时那份服务代码与现在的差多少」这一层没做。
- **同一份 `SHA256SUMS` 里的权重行**。本文核了清单**覆盖完整**（§13.3a）、**10 个非权重条目逐字节通过**（§13.3a 后半）与**逐行异同**（§13.3b），但剩下 5 行要下载 9.68 GB 的权重才算得出，没有做五个带清单的仓都已逐仓跑过 `shasum -a 256 -c`，通过的是**非权重的那 10（或 11）行**；27B 那份 `md5.txt` 的 14 行**没有验**（它带 `*` 的 BSD 形态可以 `md5 -c`，本文没做）。
- **顺序偏置的翻转率**：仓里有 `reversed` 诊断臂与 `eval_option_orders` 声明（所有代都是 `["canonical","reversed"]`），但提交的产物里没有翻转率数字。
- **ECE 分箱在同一仓里不一致**：`rsijev/metrics.py:102` 的 `ece(bins=10)` 与 `rsijev/calibrate.py:176` / `rsijev/exit_policy.py:290` 的 15 箱不是同一把尺。本文没有核出发布数字各自用的是哪一把。
- **`meta.json` 里两个字段的「哪一年代改了约定」**。§13.3(e) 记了 `release.truncation` 与 `tapped_layer` 两处随代际变义的字段，但这两个字段**没有任何代码读**（`serve/` 只从 `meta["release"]` 取 `vision` / `max_length_text` / `adaptive`），所以本文只能记下「读它的人会得到相反结论」，说不出它们各自还有没有下游消费者。

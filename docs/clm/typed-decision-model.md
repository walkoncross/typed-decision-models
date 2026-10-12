# CLM 是如何把一次决策读成「状态嵌入 · 候选嵌入」的点积的

> 代码库：<https://github.com/Contrastive-LM/CLM>
> 对应 `README.md` 第 20–23 行：*Contrastive Language Models (CLMs) are a new class of **System One
> model** trained with a **contrastive learning** objective that connects **states and actions**.
> This repo serves **CLM-8B** behind a TypeSafe-compatible API.*
> 本文按 `src/clm/`（推理）与 `train/`（训练）的实际代码逐环节说明这句话的实现方式；
> 文中出现的文件名与行号（`engine.py:134`、`schema.py:104`、`finetune.py:177`、`cache.py:133` 等）
> 均相对该仓库根目录。

与前面八篇最大的不同：**这条线上没有「读位置」这件事可以标定，也没有词表可以掩码。**
决策不是从一次前向的 logits 里「读」出来的，而是**两个嵌入向量的一次点积**；
于是「候选」可以离线算好并长期缓存，「状态」才需要现算。
CLM 把 RYOTIDE 那条读法里的三个环节（读哪个位置、掩码到标签集、重归一化）**整个换掉**，
换成了一套缓存与算术：缓存要按「文本 + 头 + 头的代次」做键，而概率仍然要落到请求声明的选项集上。

先说结论式的三句话，后面逐条给代码位置：

1. 「typed」不来自分词器、也不是掩码出来的：候选集**就是请求里那一组文本**，
   每条候选文本被整段编码成一个向量（§2）。没有单 token 约束，也没有选项数上限。
2. 决策本身**没有前向**：两个 head 是 2 层 MLP，`Engine.answer` 只做 `K+1` 次编码（去重后）
   与 `K` 次点积，softmax 在 host 上用 Python 浮点算（§4、§5）。
3. 这条线上**没有任何标定链**：没有温度拟合、没有 ECE、没有发布配方。
   唯一叫 `temperature` 的东西是**每个请求可以传的一个入参**，默认 1.0，而且**响应里不回声**（§10）。

## 本文的事实来源

| 类别 | 来源 | 本文中的标记 |
|---|---|---|
| 代码引用 | `src/clm/**`（推理与服务）、`train/**`（训练）、`tools/`、`evaluation/`、`preprocessing/`、`examples/t_rex/**` | `文件:行` |
| 可实跑得到的产物 | `examples/t_rex/results/*.json` 是 **committed 的跑分记录**（纯 JSON、容量小）；`README.md` 里的示例数字与竞技场日志行是**已打印的产物** | 标注「实测」，并注明是「在 committed 产物上重算」 |
| 公式与常量 | `README.md`、`docs/FINETUNING.md` | `README:行` |
| 未跑 | **本机没有 GPU、没有下载权重、没有起过 `clm-serve`** | 照例不给任何前向产生的概率数值 |

本文所有「实测」都只有两类动作：**读 committed 的 JSON 用标准库算术重算**（§14.1）、
**对已打印的产物做手算核对**（§14.2、§14.3）。没有运行上游的任何程序——没有 `import clm`、
没有跑 `train/finetune.py`、没有起服务、没有下载 `CLM_v0.1-8B.pt`。命令在 §13、§14 原样列出。

---

## 0. 与 RYOTIDE 的分野

对照对象是 [`docs/ryotide/typed-decision-model.md`](../ryotide/typed-decision-model.md)——
本文凡称「RYOTIDE §N」均指该篇的节号。

| | RYOTIDE | CLM |
|---|---|---|
| 是什么 | 一个读法：把冻结 LLM 读成分类器 | 一个读法 + 一套缓存：把决策读成两个嵌入的点积 |
| 权重 | 公开 LLM，全程 frozen，**零新增参数** | 公开 Qwen3-8B **编码器**（frozen）+ **两个自训的 20M 投影头**（`Contrastive-LM/CLM-v0.1-8B`，约 75 MB） |
| 是否训练 | 否 | 是——但训的不是「怎么读」，是**两个投影**；训练目标就是推理时那一个式子（`finetune.py:177` 与 `engine.py:134`） |
| 决策的运算 | 一次前向 → 取标记 token 的 logits | `head(state) · head(candidate)`，**没有参与决策的前向**（§4） |
| 「typed」怎么保证 | 标签必须落在**单 token**，否则退化成字母标记 | 请求声明的 `criteria` / `levels`；**没有词表，所以没有掩码这一步**（§2） |
| 读位置 | **运行期标定**：扫 9 前缀 × 2 表面形式 | **不在本仓**：编码器的池化位置由 vLLM 决定，仓里只有 `--runner pooling` 一句；决策侧根本没有「位置」这个概念（§3） |
| 候选数上限 | 260（两 token 两步读；>26 另有 `grouped`） | **无硬上限**（同一份 harness 里 CLM 的 `max_options` 是 `None`、Jev 是 255，`common.py:42,45`） |
| 顺序偏置 | 多 order 共享前缀 fork，概率平均 | **不存在该变量**：逐候选独立编码、逐候选点积，候选之间零交互（§6） |
| 置信度 | 读出来的，只有 T 能改 | 训出来的（两个头），但**没有标定链**：T 是调用方的入参、不随权重发布、不回声（§10） |
| 多问题共享 state | `run_many()`，公共前缀预填充一次 | **不做**：每题一个 `state_text`；被共享的是**候选向量**，而且跨请求（§7、§8） |
| 成本模型 | 每题一次前向 | `K+1` 次编码（去重后）+ 一个预留的显存竞技场（§8） |

一句话：**RYOTIDE 回答「怎么把 LLM 读成分类器」，这一路回答「怎么把分类器拆成两个可以各自缓存的嵌入」。**
共享的底线还是那一条——一次前向、不生成任何 token、输出封闭在已知选项集上（§11），
但「一次前向」在这里要改写成「一次编码」，因为决策本身已经没有前向了。

---

## 1. 端到端数据流

一次 `POST /v1/systemone`，从 HTTP 到响应：

```
  client ──► FastAPI  clm-serve（:8700）                        server.py:96-119
             auth()  401                                           :78-80
             body 必须是 {state, model, questions}                 :103-104
             temperature = float(body.get("temperature", 1.0))      :106
             run_in_executor(engine.answer, ...)                    :111-112  ← 不阻塞事件循环
               │
               ▼  Engine.answer(state, questions, model, T)         engine.py:104
   ┌──────────────────────────────────────────────────────────────────────┐
   │ ① build_pairs(state, questions)                      schema.py:104   │
   │      state_text = to_text(state) + "\n\n" + to_text(q.instructions)  │
   │                                                   schema.py:62-65    │
   │      keys, texts = candidates(q)                    schema.py:75-101 │
   │        choice: keys = list(criteria)，每条候选文本 = 描述或键        │
   │        score : keys = "0","1",...   候选文本 = 各档位文本            │
   │        noul  : keys = ("false","true")，候选文本 = 渲染好的句子      │
   │      → 每题一份 (state_text, keys, texts)                            │
   │ ② states = 各题的 state_text；cands = 所有题的候选文本拼一列         │
   │                                                      engine.py:119-120│
   │ ③ 取向量（两层缓存：显存竞技场 → 进程内 LRU → 编码器）               │
   │      arena.get(namespace, dim, texts, compute)        cache.py:128   │
   │        compute → Embedder.embed(missing)              engine.py:76-80│
   │          └─ POST /v1/embeddings {input: [text...], encoding_format:  │
   │             "base64", truncate_prompt_tokens: 2048}  embedder.py:40-57
   │             → l2(v)：编码器自己的向量也在 host 侧归一化  embedder.py:56
   │        → head.project_*：MLP + F.normalize            heads.py:104-115│
   │ ④ 逐题：cos = za[k:k+K] @ zq[i]                       engine.py:134  │
   │    answer_from_logits(q, keys, (scale*cos/T).tolist())          :136 │
   │      softmax(...)                                    schema.py:115-119│
   │      answer_from_probs(...)                          schema.py:131-145│
   │ ⑤ {"model","answers","usage":{"billing_units",
   │     "input_tokens", "output_tokens": 0}}             engine.py:137-138│
   └──────────────────────────────────────────────────────────────────────┘
               │
               ▼  JSONResponse + X-CLM-Latency-Ms           server.py:119
```

进程边界是这条路线的一个结构事实（`README.md:217-223` 的框图）：

```
browser ──► clm-serve  (heads, torch, :8700)   GET / (playground)
client  ──►                                    POST /v1/systemone · /v1/rank · /v1/models · /health
               │       state head + action head（20M 参数，hot-reload），两层向量缓存
               ▼
          vLLM Qwen3-8B pooling server（GPU, :8090）  /v1/embeddings
```

**唯一跑模型的是右边那个进程，而它不是这个仓库。** 本仓跑的是两个 MLP（`heads.py:41-65`）。
这条分界在后面的每一节都会再出现：读法里凡是与「模型怎么读」有关的部分，
CLM 都推给了编码器服务，本仓只剩「拿向量算什么」。

另外两条入口，与 HTTP 平行：

- **进程内**：`Engine.answer(...)` 直接可用，不走服务（`engine.py:104`，`README.md:104-112`）；
- **`/v1/rank`**：同一个原语的裸形式。它把候选包装成一个 `choice` 问题
  （键是下标、值是候选原文），再把 `probabilities` 排序（`engine.py:140-151`）。
  注意客户端与引擎的参数顺序不同：`CLMClient.rank(context, question, answers)`
  （`client.py:183`）对 `Engine.rank(state, candidates, instructions)`（`engine.py:140`）——
  服务端是按下标位置转发的（`server.py:144`）。

---

## 2. 「typed」从哪来：请求声明的 `criteria`，加一个不存在的词表

RYOTIDE 的类型化是**借来的**：它要求标签能落到分词器的单个 token 上，落不到就退化成字母标记。
CLM 没有这条约束，也没有它带来的麻烦——被评分的从来不是标签文本，而是**为它算出的一个向量**。

### 2.1 题型与封闭性：三个题型，一个字段

`QUESTION_TYPES = ("noul", "choice", "score")`（`schema.py:23`），
校验只有一处：

```python
def _norm_question(q: dict) -> dict:
    t = q.get("type")
    if t not in QUESTION_TYPES:
        raise ValueError(f"unknown question type {t!r}; expected one of {QUESTION_TYPES}")
    return q
```

`schema.py:68-72`——**只查 `type`，其余字段一律不查**。
多传的字段被静默忽略（与 vLLM Semantic Router 那条「多一个字段即报错」正好相反）。
题面与选项分别由 `instructions` 与 `criteria` 承担（`schema.py:75-101`）：

| 题型 | `criteria` | 选项键 | 候选文本（送进 action head 的东西） |
|---|---|---|---|
| `choice` | 非空 map（否则 `ValueError`，`schema.py:80-81`） | `list(crit)`，**保持请求给的顺序** | 该选项的描述；描述为空则退回**键本身**（`schema.py:86`） |
| `score` | 有序 list，≥2 项（否则 `ValueError`，`schema.py:89-90`） | `"0","1",...` | 档位文本原文，**没有前缀**（`schema.py:92`） |
| `noul` | 可选 `{"true":…, "false":…}` | `("false","true")` | `f"{k}: {…}"` 渲染句（`schema.py:96-100`） |

三处细节值得单独记下来，因为它们决定了「typed」到底封闭在什么上：

- **`noul` 的候选顺序是 `false` 在前**（`NOUL_KEYS = ("false", "true")`，`schema.py:24`），
  而答案取的是 `dist["true"]`（`schema.py:137`）。不传描述时，两条候选文本被拼成
  `"false: No. This is false: <instructions>"` 与 `"true: Yes. This is true: <instructions>"`
  （`schema.py:99`）——**键名被前缀进候选文本，然后整句送进编码器**。
  这是这条线上最像「把类型信息写进文本」的一处，而它写的是一句带冗余前缀的话。
- **`score` 的档位序号只存在于键里**，候选文本本身不含序号（`schema.py:92`）。
  action head 对顺序不敏感（§6），所以「第 2 档」的身份完全靠它的文本；
  两档文本相同就等于两个无法区分的选项。
- **`choice` 的候选文本可以是键**（描述为空时）。这是唯一一处「选项名进入编码器」的路径：
  API 文档把这条写成了规格——*each option is embedded as its description, or its key when
  the description is empty*（`README.md:394`）。

### 2.2 封闭性不是被掩码保证的，是没有别的东西可算

`answer_from_probs` 用 `zip(keys, probs)` 组装分布（`schema.py:135`）：

```python
probs = [float(p) for p in probs]
dist = {k: p for k, p in zip(keys, probs)}
```

`keys` 来自 `candidates(q)`，`probs` 是那条公式在 K 个候选上的 softmax。
**没有任何一条路径能产生一个不在 `keys` 里的标签**——不是因为它被掩掉了，
而是因为整条链上从来没有出现过词表：没有 logits，没有 vocab 维度，没有 token id。
RYOTIDE 的 #1/#2 两条不变量（标记上的质量、前缀与表面形式成对扫）在这里连**对象都不存在**（§11）。

代价换到了别处：

- **每个选项都要编码一次**。K 个选项 = K 段文本要过编码器（首次；之后靠缓存，§8）。
  所以「选项数」在这里不是「一步读几个 logit」，而是「要准备几个向量」。
- **没有选项数上限**。`common.py` 里两个端点的能力表写得很直白：CLM 的 `max_options` 是
  `None`、Jev 是 `255`（`common.py:42,45`），而 `Judge.choose` 只在
  `len(keys) > cap` 时才走分块两级近似（`common.py:180-199`）——对 CLM 这条路，那个分块分支
  **永远不进**。`Engine.answer` 里也确实没有对 `len(texts)` 的任何检查（`engine.py:133-136`）。
  这是八篇里唯一一条「候选数不受接口约定限制」的路线（其上限由显存与 batch 决定，不由契约决定）。
- **重复候选文本无法区分**。两条文本相同 ⇒ 编码器给出同一个向量 ⇒ 同一个投影 ⇒
  `engine.py:134` 的两个点积逐位相同 ⇒ 两个概率严格相等，argmax 落到请求顺序里的第一个
  （`schema.py:139` 的 `max(range(len(probs)), key=…)`）。
  全仓**没有任何检查**发现这件事——对比 Laya 的 `options_distinct` / `tokens_per_option`
  那两个诊断量（其 §2），这里少了一整类守卫。

---

## 3. 读位置：它被拆成两个问题，其中一个不在本仓

RYOTIDE 把「读哪里」当成一个可以标定的自由度（扫 9 前缀 × 2 表面形式，取 marker mass 最高的）。
CLM 里这个词要拆成两问，答案完全不一样。

### 3.1 编码器池化取哪个 token：由 vLLM 决定，本仓不置一词

训练侧的配方是明确的：**取尾部**。

```python
def state_ids(self, state) -> list[int]:
    if isinstance(state, str):
        return self.text_ids(state, keep="tail")
    ids = self._flatten(self.tok.apply_chat_template(state, tokenize=True, add_generation_prompt=False))
    return ids[-self.cap:]
```

`embed_utils.py:45-49`，`keep="tail"` → `ids[-self.cap:]`（`embed_utils.py:43`），
`cap = max_len - 1`（`embed_utils.py:28`）。`choice` 任务对**所有**文本都走尾部保留：

```python
cache.add(todo, backend.embed([recipe.text_ids(t, keep="tail") for t in todo]))
```

`finetune.py:476`——state 与候选一视同仁。

而服务侧的池化方式是**交给 vLLM 的一句话**：

```bash
vllm serve Qwen/Qwen3-8B --served-model-name qwen3-8b --runner pooling --max-model-len 2048 --port 8090 &
```

`README.md:59`；`serve_qwen3_8b.sh:18-23` 同样是 `--runner pooling` 加若干内存参数。
**没有一处指定池化取哪个位置**——它是 vLLM 对该模型自己的默认值。
本仓对这个默认值的全部记录只有散文：`heads.py:11` 与 `embedder.py:1-8` 的 docstring 说
*l2-normalised*、*last-token pooling*，`serve_qwen3_8b.sh:4` 的注释说
*LAST-token pooling + prefix cache, same as the precompute*。

于是这条线上「读位置」的地位是：**由另一个进程的默认值决定，本仓既不设定也不断言，
运行期更没有任何诊断量**。vLLM 对 `Qwen/Qwen3-8B` 在 `--runner pooling` 下默认取哪个位置，
本文**未核实**（本机无 GPU、无权重）。可核实的只有三件事：

1. 训练配方取尾部（`embed_utils.py:43,49`、`finetune.py:476`）；
2. 服务侧不指定（`serve_qwen3_8b.sh:18`、`README.md:59`）；
3. 检查点里**不记录**编码器身份——见下。

### 3.2 检查点不记录它配的是哪个编码器（`clm` 路径）

`clm` 任务写进 `cfg` 的字段是：

```python
"cfg": {"width": width, "depth": depth, "projection_dim": args.proj,
        "hidden_size": hidden, **head_options, "init_ckpt": args.init_ckpt,
        "sampler": args.sampler, "batch": args.batch, "lr": lr,
        "train_dir": emb_dir}
```

`finetune.py:315-318`——有 `hidden_size`、有 `train_dir`，**没有 `embed_model`**。
`choice` 任务的 `cfg` 里有（`finetune.py:590-592` 含 `embed_model` 与 `max_len`）。
所以一个 `clm` 头的编码器身份只能顺着 `train_dir/metadata.json` 里的 `model_name`
去反查（`finetune.py:144-146` 写入的那个字段），检查点本身说不出自己配谁。

而这条约束被上游写成一句话放在文档里：

> Any head in the same checkpoint format … can be served with `--ckpt`;
> **a head only makes sense with the encoder and pooling it was trained against.**

`README.md:302-306`。**这句话在代码里没有任何对应物**：`HeadPair._load` 只校验
`cfg` 里那几个架构字段（`heads.py:81-95`），`Engine.__init__` 不比较编码器，
`/health` 只报 `embedder: healthy()`（`server.py:84`）——而 `healthy()` 只打一次
`GET /v1/models`（`embedder.py:82-87`）。**换一个池化方式不同的编码器，
服务会照常起来、照常返回格式正确、数值看似合理的分布。**

### 3.3 决策侧没有「位置」这个概念

这一半反而完全显式：决策读的是 `K+1` 个向量，没有坐标可谈。

```python
cos = za[k:k + len(texts)] @ zq[i]
k += len(texts)
answers[qid] = answer_from_logits(questions[qid], keys, (scale * cos / temperature).tolist())
```

`engine.py:134-136`。`zq[i]` 是第 `i` 题的状态投影，`za[k:k+K]` 是那题 K 个候选的投影，
`@` 是点积。**没有 index、没有 gather、没有「第几个 token」**——
RYOTIDE 的读位置标定、SemIf-OpenJev 的槽位下标、Semantic Router 的 `markers`/`endpoints`
在这条线上都没有对应物。

顺带记一处截断口径：服务侧把截断委托给 vLLM 的 `truncate_prompt_tokens`
（`embedder.py:41-43`，默认 2048，`server.py:173`），而训练侧的 `cap` 是 `max_len - 1 = 2047`
（`embed_utils.py:28,43`）。差一个 token，只在边界上可见；更要紧的是**截断保留哪一端**：
训练侧显式保尾，服务侧这个参数由 vLLM 决定侧面，本仓没有配置它，也没有记录它。**未核实**。

---

## 4. 一次前向：决策时没有前向，只有 `K+1` 次编码与 `K` 次点积

RYOTIDE 的 §4 是「一次前向只读最后一个位置」。这一节在这里要换标题，
因为**参与决策的部分根本没有前向**。

### 4.1 两个 head 是 2 层 MLP

```python
class Head(nn.Module):
    def __init__(self):
        super().__init__()
        self.inp = nn.Linear(hidden, width)
        self.hidden = nn.ModuleList(nn.Linear(width, width) for _ in range(depth - 2))
        self.norms = nn.ModuleList((nn.LayerNorm(width) if layernorm else nn.Identity())
                                   for _ in range(depth - 2))
        self.out = nn.Linear(width, proj)
        self.act = act()
        self.residual = residual

    def forward(self, x):
        x = self.act(self.inp(x))
        for lin, nrm in zip(self.hidden, self.norms):
            h = self.act(nrm(lin(x)))
            x = x + h if self.residual else h
        return self.out(x)
```

`heads.py:47-65`。`HIDDEN = 4096`、`PROJ_DIM = 512`（`heads.py:21-22`），
默认 `width=1536, depth=3`（`finetune.py:259-260,496`）——两个 20M 量级的 MLP。
前向只在 `torch.no_grad()` 里跑一次（`heads.py:104-109`），出来就 L2 归一化：

```python
def _project(self, head, x: np.ndarray):
    """-> [n, proj] L2-normalised projections, left on ``self.device``."""
    import torch
    self.ensure()
    with torch.no_grad():
        return torch.nn.functional.normalize(head(torch.from_numpy(x).to(self.device)), dim=-1)
```

`heads.py:104-109`。**归一化的是投影，不是编码器输出**——编码器那边另有一次 `l2()`
（`embedder.py:25,56`），所以从文本到点积一共经过两次归一化，起作用的这次在这里。

### 4.2 决策的公式，就是训练的公式

推理：`scale * cos / temperature`（`engine.py:136`），其中
`scale = float(torch.as_tensor(ck["logit_scale"]).float().exp().clamp(max=100.0))`（`heads.py:95`）。

训练：同一行。

```python
def _clm_loss(state_head, action_head, logit_scale, states, actions, codes):
    state_z = F.normalize(state_head(states), dim=-1)
    action_z = F.normalize(action_head(actions), dim=-1)
    logits = logit_scale.exp().clamp(max=100.0) * state_z @ action_z.t()
    labels = torch.arange(len(states), device=states.device)
    same = codes.unsqueeze(0) == codes.unsqueeze(1)
    same.fill_diagonal_(False)
    forward = F.cross_entropy(logits.masked_fill(same, float("-inf")), labels)
    backward = F.cross_entropy(logits.t().masked_fill(same, float("-inf")), labels)
    return (forward + backward) / 2, logits, labels
```

`finetune.py:174-183`。这是 InfoNCE：`scale` 是温度倒数（README 的 $\tau$ 的倒数），
`labels = arange(B)` 让「第 i 个状态配第 i 个动作」成为正样本对，双向各算一次交叉熵。
**推理时的 `scale * cos` 就是这个 `logits` 的同一项**，`temperature` 是推理时才出现的额外除数。
同一条式子还出现在 `choice` 任务里（`finetune.py:510-513`）与评测里（`bon_eval.py:53-55`）。

三个可直接从代码读出的性质：

- **投影被归一化，所以 `cos ∈ [-1, 1]`，`logits ∈ [-scale, scale]`**，
  而 `scale` 被 `clamp(max=100.0)` 限在 100（`heads.py:95`、`finetune.py:177` 两处都有这个 clamp）。
  `logit_scale` 的初值是 `math.log(1 / 0.07)`（`finetune.py:289,503`），即 `scale ≈ 14.29`；
  训练可以把它推到上限。到上限时 T=1 的分布几乎必然是 one-hot——**实际训到多少，未核实**
  （需要权重或训练记录）。
- **`clamp` 是静默的**：如果某个检查点的 `logit_scale` 超过 `log(100)`，
  服务端用的 `scale` 会小于 `exp(logit_scale)`，而响应里没有任何地方报出它用了多少（§10）。
- **`temperature` 是除数**，作用在 `scale*cos` 上：`T>1` 压平、`T<1` 变尖（`engine.py:136`）。
  范围 `(0, 100]`，`Engine.answer` 与 `MockEngine.answer` 各校验一次
  （`engine.py:109-110`、`playground_mock.py:86`），服务端先在解析处判一次数字类型（`server.py:106-108`）。

### 4.3 「一次前向」在这里变成了什么

`README.md:209-211` 把这条路线自己的成本模型写清楚了：

> Each encoder is a frozen LLM backbone plus a 20M-parameter trainable projection head,
> so inference is **one embedding per fresh text and a dot product per cached candidate**.

于是 RYOTIDE 的不变量 #3（只读最后一个位置）在这里没有对象，而**新的成本结构**是：

| | 每个请求要付出什么 |
|---|---|
| 状态侧 | 每题一次编码（`state_text` 各不相同，§7）；`zq` 一次性算好 |
| 候选侧 | **首次** K 次编码；之后 0 次（竞技场命中，§8） |
| 决策 | K 次点积 + 一次 host 侧 softmax |
| 生成 | 0 —— `output_tokens: 0` 是字面量（`engine.py:138`） |

`README.md:466-473` 给了一张 p50 的表（同一张 RTX 4090、固定候选集）：
新状态每次调用 28.6 → 28.0 ms、**重访状态 1.7 → 0.6 ms**（3 个候选）与 2.0 → 0.7 ms（50 个候选）。
那是上游记录的数字，**本文不重跑**；但那个方向性（重访便宜、新状态照付编码器）
在 §14.1 的 committed 记录上能独立看见。

---

## 5. 掩码 + softmax：没有掩码，softmax 在 host 上用 Python 浮点算

RYOTIDE 的 §5 是「掩码 + 重归一化」。这一节在这里只剩后半句，而且后半句只有一步。

### 5.1 掩码那一步不存在

没有词表 ⇒ 没有需要打到 `-inf` 的位置 ⇒ 没有「掩码后重归一化」这个两段式结构。
「重归一化」在这里不是补救，而**就是**唯一的归一化：

```python
def softmax(logits: list[float]) -> list[float]:
    m = max(logits)
    e = [math.exp(v - m) for v in logits]
    z = sum(e)
    return [v / z for v in e]
```

`schema.py:115-119`。输入正好是那 K 个候选的 `scale*cos/T`（`engine.py:136` 的 `.tolist()`）。
这是这条线上一处容易被忽略的实现事实：**概率是在 host 上、用 Python 的 `math.exp` 算的**，
不是 `torch.softmax`——设备侧只算点积，然后 K 个 float32 被 `.tolist()` 搬回 host
（`engine.py:136`），在 Python 浮点里做 max 减法与归一化。
`answer_from_logits` 只是把它接到答案组装上（`schema.py:148-149`）。

因此那条不变量的落位是：**结构上恒成立，且没有可调的地方**——
softmax 的自变量就是那 K 个数，不可能有多余质量跑到别处。

### 5.2 答案成形，与 `confidence`

```python
def answer_from_probs(q: dict, keys: list[str], probs: list[float]) -> dict:
    """Assemble the Answer object for question ``q`` from its option distribution."""
    t = q["type"]
    probs = [float(p) for p in probs]
    dist = {k: p for k, p in zip(keys, probs)}
    if t == "noul":
        return {"type": "noul", "noul": dist["true"]}
    if t == "choice":
        j = max(range(len(probs)), key=probs.__getitem__)
        return {"type": "choice", "choice": keys[j], "confidence": confidence(probs), "probabilities": dist}
    levels = q["criteria"]
    score = sum(i * p for i, p in enumerate(probs))
    legend = {str(i): c if isinstance(c, str) else to_text(c) for i, c in enumerate(levels)}
    return {"type": "score", "score": score, "confidence": confidence(probs), "legend": legend,
            "probabilities": dist}
```

`schema.py:131-145`。四处要记：

- **`noul` 的答案里没有 `confidence`**，只有 `noul: p_true`（`schema.py:137`）。
  `choice` 与 `score` 才有。所以三题的字段并不齐——同一个接口下，一题有置信度、另一题没有。
- **`score` 是期望档位**：`sum(i * p for i, p in enumerate(probs))`，值域 `[0, n-1]`，
  配一个 `legend` 把下标映回档位文本（`schema.py:142-144`）。
- **`confidence = top - mean(其余)`，再钳到 `[0,1]`**：

  ```python
  def confidence(probs: list[float]) -> float:
      """TypeSafe-style confidence: top probability minus the mean of the rest."""
      if len(probs) < 2:
          return 1.0
      j = max(range(len(probs)), key=probs.__getitem__)
      rest = [p for i, p in enumerate(probs) if i != j]
      return max(0.0, min(1.0, probs[j] - sum(rest) / len(rest)))
  ```

  `schema.py:122-128`。这是上游按 TypeSafe 的口径复刻的一个式子（`README.md:397` 写成
  *`confidence` = top probability minus the mean of the others*）。两点后果：
  **①平坦分布一律报 0.0**（钳位把「有多平」这一信息丢掉了）；
  **②只有一个候选时恒为 1.0**（`len(probs) < 2` 分支）——而单候选 `choice`
  在 `schema.py:80-81` 是合法的（只要 `criteria` 非空）。
- **同一个分布有两个归约**：响应里的 `score` 是**期望档位**，
  而 `label_of` 对 `score` 取的是 **argmax 档位**（`schema.py:159-160`）：

  ```python
  def label_of(answer: dict) -> str:
      """Discrete label of an answer (for scoring): choice key, score level, or 'true'/'false'."""
      t = answer["type"]
      if t == "choice":
          return answer["choice"]
      if t == "noul":
          return "true" if answer["noul"] >= 0.5 else "false"
      p = answer["probabilities"]
      return max(p, key=p.__getitem__)
  ```

  两者在平坦分布上会给不同的档位。这是 §10.3 那条「算得出来没接进来」的一半。

### 5.3 三处静默的边界

按「违反时数字会看似合理却是错的」这个标准，这里有四处没有守卫的地方，全部可从代码确认：

1. **`noul` 的判正阈值是 0.5**（`schema.py:158`），而它读的是 `noul` 字段而不是分布——
   没问题，但要注意 `label_of` 与 `probabilities_of`（`schema.py:163-166`）构造的那份
   `{"false": 1-p, "true": p}` 在其他地方并不存在：响应里 `noul` 题**根本没有 `probabilities`**。
2. **重复候选文本**（§2.2）：两个概率严格相等，argmax 给第一个，无诊断。
3. **`confidence` 的钳位**（§5.2）：0.0 与「真的算出来是负的」不可区分。
4. **`temperature` 不回声**（`engine.py:137-138`）：响应只有 `model` / `answers` / `usage`
   （`server.py:119` 再加一个延迟头）。同一个请求换一个 T，得到的 JSON 里没有任何字段能区分。

---

## 6. 顺序偏置：不存在该变量，但并列的裁决依赖顺序

RYOTIDE 用多 order 共享前缀 fork 再平均来消顺序偏置；SemIf-OpenJev 不消除只度量；
Semantic Router 刻意保留。CLM 的情况是**这个变量在架构上不存在**，理由可以直接指到两行代码：

```python
za = self._cached(f"{ns}/action", dim, cands, tokens, head.project_actions)   # engine.py:130
...
cos = za[k:k + len(texts)] @ zq[i]                                           # engine.py:134
```

第一条把**每一个候选文本各自**映射成一个向量（`head.project_actions` 走的是
`heads.py:104-109` 的逐行 `F.normalize(head(x))`）；第二条把状态向量与每个候选向量**逐个点积**。
**候选之间没有任何一项交互**：第 j 个候选的得分里不出现第 k 个候选。
所以把 `criteria` 的顺序打乱，每个选项的概率在数学上不变，
只是 `keys` 与 `probs` 的排列跟着变（`schema.py:85-87` 保序、`schema.py:135` 按位 zip）。

唯一要小心的技术细节：`softmax` 的 `max` 与 `sum` 是在一个被置换过的列表上做的
（`schema.py:116-118`），浮点加法不满足结合律，所以**打乱顺序可能改变末位**。
这一条本文没有实测（需要权重），只作为「严格说来不是逐位不变」的注记。

训练侧同样没有顺序先验，而且是两种口径：

- `clm` 任务：`_clm_loss` 把整个 batch 当成一个检索池，正样本由 `arange` 定义（`finetune.py:177-183`），
  位置只决定「谁是正样本」，不进入任何分数；
- `choice` 任务：`infonce_loss` 把 batch 内所有候选文本用 `torch.unique` 收成一个池
  （`finetune.py:537-540`），一个候选的分数只取决于它的**文本**，与它出现在哪些题的哪一位无关。
  另一条 `softce` 损失是逐题的 softmax（`finetune.py:510-517`），顺序只用来对齐下标。

**真正依赖顺序的地方只有并列的裁决**，而且是两处、彼此独立：

- `schema.py:139` 的 `max(range(len(probs)), key=probs.__getitem__)`：取第一个最大值。
  所以重复候选文本（§2.2）时，被选中的是**请求顺序里的第一个**。
- `examples/t_rex` 的 `proposed = max(ACTIONS, key=lambda a: (p[a], -ACTIONS.index(a)))`
  （`brain.py:32`）：并列时取 `ACTIONS = ("jump","duck","run")` 里靠前的那个
  （`planner.py:32`）。这是这个示例自己的一层 tie-break，与 `schema.py` 那处无关——
  **同一个仓库里两处并列裁决，两套顺序。**

---

## 7. 多问题共享 state：不共享 state，共享的是候选

RYOTIDE 的 `run_many()` 把公共前缀预填充一次、计费只算一次；Semantic Router 把它做成一档
`shared_context` profile。CLM **不做**这件事，而代价的形态与别家不同。

### 7.1 每题一个 `state_text`，所以没有共同的 state

```python
def state_text(state: Any, instructions: Any) -> str:
    """Context first, question last — the layout the heads were trained on."""
    s, i = to_text(state).strip(), to_text(instructions).strip()
    return f"{s}\n\n{i}" if s and i else (s or i)
```

`schema.py:62-65`；`build_pairs` 对每一题各调用一次（`schema.py:104-112`）。
所以 N 个问题产生 N 段**互不相同**的状态文本（只要 `instructions` 不同），
`engine.answer` 把它们整列送去编码（`engine.py:119,129`），去重只对**完全相同的文本**生效
（`Embedder.embed` 的 `dict.fromkeys`，`embedder.py:64`）。

**没有任何前缀共享**：N 题就是 N 段文本，每段都从 `state` 开头重复一遍。
本仓没有 RYOTIDE 那种「公共前缀预填充一次」的机制，也没有为它计费的折扣
（`billing_units = len(questions)`，`engine.py:138`）。
把 state 头部交给 vLLM 的前缀缓存是**可选**的，而且两份启动命令不一致：
`README.md:59` 的 quickstart **没有** `--enable-prefix-caching`，
`serve_qwen3_8b.sh:19` 有。**对 pooling runner 是否生效、生效到什么程度，本文未核实。**

### 7.2 真正被共享的是候选向量，而且跨请求

一次请求里所有题的候选文本被拼成一列统一编码：

```python
states = [p[0] for p in pairs.values()]
cands = [t for p in pairs.values() for t in p[2]]
```

`engine.py:119-120`。所以同一题内、以及不同题之间的**重复候选文本**只算一次编码。
但这条优化的主战场不在这里，而在 §8：agent 循环里**动作集基本不变、状态一直变**，
于是候选向量第一次请求之后就基本不再产生编码开销。这就是上游那句
*States and actions are disaggregated, so their embeddings are cached and reused independently*
（`README.md:32-33`）的全部含义。

### 7.3 `usage.input_tokens` 是缓存相关的数

```python
def _cached(self, namespace: str, dim: int, texts: list[str], tokens: list[int], project=None):
    """Vectors for ``texts``, from the arena where possible; ``tokens`` collects misses."""
    def compute(missing: list[str]):
        emb, spent = self.embedder.embed(missing)
        tokens.append(spent)
        return project(emb) if project else self._to_device(emb)
    ...
```

`engine.py:75-84`；`tokens` 最后被求和成 `usage.input_tokens`（`engine.py:138`）。
`Embedder.embed` 只对**未命中**的文本发请求，并把 vLLM 报的 `prompt_tokens` 返回
（`embedder.py:59-80`，取的是 `usage.prompt_tokens`，`embedder.py:57`）。

结论：**`input_tokens` 不是这个请求的函数，而是「这个请求 + 当前缓存状态」的函数。**
`README.md:90` 自己把这一点写进了 quickstart 的注释：
*`# 38 58.1     (106 tokens on a cold cache: option texts are embedded once)`*——
同一个调用冷缓存 106、热缓存 38。这是九篇里唯一一条把「同一请求两种 token 数」摆在首页的实现。

还有一处更细的会计问题，只有读代码才看得见：竞技场的 thrashing 分支会**再算一遍**，
于是同一批文本被计两次 token（`cache.py:150-155` 与 `engine.py:76-80` 的 `tokens.append` 在
同一次 `_cached` 调用里执行两遍）。见 §8.3。

---

## 8. 向量竞技场：CLM 版的「KV cache 预算」

别家的 cache 一节讲的是 KV 怎么复制、怎么共享前缀。CLM 没有 KV cache（它不是自回归的），
它有的是**一个自己造的、启动时就占满、之后永不增长的显存池**。

### 8.1 一次分配，按宽度切池

```python
self.flat = torch.zeros(self.reserved_bytes // self.item, device=self.device, dtype=self.dtype)
```

`cache.py:101`。预算来自一个分数（占设备内存的 `0.02`，默认）或一个绝对大小（`512MiB`），
`0` 表示关掉（`cache.py:36-53`、`DEFAULT_BUDGET = "0.02"` 于 `cache.py:27`）。
实际占用还要与「当前空闲的 90%」取小，给编码器和 head 留地方（`cache.py:99-100`）。
`Engine._reserve` 按宽度切池：

```python
for dim in sorted({h.proj_dim for h in self.heads.values()}):
    arena.reserve(dim, RAW_SHARE if dim == HIDDEN else (1.0 - RAW_SHARE))
arena.reserve(HIDDEN, RAW_SHARE)       # clm-raw works in the encoder's own space
```

`engine.py:69-71`，`RAW_SHARE = 0.125`（`engine.py:28`）。
两个池的份额是 7/8（512 维投影）与 1/8（4096 维原始嵌入）——
**因为 4096 维的行宽是 512 维的 8 倍，1/8 的字节数换来的是 1/56 的行数**：

| 池 | 行数（`README.md:454` 打印的） | 字节数 | 占比 |
|---|---|---|---|
| 512 维 | 215,764 | 215764 × 512 × 4 = 441,884,672 B | 87.5% |
| 4096 维 | 3,852 | 3852 × 4096 × 4 = 63,111,168 B | 12.5% |
| 合计 | — | 504,995,840 B ≈ **505.0 MB** | 与打印值一致（§14.3） |

行数比 215,764 / 3,852 = 56.01，正是 7 × (4096/512) = 56。

### 8.2 键 = 头的身份 + 文本；hot-reload 靠代次失效

```python
keys = [f"{namespace}\x00{t}" for t in texts]
```

`cache.py:133`，而 namespace 是

```python
@property
def namespace(self) -> str:
    """Identity of these exact weights, for cache keys."""
    return f"{self.name}@{self.generation}"
```

`heads.py:122-125`；`generation` 每次 (re)load 加一（`heads.py:93`），
而 (re)load 由 mtime 变化触发（`heads.py:97-102`）。所以：

- 多个 head 共用一个池（键里有名字）；
- **hot-reload 之后旧行的键不再匹配**，不需要显式失效，LRU 慢慢回收
  （每个池一个 LRU，`Pool.claim` 于 `cache.py:66-73`）。
  这一条与上游文档说的一致：*Entries are keyed by head and generation, so several heads share
  the arena and a hot-reloaded head stops matching rows its previous weights produced*（`README.md:462-463`）。
- `/health` 报占用与命中率（`server.py:86`，`VectorArena.stats` 于 `cache.py:160-166`；
  启动日志也打一行，`server.py:213-216`）。

### 8.3 两条只在读代码时看得见的性质

**① 缓存里存的是投影，不是编码器输出**（除了 `clm-raw`）。
`engine.py:129-130` 传进去的 `project` 是 `head.project_states` / `head.project_actions`，
`compute` 的返回值被直接写进池（`cache.py:148-149`）。
所以命中一次省掉三件事：编码器调用、host→device 拷贝、head 的前向——
上游把这句写在 `README.md:466`。宽度的选择也由此而来：默认 head 的池是 `proj_dim`=512。

**② thrashing 分支会重算，并且重复计入 token**：

```python
with self._lock:
    rows = [pool.slots.get(k) for k in keys]
    if any(r is None for r in rows):
        return compute(texts)                # thrashing; answer without the arena
```

`cache.py:150-155`。注释是诚实的（*thrashing; answer without the arena*），
但这个分支有两个后果：重算的向量**不回填**池，以及 `compute` 里的 `tokens.append(spent)`
（`engine.py:77-79`）会在这同一次 `Engine._cached` 调用里**第二次**追加 token 数，
使 `usage.input_tokens` 把同一批文本算两遍。
这只在「填充与淘汰在同一请求内竞争」时发生，概率低；但它意味着
**`input_tokens` 不是「编码器实际处理的 token 数」的严格上界之外的保证**——
更准确的说法是：它是「编码器实际处理的 token 数」，而在 thrashing 时编码器确实处理了两遍。

---

## 9. 服务化 / fail-loud 风格

### 9.1 接口与错误码

| 路由 | 说明 | 位置 |
|---|---|---|
| `POST /v1/systemone` | 主入口：`{state, model, questions[, temperature]}` | `server.py:96-119` |
| `POST /v1/rank` | 同一原语的裸形式：`{context, question, answers[]}` | `server.py:121-153` |
| `GET /v1/models` | 已加载的 head 列表 + `clm-raw` | `server.py:91-94`，`engine.py:91-98` |
| `GET /health` | `ok` / `embedder` / `models` / `cache`（未 mock 时） | `server.py:82-89` |
| `GET /` + 静态 | playground（`--no-ui` 可关），路由最后挂载 | `server.py:155-162` |

错误码四档，都是显式的：`401` 键不对（`server.py:78-80`）、
`422` 非 JSON / 形状不对 / `temperature` 不是数 / 引擎抛的 `ValueError`·`KeyError`·`TypeError`·
`AttributeError` / 未知模型（`server.py:101-116`）、`502` 编码器不可达（`server.py:117-118`）。
服务端耗时走 `X-CLM-Latency-Ms` 头（`server.py:119`）。

`EmbedderError` 是这条线上唯一自定义的异常，在「连不上」与「非 200」两处抛出
（`embedder.py:44-49`），服务端把它翻译成 502——所以**编码器挂掉时不会静默返回一个分布**，
这是这条路线在别处最缺的那类守卫里做得对的一件。

### 9.2 fail-loud 的四件好事

- **下载是可关的**：`--no-download` 让 `clm-serve` 直接失败而不是去拉权重
  （`server.py:185,194-195`），没有任何检查点时 `SystemExit("no checkpoint: …")`（`server.py:206-207`）。
- **启动时把状态全打出来**：模型与设备、编码器是否可达、鉴权开关、竞技场占用与两个池的容量、
  监听地址（`server.py:210-222`）。
- **播放页的假数据会自报**：`tools/playground_mock.py` 用字符 n-gram 冒充编码器，
  但它在 `/health` 上报 `"mock": true`（`server.py:87-88`，`playground_mock.py:67`）、
  在描述里写 `MOCK — character n-grams, not a contrastive language model`（`playground_mock.py:76-80`）、
  并在 UI 里挂警告横幅，docstring 第一句就是 *for working on the UI, not for measuring anything*
  （`playground_mock.py:1-9`）。
- **CORS 默认关，且把理由写在代码里**：*an API key otherwise travels in a header any page would
  then be free to send*（`server.py:66-70`）。

### 9.3 fail-loud 的边界：五处没有守卫的地方

这一节是本文认为最该被引用的一节——按「违反时数字会看似合理却是错的」排：

| # | 没有守卫的东西 | 证据 | 后果 |
|---|---|---|---|
| 1 | **head 与编码器（含池化方式）的配对** | 检查点不记 `embed_model`（`finetune.py:315-318`）；`HeadPair._load` 只读架构字段（`heads.py:81-95`）；`/health` 只报 `healthy()`（`server.py:84`，实现见 `embedder.py:82-87`） | 换一个池化位置不同的编码器，服务照常起、照常给分布 |
| 2 | **编码器取哪个位置** | 训练侧保尾（`embed_utils.py:43,49`）；服务侧只有 `--runner pooling`（`serve_qwen3_8b.sh:18`） | 读位置错了没有任何诊断量 |
| 3 | **候选文本重复** | 无任何检查（§2.2） | 静默并列，取请求顺序第一个 |
| 4 | **`temperature` 不回声** | `engine.py:137-138`、`server.py:119` | 一个数离开服务后无法回源到 T |
| 5 | **`scale` 的 clamp 静默生效** | `heads.py:95` 的 `.clamp(max=100.0)` | 响应里没有任何字段报出实际用的 `scale` |

对比一下别家：RYOTIDE 有 `marker_mass`，Laya 有 `options_distinct`/`tokens_per_option`，
SemIf-OpenJev 有 `allowed_token_mass` + `full_vocab_argmax_id`，openjev-hf 有 `/v1/version`
把全部常数与 flag 连摘要一起报出，Semantic Router 有加载期 golden 自检。
**CLM 在这条轴上是九篇里最薄的一条**：它有 `/health` 的缓存统计与 `billing_units`，
但那两个量都不指向「这次读得对不对」。

---

## 10. 置信度从哪来，又怎么被改写

### 10.1 一条链

| 环节 | 改数值？ | 改 argmax？ | 位置 |
|---|---|---|---|
| ① **训练本身**：`state_head` / `action_head` / `logit_scale` | **是** | **是** | `finetune.py:174-183,287-298`（`clm`）、`finetune.py:495-513`（`choice`） |
| ② **`scale = exp(logit_scale)` 的 `clamp(max=100)`** | 是（只在超过 100 时） | 否（单调缩放） | `heads.py:95` |
| ③ **`temperature`**：`scale*cos/T` | **是** | 否（softmax 单调） | `engine.py:136` |
| ④ **`confidence` 公式**：`top - mean(其余)`，钳到 `[0,1]` | 是（改名不换分布，但钳位丢信息） | 否 | `schema.py:122-128` |
| ⑤ 任何形式的标定（温度拟合 / ECE / 采信判据） | — | — | **不存在** |

第 ①环是这条线上的主环节，而它的性质与 RYOTIDE 正相反：
RYOTIDE 的分布是模型自己的 softmax，训练不参与，所以温度**不可能**改准确率；
CLM 的分布来自两个被训练过的投影，`scale` 也是学出来的（`finetune.py:289,296`），
所以**训练既改数值也可能改 argmax**。第 ③环仍然是纯单调的，改不了 argmax。

### 10.2 没有标定链，只有一个每请求的旋钮

这是 CLM 与另外八篇差别最大的一处。别家都把「T 从哪来」当成一个要交代的问题——
离线拟合 + 接受判据（RYOTIDE）、拟合后无条件发布（Open-Jev、Laya）、
写在模型包里（Quyet、Semantic Router）、写进发布配方（OpenJev(HF)）。
CLM 的答案更简单：**`temperature` 是调用方的入参，默认 1.0，不随权重发布，也不写进响应。**

```python
def answer(self, state: Any, questions: dict[str, dict], model: str = DEFAULT_MODEL,
           temperature: float = 1.0) -> dict:
    """-> {"model", "answers": {id: Answer}, "usage"}; raises ValueError on bad questions."""
    if not questions:
        raise ValueError("questions must not be empty")
    if not (0 < temperature <= 100):
        raise ValueError("temperature must be in (0, 100]")
```

`engine.py:104-110`；服务端从 body 里取，缺省 1.0（`server.py:106`）。
于是「这个概率是在哪个 T 下产生的」这个问题**只能问调用方**。
对它自己 committed 的那份记录，这件事反而可以确定：T-Rex 的请求体里根本没有这个键——

```python
response = self.client.post(
    "/v1/systemone", json={"model": self.request_model, "state": state, "questions": questions}
)
```

`examples/t_rex/trex/backends.py:127-129`；全仓 `examples/` 下 grep `temperature` 零命中。
所以 **§14.1 复算的那批 T-Rex 数字用的是默认 T = 1.0**——这不是推测，是「请求体没有这个键 +
服务端默认 1.0」两个事实的合成。

### 10.3 一处「算了也导出了，但仓里没人用」

```python
def label_of(answer: dict) -> str:
    """Discrete label of an answer (for scoring): choice key, score level, or 'true'/'false'."""
```

`schema.py:152`，以及 `probabilities_of`（`schema.py:163`）。
**全仓没有任何调用点，而且连包级公开面都没有它们**：

```bash
grep -rn "label_of\|probabilities_of" .     # 只有 schema.py:152 与 schema.py:163 两处定义
grep -n  "label_of\|probabilities_of" src/clm/__init__.py   # 无匹配（退出码 1）
```

对比同一个模块里的邻居：`build_pairs` / `candidates` / `state_text` / `answer_from_logits` /
`answer_from_probs` 都在 `__init__.py:17` 被导入、在 `__all__`（`__init__.py:20-22`）里列出，
这两个没有。所以它们的可达路径只有 `clm.schema.label_of`，不在包的门面上。
值得记下的是它与 §5.2 那条口径问题的关系：
**库对外提供的「取离散标签」和响应里给出的「`score`」是两个不同的归约**
（argmax 档位 vs 期望档位）。下游拿哪个去算准确率，会得到不同的数——
而本仓不给示范，因为没有调用点。

---

## 11. 四条不变量的落位

RYOTIDE §13 的四条，在 CLM 上的落位如下（判据不足的写「概念不适用」并给等价命题）。

| RYOTIDE §13 不变量 | 在 CLM 上 |
|---|---|
| **#1** 读位置必须落在标记上（诊断量 `marker_mass`） | **概念不适用**：没有词表分布，所以没有「质量落在标记上」这个量。改写成等价的命题：**「编码器的池化位置必须与训练时一致」**——而这一条**在代码里完全没有守卫**，既不断言、也无诊断量，只有 `README.md:305-306` 的一句话（§3.1、§3.2） |
| **#2** 前缀与标记的表面形式耦合，必须成对扫 | **不适用**：标记就是文本，候选文本被整段编码，不存在「表面形式」这个自由度。改写成 CLM 版的等价命题，而且这条是**正的**：**渲染布局必须训练与推理一致**——而它由「两条路径共用同一个 `build_pairs`」保证：`train/adapters.py:19,74` 与 `src/clm/engine.py:118` 调的是同一个函数（`schema.py:104`）。这是这条线上少见的、靠**共享代码**而不是靠约定维持的一致性 |
| **#3** 只读最后一个位置 | **改写**：「最后一个位置」只出现在编码器的池化里，而那个选择**不在本仓**（§3.1）。决策侧读的是 `K+1` 个向量、做的是 `K` 次点积（`engine.py:134`），没有位置可读（§4.3） |
| **#4** 比较概率 / ECE 前先确认 `temperature` | **成立，而且比别处更难确认**：T 不随权重发布，是**每请求的入参**（`engine.py:105`），默认 1.0，**响应里不回声**（`engine.py:137-138`）。committed 记录能定下 T=1.0 只因为请求体里没有这个键（§10.2） |
| （本条新增） | **候选文本必须互不相同**——两条相同文本在数学上不可区分（逐候选编码 + 逐候选点积），会静默并列并被顺序裁决成第一个（§2.2、§6）。本仓无守卫、无诊断量 |

---

## 12. 文件对照

| 文件 | 行数 | 职责 |
|---|---|---|
| `src/clm/schema.py` | 166 | **核心**：题面渲染（`to_text` / `state_text`）、候选构造（`candidates` / `build_pairs`）、`softmax`、`confidence`、答案组装（`answer_from_probs`） |
| `src/clm/engine.py` | 151 | `Engine.answer` / `Engine.rank`：取向量、点积、组装；竞技场预留（`_reserve`） |
| `src/clm/heads.py` | 184 | `Head` 架构、检查点加载 / hot-reload / 下载；`scale` 在这里从 `logit_scale` 算出 |
| `src/clm/embedder.py` | 86 | `/v1/embeddings` 客户端 + 200k 条目的 LRU（按文本键）；`l2()` |
| `src/clm/cache.py` | 166 | 显存竞技场：一次分配、按宽度切池、LRU、按 `namespace\0text` 键、thrashing 分支 |
| `src/clm/server.py` | 228 | FastAPI 应用、鉴权、错误码、`X-CLM-Latency-Ms`、静态播放页、`clm-serve` 入口 |
| `src/clm/client.py` | 203 | 纯 HTTP 客户端 + `Noul` / `Choice` / `Score` 与三种答案类型（不需要 torch） |
| `src/clm/__init__.py` | 35 | 公开面；`Engine`/`Embedder`/`HeadPair` 懒加载 |
| `src/clm/static/*` | 925+212+554 | 播放页（`app.js` 里能看到字段怎么被渲染：`noul` 只有一条概率条、`score` 有指针刻度尺） |
| `train/finetune.py` | 749 | 两个任务：`clm`（轨迹 InfoNCE）与 `choice`（typed-decisions，`infonce`/`softce` × `soft`/`hard`）；写检查点 |
| `train/adapters.py` | 87 | 轨迹读取；`typed_decision_examples` 复用 `build_pairs` |
| `train/embed_utils.py` | 106 | **训练侧 token 配方**（`Recipe`：保尾、`cap = max_len - 1`）与两个 embedding 后端 |
| `evaluation/bon_eval.py` | 320 | best-of-N 选择率评测；同一公式（`step_scores`）、按轨迹末 `window` 步求均值 |
| `preprocessing/hf_embeddings.py` | 205 | 嵌入目录 ↔ HF parquet 数据集互转 |
| `tools/playground_mock.py` | 131 | 假编码器版本的播放页；自报 `mock: true` |
| `examples/common.py` | 210 | `Judge`：重试 / 缓存 / 统计（**在 t_rex 路径上未被使用**，§14.4） |
| `examples/t_rex/**` | 2812 | 实时游戏示例：planner + 子进程 brain + 60 FPS 主循环 + committed 记录（其中 `trex/engine.py` 775 行是游戏本体、`trex/pilot.py` 735 行是实时 pilot） |

---

## 13. 亲手跑一遍

前面几篇的惯例是给可复现的命令。CLM 的官方路径需要一张 GPU 与 vLLM，本文**没有跑过**；
下面按「跑不了/跑得了」分开列。

### 13.1 官方路径（需要一张 GPU，本文未执行）

```bash
pip install contrastive-lm                      # 或 pip install -e .  从 clone 装

# 1) 编码器：一个 vLLM pooling 服务（GPU）。注意这一行不指定池化方式（§3.1）
vllm serve Qwen/Qwen3-8B --served-model-name qwen3-8b --runner pooling --max-model-len 2048 --port 8090 &
# 仓库自带的启动脚本，多了 enforce-eager / prefix-caching / 内存上限
GPU=0 PORT=8090 UTIL=0.35 ./serve_qwen3_8b.sh

# 2) 决策服务（首次运行会下载约 75 MB 的参考头）
clm-serve                                       # :8700；--no-ui 只留 API，--action-cache 0 关掉竞技场

# 3) 一次请求（README.md:71-91 的同一个例子）
curl -s localhost:8700/v1/systemone -H 'content-type: application/json' -d '{
  "state": "Customer: my invoice was charged twice and nobody answers the phone!",
  "questions": {
    "urgency":     {"type": "noul",   "instructions": "Is this urgent?"},
    "department":  {"type": "choice", "instructions": "Which team should handle this?",
                    "criteria": {"billing": "Charges, invoices, refunds",
                                 "technical": "Bugs and outages"}},
    "frustration": {"type": "score",  "instructions": "How frustrated is the customer?",
                    "criteria": ["Calm", "Frustrated", "Very angry"]}}}' | python3 -m json.tool

# 4) 实时游戏：CLM 与 Jev 在同一个 harness 下的 committed 记录就是这么生成的
python examples/t_rex/run.py --model clm        # 5 seeds × 60 s，实时，shield 开
python examples/t_rex/run.py --model clm --lockstep 6   # 冻结游戏等答案（把延迟从指标里拿掉）
```

### 13.2 不需要 GPU 的部分（本文执行过的只有这一类）

```bash
# 播放页的假编码器版本——路由与答案 schema 是真的，数字是噪声（自报 mock: true）
pip install fastapi uvicorn numpy
python tools/playground_mock.py --port 8700

# 竞技场尺寸的算术核对（§14.3）：对 README.md:454 打印的那一行做手算
python3 -c "print(215764*512*4 + 3852*4096*4)"     # -> 504995840  ≈ 505.0 MB
python3 -c "print(215764/3852)"                     # -> 56.01      ≈ 7 × 4096/512

# committed 记录的复算（§14.1），只用标准库
python3 - <<'PY'
import json, statistics as st
for f in ("examples/t_rex/results/clm_realtime.json", "examples/t_rex/results/jev_realtime.json"):
    d = json.load(open(f)); R = d["results"]; col = lambda k: [r[k] for r in R]
    print(f, "| decisions", sum(col("decisions")), "| p50 中位数", st.median(col("latency_ms_p50")),
          "| tokens/decision", [round(r["input_tokens"]/r["decisions"], 3) for r in R])
PY
```

**没有跑过的**：`import clm`、`clm-serve`、`train/finetune.py`、`evaluation/bon_eval.py`、
任何一次前向、任何权重下载。所以本文不出现任何一个由前向产生的概率数值——
§14 里出现的每个概率都来自 `README.md` 的引文或 committed JSON，且都标明来源。

---

## 14. 实例走查

### 14.1 committed 的两份 T-Rex 记录（CLM vs Jev）

来源：`examples/t_rex/results/clm_realtime.json` 与 `jev_realtime.json`；
上游在 `examples/t_rex/README.md:33-35` 记下了两次运行的时间与机器：
**CLM 2026-09-23，`clm-latest` = `CLM_v0.1-8B.pt`，Qwen3-8B 编码器在一张 RTX 4090 上；
Jev 2026-09-22，`jev-latest` 应答为 `jev-1.13.0`；延迟是客户端逐请求测的。**
JSON 里没有日期字段，`model` 字段分别写着 `clm-latest` 与 `jev-1.13.0`。

配置（两份相同）：`mode: realtime`、`shield: true`、`course_style: original`、
`prompt: labeled`、`duration_s: 60.0`、`seeds: 5`、`inflight: 6`；
`endpoint` 五次运行逐字节相同（CLM 是 `http://127.0.0.1:8700 · clm-latest · 6 requests in flight`，
Jev 是 `https://api.typesafe.ai · …`）——**同一份记录里五次运行打的是同一个服务进程**，
这一点下面要用到。

| 复算量 | CLM | Jev | 复算式 |
|---|---|---|---|
| `decisions` 五次 | 3048 / 3244 / 3405 / 3501 / 3511 | 1097 / 1175 / 1056 / 1162 / 1105 | — |
| 合计 | 16,709 | 5,595 | `sum` |
| 均值（= `summary.mean_decisions`） | 3341.8 | 1119.0 | 逐位吻合 |
| `latency_ms_p50` 中位数（= summary） | 16.5 | 149.8 | `statistics.median` 吻合 |
| `latency_ms_p95` 范围 | 16.9 – 67.4 | 184.4 – 200.3 | — |
| `model_ms_p50` 中位数（= summary） | **2.6** | **131.9** | `statistics.median` 吻合 |
| `agreement_with_planner` 均值（= summary） | **0.658** | **0.987** | 0.6582 / 0.9870，summary 取 3 位 |
| `answers_discarded` 合计（= summary） | 1250 | 2926 | 逐位吻合 |
| `input_tokens` 合计 | 89,878 | 3,103,560 | — |
| **tokens / 决策** | **5.38** | **554.7** | 89,878 / 16,709 与 3,103,560 / 5,595 |

四条从这张表里直接读出来的东西：

1. **端到端 p50 的 9.1 倍**：149.8 / 16.5 = 9.08。这正好落在上游那句
   *up to **9×** lower latency*（`README.md:29`、`README.md:145`）上——
   **这个「9×」在 committed 产物上是可复算的**。模型侧的差更大：131.9 / 2.6 = **50.7 倍**；
   两者不等，是因为 `latency_ms` 是「从发出到在游戏里落地」（`pilot.py:150`），
   包含排队与在飞等待，而 `model_ms` 是那一次 HTTP 调用本身（`backends.py:126-130`）。
2. **每条决策的编码器 token：5.38 对 554.7，差 103 倍**。
   这是「状态与动作解耦」那句话在 committed 产物上的样子：固定的三个动作描述几乎不再重复编码。
3. **CLM 的 tokens/decision 逐 run 单调下降**：15.593 → 6.991 → 3.003 → 1.408 → 1.286；
   **Jev 是平的**：563.1 / 542.9 / 573.0 / 540.3 / 556.5。
   五条记录按 seed 0→4 排列，而 CLM 那五次打的是同一个服务进程（`endpoint` 逐字节相同），
   所以这条曲线与「竞技场 + 进程内 LRU 按文本缓存、跨 run 累积」一致（§8.2）——
   第一次运行付了冷缓存的账（`README.md:90` 的 106 vs 38 是同一件事的微观版本），
   第五次只花 1.29 token/决策。Jev 那边没有可复用的东西，所以是平的。
   （**为什么恰好单调**：本文只做到「与缓存累积一致」这一步，没有逐条追踪状态文本的命中率。）
4. **快的那条不是准的那条。** `agreement_with_planner` 是
   「模型自己给出（未被 shield 替换）的动作与 planner 最优动作是否等效」
   （`brain.py:32,44` 与 `same_effect`，`brain.py:19-21`），
   而 planner 是确定性物理标注器。**CLM 0.658、Jev 0.987**；
   对应地，shield 介入的规模也差两个量级（见下）。
   上游自己把这条边界写在 `examples/t_rex/README.md:25-27`：
   *the survival number measures the combined system and the agreement and intervention rows
   measure the model*。所以 **`survived = 5` 不能当成 CLM 会玩这个游戏的证据**：
   它是一个「模型 + shield」的合成结果。

### 14.2 一个术语在两层是两个量：`shield_interventions`

从 §14.1 的表里看不出来、但值得单独记一笔：**`shield_interventions` 这个名字在 run 层与
summary 层指的不是同一个量**。

- run 层：`p["shield_interventions"]` = `Stats.interventions` = `decision.intervened` 之和
  （`pilot.py:492`、`brain.py:43`），即 **think 时刻**那次替换；
- summary 层：`Σ(shield_interventions + arrival_saves + emergency_saves)`（`run.py:131`），
  即**三个计数器之和**——落地时刻的替换（`pilot.py:169-173`）与紧急盾（`pilot.py:249-258`）
  也被加进来了。

复算：CLM 的 run 层是 46+86+86+72+74 = **364**，`arrival_saves` 是 760+807+889+1029+1034 = 4519，
`emergency_saves` 全 0，4519 + 364 = **4883** = summary 值。
Jev 更极端：run 层的 `shield_interventions` **五次全是 0**，
而 summary 的 28 全部来自 `arrival_saves`(18) + `emergency_saves`(10)。
所以 `summary.shield_interventions` 对 Jev 而言**不包含任何一次 shield 在 think 时刻的介入**。

### 14.3 committed 产物是产物的一个子集

`Pilot.Stats.summary` 算出来但**没有写进 JSON** 的字段（`pilot.py:514-544` 对比 `run.py:77-86`）：
`runs_finished`、`current_score`、`mean_finished_score`、`planner_ms_p50` / `planner_ms_p95`、
`crash_traces`、`usd`（按 token 单价算的钱，`pilot.py:543`）。
另外：

- **没有逐次延迟数组**，只有每 run 的 `latency_ms_p50` / `p95` / `model_ms_p50` 三个标量；
  原始序列只活在内存里（`pilot.py:476-482` 的两个 deque 与两个 list）。
  所以「p50 的中位数」是能被复算的，而「整体的 p50」**不能**——
  那需要所有原始延迟，JSON 里没有。
- **本批五次全是零死亡**，所以 `scores` 是单元素 `[697]`（`run.py:78`），`best_score` 全是 697。
  分数这一列因此没有区分度，两组记录都一样。
- **没有 `frames` 字段**。帧数只能从 `game_seconds = 60.0` 与 `FPS = 60`
  （`examples/t_rex/trex/engine.py:13`）换算出
  3600 帧；CLM 的 3341.8 次决策对应约 0.93 次/帧，Jev 的 1119.0 对应 0.31 次/帧——
  与两者的延迟比（9.1 倍）方向一致。
- `percentile` 是**最近秩** `ordered[min(len-1, int(q*len))]`（`pilot.py:507-512`），
  不是插值中位数；偶数长度时取偏上的那个。引用这两个 p50 时按此口径看。

### 14.4 一处文档与代码不符

`examples/t_rex/README.md:5-6` 写：

> both players send the same request through **one client** ([`examples/common.py`](../common.py),
> built on `clm.CLMClient`); only the base URL and API key differ.

代码不是这样：`run.py:38` 只从 `common` 拿 `load_env, write_json` 两个工具函数；
两个玩家的真实客户端是 `examples/t_rex/trex/backends.py` 里**自己那个 `httpx.Client`**
（`backends.py:110-115,125-137`），POST 的 body 是 `{"model", "state", "questions"}`。
`common.py` 的 `Judge`（带重试、sqlite 答案缓存、token 统计，`common.py:100-199`）
在 `examples/t_rex/` 下**零调用点**（全仓 grep `Judge` 在 t_rex 内只命中一句英文动词）。
后果是这份记录里没有重试、没有答案缓存，非 200 直接抛
（`backends.py:131-132`，`timeout=5.0` 于 `backends.py:100`），
延迟统计来自 `pilot.Stats` 而不是 `Judge.stats`。
`README.md:368` 对 `common.py` 的描述（*one client for both endpoints*）在**没有 t_rex 的那条路径上**是成立的。

### 14.5 `README.md` 示例数字的自洽性（引文核对，非实测）

`README.md:86-90` 打印了四个数，能做的只有自洽性核对：

| 打印值 | 核对 |
|---|---|
| `urgency.noul = 0.41022` | 落在 `[0,1]`，与 `noul: p_true` 的定义一致（`schema.py:137`） |
| `department.probabilities = {'billing': 0.93878, 'technical': 0.06122}` | **两者之和恰好 1.0**（打印到 5 位小数） |
| 同题的 `confidence`（未打印） | 按 `schema.py:122-128` 应为 0.93878 − 0.06122 = **0.87756**；这是对公式的套用，不是记录值 |
| `frustration.score = 1.98386` | 三档 rubric（0..2）下的期望档位，落在 `[0,2]` 且与 `score = Σ i·p_i` 一致（`schema.py:142`） |
| `usage.input_tokens = 38`，注释说冷缓存 106 | 与「只计未命中」的实现一致（§7.3、`README.md:90`） |

这些是**上游打印的值**，本文没有权重、没有复现它们，只核对了「它们与代码里的公式是否自洽」。

### 14.6 这一节能证明什么、不能证明什么

**能证明**（全部在 committed 产物或已打印产物上重算，命令见 §13.2）：

- CLM 与 Jev 在同一 harness、同一时长、同一在飞请求数下，端到端 p50 之比是 9.08，
  与上游宣称的「9×」吻合；
- CLM 的每条决策编码器 token 比 Jev 少 103 倍，且 CLM 侧存在逐 run 单调的缓存收益曲线；
- **CLM 与确定性 planner 的一致率是 0.658，Jev 是 0.987**；shield 介入（三个计数器合并）
  CLM 4883 次、Jev 28 次。所以这两组记录里的「5/5 存活」不能用来比较两个模型的决策质量；
- `summary.shield_interventions` 与 run 层的同名字段是两个量。

**不能证明**：

- 不能证明 CLM 的读法比别家「准」——这份记录里唯一的正确性信号是
  `agreement_with_planner`，而它只有 0.658，且那是**标注了答案**的题面
  （`labeled` 模式下最优动作的描述里就写着 `Best.`，`backends.py:40-47`）。
  换成 `--prompt guided` 会给出另一组数；本文没有那组数。
- 不能证明任何概率数值的正确性：全部概率来自上游 README 的引文。
- 也不能把它与另外八篇的准确率并排：那是 JevBench 一类数据集上的数，这里是游戏内一致率。
  上游的 JevBench 类结果（`README.md:138-162` 的 zero-shot 与 agentic 两组）**没有**随仓committed，
  本文无法核对。

---

## 15. 本文不做什么

- **不跑上游代码**：没有 `import clm`、没有起 `clm-serve`、没有跑训练或评测、
  没有下载 `CLM_v0.1-8B.pt`、没有 GPU。本文所有数字要么是 committed JSON 上的标准库算术，
  要么是已打印产物的手算核对，要么是 `文件:行` 引用。
- **不给前向产生的数值**：概率、准确率、`cos` 的实际分布、`scale` 训练后的实际取值，
  一律不填。凡是需要权重或 GPU 才能得到的量，本文只说机制。
- **未核实的三处**（明确列出，不用推测填空）：① vLLM 对 `Qwen/Qwen3-8B` 在 `--runner pooling`
  下默认池化哪个位置；② `truncate_prompt_tokens` 截断保留哪一端；
  ③ `README.md:454` 的 `505.0 MB` 与「24 GiB 卡的默认 2%」算出的 515.4 MB 差约 2%——
  两张池的字节数与行数比都能对上（§8.1、§13.2），但那个绝对值的来源（哪张卡、
  还是非默认预算）本文无法确认，本机也没有 `mem_get_info` 可取。
- **不评价选型**：本文只写这条读法怎么工作、代价落在哪里。

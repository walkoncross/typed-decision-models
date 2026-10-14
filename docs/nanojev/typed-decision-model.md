# NanoJev：K 条候选路径各读自己的 EOS 位，一个标量头吐出 K 个数

> 对应 README「关于各篇文档」一节
> 代码库：<https://github.com/TianyuCodings/NanoJev>

本文按 `scripts/` 与 committed 的 `research/`、`results/` 的实际内容逐环节说明这条实现的方式；文中出现的文件名与行号（`train_toy_decisions.py:115`、`predict_toy_decisions.py:264`、`serve_decisions.py:44` 等）均相对该仓库根目录。与 [RYOTIDE](../ryotide/typed-decision-model.md) 的对照是逐节的。

---

## §0 与 RYOTIDE 的分野

RYOTIDE 是**零训练**：冻结 LLM、运行期扫位置标定读点、把末位词表 logits 掩码到标签行再重归一化。NanoJev 把这条链上的三件事同时换掉了，而且换法与已收的另外二十条都不同。

| | RYOTIDE | NanoJev |
|---|---|---|
| 「typed」从哪来 | 分词器里的单 token 标签 | **请求声明的 `criteria`**（choice 的键 / score 的数组长度） |
| 候选是一条还是 K 条 | 一条序列，读末位的词表 logits | **K 条独立序列**，每条读自己的末位 |
| 词表维度 | 出现（掩码到标签行） | **从不出现**（骨干连 `lm_head` 都没有） |
| 读位置 | 运行期标定（扫 9 前缀 × 2 表面形式） | `encode` 拼序列时定死，运行期无标定 |
| 权重 | 冻结 | 自训（0.6B 全量微调，无 LoRA） |
| 置信度 | 读出来的 | **训出来的**；但发布路径上**没有任何二次改写** |
| 诊断量 | `marker_mass` | 训练读法上**概念不适用**；但同一个仓里另有一条基线读法**在报它** |

一句话立场：**NanoJev 把「一个位置」拆成了 K 个位置，代价是候选之间不再共享任何前缀**——于是 RYOTIDE 那条「掩码把越界的质量挡回去」的整个话题在这条链上没有对象，而换来的是「候选集完全由请求定义、数目只受 255 与 10 两个接口常量约束」。这也是它为什么敢把 `State` 段在每条候选路径里各写一遍（其 §7）。

还有一层不属于读法、但决定了这个仓长什么样：**它是一个从零建起来的复现项目，而不是一份服务**。`scripts/` 下 85 个文件里，真正构成读法的只有 `train_toy_decisions.py` 与 `predict_toy_decisions.py` 两个（`predict` 从 `train` 里 `importlib` 取 `DecisionModel`，`predict_toy_decisions.py:165-173`），其余大半是数据集构造、评测、探针、打包与「把同一件事换个姿势再证一遍」的检查脚本。所以本文的「实测」几乎全部落在**它自己提交的那些检查产物**上——这是本仓已收各篇里 commit 进仓库的证据密度较高的一档，但代价是这些产物的**权重大多没有随之发布**（其 §14.1）。

---

## 本文的事实来源

| 类别 | 来源 | 本文中的标记 |
|---|---|---|
| 代码引用 | 直接读 `scripts/*.py`、`scripts/*.mjs`、`web/*.js`、`docs/*.md` | `文件:行` |
| 可离线核对、不跑上游程序 | 仓内 `scripts/*.py` 的 sha256、`research/**`、`results/**` 里记录的字段 | 「实测」 |
| 在 committed 产物上重算 | `research/live_service_check_v3.json`、`research/parallel_example_v3.json`、`research/web_v3_live_browser_response.json`、`research/dynamic_candidates_v2.json`、`research/trained_invariance_timing.json`、`research/qwen_tree_check.json`、`research/navigation_v3_native_qwen_greedy.json` | 「实测（committed 产物）」 |
| 公式与常量 | `docs/*.md`、`README.md`、`data` 缺席时以 `config.json` 为准 | `文件:行` |
| 模型事实 | Hugging Face API、模型卡、随仓 `SHA256_MANIFEST.json` / `MODEL_MANIFEST.json`，以及对 `best.safetensors` 的 **HTTP Range 头部**（见 §14.4） | 「实测（Hub）」 |

**本文不给出任何由前向产生的概率数值**——那需要权重与 GPU，本机没有，上游也没有把 §14 那几条实例响应所对应的权重发布出来（§14.1）。凡引用小数，都是**上游自己在 committed 产物里记下的数**，并逐条标出来源。

---

## §1 端到端数据流

```
POST /api/evaluate            { "states": [ {id, state, questions:{qid: {type, instructions, criteria}}} ] }
  │  serve_decisions.py:32-52
  │  ① 长度 ≤ 2,000,000 B（:36） ② 同源检查（:37-38）
  │  ③ validate_request（predict_toy_decisions.py:34-82）——纯 schema，落 fail-loud
  │  ④ 限额 32 states / 96 questions / 256 candidate paths（:44）
  ▼
DecisionPredictor.predict(payload, batch_questions=0, temperature=1.0)   predict_toy_decisions.py:241
  │
  ├─ prepare_examples（:85-119）  每个问题 → K 条「候选路径」
  │     segments = [ "State:\n{state}\n", "Question type: {type}\nQuestion:\n{instructions}\n" ]
  │     prefix   = Σ encode(segment, add_special_tokens=False)          # 不走 chat 模板
  │     leaf_t   = prefix + encode("Candidate:\n{text}\nDecision:") + [eos_token_id]
  │     • choice : K = len(criteria)，text = "键: 描述"
  │     • boolean: K = 2，但**只有一条路径**，text = "The proposition is true."
  │     • score  : K = len(criteria)，text = 该档描述（索引不进文本）
  │     超 max_length → 抛错，不截断（:114-115）
  │
  ├─ complete_question_batches（:122-128）batch_questions=0 → 一次装下全部问题
  │
  ├─ 一次前向 DecisionModel.forward（train_toy_decisions.py:104-141）
  │     tokens = [Σ all leaf_tokens] 条序列，右填充到 width，attention_mask 显式给（:107-112）
  │     hidden = backbone(input_ids, attention_mask, use_cache=False).last_hidden_state
  │     leaves = hidden[arange, lengths-1]              ◄── 每条路径读自己的末位
  │     h      = scatter 到 [n_questions, kmax, hidden]（:116-124）
  │     z      = scalar(norm(h)).squeeze(-1)          ◄── 每题一个长度为 kmax 的标量向量
  │     choice 题再加 set-attention 的 delta（:127-133）
  │     boolean 题：logits = [0, z]                     （:134-138）
  │     返回 torch.stack(out).masked_fill(~valid, -1e9) （:141）
  │
  ├─ 逐题 softmax：scores = values[:k].float(); (scores / temperature).softmax(-1)   predict:261-264
  │
  └─ answer_from_probabilities（:131-146）
        boolean → p_true = p[1], value = bool(argmax)
        choice  → choice = ids[argmax]
        score   → score = Σ i·p_i（期望值），level = argmax（两个不同的数）
        Σp 与 1 的偏差 > 1e-5 → 抛错（:135-136）
  ▼
{ schema_version, checkpoint{base_model, base_revision, set_head}, temperature{value, fitted_by_this_command:false},
  execution{forward_passes, autoregressive_decode_steps:0, prefix_sharing:false, network_model_calls:0, …},
  states:[{id, answers:{qid: {type, probabilities, value/choice/level/score, p_true}}}] }
```

响应里**没有 `usage`、没有 `output_tokens`、也没有 `confidence`**——这一点与「同一个字段名在不同实现里是两个量」无关，是**这个字段在这里根本不存在**（其 §9、§10）。

---

## §2 「typed」从哪来

答案：**请求里那张 `criteria`，没有别的东西**。

- `choice`：`ids = list(q["criteria"])`（键序即候选序），`texts = [f"{key}: {q['criteria'][key]}" for key in ids]`（`predict_toy_decisions.py:99-100`）。校验器要求 2–255 项、键与描述都非空（`:70-75`）。
- `score`：`ids = [str(i) for i in range(len(criteria))]`，`texts = q["criteria"]`（`:101-103`）。校验器要求 2–10 项（`:76-81`）。代码里那句注释把设计意图写死了：`texts = q['criteria']  # Never inject ordinal index or adjacent levels.`（`train_toy_decisions.py:43`）。
- `boolean`：`ids = ['false','true']`，`texts = ['The proposition is true.']`——**只有一条语义路径**（`:97`，以及 `test_question_contract.py:103` 的 `assertEqual(len(example["leaf_tokens"]), 1)`）。可选的 `criteria` 只往 `segments[1]` 追加 `False criterion: …` / `True criterion: …`，且**顺序固定为 false 再 true**（`:106-109`），与调用方给键的顺序无关。

于是三类本仓已收各篇反复出现的约束在这里同时没有对象：

| 约束 | 在别处是什么 | 在这里 |
|---|---|---|
| 标签必须单 token | RYOTIDE / decider / 几乎全族 | **不存在**：标签不是 token，是「一条候选路径的末位」 |
| 字母表 / 码表上限（26 / 52 / 255） | Laya、SemIf-OpenJev、OpenJev(HF)、Nimble… | **不适用**：只有 2–255（choice）与 2–10（score）两个**接口**常量，与分词器无关 |
| 选项文本必须单 token | 没有一家要求过——要单 token 的从来是**标签** | **这里连标签都没有**：候选是一整段文本，长度只受 `max_length` 约束；而 `option_key` 也被写进文本，所以**改一个键就能移动答案**（§2 末段） |

代价与 RSI-Jev 是同一处，而且这里更直接：**选项键会被写进候选文本**（`f"{key}: {…}"`），改一个键就改了那条路径的 token，于是可以移动答案。`test_question_contract.py:146-151` 把这条性质测成了「重排选项**保住了每条具名 leaf**」——保住的是「同名键 → 同一条 leaf」，不是「答案不变」。

还有一处与全族都不同的是**候选数 K 本身是头的输入**：`set_head == 'attention'` 时，`log_k` 作为一个额外特征拼进 `set_project`（`train_toy_decisions.py:129-130`）。也就是说这个头**知道自己在几个选项里选**，而 RYOTIDE / Jeff / Kev / decider 那条线上的头对 K 无感（K 只体现在 softmax 的分母上）。

---

## §3 读位置

**读位置是 `encode` 拼序列时算出来的：每条候选路径的最后一个 token。**没有标定、没有候选位置、没有阈值。

```python
# train_toy_decisions.py:113-115（predict_toy_decisions.py:258-261 同构）
hidden = self.backbone(input_ids=tokens, attention_mask=attention, use_cache=False).last_hidden_state
leaves = hidden[torch.arange(len(paths), device=device), lengths-1]
```

`lengths-1` 指向的是每条 leaf 末尾那个 `tokenizer.eos_token_id`（`:56-57`）。所以：

- 「最后一个位置」在这条链上要**改写成「每条候选路径自己的最后一个位置」**——一次决策读 **K 个位置**，K 由请求定。这与 decider 的「K 个 `Answer k: (` 槽位」、RSI-Jev 的「1 + K 个位置」是同一类改写，但机制第三种：那条序列是**整条复制**出来的，`prefix` 一模一样，只在尾巴上换一个候选文本。
- 这条路**不经过 chat 模板**。`prepare_examples` 只调 `tokenizer.encode(t, add_special_tokens=False)`，`apply_chat_template` 在整条训练读法里一次都不出现（全仓只有基线读法用它，其 §11）。
- 因此 RYOTIDE 那个「扫 9 前缀 × 2 表面形式找 marker mass 最高的位置」的机件、以及本仓多篇绕不开的「`enable_thinking=False` 怎么接」这个问题，在这条链上**同时没有对象**：位置不是找出来的，是拼出来的。
- 位置对了没有，是**构造性**的，不需要运行期诊断量；真正需要诊断的是「候选文本有没有真的进到序列里」——那由 `test_question_contract.py` 在字符级分词器上离线断言（`:89-96`、`:134-151`）。

---

## §4 一次前向：三种题型、三种 logits 形状

主干只有一次 `self.backbone(...)`（`train_toy_decisions.py:113`），全部的候选路径平铺成一个 batch：

```python
paths = [ids for ex in examples for ids in ex['leaf_tokens']]        # :105
```

所以「一次前向」是**字面成立**的：`execution.forward_passes` 是批数（`:256-258`、`predict:276`），而每一批只有一次骨干调用。响应里对应 `autoregressive_decode_steps: 0`。这一点在 committed 产物里有直接证据：`research/live_service_check_v3.json` 两次调用都是 `forward_passes: 1`、`candidate_paths: 21`、`autoregressive_decode_steps: 0`。

**但三种题型吐出来的 logits 形状不一样**，这是本族里少见的一处：

| 题型 | 路径数 | logits | 有无候选间交互 |
|---|---|---|---|
| `choice` | K = len(criteria) | K 个标量 | **有**：set-attention 的 delta（`:127-133`） |
| `score` | K = len(criteria) | K 个标量 | 无（同一段代码里 `choice` 列表不含 score，`:127`） |
| `boolean` | **1** | **2** 个：`[0, z]` | 无 |

`boolean` 那一条是硬构造出来的：

```python
# Boolean has one semantic path and one scalar, representing logits [0,z].
out.append(F.pad(torch.stack([z[i, 0] * 0, z[i, 0]]), (0, kmax-2)))     # :134-138
```

所以 `p_true = sigmoid(z)`，`z` 就是 log-odds。离线评估器认这个约定并显式还原它：`extract_logits` 里 `value = [0.0, float(value)]  # scalar 必须表示 log(p_true/p_false)`（`evaluate_toy_decisions.py:47-48`）。

**set head 只在 choice 上生效**，它的输入是 `cat([h, log_k])`（`train_toy_decisions.py:129-130`），自注意力对候选轴做混合，`set_output` 是**零初始化**（`:101-102`：*Only the final residual projection starts at zero; its upstream layers are nonzero*），所以训练第 0 步时 `delta ≡ 0`，整条链退化成「K 个独立标量」——和 `set_head='none'` 逐位相同。这是一处**构造性的恒等起点**，与 Jeff 的 `slice_init_`（让训练起点等于零样本读法）是同一个设计手法，只是这里的对照物是「没有候选间交互的头」。

---

## §5 掩码 + softmax

这里要把三件事分开，因为它们的对象各不相同。

**① 「掩码到已知标签集」概念不适用。** 头的输出维度就是 K：`scalar: Linear(hidden, 1)` 逐候选打分（`train_toy_decisions.py:92`），所以「标签之外的质量」在数学上不存在。代码里唯一那个 `-1e9` 是

```python
return torch.stack(out).masked_fill(~valid, -1e9), valid      # :141
```

对象是**批内候选轴的补位**（`valid[i, :len(candidate_ids)] = True`，`:123`），不是标签集；而且下一步 `scores = values[:k]`（`predict:261`）直接把它切掉了，它连 softmax 都进不去。所以 `marker_mass` 这个词在这条链上找不到落点——与 Kev / decider / Nimble / pplx-decider / RSI-Jev / JevAny-native 同一个结论，但机制是**又一种**：那几家是「头只输出 K 个标量」，这里是「**每条候选一条独立序列 + 一个共享的标量投影**」。差别在候选之间有没有共享计算：Kev 的 pointer 头让候选互相关注同一段 state，这里的 K 条序列在骨干里**互相看不见**。

**② 重归一化的对象是候选，而且是构造性的。** `probabilities = (scores / temperature).softmax(-1)`（`predict:264`）落在长度恰为 K 的向量上，所以 Σp = 1 是结果而不是被维护的不变量。校验器仍留了一道门：`abs(fsum(probabilities) - 1.0) > 1e-5 → raise ValueError("模型概率总和不为1")`（`:135-136`），且**不修不补**——抛错、不返回部分预测。

**③ 「不物化词表维度」在这条链上是彻底的。** 检查点里 322 个张量、`lm_head` 一个键都没有（§14.4 在 Hub 上的实测）；骨干是 `lm.model`（`train_toy_decisions.py:307`）或 `AutoModel.from_config(body_config)`（`predict_toy_decisions.py:222`），后者的注释把理由写明：`from_config 只构造结构；全部参数由 best.safetensors 加载，不用 from_pretrained 下载底座。`

**三个服务旋钮会改数值，但都不在这段代码里**：`temperature`（默认 1.0，其 §9）、`precision`（bf16/fp32，`:203-205`、`:257`）、`batch_questions`（改批组成，其 §14.2）。它们都不是「对发布值做的事后加工」，是**入参**——这也意味着同一个请求换一个 `--batch-questions` 就会得到不同的概率，而上游把它明确写进了执行元数据（`execution.batch_questions_limit`）。

### §5.1 三种 ECE 定义在同一个仓里并存

这一条值得单独记，因为它正是本仓总表 §5 那一类口径差的又一例：

| 位置 | 分箱 | 「置信度」对齐的是谁 |
|---|---|---|
| `evaluate_toy_decisions.py:120-163` | **10** 等宽，最后一箱含 p=1 | **正确答案**（`confidence` vs `accuracy`） |
| `evaluate_pipeline_decisions.py:134-138` | **10** 等宽 | 两个都要：`deterministic_ece` 对齐 one-hot 真值，`analytic_expected_ece` 对齐**该题已知的条件概率** |
| `benchmark_calibrated_learning.py:78-90`（`--ece-bins` 默认**就是 15**，`benchmark_calibrated_learning.py:210`） | **15** 等宽 | 正确答案 |

第二行那个 `analytic_expected_ece` 与另外两个**不是同一个量**：它算的是「预测置信度」与「已知 q 在所选事件上的概率」之差（`:164` 的注释写明 *no sampled outcomes*），所以它不需要观测结果，也就不是通常意义上的 ECE。

---

## §6 顺序偏置

**结构上就没有这个变量**，而且比 Laya 更彻底——Laya 是在训练时让位置对选项不可见，这里是**每条候选各占一条序列**，重排候选列表根本不改变任何一条 leaf 的 token。

唯一的跨候选机件是 set head 的 self-attention，它对候选轴是**等变**（permutation-equivariant）而不是不变的：把候选顺序换一下，输出跟着换一下，折回原名之后应该逐位相同。

上游把这件事测了，而且测出了一个**不是 0 的残差**：

- `candidate_permutation_mapped_back`：max logit |Δ| = **0.0**，max probability |Δ| = **1.19e-7**，argmax 翻转 **0**（`research/trained_invariance_timing.json`）。
- 那个 1.19e-7 不是「有偏置」，是**浮点求和顺序**：attention 对 K 个候选做的加权和在重排后换了一个加法次序，float32 下最后几位不一样。理论值 0、实测 1.19e-7，两者同时报出来，比只报一个「0.0」诚实。
- `boolean` 题根本不参与：`permute_candidates` 里 `if ex['type'] == 'boolean': continue  # Boolean is one scalar with a fixed [false,true] interface.`（`check_trained_invariance.py:112-113`），而它本来也只有一条路径。
- 训练期**没有做任何顺序打散**——与 Kev / pplx-decider / metask-jev-rain / JevAny 都不同（那几家靠每个 epoch / 每个 step 打散来消解训练期的顺序信号）。这里不需要：候选顺序压根不进入任何一条序列。

`score` 题另有一处相关的设计：档位索引不进文本（`train_toy_decisions.py:43`），而 `answer_from_probabilities` 报的 `level` 是 argmax 下标、`score` 是 Σ i·p_i（`predict:144-145`）。所以**移动某一档描述的位置会改它报出来的下标，但不会改它的 leaf**——`test_question_contract.py:154-165` 就是把这条断言写成了测试。

---

## §7 多问题共享 state

**不做。而且是本仓已收各篇里「不做」得最朴素的一条：state 既重复编码，也重复计费。**

- `execution.prefix_sharing` 是一个**字面量 `False`**（`predict_toy_decisions.py:277`、`train_toy_decisions.py:259`），配置里也写着 `'parallelism': 'flat complete candidate paths in one backbone forward; no prefix sharing'`（`train_toy_decisions.py:316`、`train_pipeline_decisions.py:502`）。
- 一条 state 在一个问题里被复制 K 次（每条候选路径一份），一个问题在两个问题里再复制一次。没有任何 token 级的缓存。
- **但同一次请求里的所有问题确实在同一次前向里跑**：`batch_questions=0` 时 `size = len(examples)`（`predict:125`），服务端也把它当默认（`serve_decisions.py:45` 不传）。所以省下的是 kernel 启动与 Python 循环，不是 token——这与 RYOTIDE 的 `run_many()`（公共前缀只 prefill 一次、计费只算一次）和 SemIf-OpenJev 的 `shared`（一次 prefill + cache 复制）是两回事。

**「题与题互不可见」在这里是可以数值上证明的**，因为每条 leaf 是一条独立序列、骨干里没有跨题注意力：

- `unrelated_state_mutation_isolation`：把另一个 state 的 state 前缀里三个 token 换掉，本题 21 个问题的答案 max logit |Δ| = **0.0**、max prob |Δ| = **0.0**（`research/trained_invariance_timing.json`；变异构造见 `check_trained_invariance.py:122-146`）。
- `state_and_question_reordering`：把状态与问题整个反序重排，max logit |Δ| = **0.0**、max prob |Δ| = **0.0**。
- 对照是 Clef：那里 4 层 decoder 在 `fields` 之间做自注意力，删掉一题剩下题目的答案**可以变**。

### §7.1 前缀共享被独立验证过，但服务路径不用它

这是本文的一处「证得出来 ≠ 接进来了」：

- `check_tree_attention.py` 用纯 Python 的 2 层玩具注意力做树形前向 vs 平铺前向的对照，产物 `research/tree_attention_check.json` 记 `reorder_max_abs_error: 0.0`、`forward_max_abs_error: 0.0`、12 项有限差分导数最大误差 `0.0`，另有两个**反例探针**：故意用错的全局因果掩码会带来 `0.3040` 的误差、故意用错的 RoPE 位置会带来 `0.2150`——即这个检查对「掩码写错」是敏感的，不是恒过。
- `check_qwen_tree.py` 在**随机初始化的微型 Qwen3** 上做同一对照（`vocab 128 / hidden 64 / 2 层`，`research/qwen_tree_check.json`）：9 条 leaf、平铺 66 token、树形 38 token；float32 下 leaf hidden max |Δ| = **4.8e-7**、logits max |Δ| = **2.2e-7**、loss 差 **0.0**、25 个参数的梯度 max |Δ| = **6.3e-7**；bf16 下 leaf hidden / logits / 重排全部 **0.0**、梯度 max |Δ| = **0.0098**。
- 而**发布路径一次都没用它**：每个 committed 产物都写着 `prefix_sharing: false`、`tree_attention: false`，`check_trained_invariance.py` 的开头就注明 *Inference repeats full candidate paths; it does not implement tree attention or prefix KV sharing*（`:5-6`），`train_toy_decisions.py:5` 的模块 docstring 则说 *This reference implementation repeats prefixes; tree sharing is a separately verified optimization.*

也就是说：**优化被验证到「逐位相同（bf16 下连逐位都相同）」的程度，但没有接进任何一条能跑出数的路径**。代价是可以量出来的——`research/trained_invariance_timing.json` 的 timing 段：1 状态 3 题 7 路径 p50 **26.89 ms**（111.6 题/秒）、8 状态 24 题 68 路径 p50 **84.72 ms**（283.3 题/秒）。这是「warm GPU forward + 张量装配 + 主机到设备拷贝」，不含分词、不含 HTTP。

---

## §8 服务化 / fail-loud 风格

服务面薄得只有 73 行（`scripts/serve_decisions.py`），但守卫全在**读法层**，不在 HTTP 层：

| 守卫 | 位置 | 违约时 |
|---|---|---|
| 请求体 0 < len ≤ 2,000,000 B | `serve_decisions.py:36` | 400 |
| `Origin` 与 `Host` 同源，否则拒 | `:37-38` | 400 |
| 限额 32 states / 96 questions / 256 candidate paths | `:44` | 400 |
| `{"states": [...]}` 且**只**含 `states` 键 | `predict_toy_decisions.py:34-36` | 400 |
| state `id` 唯一非空、`state` 非空 | `:44-51` | 400 |
| `choice` criteria 2–255、`score` 2–10、`instructions` 非空 | `:70-81` | 400 |
| JSON 拒绝重复键、拒绝 `NaN`/`Infinity` | `:12-27` | 400 |
| 候选路径超过 `max_length` | `:114-115` | 400（**不截断**） |
| logits 出现非有限值 | `:262-263` | 400（**不返回部分预测**） |
| Σp 与 1 偏差 > 1e-5 | `:135-136` | 400 |
| 其他异常 | `serve_decisions.py:50-51` | **500**，文案写明 *No teacher fallback was used.* |

`GET /api/health` 返回 `{ready, model_loaded_once: true, provider_calls: 0}`（`:23-24`），`GET /*` 是静态页，路径穿越由 `target.is_relative_to(web_root)` 挡掉（`:27`）。这些都有一份**不跑 GPU** 的合同检查：`check_server_contract.py` 用显式标注的 CPU 替身（`Engine` 类）起真 HTTP server，断言 5 条（health / 静态页 / `../.env` 被拒 / 正常推理路由 / 无效 schema 在引擎之前被拒），产物 `research/server_contract_check.json` 记 `engine_calls: 1`。

`check_live_service.py` 则是真 GPU 版：连发两次同样的请求，断言 `forward_passes==1 && autoregressive_decode_steps==0 && network_model_calls==0 && persistent_model_load_count==1`、`inference_call_index` 递增 1，并把逐题概率的重复差记成 `max_repeated_probability_delta`——`research/live_service_check_v3.json` 里这个值是 **0.0**。

没有的东西也要写下来：**没有鉴权、没有版本端点、没有 prompt 哈希、没有 `prompt_hash` 之类的身份指纹**。唯一的身份线索是响应里 `checkpoint.directory` 那个**服务端本地路径**（`predict_toy_decisions.py:268`）——它既不是内容哈希，也不指向任何可下载的东西。这与 Nimble 把 prompt 源码 sha256 钉进 `schema_config.json` 是两个极端。

### §8.1 浏览器演示：只重放，唯一的实时通道是一个表单

四个页面（`web/index.html` 的决策实验室、`arcade.html`、`comparison.html`、`side-by-side.html`）的 JS **没有任何模型代码**，各自只 `fetch` 一份随仓 JSON 并渲染：`app.js:365` → `demo_results.json`、`arcade.js:354` → `arcade_results.json`、`comparison.js:55` → `comparison_results.json`、`side-by-side.js:312` → `side_by_side_results.json`。`web/app.js:3` 的注释把这件事写明：*All model outputs come from recorded results or the live service.*；`web/arcade.js:1` 更直接：*Recorded-state playback only. The data file owns every action and outcome.* 生成这些产物的一侧也不跑模型：`build_demo_artifact.py:2` 是 *Package actual evaluated episodes; never synthesize replacement predictions.*，`build_arcade_demo.py:4` 是 *No model inference, API calls, episode search, or target generation is performed.*（它导入环境模块只为了 CPU 重放校验轨迹，`:70-107`、`:196-251`）。

唯一的实时通道是页面上的 **Live endpoint** 表单：`app.js:421-433` 把文本框里的 JSON `POST` 给 `/api/evaluate`，也就是 §8 那个本地 Python 服务。三条约束值得记：①`location.protocol === 'file:'` 时**直接早退**并提示 *Start an HTTP model service before using the live endpoint.*（`:427`）——**不是**降级到录制数据；②超时 120 s（`:431`）；③收到 404/405/501 时抛 `service_missing` 并提示静态服务器不提供实时 API（`:434`、`:448`）。而录制数据的加载失败（`file://` 打开）走的另一条路：`:388` 提示 *Browsers cannot load the results JSON from file://*，然后渲染空态并禁用模型选择，`:381` 那句 *The page does not insert simulated successes.* 说明它宁可空着也不编。

展示层对概率只做三件事，**都不改数值**：①**显示舍入**（`app.js:239` 的 `(p*100).toFixed(2)%`，其余三个页面 1 位小数）；②**条宽限幅** `Math.min(1, Math.max(0, p))`（`:241`，只影响 CSS 宽度）；③**展示上限 255 条**（`:234` 的 `entries.slice(0, 255)`）。**不重归一化**这一点是写在界面上的：`:269` 判定 `mass !== null && Math.abs(mass - 1) > 1e-5`，`:272` 的文案是 *The recorded probabilities sum to … and are displayed without renormalization.*——**只提示，不缩放**；`comparison.js:38` 同构。这与 §14.6 里对照臂「显式归一化并记账」恰好相反：**演示层选择展示原值并声明它可能不为 1**。

三条浏览器端的检查也随仓提交，且都不跑 GPU：`research/web_live_browser_check.json` 与 `web_v3_live_browser_check.json` 记的是「真实 Chrome POST 到真服务」的往返（`http_status: 200`、`page_errors: []`、`provider_calls: 0`、`latency_ui: "132 ms · 浏览器往返"`），`research/web_empty_browser_check.json` 记的是**空数据与错误态的六条断言**（缺数据时显示空态并禁用播放与模型选择、并行视图不捏造批次、非法 JSON 本地拒绝、静态服务明确提示实时 API 缺失、390 px 窄屏无横向溢出、错误 schema 明确提示）。

---

## §9 置信度从哪来，又怎么被改写

### 9.1 是训出来的

头是训练产物，训练同时在改数值**和**改 argmax。这一点由 loss 的形态直接决定：`grouped_calibrated_loss` 在每题自己的完整候选集上做 `log_softmax`，目标是软分布或 one-hot（`calibrated_objectives.py:63-85`），梯度含 `(softmax(z) − t)` 项——不是某个「读出之后再校正」的环节。

### 9.2 但发布路径上一个数都没被改写

有一条很短的链表，短到只有一环：

| 环节 | 谁在改 | 证据 |
|---|---|---|
| `z` → `/temperature` | **调用方**（CLI / serve 的入参） | `predict_toy_decisions.py:264` |
| → `softmax` | 无 | 同上 |
| → 响应 `probabilities` | **无** | `answer_from_probabilities:131-146` 只做 argmax 与 Σ i·p_i |
| → 响应其他字段 | **无**（`confidence` 字段不存在） | 同上 |

温度这一格要说得更死：**默认就是 1.0，而且响应自己声明它不是拟合来的**——

```python
"temperature": {"value": float(temperature), "fitted_by_this_command": False,
                "note": "显式应用给定标量；默认1不表示模型已校准。"}          # predict_toy_decisions.py:270-271
```

拟合温度的实现**在仓里，但在另一条链上**：`evaluate_toy_decisions.py:223-243`，规则是「只用 calibration split、81 点对数网格 `[0.25, 4.0]`、同损失取离 1 更近的那个、方法名 `shared_scalar_logspace_grid`」，产物字段还带 `at_grid_boundary`（`:242`）。而后加的那条 pipeline 评估器**明确不干这件事**：`'selection': 'This evaluation does not fit temperatures or select checkpoints.'`（`evaluate_pipeline_decisions.py:169`）。训练产物的 `summary.json` 里写的也是 `temperature: 1.0, temperature_fitted: False`（`train_pipeline_decisions.py:559-560`）。

所以：**温度是「读出来的」这一侧唯一能改数值的旋钮，而它在这条链上没有被拧过**。这不是「校准过了」，是「校准这件事被留在了服务之外，并且默认值不等于已校准」。

### 9.3 唯一叫 `confidence` 的地方在离线评估器里

`evaluate_pipeline_decisions.py:123` 会算 `s['confidence'] = p[best]`，但只对 `deterministic_truth` 题算，而且 `:127` 显式把 `confidence` / `target_top_correctness` / `analytic_confidence` / `analytic_top_probability` 从被平均的字段里剔除——它们只进分箱 ECE，不进汇总数字。这就是它与全族其它篇的分野：**别家是「同名两个量」，这里是「这个名字根本不出现在服务面上」**。

### 9.4 teacher 一侧倒是有 `confidence`

从 TypeSafe 网关取标签的那条链把 `providerMetadata.typesafe.confidence` 原样落盘（`teachers.mjs:194`），连同 `rounding.probabilityDecimals`（`:189`）。但它**只进标签文件，从不进本地响应**——本地读法对 teacher 的依赖只在训练期，且训练期对 teacher 概率的处置是「identity rounded proxy 或整条隔离」，见 §14.6。

---

## §10 四条不变量的落位

| RYOTIDE 的不变量 | 在 NanoJev 里的状态 | 依据 |
|---|---|---|
| **一次前向** | **成立**，且服务面把它当元数据上报 | `execution.forward_passes`（`predict:276`）、`autoregressive_decode_steps: 0`（`:277`） |
| **只读最后一个位置** | **要改写**：读的是**每条候选路径自己的末位（EOS）**，一次决策 K 个位置 | `train_toy_decisions.py:115`；位置由 `prepare_examples` 拼出来，运行期无标定（其 §3） |
| **掩码到已知标签集** | **概念不适用**（第 N 种机制）：头输出维度就是 K，且 K 条序列在骨干里互不可见 | `train_toy_decisions.py:92`、`:141`；其 §5 |
| **重归一化** | **成立，对象是候选**；Σp=1 是构造结果，另有 1e-5 容差的门兜底 | `predict:264`、`:135-136` |
| **`output_tokens = 0`** | **字段不存在**：响应里没有 `usage` | 全仓对 `output_tokens` / `input_tokens` 零命中；README 与各产物都没有这个字段 |

另有一处**本族新增**、需要单独记的：**「一次前向」还按候选数分档**。这里不是像 metask-jev-rain 那样按 128 个 token id 一档拆探针，而是**按 `batch_questions` 拆批**——`--batch-questions N` 让同一批问题分成 `ceil(题数/N)` 次前向。`check_dynamic_candidates.py` 的产物里 `actual_forward_calls: {"decision_model": 1, "backbone": 1}` 就是「5 个问题、K 从 2 到 255、346 条路径、一次前向」的实测记录（`research/dynamic_candidates_v2.json`）。

---

## §11 第二条读法：零训练的字母槽位基线（本族新增的一处「一仓两读法」）

**同一个仓里还有一条完全独立、零训练的读法**，它是作为对照臂存在的，且被服务化了：

```python
# train_toy_decisions.py:194-233  frozen_native_baseline
label_ids = [tokenizer.encode(letter, add_special_tokens=False)[0] for letter in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ']
...
prompt = f"State:\n{state}\nQuestion:\n{instructions}\nOptions:\n{options}\nAnswer with only the option letter."
ids = tokenizer.apply_chat_template([...], add_generation_prompt=True, enable_thinking=False, ...)
...
h = lm.model(input_ids=tokens, attention_mask=mask, use_cache=False).last_hidden_state
h = h[torch.arange(len(paths)), lengths-1]
# Only the required output rows, not a full vocabulary projection.
z = F.linear(h, lm.get_output_embeddings().weight[label_ids]).float()      # :231
```

这条读法**和 RYOTIDE 是同一族**：单 token 字母标签、末位 hidden、只投影需要的行（`:231` 的注释 *Only the required output rows, not a full vocabulary projection*）。它的服务化版本是 `NativeQwenPredictor`（`evaluate_native_qwen_navigation.py:93-160`），而它比 RYOTIDE 多报**一个量**：

```python
offered       = logits[i, indices]                        # 只取候选那几个 token id
conditional   = offered.softmax(-1)                       # 条件分布
unconditional = (offered - logits[i].logsumexp(-1)).exp() # 全词表下的无条件概率
'offered_token_mass': math.fsum(unconditional)            # :143-152
```

`offered_token_mass` 就是 RYOTIDE 的 `marker_mass` 的同类量——**A–D 四个字母在整份词表里占的份额**。而且它在这条链上的取值说明了一件事：**冻结的 Qwen3-0.6B 在「Answer with only the option letter.」这种问法下，97%–99% 的质量确实落在四个字母上，剩下 1%–3% 落在别处。**

实测（`research/navigation_v3_native_qwen_greedy.json`，1,478 次决策）：

- `Σ native_option_unconditional_probs` 与 `offered_token_mass` 逐条相等（1478/1478，容差 1e-6）。
- `probabilities` = `softmax(native_option_logits)` 逐条成立，最大偏差 **7.6e-8**（float32 softmax 的舍入）。
- `choice` = argmax，1478/1478。
- `offered_token_mass` 区间 **[0.97242, 0.99479]**，中位数 **0.98905**。
- 提示长度 216–315 token；`A/B/C/D` 的 token id 是 **32/33/34/35**（`research/navigation_v3_native_qwen_preflight.json` 的 `token_ids`，且该文件记 `GPU_calls: 0`——它是纯分词预检）。

**两条读法的处置差异**（这是本节真正的信息量）：

| | 训练读法（§2–§9） | 基线读法（本节） |
|---|---|---|
| 词表投影 | 从不发生（`lm_head` 不在检查点里） | **每次读 `logsumexp` 都要整份词表**（`:133`、`:143`） |
| chat 模板 | 从不使用 | 使用，且显式 `enable_thinking=False`（`:40-42`） |
| 位置 | 每条候选路径的末位 | **整条 prompt 的末位**（`:131`） |
| 候选上限 | 2–255 / 2–10 | **2–4**（`:24`，且候选按字典序映射到 A–D） |
| 单 token 校验 | 不适用 | 有：`len(encoded)==1 and decode(encoded)==label`，四个 id 互异（`:63-69`） |
| `marker_mass` 同类量 | **概念不适用** | `offered_token_mass`，逐题上报 |

同一个仓、同一个基座、同一个环境，**一条读法把词表整个扔掉，另一条每次都要走一遍词表**——本仓已收的各篇里，「两条读法并存」还有两处，但都不是这一种：JevAny 是**同一个运行时的两条链**（同一条 `lm_head` 上的两种取法，换一个 `--readout` 开关），RSI-Jev 是把零样本读法 `LogprobReadout` **保留在同一个架构模块里当基线**（零可训练参数，与训练头并列在 `rsijev/arch.py`）——那两条都还在**同一种架构**里。这里是**两种根本不同的架构**放在同一个仓里当对照：一条彻底不要词表，另一条每次决策都要走一遍词表。

---

## §12 文件对照表

| 文件 | 行数 | 职责 |
|---|---:|---|
| `scripts/train_toy_decisions.py` | 363 | **`DecisionModel` 的定义就在这里**（`:86-141`）；另有冻结基线读法 `frozen_native_baseline`（`:194-233`）、基准计时（`:236-259`）与训练主循环（`:262-363`）。推理侧靠 `importlib` 从这里取 `DecisionModel`（`predict_toy_decisions.py:165-173`） |
| `scripts/predict_toy_decisions.py` | 327 | **推理主体**：`validate_request`（`:34-82`）、`prepare_examples`（`:85-119`，与 trainer 逐段同构）、`complete_question_batches`（`:122-128`）、`answer_from_probabilities`（`:131-146`）、`DecisionPredictor`（`:176-282`） |
| `scripts/train_pipeline_decisions.py` | 567 | 后加的 pipeline trainer：三种目标（`teacher` / `gold_distribution` / `observed_outcome`）× 三种损失（`ce` / `brier` / `paired_brier_pg`）、complete-question 微批打包（`:236-254`）、dev-only checkpoint 选择 |
| `scripts/calibrated_objectives.py` | 85 | 三种损失的实现；`paired_brier_policy_loss` 是**自陈的 RLCD 研究候选**，不是官方配方（`:29-60`，docstring 首行写明） |
| `scripts/evaluate_toy_decisions.py` | 336 | 纯标准库离线评估：**唯一会拟合温度的地方**（`:223-243`）、10 箱 ECE、按 state 聚类的 bootstrap |
| `scripts/evaluate_pipeline_decisions.py` | 178 | 后加的评估器：三种目标分开算、两种 10 箱 ECE（`:134-138`），**明说不拟合温度、不选 checkpoint**（`:169`） |
| `scripts/serve_decisions.py` | 73 | 标准库 HTTP 服务：`/api/health` + 静态页 + `POST /api/evaluate`，限额与错误码 |
| `scripts/check_server_contract.py` | 45 | 用 CPU 替身起的 HTTP 合同检查（**不跑 GPU**） |
| `scripts/check_live_service.py` | 30 | 真 GPU 版：连发两次、比较逐题概率，产出 `max_repeated_probability_delta` |
| `scripts/check_trained_invariance.py` | 352 | 6 类不变性 + 精度诊断 + 计时，产物 `research/trained_invariance_timing.json` |
| `scripts/check_dynamic_candidates.py` | 174 | 一次前向里 K ∈ {2,5,20,64,255} 的输出基数检查 |
| `scripts/check_tree_attention.py` | 202 | 纯 Python 玩具验证树形前向 = 平铺前向（**含两个反例探针**） |
| `scripts/check_qwen_tree.py` | 332 | 随机微型 Qwen3 上的同一验证 + 梯度对照 |
| `scripts/evaluate_native_qwen_navigation.py` | 233 | **第二条读法**的服务化：字母槽位基线 + `offered_token_mass` |
| `scripts/teachers.mjs` | 238 | 唯一调用 TypeSafe 网关的客户端：`{state, questions}` 进、`native_probs` + `rounding` + `confidence` 出（`:180-198`） |
| `scripts/test_question_contract.py` | 231 | 字符级分词器上的离线编码合同测试（**不 import torch、不下载**） |
| `scripts/benchmark_calibrated_learning.py` | 284 | CPU 上的三种损失对照臂基准（**15 箱** ECE） |
| `web/*.js`（4 个：app / arcade / comparison / side-by-side） | — | 浏览器演示：**只重放随仓 JSON**，唯一的实时通道是 `app.js:433` 的 `/api/evaluate`（其 §8.1） |

---

## §13 亲手跑一遍

以下命令都取自上游自己的产物／docs，**本文没有执行它们**（需要 CUDA、需要权重）。

**服务与推理**（`docs/SCALED_GAMES.md`、`docs/GAME_RELEASE.md:64-81`）：

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/serve_decisions.py \
  --checkpoint-dir checkpoints/NanoJev --precision bf16 --disable-native-triton
# → http://127.0.0.1:8765 ，POST /api/evaluate

CUDA_VISIBLE_DEVICES=0 python scripts/predict_toy_decisions.py \
  --checkpoint-dir checkpoints/NanoJev --input research/toy_inference_example.json \
  --temperature 1.0 --precision bf16
```

**不跑 GPU 的那几条**（`docs/TYPESAFE_CONTRACT.md:28-32`、`docs/RLCD_EXPERIMENT.md:73-94`）：

```bash
python3 -m unittest discover -s scripts -p test_question_contract.py -v   # 只用字符级分词器
python3 scripts/train_pipeline_decisions.py --self-check                  # 纯 CPU 的 schema/梯度检查
python3 scripts/train_pipeline_decisions.py --input <data> --validate-only
python3 scripts/check_server_contract.py                                  # CPU 替身 + 真 HTTP
python3 scripts/test_calibrated_objectives.py --mc-repeats 2000 --output results/calibrated_objectives_check.json
python3 scripts/benchmark_calibrated_learning.py --steps 600 --seeds 17 18 19
```

**从 Hub 取权重**（`docs/GAME_RELEASE.md:28-47`、`TRAINING_RECIPE.md`）：

```python
from huggingface_hub import snapshot_download
snapshot_download(repo_id="C-Tianyu/NanoJev", local_dir="checkpoints/NanoJev",
                  allow_patterns=["variants/local_atomic_seed17/*", "GAMES_MODEL_MANIFEST.json"])
```

模型的加载方式是**先建结构、再灌权重**，理由写在注释里（`predict_toy_decisions.py:221-226`）：`from_config` 只构造结构，全部参数由 `best.safetensors` 加载，**不用 `from_pretrained` 下载底座**。HF 模型卡的加载示例走的是另一条：从发布目录里 `sys.path.insert(0, path + "/source/scripts")` 再 `from predict_toy_decisions import DecisionPredictor`——而那个 `source/scripts/predict_toy_decisions.py` 与本地 clone 那份**逐字节相同**（其 §14.5）。

---

## §14 实例走查与机械核对

### 14.1 三条实例响应的来源，与一处必须说清的限制

本文的三条实例来自仓内 committed 的完整响应：

| 产物 | 状态数 / 题数 / 候选路径 | `inference_call_index` | 服务端评测耗时 |
|---|---:|---:|---:|
| `research/live_service_check_v3.json` responses[0] | 3 / 9 / 21 | 1 | 0.2852 s（首次，含热身） |
| 同上 responses[1] | 3 / 9 / 21 | 2 | 0.0475 s |
| `research/web_v3_live_browser_response.json` | 3 / 9 / 21 | 3 | 0.0657 s（**真实浏览器 POST**，`web_v3_live_browser_check.json` 记 `page_errors: []`、`http_status: 200`） |

三条用的是**同一份 checkpoint**（目录名 `runs/v3_gold_coords_multi_seed17`），而它的 sha256 是 `8375fcde…`（由 `research/parallel_example_v3.json` 的 `checkpoint_sha256` 与 `research/navigation_v3_regression.json` 的 `runs.v3_gold_coords_multi_seed17.checkpoint_sha256` 两处记录同一值）。

**限制**：这个 checkpoint **没有发布**。Hub 上现存的两个 root 级权重分别是 `fff62d14…`（旧 root，= `runs/v3_teacher_coords_multi_seed17`，见 `research/nanojev_comparison_verification.json` 的 `checkpoint_sha256` 与 `navigation_v3_regression.json`）与 `f68c47d6…`（当前 `unified-games-v1` root，见 `SHA256_MANIFEST.json`），另有 `stage1/best.safetensors` = `231b5178…`（= `runs/v2_teacher_seed17`，由 `research/dynamic_candidates_v2.json` 与 `navigation_v3_regression.json` 两处互证）。所以**这三条实例无法在已发布权重上重放**——本文只做「在响应内部自洽性」的复算，不做重放。

### 14.2 响应内部的三条恒等式，与「重复请求逐位相同」

在 `live_service_check_v3.json` 的 9 条答案上逐条重算（纯标准库）：

| 检查 | 结果 |
|---|---|
| `Σ probabilities = 1`（容差 1e-6） | 9/9 通过 |
| `boolean`：`p_true == probabilities["true"]`，`value == (argmax == "true")` | 3/3 通过 |
| `choice`：`choice == argmax` | 3/3 通过 |
| `score`：`score == Σ i·p_i`（**逐位相等，容差 0**） | 3/3 通过；`level == argmax` | 
| 重复请求：`max_repeated_probability_delta` | **0.0** |

第三条的逐位相等值得单独看一眼，因为它涉及 float32 求和次序：`risk` 的 `score = 0×0.9999549388885498 + 1×2.6688041543820873e-05 + 2×1.834240356402006e-05 = 6.337284867186099e-05`，与响应里那个数**完全相同**。三题都如此。

第四条还可以再加一条更强的：把 `web_v3_live_browser_response.json`（第 3 次调用，来自真实浏览器）与 `live_service_check_v3.json` 的 responses[0]（第 1 次调用）逐题逐候选比，**最大 |Δp| = 0.0**——24 个数（3 状态 × 3 题，候选数 7 + 8 + 9）全部逐位相同。也就是说：**同一形状的请求重复三次（含一次跨进程、跨 WebSocket 的浏览器往返），概率逐位不变**。

### 14.3 batch 形状是一个被测量过的自由度

上面那句「逐位相同」有一个前提：**请求形状相同**。换一个形状就不一样了。把 `parallel_example_v3.json`（6 状态 / 18 题 / 44 路径，一次前向）与 `live_service_check_v3.json`（3 状态 / 9 题 / 21 路径）里**共有的那 3 个 state** 对齐比较：

| | |
|---|---|
| 最大 \|Δp\| | **0.002733**（`known_chance_v2:test:1` 的 `outcome`、候选 `white`） |
| 次大 | 0.001165（同题 `group` 的档位 `2`）、0.001048（同题 `outcome` 的 `cyan`） |
| argmax 翻转 | **0** |
| `device` / `execute` 两题 | 逐位相同 |

这与仓里另一份更极端的对照是同一类量：`trained_invariance_timing.json` 的 `single_question_vs_eight_state_batch` 报 max logit |Δ| **0.2695**、max prob |Δ| **0.0151**，**超出上游自己预登记的容差**（logit 0.02 / probability 0.005），于是触发了精度诊断：关掉 autocast 用 fp32 重跑，单问题 vs 8 状态一批的差降到 max prob |Δ| **4.46e-6**（在容差内），而 bf16 批 vs fp32 批是 max prob |Δ| **0.0137**。上游把这条写成了 `interpretation_rule: 'Small FP32 single/batch differences support finite-precision or kernel-shape sensitivity; this is not a full kernel root-cause proof.'`，并让 `all_invariance_checks_within_diagnostic_tolerance` 老实报 **false**。

这是本文想说清的一件事：**「概率逐位可复现」在这条链上成立，但成立的范围是「同一精度 + 同一批组成」**。`TYPESAFE_CONTRACT.md:9` 自己也划了这条线：*These tests establish input isolation, not bitwise output equality across GPU batch shapes.* 把这条与 §14.2 并读，才是完整的——argmax 在两个方向上都没翻过（0 次），翻的只有小数位。

### 14.4 容器算术：头是 200,578 个参数，而它可以被三个数独立夹出来

对 Hub 上的 `C-Tianyu/NanoJev` 用 **HTTP Range 只取 `best.safetensors` 的头部**（8 字节长度 + 37,280 字节 JSON，不下载权重）：

| | `4a19595e…`（旧 release revision） | `unified-games-v1`（当前） |
|---|---|---|
| header 长度 | 37,280 B | 37,280 B |
| 张量数 | 322 | 322 |
| dtype | **全部 F32**（322/322） | **全部 F32** |
| Σ data_offsets | 2,385,001,992 B | 2,385,001,992 B |
| 8 + header + data | **2,385,039,280 B**（与逐文件 size 逐字节相同） | 同 |
| 张量元素合计 | **596,250,498** | **596,250,498** |

头部的 12 个张量，形状全部读出：

```
norm.weight [1024]  norm.bias [1024]  scalar.weight [1,1024]  scalar.bias [1]
set_project.weight [128,1025]  set_project.bias [128]
set_attention.in_proj_weight [384,128]  set_attention.in_proj_bias [384]
set_attention.out_proj.weight [128,128]  set_attention.out_proj.bias [128]
set_output.weight [1,128]  set_output.bias [1]
```

把它们加起来：`2,048 + 1,025 + 131,328 + 49,536 + 16,512 + 129 = ` **200,578**。

而这三个数可以互相夹：

```
Hub 上 Qwen/Qwen3-0.6B 的 safetensors metadata total      751,632,384
  − vocab_size × hidden_size = 151,936 × 1024            = −155,582,464
                                                         =  596,049,920   ← 骨架（AutoModel，无 lm_head）
仓内 research/navigation_v3_native_qwen_summary.json
  model_provenance.parameter_count                       =  596,049,920   ✓ 逐位相同
  + 200,578（上表 12 个张量的和）
                                                         =  596,250,498   ← DecisionModel
Hub best.safetensors 的 322 个张量元素合计                 596,250,498   ✓ 逐位相同
仓内 stage1/config.json 的 parameter_count                596,250,498   ✓
仓内 research/trained_invariance_timing.json
  training_config.parameter_count                        =  596,250,498   ✓
```

五个来源、四个文件、两个方向，全部落在同一个整数上。顺带被这几步证明的两件事：**（a）检查点里确实没有 `lm_head`**——322 个张量 = 骨干 310（28 层 × 11 + `embed_tokens` + 末层 `norm`）+ 头 12；**（b）发布的权重是 F32 张量**，与 `config.json` 的 `parameter_storage: float32` 一致（`forward_autocast: bfloat16` 是前向侧的事）。

（另记一处与「参数量」容易被混淆的：`Qwen/Qwen3-0.6B` 的 `config.json` 写着 `tie_word_embeddings: true`，但它的 safetensors 里 `lm_head` 仍然单独存了一份，所以 metadata 的 `total` 比 `sum(p.numel() for p in model.parameters())` 多出整整 `151,936 × 1024`。差值 155,582,464 是这么来的，不是算错。）

### 14.5 哈希链：脚本对得上，训练器对不上

**对得上的那些**（仓内脚本 sha256 vs committed 产物里记的 sha256，逐字节相同）：

| committed 产物 | 字段 | 指向 |
|---|---|---|
| `research/trained_invariance_timing.json` | `script_sha256` | `scripts/check_trained_invariance.py` |
| `research/dynamic_candidates_check.json` + `_v2.json` | `script_sha256` | `scripts/check_dynamic_candidates.py` |
| `research/qwen_tree_check.json` | `script_sha256` | `scripts/check_qwen_tree.py` |
| `research/navigation_v3_native_qwen_*.json` | `script_sha256` | `scripts/evaluate_native_qwen_navigation.py` |
| 同上 | `controller_script_sha256` | `scripts/evaluate_navigation_v3.py`（也 = `navigation_v3_oracle/random.json` 的 `script_sha256`） |
| `research/navigation_v3_jev_api_*.json` | `script_sha256` | `scripts/evaluate_live_jev_navigation.py` |
| `research/nanojev_comparison_verification.json` | `posthoc_dependency_source_hashes` 三项 | `build_navigation_v3.py` / `rollout_jev_navigation.py` / `jev_navigation_worker.mjs`，**三项全中** |

**发布侧也有一条很干净的**：当前发布 `config.json` 里的

```
"implementation_sha256": "a4aedc3b7ad25d875af67943bef6dbc017799f46c4ae0a845fc200e1866fa6cd"
```

就是发布目录里 `source/scripts/train_unified_games.py` 的 sha256（52,461 字节，逐字节相同）。而且发布里的**读法本身**与本地 clone 是同一份：

| 文件 | 本地 clone | HF `source/scripts/` | |
|---|---|---|---|
| `predict_toy_decisions.py` | 17,961 B | 17,961 B | **sha256 相同** |
| `train_toy_decisions.py` | 20,890 B | 20,890 B | **sha256 相同** |
| `serve_decisions.py` | 4,231 B | 4,231 B | **sha256 相同** |
| `teachers.mjs` | 11,444 B | 11,444 B | **sha256 相同** |
| `train_pipeline_decisions.py` | 33,939 B | **33,988 B** | **不同** |

**对不上的那些，才是这一段真正要说的事。** 三个 committed 产物各自记了一个 `trainer_sha256`：

| 产物 | `trainer_sha256`（前 12 位） |
|---|---|
| `research/dynamic_candidates_check.json` | `ab55aeaecef4` |
| `research/dynamic_candidates_v2.json` | `2753a49b89d6` |
| `research/trained_invariance_timing.json` | `054db338f8d5` |
| **仓内当前的 `scripts/train_toy_decisions.py`** | **`4f39babd5575`** |

**四个互不相同。** 也就是说：`check_*` 那些验证脚本自那次运行以来没动过（所以哈希还能对上），而**产生被验证数字的那个训练器，前后至少改过三版，且没有一版等于仓里现在这一版**。同一天（2026-09-17）产出的两份产物 `dynamic_candidates_check.json` 与 `_v2.json` 记的 `trainer_sha256` 就已经不同——所以这不是「后来改了」，而是在那几小时里就在改。这与 pplx-decider 那处「同一份发布里 `model.py` 有两个 sha256」是同一类问题的**加强版**：那里是两处记录、一份文件；这里是三处记录、四处版本。

还有一处较小的同类：`research/navigation_v3_regression.json` 记的 `evaluation_script_sha256`（= 各 run 的 `script_sha256`）是 `8e0d7bca…`，而仓内 `evaluate_navigation_v3.py` 现在是 `5c1b413e…`（同一文件在 `navigation_v3_oracle/random.json` 与 `native_qwen` 的 `controller_script_sha256` 里就是 `5c1b413e…`）。所以**那份回归是更早的一版评估器跑出来的**。

### 14.6 归一化在 teacher 一侧是**显式的一步**，并且被逐决策计数

本地读法那边「Σp = 1」是构造结果（§5）。对照臂走的不是同一条路：Jev 网关回的是一组**声明了小数位、和不一定等于 1** 的舍入值，而这条链**显式地归一化它，并把「归一化了没有」当成一个可核对的字段**：

- `scripts/jev_navigation_worker.mjs:41` 用 `native[k] / sum` 得到控制用的分布，同时把 `native_probabilities`、`native_sum` 与 `normalization_applied: Math.abs(sum-1) > 1e-12` 一起写进答案，`target_kind` 写着 `normalized_rounded_controller_proxy`（`:43-44`）。所以**原生值没有被丢掉，也没有被冒充成最终分布**——两者在同一个对象里并排。
- `rollout_jev_navigation.py:143` 再把 `raw_probability_sum: total` 逐决策落到轨迹里；协议字段明写 `'sampling': 'categorical from normalized rounded Jev action probabilities; native values and sums retained in answers.action'`（`:201`）。
- `verify_nanojev_comparison.py:192-199` 把这一步钉成校验：`normalization_applied` 必须**恰等于** `|sum − 1| > 1e-12`，且 `|native/sum − probabilities| ≤ 1e-12`，否则抛错；两个计数器（`native_nonunit_sum` / `native_sum_differs_exactly`）就是从这条判据里来的。
- 产物里的计数（`research/nanojev_comparison_verification.json`，jev/greedy 与 jev/sample 两臂）：`native_nonunit_sum_decisions` = **0**、`normalization_correction_decisions` = **0**，而 `exact_nonunit_sum_decisions` = **3** 与 **6**。读法：**材料性偏差一例都没有（|sum−1| > 1e-12 从未成立），只有 3 与 6 次在浮点意义上「和不是恰好 1.0」**——所以这一步在这次运行里是恒等变换，但它**被记下来了**，而不是被默认掉。

这一点与训练侧对 teacher 概率的处置合起来才完整。训练侧是「**identity rounded proxy 或整条隔离**」：`train_pipeline_decisions.py:127-138` 先用 `require_unit_sum=False` 读进来、再按声明小数位校验、再判 `sum(round(p*scale)) == scale`，不成立就把**这一臂**隔离（`teacher_probs = None`），既不归一化也不裁剪、也不补均匀分布；`train_toy_decisions.py:72` 的注释是 *Identity proxy at the provider's declared decimal precision; no fabricated labels.* 于是同一份 teacher 数据在两处被两种方式对待：**训练时不改（不合规就整条不用），对照臂里显式改并记账**——差别在于对照臂要拿它去驱动一个控制器，所以它必须是一个真正的分布。

同一份产物还诚实记了一处时序问题：`source_hash_fields_not_recorded_at_run: <3 items>` 与 `source_provenance_limit: 'Dependency hashes were not recorded in the original API run; these are post-hoc source audit hashes.'`——依赖哈希是**事后补**的（补出来的三项恰好都等于仓内当前版本，见 §14.5），而每一次真实请求与 API 回执是独立核过的。

---

## §15 一句话总结

NanoJev 是本仓已收各篇里**唯一一条把「读一个位置」换成「每条候选各读一条完整序列」的路线**：标签集是请求的 `criteria`（与 Kev / RSI-Jev 同侧）、词表维度从不出现（与 Kev / decider / pplx-decider / RSI-Jev 同侧，但机制不同）、顺序偏置在构造上不存在而不是被消解（与 Laya 同侧但更彻底）、`confidence` 这个字段**根本不存在**（与全族都不同）。它同时是唯一一条**在同一份代码里既扔掉词表、又每次走一遍词表**的——训练读法是前者，零训练基线是后者，而后者把 RYOTIDE 的 `marker_mass` 以 `offered_token_mass` 的名字报了出来，中位数 **0.9891**。

代价写在同一批文件里：**前缀共享被验证到逐位相同却一次都没接进服务路径**（§7.1）；**产生那些验证数字的训练器改过至少三版，没有一版留在仓里**（§14.5）；**本文走查的实例响应所对应的权重没有发布**（§14.1）。

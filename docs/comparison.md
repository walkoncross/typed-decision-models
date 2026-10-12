# 七条路线对照总表

> 本文是**索引与总表**，不是又一篇走查：只放跨实现才看得见的对照，论证留在各篇里。
> 需要细节时按文中的节号回到对应文档，本文不重复它们的推理过程。

收录的七条路线，各自有一篇逐环节走查：

| 实现 | 走查 | 一句话 |
|---|---|---|
| RYOTIDE | [`ryotide/typed-decision-model.md`](ryotide/typed-decision-model.md) | 零训练，把冻结 LLM **读**成分类器 |
| Jeff | [`jeff-serve/typed-decision-model.md`](jeff-serve/typed-decision-model.md) | 在 RYOTIDE 那条读法之上**训**一个决策头 |
| Open-Jev | [`open-jev/typed-decision-model.md`](open-jev/typed-decision-model.md) | 把决策拆成 N 个 **pointwise 是非题**，再训一个标量头 |
| Quyet | [`quyet/typed-decision-model.md`](quyet/typed-decision-model.md) | 自训权重，encoder 用槽位、LLM 用字母 |
| Laya | [`laya/typed-decision-model.md`](laya/typed-decision-model.md) | 双向编码器，把决策做成**填空题**：每选项一个 `[MASK]` 槽位 |
| SemIf-OpenJev | [`semif-openjev/typed-decision-model.md`](semif-openjev/typed-decision-model.md) | 零训练，读末位的**字母槽位** `A`–`P`；把读法铺到三个运行时 + 一个浏览器 |
| OpenJev(HF) | [`openjev-hf/typed-decision-model.md`](openjev-hf/typed-decision-model.md) | 微调过的 27B + 一个 vLLM 前面的 HTTP 补丁：读**第一个输出位置**上的 52 个字母 |

共同底线原本写作两条：**一次前向**、**不生成任何 token（`output_tokens = 0`）**，以及**输出封闭在已知选项集上**。
两处例外都在「确实采了一个 token」这一侧：SemIf-OpenJev 的浏览器路径用 `max_tokens: 1` 换来一个能返回
logprobs 的接口，读的仍是被迫输出的那一个 token 的 `top_logprobs`，而不是采样结果（其 §9）；
OpenJev(HF) 对 vLLM 发 `max_tokens: 1`，响应里 `"output_tokens": 0` 是**硬编码**的（`shim.py:325`，其 §4）。
所以更准确的说法是 **「不靠解析生成文本来得到分数」**：七条都满足这一条。SemIf-OpenJev 的浏览器演示页
与 OpenJev(HF) 的 `/v1/chat/completions` 都另有生成通道，但那是各自立起来当对照或透传的东西，不是读法。

> 「只读最后一个位置」**不是**共同底线。七篇里 RYOTIDE、Jeff、Open-Jev、OpenJev(HF) 读末位
> （最后一者的「末位」是**第一个输出位置**，在因果 LM 里是同一个分布，差别在它靠微调把语义搬到那里，其 §3）；
> Quyet 的 encoder 路径和 Laya 读的是各选项自己的槽位（各一篇的 §5 / §4）。SemIf-OpenJev 两条都算：
> `direct` / `serial` / `mlx` 读末位，但 `shared` 模式读的是**每条序列自己**的末位——批内有 padding 时
> 与 `-1` 不是同一个位置（其 §7.3）。差别集中在**读哪个位置**和**置信度从哪来**。

---

## 1. 主对照表

| 维度 | RYOTIDE | Jeff | Open-Jev | Quyet | Laya | SemIf-OpenJev | OpenJev(HF) |
|---|---|---|---|---|---|---|---|
| **权重来源** | 公开 LLM（Qwen3.5 / Gemma 4），全程 frozen | Gemma 4 31B + LoRA，**已合并**后发布 | Qwen3.5 + LoRA + 标量决策头（走查以 2B 为例） | 自训权重（`chinhnc/Quyet-1.0-*`） | 自训权重（`convaiinnovations/laya`，三个 checkpoint） | 公开 LLM（Qwen3.5-4B BF16），全程 frozen；另有 0.6B / 2B / 27B-EXL3 与浏览器量化 GGUF | 自训：`Qwen/Qwen3.8-27B` 微调（约 54 GB BF16；另有 FP8 / MLX / GGUF 四个量化仓） |
| **是否训练** | 否，零新增参数 | 是：训一个决策头 | 是：LoRA + 一个标量头 | 是：打分头 / 字母读法 | 是：双向编码器 + 决策头，RLCD（其 §11） | 否，零新增参数（`core.py:132` 只 `model.eval()`） | 是：在 Qwen3.8-27B 上微调（训练代码不在发布仓里） |
| **决策机制** | 读**标签 token** 的 logits → 掩码 → softmax | 读末位 hidden state → **决策头** → softmax | **逐候选 pointwise 打分** → softmax | 槽位 logits（encoder）/ 字母 logits（LLM）→ 掩码 → softmax | 每选项一个 `[MASK]` 槽位 → 共享 `scorer` 压成标量 → softmax | 读末位的**字母槽位** `A`–`P` → **取下标**（无显式掩码）→ softmax（其 §5） | 读**第一个输出位置**上 ≤52 个字母 logits → 除 T → softmax；服务端 `allowed_token_ids` 真掩码（其 §5） |
| **「typed」从哪来** | 分词器：标签必须单 token，否则映射成字母标记 | 单 token 标签表 255（26 字母 + 229 双字母） | 请求声明的 `answer_keys` + 响应概率校验 | 训练时固定的槽位：`[MASK]` / 字母 `A..J` | `[MASK]` 槽位；**与选项文本长度、分词器都无关**（其 §2） | 请求声明的 `options` + 固定字母表；分词器**三重校验**（单 token / 往返 / 边界），llama.cpp 再加 GGUF 逐 token 比对（其 §2） | 请求里的 `criteria`（map / 数组 / 两项）+ 固定 52 字母（`A`–`Z`、`a`–`z`）；只验 token id 唯一，不验往返（其 §2） |
| **读位置怎么定** | **运行期标定**：扫 9 前缀 × 2 表面形式，取 marker mass 最高（其 §3） | **训练期固定**：与 prompt 格式一起冻结（其 §4） | **无标定**：`Yes or No` 是固定问句（其 §9.2） | **训练期固定**：运行期只有 `prompt_version` 一个常量（其 §0） | **训练期固定**：位置是构造出来的 `marker_pos`；布局写进 checkpoint config（其 §3） | **无标定**：固定末位；`allowed_token_mass` 只**上报**、从不参与选择（其 §3） | **训练期固定**：微调让第一个输出位置携带决定；运行期**无任何诊断量**，防线只是每处都传 `enable_thinking: False`（其 §3） |
| **读出来的是什么** | 末位**词表 logits**，再按标记 id 取出 | 末位 **hidden state**，天然不物化 vocab 维度 | 末位 hidden state → 头 → **每候选一个标量** | encoder：每槽位一个标量；LLM：末位字母 logits | **每个选项一个槽位**的隐状态 → 标量（K 个位置，不是末位） | 末位**词表 logits**，按槽位 id 取出 ≤16 个数 | **第一个输出位置**的词表 logits，按候选字母 token id 取出 ≤52 个数 |
| **候选数上限** | 260（`A0`–`Z9` 两 token 两步读；>26 另有 `grouped` 近似读法） | 255（标签表上限；头槽位默认 16、上限 255） | 255（接口约定，一步打分） | 10（`A..J` / 槽位数，超出直接报错） | **无硬上限**：受 token 预算与「槽位不许塌陷」约束（其 §2） | 16（`LETTERS = "ABCDEFGHIJKLMNOP"`，`core.py:11`）；**浏览器路径 20**（`worker.js:28`）——同一个产品两条路径不一样（其 §2） | 52（一步，26 大写 + 26 小写）；>52 走**分块两级近似**，且字母**逐块重置**（其 §9） |
| **顺序偏置** | 多 order 共享前缀 fork，折叠回真实标签后**概率平均**（其 §6） | 只给 4 种 yes/no 渲染**开关**，本身不平均（其 §6） | **不存在该变量**（pointwise 无顺序，其 §4） | 未涉及（其 §4 讲的是两条路径的标签槽位顺序，不是选项顺序偏置） | **训练时消解**：`parallel` 布局让位置对选项不可见（其 §6） | **不消除，只度量**：反转选项顺序翻转 **10/36** 的 argmax、平均最大概率位移 0.2648，并公开进 committed 报告（其 §6） | **训练时压低**：翻转率 18.5% → **2.3%**（`README.md:109`）；代码留了 `READOUT_PERMS` 平均开关，但**发布配置是 1、不平均**（其 §7） |
| **置信度来源** | **读出来的**：只有 temperature 能改数值 | **训出来的**：头的 A、b；训练后只剩 serving rule 两项 | **训出来的**：软目标 CE + Brier；temperature 二次校正 | **训出来的**：T 写在 `quyet_config.json`，随权重发布 | **训出来的**：RLCD；但**两个都叫 `confidence`**，另有 `x_jev_confidence`（其 §10） | **读出来的**：T 逐 workload 拟合（1.2342 / 2.4990 / 1.7086），带 group-disjoint 折外 ECE 与 group bootstrap 区间，无条件发布（其 §12） | **训出来的**：训练在改 argmax；T=0.85 与 noul 的 (1.829074, 0) 是训练之后施加的常数，且**代码默认值全都不同**（其 §13） |
| **诊断量 / 守卫** | `marker_mass`（全词表落在标记上的份额） | 无同等量；`runtime` 只有 `readout` / `probability_origin` / `serving_rule` | 概率校验 + `max_length` 硬报错 + 拒绝服务未加载模型 | 截断上报 `truncated` + `warnings`；加载期断言字母单 token | `options_distinct` vs `options`、`tokens_per_option`（槽位塌陷）+ `state_tokens_dropped` | `allowed_token_mass` + `full_vocab_argmax_id`（**只在 serial / llama.cpp 后端产出**）；槽位三重校验 + 加载完整性自检 | `GET /v1/version` 报 model dir + 全部常数 + 全部 flag + **补丁自身 sha256**，且同一串摘要写进每条回答；`ReadoutIncomplete` 缺分即失败（其 §12） |
| **多问题共享 state** | `run_many()`：公共前缀预填充一次，**计费只算一次**（其 §7） | **不做**：朴素逐题循环，state 重复编码、全额计费（其 §7） | 用两级 LCP 前缀缓存，**默认关闭**（误差超容限，其 §7） | **不做**：逐问题累计 token，state 重复编码（其 §8） | state 只**分词**一次，但**计费逐题累计**；双向编码器无法复用前缀（其 §7） | `shared` 模式：一次 prefill + `reorder_cache(all-zeros)` 复制分支 + 只算指定末位；**8.6× 加速换来 5–6/777 的 argmax 漂移**（其 §7） | 每题各一次 prefill，复用交给 **vLLM 前缀缓存**；补丁只做 `prewarm`、784 块对齐（默认关）与 `SHIM_STAGGER`（其 §8） |
| **路由（逐请求挑 checkpoint）** | 无 | 无 | 无 | 无（`kind` 在 config 里定死） | **有**：纯 Python 启发式，不跑模型（其 §8） | 无（`--backend` 是启动参数，不是逐请求路由） | 无（一个进程服务一个模型目录；目录名进 `/v1/version`） |
| **后端** | MLX（参考）+ torch | torch（CUDA），cuDNN attention 默认关 | torch + 自研 Triton final-row 内核 | encoder 与 LLM 两条**并行** runtime | torch 四种：eager / compile / tilelang / onnx | 三种：torch / MLX / llama.cpp，另有浏览器 WASM（wllama） | vLLM 0.29.0（FP8，参考配方）+ MLX 替代路径（`shim_mlx.py` 只换 client） |
| **服务接口** | `/v1/systemone` + `/health`（含 `prompt_hash`），另有 MCP | `/v1/systemone` + `GET /`（**无** `prompt_hash`），单线程串行 | `/v1/systemone` + `/health`，有界批处理 | CLI（JSON Lines 进出）+ `examples/gem_run` | FastAPI `/v1/systemone` + `/batch` + `/health`，另有 stdio MCP | **CLI**（`semif-score`）+ 只增不改的 JSONL；无 HTTP 服务、无 `/health`（其 §11） | 标准库 `http.server` 起的补丁：`/v1/systemone` + `/v1/version` + `/v1/prewarm` + `/v1/chat/completions`；**不设 token 就没有鉴权**（其 §12） |

---

## 2. 谁在改发布出去的数值

这张表把七篇各自的「置信度链条」并排放。**跨实现比较概率之前，必须先在各自那一列里确认全部环节**。

| 实现 | 改数值的环节 | 改 argmax？ | 关键约束 |
|---|---|---|---|
| **RYOTIDE**（其 §9.0） | ①温度 ②上下文校准 bias ③第二模型一致性重加权 | ①否 ②理论可以 ③否 | 已发布 preset 里**只有温度生效**；bias 未接入（见 §4.5），重加权仅离线测量 |
| **Jeff**（其 §9.0） | ①**训练本身**（头的 A、b）②`temperature` kind ③serving rule 温度 ④`noul_edge` 带内推移 | ①**是** ②否 ③否 ④否 | 训练之后只有 ③④ 能动数值；④**会破坏单调性**（见下） |
| **Open-Jev**（其 §6） | ①**训练本身**（软目标 CE + Brier）②温度校正 | ①是 ②否 | 温度**无条件写入**，无接受 / 拒绝门槛 |
| **Quyet**（其 §5） | ①**训练本身** ②`softmax_t` 温度 | ①是 ②否 | `softmax_t` 是**唯一**改发布数值的步骤；缺键 = 1.0 = 未校准 |
| **Laya**（其 §10.1） | ①**训练本身**（RLCD）②温度 ③逐语言温度 ④分箱图 | ①**是** ②否 ③否 ④否 | 温度**无条件发布**，无接受 / 拒绝门槛；`clamp` 自陈「只限界损害，不校准任何东西」（其 §10.3） |
| **SemIf-OpenJev**（其 §12） | ①原生 `softmax(option_logits)` ②温度重写 `softmax(logits / T)` | ①否 ②否（单调） | **没有「训练本身」这一环**，所以温度在数学上不可能改 argmax；改写过的每一行**自带** `calibration: {method, temperature}`（`calibrate.py:346`），且原文件不动 |
| **OpenJev(HF)**（其 §13） | ①**训练本身**（微调 Qwen3.8-27B）②`logprob / T` ③noul 的 `logit/T + bias` ④`round(·,4)` ⑤`READOUT_PERMS` 平均（发布配置为 1、未启用） | ①**是**（18.5% → 2.3% 就是它改的）②③否 ④极近并列时可能 ⑤**是**，但未启用 | 三个常数的**文件默认值与发布值全都不同**（T 1.1→0.85、noul 3.0→1.829074、bias −0.4→0）；且 T 与 noul 斜率是在 `READOUT_TARGETED=1` 下拟合的，混用即为错配 |

温度的具体值，七篇的来源各不相同，**没有一个可以跨实现借用**：

| 实现 | T 从哪来 | 实际值（含被拒的） |
|---|---|---|
| RYOTIDE | 离线拟合 + **预先注册的接受 / 拒绝判据** | Gemma 1.924 采纳、Qwen9 0.667 采纳、Qwen3.5-4B 拟合 1.592 但**留出集变差被拒**，保持 1.0 |
| Jeff | `serving_rule.json` 配置（另有逐输入学出来的 `temperature` kind） | 逐类型可配，缺失即恒等 |
| Open-Jev | `train.py` 拟合后直接发布 | 2B 发布值 **1.7360720317205194** |
| Quyet | 写在模型自带的 `quyet_config.json` | 分桶：encoder 用 `<type>:<桶>`，LLM 用纯题型键 |
| Laya | LBFGS/NLL 离线拟合，**无接受 / 拒绝判据**（`train.py:1138`） | 三层：逐题型 + 逐选项数分桶 + 逐语言；`choice:11+` 拟合出 **0.1006**，被 `clamp` 拒到 `TEMP_MIN = 0.5`（其 §10.3） |
| SemIf-OpenJev | 黄金分割搜索最小化 mean NLL，`T ∈ [0.05, 20]`，numpy only、不加载模型（`calibrate.py:97`） | **逐 workload**：authored 144 → **1.2342236287881765**、WANLI 256 → **2.499030870405456**、Every 154 → **1.7086356884405622**；池化负对照 554 行 → **1.9647556181545844**（三个 workload 的「池化显著更差」全为 `false`） |
| OpenJev(HF) | 环境变量，写进发布配方（`serve/SERVE.md:58-66`）；拟合口径只留一句注释 | 发布 **T=0.85** / noul **(1.829074, 0)**；**代码默认 T=1.1 / (3.0, −0.4)**；`results/noul_calibration.json`（注释所称的拟合产物）**不在发布仓里** |

### 温度的代价，七条记录四种表现

- RYOTIDE §12(f)：同一条记录顶部概率 T=1 时 **0.999917**、T=1.924 时 **0.981007**；
- Open-Jev §10(i)：255 候选下顶部概率 T=1.736 时 **0.107047**、还原 T=1 后 **0.575983**（T>1 的压平效应随候选数放大）；
- Jeff §9.1：100000 组随机 `(p, T)` 中 argmax **改变 0 次**——温度不改准确率，所以能被单独拟合与事后报告；
- Laya §10.3：`choice:11+` 的 **0.1006** 会把真实的 0.24 发布成 **0.99**，clamp 到 `TEMP_MIN=0.5` 之后仍发布成 **0.4993**——源码原话是 *The clamp bounds the damage; it does not calibrate anything*；
- 而 Jeff 多一项 `noul_edge`：`0.7999 → 0.801`，`0.8001 → 0.8001`，**raw 更大报出去更小**，跨样本比较被它改写的 P(yes) 不可靠；
- SemIf-OpenJev §12：同一批 144 行，T=1 时平均 top-1 是 **0.8628**、T=1.2342 时是 **0.8333**，argmax 翻转 **0 次**——这正是「零训练 ⇒ 温度只动置信度」的实证；而它逐 workload 的三个 T 相差一倍以上（1.2342 vs 2.4990），**拿错哪一个都会得到看似合理的错数**；
- OpenJev(HF) §5/§13：T 与 noul 斜率是**在 `READOUT_TARGETED=1` 下拟合**的，而那个开关的**文件默认是关的**（`shim.py:21`）；默认关时读数协议换成 `top_logprobs` + 掉出 top-K 给 `−30.0` 兜底（`shim.py:164`）。所以「跑起来看着正常」的那套默认，用的是**为另一套读数拟合的常数**。

---

## 3. 四条不变量的跨篇落位

RYOTIDE §13 那四条是这组文档的共同骨架。它们在另外六篇里的落位**并不整齐**——有两条在「打分量不是词表分布」的实现里根本不适用，需要改写成等价命题；而 Laya 甚至连 #3 都要改写，因为它读的就不是最后一个位置。

| RYOTIDE §13 不变量 | Jeff（其 §13） | Open-Jev（其 §9） | Quyet（其 §9） | Laya（其 §12） | SemIf-OpenJev（其 §13） | OpenJev(HF)（其 §14） |
|---|---|---|---|---|---|---|
| **#1** 读位置必须落在标记上（诊断量 `marker_mass`） | 结构性成立，但**无诊断量** | **概念不适用**：打分量是 head 的标量，不是词表分布，没有 `marker_mass` 这个东西 | 改写为「只有选项槽位的 logit 进入 softmax」 | **结构性成立，且有诊断量**：`options_distinct` vs `options`（槽位塌陷）、`tokens_per_option` | **成立，诊断量存在但用途不同**：`allowed_token_mass` + `full_vocab_argmax_id` **只上报、不参与选位置**；且只在 serial / llama.cpp 两条后端产出（其 §3） | **结构性成立，但完全没有诊断量**。位置由微调固定；唯一的守卫是 `assert` 候选 token id 唯一（`shim.py:150`），而它查的是**槽位**不是**位置**（其 §14） |
| **#2** 前缀与标记表面形式耦合，必须成对扫 | 成立，且被**训练固化**；代价是不能换 prompt 格式 | **不适用**：没有 `SUFFIXES`/`PATTERNS` 扫描，也没有运行期标定 | 改写为「标签必须落到单 token（LLM）/ 单个 MASK 槽位（encoder）」 | **不适用**：标记是 token id 不是字符串，不存在「表面形式」这个自由度 | **改写为「断言掉这个自由度」**：不扫，而是断言 `encode(prompt + letter) == ids + [slot]`（`direct.py:42`）；llama.cpp 再加一层 GGUF 逐 token 比对 | **不适用**：标记是 token id（`letter_ids()`）。同类耦合换成「字母 token 与 `TOKENIZER` 绑定」——它同时决定读哪个 id 与 `/v1/version` 报什么名字（其 §13.2） |
| **#3** 只读最后一个位置 | 成立，且更彻底（连 vocab 维度都不物化） | 成立 | 成立 | **改写**：不是「只读一个位置」，而是每条序列读它自己的 K 个槽位；「不物化词表维度」这条同样成立，机制不同 | 成立；**但 `shared` 模式读的是「每条序列自己的末位」而不是批的 `-1`**（批内右填充时两者不同，`shared.py:50,142-144`），MLX 同理 | **改写为「只读第一个输出位置」**：`max_tokens=1`（`shim.py:151`），在因果 LM 里与末位是同一个分布，但语义靠**微调**搬过去；「不物化词表维度」不适用（词表维度在 vLLM 里） |
| **#4** 比较概率 / ECE 前先确认 temperature | 成立，且**多一项 `noul_edge`** | 成立，`temperature.json` 随 checkpoint 发布 | 成立，T 在 `quyet_config.json` | 成立，但**要确认的不止 T**：还有两个 `confidence` 量与该用哪一个、`x_jev_confidence` | 成立，且最容易确认：三个 T 逐 workload 不同，但**改写过的每一行自带 `calibration: {method, temperature}`**；先看文件名里有没有 `.calibrated` | 成立，且是七条里查得最彻底的：`/v1/version` 报全部常数与 flag，摘要还写进每条回答的 `model` 字段。**但前提是调用方真的去读**——不读就会用文件默认值（其 §13.2） |
| （各自新增的） | **单 token 标签表与分词器绑定**——换分词器必须重标 | — | **只有 state 被截断，且截断必须被上报** | **布局是权重的属性**（`option_layout` 写死在 config）；**归一化常量**（`k / 255.0`、桶边界 `2/3-5/6-10/11+`） | **`output_tokens = 0` 有例外**（浏览器路径 `max_tokens: 1`）；**发布的产物必须与发布的代码同版**——committed 的 `direct-authored144.jsonl` 带 `cache_class` 等 HEAD 已不产出的字段（其 §13） | **发布的常数必须与发布的读数配对**——三个常数的默认值全不等于发布值，且 T 是在另一个读数协议下拟合的；**仓库自带的清单要与当前内容一致**——`MANIFEST.json`/`SHA256SUMS` 的 `README.md` 条目已过期（其 §13.2/§13.3） |

---

## 4. 同一个坑，各踩各的

这一节是并读七篇才有价值的部分：同一个失效模式在七条路线上各出现一次，**修法不同、能不能被发现也不同**。

### 4.1 `enable_thinking=False`：一条坑，几种接法与两种不适用

读位置落进 `<think>` 块，概率质量到不了选项标记上——各条路线都踩过或绕开过，发现方式与修法各不相同；Laya 则根本没有这个位置（表末一行）。

| 实现 | 怎么发现 | 怎么修 |
|---|---|---|
| RYOTIDE | **靠 `marker_mass`**：只有约 15% 质量到达标记，而适配器照样返回格式正确的分布、harness 照样打分（分数 27.4 → 61.8） | `_tokens()` 传 `enable_thinking=False`，模板不接受时回退 |
| Jeff | 不靠诊断量，**设计上消掉**：读位置固化在模板末尾 | 头就训在那个位置上；代价是**失去对应的诊断量** |
| Open-Jev | 不适用（无标记概念） | `model.py:234` 传 `enable_thinking=False` |
| Quyet | 不靠诊断量 | prompt 构造处**写死**，`TypeError` 时回退 |
| Laya | **这个坑不存在**：`laya/` 核心没有 `enable_thinking` / `chat_template` / `padding_side`（grep 为空） | 不适用——它不是生成模型，序列由 `build_sequence` 手工拼装，没有「思维块」这个位置可以落错 |
| SemIf-OpenJev | 不靠诊断量 | **同一个仓库里三种接法**：① torch 路径传 `apply_chat_template(..., enable_thinking=False)`（`direct.py:36`）；② reranker 路径**根本不走模板**，把 `<think>\n\n</think>\n\n` 手写进硬编码的 Qwen3-Reranker 字符串（`reranker.py:15`）——「消掉思维块」比「关掉思维块」更彻底；③ 浏览器路径交给 wllama 的 `chat_template_kwargs`（`worker.js:150`）（其 §8 / §9） |
| OpenJev(HF) | **不靠诊断量——所以这个坑在这里最危险**：模型卡明说位置是训练出来的（`README.md:191`），运行期没有任何量能告诉你读错了地方 | 靠「每一处都不漏传」：补丁内三处 `chat_template_kwargs={"enable_thinking": False}`（`shim.py:111,152,159`），透传路由再**覆盖**一次（`shim.py:275`）。而模型的 `chat_template.jinja:46-47` 在**不传时默认开 thinking、档位 `xhigh`**，并把 assistant 前缀写成**没有闭合的** `<think>\n`（`:168`）——漏传一处的后果就是读位置落进思维块（其 §3） |

### 4.2 填充方向：两条都精确，依赖的性质不同

| 实现 | 填充方向 | 为什么不影响结果 |
|---|---|---|
| RYOTIDE | **右填充且不传 mask**（torch 批量读） | 因果模型看不到自己之后的 padding |
| Open-Jev | **右填充**（`padding_side = "right"`） | 同上，不污染真实末位置 |
| Quyet | **左填充且必须传 mask** | 位置整体平移，RoPE 只看相对距离；padding 由 mask 屏蔽 |
| Laya | **右填充且必须传 mask**（`pad_batch`，`backends/base.py:41-71`） | 它是**双向**编码器，padding 在左侧也会被 attend 到，所以 `right` 并不自动安全，必须靠 mask；另外填充行**特意留一个活 marker**（`:59-68`），否则全掩码行变 NaN——而批里一个 NaN 就足以毒掉 CUDA graph 的 replay |
| SemIf-OpenJev | **两条方向在同一个仓库里并存**：`shared` **右填充** + 显式 `position_ids`（padding 位给 0），读**记录下来的每条末位**（`shared.py:47-50`）；reranker **左填充**，读 `[:, -1, :]`（`reranker.py:64-67,77`） | 右填充那条靠因果性 + 位置显式给定（`attention_mask` 也传了）；左填充那条靠 mask。真正的坑不在方向，在**「末位」这个词**：批内有 padding 时，`shared` 模式必须读每条自己的末位，读 `-1` 会读到 padding（其 §7.3） |
| OpenJev(HF) | **不适用**：这份代码**不组批**——它按题并发请求（`shim.py:320` 的 `ThreadPoolExecutor.map`），每个请求一个 prompt，填充由 vLLM 内部处理 | 补丁里唯一与「对齐」有关的东西是 `SHIM_PAD`（`shim.py:92-102`）：往共享前缀里塞 `" pad"` 填充，把 header+prefix 顶到 784 的块边界，为的是**缓存块命中**而不是批内对位。它甚至为此对 BPE 合并做了三次迭代修正，且**发布配置里是关的**（其 §8.2） |

Quyet §6 给了实测（随机初始化 2 层 Qwen2）：位置整体 +36 → 差 **0.0**；左填充 + mask → **5.96e-08**；左填充**漏传 mask** → **0.166**。所以「一边能省掉 mask、另一边不能」不是风格差异，是被依赖的性质不同。

### 4.3 线性注意力 / 混合架构的 cache 坑

| 实现 | 坑 | 结论 |
|---|---|---|
| RYOTIDE | `ArraysCache`（线性注意力层，固定大小 recurrent state、无 offset 可回卷） | 只复制**列表**即可——槽位是写时重绑而非就地修改 |
| RYOTIDE | `RotatingKVCache`（Gemma 4 的滑动窗口层） | 必须**给每个分支自己的 buffer**：环形缓冲绕回后 `state` 交回的就是 buffer 本身，就地写会**写花父级** |
| RYOTIDE | `trim_to()` 遇不可回卷的层 | **直接报错**，而不是静默 no-op——否则一个问题会泄漏进下一个 |
| Open-Jev | transformers 5.10.2 里 `LinearAttentionLayer` 未实现 `batch_repeat_interleave` | `fork_cache` 手写；浅拷贝或只拷 KV 会让递推状态**别名**，污染后续候选 |
| Open-Jev | 库用 **conv dtype（bf16）** 分配递推状态，而 delta-rule 状态本应是 FP32 | `RecurrentDtypeLayer` 精度补丁，强制保留 kernel 输出的 dtype |
| Jeff / Quyet | 无此坑 | 每问题独立完整前向、无 KV 分支，用简单性换掉了这一整块复杂度 |
| Laya | 无此坑 | **编码器根本没有 KV cache**，也没有递推状态。代价不在内存而在计算：双向编码器**无法复用前缀**——每个 token 的表示都依赖整段 state，换个 state 就得整段重算（其 §7） |
| SemIf-OpenJev | 分支复制写成 `cache.reorder_cache(torch.zeros(len(rows)))`（`shared.py:128`）——语义就是「把第 0 行复制 N 份」 | 对纯注意力 cache 够用；换成带递推状态的 cache，复制的就是状态本身。所以有**两道守卫**：`cache.get_seq_length() != len(prefix)` 与 `reorder_cache` 不可调用时**直接报错**（`shared.py:95-98`，原话 *Native cache does not support duplicate branch selection*）。MLX 侧改用 `entry.merge([entry]*N)` + `entry.prepare(lengths, right_padding)` 显式告知每条分支的真实长度（`mlx_backend.py:215-217`）；llama.cpp 的 cache 不透明，只能整段 `save_state` / `restore_state`（`llamacpp_backend.py:380-391`） |
| SemIf-OpenJev | **加速本身是有损的**：`serial` / `parallel` 相对 fresh 有 5/777 与 6/777 的 argmax 翻转、最大概率差 0.0934 / 0.0624 | 源码把容差设成**故意不设门槛**（*Tolerance is deliberately non-gating*），README 也照写「BF16 execution changed 5–6 of 777 argmaxes」。这是七条里唯一明说「快 8.6 倍，代价是 6 个决定变了」的地方（其 §7.5） |
| OpenJev(HF) | **坑在模型里，但处理它的不是这份代码**：`config.json:16,22,93` 是混合架构——每 4 层一个 `full_attention`、其余 `linear_attention`，且 `mamba_ssm_dtype: float32` | 缓存整个交给 **vLLM**（参考配方里的 `--enable-prefix-caching` 与 `--gdn-prefill-backend triton`，`README.md:152-154`）。补丁只做一件相关的事：把共享前缀对齐到 **784** 的块边界（`shim.py:82`），而那个 784 在 `config.json` 里**没有对应字段**，注释只说它是 *the cache block of this model*——**推导未核实**（其 §8.2） |

### 4.4 置信度：读出来的 vs 训出来的

这是七条路线最根本的一条分界，也决定了「温度能不能事后单独拟合」：

- **RYOTIDE 读**：分布是模型自己的 softmax，训练不参与。于是温度**不可能改准确率**，可以被离线拟合、被预先注册的判据接受或拒绝（§2）。
- **SemIf-OpenJev 也读**，而且是七条里**唯一连「训练本身」这一环都不存在**的（另一条零训练的 RYOTIDE 亦然，但 RYOTIDE 还带上下文校准 bias 与第二模型重加权两个潜在环节，SemIf-OpenJev 的链条上只剩温度一环）。代价是它的**拟合比 RYOTIDE 更宽松**：三个 T 无条件发布，没有接受 / 拒绝判据；换来的是它把折外 ECE、group bootstrap 区间和池化负对照一并提交（其 §12）。它也因此能实测出「T=1.2342 在 144 行上翻转 0 个 argmax、平均 top-1 从 0.8628 降到 0.8333」这种干净的单调性证据。
- **Jeff / Open-Jev / Quyet / Laya / OpenJev(HF) 训**：训练本身就在改数值、也可能改 argmax。温度退化成**训练之后的二次校正**，不再是唯一变量。
  - OpenJev(HF) 是这一组里**把「训练改了 argmax」量化得最直接**的一条：同一批选项打乱顺序，微调前翻转 18.5%、微调后 2.3%（`README.md:109`）——温度不可能做出这种改变，它是训练做的（其 §7）。代价随之而来：它发布的三个常数**与代码默认值全不相同**，且是在另一个读数协议下拟合的（§2）。
  - Open-Jev 对此最宽松：拟合值**无条件写入**，没有接受 / 拒绝门槛（其 §6 注：*"Calibration temperature is not a held-out capability or safety result"*）。
  - Laya 同样**无条件发布**（`train.py:1138`），且发布路径上的 `clamp` 自己承认 *The clamp bounds the damage; it does not calibrate anything*（其 §10.3）。
  - Quyet 把它固化进权重包：T 与「缺键 = 1.0 = 未校准」的默认一起发布。
- **Laya 另有一层别家没有的麻烦**：同一条答案里**两个字段都叫 `confidence`**，只有一个（`answer_confidence`）是温度拟合与 ECE 的对象，另一个是归一化熵；服务侧还会注入第三个 `x_jev_confidence`。所以 §3 的 #4 在 Laya 上要改写成「确认用哪一个」，而不只是「确认 T」。

### 4.5 「算得出来 ≠ 接进来了」

七篇各自记了一处「代码写了却不在生效路径上」或「接了却没有收益」，性质各不相同：

| 实现 | 位置 | 性质 |
|---|---|---|
| RYOTIDE | `calibration_bias()`（其 §9.3） | **全仓零调用点**，且六臂实验实测**没赢**（letters 0.675 vs letters+cc 0.600）——是「试过之后的结果」，不是忘了接线 |
| Jeff | `typed-*` 两种头（其 §3） | 发布路径上**不可达**：`Decider.decide` 调 `option_logits(h, k)` 时从不传 `q`，而 typed 头要求显式传入 |
| Quyet | encoder 的 `MASKED_LOGIT`（其 §3.1） | 在主路径上是**冗余**：`runtime` 只取 `lg[j, :len(markers)]`，补齐槽位本来就进不了 softmax——写了但改变不了任何数值 |
| Laya | `loss="soft-ce"` 路径（其 §11） | **加了没收益**：源码记 *#741 measured no gain from the extra term on the typed-decisions split*——策略梯度项在那份 split 上没测出收益，被诚实地记成负结果 |
| Open-Jev | 反面教材（其 §8） | `device_map="auto"` 让 12 个模块落到 disk、决策头留在 `meta`，**而服务仍然对 `/health` 回 "ready"，并从没加载的模型返回自信的数字** |
| SemIf-OpenJev | 诊断量 `allowed_token_mass` / `full_vocab_argmax_id`（其 §3） | **不在默认生效路径上**：只有 `serial.py:120-123` 与 `llamacpp_backend.py:285-286` 产出，`direct` / `shared` / `mlx` 都不产出；而 README 的 Quick start 走的正是 `--mode direct`。它是「量算得出来、也确实有用，但最常用的那条读法不上报它」——与 Laya 那条（接了没收益）方向相反，是**有收益但没接到默认路径** |

| OpenJev(HF) | 读数协议 `READOUT_TARGETED` 与三个常数（其 §13.2） | **默认路径与发布路径不是同一套**：文件默认 `TARGETED` 关、T=1.1、noul=(3.0,−0.4)；发布配方是 `TARGETED=1`、T=0.85、noul=(1.829074, 0)，而 T 与 noul 斜率**正是在 TARGETED 打开时拟合的**。`python helper/shim.py` 直接跑起来能得到格式完全正确、数字看似合理的回答，用的却是为另一套读数拟合的常数 |

前三条是「写了没生效」，Laya 那条是「接了没收益」，SemIf-OpenJev 那条是「有效但有条件」，OpenJev(HF) 那条是「接了但要配对才生效」，Open-Jev 那条是「生效了但是没有效」——最后一条最难发现，因为它连错都不报。

---

## 5. 数字的可比较性：六处口径不一致

跨实现引用概率、准确率、ECE 之前，有六组口径必须先对齐，否则会得到看似合理的错数。

1. **温度**（§2）。七篇都记录了同一个现象：同一条记录、同一个 argmax，顶部概率可以差一个量级。
2. **ECE 分箱**。RYOTIDE 走的是 vendored JevBench 的 **10 个等宽分箱**（`vendor/jevbench/jevbench/metrics.py:55`）；Open-Jev 与 Laya 各自实现的是 **15 箱**（分别 `metrics.py:98` 的 `n_bins=15`、`common.py:766` 的 `bins: int = 15`）。**10 箱与 15 箱的数字不可直接互比**（Open-Jev §6 明确记了这一点）。SemIf-OpenJev 用的是 **10 箱**（`benchmarks/calibrate.py:134` 与 `benchmarks/evaluate.py:135` 都是 `bins=10`）——它与 RYOTIDE 在这一层同口径，与另外两条不同。
3. **calib 轴的公式**也可能不是一个。RYOTIDE 记录了看板上它那一列的 calib 与官方列**并非同一个式子**（本地评分只用 ECE，官方在能拿到 gold 分布时还加一项 TVD，其 §9.5）。
4. **归一化常量与分桶边界**。Quyet 与 Laya 都用 `2 / 3-5 / 6-10 / 11+` 这套选项数分桶（Quyet 其 §5、Laya 其 §10.2），而 Laya 的 act 头还用 `k / 255.0` 归一化选项数（`common.py:650`）。同一个「11+ 桶」在两边指的不一定是同一组候选规模。
5. **「已校准」与「未校准」是两份文件，不是同一个数的两个版本。** SemIf-OpenJev 把温度改写过的概率写进**另一个文件**（`results/raw/calibration/*.calibrated.jsonl`），原文件不动，并且在改写过的每一行留下 `calibration: {method, temperature}`（`calibrate.py:345-346`）。引用一个概率之前，先看文件名里有没有 `.calibrated`——这与「确认 T 是多少」（第 1 条）不是同一件事：**T 是参数，文件是身份**。
6. **「代码里的默认值」与「发布配方」是两套数。** OpenJev(HF) 的三个标定常数（T 1.1 → **0.85**、noul 3.0 → **1.829074**、bias −0.4 → **0**）与两个行为开关（`READOUT_TARGETED` 默认关 → 发布 **开**、`READOUT_INSTR_STYLE` 默认 JSON → 发布 **pyrepr**）的默认值**全都不同于发布值**（其 §13.2），而 T 与 noul 斜率正是在 `TARGETED` 打开时拟合的。**照 README 抄那两行命令**与**直接 `python helper/shim.py`** 得到的是两套数，且后者不会报错。这是七条里唯一一条把「怎么跑」本身变成口径问题的地方。

另外六处「数字归属」的坑，各篇都自己声明了，引用时不要越过：

- RYOTIDE §12 的两条记录是 `temperature 1.0` 的 run，**不是**已发布 preset 的 1.924；
- Open-Jev §10 的两条记录 `max_length` 配置不同（16384 vs 4096），只是都没触及上限；
- Jeff **没有跑过模型前向**（权重约 61 GB 未下载），全文**不给任何前向产生的概率数值**；
- Laya 同样**没有跑过前向**（本机无 torch、无 checkpoint）。它那篇 §6 的 3.59 与 §7 的 3.1 倍是**源码注释里记录的实测**，不是该文跑出来的——引用时按 `common.py:280-288` 与 `common.py:416-424` 回源核对；
- SemIf-OpenJev 的 `results/raw/predictions/direct-authored144.jsonl` 文件名写着 `direct`，实际 `serving_config` 是 `native-state-prefix-cache-v1`（即 `serial` 路径），并带有 HEAD 代码**已不再产出**的 `cache_class` / `encode_seconds` 字段，`model` 里还有 HEAD 根本不支持的 `adapter_*` 槽位（实测取值全为 `null`）——**它记录的是更早的一次运行，不能用 HEAD 逐字段复现**（其 §13）。校准层读的正是这份文件，所以那三个 T 也带着同样的版本印记；
- OpenJev(HF) 自带的 `MANIFEST.json` 与 `SHA256SUMS` 里 **`README.md` 的条目已过期**：清单记的是 12158 字节 / `3753084b…`，仓库当前是 13007 字节 / `bff7f997…`（`git cat-file -s HEAD:README.md` 也是 13007）。其余 12 个非 LFS 文件与清单**逐字节相符**（含 `helper/shim.py`）。所以卡片上那些数字与清单不是同一次生成的——**引用前先确认清单覆盖到它**（其 §13.3）。它另有两处声称的拟合产物（`results/noul_calibration.json`、以及 *4 ≈ +16 pts acc* 那条注释）**根本不在发布仓里**，无法回源（其 §16.3）。

---

## 6. 怎么用这几篇

- **想理解一条路线**：直接读它那篇，七篇都能独立读——它们各自的 §0 已经把与 RYOTIDE 的分歧讲清了。
- **想选型**：从 §1 的主对照表进，按「要不要训练」「候选数上限」「能不能换 prompt 格式」「多问题要不要共享 state」「要不要逐请求挑 checkpoint」五个问题筛。
- **想踩坑**：读 §4。这一节的每一条在七条路线上都真实发生过，且**修法不可互相照搬**——尤其 §4.1 的 `enable_thinking`（OpenJev(HF) 的模板在不传时**默认开 xhigh 推理**且 assistant 前缀不闭合）、§4.2 的填充方向（Laya 是双向编码器，右填充**并不**自动安全；SemIf-OpenJev 同一个仓库里两条方向并存）和 §4.3 的 cache 分支复制。
- **想引用数字**：先读 §5，确认温度、ECE 口径、分桶边界、**跑的是哪套默认值**和记录来源。
- **本文不做什么**：不复制各篇的论证，不做七路之外的推断。任何一条结论要落到具体代码，回到对应走查里按行号核对。

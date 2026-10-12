# 五条路线对照总表

> 本文是**索引与总表**，不是又一篇走查：只放跨实现才看得见的对照，论证留在各篇里。
> 需要细节时按文中的节号回到对应文档，本文不重复它们的推理过程。

收录的五条路线，各自有一篇逐环节走查：

| 实现 | 走查 | 一句话 |
|---|---|---|
| RYOTIDE | [`ryotide/typed-decision-model.md`](ryotide/typed-decision-model.md) | 零训练，把冻结 LLM **读**成分类器 |
| Jeff | [`jeff-serve/typed-decision-model.md`](jeff-serve/typed-decision-model.md) | 在 RYOTIDE 那条读法之上**训**一个决策头 |
| Open-Jev | [`open-jev/typed-decision-model.md`](open-jev/typed-decision-model.md) | 把决策拆成 N 个 **pointwise 是非题**，再训一个标量头 |
| Quyet | [`quyet/typed-decision-model.md`](quyet/typed-decision-model.md) | 自训权重，encoder 用槽位、LLM 用字母 |
| Laya | [`laya/typed-decision-model.md`](laya/typed-decision-model.md) | 双向编码器，把决策做成**填空题**：每选项一个 `[MASK]` 槽位 |

共同底线只有两条：**一次前向、不生成任何 token（`output_tokens = 0`）**，以及**输出封闭在已知选项集上**。

> 「只读最后一个位置」**不是**共同底线。五篇里只有 RYOTIDE、Jeff、Open-Jev 读末位；Quyet 的 encoder 路径
> 和 Laya 读的是各选项自己的槽位（各一篇的 §5 / §4）。差别集中在**读哪个位置**和**置信度从哪来**。

---

## 1. 主对照表

| 维度 | RYOTIDE | Jeff | Open-Jev | Quyet | Laya |
|---|---|---|---|---|---|
| **权重来源** | 公开 LLM（Qwen3.5 / Gemma 4），全程 frozen | Gemma 4 31B + LoRA，**已合并**后发布 | Qwen3.5 + LoRA + 标量决策头（走查以 2B 为例） | 自训权重（`chinhnc/Quyet-1.0-*`） | 自训权重（`convaiinnovations/laya`，三个 checkpoint） |
| **是否训练** | 否，零新增参数 | 是：训一个决策头 | 是：LoRA + 一个标量头 | 是：打分头 / 字母读法 | 是：双向编码器 + 决策头，RLCD（其 §11） |
| **决策机制** | 读**标签 token** 的 logits → 掩码 → softmax | 读末位 hidden state → **决策头** → softmax | **逐候选 pointwise 打分** → softmax | 槽位 logits（encoder）/ 字母 logits（LLM）→ 掩码 → softmax | 每选项一个 `[MASK]` 槽位 → 共享 `scorer` 压成标量 → softmax |
| **「typed」从哪来** | 分词器：标签必须单 token，否则映射成字母标记 | 单 token 标签表 255（26 字母 + 229 双字母） | 请求声明的 `answer_keys` + 响应概率校验 | 训练时固定的槽位：`[MASK]` / 字母 `A..J` | `[MASK]` 槽位；**与选项文本长度、分词器都无关**（其 §2） |
| **读位置怎么定** | **运行期标定**：扫 9 前缀 × 2 表面形式，取 marker mass 最高（其 §3） | **训练期固定**：与 prompt 格式一起冻结（其 §4） | **无标定**：`Yes or No` 是固定问句（其 §9.2） | **训练期固定**：运行期只有 `prompt_version` 一个常量（其 §0） | **训练期固定**：位置是构造出来的 `marker_pos`；布局写进 checkpoint config（其 §3） |
| **读出来的是什么** | 末位**词表 logits**，再按标记 id 取出 | 末位 **hidden state**，天然不物化 vocab 维度 | 末位 hidden state → 头 → **每候选一个标量** | encoder：每槽位一个标量；LLM：末位字母 logits | **每个选项一个槽位**的隐状态 → 标量（K 个位置，不是末位） |
| **候选数上限** | 260（`A0`–`Z9` 两 token 两步读；>26 另有 `grouped` 近似读法） | 255（标签表上限；头槽位默认 16、上限 255） | 255（接口约定，一步打分） | 10（`A..J` / 槽位数，超出直接报错） | **无硬上限**：受 token 预算与「槽位不许塌陷」约束（其 §2） |
| **顺序偏置** | 多 order 共享前缀 fork，折叠回真实标签后**概率平均**（其 §6） | 只给 4 种 yes/no 渲染**开关**，本身不平均（其 §6） | **不存在该变量**（pointwise 无顺序，其 §4） | 未涉及（其 §4 讲的是两条路径的标签槽位顺序，不是选项顺序偏置） | **训练时消解**：`parallel` 布局让位置对选项不可见（其 §6） |
| **置信度来源** | **读出来的**：只有 temperature 能改数值 | **训出来的**：头的 A、b；训练后只剩 serving rule 两项 | **训出来的**：软目标 CE + Brier；temperature 二次校正 | **训出来的**：T 写在 `quyet_config.json`，随权重发布 | **训出来的**：RLCD；但**两个都叫 `confidence`**，另有 `x_jev_confidence`（其 §10） |
| **诊断量 / 守卫** | `marker_mass`（全词表落在标记上的份额） | 无同等量；`runtime` 只有 `readout` / `probability_origin` / `serving_rule` | 概率校验 + `max_length` 硬报错 + 拒绝服务未加载模型 | 截断上报 `truncated` + `warnings`；加载期断言字母单 token | `options_distinct` vs `options`、`tokens_per_option`（槽位塌陷）+ `state_tokens_dropped` |
| **多问题共享 state** | `run_many()`：公共前缀预填充一次，**计费只算一次**（其 §7） | **不做**：朴素逐题循环，state 重复编码、全额计费（其 §7） | 用两级 LCP 前缀缓存，**默认关闭**（误差超容限，其 §7） | **不做**：逐问题累计 token，state 重复编码（其 §8） | state 只**分词**一次，但**计费逐题累计**；双向编码器无法复用前缀（其 §7） |
| **路由（逐请求挑 checkpoint）** | 无 | 无 | 无 | 无（`kind` 在 config 里定死） | **有**：纯 Python 启发式，不跑模型（其 §8） |
| **后端** | MLX（参考）+ torch | torch（CUDA），cuDNN attention 默认关 | torch + 自研 Triton final-row 内核 | encoder 与 LLM 两条**并行** runtime | torch 四种：eager / compile / tilelang / onnx |
| **服务接口** | `/v1/systemone` + `/health`（含 `prompt_hash`），另有 MCP | `/v1/systemone` + `GET /`（**无** `prompt_hash`），单线程串行 | `/v1/systemone` + `/health`，有界批处理 | CLI（JSON Lines 进出）+ `examples/gem_run` | FastAPI `/v1/systemone` + `/batch` + `/health`，另有 stdio MCP |

---

## 2. 谁在改发布出去的数值

这张表把五篇各自的「置信度链条」并排放。**跨实现比较概率之前，必须先在各自那一列里确认全部环节**。

| 实现 | 改数值的环节 | 改 argmax？ | 关键约束 |
|---|---|---|---|
| **RYOTIDE**（其 §9.0） | ①温度 ②上下文校准 bias ③第二模型一致性重加权 | ①否 ②理论可以 ③否 | 已发布 preset 里**只有温度生效**；bias 未接入（见 §4.5），重加权仅离线测量 |
| **Jeff**（其 §9.0） | ①**训练本身**（头的 A、b）②`temperature` kind ③serving rule 温度 ④`noul_edge` 带内推移 | ①**是** ②否 ③否 ④否 | 训练之后只有 ③④ 能动数值；④**会破坏单调性**（见下） |
| **Open-Jev**（其 §6） | ①**训练本身**（软目标 CE + Brier）②温度校正 | ①是 ②否 | 温度**无条件写入**，无接受 / 拒绝门槛 |
| **Quyet**（其 §5） | ①**训练本身** ②`softmax_t` 温度 | ①是 ②否 | `softmax_t` 是**唯一**改发布数值的步骤；缺键 = 1.0 = 未校准 |
| **Laya**（其 §10.1） | ①**训练本身**（RLCD）②温度 ③逐语言温度 ④分箱图 | ①**是** ②否 ③否 ④否 | 温度**无条件发布**，无接受 / 拒绝门槛；`clamp` 自陈「只限界损害，不校准任何东西」（其 §10.3） |

温度的具体值，五篇的来源各不相同，**没有一个可以跨实现借用**：

| 实现 | T 从哪来 | 实际值（含被拒的） |
|---|---|---|
| RYOTIDE | 离线拟合 + **预先注册的接受 / 拒绝判据** | Gemma 1.924 采纳、Qwen9 0.667 采纳、Qwen3.5-4B 拟合 1.592 但**留出集变差被拒**，保持 1.0 |
| Jeff | `serving_rule.json` 配置（另有逐输入学出来的 `temperature` kind） | 逐类型可配，缺失即恒等 |
| Open-Jev | `train.py` 拟合后直接发布 | 2B 发布值 **1.7360720317205194** |
| Quyet | 写在模型自带的 `quyet_config.json` | 分桶：encoder 用 `<type>:<桶>`，LLM 用纯题型键 |
| Laya | LBFGS/NLL 离线拟合，**无接受 / 拒绝判据**（`train.py:1138`） | 三层：逐题型 + 逐选项数分桶 + 逐语言；`choice:11+` 拟合出 **0.1006**，被 `clamp` 拒到 `TEMP_MIN = 0.5`（其 §10.3） |

### 温度的代价，五条记录四种表现

- RYOTIDE §12(f)：同一条记录顶部概率 T=1 时 **0.999917**、T=1.924 时 **0.981007**；
- Open-Jev §10(i)：255 候选下顶部概率 T=1.736 时 **0.107047**、还原 T=1 后 **0.575983**（T>1 的压平效应随候选数放大）；
- Jeff §9.1：100000 组随机 `(p, T)` 中 argmax **改变 0 次**——温度不改准确率，所以能被单独拟合与事后报告；
- Laya §10.3：`choice:11+` 的 **0.1006** 会把真实的 0.24 发布成 **0.99**，clamp 到 `TEMP_MIN=0.5` 之后仍发布成 **0.4993**——源码原话是 *The clamp bounds the damage; it does not calibrate anything*；
- 而 Jeff 多一项 `noul_edge`：`0.7999 → 0.801`，`0.8001 → 0.8001`，**raw 更大报出去更小**，跨样本比较被它改写的 P(yes) 不可靠。

---

## 3. 四条不变量的跨篇落位

RYOTIDE §13 那四条是这组文档的共同骨架。它们在另外四篇里的落位**并不整齐**——有两条在「打分量不是词表分布」的实现里根本不适用，需要改写成等价命题；而 Laya 甚至连 #3 都要改写，因为它读的就不是最后一个位置。

| RYOTIDE §13 不变量 | Jeff（其 §13） | Open-Jev（其 §9） | Quyet（其 §9） | Laya（其 §12） |
|---|---|---|---|---|
| **#1** 读位置必须落在标记上（诊断量 `marker_mass`） | 结构性成立，但**无诊断量** | **概念不适用**：打分量是 head 的标量，不是词表分布，没有 `marker_mass` 这个东西 | 改写为「只有选项槽位的 logit 进入 softmax」 | **结构性成立，且有诊断量**：`options_distinct` vs `options`（槽位塌陷）、`tokens_per_option` |
| **#2** 前缀与标记表面形式耦合，必须成对扫 | 成立，且被**训练固化**；代价是不能换 prompt 格式 | **不适用**：没有 `SUFFIXES`/`PATTERNS` 扫描，也没有运行期标定 | 改写为「标签必须落到单 token（LLM）/ 单个 MASK 槽位（encoder）」 | **不适用**：标记是 token id 不是字符串，不存在「表面形式」这个自由度 |
| **#3** 只读最后一个位置 | 成立，且更彻底（连 vocab 维度都不物化） | 成立 | 成立 | **改写**：不是「只读一个位置」，而是每条序列读它自己的 K 个槽位；「不物化词表维度」这条同样成立，机制不同 |
| **#4** 比较概率 / ECE 前先确认 temperature | 成立，且**多一项 `noul_edge`** | 成立，`temperature.json` 随 checkpoint 发布 | 成立，T 在 `quyet_config.json` | 成立，但**要确认的不止 T**：还有两个 `confidence` 量与该用哪一个、`x_jev_confidence` |
| （各自新增的） | **单 token 标签表与分词器绑定**——换分词器必须重标 | — | **只有 state 被截断，且截断必须被上报** | **布局是权重的属性**（`option_layout` 写死在 config）；**归一化常量**（`k / 255.0`、桶边界 `2/3-5/6-10/11+`） |

---

## 4. 同一个坑，各踩各的

这一节是并读五篇才有价值的部分：同一个失效模式在五条路线上各出现一次，**修法不同、能不能被发现也不同**。

### 4.1 `enable_thinking=False`：一条坑，三种接法、两种不适用

读位置落进 `<think>` 块，概率质量到不了选项标记上——四条路线都踩过，发现方式与修法各不相同；Laya 则根本没有这个位置（表末一行）。

| 实现 | 怎么发现 | 怎么修 |
|---|---|---|
| RYOTIDE | **靠 `marker_mass`**：只有约 15% 质量到达标记，而适配器照样返回格式正确的分布、harness 照样打分（分数 27.4 → 61.8） | `_tokens()` 传 `enable_thinking=False`，模板不接受时回退 |
| Jeff | 不靠诊断量，**设计上消掉**：读位置固化在模板末尾 | 头就训在那个位置上；代价是**失去对应的诊断量** |
| Open-Jev | 不适用（无标记概念） | `model.py:234` 传 `enable_thinking=False` |
| Quyet | 不靠诊断量 | prompt 构造处**写死**，`TypeError` 时回退 |
| Laya | **这个坑不存在**：`laya/` 核心没有 `enable_thinking` / `chat_template` / `padding_side`（grep 为空） | 不适用——它不是生成模型，序列由 `build_sequence` 手工拼装，没有「思维块」这个位置可以落错 |

### 4.2 填充方向：两条都精确，依赖的性质不同

| 实现 | 填充方向 | 为什么不影响结果 |
|---|---|---|
| RYOTIDE | **右填充且不传 mask**（torch 批量读） | 因果模型看不到自己之后的 padding |
| Open-Jev | **右填充**（`padding_side = "right"`） | 同上，不污染真实末位置 |
| Quyet | **左填充且必须传 mask** | 位置整体平移，RoPE 只看相对距离；padding 由 mask 屏蔽 |
| Laya | **右填充且必须传 mask**（`pad_batch`，`backends/base.py:41-71`） | 它是**双向**编码器，padding 在左侧也会被 attend 到，所以 `right` 并不自动安全，必须靠 mask；另外填充行**特意留一个活 marker**（`:59-68`），否则全掩码行变 NaN——而批里一个 NaN 就足以毒掉 CUDA graph 的 replay |

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

### 4.4 置信度：读出来的 vs 训出来的

这是五条路线最根本的一条分界，也决定了「温度能不能事后单独拟合」：

- **RYOTIDE 读**：分布是模型自己的 softmax，训练不参与。于是温度**不可能改准确率**，可以被离线拟合、被预先注册的判据接受或拒绝（§2）。
- **Jeff / Open-Jev / Quyet / Laya 训**：训练本身就在改数值、也可能改 argmax。温度退化成**训练之后的二次校正**，不再是唯一变量。
  - Open-Jev 对此最宽松：拟合值**无条件写入**，没有接受 / 拒绝门槛（其 §6 注：*"Calibration temperature is not a held-out capability or safety result"*）。
  - Laya 同样**无条件发布**（`train.py:1138`），且发布路径上的 `clamp` 自己承认 *The clamp bounds the damage; it does not calibrate anything*（其 §10.3）。
  - Quyet 把它固化进权重包：T 与「缺键 = 1.0 = 未校准」的默认一起发布。
- **Laya 另有一层别家没有的麻烦**：同一条答案里**两个字段都叫 `confidence`**，只有一个（`answer_confidence`）是温度拟合与 ECE 的对象，另一个是归一化熵；服务侧还会注入第三个 `x_jev_confidence`。所以 §3 的 #4 在 Laya 上要改写成「确认用哪一个」，而不只是「确认 T」。

### 4.5 「算得出来 ≠ 接进来了」

五篇各自记了一处「代码写了却不在生效路径上」或「接了却没有收益」，性质各不相同：

| 实现 | 位置 | 性质 |
|---|---|---|
| RYOTIDE | `calibration_bias()`（其 §9.3） | **全仓零调用点**，且六臂实验实测**没赢**（letters 0.675 vs letters+cc 0.600）——是「试过之后的结果」，不是忘了接线 |
| Jeff | `typed-*` 两种头（其 §3） | 发布路径上**不可达**：`Decider.decide` 调 `option_logits(h, k)` 时从不传 `q`，而 typed 头要求显式传入 |
| Quyet | encoder 的 `MASKED_LOGIT`（其 §3.1） | 在主路径上是**冗余**：`runtime` 只取 `lg[j, :len(markers)]`，补齐槽位本来就进不了 softmax——写了但改变不了任何数值 |
| Laya | `loss="soft-ce"` 路径（其 §11） | **加了没收益**：源码记 *#741 measured no gain from the extra term on the typed-decisions split*——策略梯度项在那份 split 上没测出收益，被诚实地记成负结果 |
| Open-Jev | 反面教材（其 §8） | `device_map="auto"` 让 12 个模块落到 disk、决策头留在 `meta`，**而服务仍然对 `/health` 回 "ready"，并从没加载的模型返回自信的数字** |

前三条是「写了没生效」，Laya 那条是「接了没收益」，Open-Jev 那条是「生效了但是没有效」——最后一条最难发现，因为它连错都不报。

---

## 5. 数字的可比较性：四处口径不一致

跨实现引用概率、准确率、ECE 之前，有四组口径必须先对齐，否则会得到看似合理的错数。

1. **温度**（§2）。五篇都记录了同一个现象：同一条记录、同一个 argmax，顶部概率可以差一个量级。
2. **ECE 分箱**。RYOTIDE 走的是 vendored JevBench 的 **10 个等宽分箱**（`vendor/jevbench/jevbench/metrics.py:55`）；Open-Jev 与 Laya 各自实现的是 **15 箱**（分别 `metrics.py:98` 的 `n_bins=15`、`common.py:766` 的 `bins: int = 15`）。**10 箱与 15 箱的数字不可直接互比**（Open-Jev §6 明确记了这一点）。
3. **calib 轴的公式**也可能不是一个。RYOTIDE 记录了看板上它那一列的 calib 与官方列**并非同一个式子**（本地评分只用 ECE，官方在能拿到 gold 分布时还加一项 TVD，其 §9.5）。
4. **归一化常量与分桶边界**。Quyet 与 Laya 都用 `2 / 3-5 / 6-10 / 11+` 这套选项数分桶（Quyet 其 §5、Laya 其 §10.2），而 Laya 的 act 头还用 `k / 255.0` 归一化选项数（`common.py:650`）。同一个「11+ 桶」在两边指的不一定是同一组候选规模。

另外四处「数字归属」的坑，各篇都自己声明了，引用时不要越过：

- RYOTIDE §12 的两条记录是 `temperature 1.0` 的 run，**不是**已发布 preset 的 1.924；
- Open-Jev §10 的两条记录 `max_length` 配置不同（16384 vs 4096），只是都没触及上限；
- Jeff **没有跑过模型前向**（权重约 61 GB 未下载），全文**不给任何前向产生的概率数值**；
- Laya 同样**没有跑过前向**（本机无 torch、无 checkpoint）。它那篇 §6 的 3.59 与 §7 的 3.1 倍是**源码注释里记录的实测**，不是该文跑出来的——引用时按 `common.py:280-288` 与 `common.py:416-424` 回源核对。

---

## 6. 怎么用这几篇

- **想理解一条路线**：直接读它那篇，五篇都能独立读——它们各自的 §0 已经把与 RYOTIDE 的分歧讲清了。
- **想选型**：从 §1 的主对照表进，按「要不要训练」「候选数上限」「能不能换 prompt 格式」「多问题要不要共享 state」「要不要逐请求挑 checkpoint」五个问题筛。
- **想踩坑**：读 §4。这一节的每一条在五条路线上都真实发生过，且**修法不可互相照搬**——尤其 §4.2 的填充方向（Laya 是双向编码器，右填充**并不**自动安全）和 §4.3 的 cache 别名。
- **想引用数字**：先读 §5，确认温度、ECE 口径、分桶边界和记录来源。
- **本文不做什么**：不复制各篇的论证，不做五路之外的推断。任何一条结论要落到具体代码，回到对应走查里按行号核对。

# JevBench（Benchmark Heaven）指标与打分口径

本文说明 **Benchmark Heaven 的 JevBench** 各项关键评价指标**如何计算**，逐条给出代码位置与公式，便于核对与复算。

> **命名消歧。** 本篇说的是 **Benchmark Heaven 的 JevBench**（上游仓库 `fstandhartinger/jevbench`，看板 <https://benchmarkheaven.com/jev-models>）。另有一个同名的 **metamorphic coherence benchmark**（<https://jevbench.github.io/>），它测的是*一致性*而非准确率，与本篇不是同一个项目 —— 本仓 [`jev-benchmarks.md`](jev-benchmarks.md) 第 1 条索引的是后者。

## 溯源

| 项 | 值 |
|---|---|
| 上游仓库 | <https://github.com/fstandhartinger/jevbench> |
| 快照 commit | `7625ecb1b075cfe4d09d34141d353325825eea1d` |
| 上游源路径 | **无**：本篇不是副本，是读上游 `jevbench/`（`scoring` / `metrics` / `summarize` / `runner` / `budget` / `composite_v12`–`v14`）与 `docs/METHOD-v1.5*.md` 在本仓库直接写成的；上游 `docs/` 下没有对应的原件 |

本篇所有 `文件:行号` 形式的引用（如 `jevbench/composite_v13.py:54`）**均相对上游 jevbench 仓库的根目录**，本仓库只留文档，不含任何源码。行号对应上表快照 commit；上游若已前进，请按该 commit 比对。

**阅读前须知（版本口径）**

- 上游**已实现**的打分链路是 v1.2 → v1.3 → v1.4（`jevbench/composite_v12.py`、`composite_v13.py`、`composite_v14.py`）。
- **v1.5 的冻结方法**见上游 `docs/METHOD-v1.5.md`（<https://github.com/fstandhartinger/jevbench/blob/main/docs/METHOD-v1.5.md>），其打分器未包含在公开仓库中。本篇中 v1.5 的公式均以该文件（及 `docs/METHOD-v1.5-README.md` 所列修订件）为准。
- 上游 README 中当前公布的版本为 **v1.6.1**，其方法文档不在公开仓库内；本篇只在与上游代码/文档有明确对应处提及。
- **不要跨版本比较分数**：各版本的任务集与打分定义都不同，历史版本保留其原始定义。

---

## 0. 计算流水线总览

所有指标由四层构成，下层是上层的输入：

```
① 逐题打分   scoring.py        分布/标签 → valid / correct / predicted / ordinal_ev
        │
② 聚合指标   metrics.py, summarize.py   准确率、Brier、ECE、MAE、延迟、成本、复述一致性
        │
③ 四个轴     composite_v12/v13/v14.py   Intelligence / Calibration / Speed / Cost（0–100）
        │
④ 综合分     同上               JevBench Score（几何平均或调和平均 + 门槛惩罚）
        │
⑤ 统计判定   配对分层 bootstrap  并列标记、"#1" 判定
```

所有轴与综合分都取值在 **0–100**，并做 `clamp` 截断（`composite_v13.py:50`）。

---

## 1. 逐题打分（`jevbench/scoring.py`）

这是所有指标的原子单位：把系统对一个决策题的原始输出，转成 `valid / correct / predicted / ordinal_ev` 四个字段。

### 1.1 分布合法性：双容差规则

`validate_probs()`（`scoring.py:27`）要求：

1. 键集合与题目 `labels` **完全一致**（既不能缺、也不能多）；
2. 每个值是 `[0,1]` 内的有限浮点数（`bool` 被显式拒绝）；
3. 和 = 1，容差 `SUM_TOL = 1e-3`。

实践中发现固定 1e-3 主要在筛选"三位小数四舍五入"（如九选项题的 0.999），因此采用**双容差**（`scoring.py:11-20`）：

| 标记 | 容差 | 含义 |
|---|---:|---|
| `strict_valid` | `SUM_TOL = 1e-3` | 预注册的严格规则，作为对照列公布 |
| `valid`（头条） | `RENORM_TOL = 2e-2` | 落在 2% 带内的分布视为"四舍五入"，**重新归一化**后正常打分，`renormalized=True` |
| 带外 | — | `valid=False`，**计错**，绝不合成概率 |

重新归一化只做缩放 `p_k / Σp`；带外或全零分布直接判错（`scoring.py:89-101`）。

### 1.2 判定与期望值

- **argmax**（`scoring.py:59`）：取概率最大的标签；**并列时取字典序最小的标签**，保证确定性。
- **期望值 EV**（`scoring.py:68`）：`Σ k · p_k`，`k` 是等级索引的字符串形式。仅 `score` 类型题使用。
- `score_task()`（`scoring.py:77`）对 `score` 题**同时**给出 `predicted = argmax` 和 `ordinal_ev`；`correct` 依据 argmax 与金标是否相等。

### 1.3 标签型系统（无分布）

`score_label()`（`scoring.py:127`）用于只返回标签、不返回概率的系统（如 Needle 3）：

- 标签在标签集内 → `valid=True`；**`probs=None`**，因此所有校准类指标对它自动跳过；
- 弃答（`label is None`）→ `valid=True` 但**计错**；
- 不产生任何概率，也**不会**在下游被补成分布。

### 1.4 计错规则汇总

以下情形一律计错（且不修复）：分布非法、超容差、弃答、请求失败、schema 不合法。例外是 HTTP 422 —— 视为"系统拒绝该输入"的**错误答案**，不计入基础设施错误累计（`runner.py:59-63`）。

---

## 2. 聚合指标（`jevbench/metrics.py`、`jevbench/summarize.py`）

`summarize.metric()`（`summarize.py:6`）把逐题记录汇总为下表字段。

### 2.1 准确率与覆盖率

| 指标 | 计算 | 代码 |
|---|---|---|
| `accuracy` | 正确数 / **可评分题数**（`expected is not None` 且无 `exclude_reason`） | `summarize.py:14` |
| `macro_accuracy` | 各家族准确率的**未加权平均**，只计入 `n_scorable > 0` 的家族 | `metrics.py:26`、`summarize.py:31` |
| `coverage` | 已尝试 / 计划题数 | `summarize.py:14` |
| `schema_validity` | `valid` 数 / 已尝试（头条口径） | `summarize.py:14` |
| `schema_validity_strict` | `strict_valid` 数 / 已尝试（预注册口径） | `summarize.py:14` |
| `n_renormalized` | 触发重新归一化的题数 | `summarize.py:14` |
| `operational_success` | `ok` 数 / 已尝试 | `summarize.py:14` |

说明：分母是**可评分**题数而非全部题数；`expected=None` 的题只进覆盖率、不进准确率。

### 2.2 Brier 分数

`brier_score()`（`metrics.py:32`）为多分类平方和：

```
Brier = Σ_k (p_k − y_k)²          # y_k = 1 当且仅当 k 为金标
```

二分类（`noul`）刻意采用**二分类两标签约定**，使数值与多分类可比：

```
Brier = (p_yes − y)² + (p_no − (1 − y))² = 2 · (p_yes − y)²
```

聚合值 `brier_mean` 对所有返回了分布的**可评分**题取算术平均（`summarize.py:8-14`）。标签型系统无分布，被排除而非计 0。

### 2.3 ECE（期望校准误差）

`ece_top_label()`（`metrics.py:55`）：

1. 只取**最高标签置信度**（top-label confidence，`scoring.py:73`）；
2. 分成 **10 个等宽区间**，`idx = min(int(conf × 10), 9)`，置信度先裁到 `[0,1]`；
3. 按样本量加权：

```
ECE = Σ_b (n_b / N) · |acc_b − mean_conf_b|
```

空箱不参与（返回 `None` 而非 0），同时输出每箱的 `n / mean_confidence / accuracy` 供检查。校准的样本数 `calibration_n` 一并公布。

### 2.4 序数平均绝对误差

`ordinal_mae()`（`metrics.py:95`）对 `score` 类型题计算 `mean(|EV − 金标等级|)`，与 argmax 准确率**并列报告**（`summarize.py:12`），不是替代关系。

### 2.5 延迟

`latency_summary()`（`metrics.py:114`）用线性插值百分位（`metrics.py:102`）给出：

- `p50_s`、`p95_s`：跨所有尝试，含网络时间的调用方墙钟时间；
- `latency_failures`：失败请求单独汇总；
- `prepare_s`：仅少数适配器有的一次性装配时间（如 ProgramAsWeights 编译程序），**不计入决策延迟**（`runner.py:16-22`）。

### 2.6 成本

单题成本在 `runner.py:31-42` 计算：`input_tokens × 输入单价 / 1e6 + output_tokens × 输出单价 / 1e6`，口径标记为 `derived_usage_times_tariff`。

关键约定：

- **无计费路径**（公开 demo、自有 CPU、包月订阅）成本为 **`null`**，标记 `zero_route_fee_compute_excluded`，理由是"未计量 ≠ 免费"，画成 0 是无法溯源的主张；
- 解析失败但已被计费的请求仍按预留额结算，不留空；
- 聚合价格 `price_per_1000_decisions_usd = Σcost × 1000 / n`，且**只有当所有记录的成本都非 `null` 时**才给出（`summarize.py:14`）。

### 2.7 复述一致性 / 一致性

同一场景的改写题共享 `group` id。仓库内有两套实现，注意区分：

| 函数 | 字段 | 含义 |
|---|---|---|
| `summarize.agreement()`（`summarize.py:15`） | `agreement` | 两者**预测标签相同**的比例（分母 = 双双合法的对数） |
| 同上 | `both_correct_rate_all_pairs` | 两者**都答对**的比例（分母 = **全部**配对组数） |
| `metrics.paraphrase_consistency()`（`metrics.py:123`） | `consistency` | **both-correct** 率：任一成员 `expected=None` 则该对不可评分，计入 `unscorable_expected_none`，不进分子 |

实际进入发布摘要的是 `summarize.agreement()`（`summarize.py:31`）。之所以同时报两个数：**只看"答案相同"会奖励一个稳定答错的模型**。

---

## 3. 四个轴（`jevbench/composite_v1*.py`）

### 3.1 Intelligence

公式随版本演进，最新（v1.5 冻结）如下。

**v1.2**（`composite_v12.py:51`）—— 分层加权准确率：

```
I = 100 · Σ_t w_t · acc_t / Σ_t w_t       w: easy .14 / standard .28 / judge .28 / hard .30
```

未跑的层被剔除，剩余权重**重新归一化**。

**v1.3**（`composite_v13.py:54`）—— 加入机会基线校正：

```
CC = clamp( 100 · (acc − c) / (1 − c) )
```

`c` 由**选项数直方图**算出（`composite_v13.py:12-27`）：`c = 平均(1 / 选项数)`，即逐题均匀猜测基线。四层基线的可复算值：

| 层 | 选项数分布 | 层基线 c |
|---|---|---:|
| easy | 2:18, 4:13, 5:41 | 0.2840 |
| standard | 2:32, 4:40, 5:12, 6:12 | 0.3167 |
| judge | 2:68, 9:78 | 0.2922 |
| hard | 2:77, 3:26, 4:73, 5:38, 6:6 | 0.3364 |

**v1.4**（`composite_v14.py:24`）—— 混入密封集，并加过拟合惩罚：

```
sealed_cc = 100 · max(0, (acc_sealed − 0.293) / (1 − 0.293))
combined  = 0.80 · I_v13 + 0.20 · sealed_cc
gap       = 100 · (acc_public − acc_sealed)              # 原始百分点
penalty   = max(0, gap − 25) / 100                        # 每超 1 点扣 1%
I         = combined · max(0, 1 − penalty)
```

常数见 `composite_v14.py:11-15`（`SEALED_WEIGHT=0.20`、`SEALED_CHANCE=0.293`、`GAP_ALLOWANCE_POINTS=25`）。

**v1.5**（冻结方法 §3）—— 改为**逐题**基线 + 分请求类型 + 开/密封各半：

| 类型 | 公式 |
|---|---|
| Choice | `CC = 100 · (acc − c̄) / (1 − c̄)`，`c̄` 为该层各题 `1/选项数` 的均值 |
| Noul | `P(yes) ≤ 0.20` 判 No，`≥ 0.80` 判 Yes，**中间为弃答、计错**；`CC = 100 · (acc − 0.5) / 0.5` |
| Score | `CC = 100 · (1 − mean nMAE / mean nMAE_chance)`；`nMAE = \|pred − gold\| / (K − 1)`，`nMAE_chance` = 该题 K 个等级均匀猜测的 nMAE 均值。有分布时 `pred` 取期望位置，否则取等级标签 |

聚合：

```
CC_t,x   = Σ_tier w_tier · CC_t,x,tier        w: easy .10 / standard .20 / judge .30 / hard .40
I_x      = Σ_t  W_t · CC_t,x                  W: Choice .50 / Noul .25 / Score .25（按支持类型重归一化）
base     = 0.50 · I_open + 0.50 · I_sealed
```

过拟合惩罚改为**相对全场**（口径为同批次全覆盖率系统的 gap 中位数 `G_med`，v1.5.0 冻结后复用）：

```
excess       = (I_open − I_sealed) − G_med
penalty      = max(0, 1 − max(0, excess − 8) / 100)
Intelligence = clip(base · penalty, 0, 100)
```

> **头条权重修订**：v1.5 公开发布采用等轴等类型修订件（上游 `docs/METHOD-v1.5-ADDENDUM-HEADLINE-A-EQUAL-TYPES.md`，<https://github.com/fstandhartinger/jevbench/blob/main/docs/METHOD-v1.5-ADDENDUM-HEADLINE-A-EQUAL-TYPES.md>），即 Choice / Noul / Score **各占 1/3**（原冻结的 50/25/25 作为次要视图）。层权重 10/20/30/40 与开/密封 50/50 不变。Intelligence **按层不截断**，因此低于机会基线的噪声不会被抹平。

不支持的类型记为 "Not supported" 并**排除**，绝不记 0；只有覆盖三种类型的系统进入全覆盖率排名。

#### 3.1.1 层权重 14/28/28/30 的来源

这套权重只在 v1.2 冻结时定下，此后 v1.3、v1.4 沿用未改。仓库对它记录的完整程度不一致，**分清楚哪些有出处、哪些没有**。

**（a）算术推导——有记录。** `composite_v12.py:5` 的模块 docstring：

> hard 30 %, easy 14 %, standard 28 %, judge 28 % (**the 70 % left after hard, split 1 : 2 : 2**)

即两步：先固定 `hard = 30%`，再把剩下的 70% 按 `easy : standard : judge = 1 : 2 : 2` 分，得 70×1/5 = 14、70×2/5 = 28、70×2/5 = 28。因此**自由参数只有两个**：hard 占 30%，以及 1:2:2 这个比例。

**（b）取值的历史——有变更记录，无论证。** `RESULTS-v1.2.md:462`：

> Intelligence weights hard 30 % (**was 50 %**); the rest 1 : 2 : 2 over easy : standard : judge.

对照 `README.md:140`，v1.2-wip 阶段是「Balanced 33:33:33 Main Score with **hard 50 % of Capability**」。即 hard 从 50% 下调到 30%，其余三层从"等分 50%"变为"1:2:2 分 70%"。上游仓库**没有**留下为什么选 30% 和 1:2:2 的冻结前论证。

**（c）"为什么要难题层"的经验依据——有记录。** `RESULTS-v1.1.3.md:65`（v1.1 时代的前瞻观察）：在那套 220 题硬层上，各 entrant 得 44–66%，而 Jev 74.1%、GPT-5.6 Luna 94.5%，并明确写道「这里的接近 100% 的天花板掩盖了真实差距」。这是"区分度集中在 hard"这一判断的经验起点。

**（d）背后的原则——只在 v1.5 被写出来。** `METHOD-v1.5.md:174-177`（改权为 10/20/30/40 时给出的理由）：

> **Judge + hard 占 70%。** easy 和 standard 题区分系统的能力很弱：v1.4.2 中全场均值分别是 0.95 和 0.82 准确率，它们主要确认基本能力。judge 与 hard 承载区分信号，且代表 Jev 级系统值得部署的那类决策。**easy 保留 10%，以免一个连基础都做不对的系统被运气好的难题答案救回来。**

这条原则同样解释了 14/28/28/30 的**排序**（easy 最低、judge = standard、hard 最高），但仓库中没有把它与 v1.2 的具体数字挂钩的段落。

**（e）两个可验证的观察。**

- **不是按题量比例分配。** 各层题量 easy 72 / standard 96 / judge 146 / hard 220（`METHOD-v1.5.md:43`），按比例应为 13.5 / 18.0 / 27.3 / 41.2。实际权重把 standard 从 18% 拉到 28%、把 hard 从 41% 压到 30% —— standard 与 judge 被刻意置于**同等**权重，与各自题量无关。所以这是难度判断，不是抽样比例。
- **同一组比例被复用为难度剖面。** v1.5 的密封集抽样也按 14/28/28/30 分层（`METHOD-v1.5.md:59`），因此 14/28/28/30 不只是打分权重，同时被当作分层抽样的目标结构。

| 问题 | 仓库能否回答 |
|---|---|
| 14/28/28 如何从 30% 推出 | 能，`composite_v12.py:5` |
| 为何 hard 取 30% 而非 50% | 仅有变更记录（`RESULTS-v1.2.md:462`），无设计论证 |
| 为何取 1:2:2 | **无记录** |
| 为何难题层应占更高权重 | 有，`RESULTS-v1.1.3.md:65`（经验）与 `METHOD-v1.5.md:174`（原则，服务于 v1.5 的新数值） |

#### 3.1.2 这组权重是否合理（用 v1.2 数据实测）

§3.1.1 说明了权重**从哪来**；这里检验它**站不站得住**。检验口径：取 `results/v1.2/jevbench-v1.2-results.json` 中全部 `ranked` 且四层齐全的系统（48 条），逐层算机会校正分 CC 的**系统间标准差**作为区分度。复算脚本见 §8。

**（a）区分度并不支持 hard 拿最高权重。**

| 层 | 题量 | 权重 | 全场均值(原始) | CC 均值（未截断） | 区分度 SD（未截断） | SD（0 截断后，即打分器实际所见） |
|---|---:|---:|---:|---:|---:|---:|
| easy | 72 | **0.14** | 0.929 | 90.1 | 24.3 | 24.3 |
| standard | 96 | 0.28 | 0.799 | 70.6 | 29.5 | 29.4 |
| judge | 146 | 0.28 | 0.756 | 65.5 | **32.0** | 29.7 |
| hard | 220 | **0.30** | 0.540 | 30.7 | 26.1 | 25.9 |

（表中为未截断值；打分器实际使用的是 0 截断后的版本，judge 的 CC 均值为 66.4、hard 为 30.9。）

- **easy 压到最低是对的**：它已接近饱和（CC 均值 90.1），区分度垫底，给高权重只会让平庸系统搭便车 —— 与 `METHOD-v1.5.md:174` 写下的原则一致。
- **但 hard 的区分度只排第三**（26.1），实际却拿了最高权重（30）。区分度最高的是 **judge（32.0）**。
- v1.2–v1.4 的逐层 CC 带 `clamp(0,100)`，judge 层有 3/48 个系统被截到 0（SD 由 32.0 压到 29.7），hard 层 5/48（26.1 → 25.9）。截断**低估** judge 的区分度。v1.5 明确改为"逐层不截断"，正是在修这一点。

**（b）三种自然基准都不吻合。**

| 基准 | easy | standard | judge | hard |
|---|---:|---:|---:|---:|
| **实际权重** | 14 | 28 | 28 | 30 |
| 按区分度比例 | 21.7 | 26.4 | 28.6 | 23.3 |
| 按题量比例 | 13.5 | 18.0 | 27.3 | 41.2 |
| 按逆方差（噪声最优） | ≈ 题量比例 | | | |

- 相对**区分度**：hard 被高配（30 vs 23.3），easy 被低配（14 vs 21.7）。
- 相对**题量**：standard 被抬到 1.55 倍，hard 被压到 0.73 倍。standard + judge 只占 45% 的题量，却贡献约 **58%** 的 Intelligence 方差预算（各层方差占比：easy 2.4%、standard 30.9%、judge 26.7%、hard 40.0%）。

结论：这套权重既不是在**最大化区分度**，也不是在**最小化噪声**，它编码的是一个**价值判断**（"难题代表的决策更值得部署"）。这个判断本身站得住，但与任何一种可从数据推出的基准都不重合。

**（c）最弱的一环：standard = judge 被给了完全相同的 28%。**

两者在三项指标上都不同 —— 题量 96 vs 146、区分度 29.5 vs 32.0、CC 均值 70.6 vs 65.5。唯一能支持"等同"的说法是"两者的决策价值相当"，但那是**另一条原则**，不是 §3.1.1(d) 引用那条（那条按区分度排序，会把 standard 排在 judge 之下）。v1.5 改成 20 vs 30，方向正确。

**（d）后果一：对最终榜单几乎无影响。**

把权重换成等权 / 按题量 / v1.5 的 10/20/30/40，用其余三轴不变重算综合分：

| 替换为 | ΔIntelligence 平均 | ΔIntelligence 最大 | 综合分 Spearman ρ | 最大位次变动 | Top-5 集合 |
|---|---:|---:|---:|---:|---|
| 等权 25/25/25/25 | 4.26 | 9.01 | 0.990 | 5 | 相同 |
| 按题量比例 | 4.56 | 8.39 | 0.995 | 5 | 相同 |
| v1.5 10/20/30/40 | 4.84 | 8.13 | 0.992 | 5 | 相同 |

只看 Intelligence 轴排序时 ρ = 0.998–0.999、最大位次变动 2–4。所以这组权重**没有扭曲头条**，不构成"有害"。

**（e）后果二：对 Intelligence 轴值是一个远超噪声的杠杆。**

ΔIntelligence 平均 **4–5 CC 点**、最大约 **9 点**。对照方法自己算出的配对差值 95% 半宽 **±2.8**（单系统 SE ≈ 1.6，`METHOD-v1.5.md:117,122`），这个变动**远大于测量误差**。也就是说 "Intelligence 74.4" 这个数字里嵌着一个与噪声同量级甚至更大的编辑选择。

而且它不中性：`corr(ΔI, hard 准确率) = +0.37` —— 调高 hard 权重重奖难题强的系统。这符合设计意图，但它意味着权重选择是一次**有方向的倾斜**，理应披露。

**（f）判断汇总。**

| 维度 | 评价 |
|---|---|
| 方向（easy 最低、难题层更重要） | **合理**，有数据支持 |
| 不按区分度 / 题量 / 方差推导 | **可接受**（它是价值判断而非估计量），但应说明 |
| 具体取 30% 与 1:2:2 | **任意**，落在很宽的合理区间内 |
| standard = judge | **最可质疑**，与后来写下的原则冲突 |
| 对最终榜单的影响 | **可忽略**（ρ ≈ 0.99） |
| 对 Intelligence 轴值的影响 | **4–5 分，远超噪声** |
| 记录完整性 | **不合格**：与噪声同量级的编辑杠杆，无冻结前论证 |

一句话：这套权重**能用、方向对、没造成伤害**，但它是一次约 4–5 分的手动调节，既未论证也未披露 —— 问题出在过程，不在数字。

> **适用边界**：以上敏感性分析建立在 v1.2 这 52 个系统的分布上。若参赛池高度集中在某一层，权重的影响会更大；区分度一列的绝对值也受逐层 0 截断规则影响。

### 3.2 Calibration

**v1.3 及以后**（`composite_v13.py:95`、`composite_v12.py:89`）：

```
Calibration = ( max(0, 100·(1 − ECE/0.5)) + 100·(1 − mean_TVD) ) / 2
```

其中 TVD（`composite_v13.py:91`）：

```
TVD(p, q) = 0.5 · Σ_label |p_label − q_label|
```

- 若 `mean_tvd` 为空，则只取 ECE 部分；
- **标签型系统无分布 → `None`**，在综合分中计 0（`composite_v12.py:97` 的 `max(axis or 0, 1)`）。

**v1.4**（`composite_v14.py:33`）把密封集测量混入：

```
Calibration = old + (new − old) · min(1, 0.20 / 0.35)     # ≈ old + 0.5714·(new − old)
```

**v1.5**（冻结方法 §5）改为**分类型的校准**，在公开集与密封集**合并**样本上计算（降噪），再按类型权重合成：

- Choice：ECE 与 TVD（概率题上）；
- Noul：`P(yes)` 上的 ECE，**Brier 作为附带报告**；
- Score：归一化 RPS 与顶层 ECE。

标签型输出贡献 0。

### 3.3 Speed

`composite_v12.py:68-76` / `composite_v13.py:68-82`，v1.5 沿用：

```
score(s) = clamp( 100 − 20 · log10(s / 0.1) )      # 0.1 s = 100 分，每慢 10 倍扣 20 分
Speed    = ( score(p50) + score(p95) ) / 2
```

延迟先做**端点调整** `adjusted_latency()`：

| 端点类型 | 处理 |
|---|---|
| `api`（生产 API） | 不变 |
| 自有 GPU / CPU | `×2`，再 `+0.15 s` |
| 作者 demo 服务器 / 自托管 | `×2` |

`×2` 是**假设而非测量**（我们自己跑时串行、无并发负载），原始 p50/p95 与调整后数值同时公布。v1.5 的测量范围是**全部公开集 standard + judge 题**的串行运行。

### 3.4 Cost

`composite_v12.py:79` / `composite_v13.py:85`，v1.5 沿用：

```
Cost = clamp( 100 − 30 · log10( (USD per 1,000 decisions) / 0.001 ) )
```

$0.001 / 1,000 次决策 = 100 分，每贵 10 倍扣 30 分。**价格缺失是错误，绝不自动记 100**（代码直接 `raise ValueError`）。v1.5 的 token 量在全部 1,624 次决策上汇总，价格按 9 月 24 日价格规则（无公开可订购价格者用基座模型参考价估计，并标注为估计）。

---

## 4. 综合分

### 4.1 v1.2 / v1.3：加权几何平均

`composite_v12.py:96`、`composite_v13.py:108`：

```
JevBench Score = exp( Σ_axis (w_axis / W) · ln( max(axis, 1) ) ) × near_chance_multiplier
```

- `max(axis, 1)`：把 0 分夹到 1 分，避免 `ln 0`；语义上"缺轴（标签型 Calibration）计 0 → 落到地板 1"；
- 近机会惩罚（`composite_v13.py:102`）：

```
near_chance_multiplier(I) = 1                       若 I ≥ 50
                          = (max(I, 0) / 50)²       若 I < 50
```

几何平均意味着**任一弱轴会把总分强力拉低**。

### 4.2 v1.4 及以后：加权调和平均 + 三重门槛

v1.4 起改用**加权调和平均**（幂平均 p = −1），并加独立门槛（`composite_v14.py:40`）：

```
H = Σw / Σ (w_axis / axis)          # 任一加权轴 ≤ 0 → H = 0；任一轴缺失 → None
H ×= (I/50)²       若 I < 50
H ×= (S/50)²       若 S < 50
H ×= (C/50)²       若 C < 50
```

门槛值 `JEV_CLASS_THRESHOLD = 50.0`（`composite_v14.py:15`）。要点：

- 调和平均 ≤ 算术平均，且**对短板更敏感**；等权 25/25/25/25 **不等于**算术平均（上游 README 明确提示）；
- 三个门槛分别惩罚智力、速度、成本过低的系统 —— 一个又快又便宜但判不准的系统不会因为两项高分而排到前面。

### 4.3 权重预设

`PRESETS`（`composite_v12.py:36-43`，v1.3/v1.4 同）：所有预设都用**同一种**合成方式重算，只是权重不同。

| 预设 | I / C / S / $ |
|---|---|
| JevBench Score (25:25:25:25) | 0.25 / 0.25 / 0.25 / 0.25 |
| Balanced 33:33:33 (no calibration) | 1/3 / 0 / 1/3 / 1/3 |
| Emphasis on Accuracy 60:20:20 | 0.6 / 0 / 0.2 / 0.2 |
| Emphasis on Speed 20:60:20 | 0.2 / 0 / 0.6 / 0.2 |
| Emphasis on Cost 20:20:60 | 0.2 / 0 / 0.2 / 0.6 |
| Intelligence only | 1.0 / 0 / 0 / 0 |

v1.5 在此之上冻结了三个正式选项（见冻结方法 §6）：

| 选项 | I / C / S / $ | Intelligence 门槛 |
|---|---|---|
| A：等权（**公布头条**，经修订件确认） | 25/25/25/25 | < 50 → (I/50)² |
| B：有效性加权（次要视图） | 40/20/20/20 | < 50 → (I/50)² |
| C：等权 + 更严地板 | 25/25/25/25 | < 60 → (I/60)² |

看到 v1.5 结果后把头条从 B 换成 A/C 属于"受结果影响的改动"，需要带公开说明的版本化方法修订。

### 4.4 当前公布版本（v1.6.1，来自上游 README）

上游 README 当前公布的头条口径与上述不同，且上游公开仓库内**无**对应方法文档：

- **Capability Score** = **Intelligence 与 Calibration 的算术平均**，是 open-weights 榜单的头条；
- **Composite Score** = 四轴加权调和平均 + 平方门槛惩罚（同上 §4.2），在 API 榜单上领先；
- 两个排名门槛：成本上限 **USD 0.0646 / 1,000 次决策**，调整后中位延迟上限 **1.23 s**。

---

## 5. 统计判定：配对分层 bootstrap

v1.5 冻结（方法 §2）：

- `B = 2,000` 次重采样，按 **split × type × tier** 分层重采样**题目**；
- **所有系统使用同一次重采样**（配对，降低差值方差）；
- 每次重采样重算 Intelligence、Calibration 与综合分；**Speed 与 Cost 固定**（噪声未建模）。

判读规则：

- 相邻名次的差值 95% 区间**包含 0** → 标 `≈`（统计并列）；名次仍按点估计排序；
- 只有在一系统于 **≥ 95%** 的重采样中胜过第 2 名，才称其为 **"#1"**；否则措辞为 "joint leaders (statistical tie)"；
- 同一规则适用于帖子中的 "beats" 断言。

背景噪声（方法 §2）：单系统 Intelligence 的 SE ≈ 1.6；开/密封差值的 SE 为 3.2 点（这也是过拟合豁免从 25 点收紧到 8 点的依据）。**v1.5 不主张 1 点差异是真实的**——要分辨 1 点差异约需每系统 12,000 次决策。

---

## 6. 版本沿革速查

| 指标 | v1.2 | v1.3 | v1.4 | v1.5（冻结） |
|---|---|---|---|---|
| Intelligence | 100×分层加权准确率（14/28/28/30） | 逐层机会校正 CC | +20% 密封集混合 + gap>25 惩罚 | 分类型 CC（Choice/Noul/Score）、逐题基线、层权 10/20/30/40、开/密封 50/50、惩罚相对 `G_med`（豁免 8 点） |
| Calibration | (ECE 项 + TVD 项)/2 | 同 v1.2 | 与旧值按 0.20/0.35 混合 | 分类型（ECE/TVD、P(yes) ECE+Brier、RPS+ECE），公开+密封合并 |
| Speed | 20 分/十倍，p50 与 p95 取均值 | 同 v1.2 | 同 v1.2 | 同 v1.2（测量集为公开 standard+judge） |
| Cost | 30 分/十倍 | 同 v1.2 | 同 v1.2 | 同 v1.2（1,624 次决策汇总） |
| 综合分 | 加权**几何**平均 + 近机会乘子 | 同 v1.2 | 加权**调和**平均 + I/S/C 三重门槛 | 同 v1.4；头条权重经修订件定为等轴等类型 |
| 并列判定 | 不显示 | 不显示 | 不显示 | 配对分层 bootstrap（B=2000） |

---

## 7. 代码位置索引

| 内容 | 位置 |
|---|---|
| 分布校验、双容差、argmax、EV、逐题打分 | `jevbench/scoring.py:11`、`:27`、`:59`、`:68`、`:77`、`:127` |
| 准确率、Brier、ECE、MAE、百分位、复述一致性 | `jevbench/metrics.py:26`、`:32`、`:55`、`:95`、`:102`、`:123` |
| 汇总字段、家族/分片聚合、一致性 | `jevbench/summarize.py:6`、`:15`、`:24` |
| 串行执行、计时、成本口径、错误规则 | `jevbench/runner.py:13`、`:31`、`:43`、`:59` |
| 预算预留/结算、价格表 | `jevbench/budget.py:8`、`:16` |
| v1.1：三个子基准 + 主分（算术） | `jevbench/composite.py` |
| v1.2：四轴 + 几何平均 | `jevbench/composite_v12.py` |
| v1.3：机会校正 + 近机会乘子 | `jevbench/composite_v13.py` |
| v1.4：密封集混合 + 调和平均 + 门槛 | `jevbench/composite_v14.py` |
| v1.5 冻结方法 | `docs/METHOD-v1.5.md` |
| v1.5 头条修订（等轴等类型） | `docs/METHOD-v1.5-ADDENDUM-HEADLINE-A-EQUAL-TYPES.md` |
| 指标性质与回归测试 | `tests/test_metric_properties.py`、`tests/test_composite_v12.py`、`tests/test_composite_v13.py`、`tests/test_v14_release.py` |
| 已发布结果的复算校验 | `tests/test_published_reproduction.py` |

---

## 8. 附：§3.1.2 的复算脚本

只读 `results/v1.2/jevbench-v1.2-results.json`，不写入任何文件。**在 jevbench 仓库根目录执行**：

```python
python3 jevbench_weights_check.py
```

```python
"""复算本文 §3.1.2 的四张表。"""
import json
import math
import os
import statistics
import sys

sys.path.insert(0, os.getcwd())  # 需在 jevbench 仓库根目录运行

from jevbench.composite_v13 import TIER_CHANCES, chance_corrected_accuracy

N = {"easy": 72, "standard": 96, "judge": 146, "hard": 220}
W0 = {"easy": 0.14, "standard": 0.28, "judge": 0.28, "hard": 0.30}
ALT = {
    "等权 25/25/25/25": {"easy": 0.25, "standard": 0.25, "judge": 0.25, "hard": 0.25},
    "按题量比例": {t: N[t] / 534 for t in N},
    "v1.5 10/20/30/40": {"easy": 0.10, "standard": 0.20, "judge": 0.30, "hard": 0.40},
}

data = json.load(open("results/v1.2/jevbench-v1.2-results.json"))
S = [s for s in data["systems"] if s.get("ranked") and s.get("tiers")]


def I_of(tiers, w):
    """按权重 w 聚合机会校正后的分层 Intelligence。"""
    num = den = 0.0
    for t, wt in w.items():
        if tiers.get(t) is None or wt == 0:
            continue
        num += wt * chance_corrected_accuracy(tiers[t], TIER_CHANCES[t])
        den += wt
    return num / den if den else None


def spearman(a, b):
    n = len(a)
    d = sum((a[k] - b[k]) ** 2 for k in a)
    return 1 - 6 * d / (n * (n * n - 1))


print("=== 表(a) 各层区分度 ===")
for t in ("easy", "standard", "judge", "hard"):
    acc = [s["tiers"][t] for s in S if s["tiers"].get(t) is not None]
    raw = [100 * (a - TIER_CHANCES[t]) / (1 - TIER_CHANCES[t]) for a in acc]
    print(f"{t:9s} n={N[t]:3d} w={W0[t]:.2f} 均值={statistics.mean(acc):.3f} "
          f"CC均值={statistics.mean(raw):5.1f} SD未截断={statistics.pstdev(raw):5.1f} "
          f"SD截断={statistics.pstdev([max(0.0, x) for x in raw]):5.1f} "
          f"截到0={sum(1 for x in raw if x < 0)}/{len(raw)}")

print("\n=== 表(b) 三种基准 ===")
sd = {t: statistics.pstdev([100 * (s["tiers"][t] - TIER_CHANCES[t]) / (1 - TIER_CHANCES[t])
                            for s in S if s["tiers"].get(t) is not None])
      for t in N}
tot = sum(sd.values())
print("实际   ", {t: round(W0[t] * 100, 1) for t in N})
print("按区分度", {t: round(100 * sd[t] / tot, 1) for t in N})
print("按题量  ", {t: round(100 * N[t] / 534, 1) for t in N})

print("\n=== 方差占比 ===")
shares = {t: [] for t in N}
for s in S:
    parts = {t: W0[t] ** 2 * s["tiers"][t] * (1 - s["tiers"][t]) / N[t] for t in N}
    tot_p = sum(parts.values())
    for t in N:
        shares[t].append(parts[t] / tot_p)
print({t: round(100 * statistics.mean(shares[t]), 1) for t in N})

print("\n=== 表(d) 换权重的影响 ===")
base_I = {s["key"]: I_of(s["tiers"], W0) for s in S}
rank_I0 = {k: i for i, (k, _) in enumerate(sorted(base_I.items(), key=lambda x: -x[1]))}
rows = [s for s in S if all(s["axes"].get(k) is not None
                            for k in ("intelligence", "calibration", "speed", "cost"))]


def geo(axes):
    return math.exp(sum(0.25 * math.log(max(axes[k] or 0, 1))
                        for k in ("intelligence", "calibration", "speed", "cost")))


def rank_of(order):
    return {s["key"]: i for i, s in enumerate(order)}


base_c = sorted(rows, key=lambda s: -geo({**s["axes"], "intelligence": I_of(s["tiers"], W0)}))
rc0 = rank_of(base_c)
for name, w in ALT.items():
    d = [abs(base_I[s["key"]] - I_of(s["tiers"], w)) for s in S]
    new_I = {s["key"]: I_of(s["tiers"], w) for s in S}
    rho_I = spearman(rank_I0, {k: i for i, (k, _) in enumerate(sorted(new_I.items(), key=lambda x: -x[1]))})
    order = sorted(rows, key=lambda s: -geo({**s["axes"], "intelligence": I_of(s["tiers"], w)}))
    rn = rank_of(order)
    print(f"{name:16s} ΔI 平均={statistics.mean(d):.2f} 最大={max(d):.2f} "
          f"| I轴 ρ={rho_I:.3f} | 综合 ρ={spearman(rc0, rn):.4f} "
          f"最大位次={max(abs(rc0[k] - rn[k]) for k in rc0)} "
          f"Top5同={set(s['key'] for s in base_c[:5]) == set(s['key'] for s in order[:5])}")

print("\n=== 表(e) 系统性倾斜 ===")
pr = [(I_of(s["tiers"], ALT["v1.5 10/20/30/40"]) - I_of(s["tiers"], W0), s["tiers"]["hard"]) for s in S]
mx, my = statistics.mean([p[0] for p in pr]), statistics.mean([p[1] for p in pr])
cov = sum((a - mx) * (b - my) for a, b in pr) / len(pr)
print(f"corr(ΔI, hard准确率) = "
      f"{cov / (statistics.pstdev([p[0] for p in pr]) * statistics.pstdev([p[1] for p in pr])):+.3f}")
```

**预期输出（v1.2 数据，2026-10 复算）**

```
=== 表(a) 各层区分度 ===
easy      n= 72 w=0.14 均值=0.929 CC均值= 90.1 SD未截断= 24.3 SD截断= 24.3 截到0=1/48
standard  n= 96 w=0.28 均值=0.799 CC均值= 70.6 SD未截断= 29.5 SD截断= 29.4 截到0=1/48
judge     n=146 w=0.28 均值=0.756 CC均值= 65.5 SD未截断= 32.0 SD截断= 29.7 截到0=3/48
hard      n=220 w=0.30 均值=0.540 CC均值= 30.7 SD未截断= 26.1 SD截断= 25.9 截到0=5/48

=== 表(b) 三种基准 ===
实际    {'easy': 14.0, 'standard': 28.0, 'judge': 28.0, 'hard': 30.0}
按区分度 {'easy': 21.7, 'standard': 26.4, 'judge': 28.6, 'hard': 23.3}
按题量   {'easy': 13.5, 'standard': 18.0, 'judge': 27.3, 'hard': 41.2}

=== 方差占比 ===
{'easy': 2.4, 'standard': 30.9, 'judge': 26.7, 'hard': 40.0}

=== 表(d) 换权重的影响 ===
等权 25/25/25/25   ΔI 平均=4.26 最大=9.01 | I轴 ρ=0.999 | 综合 ρ=0.9899 最大位次=5 Top5同=True
按题量比例         ΔI 平均=4.56 最大=8.39 | I轴 ρ=0.998 | 综合 ρ=0.9945 最大位次=5 Top5同=True
v1.5 10/20/30/40   ΔI 平均=4.84 最大=8.13 | I轴 ρ=0.999 | 综合 ρ=0.9923 最大位次=5 Top5同=True

=== 表(e) 系统性倾斜 ===
corr(ΔI, hard准确率) = +0.366
```

> 脚本依赖 `jevbench.composite_v13`（`TIER_CHANCES` 与 `chance_corrected_accuracy`）。若上游版本变更导致数字漂移，先核对 `composite_v13.py:12` 的选项数直方图与 `:11` 的层权重是否仍是本文所记的值。

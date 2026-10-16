# Jev benchmarks：外部基准与看板索引

本仓自己的走查只读代码与已发布的产物，**不跑上游程序、不出分数**（见 [`docs/comparison.md`](../docs/comparison.md) §5）。要查"某个模型在这类基准上考了多少分"，得去下面这些外部页面。

本文只做**链接索引**：每条给出页面标题与它自己的自述，不复制分数、不做跨页排名对比。任何要引用数字的场合，回到对应页面按它自己的口径核。

| # | 名称 | 链接 | 类型 |
|---|---|---|---|
| 1 | JevBench | <https://jevbench.github.io/> | 基准本体：一致性测试套件 + 排行榜 + 技术报告 + Python 包 |
| 2 | JevBench 看板（Benchmark Heaven） | <https://benchmarkheaven.com/jev-models> | 第三方看板，逐行标架构类 |
| 3 | Decision Model Leaderboard | <https://clef-evals.workers-ai-mle.workers.dev/> | 社区 Decision Index 排行榜（含 Cloudflare Clef） |
| 4 | Jev Decision Index（HF Space） | <https://huggingface.co/spaces/multimodalart/jev-decision-index> | 开源复现的逐模型看板 |
| 5 | Open-Jev benchmarks | <https://zefan-cai.github.io/open-jev/benchmarks/> | 上游自己的基准结果页（与本仓 [`docs/open-jev/`](../docs/open-jev/typed-decision-model.md) 同一上游） |
| 6 | Microsoft-Decision-1 | <https://commandline.microsoft.com/microsoft-decision-1-model-foundry/> | 厂商发布文（决策打分模型） |
| 7 | Decision Index 0.3（BenchLM.ai） | <https://benchlm.ai/benchmarks/decisionindex> | 第三方**镜像**看板：把第 4 条那份榜钉在一个 commit 上重排 |

---

## 1. JevBench —— 基准本体

<https://jevbench.github.io/>

页面标题：**JevBench: coherence tests for typed decision models**

它测的是**一致性（consistency），不是准确率、也不是速度**，所以**不需要 gold label**。做法是把请求里**问题那一侧**做变换（绝不动它所描述的那份输入），变换对答案的影响由概率论或选择理论**固定下来**，再看模型的答案偏离那条定律多少个概率单位——**0.05 以内算通过**。

50 条定律分五维，按"连接两个被比较答案的运算"划分：

| 维度 | 含义 |
|---|---|
| REP 表示 | 同一批事件换个说法（改写、翻译、重排或重命名选项）应得同一个答案 |
| BAT 批独立 | 一道题的答案不取决于同批问了哪些题、顺序、题数 |
| MEA 概率测度 | 答案可加：问题与它的否定和为 1；合并选项则概率合并 |
| LOG 逻辑序 | 合取不比它的任一部分更可能；更严的阈值不比更松的更可能 |
| CHO 选择集条件化 | 增加不相关选项不应改变原有选项之间的相对关系 |

入口：Leaderboard / Profiles / Findings / Models / Get started / Report / GitHub / Dataset；报告有 PDF；安装是 `pip install jevbench`；数据在 Hugging Face。

## 2. JevBench 看板 —— Benchmark Heaven

<https://benchmarkheaven.com/jev-models>

页面标题：**JevBench — AI decision model benchmark & leaderboard | Benchmark Heaven**

第三方维护的看板。本仓 [`docs/comparison.md`](../docs/comparison.md) §1.1 用的是它的 `/jev-models/all` 视图（该处记为 **board v1.7.46、2026-10-13 更新、共 192 行**）：看板给**每一行**标一个**架构类**，八类只表架构、不改分数与排名。

抓取时注意：该站要带代理（`--proxy 127.0.0.1:7890 -k`），且类别写在 tooltip 标题里而不是单元格文本里。

## 3. Decision Model Leaderboard

<https://clef-evals.workers-ai-mle.workers.dev/>

页面标题：**Decision Model Leaderboard**

页面自述：*Decision models ranked on the community Decision Index suite, including Cloudflare's Clef.* 社交卡片的 alt 写作 *Decision Index vs median latency for Clef, Clef-flash, Jev and selected open models* ——即这张榜把**质量（Decision Index）与延迟（median latency）**并置。

页面是个 SPA（`#root` 挂载、资源走 `/assets/index-*.js`），**内容不在 HTML 里**，直接抓 HTML 只能拿到 meta。

## 4. Jev Decision Index —— Hugging Face Space

<https://huggingface.co/spaces/multimodalart/jev-decision-index>

页面标题：**Jev Decision Index - a Hugging Face Space by multimodalart**

Space 自述：浏览一个给 **Jev 模型的开源复现**排名的看板，数据来源分三类——**公开基准、私有测试、新领域任务**；可以看逐模型的详细拆解并切换视图。

> 访问注意：本机直连 huggingface.co 会超时，走 Clash 代理 `127.0.0.1:7890`。

## 5. Open-Jev benchmarks —— 上游自己的结果页

<https://zefan-cai.github.io/open-jev/benchmarks/>

页面标题：**Open-Jev — Benchmark results and scope**
自述：*This index consolidates completed measurements from the published reports. Each table keeps its own metric and denominator. Tables label each checkpoint.*

它汇总的**不只是 JevBench**：JevBench & JF100、TREC 检索、延迟、对照与留出（controls & holdouts）、独立版本、算子效率、本地支持试验、全内部评测——**每组表各自保留自己的指标与分母**，并逐表标注是哪个 checkpoint。

这一条和本仓的口径最贴，值得单独留意两点：

- **它把「版本」摆到台面上**：页面明写 *Keep benchmark versions separate*。那 231 张公开表用的是**历史** JevBench commit `f8ce713`；v1.4 报告里 9B 的 29.9 %（sealed）对 77.5 %（public）**两边的难度构成不同**，页面自己声明 *This remains a historical record, not the current result*。另一份独立快照标的是 JevBench **v1.5.4**、观测于 2026-10-02 PDT，904 公开 + 720 sealed。
- **它就是本仓 [`docs/open-jev/`](../docs/open-jev/typed-decision-model.md) 那一篇的上游**（[Zefan-Cai/Open-Jev](https://github.com/Zefan-Cai/Open-Jev)）自己出的结果页——所以那条路线的走查与它这里的分数据出于同一方，**不是独立复现**。

## 6. Microsoft-Decision-1 —— 厂商发布文

<https://commandline.microsoft.com/microsoft-decision-1-model-foundry/>

页面标题：**Microsoft-Decision-1: Our model for fast decision-making**
正文标题：*Introducing Microsoft-Decision-1, our model for fast decision-making*
署名：Achint Srivastava（VP of Software Engineering, Office of the CTO, Microsoft），文内日期 **2026.10.09**
自述：*a decision-scoring model*，在结构化决策任务上以延迟与质量取胜，宣称同时优于 LLM 与其他决策模型。

注意 URL 里带 `model-foundry`，但正文大标题是 "Introducing Microsoft-Decision-1" ——**引用时以正文为准**。

## 7. Decision Index 0.3 —— BenchLM.ai 的镜像页

<https://benchlm.ai/benchmarks/decisionindex>

页面标题：**Decision Index 0.3 Full Scores | BenchLM.ai**
meta 自述：*Compare the 112 systems on Decision Index 0.3. The full score combines public and private task aggregates and stays separate from general model rankings.*
页内日期栏：**Last updated: October 6, 2026 · mirrored from the public benchmark leaderboard**

这一条与上面第 4 条**是同一份榜**：页内三条外链全指向那个 Hugging Face Space（榜单本体、`methodology.html`、Space README），数据源还**钉在一个 commit 上**（页内写作 *Pinned data source at commit f03e5a6*，链到 Space 该 revision 的 `data/` 目录）。所以它**不是独立测量**，是第三方镜像——引用时按这一层理解，别把它当成第二份证据。

它自己的口径说明（页内「How we show Decision Index 0.3」）：

- **合成权重**：37 个公开基准 20 %、**同技能的私有测试 50 %**、新领域私有任务 30 %（页面另给一个 `20% public · 80% private` 的摘要）。
- 源方报告 **110,201 条决策**；分数做过**机会校正（chance-corrected）与覆盖度校正（coverage-adjusted）**，未支持或未作答的请求**计为错**。
- 它自己声明镜像的边界：聚合值含私有任务结果，但**题目内容与逐题答案不公开**，那部分**无法从公开榜单独立复核**。
- 最后一句留给 BenchLM 自己：这一列**显示但不计入**它的加权总分（*excluded from the scoring formula*），所以不动它自家那套模型总排名。

抓取注意：与第 3 条相反，这一页**内容在 HTML 里**（服务端渲染），直接抓就拿到全表；页面另给一份 markdown 版（`<link rel="alternate" type="text/markdown" href="https://benchlm.ai/md/benchmarks/decisionindex.md">`）。本机直连可通，**不用挂代理**。

站点背景：BenchLM.ai 首页自述 *LLM Leaderboard & AI Model Benchmarks*，本页属其 Decision Models 类目。

---

## 核验记录

- 前六条链接于 **2026-10-14** 逐一请求，**均返回 200**；第 7 条于 **2026-10-16** 请求，返回 200。
- 各条标题、自述、日期均取自**页面自身**的 `<title>` / meta / 正文，不是转述。
- 本文**不含**任何分数、排名或跨页对比——那些一律回到上表对应页面按其口径核。

---

*本文件只做链接索引，不构成对任何基准口径的背书。*

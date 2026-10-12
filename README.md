# typed-decision-models

搜集 **jev 类模型 / 类型化决策模型(typed decision model)** 的实现方法。

这类模型的共同点是:不把 LLM 当生成器用,而是把同一份权重当**分类器**读——一次前向、只取读位置的 logits、掩码到固定标签集、重归一化。于是输出被封闭在一个已知标签集上,没有文本需要解析,也没有东西可以幻觉。代表场景是 JevBench 式的"给定一段 state,在若干选项中选一个"。

## 目录

| 实现 | 文档 | 内容 |
|---|---|---|
| RYOTIDE | [`docs/ryotide/typed-decision-model.md`](docs/ryotide/typed-decision-model.md) | 逐环节实现走查:15 节,从"读位置标定""掩码 + 重归一化""KV cache 分支策略"到"置信度从哪来、又怎么被度量",含实例走查、四条不变量与文件对照表 |
| Jeff | [`docs/jeff-serve/typed-decision-model.md`](docs/jeff-serve/typed-decision-model.md) | 逐环节实现走查:15 节,从"决策头 = 把「读」变成可训练参数"到"四条不变量在 Jeff 里的状态",含 choice / noul / 27 选项三条实例走查与文件对照表,全程与 RYOTIDE 逐节对照 |
| Open-Jev | [`docs/open-jev/typed-decision-model.md`](docs/open-jev/typed-decision-model.md) | 逐环节实现走查:12 节,从"Yes−No 播种决策头""pointwise 逐候选打分""两级 LCP 前缀缓存"到"fail-loud 风格",含 4 候选与 255 候选两条实例走查与温度还原 |
| Quyet | [`docs/quyet/typed-decision-model.md`](docs/quyet/typed-decision-model.md) | 逐环节实现走查:12 节,从"两条 runtime(encoder / LLM)""MASK 槽位与字母槽位""左填充的逐位精确性"到"四条不变量",每个环节与 RYOTIDE 对照 |
| Laya | [`docs/laya/typed-decision-model.md`](docs/laya/typed-decision-model.md) | 逐环节实现走查:14 节,从"`[MASK]` 槽位而不是单 token 标签""读 K 个槽位而不是最后一个位置""顺序偏置在训练时消解"到"RLCD 到底是什么",含路由器、fail-loud 守卫表与四条不变量的落位 |
| SemIf-OpenJev | [`docs/semif-openjev/typed-decision-model.md`](docs/semif-openjev/typed-decision-model.md) | 逐环节实现走查:18 节,从"字母槽位 A–P 的三重校验""固定末位、诊断量只上报不标定""共享 state 的三种 cache 复制策略"到"浏览器路径是唯一违反 `output_tokens = 0` 的一条",含 reranker 对照臂、逐 workload 温度链与四条不变量的落位 |
| OpenJev(HF) | [`docs/openjev-hf/typed-decision-model.md`](docs/openjev-hf/typed-decision-model.md) | 逐环节实现走查:18 节,从"52 个字母槽位与请求里的 `criteria`""两条读数协议(TARGETED 与 top-K 兜底)""超过 52 个选项的分块两级近似"到"代码默认值 ≠ 发布配方常数",含 agent state 压缩与截图通道、`/v1/version` 自证、四条不变量的落位 |
| vLLM Semantic Router | [`docs/semantic-router/typed-decision-model.md`](docs/semantic-router/typed-decision-model.md) | 逐环节实现走查:17 节,从"三代模型 × 三套读出协议(encoder 的 `[MASK]` 槽位 / decoder 的选项端点 / Vela 2.0 的八标记)""四档 profile:exact 目标逐字节相同"到"Go 侧不用模型报的 `confidence`,自己从 `probabilities` 重算",含加载期 golden 自检、committed golden answers 上的重算实例与 JevBench 闭环 |
| CLM | [`docs/clm/typed-decision-model.md`](docs/clm/typed-decision-model.md) | 逐环节实现走查:15 节,从"决策是状态嵌入与候选嵌入的一次点积,没有词表也没有掩码""读位置不在本仓:它由 vLLM 的池化默认值决定,而检查点不记录编码器""softmax 在 host 上用 Python 浮点算"到"整条链上没有标定,`temperature` 只是每请求的入参且不回声",含预留显存竞技场、两处会重复计 token 的路径、committed T-Rex 记录上的复算(与 Jev 同 harness 的 9.08 倍 p50、103 倍的每决策 token、0.658 对 0.987 的 planner 一致率) |
| Kev | [`docs/kev/typed-decision-model.md`](docs/kev/typed-decision-model.md) | 逐环节实现走查:15 节,从"用户声明的每个选项就是一段 `<opt>…</opt>` 跨度""骨干用 `.model` 加载,vocab head 被丢掉,词表维度从头到尾不存在""没有掩码也没有重归一化,softmax 直接落在头给出的 K 个标量上"到"顺序偏置训练期打散、运行期只度量而发布卡片不给数字",含发布检查点 77 个选项的读数、documents-v1 一条完整实例、四条不变量落位,以及三类不跑上游程序的「实测」(committed 行上重算公式、模型卡与发布清单 sha256 对照、发布清单与 SHA256SUMS 互证) |
| 跨实现对照 | [`docs/comparison.md`](docs/comparison.md) | 十路总表:主对照表、谁在改发布数值、四条不变量的跨篇落位、同一坑的各家修法、数字可比较性口径。只做索引与总表,论证仍在各篇 |

## 关于各篇文档

十篇都是从**只读代码**出发写的实现走查,不是复述 README。每篇的代码引用(`classify.py:42`、`head.py:160`、`api.py:48`、`questions.py:38`、`common.py:634` 等)均相对**其对应仓库**的根目录,本仓库只留文档副本,不含任何源码。

| 文档 | 上游代码库 | 来源 |
|---|---|---|
| `docs/ryotide/typed-decision-model.md` | [csabag/ryotide](https://github.com/csabag/ryotide) | 仓库 `docs/typed-decision-model.md`(分支 `docs`)的副本,上游仍在持续更新,同步时按此路径比对 |
| `docs/jeff-serve/typed-decision-model.md` | [j-geuter/jeff-serve](https://github.com/j-geuter/jeff-serve) | 本地未提交稿,收录即移入本仓库 |
| `docs/open-jev/typed-decision-model.md` | [Zefan-Cai/Open-Jev](https://github.com/Zefan-Cai/Open-Jev) | 本地未提交稿,收录即移入本仓库 |
| `docs/quyet/typed-decision-model.md` | [ncchinh/quyet](https://github.com/ncchinh/quyet) | 本地未提交稿,原名 `docs/implementation.md`,收录时移入本仓库并改名为主题名 |
| `docs/laya/typed-decision-model.md` | [NandaKishorM/laya](https://github.com/NandaKishorM/laya) | **非副本**:读 `laya/` 源码在本仓库直接写成,上游 `docs/` 下没有对应文章 |
| `docs/semif-openjev/typed-decision-model.md` | [TheoLeeCJ/SemIf-OpenJev](https://github.com/TheoLeeCJ/SemIf-OpenJev) | **非副本**:读 `src/semif_phase1/` 源码在本仓库直接写成,上游 `docs/` 下只有 METHOD/CALIBRATION/RESULTS 等项目文档,没有对应的走查稿。注意与上表的 `docs/open-jev/` 是两个不同仓库 |
| `docs/openjev-hf/typed-decision-model.md` | [openjev/openjev](https://huggingface.co/openjev/openjev) | **非副本**:读 `helper/` 与 `serve/` 源码在本仓库直接写成。上游是 **Hugging Face 上的模型仓**(权重 + 这两个代码目录),没有对应的走查稿。注意 `docs/open-jev/`、`docs/semif-openjev/` 与它是**三套不同代码** |
| `docs/semantic-router/typed-decision-model.md` | [vllm-project/semantic-router](https://github.com/vllm-project/semantic-router) | **非副本**:读 `src/model-runtime/vllm_srun/`、`src/semantic-router/pkg/` 与 `tools/calibration/` 源码在本仓库直接写成。上游 `website/docs/` 下只有面向用户的项目文档(含一份 `/v1/decisions` 的参考页),没有对应的走查稿。它是十篇里唯一一篇「运行时 + 控制面」而非单个读法的走查 |
| `docs/clm/typed-decision-model.md` | [Contrastive-LM/CLM](https://github.com/Contrastive-LM/CLM) | **非副本**:读 `src/clm/`、`train/`、`examples/t_rex/` 源码在本仓库直接写成。上游 `docs/` 下只有 `FINETUNING.md` 一份**给 agent 的自动调参实验说明**(不是项目文档,更不是走查稿)。它是十篇里唯一一条「决策 = 两个嵌入的点积」的路线,也是唯一一条自带 committed 跑分记录(T-Rex 实时游戏)让走查可以在其上重算的 |
| `docs/kev/typed-decision-model.md` | [jaredpalmer/kev](https://github.com/jaredpalmer/kev) | **非副本**:读 `kev/`(model / api / serve / predictors / metrics / train / data / checkpoint / anchors / calibrate / mlx_model)、`scripts/` 与 committed 的 `runs/`、`docs/model-cards/` 源码在本仓库直接写成。上游 `docs/` 下只有模型卡、发布说明与 `autoresearch.md` 等项目文档,没有对应的走查稿。它是十篇里唯一一条**读法在词表之外**的路线(骨干加载时丢掉 vocab head),也是唯一一条把「已发布数字 → committed 证据」做成机器校验清单(`docs/claims.json` + `scripts/verify_claims.py`)的 |

前四篇正文均未改动,只在文首补了代码库地址与「行号相对仓库根目录」的说明,并把指向 RYOTIDE 那篇的路径改写为本仓库内的相对路径。Laya、SemIf-OpenJev、OpenJev(HF)、Semantic Router、CLM 与 Kev 六篇同理在文首带代码库地址与行号基准说明,但它们没有上游原件可比对——改它们就是在改唯一的一份。这六篇「非副本」里,只有 OpenJev(HF) 的上游不是 Git 代码仓,`git clone` 得到的是权重仓(LFS 需跳过);Semantic Router 的上游是 Git 仓,但它是十篇里体量最大的一个:一个 Go 控制面 + 一个 Python 运行时 + 一套 JevBench 闭环评测,本文只走「类型化决策」这一条线。CLM 的上游同样没有走查稿(`docs/` 下只有一份给 agent 的自动调参说明),但它与另外九篇有一处不同:**它把跑分记录连同结果一起提交进了仓库**(`examples/t_rex/results/*.json`),所以那篇的实例走查是在 committed 产物上重算,而不是引上游文档里的数字。Kev 与它相邻:上游**没有**提交评测记录的那份完整转储,但提交了 `rows.json` / `report.json` 这类**读数摘要**与一份发布清单(`runs/release/kev-1.0-manifest.json` + `SHA256SUMS.txt`),所以 Kev 那篇的「实测」是在这些 committed 产物上重算公式、并对发布清单做 sha256 对照。

十条路线可以放在一起读,分歧集中在「读位置怎么定、置信度从哪来」:

- **RYOTIDE**:零训练,把冻结 LLM 读成分类器——运行期标定读位置、掩码 + 重归一化,`marker_mass` 是它的诊断量;
- **Jeff**:在 RYOTIDE 那条读法之上**训一个决策头**,`slice_init_` 让训练起点逐位等于零样本读法;
- **Open-Jev**:把决策拆成 N 个 pointwise 的是非题,再训一个 `Yes − No` 播种的标量头读成校准打分;
- **Quyet**:自训权重,encoder 用 `[MASK]` 槽位、LLM 用字母 `A..J`,温度随权重一起发布;
- **Laya**:双向编码器,把决策改造成**填空题**——每个选项一个 `[MASK]` 槽位,一次前向读 K 个标量;顺序偏置在训练时用 `parallel` 布局消解,另有一个不跑模型的**路由器**逐请求挑 checkpoint;
- **SemIf-OpenJev**:同样零训练,但把读法工程化到**三个运行时 + 一个浏览器**,并配一条独立的校准链——读末位的字母槽位 `A`–`P`,顺序偏置不消除只度量(反转选项顺序翻转 10/36 的 argmax),浏览器那条路径是十篇里唯一**会生成 1 个 token** 的;
- **OpenJev(HF)**:微调过的 27B + 一个挡在 vLLM 前面的 HTTP 补丁——读**第一个输出位置**上 52 个字母的 logits,训练把顺序偏置从 18.5% 压到 2.3%,而 T=0.85 / noul 的 (1.829074, 0) 是训练之后施加的常数,且**代码里的默认值与发布配方全都不同**;
- **vLLM Semantic Router**:不是一篇读法而是一个**运行时**——同一个 `/v1/decisions` 底下挂三代模型、三套读出协议(encoder 的 `[MASK]` 槽位 / decoder 的选项端点 / Vela 2.0 的八个标记 token)与四档数值 profile,`exact` 的目标是与发布包**逐字节相同**;Go 控制面拿到 typed answer 后**不用**模型报的 `confidence`,而是自己从 `probabilities` 重算一个,并拿 JevBench 标定自己的升级门;
- **CLM**:把决策整个换掉——不读 logits、没有词表,一次决策是**状态嵌入与候选嵌入的一次点积**;于是候选可以长期缓存、状态才需要现算,`marker_mass` 这类诊断量在这里没有对象,而**读位置被推给了编码器服务的池化默认值**(本仓不设定、不断言、不记录),整条链上也没有任何标定,`temperature` 只是每请求的入参且不回声。
- **Kev**:换掉的是「标签集从哪来」——**用户声明的每个选项就是一段 `<opt>…</opt>` 跨度**,骨干加载时直接丢掉 vocab head(`model.py:257`),所以既没有词表可以掩码、也没有「重归一化」这一步,softmax 落在头给出的恰好 K 个标量上;它付的代价是**必须训一个头**(`PointerHead` 的 q/k),换来的是候选集不受分词器约束、**读位置由 `encode` 在构造序列时算出来**因而运行期没有可标定的自由度;顺序偏置在训练期每 epoch 打散(外加一个构造性不变的开关 `option_isolation`,默认关且混合骨干不支持),运行期只度量、而**发布卡片不给翻转率**。

共同底线只有一条,也是 RYOTIDE 第 13 节那四条不变量的来源:**一次前向、只读最后一个位置、掩码到已知标签集、重归一化,`output_tokens = 0`**。违反时数字会看似合理却是错的——十条路线都在同一个主题上记录了各自的踩坑方式。两处例外都在「会生成 1 个 token」这一侧:SemIf-OpenJev 的浏览器路径用一次 1-token 的解码换一个能返回 logprobs 的接口,读的仍是被迫输出的那个 token 的 `top_logprobs`(其 §9);OpenJev(HF) 对 vLLM 发 `max_tokens: 1`,而响应里 `output_tokens: 0` 是**硬编码**的(其 §4)。Semantic Router 是这条底线上**最干净**的一条:`output_tokens` 在响应组装处硬编码为 0(其 §2.2),三套读法没有一条会生成 token。CLM 是另一头的极端:它连「一次前向」都不必改写——决策本身没有前向,`output_tokens: 0` 是字面量(其 §4),四个词里只剩「输出封闭在已知选项集上」还需要讲,而它靠的是**根本没有词表可越界**(其 §2.2)。Kev 是**字段层面唯一不成立**的一条:它确实不生成 token,但 `usage.output_tokens` 被定义成「序列化后答案 JSON 的 token 数」(`api.py:163-165`,README 第 284 行明说 *counts tokens in the serialized answers, not generated tokens*)——机制层面成立,字段层面不成立(其 §9.4);同时它也是第一条**连「掩码 + 重归一化」都不存在**的路线,所以那四个词在它身上只剩「一次前向」与「输出封闭在选项集上」。

并读才看得见的东西(`enable_thinking=False` 的诸种接法、填充方向、混合架构的 cache 分支复制、温度口径、ECE 分箱不一致、已校准与未校准是两份文件、**代码默认值 ≠ 发布配方**、**同一个字段名在不同代际是两个公式**、**同一个字段名在两层是两个量**)汇总在 [`docs/comparison.md`](docs/comparison.md)。

## 收录约定

新增实现或案例时,在 `docs/<实现名>/` 下建目录,一事一文,并在上表登记一行。同一实现的文档放在同一子目录里,文件名直接写主题(如 `typed-decision-model.md`、`calibration.md`),不再重复实现名。

跨实现的总表放 `docs/` 顶层(如 `docs/comparison.md`),不进任何单个实现的子目录——它不属于某一个实现,且这样各篇更新时顶层只需改表格格子。总表定位为**索引**,不复制各篇的论证,需要论证的地方链回原文。

正文取自上游的副本保持逐字不动,改动只限文首的代码库地址、行号基准说明和库内相对链接,以便按路径与上游比对。代码引用带文件名与行号,概率、准确率一类的数字标注来源 run 与 temperature。

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
| 跨实现对照 | [`docs/comparison.md`](docs/comparison.md) | 六路总表:主对照表、谁在改发布数值、四条不变量的跨篇落位、同一坑的各家修法、数字可比较性口径。只做索引与总表,论证仍在各篇 |

## 关于各篇文档

六篇都是从**只读代码**出发写的实现走查,不是复述 README。每篇的代码引用(`classify.py:42`、`head.py:160`、`api.py:48`、`questions.py:38`、`common.py:634` 等)均相对**其对应仓库**的根目录,本仓库只留文档副本,不含任何源码。

| 文档 | 上游代码库 | 来源 |
|---|---|---|
| `docs/ryotide/typed-decision-model.md` | [csabag/ryotide](https://github.com/csabag/ryotide) | 仓库 `docs/typed-decision-model.md`(分支 `docs`)的副本,上游仍在持续更新,同步时按此路径比对 |
| `docs/jeff-serve/typed-decision-model.md` | [j-geuter/jeff-serve](https://github.com/j-geuter/jeff-serve) | 本地未提交稿,收录即移入本仓库 |
| `docs/open-jev/typed-decision-model.md` | [Zefan-Cai/Open-Jev](https://github.com/Zefan-Cai/Open-Jev) | 本地未提交稿,收录即移入本仓库 |
| `docs/quyet/typed-decision-model.md` | [ncchinh/quyet](https://github.com/ncchinh/quyet) | 本地未提交稿,原名 `docs/implementation.md`,收录时移入本仓库并改名为主题名 |
| `docs/laya/typed-decision-model.md` | [NandaKishorM/laya](https://github.com/NandaKishorM/laya) | **非副本**:读 `laya/` 源码在本仓库直接写成,上游 `docs/` 下没有对应文章 |
| `docs/semif-openjev/typed-decision-model.md` | [TheoLeeCJ/SemIf-OpenJev](https://github.com/TheoLeeCJ/SemIf-OpenJev) | **非副本**:读 `src/semif_phase1/` 源码在本仓库直接写成,上游 `docs/` 下只有 METHOD/CALIBRATION/RESULTS 等项目文档,没有对应的走查稿。注意与上表的 `docs/open-jev/` 是两个不同仓库 |

前四篇正文均未改动,只在文首补了代码库地址与「行号相对仓库根目录」的说明,并把指向 RYOTIDE 那篇的路径改写为本仓库内的相对路径。Laya 与 SemIf-OpenJev 两篇同理在文首带代码库地址与行号基准说明,但它们没有上游原件可比对——改它们就是在改唯一的一份。

六条路线可以放在一起读,分歧集中在「读位置怎么定、置信度从哪来」:

- **RYOTIDE**:零训练,把冻结 LLM 读成分类器——运行期标定读位置、掩码 + 重归一化,`marker_mass` 是它的诊断量;
- **Jeff**:在 RYOTIDE 那条读法之上**训一个决策头**,`slice_init_` 让训练起点逐位等于零样本读法;
- **Open-Jev**:把决策拆成 N 个 pointwise 的是非题,再训一个 `Yes − No` 播种的标量头读成校准打分;
- **Quyet**:自训权重,encoder 用 `[MASK]` 槽位、LLM 用字母 `A..J`,温度随权重一起发布;
- **Laya**:双向编码器,把决策改造成**填空题**——每个选项一个 `[MASK]` 槽位,一次前向读 K 个标量;顺序偏置在训练时用 `parallel` 布局消解,另有一个不跑模型的**路由器**逐请求挑 checkpoint;
- **SemIf-OpenJev**:同样零训练,但把读法工程化到**三个运行时 + 一个浏览器**,并配一条独立的校准链——读末位的字母槽位 `A`–`P`,顺序偏置不消除只度量(反转选项顺序翻转 10/36 的 argmax),浏览器那条路径是六篇里唯一**会生成 1 个 token** 的。

共同底线只有一条,也是 RYOTIDE 第 13 节那四条不变量的来源:**一次前向、只读最后一个位置、掩码到已知标签集、重归一化,`output_tokens = 0`**。违反时数字会看似合理却是错的——六条路线都在同一个主题上记录了各自的踩坑方式。唯一的例外是 SemIf-OpenJev 的浏览器路径:它用一次 1-token 的解码换一个能返回 logprobs 的接口,读的仍是被迫输出的那个 token 的 `top_logprobs`(其 §9)。

并读才看得见的东西(`enable_thinking=False` 的诸种接法、填充方向、混合架构的 cache 分支复制、温度口径、ECE 分箱不一致、已校准与未校准是两份文件)汇总在 [`docs/comparison.md`](docs/comparison.md)。

## 收录约定

新增实现或案例时,在 `docs/<实现名>/` 下建目录,一事一文,并在上表登记一行。同一实现的文档放在同一子目录里,文件名直接写主题(如 `typed-decision-model.md`、`calibration.md`),不再重复实现名。

跨实现的总表放 `docs/` 顶层(如 `docs/comparison.md`),不进任何单个实现的子目录——它不属于某一个实现,且这样各篇更新时顶层只需改表格格子。总表定位为**索引**,不复制各篇的论证,需要论证的地方链回原文。

正文取自上游的副本保持逐字不动,改动只限文首的代码库地址、行号基准说明和库内相对链接,以便按路径与上游比对。代码引用带文件名与行号,概率、准确率一类的数字标注来源 run 与 temperature。

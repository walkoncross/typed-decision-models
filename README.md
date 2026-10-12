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

## 关于各篇文档

四篇都是从**只读代码**出发写的实现走查,不是复述 README。每篇的代码引用(`classify.py:42`、`head.py:160`、`api.py:48`、`questions.py:38` 等)均相对**其对应仓库**的根目录,本仓库只留文档副本,不含任何源码。

| 文档 | 上游代码库 | 副本来源 |
|---|---|---|
| `docs/ryotide/typed-decision-model.md` | [csabag/ryotide](https://github.com/csabag/ryotide) | 仓库 `docs/typed-decision-model.md`(分支 `docs`),上游仍在持续更新,同步时按此路径比对 |
| `docs/jeff-serve/typed-decision-model.md` | [j-geuter/jeff-serve](https://github.com/j-geuter/jeff-serve) | 本地未提交稿,收录即移入本仓库 |
| `docs/open-jev/typed-decision-model.md` | [Zefan-Cai/Open-Jev](https://github.com/Zefan-Cai/Open-Jev) | 本地未提交稿,收录即移入本仓库 |
| `docs/quyet/typed-decision-model.md` | [ncchinh/quyet](https://github.com/ncchinh/quyet) | 本地未提交稿,原名 `docs/implementation.md`,收录时移入本仓库并改名为主题名 |

四篇正文均未改动,只在文首补了代码库地址与「行号相对仓库根目录」的说明,并把指向 RYOTIDE 那篇的路径改写为本仓库内的相对路径。

四条路线可以放在一起读,分歧集中在「读位置怎么定、置信度从哪来」:

- **RYOTIDE**:零训练,把冻结 LLM 读成分类器——运行期标定读位置、掩码 + 重归一化,`marker_mass` 是它的诊断量;
- **Jeff**:在 RYOTIDE 那条读法之上**训一个决策头**,`slice_init_` 让训练起点逐位等于零样本读法;
- **Open-Jev**:把决策拆成 N 个 pointwise 的是非题,再训一个 `Yes − No` 播种的标量头读成校准打分;
- **Quyet**:自训权重,encoder 用 `[MASK]` 槽位、LLM 用字母 `A..J`,温度随权重一起发布。

共同底线只有一条,也是 RYOTIDE 第 13 节那四条不变量的来源:**一次前向、只读最后一个位置、掩码到已知标签集、重归一化,`output_tokens = 0`**。违反时数字会看似合理却是错的——四篇都在同一个主题上记录了各自的踩坑方式。

## 收录约定

新增实现或案例时,在 `docs/<实现名>/` 下建目录,一事一文,并在上表登记一行。同一实现的文档放在同一子目录里,文件名直接写主题(如 `typed-decision-model.md`、`calibration.md`),不再重复实现名。代码引用带文件名与行号,概率、准确率一类的数字标注来源 run 与 temperature。

# typed-decision-models

搜集 **jev 类模型 / 类型化决策模型(typed decision model)** 的实现方法。

这类模型的共同点是:不把 LLM 当生成器用,而是把同一份权重当**分类器**读——一次前向、只取读位置的 logits、掩码到固定标签集、重归一化。于是输出被封闭在一个已知标签集上,没有文本需要解析,也没有东西可以幻觉。代表场景是 JevBench 式的"给定一段 state,在若干选项中选一个"。

## 目录

| 实现 | 文档 | 内容 |
|---|---|---|
| RYOTIDE | [`docs/ryotide/typed-decision-model.md`](docs/ryotide/typed-decision-model.md) | 逐环节实现走查:15 节,从"读位置标定""掩码 + 重归一化""KV cache 分支策略"到"置信度从哪来、又怎么被度量",含实例走查、四条不变量与文件对照表 |

## 关于 `docs/ryotide/typed-decision-model.md`

该文是对 RYOTIDE(`github.com/csabag/ryotide`)实现的逐文件说明,按 `src/ryotide/` 的实际代码组织。其中的代码引用(`classify.py:42`、`jevbench_adapter.py`、`bench/fit_temperature.py` 等)均指 RYOTIDE 仓库内的路径,本仓库只留文档副本,不含其源码。副本取自 ryotide 仓库 `docs/typed-decision-model.md`(分支 `docs`),该文仍在原仓库持续更新,需要同步时按此路径比对。

核心立场浓缩成一句:同一个 KV cache 既能生成也能判类,区别只在**怎么读最后一个位置**。文中第 13 节列的四条不变量是踩坑总结——违反时数字会看似合理却是错的。

## 收录约定

新增实现或案例时,在 `docs/<实现名>/` 下建目录,一事一文,并在上表登记一行。同一实现的文档放在同一子目录里,文件名直接写主题(如 `typed-decision-model.md`、`calibration.md`),不再重复实现名。代码引用带文件名与行号,概率、准确率一类的数字标注来源 run 与 temperature。

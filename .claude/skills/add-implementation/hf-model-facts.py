#!/usr/bin/env python3
"""核对走查文档里的「模型描述」——把 Hugging Face 的 API 当成唯一事实来源。

用法::

    # 1) 从文档里抽候选仓名（只做正则，不联网）
    python3 hf-model-facts.py repos docs/laya/typed-decision-model.md

    # 2) 逐个问 Hub，打印 base_model / 参数量 / 逐文件体积 / sha
    python3 hf-model-facts.py facts Qwen/Qwen3.5-9B convaiinnovations/laya

**必须显式覆盖 endpoint 并走代理**：本机 shell 里常设着 `HF_ENDPOINT=https://hf-mirror.com`，
不覆盖就会去问镜像——镜像可能停在旧 revision，缺文件时直接抛 `LocalEntryNotFoundError`（看着像断网）。

    env HF_ENDPOINT=https://huggingface.co https_proxy=http://127.0.0.1:7890 \
        python3 hf-model-facts.py facts <repo> [<repo> ...]

`facts` 只读元数据，**不下载任何文件**（含权重）。要读某个小文件本身，用 hf CLI：

    env HF_ENDPOINT=https://huggingface.co https_proxy=http://127.0.0.1:7890 \
        hf download <repo> --include "*.json" --include "README.md" --local-dir /tmp/<repo>

抓下来的快照**不要提交进仓库**：revision 会变，进仓库就是一份会过期却看着权威的数据。
"""
from __future__ import annotations

import os
import re
import sys

MIRROR_HINTS = ("hf-mirror.com", "modelscope")
SKIP_EXT = (".py", ".md", ".json", ".jinja", ".txt", ".safetensors", ".sh", ".yaml", ".yml", ".pt",
            ".gz", ".bin", ".csv", ".jsonl", ".parquet", ".lock", ".toml", ".cfg", ".ini", ".cu", ".cpp")
# 明显是仓内路径而非「组织/模型名」的首段
SKIP_HEAD = {"docs", "src", "tests", "test", "examples", "example", "runs", "service", "registry",
             "data", "model", "models", "adapter", "adapters", "scripts", "tools", "benchmarks",
             "evaluation", "training", "website", "deploy", "requirements", "assets", "config",
             "git", "http", "https", "www", "huggingface", "github", "npm", "node_modules", "completion"}
TOKEN = re.compile(r"\b([A-Za-z0-9][A-Za-z0-9._-]{0,38})/([A-Za-z0-9][A-Za-z0-9._-]{0,60})\b")


def warn_if_mirror() -> None:
    ep = os.environ.get("HF_ENDPOINT", "")
    if any(h in ep for h in MIRROR_HINTS):
        print(f"!! HF_ENDPOINT={ep} 是镜像；核对 revision 前请覆盖成 https://huggingface.co",
              file=sys.stderr)


def cmd_repos(paths: list[str]) -> None:
    seen: dict[str, set[str]] = {}
    for p in paths:
        text = open(p, encoding="utf8").read()
        for m in TOKEN.finditer(text):
            org, name = m.group(1), m.group(2)
            if org in SKIP_HEAD or name.endswith(SKIP_EXT) or org.endswith(SKIP_EXT):
                continue
            # 组织名基本不以数字开头；这一步把 `2/3`、`0.8B/2B`、`413/503` 这类比值/分数滤掉
            if not re.match(r"^[A-Za-z]", org) or name.isdigit() or len(org) < 2:
                continue
            # 模型名几乎总带一个大写字母或数字；这一步滤掉 `yes/no`、`path/to`、`decider/data` 这类路径/词对
            if not re.search(r"[A-Z0-9]", name):
                continue
            seen.setdefault(f"{org}/{name}", set()).add(p)
    for repo in sorted(seen):
        print(f"{repo}\t{','.join(sorted(seen[repo]))}")
    print(f"# 候选 {len(seen)} 个——用 `facts` 逐个问 Hub，404 的说明是路径而不是仓名", file=sys.stderr)


def cmd_facts(repos: list[str]) -> None:
    warn_if_mirror()
    from huggingface_hub import HfApi

    api = HfApi()
    for repo in repos:
        try:
            info = api.model_info(repo, files_metadata=True)
        except Exception as e:                                    # noqa: BLE001
            print(f"{repo}\t未找到（{type(e).__name__}）——不是 Hub 上的仓名？")
            continue
        card = {}
        try:
            card = info.card_data.to_dict() if info.card_data else {}
        except Exception:                                          # noqa: BLE001
            pass
        st = getattr(info, "safetensors", None) or {}
        sib = sorted(((s.rfilename, s.size or 0) for s in (info.siblings or [])), key=lambda t: -t[1])
        weights = sum(sz for f, sz in sib if f.endswith((".safetensors", ".bin", ".gguf", ".pt")))
        print(f"## {info.id}")
        print(f"   sha={info.sha}  lastModified={getattr(info, 'last_modified', None)}")
        print(f"   base_model={card.get('base_model')}  relation={card.get('base_model_relation')}  "
              f"license={card.get('license')}  library={getattr(info, 'library_name', None)}")
        params = st.get("total") if isinstance(st, dict) else None
        print(f"   params_total={params}  权重文件合计={weights} B = {weights / 1e9:.2f} GB "
              f"({weights / 2**30:.2f} GiB)")
        cfg = info.config or {}
        keep = {k: cfg[k] for k in ("architectures", "model_type", "hidden_size", "num_hidden_layers",
                                    "vocab_size", "max_position_embeddings") if k in cfg}
        if keep:
            print(f"   config: {keep}")
        for f, sz in sib[:6]:
            print(f"   {sz:>14,} B  {f}")


def main() -> int:
    if len(sys.argv) < 3:
        print(__doc__)
        return 2
    mode, rest = sys.argv[1], sys.argv[2:]
    if mode == "repos":
        cmd_repos(rest)
    elif mode == "facts":
        cmd_facts(rest)
    else:
        print(__doc__)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

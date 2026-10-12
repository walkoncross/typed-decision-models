#!/usr/bin/env bash
# 收录规范校验。在仓库任意位置执行：
#   .claude/skills/add-implementation/check.sh
# 退出码 0 = 全部通过；1 = 有 FAIL。
set -uo pipefail

ROOT=$(git rev-parse --show-toplevel 2>/dev/null || pwd)
cd "$ROOT" || exit 2

fail=0
ok()  { printf '  ok    %s\n' "$1"; }
bad() { printf '  FAIL  %s\n' "$1"; fail=1; }

# 收录在 docs/<实现名>/ 下的文档（comparison.md 等跨实现文档在顶层，不参与）。
impl_docs() { find docs -mindepth 2 -name '*.md' | sort; }

echo "== 1. 编码（应为 UTF-8）=="
while IFS= read -r f; do
  enc=$(file -I "$f" | sed -n 's/.*charset=//p')
  if [ "$enc" = "utf-8" ]; then ok "$f"; else bad "$f 编码=${enc}（应为 utf-8）"; fi
done < <(find docs -name '*.md' | sort)

echo "== 2. 目录结构（每个实现子目录下至少一篇）=="
for d in docs/*/; do
  n=$(find "$d" -maxdepth 1 -name '*.md' | wc -l | tr -d ' ')
  if [ "$n" -ge 1 ]; then ok "${d}（${n} 篇）"; else bad "${d} 下没有 .md"; fi
done

echo "== 3. 文件名（写主题名，不重复实现名，不用泛名）=="
while IFS= read -r f; do
  base=$(basename "$f"); impl=$(basename "$(dirname "$f")")
  if printf '%s' "$base" | grep -qiE '^(implementation|impl|index|doc|docs|document|readme|notes|misc)\.md$'; then
    bad "$f 文件名过于泛化（应写主题名，如 typed-decision-model.md）"
  elif printf '%s' "$base" | grep -qi -- "$impl"; then
    bad "$f 文件名重复了实现名（${impl}）"
  else
    ok "$f"
  fi
done < <(impl_docs)

echo "== 4. 文首两行（代码库地址 + 行号基准说明）=="
while IFS= read -r f; do
  head20=$(head -20 "$f")
  miss=""
  printf '%s' "$head20" | grep -q '^> 代码库：<http' || miss="缺「> 代码库：<URL>」"
  printf '%s' "$head20" | grep -q '均相对该仓库根目录' || miss="$miss 缺行号基准说明"
  if [ -z "$miss" ]; then ok "$f"; else bad "$f $miss"; fi
done < <(impl_docs)

echo "== 5. 正文不含本机绝对路径 =="
# 只查真正的用户目录绝对路径。文中的 /path/to/xxx、/opt/... 是示例占位，不算问题；
# 不查 Windows 盘符形式——它会把 Python 字符串里的 "Answer:\n" 这类误判成路径。
if hits=$(grep -rnE '/Users/|/home/' docs --include='*.md'); then
  bad "发现绝对路径："
  printf '%s\n' "$hits" | sed 's/^/        /'
else
  ok "无绝对路径"
fi

echo "== 6. 相对链接可解析 =="
# 只查收录文档（docs/ 与 README）：skill 正文里的模板路径是占位符，不是待解析链接。
broken=$(while IFS= read -r f; do
  d=$(dirname "$f")
  grep -oE '\]\([^)]*\.md[^)]*\)' "$f" 2>/dev/null \
    | sed -E 's/^\]\(([^)#]*\.md).*/\1/' \
    | while IFS= read -r l; do
        # 跳过空串、外部链接、以 / 开头的站点绝对路径
        if [ -n "$l" ] && [ "${l#http}" = "$l" ] && [ "${l#/}" = "$l" ]; then
          [ -f "$d/$l" ] || echo "$f -> $l"
        fi
      done
done < <(find docs -name '*.md'; echo README.md))
if [ -z "$broken" ]; then ok "全部可解析"; else bad "断链："; printf '%s\n' "$broken" | sed 's/^/        /'; fi

echo "== 7. README 目录表已登记 =="
while IFS= read -r f; do
  if grep -qF "$f" README.md; then ok "$f"; else bad "README.md 未登记 $f"; fi
done < <(impl_docs)

echo "== 8. comparison.md 已覆盖每个实现 =="
for d in docs/*/; do
  impl=$(basename "$d")
  if grep -qF "$impl/" docs/comparison.md; then ok "$impl"; else bad "comparison.md 未覆盖 $impl"; fi
done

echo "== 9. 代码库地址与 README 一致 =="
while IFS= read -r f; do
  url=$(grep -m1 -oE '^> 代码库：<[^>]*>' "$f" | sed -E 's/.*<([^>]*)>/\1/')
  if [ -z "$url" ]; then bad "$f 取不到 URL"; continue; fi
  rurl=$(grep -F "$f" README.md | grep -oE 'https://github\.com/[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+' | head -1)
  if [ "$url" = "$rurl" ]; then ok "$f  $url"; else bad "$f 文首=$url  README=$rurl"; fi
done < <(impl_docs)

echo
if [ "$fail" -eq 0 ]; then echo "全部通过。"; else echo "有 FAIL，见上。"; fi
exit "$fail"

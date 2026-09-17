#!/usr/bin/env bash
set -eo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

failed() {
  local result=$?
  if (( result != 0 )); then
    printf '\n启动未完成，请查看上面的错误信息。\n'
    if [[ -t 0 ]]; then read -r -p '按回车关闭此终端……' _reply || true; fi
  fi
  exit "$result"
}
trap failed EXIT

if [[ "${1:-}" == '--help' ]]; then
  printf '用法：./start_104_and_view.sh [--check]\n默认连接 104 启动/复用建图，再打开 237 显示；--check 只检查 104 状态。\n'
  exit 0
fi
if [[ $# -gt 1 || ( $# -eq 1 && "$1" != '--check' ) ]]; then
  printf '参数不支持；使用 --help 查看用法。\n' >&2
  exit 2
fi
remote_action=mapping
if [[ "${1:-}" == '--check' ]]; then remote_action=check; fi
"$ROOT/run_remote104.sh" "$remote_action"
if [[ "${1:-}" == '--check' ]]; then exit 0; fi

if /usr/bin/python3 - <<'PY'
import json
import viewer
state = json.loads(viewer.STATE.read_text()) if viewer.STATE.exists() else {}
children = state.get('children', {})
raise SystemExit(0 if viewer.alive(state) and children and all(viewer.alive(p) for p in children.values()) else 1)
PY
then
  printf '237 的 104 显示窗口已在运行，请切换到标题含“104 车机”的窗口。\n'
  exit 0
fi
printf '打开 237 显示窗口。关闭窗口或按 Ctrl+C 只停止显示，104 建图继续运行。\n'
./start.sh

#!/usr/bin/env bash
set -o pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
case "${1:-}" in
  '') action=control ;;
  --check) action=control-check ;;
  --help) printf '用法：start_104_control.sh [--check]\n独立打开 104 实车手动控制窗口；--check 只检查底盘状态。\n'; exit 0 ;;
  *) printf '参数不支持，使用 --help 查看用法。\n' >&2; exit 2 ;;
esac
export DISPLAY="${DISPLAY:-:0}"
if [[ -z "${XAUTHORITY:-}" && -f "/run/user/$(id -u)/gdm/Xauthority" ]]; then
  export XAUTHORITY="/run/user/$(id -u)/gdm/Xauthority"
fi
"$ROOT/run_remote104.sh" "$action"
result=$?
if (( result != 0 )); then
  printf '\n手动控制入口未完成，请查看上面的错误。\n'
  if [[ -t 0 ]]; then read -r -p '按回车关闭终端……' _reply || true; fi
fi
exit "$result"

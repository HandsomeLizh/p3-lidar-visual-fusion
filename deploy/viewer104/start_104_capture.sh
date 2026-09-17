#!/usr/bin/env bash
set -o pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
"$ROOT/run_remote104.sh" capture
result=$?
if (( result == 0 )); then
  printf '\n104 采集已启动/复用。接下来单独打开“02 启动104建图并显示”。\n'
else
  printf '\n采集启动失败，请查看上方错误。\n'
fi
if [[ -t 0 ]]; then read -r -p '按回车关闭此终端（已启动的采集继续运行）……' _reply || true; fi
exit "$result"

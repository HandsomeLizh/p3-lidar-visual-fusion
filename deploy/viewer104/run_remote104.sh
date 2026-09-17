#!/usr/bin/env bash
set -eo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
case "${1:-}" in
  capture) remote_command='bash /home/yanfa/P3/lidar_visual_fusion/hardware_sensors.sh start' ;;
  mapping) remote_command='bash /home/yanfa/P3/lidar_visual_fusion/start_for_237.sh' ;;
  check) remote_command='bash /home/yanfa/P3/lidar_visual_fusion/start_for_237.sh --check' ;;
  control) remote_command='bash /home/yanfa/P3/lidar_visual_fusion/start_manual_control.sh' ;;
  control-check) remote_command='bash /home/yanfa/P3/lidar_visual_fusion/start_manual_control.sh --check' ;;
  *) printf '用法：run_remote104.sh capture|mapping|check|control|control-check\n' >&2; exit 2 ;;
esac
display_options=()
if [[ "$1" == control ]]; then display_options=(-Y); fi
key="${P3_104_SSH_KEY:-$HOME/.ssh/p3_104_ed25519}"
if [[ ! -r "$key" ]]; then
  printf '未找到 237→104 专用 SSH 密钥：%s。请按桌面文档恢复免密配置。\n' "$key" >&2
  exit 1
fi
printf '使用专用 SSH 密钥连接 104（192.168.100.104），无需输入密码。\n'
exec ssh -t "${display_options[@]}" -o ConnectTimeout=8 -o ServerAliveInterval=10 -o ServerAliveCountMax=3 \
  -i "$key" -o IdentitiesOnly=yes -o BatchMode=yes \
  -o ForwardAgent=no -o StrictHostKeyChecking=yes \
  -o UserKnownHostsFile="$ROOT/config/ssh_known_hosts104" \
  yanfa@192.168.100.104 "$remote_command"

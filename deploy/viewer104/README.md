# 237 上的 104 专用窗口

目录：`/home/yanfa/P3/viewer104`。104 接真实传感器并建图；本目录接收和显示。237 自己的算法使用原入口和另一 ROS 域，拥有自己的窗口。

## 三个独立入口

在 237 桌面终端执行，每个入口单独使用一个终端：

```bash
cd /home/yanfa/P3/viewer104
./start_104_capture.sh       # 启动/复用 104 采集
./start_104_and_view.sh      # 采集就绪后，启动/复用 104 建图并打开专用窗口
./start_104_control.sh       # 需要手动驾驶时，打开 104 原底盘控制面板
```

237→104 已配置专用 SSH 密钥，无需输入 SSH 密码。桌面三个编号图标执行相同操作。详细操作和手动控制说明见桌面的 `237_启动104建图与显示.md`。

104 已经在建图时，只打开或关闭显示：

```bash
./start.sh
./status.sh
./stop.sh
```

窗口标题为“104 车机 · 定位与建图 · ROS 域 59”。关闭它只停止本目录的显示进程。237 自己的窗口、104 建图和底盘分别管理。

## 话题隔离

- 固定接收 104 的域 59。237 自己的算法应使用其他域；本次核查使用 75。
- 内部显示点云 `/viewer104_237/rviz_cloud`、标记 `/viewer104_237/markers`，节点位于 `/viewer104_237`；RViz 配置、运行目录、锁和进程记录都在本目录。
- 104 当前三维与高程建图只用 LiDAR，点云为 `/T3/mapping/lidar_map`；双目补点及 `/T3/mapping/stereo_map` 发布已关闭。定位仍用视觉、LiDAR、真实 IMU。局部/全局高程分别为 `/Car/T3/mapping/grid_map`、`/Car/T3/mapping/global_grid_map`。外发局部图 64×64 米，右侧跟车视野 32×32 米。
- 专用 DDS 配置采用 1400 字节 UDP 包、1280 字节 DDS 分片，接收缓冲请求 10 MiB。237 的 `net.core.rmem_max` 已配置为 10485760；系统持久设置在 `/etc/sysctl.d/90-p3-viewer104-receive.conf`。
- RViz 点大小为 1.5 像素，保存在 `config/p3_visual_window.rviz`。104 建图程序也使用专用小包传输配置，避免相机数据分片重传影响地图接收。
- 专用窗口从 `/viewer104_transport/*` 接收无损压缩的显示栅格，以及最高 2 Hz、320×240 的 JPEG 相机预览。104 原始定位输入、规划端 64 m GridMap 和高程数值均不改变。`start_for_237.sh` 会为已有建图补启动这一路显示传输，无需重建地图。
- 地图短暂断流时，画面最多保留 30 秒，并标注“地图更新中断”；这段时间暂停目标发送。恢复后自动使用新图。地图下方显示已知格数量和最近接收时间，运行目录 `runtime/grid_display_status.json` 记录接收计数。
- 操作者设置的 P4 目标点发到域 59 的 `/Car/T4/rviz_goal`，只用于 104 配套规划。
- 启动入口会覆盖继承的其他项目 ROS 域、网络配置和显示运行目录。

## 没有地图时

确认 104 正在建图，两台网络互通，然后在显示运行时执行：

```bash
source /home/yanfa/P3/viewer104/env.sh
python3 /home/yanfa/P3/viewer104/probe.py
```

探针观察 20 秒，只订阅；结果写入 `results/receive_check.json`。窗口日志在 `results/日期_时间/`。本机联网网卡当前为 `wlP1p1s0`，更换时更新本目录 `config/cyclonedds.xml`。仅复用 237 的 ROS/Qt/消息依赖，编译专用窗口用 `./build.sh`。

## SSH 登录

在 237 的 `yanfa` 终端用 `ssh p3-104` 或 `ssh yanfa@192.168.100.104` 免密登录。私钥位于 `/home/yanfa/.ssh/p3_104_ed25519`，只留在 237，不随项目分发。104 的公钥授权限定来源 `192.168.100.237`。三入口都使用该密钥，保留主机身份校验；不修改远程桌面的密码。

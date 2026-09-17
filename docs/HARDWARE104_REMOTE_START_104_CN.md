# 104 建图：本机与 237 远程启动

104 地址：`yanfa@192.168.100.104`。
项目目录：`/home/yanfa/P3/lidar_visual_fusion`。
当前三维点云和高程图仅使用 LiDAR，双目三维补点已关闭；定位继续使用视觉、LiDAR 和真实 IMU。发给规划端的局部 GridMap 为以车为中心的 **64×64 米、320×320 格**；右侧显示独立跟车查看 **32×32 米**。

## 推荐：在 237 操作

两台开机并处于可互通的局域网。在 237 桌面先双击 **“01 启动104传感器采集”**，成功后再双击 **“02 启动104建图并显示”**。237→104 已配置专用 SSH 密钥，无需输入 SSH 密码。

刚上电时雷达需要初始化，建议等约 20 秒再启动建图；若提示传感器未就绪，稍后重新执行建图入口。当前三维点大小为 1.5 像素，已保存在配置中。

也可以在 **237 的桌面终端**执行：

```bash
cd /home/yanfa/P3/viewer104
./start_104_capture.sh       # 单独启动采集
# 采集成功后再执行：
./start_104_and_view.sh      # 启动/复用建图与显示
```

需要手动驾驶时，在 237 单独双击“03 打开104手动底盘控制”，或另开终端执行 `/home/yanfa/P3/viewer104/start_104_control.sh`。它通过免密 SSH 显示原控制面板。采集、建图、手动控制是三个独立入口；104 已经在建图时复用现有地图，不重新建立原点。专用私钥只保存在 237 的 `.ssh/p3_104_ed25519`，104 安装对应公钥。

## 在 104 本机启动，界面放在 237

终端 A：只启动/复用双目、LiDAR、IMU 与相机触发。这里只向 ROS 发布，不自动录 bag 或保存原始图片。

```bash
cd /home/yanfa/P3/lidar_visual_fusion
./hardware_sensors.sh start
```

终端 B：只启动/复用建图。

```bash
cd /home/yanfa/P3/lidar_visual_fusion
./start_for_237.sh
```

此入口检查实际传感器数据；采集未就绪时提示错误，不自动启动采集。随后在 237 执行：

```bash
cd /home/yanfa/P3/viewer104
./start.sh
```

## 在 104 本机启动并显示

只在当前没有本工程运行时使用：

```bash
cd /home/yanfa/P3/lidar_visual_fusion
./hardware_sensors.sh start
P3_HARDWARE_DOMAIN=59 ./start_hardware.sh --rviz
```

如果提示已有运行，先用 `./status.sh` 检查。确实需要新建地图时，再执行 `./stop.sh` 后重新启动。

## 状态和停止

在 104 的终端，或从 237 通过 `ssh yanfa@192.168.100.104` 登录后：

```bash
cd /home/yanfa/P3/lidar_visual_fusion
./status.sh
./stop.sh
```

`./status.sh` 只查看；`./stop.sh` 停止这套建图及它启动的窗口，传感器驱动保留。需要只关闭 237 显示，执行 237 上的 `/home/yanfa/P3/viewer104/stop.sh`。

## 独立手动／底盘控制

在 **104 图形桌面终端**执行：

```bash
cd /home/yanfa/P3/lidar_visual_fusion
./start_manual_control.sh          # 打开原 Mars_Car2 控制面板
# 只看进程状态，不启动硬件或窗口：
./start_manual_control.sh --check
```

从 237 使用专用 `start_104_control.sh`，窗口会通过 SSH 显示在 237。不要在没有图形转发的普通 SSH 终端直接打开 GUI。

- 底盘驱动：`/home/yanfa/program/Mars_Car2`；原控制面板：`/home/yanfa/program/UI/src/mars_car_gui.py`。
- 复用已有底盘驱动；首次启动驱动沿用其电机上电、上位机控制和编码器初始化流程。
- 初始不发送行驶指令；先点击所用区域的“停止”并确认车辆停稳，再关闭面板。关闭面板或 SSH 断线不等于停车。
- 不自动启动 P4。采集、建图的启停不会启停底盘。
- 实际核验：域 19 的 `/cmd_vel` 由 `Car_node` 订阅；`/Car/T5/chassis_detail` 有约 20 Hz 反馈。未执行行驶指令验证。

## 地图与记录

- ROS_DOMAIN_ID：59。
- LiDAR 累积点云：`/T3/mapping/lidar_map`。双目补点已关闭，`/T3/mapping/stereo_map` 当前不发布。
- 局部高程图：`/Car/T3/mapping/grid_map`，车周围 64×64 米，约每秒更新；右侧显示只看 32×32 米。
- 全局高程图：`/Car/T3/mapping/global_grid_map`，约每 2 秒更新。
- 正式定位：`/T3/semantic/current_pose`。
- 当前运行目录可用 `./status.sh` 查看；日志、地图、统计保存在对应 `results/run_日期_时间/`。

**每次重新启动建图都会创建新地图原点，P4 使用前需要重新对齐。** 只重开 237 显示不会改变 104 地图原点。启动脚本不会自动驾驶。

237 专用窗口仍接收域 59 的上述地图和位姿，但内部 RViz 点云、标记已隔离到 `/viewer104_237/*`。104 本机显示可以同时开启。237 自己的算法应使用其他 ROS 域；专用窗口的停止脚本只管理 `viewer104` 记录的进程。

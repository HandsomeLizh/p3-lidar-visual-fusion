# 104 建图：本机与 237 远程启动

104 地址：`yanfa@192.168.100.104`。
项目目录：`/home/yanfa/P3/lidar_visual_fusion`。
新版地图仅使用双目三维点累积点云与高程；定位使用视觉、LiDAR 和真实 IMU。

## 推荐：在 237 操作

两台开机并处于可互通的局域网。在 237 桌面先双击 **“01 启动104传感器采集”**，成功后再双击 **“02 启动104建图并显示”**，按提示输入 104 的 SSH 密码。

也可以在 **237 的桌面终端**执行：

```bash
cd /home/yanfa/P3/viewer104
./start_104_capture.sh       # 单独启动采集
# 采集成功后再执行：
./start_104_and_view.sh      # 启动/复用建图与显示
```

采集与建图是两个独立入口；机器人控制使用现有独立控制程序，不会被它们自动启动。104 已经在建图时复用现有地图，不重新建立原点。密码不写入脚本。

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

## 地图与记录

- ROS_DOMAIN_ID：59。
- 双目累积点云：`/T3/mapping/stereo_map`。
- 局部高程图：`/Car/T3/mapping/grid_map`，车周围 32×32 米，约每秒更新。
- 全局高程图：`/Car/T3/mapping/global_grid_map`，约每 2 秒更新。
- 正式定位：`/T3/semantic/current_pose`。
- 当前运行目录可用 `./status.sh` 查看；日志、地图、统计保存在对应 `results/run_日期_时间/`。

**每次重新启动建图都会创建新地图原点，P4 使用前需要重新对齐。** 只重开 237 显示不会改变 104 地图原点。启动脚本不会自动驾驶。

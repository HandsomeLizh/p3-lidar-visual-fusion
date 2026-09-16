# LiDAR＋视觉建图：简明使用说明

默认使用 **VoxelMap＋XFeat/LighterGlue**，当前不需要 IMU。已编译，可直接启动。

## 1. 连接远程

在 Windows 终端执行：

```bash
ssh yanfa@192.168.100.216
cd /home/yanfa/P3/lidar_visual_fusion
```

以下命令均在这个远程目录执行。RViz 窗口显示在远程机器桌面上。

## 2. 历史 bag 建图

**先推荐运行 short bag：**

```bash
./start_short_bag.sh fusion
```

自动启动视觉＋LiDAR 融合、高程图、点云图和原 P3 的 RViz 界面。

其他选择（任选一个，切换前先执行 `./stop.sh`）：

```bash
# 只用 LiDAR，回放同一个 short bag
./start_short_bag.sh lidar

# 20260824_235238 的前约 10 分钟，视觉＋LiDAR
./start_long_bag.sh
```

保持默认原速回放即可。每次结果目录会在启动终端打印，也可通过 `./status.sh` 查看。

## 3. 仿真车边走边建图

**UE 已启动后，开两个 SSH 终端。两个脚本分别管理自己的模块，车辆由人控制。**

### 终端 A：一键启动采集、车辆控制和控制窗口

```bash
ssh yanfa@192.168.100.216
cd /home/yanfa/P3/lidar_visual_fusion
./start_simulation_sources.sh
```

这个脚本统一设置 **ROS_DOMAIN_ID=10**，连接 UE 的 `192.168.10.22:6665`（采集）和 `:6668`（车辆控制）。它会启动缺少的节点、复用配置一致的已有节点，并在远程桌面打开“月球车控制 GUI”。

启动脚本不选择驾驶模式、不设置非零速度；由人操作 GUI。界面沿用原 P3 的速度滑条、模式按钮和紧急停车按钮；关闭 GUI 时仍执行原界面的停车逻辑。GUI 左侧地图区需要它自己的 `/map`，融合高程图和点云请看 RViz。

- 新启动了节点时，保持终端 A 打开。日志路径会在终端打印。
- 若提示旧节点处于**域 0**等配置冲突，请在原启动终端按 **Ctrl+C** 结束提示的旧节点，再运行本脚本；脚本不会擅自停止旧进程。
- 仅检查节点是否已启动、配置是否一致：`./start_simulation_sources.sh --check`。
- 新启动 capture 时默认请求上限 2 Hz、额外等待 0 秒；这不保证 UE 能以 2 Hz 完成传输。已有 capture 会被复用，其参数不会被偷偷改写。
- 已构建的接收端优化版会自动使用，日常命令不变；新机器的构建方法见 [环境说明](SETUP_CN.md)。

### 终端 B：启动自己的建图和 RViz

```bash
ssh yanfa@192.168.100.216
cd /home/yanfa/P3/lidar_visual_fusion
./start_live.sh
```

这条命令启动 **实时数据转接、视觉＋LiDAR 建图和 RViz**，持续运行到手动停止。车辆使用原来的人工控制界面或控制程序驾驶。

- 建图程序不发送行驶、模式切换或停车指令，也不申请车辆控制权。
- 实时源域为 10，建图域为 57；数据转接自动启动。
- 激光独立转接，视觉缺帧不会阻塞激光输入。
- 结果保存在 `results/live_mapping_日期_时间_编号/`。
- 默认不录制原始数据。如需留包，使用 `./start_live.sh --record-input`，注意磁盘空间。
- 旧命令 `./test_live_simulation.sh` 也已改为同一人工驾驶入口，不再自动行驶。

结束时：**先由人停车，再在终端 B 按 Ctrl+C**，等待保存地图并关闭建图、RViz。终端 A 的采集和车辆控制继续运行；需要关闭它们时，再在终端 A 按 Ctrl+C，只有本次新启动的节点会关闭。

## 4. 查看状态、保存和停止

bag 回放运行期间：

```bash
./status.sh
./save_map.sh
./stop.sh
```

停止后导出地图，将下面的 `results/你的结果目录` 换成终端显示的本次目录：

```bash
./export_map.sh results/你的结果目录
```

人工驾驶实时建图按 Ctrl+C 结束后已经自动保存、停止本次建图，直接执行导出即可。

| 文件 | 内容 |
|---|---|
| `lidar_map.pcd` | 导出的 LiDAR 点云地图 |
| `global_grid_map.npz` | 导出的全局高程栅格；地图过大时使用分块存储 |
| `global_grid_map.sqlite3` | 持久化栅格地图 |
| `trajectory_map.tum` | 估计轨迹 |
| `verification.json` | 位姿、TF、地图发布等接口检查 |
| `live_session_status.json` | 实时接收、转发数量和建图状态 |
| `resources_latest.json` | 当前内存、磁盘等资源统计 |

`verification.json` 需要启动时加 `--verify`；普通运行不收集完整验证记录，减少额外占用。

## 5. 给规划端的接口

**ROS_DOMAIN_ID：57；地图坐标系：`map`；距离单位：米。**

| 话题 | 内容／类型 |
|---|---|
| `/T3/semantic/current_pose` | 当前位姿，`nav_msgs/msg/Odometry` |
| `/T3/semantic/trajectory` | 轨迹，`nav_msgs/msg/Path` |
| `/T3/semantic/incremental_map` | 增量地图，`t3_interfaces/msg/IncrementalSemanticMap` |
| `/T3/mapping/elevation_map` | 高程图，`grid_map_msgs/msg/GridMap` |
| `/T3/mapping/lidar_map` | 点云图，`sensor_msgs/msg/PointCloud2` |
| `/Car/T3/mapping/grid_map` | 跟随车辆的局部栅格，`grid_map_msgs/msg/GridMap`，`odom` 坐标系 |
| `/Car/T3/mapping/global_grid_map` | RViz 使用的累计全局高程栅格，`grid_map_msgs/msg/GridMap`，`map` 坐标系 |

跨机器联调时，bag 启动命令末尾加 `--network`；规划端设置 `ROS_DOMAIN_ID=57`、`ROS_LOCALHOST_ONLY=0`。两端需能互通，并安装对应消息包。实时建图已启用网络发现。

### 采集与控制话题：ROS_DOMAIN_ID=10

| 话题 | 内容／类型 |
|---|---|
| `/Car/T5/Cam_Left/image_raw/color` | 左图，`sensor_msgs/msg/Image` |
| `/Car/T5/Cam_Right/image_raw/color` | 右图，`sensor_msgs/msg/Image` |
| `/Car/T5/OS1/points` | 原始点云，`sensor_msgs/msg/PointCloud2` |
| `/Car/T5/TOF_Left/image_raw/depth`、`/Car/T5/TOF_Right/image_raw/depth` | 原始 ToF 深度，`sensor_msgs/msg/Image`；深度单位尚未确认，暂不入图 |
| `/car/odom` | 仿真车辆遥测位姿，`nav_msgs/msg/Odometry`；不是本算法估计结果 |
| `/car/cmd_vel` | 人工／外部控制程序的速度指令入口，`geometry_msgs/msg/Twist` |

### 主要节点

| ROS 域 | 节点名称 | 作用 |
|---|---|---|
| 10 | `/sensor_capture_node` | 采集 UE 传感器 |
| 10 | `/lunar_car_node` | 车辆控制与遥测 |
| 10 | `/lunar_car_gui_node` | 人工控制窗口 |
| 10 → 57 | `/fusion_live_sensor_source`、`/fusion_live_sensor_transfer` | 仅转接传感器数据 |
| 57 | `/fusion_sensor_adapter` | 输入适配 |
| 57 | `/fusion_voxelmap` | LiDAR 里程计 |
| 57 | `/fusion_learned_odometry` | 视觉里程计 |
| 57 | `/ekf_filter_node`、`/fusion_odometry_guard` | 融合、检查并发布正式位姿 |
| 57 | `/fusion_terrain_mapper` | 高程、点云和增量地图 |
| 57 | `/t3_visual_monitor`、`/t3_rviz` | P3 可视化 |

在另一个 SSH 终端查看建图话题、节点：

```bash
cd /home/yanfa/P3/lidar_visual_fusion
source scripts/env.sh
export ROS_DOMAIN_ID=57 ROS_LOCALHOST_ONLY=0
ros2 topic list
ros2 node list
```

查看采集和车辆控制时，将 `ROS_DOMAIN_ID` 改为 `10`。规划定位使用域 57 的 `/T3/semantic/current_pose`。

## 6. 地图更新频率

- 入图跟随有效 LiDAR 帧，使用相同时间戳的融合位姿；图像用于改善位姿。当前入图前等待 1.6 秒，让视觉修正进入融合。
- 高程、点云和增量地图每 2 秒检查发布，有新内容才发送，最高约 0.5 Hz。
- RViz 使用沿途累计的全局地图，独立定时器每 2 秒触发发布；压缩落盘异步执行。定时器可能因同进程建图耗时发生抖动，不承诺硬实时。
- 规划端仍使用跟随车辆的局部 64×64 米栅格。未观测区域保持未知。
- 当前 capture 请求上限为 2 Hz；2026-09-16 驻车场景实测，接收端优化后由平均 1.83 秒降到 1.36 秒一批（约 0.74 Hz），仍未达到 2 Hz。缩短地图发布周期不能补出尚未采到的观测。
- LiDAR 定位退化、点云仍有效时，可由合格视觉接续定位；点云完全断流时暂停几何地图新增，目前没有纯双目深度后备建图。

## 7. ToF 与 IMU 当前状态

- ToF 外参在原 P3 的 `workspace/src/t3_semantic_mapping/config/pointcloud_extrinsic.yaml`：`tof_left` 对应 TOF5，`tof_right` 对应 TOF4。它们来自 `config/车体前右下-标定数据20260817(1).xlsx`，已转换为米和 ROS 车体系。
- 两路深度已收到，但按现有 capture 的 `0.01 米/单位` 转换后，地面比 LiDAR 低约 2.5 米。仿真深度编码尚未确认，因此暂未启用 ToF 补图。
- UE 的 `meta.json` 有姿态、速度和角速度字段；当前捕获的文件没有加速度字段。检查时 `/imu/data` 没有发布者。附近端口未找到新服务不代表仿真一定没有 IMU。
- 实时配置已增加图像辅助静止检测：两路图像特征位移、可靠视觉运动和点云变化连续 3 帧一致，才把该帧视觉运动约束替换为零速度。证据异常、变化或过期立即解除；可在 `/fusion/status` 的 `stationary` 字段查看。它不是任意场景下绝对可靠的停车判定。

细节见 [在线漂移与传输核查](live_drift_and_transport.md)。

## 常见情况

- 提示已有任务运行：先 `./status.sh` 查看，再 `./stop.sh`，然后启动新任务。
- RViz 没有地图：先确认数据源有输出；日志位置可通过 `./status.sh` 查看。
- 修改算法代码后：运行 `./build.sh`，编译完成再启动。

日常使用先掌握：**启动 → 看 RViz → 保存 → 停止 → 导出**。

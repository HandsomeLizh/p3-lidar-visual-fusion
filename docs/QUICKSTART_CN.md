# LiDAR＋视觉建图：简明使用说明

更新：2026-09-16。算法为 **VoxelMap＋XFeat/LighterGlue**；仿真配置当前不融合 IMU。日常保留两个启动入口：**采集＋人工控制**、**建图＋RViz**；需要选点导航时另外启动 P4。

## 1. 连接机器

在 Windows 终端连接；下文命令均在远程执行：

```bash
ssh yanfa@192.168.100.216
cd /home/yanfa/P3/lidar_visual_fusion
```

UE 需已启动。控制窗口和 RViz 显示在**远程机器桌面**，SSH 终端本身不显示这些窗口。同一台仿真车由一位操作人员控制；有人使用时先协调，不要同时启动测试或重启程序。

## 2. 实时采集、行驶和建图

### 终端 A：采集＋车辆控制窗口

```bash
cd /home/yanfa/P3/lidar_visual_fusion
./start_simulation_sources.sh
```

自动启动缺少的 capture、车辆桥接和控制 GUI，复用配置一致的已有节点。连接 UE `192.168.10.22`：采集端口 **6665**，控制与反馈端口 **6668**，ROS 域 **10**。

### 终端 B：自己的建图＋RViz

```bash
cd /home/yanfa/P3/lidar_visual_fusion
./start_live.sh
```

自动转接数据到 ROS 域 **57**，启动定位、建图和 P3 组合窗口。等待图像、位姿和地图出现后，使用终端 A 打开的 GUI 人工驾驶。两个启动脚本都不会主动发出行驶目标或非零速度。

需要记录本次输入时，启动命令改为：

```bash
./start_live.sh --record-input
```

需要接口检查时，可加 `--verify --verify-seconds 60`。默认不记录这两类额外数据。

## 3. 接入 P4：在栅格上选点导航

车辆停稳，P3 已有定位和地图后，在终端 C 执行：

```bash
bash /home/yanfa/P4/scripts/navigation.sh start
```

等待提示 `Navigation ready`，在 **P3 组合窗口右侧栅格** 操作：

1. 点 **设置目标**。
2. 在已观测、可通行的位置按下鼠标，拖动指定车头朝向，松开发送；Esc 取消。
3. P4 规划成功后车辆开始执行。绿色是全局路线，紫色是局部路径，橙色是已行驶轨迹。
4. 黑色为障碍，深灰为未知。默认跟车显示 **32×32 米**；点 **全图** 看累计地图，点 **跟车 32 m** 恢复跟随。

“已发送”只表示目标交给 P4，是否可达、车辆是否能通过还由 P4 检查。未知或障碍区域不能选作目标。启动命令不会创建自主探索任务。

**当前 P4 仿真控制使用 `/P4/input/odometry` 的 UE 对齐反馈。** P3 独立估计位姿通过 `/Car/T3/localization/odometry` 发布，UE 绝对位置不参与本建图算法估计。P4 能走到目标，不能单独证明 P3 定位精度达标。

### 停车与结束

先在车辆控制 GUI 停车。使用过 P4 时先结束 P4：

```bash
bash /home/yanfa/P4/scripts/navigation.sh stop
```

然后在终端 B 按 **Ctrl+C**，等待保存并关闭本次建图、RViz。终端 A 可继续保留；需要结束采集和控制时，再按 Ctrl+C。它只关闭自己新启动的节点，已复用的进程仍由原启动入口管理。

**重启 P3 的顺序：停车 → 停 P4 → 保存并停 P3 → 启动 P3 → 启动 P4。** P3 重启会重置本次原点，P4 需要重新对齐。旧窗口不会自动加载新代码。

## 4. 历史 bag 建图

先确认实时建图已停止，再任选一条：

```bash
./start_short_bag.sh fusion   # short，视觉＋LiDAR
./start_short_bag.sh lidar    # short，仅 LiDAR
./start_long_bag.sh           # 20260824_235238 的前约 10 分钟
```

启动后自动显示高程和点云。切换任务前使用 `./stop.sh`。历史回放时不要启用车辆选点导航。

## 5. 状态、地图保存在哪里

```bash
./status.sh       # 当前运行目录和进程
./save_map.sh     # 运行中保存检查点
./stop.sh         # 停止自己的建图；不是车辆停车命令
```

最近一次运行可从 **`results/latest/`** 打开，真实路径以 `./status.sh` 和 `run_state.json` 为准。启动新会话会更新这个快捷入口。

本次建图完全停止后导出：

```bash
./export_map.sh results/latest
```

导出旧会话时改用其实际目录，避免 `latest` 已指向新任务。

| 文件 | 内容 |
|---|---|
| `lidar_map.pcd` | 停止后导出的完整点云 |
| `global_grid_map.sqlite3` | 持久化高程栅格 |
| `global_grid_map.npz` | 容量允许时的全局矩阵；大图保留分块形式 |
| `trajectory_map.tum` | 估计轨迹 |
| `fusion_status.json` | 当前视觉、LiDAR、融合状态 |
| `map_statistics.json` | 入图数量、地图版本、丢帧和存储状态 |
| `resources_latest.json` | 内存和磁盘情况 |
| `verification.json` | 加 `--verify` 后生成的接口检查结果 |

旧结果归档到 `results/history/日期/`；原路径与新路径在 `results/organization/`。本次整理仅限融合项目，**移动归档、不删除数据**。详见 [目录说明](DIRECTORY_CN.md)。

## 6. 常用话题与节点

建图和 P4 使用 **ROS_DOMAIN_ID=57**、`ROS_LOCALHOST_ONLY=0`。长度为米；`map → odom` 为单位变换，车体为 `base_link`。

| 话题 | 内容／类型 |
|---|---|
| `/Car/T3/localization/odometry`、`/T3/semantic/current_pose` | 正式位姿，`nav_msgs/msg/Odometry`，`odom` 系 |
| `/T3/semantic/trajectory` | 已行驶轨迹，`nav_msgs/msg/Path` |
| `/Car/T3/mapping/grid_map` | 规划用局部 **64×64 米**栅格，`grid_map_msgs/msg/GridMap` |
| `/T3/mapping/global_grid_map`、`/Car/T3/mapping/global_grid_map` | 累计全局高程与障碍层，`grid_map_msgs/msg/GridMap` |
| `/T3/mapping/lidar_map` | 有界点云预览，`sensor_msgs/msg/PointCloud2` |
| `/T3/semantic/incremental_map` | 当前滑动窗口，`t3_interfaces/msg/IncrementalSemanticMap` |
| `/Car/T4/rviz_goal` | 选点目标，`geometry_msgs/msg/PoseStamped` |
| `/Car/T4/planning/global_route`、`/Car/T4/planning/local_path` | P4 规划路线，`nav_msgs/msg/Path` |
| `/fusion/status` | 定位健康状态，`std_msgs/msg/String`，JSON 内容 |

主要节点：`fusion_voxelmap`（LiDAR）、`fusion_learned_odometry`（视觉）、`ekf_filter_node`＋`fusion_odometry_guard`（融合与正式输出）、`fusion_terrain_mapper`（建图）、`t3_visual_monitor`＋`t3_rviz`（显示）。

查看话题、节点和健康状态：

```bash
source scripts/env.sh
export ROS_DOMAIN_ID=57 ROS_LOCALHOST_ONLY=0
ros2 node list
ros2 topic list
ros2 topic echo --once /fusion/status
```

采集和控制在域 **10**：左右图 `/Car/T5/Cam_Left/image_raw/color`、`/Car/T5/Cam_Right/image_raw/color`，点云 `/Car/T5/OS1/points`，车辆反馈 `/car/odom`、`/car/telemetry`。完整协议见 [接口与资源约定](接口与资源约定.md)。

## 7. 当前默认效果与限制

- **地图更新**：有效 LiDAR 帧＋同时间戳位姿入图，等待视觉修正约 1.6 秒；局部／全局地图目标周期均为 **2 秒**，实际可能延迟。右侧 32 米是显示窗口，发给 P4 的局部栅格仍是 64 米。
- **采集频率**：请求上限 2 Hz；之前同一驻车场景实测约 **0.74 Hz**，不是稳定 2 Hz。发布频率调高不能生成新的传感器观测。
- **点云参数**：VoxelMap 根体素 **2 米**，输入降采样 **0.2 米**，迭代上限 **20 次**；左侧显示采样 **0.25 米、最多 10 万点**。这些参数作用不同。
- **互相接续**：LiDAR 失配时，质量合格、坐标连续的视觉可独立输出；视觉失效时使用合格 LiDAR。持续失配会尝试候选子图；候选失败不打断合格视觉。两路都失效时停止可信定位输出。
- **点云断流**：暂停几何地图新增，目前没有纯双目深度后备建图。未扫描区域保持未知。
- **障碍更新**：新增障碍随点云加入；清除旧障碍需要至少 3 次合格新观测，定位不可靠或未重新扫描时保留。
- **IMU／ToF**：标准 IMU 接口保留；仿真速度尚未完成一致性核验，默认不融合。ToF 深度单位与外参投影尚未核对一致，暂不补图。详见 [可选 IMU](optional_imu.md)、[速度反馈](velocity_feedback.md)。
- **本轮验收**：运动测试因机器正在被其他人使用而暂停；首段仅发生转向，未完成有效行驶轨迹，不提供本次 ATE。已有 bag 成绩和接口检查不能代替在线精度验收。

## 8. 常见情况

| 情况 | 处理 |
|---|---|
| 提示已有任务运行 | 先 `./status.sh` 核对；其他人正在用时先协调，不要直接停任务 |
| 控制窗口没打开 | `./start_simulation_sources.sh --check` 检查；确认查看的是远程桌面，缺少节点时运行启动脚本 |
| 显示“视觉接续定位” | 视觉正在接替 LiDAR；查看 `/fusion/status` 的 `output_source` 和 `localization_valid` |
| 显示“定位暂不可用” | 没有合格的正式定位结果，或健康状态超时；不只是 LiDAR 状态。视觉仍可能在跟踪，但重置后尚未接回地图坐标，或不确定性超限 |
| 定位失效、地图长时间不更新 | 停车，检查数据源和 `fusion_status.json`、`map_statistics.json` |
| 目标发出但车辆不走 | 查看 `bash /home/yanfa/P4/scripts/navigation.sh status`；确认目标区域已观测、P4 对齐就绪 |
| 新按钮或新代码没生效 | 按第 3 节顺序重启；算法源码修改后先 `./build.sh` |

进一步阅读：[文档索引](INDEX_CN.md) · [新机器环境](SETUP_CN.md) · [目录说明](DIRECTORY_CN.md)。

判断整体定位先看 `localization_valid`。`vision_enabled: true` 只说明视觉观测可用；还要检查 `visual_continuity.anchored` 和 `filtered_quality.reason`。视觉重置后，不能直接把它的新原点当成旧地图原点。

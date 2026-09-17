# LiDAR＋视觉建图：简明使用说明

更新：2026-09-17。默认不带参数的入口采用 **VoxelMap＋XFeat/LighterGlue 定位，只有双目点云建图**；也可选择本页的 **视野内 LiDAR 建图**配置，本次已启动该模式供用户手动试车。仿真配置当前不融合 IMU。日常保留两个启动入口：**采集＋人工控制**、**建图＋RViz**；P4 由使用者另行管理。

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

已有 capture 和控制时，只需启动终端 B。**只发布实际观测到的双目点云及其高程**，边走边累积；未看到的位置保持未知。当前使用学习特征的稀疏双目三角化，静止时反复看同一区域不会铺满整张地图。双目深度无效时保留已有地图、等待有效双目观测，LiDAR 定位仍可继续。详见[当前双目建图模式](stereo_mapping_20260917.md)。

如果希望**以视觉范围为中心、用 LiDAR 测量建图**，在本套 P3 停止后改用以下启动命令。当前按方向保留近处连续扫描，出现超过 0.25 米的水平回波空档后，空档外的孤立点不再进入点云、高程或 P4。15 米是候选上限，实际截断位置由连续观测决定；边缘保留显示渐变。定位继续融合视觉与完整 LiDAR：

```bash
./start_live.sh --profile config/simulation_lidar_camera_fov.yaml
```

详见[视野内 LiDAR 建图](lidar_camera_fov_20260917.md)。两种入口不能同时运行；正在使用 P4 时，需要安排新地图会话的衔接。

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
3. P4 规划成功后车辆开始执行。右侧紫色是局部路径，橙色是已行驶轨迹；左右两侧均不显示全局规划线。
4. P3 实时高程模式按高度着色，深灰为未知；障碍判定交给 P4。默认跟车显示 **32×32 米**；点 **全图** 看累计地图，点 **跟车 32 m** 恢复跟随。

“已发送”只表示目标交给 P4，是否可达、车辆是否能通过还由 P4 检查。未知区域不能选作目标；旧配置提供障碍层时也会拒绝黑格。启动命令不会创建自主探索任务。

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
| `lidar_map.pcd` | 停止后导出的完整点云；保留历史文件名，当前实时模式内容仅来自双目 |
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
| `/T3/mapping/global_grid_map`、`/Car/T3/mapping/global_grid_map` | 累计全局高度、方差和观测次数，`grid_map_msgs/msg/GridMap` |
| `/T3/mapping/lidar_map` | 有界点云预览，`sensor_msgs/msg/PointCloud2` |
| `/T3/semantic/incremental_map` | 旧分类模式的兼容消息；当前实时高程模式不发布 |
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

- **地图更新**：有效点云＋同时间戳位姿入图；局部／全局地图目标周期均为 **2 秒**，实际可能延迟。实时融合的双目建图与视觉范围 LiDAR 建图两份配置均给 P4 发布 **64×64 米**局部图（320×320 格），右侧界面独立显示车周围 **32×32 米**。窗口扩大不会把未观测区域填成平地。
- **采集频率**：请求上限 2 Hz；之前同一驻车场景实测约 **0.74 Hz**，不是稳定 2 Hz。发布频率调高不能生成新的传感器观测。
- **点云参数**：当前实时配置的 VoxelMap 根体素 **1 米**，输入降采样 **0.1 米**，迭代上限 **30 次**，收敛后提前结束；视觉范围 LiDAR 模式预览采样 **0.15 米、发布最多 15 万点、窗口渲染最多 10 万点**。默认双目模式的发布预览仍为 0.25 米。这些参数作用不同。
- **显示样式**：左侧为 **Points、2 像素**；右侧仍是按高度着色的栅格。连续性裁剪同步影响两侧和发给 P4 的高度，但不会清除 P4 已缓存的旧地图。
- **高程融合**：实时默认按不确定性加权并检查跨帧高度冲突，只发布高度、方差和观测次数。P4 自己判断障碍和通行性。见 [概率高程融合与接口](probabilistic_elevation_20260917.md)。
- **互相接续**：仿真配置小幅提高合格视觉运动增量的权重，两路正常时输出融合位姿，保留 LiDAR 的高度与姿态约束。雷达失效或融合结果不可用时，由合格连续视觉接续；视觉失效或过时则使用合格 LiDAR／EKF。持续失配会尝试候选子图；候选失败不打断合格视觉。两路都失效时停止可信定位输出。见 [视觉偏好与误障碍修复](visual_preference_20260916.md)。
- **点云断流**：所选建图来源无有效点云时暂停新增，保留历史地图。两份实时配置分别选择双目或 LiDAR，不会自动切换建图来源；未扫描区域保持未知。
- **表面更新**：与旧高度冲突的高表面需连续确认；降低已有表面需要独立的清除证据。纯视觉定位模式目前只用 LiDAR 建图，暂不自动清除旧高表面，未重扫区域保留。
- **IMU／ToF**：标准 IMU 接口保留；仿真速度尚未完成一致性核验，默认不融合。ToF 深度单位与外参投影尚未核对一致，暂不补图。详见 [可选 IMU](optional_imu.md)、[速度反馈](velocity_feedback.md)。
- **本轮验收**：已完成启动接续、关键帧恢复及路径显示的隔离回归和实际图像干扰回放。本轮不控制车辆，也没有新行驶轨迹的 ATE；历史运动记录与当前回归结果见 [项目复盘](PROJECT_EVOLUTION_CN.md)。

## 8. 常见情况

| 情况 | 处理 |
|---|---|
| 提示已有任务运行 | 先 `./status.sh` 核对；其他人正在用时先协调，不要直接停任务 |
| 控制窗口没打开 | `./start_simulation_sources.sh --check` 检查；确认查看的是远程桌面，缺少节点时运行启动脚本 |
| 显示“视觉优先定位”或“视觉接续定位” | 正在输出合格视觉位姿；前者是仿真偏好，后者是按不确定性接续。查看 `/fusion/status` 的 `output_source` 和 `localization_valid` |
| 显示“定位暂不可用” | 没有合格的正式定位结果，或健康状态超时；不只是 LiDAR 状态。视觉仍可能在跟踪，但重置后尚未接回地图坐标，或不确定性超限 |
| 定位失效、地图长时间不更新 | 停车，检查数据源和 `fusion_status.json`、`map_statistics.json` |
| 目标发出但车辆不走 | 查看 `bash /home/yanfa/P4/scripts/navigation.sh status`；确认目标区域已观测、P4 对齐就绪 |
| 新按钮或新代码没生效 | 按第 3 节顺序重启；算法源码修改后先 `./build.sh` |

进一步阅读：[文档索引](INDEX_CN.md) · [新机器环境](SETUP_CN.md) · [目录说明](DIRECTORY_CN.md)。

判断整体定位先看 `localization_valid`。`vision_enabled: true` 只说明视觉观测可用；还要检查 `visual_continuity.anchored` 和 `filtered_quality.reason`。视觉重置后，不能直接把它的新原点当成旧地图原点。

2026-09-17 关键帧恢复更新：保留当前、备用参考及最多 4 个历史关键帧；从最后一次可信跟踪起最多保留 120 秒。长间隔或历史关键帧恢复需双目几何检查及两次确认，成功后延续原坐标段；失败期间不把旧位姿当成新观测。超过保留时间或没有共同可见内容时，仍需重新建立可信坐标关系。完整回环留到下一步。详见 [关键帧恢复与启动修复](startup_recovery_20260917.md)；旧版 30 秒、两个参考的记录见 [阶段报告](visual_recovery_gap_20260916.md)。已启动会话不会自动加载之后修改的代码和参数。

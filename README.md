# P3：LiDAR＋视觉定位与建图

面向 ROS 2 Humble 的 **VoxelMap＋XFeat/LighterGlue** 融合建图模块。输出位姿、轨迹、高程栅格、LiDAR 点云和增量地图，沿用 P3 的规划接口与 RViz 显示。

## 算法

- LiDAR：VoxelMap 平面及方向协方差、配准质量检查、收敛检查；无 IMU 时失配回到最后可信状态，停止无观测支持的速度外推。逐帧定位，移动 0.1 米或转动 1 度后才向内部配准地图加入新视角。
- 视觉：轻量学习特征与稀疏匹配，检查曝光、对比度、清晰度、特征覆盖、双目几何和运动一致性。
- 融合：视觉提供车体运动约束，支持迟到观测修正；输出检查协方差、位移和转角，包括同时间戳的累计修正。正式位姿和主 TF 使用同一合格结果。视觉不合格时继续使用合格 LiDAR。
- IMU：可选；当前演示未启用经过标定的 IMU 融合。接入条件见 [可选 IMU](docs/optional_imu.md)。
- 地图：磁盘分块高程、有限内存缓存、有界点云预览和输入队列，控制长时间运行的内存增长。
- 静止检测：实时模式结合双目特征位移、视觉运动和点云变化，连续确认后使用零速度约束。可选车辆速度反馈用于否决误判；零反馈不能单独证明静止，断流时回退到图像与点云判据。
- 车辆反馈：6668 速度已知为世界坐标，但与姿态的轴向、时钟关系尚未核准；按当前要求暂停仿真反馈接入，默认关闭，不参与积分。保留可选转换代码及实车标准 IMU 接口。详见 [车辆速度反馈与 P4 核查](docs/velocity_feedback.md)。

视觉可在质量合格时接续定位；当前几何建图仍需要有效点云，尚未实现纯双目深度后备建图。

## 在现有 P3 机器上启动

先启动 UE，再开两个终端；车辆由人操作控制窗口。

**终端 A：采集＋车辆控制＋控制窗口**

```bash
cd /home/yanfa/P3/lidar_visual_fusion
./start_simulation_sources.sh
```

**终端 B：建图＋RViz**

```bash
cd /home/yanfa/P3/lidar_visual_fusion
./start_live.sh
```

采集与控制使用 ROS 域 **10**，建图使用域 **57**，传感器转接自动启动。采集脚本会复用匹配的已有进程，缺少控制窗口时补开窗口。启动脚本不选择驾驶模式、不设置非零速度；人工在 GUI 中操作。建图程序不发布车辆指令。

结束时先由人停车，再在终端 B 按 Ctrl+C，保存地图并结束建图。终端 A 管理自己启动的采集、控制和 GUI。GUI 保留原 P3 的关闭停车行为。

历史 bag：

```bash
./start_short_bag.sh fusion          # 现有 P3 的 short bag
./start_long_bag.sh                 # 现有 P3 的长 bag
./status.sh
./save_map.sh
./stop.sh
```

完整操作见 [简明使用说明](docs/QUICKSTART_CN.md)。**新机器先看 [环境与依赖](docs/SETUP_CN.md)**：仓库提供本模块源码，原 P3 的采集、控制 GUI、定制 RViz 和 bag 属于外部依赖。

## 当前 RViz 设置

组合窗口实际加载 [config/p3_visual_window.rviz](config/p3_visual_window.rviz)，由 `scripts/p3_visuals.sh` 指定；这份配置随源码版本同步。固定坐标系为 `map`，显示彩色累计点云、橙色轨迹、青色车头和车体坐标轴，初始全局视距为 300 米；窗口根据地图范围调整视角。右侧高程图沿用 P3 监视器及本模块适配。

[config/fusion.rviz](config/fusion.rviz) 是独立 RViz 的备用配置。当前组合窗口仍依赖原 P3 的 Qt 程序和显示插件，配置文件已独立保存在本仓库。

## 地图更新与规划接口

地图由有效 LiDAR 帧配合同时间戳的融合位姿更新；当前实时配置等待 1.6 秒，让视觉修正参与入图位姿。局部地图每 2 秒检查发布，有新内容才发送；规划端使用跟随车辆的 64×64 米窗口。RViz 使用累计全局地图，独立定时器每 2 秒触发发布，压缩保存交给有界后台线程。上述周期是调度目标，实际延迟还受建图耗时影响。

采集请求上限已设为 2 Hz、额外批次等待为零。2026-09-16 同一驻车场景前后各 30 批实测，接收端重叠解码后由平均 1.83 秒降到 1.36 秒一批（约 0.74 Hz），尚未达到 2 Hz。每批约 72 MB，当前 UE 数据链路为 1 Gbps。现有启动脚本自动优先使用已构建的独立采集副本，构建方法见环境说明。

| 话题（ROS 域 57） | 类型 |
|---|---|
| `/T3/semantic/current_pose` | `nav_msgs/msg/Odometry` |
| `/T3/semantic/trajectory` | `nav_msgs/msg/Path` |
| `/T3/semantic/incremental_map` | `t3_interfaces/msg/IncrementalSemanticMap` |
| `/T3/mapping/elevation_map` | `grid_map_msgs/msg/GridMap` |
| `/T3/mapping/lidar_map` | `sensor_msgs/msg/PointCloud2` |

坐标、地图查询和资源限制见 [接口与资源约定](docs/接口与资源约定.md)。地图、日志和轨迹保存在每次运行的 `results/` 子目录。

## 已完成的验证

最新 [跳变修复与体素对比](docs/jump_repair_20260916.md)：保留 **2 米根体素、0.2 米降采样、20 次迭代**。同一段历史行驶数据 LiDAR 匹配 68/68 帧成功，评分路程 14.10 米，LiDAR 位置 ATE **0.133 米（0.94%）**。2.5 米、3 米根体素未显示优势。人为失配后的停止外推、正常恢复、位姿/TF 跳变拒绝与可选 IMU 回归通过；这项成绩是回放的 LiDAR 结果，不是本轮在线融合精度验收。

2026-09-16 配准修复：同一段 68 帧在线录制数据按 1 倍速回放，融合位置 ATE 从 **0.358 m 降至 0.178 m**；评分区间约 14.10 m，ATE／路程 **1.26%**。68 帧全部入图、0 丢帧，10 项接口检查通过。LiDAR 后端处理时间中位数约 **98.5 ms**、P95 约 **136.1 ms**。这些是指定数据上的测量，不代表任意地形或高频采集性能。

- [配准修复及回归](docs/lidar_registration_fix.md)：保留修复前后比较与失败回退验证。
- [停车后漂移、地图发布与传输核查](docs/live_drift_and_transport.md)：关键帧修正、协方差一致性检查，以及 ToF／IMU 的已知限制。
- [多轨迹与图像干扰](docs/fusion_multitrajectory_report.md)：黑屏、过曝、斜强光和低照度扰动；报告对应当时版本。
- [早期在线测试](docs/live_simulation_report.md)：包含首次在线失败记录，供追查。

最新人工驾驶启动方式已实际打开控制 GUI、运行采集转接和建图，并通过 10 项实时接口检查；**尚未完成修复后的新行驶轨迹精度验收**。历史 bag 成绩与接口连通不能替代这一项。

仓库保存源码、配置、测试和文字报告；原始 bag、模型权重、运行地图、构建产物与历史备份不进入 Git。第三方来源、版本和许可证见 [THIRD_PARTY.md](THIRD_PARTY.md)。

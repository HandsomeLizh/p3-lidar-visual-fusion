# 在线纯视觉定位、LiDAR 建图

在 Orin 上启动，UE、capture 和手动车辆控制沿用当前入口：

```bash
cd /home/yanfa/P3/lidar_visual_fusion
./start_visual_only_live.sh
```

试用原 RoMa 定位时，改用 `bash start_roma_visual_live.sh`；仍使用本项目 LiDAR 建图及原界面。具体限制和 Z 坐标核验见 [RoMa 试验说明](roma_coordinate_check_20260917.md)。

- 界面：原 P3 左侧点云、右侧栅格和双目图像窗口。
- 定位：XFeat＋LighterGlue 双目里程计，接入双目图像静止确认；LiDAR、IMU、仿真真值均不参与位姿估计。
- 静止确认：两张图的光流、覆盖范围和视觉运动量连续满足要求后约束位姿；图像运动、质量不足或中断会解除。光流使用固定图像参考，避免掩盖缓慢移动。
- 双目三角化点回到左图实际视线，避免右图纵向匹配误差导致静止 PnP 偏转；关键帧按有效运动更新，不再仅因时间到期而换帧。
- 建图：只用 LiDAR 点云，使用视觉位姿投到地图。按 2026-09-17 最新要求，已关闭双目点云的发布和入图；双目图像、三角化及匹配仍用于视觉定位。
- 同一曝光时间的点云必须找到对应的有效位姿才入图。视觉短时失效时暂停新增；原点重置后保留旧地图并暂停，需要重新启动本次测试。
- 地图窗口 32×32 m，全局地图约每 2 秒发布，继续受内存保护约束。
- 2026-09-17 起默认使用[概率高程融合](probabilistic_elevation_20260917.md)。局部、全局均只输出高度、方差和观测次数，P4 负责障碍与通行判断。
- 本模式暂不自动降低已确认的旧高表面，保留当前保守更新策略。
- 输出已恢复到 ROS_DOMAIN_ID=57，输入来自 10。位置、轨迹、地图采用原 P3 对外接口，`map → odom` 为单位变换，`odom → base_link` 为纯视觉估计。
- 默认手动测试，窗口的发送目标功能关闭；使用原车辆控制窗口。需要单独隔离评估时可加 `--domain 83`。

要在原窗口选点交给 P4 规划行驶，停车后用下面的顺序启动：

```bash
cd /home/yanfa/P3/lidar_visual_fusion
./start_visual_only_live.sh --allow-goals
# 另一个终端，等待 P3 已有有效位姿和地图：
bash /home/yanfa/P4/scripts/navigation.sh start
```

右侧点 **设置目标**，在已扫描栅格按下并拖动设置朝向，松开发送；启动本身不会发目标。
P3 重启后须停稳并重启 P4 对齐。P4 当前使用 UE 对齐反馈控制车辆，P3 地图仍只用纯视觉位姿；
导航到达率不等于纯视觉定位精度。

主要话题（默认域 57）：

| 话题 | 内容 |
|---|---|
| `/Car/T3/localization/odometry` | 对外纯视觉位姿，`nav_msgs/msg/Odometry`，`odom` 坐标系，子坐标 `base_link` |
| `/T3/semantic/current_pose`、`/odometry/filtered` | 同一有效位姿的原兼容话题 |
| `/T3/semantic/trajectory` | 纯视觉轨迹 |
| `/Car/T3/debug/trajectory` | 同一轨迹的原兼容话题 |
| `/fusion/lidar` | 用于建图的规范化 LiDAR 点云 |
| `/fusion/stereo_points` | 当前关闭，无双目建图点云发布 |
| `/T3/mapping/lidar_map` | LiDAR 累计点云 |
| `/Car/T3/mapping/grid_map` | 跟车 32×32 m 高程图，含高度、方差和观测次数 |
| `/T3/mapping/global_grid_map` | 累计全局高程图，层与局部一致 |

终端 Ctrl+C 停止本次定位、建图和窗口；capture、车辆控制独立保留。

记录位于 `results/visual_only_live_时间/`，当前目录见 `visual_only_live_state.json`：

- `online_status.json`：轨迹与参考位置按时间配对，单个视觉原点内做一次不缩放刚体对齐；静止误差不能当作行驶 ATE。
- `visual_raw.jsonl`、`reference.csv`：视觉和仿真参考分别保存；参考只用于评估。
- `map_statistics.json`：`lidar_scans` 持续增长，`stereo_scans`、`stereo_points` 应保持 0。
- `learned_metrics.json`：视觉耗时、拒绝原因及定位内部的双目点数；内部有双目点不代表将其用于建图。

关闭双目入图后使用新结果目录，防止旧地图里的双目点继续保留。旧结果归档保留。

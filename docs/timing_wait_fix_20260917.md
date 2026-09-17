# 视觉晚到引起的短暂停发修复（2026-09-17）

工程：`/home/yanfa/P3/lidar_visual_fusion`。

## 原因与改动

同帧视觉结果到达晚于 LiDAR 的等待上限（原 0.45 秒）时，正常雷达帧也会被判为缺少运动参考。旧等待槽只有一张雷达帧，等待期间到达的更新雷达帧会直接丢弃。

- 两个仿真配置将 `voxelmap.independent_visual_wait_sec` 设为 **1.0 秒上限**。同帧视觉结果一到即处理，不固定等待 1 秒。
- 队列最多保存 **2 张**点云：正在等参考的帧，以及最新的后续帧。满载时替换后续帧，避免积压和持续丢掉最新输入；不会无限增加内存。
- 仍要求视觉同一 epoch、时间戳对齐、协方差合格；缺失/无效视觉仍拒绝弱几何帧。
- 新增 `reference_wait_sec`、`pending_depth`、`pending_peak`、`independent_visual_pair_available` 诊断字段。
- 0.3 m/s 运动门限、退化方向投影、收敛判定和 `output_discontinuity` 均保持。

## 验证

实际 ROS 节点、隔离 domain 88、合成平面和受控消息到达时序：

| 场景 | 结果 |
|---|---|
| 视觉晚到 0.7 秒 | 旧 0.45 秒配置拒绝连续 5 帧；新版全部接纳 |
| 2 Hz 输入，视觉晚到 0.7 秒 | 新版 6 个后续帧全部接纳、无队列丢帧 |
| 超过队列容量的输入和突发输入 | 中间帧可以丢弃，最新帧保留；队列峰值 2 |
| 视觉结果很快到达 | 本次等待约 25～31 ms，没有被固定拖到 1 秒 |
| 视觉缺失或无效 | 两帧均拒绝，位姿及配准地图保持；后续有效帧恢复 |

原始跳变记录的 90 帧回放仍全部通过，没有接纳未收敛结果；末段最大相邻位移约 7.33 mm。上述测试不是在线行驶精度或 ATE。

源码、C++ 镜像和经过测试的二进制已安装到主项目，并更新/验证 build_manifest。18:29 已按用户要求重启 P3，等待与队列修复开始生效；后续 18:51、19:09 启动的本项目会话继续包含此修复。

## 当时的切换条件与后续状态

P3 切换会重建地图坐标原点。P4 的导航/控制进程在 18:04 附近又启动过；18:11 检查为 PARKED/STOPPED，但仍持有跟踪会话信息。当时因此等待操作者安排任务和地图衔接，随后按照用户的重启要求完成切换；本轮没有停止 P4 或发送车辆控制。

18:11 检查时，左右图像和 LiDAR 已约 9 分钟没有新数据，定位为 no_active_source。UE 恢复后已收到新数据；后续发现的启动视觉坐标接续及长时间黑暗恢复问题另见 [关键帧恢复记录](startup_recovery_20260917.md)。本修复不把断流或真正过期的数据当作有效定位。

分析正在录制的 SQLite bag 时，记录器出现过 3 次 database is locked 写入报错（各一路左图、右图、LiDAR）。实时发布发生在录包写入之前；日志保留，后续分析应使用快照或已停止的 bag。

## 文件与报告

- `src/t3_voxelmap/src/voxelmap_node.cpp` 与 `scripts/voxelmap_node.cpp`
- `config/simulation_lidar_camera_fov.yaml` 与 `config/simulation_live.yaml`
- `tests/test_delayed_visual_lidar_ros.py`
- `results/timing_recovery_20260917/fixed_v2/report.json`
- `results/timing_recovery_20260917/replay_jump/report.json`
- `results/timing_recovery_20260917/deployment_manifest.json`

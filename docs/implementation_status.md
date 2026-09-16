> **阶段记录**：本文保留当时的实现与测量。当前默认配置、修复结果和多轨迹对照以 [最终多轨迹报告](fusion_multitrajectory_report.md) 为准。

# 实现与验证状态

## 已实现

- 独立目录 `/home/yanfa/P3/lidar_visual_fusion`；原项目负责的 RoMa、传感器接收与控制进程未改动。
- LiDAR 点云与高程地图、轨迹、原项目 ROS2 规划接口，以及原 P3 同一 RViz 窗口。
- 默认 VoxelMap 无 IMU ROS2 适配，复用官方概率平面地图和点面配准；Point-LIO 已退出运行与构建。
- XFeat＋LighterGlue 双目里程计；有界关键帧与图像队列，图像/几何/时间检查，失败探测与资源保护。
- 新自适应门控分别评估视觉与 LiDAR；弱 LiDAR 不再阻挡健康视觉，两路失效时报告降级。
- 地图分块持久化、有界窗口与显示、查询、保存、完整流式导出、任务进程树内存监控。
- 回放前启动接口观察器；记录非锚点视觉约束，避免将话题存在误报为视觉融合有效。

## 已有证据

- 当前 6 个 ROS2 包构建通过：`logs/build_voxelmap_default.log`；默认运行图已确认只有 VoxelMap。
- XFeat＋LighterGlue 已在 Orin CUDA 完整回放 short bag。旧融合 RMSE 2.57 m，不合格；独立视觉约 0.099 m。见 [基线报告](short_bag_test_20260915.md)。
- Point-LIO 状态规模求解及无 IMU 常速度模型数学回归：`bounded_gain_verification.json`、`output_motion_verification.json`。
- LiDAR 平地退化、丰富几何、高残差与低重叠判据：`registration_quality_verification.json`。
- 自适应 ROS 切换与来源重置回归：`../results/adaptive_guard_regression/verification.json`。最大步长 0.02 m 与合成输入一致；两路同时无效时停止合格输出。此测试没有替代实际 EKF 精度评估。
- 新 VoxelMap 完整回放输出：`../results/short_voxelmap_adaptive_20260915/`；已生成 `assessment.json`，最终 RMSE 0.112 m、9 项接口通过；详见 [实测报告](voxelmap_short_report.md)。

## 当前边界

当前仿真没有 IMU，用户已确认；PV-LIO 仅冻结官方源码，尚未完成 ROS2 运行适配或测试。实机提供真实 IMU 后再验证同步、标定与融合。

尚未完成真实异常光照/重复纹理长序列、在线行走与规划器端到端联调、30～60 分钟连续新增区域的资源验证、实机 IMU 联调。没有地表参考来认证高程误差，也没有全局回环修正。

轻量视觉在正常 short bag 可运行，不代表已证明极端环境优于 RoMa。LiDAR 和视觉同时缺少约束时，系统应诚实报告不可定位。

启动、算法来源、区别与内存约束见 [候选说明](candidate_status.md)。

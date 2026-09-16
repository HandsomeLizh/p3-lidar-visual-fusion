# VoxelMap 可选 IMU

范围：在 `P3/lidar_visual_fusion` 的 ROS2 VoxelMap 后端加入惯性预测及协方差传播。点面配准、体素地图、规划接口和 P3 RViz 入口保持同一套。当前默认视觉是 XFeat + LighterGlue 双目，它本身不直接积分 IMU；旧 `use_imu` 参数仍控制可选 VINS。新的 `imu_mode` 单独控制 VoxelMap。

## 运行行为

- `imu_mode: auto`：订阅 IMU。缺失、未标定、初始化不足、时间覆盖不完整或断流时，继续原来的 LiDAR 匀速模型，不等待 IMU 才发布。
- `imu_mode: off`：完全关闭后端惯性输入，便于消融对比。
- 有效 IMU 经过静止初始化后，用陀螺仪和加速度计预测姿态、速度和位置，并传播误差协方差。初始化默认至少 1 秒、50 个样本；同时检查加速度模长、波动、角速度和当前 LiDAR 运动估计。
- 断流回退时保留当前位姿、速度和地图；恢复默认需要 3 个完整扫描间隔。真实 IMU 零偏独立保存，避免把无 IMU 模式的角速度状态误当作陀螺零偏。
- 队列最多 4,000 个样本 / 8 秒；不保留无限增长的 IMU 历史。模式切换会保守增加协方差。

## 实车配置

复制一个当前已验证的视觉 + VoxelMap 配置，填写以下参数：

```yaml
imu_mode: auto
imu_topic: /imu/data
imu_input_frame: imu_link
imu_calibration_confirmed: true
base_from_imu:   # 填写实际测量的 4x4 刚体变换
  - [1, 0, 0, 0]
  - [0, 1, 0, 0]
  - [0, 0, 1, 0]
  - [0, 0, 0, 1]
voxelmap:
  threads: 2
  imu_time_offset_sec: 0.0
  imu_max_gap_sec: 0.05
  imu_end_tolerance_sec: 0.015
  imu_recovery_scans: 3
  imu_buffer_seconds: 8.0
  imu_buffer_samples: 4000
  imu_gyro_noise: 0.01
  imu_accel_noise: 0.1
  imu_gyro_bias_noise: 0.0001
  imu_accel_bias_noise: 0.001
```

上面的单位矩阵只是格式示例，不是实车标定。当前仿真配置保留 `imu_calibration_confirmed: false`，收到未知来源 IMU 也不会误用。时间偏移定义为 **LiDAR 时钟上的时刻 = IMU 消息时间 + imu_time_offset_sec**。噪声为连续噪声密度及随机游走的 SI 单位工程初值，需要实车标定。

IMU 消息必须使用 rad/s、m/s²，比力包含静止时的重力响应。无需使用 IMU 的四元数；允许 orientation_covariance[0] = -1。角速度或加速度缺失标记、非有限值、重复/倒序、饱和及超过当前 ROS 时间容限的样本会被拒绝。旋转外参和 IMU/LiDAR 安装位置差均参与传播；采用 IMU 原点积分后转换回 LiDAR 原点，处理转动时的安装偏移效应。

启动仍使用：

```bash
cd /home/yanfa/P3/lidar_visual_fusion
./start_voxelmap_fusion.sh --profile config/vehicle.yaml --rviz
```

通过 `/fusion/imu_status` 或 `/fusion/lidar_metrics` 的 `imu` 字段读取 `mode`、`reason`、初始化状态、使用样本数、积分时长、拒绝计数和切换计数。`imu` 表示本次扫描确实使用惯性预测；并不代表当前融合门控一定接纳了该帧 LiDAR 位姿。

## 点云去畸变边界

本次加入的是惯性预测，没有实现原始旋转扫描的逐点去畸变。支持：

- 仿真瞬时点云：`instantaneous_cloud: true`。
- 上游已经按消息时间去畸变的点云：`instantaneous_cloud: false`、`cloud_motion_compensated: true`。

实际原始滚动扫描不能当作瞬时点云。硬件模板会拒绝未经去畸变且未声明已补偿的输入；接原始雷达时还需要完成上游逐点补偿，并让定位与建图使用一致的补偿点云。这项限制与“可选 IMU 的接入、积分和断流切换”是两个工作项。

## 验证与边界

`tests/test_optional_imu.cpp` 验证无 IMU 的数学等价、重力、加速度运动、转动、旋转外参、安装偏移、协方差、断流和恢复、坏数据与有界缓存。

`tests/test_optional_imu_ros.py` 在独立 ROS_DOMAIN_ID=58 下，将同一段真实 bag 的 12 帧 LiDAR 分别送给修改前的冻结二进制、`auto` 和 `off`，比较位姿及协方差；另用明确标记的合成 IMU 验证真实 ROS 消息链路和切换。测试结果另存 `results/optional_imu_validation_20260915/`。

现有仿真 bag 没有 IMU。合成测试不证明实车精度提升；需要实际同步 IMU + LiDAR 数据后做相同路程的 ATE / 路程对比。长段融合修复及视觉干扰对照见 [融合修复报告](fusion_repair_report.md)。

模型依据：[VoxelMap 上游惯性传播实现](https://github.com/hku-mars/VoxelMap/blob/d787ee8ccfb0e509a36adb2c52bd5da97b29c39a/src/IMU_Processing.hpp)。本地可选接入、模式转换、安装偏移处理及测试是本项目扩展，不宣称复现 PV-LIO。

## 本次已完成验证

- 6 个包编译通过。
- 动力学、外参、安装偏移、协方差、缓存上限和断流恢复的 C++ 测试通过。
- 接入 IMU 阶段的历史版本在同一 bag 的 12 帧 LiDAR 上，位姿和协方差最大差异均为 0。后续方向性不确定性修复有意修改了发布协方差，不能沿用这条历史协方差结论。
- ROS 消息级切换测试通过；静止场景切换时最大相邻位置差约 1.4×10⁻⁸ 米，5 类坏输入均被拒绝。此值只用于检查合成场景的连续性，不是定位精度。
- 证据：`docs/optional_imu_verification.json` 与 `results/optional_imu_validation_20260915/`。

2026-09-16 融合修复后的消息级复测通过：相对接入 IMU 前冻结二进制，12 帧原始 LiDAR 位姿最大差异仍为 0；当前 `auto` 与 `off` 的位姿和协方差完全一致。合成 IMU 接入、断流、恢复和 5 类无效输入检查通过，静止切换最大相邻位置差为 1.39×10⁻⁸ 米。证据：`results/optional_imu_after_fusion_repair_20260916/ros_validation.json`。这些检查仍不等同于实车 IMU 精度测量。

已有完整同步 IMU bag 时，可用 `./start_voxelmap_fusion.sh --profile config/vehicle.yaml --use-sim-time` 接收外部 `/clock`，再按传感器话题白名单回放。当前 `start_bag.sh` 的历史数据播放器只负责双目和 LiDAR；它不回放 IMU。

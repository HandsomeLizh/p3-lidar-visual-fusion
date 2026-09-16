> **阶段记录**：本文保留当时的实现与测量。当前默认配置、修复结果和多轨迹对照以 [最终多轨迹报告](fusion_multitrajectory_report.md) 为准。

# 多轨迹验证：固定当前算法与参数

日期：2026-09-16。远程目录：`/home/yanfa/P3/lidar_visual_fusion`。

## 精度与完整性

沿用修复后的 VoxelMap＋XFeat/LighterGlue＋EKF，同一标定和门控参数，1 倍速回放并开启同一套 P3 RViz。LiDAR、原始视觉和融合取自同轮运行；原始前端不接收融合或真值反馈。每个方法使用完全相同的被评分时间戳，只进行一次固定尺度刚体对齐。视觉出现多个独立坐标段时不拼接出虚假的整段 ATE。

| 路线 | 路程 / 时长 | 评分帧 / 输入帧 | LiDAR / 视觉 / 融合 ATE（cm） | 融合 ATE / 路程 | 融合 P95 / 最大（cm） |
|---|---:|---:|---:|---:|---:|
| 原始修复段 | 79.41 m / 598.7 s | 174 / 174 | 4.06 / 25.88 / 4.11 | 0.0518% | 7.08 / 15.53 |
| short（近似评分） | 14.25 m / 232.4 s | 57 / 58 | 16.78 / 9.88 / 15.92 | 1.1166% | 16.07 / 93.94 |
| 同 bag 后续地形段 | 84.78 m / 639.9 s | 174 / 174 | 3.73 / 24.58 / 3.81 | 0.0450% | 7.61 / 10.99 |
| 第二天录制路线 | — | — | 评分不可用，保留失败记录 | — | — |

**short 的评分属于近似诊断。** 首帧早于参考流约 0.97 秒，其余采集时刻落在约 0.33–0.39 秒的参考缺口中；原 0.25 秒限制无法评分。保留原 `benchmark_error.json`，另存 `assessment_reference_gap_050.json`，允许最多 0.5 秒间隙内插，不外推、不调整时间偏移。不能与其他路线的厘米级差值直接做严苛排名。

short 的原始视觉 ATE 约 9.88 cm，优于 LiDAR；当前融合没有充分利用这一优势。第 54 帧（从 0 开始、1220.244 秒）原始 LiDAR 与融合均有尖峰，融合最大偏差约 0.94 m；同帧视觉几何分数约 0.84，但来源矛盾后没有送入。现有几何合格判据与冲突策略仍有不足，不能宣布所有场景切换都已解决。证据：`short_low_contrast/single_frame_spike_audit.json`。

## 时间和内存

| 路线 | LiDAR 中位数 / P95（ms） | 视觉有效跟踪帧中位数 / P95（ms） | 最后修正位姿接收中位数 / P95（ms） | 进程树 RSS 峰值（GiB，含 RViz） |
|---|---:|---:|---:|---:|
| 原始修复段 | 112.8 / 153.3 | 269.8 / 339.8 | 485.2 / 523.9 | 2.86 |
| short（近似评分） | 83.3 / 103.2 | 241.0 / 301.8 | 464.4 / 490.9 | 2.69 |
| 同 bag 后续地形段 | 111.8 / 138.0 | 249.3 / 328.7 | 480.0 / 520.5 | 2.87 |

各阶段并行，耗时不可相加。接收延迟从该帧 bag 发布开始，不是曝光到车辆动作的延迟；RSS 是采样的进程树求和，可能重复计算共享页。所有输入仍是稀疏采集，不能由这些结果推导高频实车吞吐。

## 建图与接口

| 路线 | 入图 / 输入 | 丢弃 / 待处理 | 完整点云点数 | 接口检查 |
|---|---:|---:|---:|---|
| 原始修复段 | 174 / 174 | 0 / 0 | 754,619 | 10 / 10 |
| short（近似评分） | 58 / 58 | 0 / 0 | 319,926 | 10 / 10 |
| 同 bag 后续地形段 | 174 / 174 | 0 / 0 | 725,905 | 10 / 10 |
| 第二天录制路线 | 138 / 174 | 36 / 0 | 613,077 | 10 / 10 |

接口通过说明 ROS 输出、消息类型、坐标和主 TF 检查通过，不等于定位精度合格或规划控制闭环完成。完整地图落盘，显示使用有界点云预览和局部高程。无地表真值，不给出高程绝对精度结论。

## 数据独立性与复现

- 修复段：`20260824_235238` 原始帧 0:174。
- 后续地形段：同 bag 原始帧 174:348，与修复段不重叠，但属于同一次录制。
- short：独立录制 `short_20260915_190151`，全部 58 帧。
- 第二天路线：独立 bag `20260825_232412`，固定前 174 个完整批次；有约 78 米运动。采用同一 T5 标定假设，该 bag 无独立标定快照。
- 数据选择在评分前固定，不根据成绩删除片段或调整算法。

索引、传感器配对审计、参考 CSV 与选择理由在 `test_data/multi_trajectory_20260916/`。后续地形段和第二天路线可分别执行：

```bash
cd /home/yanfa/P3/lidar_visual_fusion
./test_fusion_bag.sh clean --bag /home/yanfa/P3/roma_t3_algorithm_bundle_20260825/recordings/short_20260915_190151/bag --frame-index /home/yanfa/P3/lidar_visual_fusion/test_data/multi_trajectory_20260916/short_low_contrast/index.json --reference /home/yanfa/P3/lidar_visual_fusion/test_data/multi_trajectory_20260916/short_low_contrast/reference.csv --frames 58
./test_fusion_bag.sh clean --bag /home/yanfa/Env_X/InterFace/bags/20260824_235238 --frame-index /home/yanfa/P3/lidar_visual_fusion/test_data/multi_trajectory_20260916/heldout_terrain/index.json --reference /home/yanfa/P3/lidar_visual_fusion/test_data/multi_trajectory_20260916/heldout_terrain/reference.csv --frames 174
./test_fusion_bag.sh clean --bag /home/yanfa/Env_X/InterFace/bags/20260825_232412 --frame-index /home/yanfa/P3/lidar_visual_fusion/test_data/multi_trajectory_20260916/next_day_route/index.json --reference /home/yanfa/P3/lidar_visual_fusion/test_data/multi_trajectory_20260916/next_day_route/reference.csv --frames 174
```

命令逐项顺序运行。short 默认仍会保留严格评分失败；如需上述近似诊断，在回放停止后对输出目录执行 `python3 scripts/assess_bag_run.py OUTPUT --reference test_data/multi_trajectory_20260916/short_low_contrast/reference.csv --reference-max-gap 0.5`，并明确记录这个评分选项。

结果：`results/multi_trajectory_20260916/comparison.json`、`comparison.png` 及各轮日志；原图/干扰对照见 [融合修复报告](fusion_repair_report.md)，审查与改进建议见 [项目审查](project_audit.md)。

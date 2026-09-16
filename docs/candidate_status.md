> **阶段记录**：本文保留当时的实现与测量。当前默认配置、修复结果和多轨迹对照以 [最终多轨迹报告](fusion_multitrajectory_report.md) 为准。

# 无 IMU 仿真与实机候选

## 当前默认方案

用户已选定 VoxelMap＋轻量视觉。Point-LIO 已退出启动、配置生成、构建和包发现；旧源码及基线只作历史记录。`start_adaptive_short.sh` 现在也是 VoxelMap 的兼容入口。

## 当前运行范围

用户已确认：当前仿真没有 IMU，实机会有 IMU。当前测试只使用实测双目图像和 LiDAR，不从参考轨迹生成虚拟 IMU。

| 分支 | 本目录状态 | 当前能验证的内容 |
|---|---|---|
| Point-LIO 无 IMU 模式 | 已编译、完成 short bag 基线 | 当前移植及常速度模型的表现，不能代表完整 LIO |
| VoxelMap 纯 LiDAR模式 | ROS2 适配已编译 | 瞬时仿真点云的概率平面配准、耗时、内存 |
| PV-LIO | 官方源码已冻结，尚无 ROS2 运行适配、未编译测试 | 实机有真实 IMU、同步和标定后再接入 |
| XFeat＋LighterGlue | 已在 Orin CUDA 上运行 short bag | 双目尺度、视觉里程计、时序与图像质量检查 |

PV-LIO 官方实现需要 IMU；其 README 中纯 LiDAR 常速度移植仍为待办。VoxelMap 官方仓库提供不依赖 IMU 的 LiDAR 实现，要求点云已去畸变。当前仿真使用已确认的瞬时点云假设，不能直接推到运动中的真实旋转雷达。

来源：[PV-LIO](https://github.com/HViktorTsoi/PV-LIO)、[VoxelMap](https://github.com/hku-mars/VoxelMap)。冻结提交见 `candidate_manifest.json`。

完整 short bag 已跑完：最终 RMSE 0.112 m、VoxelMap 自身 RMSE 0.168 m。详见 [本次实测报告](voxelmap_short_report.md)。

## 新的来源选择

`selected_bag_adaptive.yaml` 和 `selected_bag_voxelmap.yaml` 启用新逻辑；旧配置保留以便复现基线。

- 视觉：图像可用性、双目极线/视差、PnP 内点与空间覆盖、重投影、双目运动复核、时间预算与连续恢复。
- LiDAR：实际点面匹配数量、比例、残差，以及配准雅可比的平移/六维条件。状态协方差与当前扫描可观测性一起使用。
- 当前扫描退化时提高 LiDAR 位姿协方差并拒绝其进入 EKF；健康视觉可独立提供运动约束。
- 视觉坏而 LiDAR 好时，停止视觉输入、使用 LiDAR。两者独立检查均通过但相互冲突时报告冲突；两者都弱时报告降级并暂停新的合格位姿/地图更新。
- 恢复时将来源对齐到连续公共坐标系一次，此后累计该来源自身的运动，避免来源重置引入跳变。

新分支使用公共坐标系中的绝对位姿约束，并以协方差融合。旧分支仍是 LiDAR 绝对位姿＋视觉差分。来源重新对齐产生的高协方差锚点不计为有效视觉约束。

可观测性阈值是工程判据，未做概率标定，也不是完整的 X-ICP。当前按整个位姿来源保守启停，尚未实现逐方向融合。概率平面地图不能凭空补足平地缺少的水平几何约束。

## 启动与评估

```bash
cd /home/yanfa/P3/lidar_visual_fusion
./build.sh
./start_voxelmap_short.sh fusion
# 完成一次后，停止本目录运行，再选择其他分支：
./stop.sh
./start_adaptive_short.sh
```

`start_voxelmap_short.sh lidar` 为纯 LiDAR 诊断模式，未启用自适应拒绝，不能把其有输出等同于可靠定位。所有 short 入口使用完整 bag、原速、原始采集间隔、域 57，以及原 P3 的同一 RViz 界面。

新入口会在回放前启动独立接口观察器；默认观察 320 秒，报告为 `verification.json`。回放结束后可创建本次输出目录中的 `verifier.stop` 提前结算观察，随后运行：

```bash
source scripts/env.sh
python3 scripts/assess_bag_run.py results/对应运行
```

原项目另一路 RoMa/仿真仍可能使用同一 Orin，耗时反映共享机器当时负载。保持其进程不变，并在报告中记录这一条件。

## 有界内存

VoxelMap 定位内部地图默认保留当前位置 80 m 内、最多 10000 个根体素，淘汰时递归释放节点；默认每格最多 1000 个样本、最大层数 4。此为结构容量限制，不是严格的字节限额。

长期高程与三维地图仍使用本目录现有磁盘分块存储；定位局部地图淘汰不会删除已经持久化的历史地图。完整进程树含 RViz 的 RSS 单独记录，不能只报神经网络张量大小。

## 尚未验证

实机 IMU 融合与标定、真实斜向强光/曝光变化、重复纹理长序列、规划器端到端使用、30～60 分钟持续新增地图、全局回环。短 bag 的位姿评分不能证明高程精度或长期内存稳定。

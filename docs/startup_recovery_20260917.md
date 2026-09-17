# 启动定位接续、关键帧恢复与右侧路径显示

日期：2026-09-17。工程：`/home/yanfa/P3/lidar_visual_fusion`。

## 修复的问题

1. 初始视觉帧只定义坐标原点，协方差很大。旧 Guard 丢弃它后，在后续 LiDAR 连续失败时，正常视觉无法接到地图坐标。
2. 在线曾有 69 帧 `underexposed`。失配超过旧 30 s 上限后，恢复图像会进入新视觉坐标段；若没有同时间的可信地图参考，正式输出仍不可用。
3. 右侧显示了用户不再需要的全局规划线；路径之前默认将 `map` 与 `odom` 等同，也可能重新加载旧会话留下的缓存路线。

## 实现

### 启动原点与正式输出分开

新增内部 `/fusion/visual_epoch_origin`，类型 `nav_msgs/msg/Odometry`。视觉节点只在双目几何初始化成功时发布身份变换事件，保持原本的高协方差；它不是位姿或零速度观测。

Guard 验证时效、坐标段、`base_link`、身份变换和高协方差，只暂存有界的原点信息。有真正的后续跟踪和同时间的合格正式参考后才建立坐标关系。普通失败帧、无关新坐标段、被正式输出拒绝的 EKF 位姿均不能借此接回。

### 有界关键帧恢复

两个实时配置同步设置：

```yaml
learned_visual:
  max_recovery_gap_sec: 120.0
  recovery_keyframes: 4
visual_continuity:
  enabled: true
  max_gap: 120.0
```

- 当前、备用参考仍先尝试；失败后才搜索最多 4 个历史关键帧，按约 2 m 平移或 0.5 rad 转角保留不同视角。
- 120 s 从最后可信跟踪计算，不要求一直静止的有效参考每隔 120 s 强制替换。
- 长间隔或历史关键帧匹配要求至少 60 个 PnP 内点、比例 0.7、覆盖率 0.25、双目三维一致比例 0.8，并继续执行原有几何与运动检查。
- 第一次找到候选只报告 `recovery_keyframe_confirming`，不更新最后可信位姿。6 s 内第二次匹配到同一参考，候选位移不超过 0.5 m 且转角不超过 0.25 rad，才确认恢复；仍受原来的速度门限限制。
- 几何失败、不相容候选和超时不会把失败观测写成新参考。失败超过保留上限则重新初始化，不能假装旧坐标关系仍然有效。
- `output_discontinuity` 没有放宽。此功能不包含全局回环、位姿图优化或历史地图变形，也不保存跨程序重启的特征库。

诊断字段包括 `retained_recovery_keyframes`、`recovered_archived_keyframe`、`recovery_confirmations`、`reference_candidates_tried`，位于 `learned_metrics.json`／`.jsonl`。正式输出仍看 `fusion_status.json` 的 `localization_valid`、`visual_continuity.anchored` 和 `filtered_quality.reason`。

### 右侧路径

- 移除右侧 `/Car/T4/planning/global_route` 订阅与绘制，图例同步更新。
- 保留紫色局部规划线、橙色已行驶轨迹和人工目标操作。
- 局部路径按声明的坐标系与时间查 TF；拒绝混合点坐标系、非有限坐标、无效变换。缺 TF 可短暂等待，不能猜测偏移。
- 局部路径使用 reliable／volatile 接收，只显示窗口启动之后的新发布，避免把 P4 之前缓存的旧原点路线直接加载回来。
- P4 当前路径可能没有时间戳。若 P4 主动重新发布旧原点、却仍标记为 `map` 的路径，接收端无法可靠识别；仍需 P4 管理一致的任务和地图会话。没有修改 P4 或发控制命令。

## 测试结果

### 几何与 Guard

- 9 项关键帧测试通过：90 s 失配保留、正常缓慢运动、旧参考恢复、两次确认、不一致候选拒绝、坏深度拒绝、4 帧历史缓存上限、超时后新坐标段。
- 最终相关单元回归共 34 项通过，包含视觉连续性、双目几何与全局／局部显示来源回退。
- 启动后 LiDAR 失效的真实 ROS Guard 测试：原逻辑后续输出 0，新逻辑输出 19；失败视觉、新原点和被拒绝 EKF 均未误接纳。
- Guard 中插入 90 s 无效视觉与无效 LiDAR：期间没有正式输出，恢复后在同一视觉坐标段接回。该测试不控制车辆。

### 实际图像的完整回放

使用已停止录制的 `results/live_mapping_20260917_182203_368919/live_input_bag`，运行真实学习模型、VoxelMap、EKF、Guard 与 mapper，隔离 ROS domain 90。

| 场景 | 结果 |
|---|---|
| 原始前 30 帧 | 29 次视觉跟踪；27 个正式时刻和 27 次入图；定位有效；无输出跳变保护事件 |
| 前 20 帧，中间人为加入按传感器时间计的 90 s 黑屏和 LiDAR 中断 | 10 帧黑屏被拒绝；第一次候选不发布；第二次确认后接回原坐标段 |
| 干扰回放结束 | 18 次有效视觉跟踪；13 个正式时刻；12 次入图；恢复后 11 个正式输出；没有另起视觉原点 |
| 黑屏期间 | 正式输出 0；未使用保留位姿伪造观测 |

最大相邻正式位置变化约 0.54 mm，来自这段驻车记录，不是绝对精度或 ATE。初始化、时效和恢复确认会跳过部分输入，本次没有把 20 帧全部写成入图成功。

### 路径与地图接口

隔离 ROS domain 69 测试通过：右侧不保存全局路线；新窗口不接收旧保留路线；带 10 m、−4 m 平移及 90° 旋转的局部路径显示坐标正确；局部 64×64 m 高程、全局回退、目标只发一次及定位无效拒绝目标均通过。

该次为无窗口测试，未测桌面帧率。合成消息循环 P95 约 35 ms，测试进程 RSS 约 158.3→158.7 MiB；不代表完整工程内存或真实 RViz 帧率。测试目标只在隔离域发布，没有车辆控制。

## 安装和运行

Python 包重新构建通过，源码与实际安装文件的 manifest 校验通过。C++ 等待队列修复此前已编译，本轮未修改 C++ 求解器。

19:09 按用户要求重启自己的 P3 和显示窗口，新会话：

```text
results/live_mapping_20260917_190905_379384
```

配置为 `config/simulation_lidar_camera_fov.yaml`，关键帧 120 s／4 帧设置已复制进本次 `profile.yaml`。随后只刷新显示进程更新图例，未重建地图原点。当前状态以 `run_state.json` 为准。

初查时 capture 连接仍存在，但日志停在等待 UE 返回新数据；用户确认 UE 正在重启。

19:16 后复查时现场已有新会话 `results/live_mapping_20260917_191613_381094`，加载本轮配置并重新收到传感器数据。初期定位有效，之后累计 33 次入图又出现恢复停顿：视觉光度合格，但与旧参考的几何匹配不足，候选约 44～56 个 PnP 内点、内点比例约 0.46～0.52，未达到恢复要求；LiDAR 同时退化并缺少可靠运动参考。31 s 被动观察中无有效正式定位、地图修订号保持 33。

这次现场数据没有证明任意失联后都可恢复；没有临时放宽几何或输出跳变门限。该限制与受控黑屏回放的通过结果同时保留，不将它们合并成“在线全部通过”。之后超过 120 s，视觉可重新初始化并跟踪，但新坐标段仍需要合格参考；现场又观察到 EKF `position_uncertain`，不能把“重新跟踪”写成“正式定位已恢复”。本轮未控制车辆，也未再次重启这份现场会话。

## 证据与复现

远程 `results/startup_recovery_20260917/`：

- `baseline_report.json`、`fixed_report.json`：启动接续前后对照。
- `keyframe_unit.log`、`dark_guard_report.json`：关键帧与正式输出回归。
- `full_replay_v3/report.json`：30 帧原图完整回放。
- `dark_full_replay/report.json`：90 s 黑屏完整链路回放。
- `ui_candidate/results/interface_verification.json`：路径、地图与目标接口。
- `keyframe_python_build.log`、`keyframe_deployment_manifest.json`、`keyframe_restart.json`：部署与重启记录。
- `final_unit.log`：最终 34 项单元回归。
- `live_after_ue_restart.json`、`live_recovery_limit.json`：UE 恢复后被动检查及现场恢复失败记录。

复现几何和 Guard 测试：

```bash
cd /home/yanfa/P3/lidar_visual_fusion
source scripts/env.sh
python3 tests/test_visual_reference_recovery.py -v
ROS_DOMAIN_ID=89 ROS_LOCALHOST_ONLY=1 \
  CYCLONEDDS_URI=file://$PWD/config/cyclonedds.xml \
  python3 tests/test_startup_visual_origin_ros.py \
  --workspace "$PWD" --report /tmp/p3_origin_recovery_report.json --assert-dark-recovery
```

完整回放用 `tests/replay_startup_recovery.py`，必须使用已经停止录制的输入目录和新的结果目录：

```bash
ROS_DOMAIN_ID=90 ROS_LOCALHOST_ONLY=1 \
  python3 tests/replay_startup_recovery.py \
  --workspace "$PWD" \
  --run results/live_mapping_20260917_182203_368919 \
  --out results/keyframe_blackout_recheck \
  --profile config/simulation_lidar_camera_fov.yaml \
  --frames 20 --blackout-seconds 90 --assert-recovery
```

仅在资源允许时运行隔离回放；它不发布在线车辆控制，但仍会占用同一块 CPU/GPU。原始录包和大型运行结果不随 Git 同步。

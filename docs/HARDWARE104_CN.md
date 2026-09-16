# 104 车机：真实传感器建图

机器：`yanfa@192.168.100.104`。目录：`/home/yanfa/P3/lidar_visual_fusion`。

此配置使用真实 Ouster 雷达及其内置 IMU、Galaxy 双目，不需要 UE 或仿真 capture。安装位置沿用 2026-08-17 标定，用户已确认未改动。

## 启动

**雷达延迟优化（2026-09-16 晚间第二轮）**：修正匹配线程参数，104 配置使用 4 个线程，输入队列保留最新 1 帧。同一段 45 帧真实点云／IMU 的位姿、协方差结果一致；单帧处理中位数 0.449→0.311 秒。当前负载下两轮真实联调的局部地图数据年龄 4.04→3.28 秒，仍需继续降低；图像本身约 0.12 秒到达。详见 [雷达延迟对照](HARDWARE104_LIDAR_LATENCY_20260916.md)。

**后续建图优化（2026-09-16 晚间）**：同输入建图计算中位数 1.36→0.49 秒，已修复接收阻塞、双目队列和位姿置信度时间匹配，并增加坏帧检查。48 项相关回归通过；当前负载下全局约 2 秒发布一次。实机双目新增栅格仍受正式位姿不确定性限制，尚未通过最终验收，详见 [建图优化与运行边界](HARDWARE104_MAPPING_20260916.md)。

**最新交付状态（2026-09-16）**：已迁移 216 的视觉参考恢复、融合约束保留和雷达邻格匹配修复，保留真实 IMU 补偿与双目稀疏补图。75 秒真实 GPU 联调通过：149/149 帧 LiDAR＋IMU 配准有效，融合入口接受 288 次视觉运动约束，地图持续发布。新版雷达处理更慢，动态精度仍待验证；对比数据和限制见 [本轮迁移记录](HARDWARE104_RECOVERY_20260916.md)，早期结果见 [原验证记录](HARDWARE104_VERIFICATION.md)。

**P4 联调注意**：复用下方脚本的小图驱动和 `config/hardware104.yaml`。重启前的 P4 会话使用旧相机驱动及原图话题，左右时间戳相差约 58 ms，双目前端没有同步成功；不要覆盖回该配置。

测试期间原有 GLIM 使用约 45～47 GiB 内存，最终于 18:20:52 被系统内存不足机制结束；本任务未停止它。18:22 检查时可用内存恢复约 52 GiB，旧视觉测试进程仍处于系统等待状态。启动脚本要求至少 **4 GiB 可用内存**，不足时会明确退出，不会替用户停止其他程序。此值是根据原模块连同窗口约 2.7 GiB 的实测峰值留出的启动余量，长期大地图仍需监控内存。

先让车静止，等待 IMU 初始化完成，再由操作人员控制车辆。

```bash
cd /home/yanfa/P3/lidar_visual_fusion
./hardware_sensors.sh start
./start_hardware.sh --rviz
```

第一个脚本复用已启动的雷达、IMU、触发器，只补开缺失的驱动。第二个启动定位、建图和组合 RViz 窗口。通过 SSH 启动时，窗口显示在车机已有桌面上。仅建图可省略 `--rviz`。

```bash
./status.sh                       # 定位、建图、界面进程
./hardware_sensors.sh status       # 本脚本启动的传感器进程
./save_map.sh
./stop.sh                         # 停止本工程的建图与界面
./hardware_sensors.sh stop         # 仅停止本脚本启动的传感器
```

这些脚本不发送车速或导航目标。已有其他程序启动的驱动不会被停止。重启建图会创建新地图坐标原点，P4 应在车停稳后重新对齐。

## ROS 域与接口

传感器沿用 **ROS_DOMAIN_ID=19**。本实机建图默认 **59**，转接自动启动；与原有 GLIM 的 TF、216 上的仿真域 57 分开。P4 联调应使用域 59，或在明确停用同域的其他定位器后统一设置。不要同时在同一域发布两套 `map/odom/base_link`。

| 用途 | 话题 | 类型 |
|---|---|---|
| 原始雷达 | `/Car/T5/OS1/points` | `sensor_msgs/msg/PointCloud2` |
| 真实 IMU | `/Car/T5/OS1/imu` | `sensor_msgs/msg/Imu` |
| 左／右小图 | `/Car/T5/Cam_Left/image_mono/mapping`、`/Car/T5/Cam_Right/image_mono/mapping` | `sensor_msgs/msg/Image` |
| 位姿 | `/T3/semantic/current_pose`、`/Car/T3/localization/odometry` | `nav_msgs/msg/Odometry` |
| 轨迹 | `/T3/semantic/trajectory` | `nav_msgs/msg/Path` |
| 规划局部高程栅格 | `/Car/T3/mapping/grid_map` | `grid_map_msgs/msg/GridMap` |
| 显示全局高程栅格 | `/Car/T3/mapping/global_grid_map` | `grid_map_msgs/msg/GridMap` |
| 全局点云 | `/T3/mapping/lidar_map` | `sensor_msgs/msg/PointCloud2` |
| 双目稀疏观测／已确认补图点 | `/fusion/stereo_points`、`/T3/mapping/stereo_map` | `sensor_msgs/msg/PointCloud2` |
| 增量地图 | `/T3/semantic/incremental_map` | `t3_interfaces/msg/IncrementalSemanticMap` |
| 惯性／时间／融合状态 | `/fusion/imu_status`、`/fusion/hardware_status`、`/fusion/status` | `std_msgs/msg/String` |

局部地图配置周期 1 秒，全局 2 秒，有新内容才发布；实际周期受计算量影响。规划局部地图仍为 64×64 米，右侧界面显示车周围 32×32 米，左侧显示全局点云。P4 目标、路线显示沿用原窗口；本次部署不启动 P4 或自动行驶。

原始输入连通性检查：

```bash
source scripts/env.sh
export ROS_LOCALHOST_ONLY=0
export CYCLONEDDS_URI=file:///home/yanfa/program/cyclonedds.xml
python3 scripts/check_hardware.py --seconds 5
```

## 实机适配

- **双目小图**：私有相机驱动将 2448×2048 RGB 全视野转换为 640×536 `mono8` 再发布。每张有效载荷约 0.34 MB，原始 RGB 约 15.04 MB；这里减少的是 ROS 图像传输，传感器到驱动的原始 GigE 图像仍按原尺寸采集。算法按输出尺寸缩放已标定的内参，保留畸变与双目外参，不采用驱动中的占位 CameraInfo。
- **时间**：雷达和内置 IMU 原报文为设备运行时间，相机为系统时间。一个小型 C++ 接收节点读取本机驱动的 DDS 发布时间，用它估计一次固定偏移，同时应用于雷达和 IMU；避免 Python 排队延迟污染时钟估计，保留传感器相对间隔。它不是硬件同步，动态精度仍须实测。时钟重置或持续漂移会终止本轮，避免给已有地图偷偷换时间基准。
- **IMU**：使用 Ouster 厂家 `imu_to_sensor_transform`，毫米转米；输入点云已经是 `sensor_frame`，不重复套用 `lidar_to_sensor_transform`。WIT IMU 原接口保留，本配置不混合两路不同 IMU。
- **扫描补偿**：将每点 `t` 纳秒转换为相对扫描起点的秒数，用覆盖整圈的 IMU 轨迹补偿到扫描末端。配准和建图使用同一补偿点云与末端时间。IMU 不足时拒绝未经补偿的实机扫描；数据重新连续后恢复补偿，断流恢复故障注入已通过。仿真瞬时点云配置仍支持无 IMU。
- **融合**：已合入 216 的稳定提交 `2faf6fa`／`a7c4577`，支持合格视觉接续正式位姿和地图更新。IMU 正常时仍用惯性预测和扫描补偿，不让视觉覆盖雷达初值。无 IMU 分支的视觉辅助子图恢复实现也保留；当前原始旋转雷达必须有可用 IMU 才能补偿入图。带 IMU 的子图重建尚未单独扩展验证。失配、跳变和不确定性检查保持开启，有真实 IMU 不代表保证不会漂移。
- **资源**：图像接收与雷达/IMU 接收分进程；点云使用 ROS 字节数组快速传输。保留有限输入队列、地图分块缓存、显示点数上限及进程内存监视器。私有依赖位于 `deps/`，不替换其他项目环境。

本轮未进一步减少点云：一帧约 11.2～11.3 万有效点，仍按 0.2 米降采样后约 2.5 万点参与配准。0.55 秒降至约 0.02 秒是同一输入下**点云预处理**耗时的改善，不是完整定位耗时；最新完整 LiDAR 后端单帧中位数约 0.32 秒。

**双目三维点入图**：104 配置已启用 `stereo_mapping.enabled`。复用左右匹配和时序几何检查通过的稀疏点，每秒最多 2 次、每帧最多 512 点；不增加深度网络。恢复到原始左相机光学坐标系后，使用正式融合位姿变换到地图。光照／视觉质量不合格、位姿过期或协方差过大时暂停补图。

默认限制深度 0.5～8 米，单点几何标准差不超过 0.15 米，加入位姿误差后高度标准差不超过 0.25 米。连续两帧确认同一格后，按不确定性加权填入未观测区域；保留单次观测误差下界，避免反复观测制造虚假的高置信度。已有 LiDAR／ToF 栅格不被视觉覆盖，后续测距数据会替换视觉补充。

局部／全局高程图、增量消息和地图保存包含这些补图格；`/T3/mapping/lidar_map` 保持纯雷达，组合窗口叠加 `/T3/mapping/stereo_map`，其预览上限 2 万点，待确认缓存上限 4096 格。`/fusion/map_status` 的 `stereo_reason` 解释等待确认、位姿不合格或已有雷达观测等状态。

有可信全局位姿且满足入图不确定性阈值时，可以在 LiDAR 点云中断后继续由双目新增稀疏地图；无法建立视觉参考或位姿误差增大时会暂停。它不是稠密双目重建，未观测空隙仍为未知，不能据此保证纯视觉长期导航。稀疏点不提供完整垂直表面包络；高差障碍依靠相邻已确认栅格判断。粗分辨率 `global_overview` 仍只包含测距输入，详细栅格及补图点才包含双目补充。

相机驱动缺少双目帧或时间戳异常时会丢弃不成对的观测。若已有旧相机驱动占用设备、没有小图话题，启动脚本会报告冲突；先在原启动终端停止它，再启动这里的驱动。

## 标定文件与构建

配置：`config/hardware104.yaml`。原始来源及 SHA256 记录在 `calibration_provenance`：

- 车体／双目：原 P3 的 `workspace/src/t3_semantic_mapping/config/t5_rgb10_rgb11.yaml`，对应 `/home/yanfa/program/transform.xlsx`。
- 雷达／内置 IMU：`/home/yanfa/program/Lidar2/src/ouster-ros/ouster-ros/config/192.168.19-metadata.json`。

静态几何检查支持当前 FRD→FLU 转换：416 个双目三维点与雷达的最近距离中位数为 0.153 米，反向朝向候选为 0.788 米，Z 轴未转换为 1.811 米。两个输入来自驻车场景的不同采样，这只是轴向合理性证据，直行／转弯联合核验与时间标定仍待完成。ToF 已找到原标定，本轮暂不加入地图。

本机 Ubuntu 22.04 / ROS 2 Humble / ARM64 / CUDA 12.6。代码改动后的构建：

```bash
./build.sh
bash scripts/build_hardware_camera.sh  # 相机驱动改动后
bash scripts/build_visual_window.sh    # 组合窗口改动后
```

运行结果在 `results/run_日期_时间/`；部署检查在 `results/hardware104_deployment/`。短时静态连通性检查不能代替行驶轨迹的 ATE 测试。

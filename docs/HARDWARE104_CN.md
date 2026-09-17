# 104 车机：真实传感器建图

机器：`yanfa@192.168.100.104`。目录：`/home/yanfa/P3/lidar_visual_fusion`。

此配置使用真实 Ouster 雷达及其内置 IMU、Galaxy 双目，不需要 UE 或仿真 capture。安装位置沿用 2026-08-17 标定，用户已确认未改动。

## 启动

**最新高程融合（2026-09-17）**：参考 237 的近地表筛选、216 的上表面与共同误差处理，按帧融合重复扫描，连续确认高度变化。默认只发布高程及质量层，规划局部图统一为 32×32 米。详细行为及验证见 [高程融合说明](HARDWARE104_HEIGHT_FUSION_20260917.md)。此前的点云缓存、时间对齐和独立双目队列优化继续保留，历史结果见 [地图延迟优化](HARDWARE104_PIPELINE_LATENCY_20260916.md)。

此前已迁移视觉参考恢复、融合约束保留与雷达邻格匹配修复，并优化雷达线程、接收队列和建图异常处理。历史结果见 [雷达延迟对照](HARDWARE104_LIDAR_LATENCY_20260916.md)、[建图运行边界](HARDWARE104_MAPPING_20260916.md)、[216 迁移记录](HARDWARE104_RECOVERY_20260916.md) 和 [早期验证](HARDWARE104_VERIFICATION.md)。

**P4 联调注意**：复用下方脚本的小图驱动和 `config/hardware104.yaml`。重启前的 P4 会话使用旧相机驱动及原图话题，左右时间戳相差约 58 ms，双目前端没有同步成功；不要覆盖回该配置。

较早测试中，GLIM 曾占用约 45～47 GiB，并于 18:20:52 被系统内存不足机制结束；本任务未停止它。晚间已有 GLIM 再次运行，本轮继续保留。启动脚本要求至少 **4 GiB 可用内存**，不足时会明确退出，不会替用户停止其他程序。此值是根据原模块连同窗口约 2.7 GiB 的实测峰值留出的启动余量，长期大地图仍需监控内存。

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
| 建图及过滤状态 | `/fusion/map_status` | `std_msgs/msg/String` |
| 惯性／时间／融合状态 | `/fusion/imu_status`、`/fusion/hardware_status`、`/fusion/status` | `std_msgs/msg/String` |

局部地图配置周期 1 秒，全局 2 秒，有新内容才发布；实际周期受计算量影响。规划局部地图和右侧近车窗口均为 32×32 米，左侧保留全局点云。GridMap 包含 `elevation`、`elevation_variance`、`height_range`、`roughness`、`observation_count`，未知高度仍为 NaN。当前实机配置不发布占用/障碍分类、语义增量及分类概览；P4 使用高程自行判断通行代价。原窗口仍支持在已知高程格上选目标，并显示 P4 路径；本次部署不启动 P4 或自动行驶。

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

**高程只用雷达**：按用户最新要求，`stereo_mapping.enabled: false`。视觉仍用于定位，但不再转换、发布用于建图的双目点云，建图端也不订阅双目补图点、不发布双目地图；LiDAR 中断时不新增高程。视觉定位本身仍需要双目几何计算。

**自车过滤**：`self_filter` 在标定后的 `base_link` 坐标中、入图前生效。范围 X 为 −1.47～0.53 米、Y 为 −1～1 米，即相机后方约 2×2 米；Z 为 0.13～1.03 米，相机下方约 0.7 米加上方 0.2 米余量。这个较大的范围来自用户估计与记录核对，不是精确车体模型，范围内的真实物体也会被过滤。过滤同时作用于点云图与高程融合，保留范围外的观测；计数见 `/fusion/map_status` 的 `self_filter.sources.lidar`。

正常启动脚本创建新地图后生效。旧地图里的车身点不自动清除；未观测的车底区域仍是未知，不会强行填成可行驶地面。定位端的原始点云、雷达配准和视觉定位设置保持原样。

2026-09-17 离线验证：146 帧旧记录中过滤 2,973 个范围内点，范围外点保持保留，过滤计算中位数 4.15 ms/帧。同一记录前六帧、固定测试位姿的建图对照中，盒内点云体素从 28 降为 0，61,623 个不与过滤盒相交的体素完全一致；建图中位数 301.3 → 306.6 ms。48 项回归及独立 ROS 几何/空帧/逐点时间检查通过。这不是实车行驶或规划成功率测试。测试结果位于 104 候选目录 `lidar_visual_fusion_hw_self_filter_20260917/results/hardware104_self_filter_20260917/`。正式建图保持暂停，237 显示未启动。

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

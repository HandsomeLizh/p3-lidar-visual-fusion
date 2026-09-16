# 104 车机：真实传感器建图

机器：`yanfa@192.168.100.104`。目录：`/home/yanfa/P3/lidar_visual_fusion`。

此配置使用真实 Ouster 雷达及其内置 IMU、Galaxy 双目，不需要 UE 或仿真 capture。安装位置沿用 2026-08-17 标定，用户已确认未改动。

## 启动

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
- **时间**：雷达和内置 IMU 原报文为设备运行时间，相机为系统时间。使用内置 IMU 的接收时间估计一次固定偏移，同时应用于雷达和 IMU；保留传感器相对间隔。它不是硬件同步，动态精度仍须实测。时钟重置或持续漂移会终止本轮，避免给已有地图偷偷换时间基准。
- **IMU**：使用 Ouster 厂家 `imu_to_sensor_transform`，毫米转米；输入点云已经是 `sensor_frame`，不重复套用 `lidar_to_sensor_transform`。WIT IMU 原接口保留，本配置不混合两路不同 IMU。
- **扫描补偿**：将每点 `t` 纳秒转换为相对扫描起点的秒数，用覆盖整圈的 IMU 轨迹补偿到扫描末端。配准和建图使用同一补偿点云与末端时间。IMU 不足时拒绝未经补偿的实机扫描；仿真瞬时点云配置仍支持无 IMU。
- **融合**：本分支先部署已有融合基线。IMU 负责惯性预测，视觉继续提供独立运动约束。216 上另一 agent 正在修改的视觉连续定位与子地图恢复，待其验证后再合并；此处未冒充已合入。保留失配、跳变和不确定性检查，有真实 IMU 不代表保证不会漂移。
- **资源**：图像接收与雷达/IMU 接收分进程；点云使用 ROS 字节数组快速传输。保留有限输入队列、地图分块缓存、显示点数上限及进程内存监视器。私有依赖位于 `deps/`，不替换其他项目环境。

相机驱动缺少双目帧或时间戳异常时会丢弃不成对的观测。若已有旧相机驱动占用设备、没有小图话题，启动脚本会报告冲突；先在原启动终端停止它，再启动这里的驱动。

## 标定文件与构建

配置：`config/hardware104.yaml`。原始来源及 SHA256 记录在 `calibration_provenance`：

- 车体／双目：原 P3 的 `workspace/src/t3_semantic_mapping/config/t5_rgb10_rgb11.yaml`，对应 `/home/yanfa/program/transform.xlsx`。
- 雷达／内置 IMU：`/home/yanfa/program/Lidar2/src/ouster-ros/ouster-ros/config/192.168.19-metadata.json`。

安装未变的确认和坐标轴验证是两件事：当前保留 FRD→FLU 转换的核验状态，需结合实际投影、直行及转弯记录完成动态核验。ToF 已找到原标定，本轮暂不加入地图。

本机 Ubuntu 22.04 / ROS 2 Humble / ARM64 / CUDA 12.6。代码改动后的构建：

```bash
./build.sh
bash scripts/build_hardware_camera.sh  # 相机驱动改动后
bash scripts/build_visual_window.sh    # 组合窗口改动后
```

运行结果在 `results/run_日期_时间/`；部署检查在 `results/hardware104_deployment/`。短时静态连通性检查不能代替行驶轨迹的 ATE 测试。

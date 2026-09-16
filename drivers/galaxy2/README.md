# Galaxy2 — 大恒双相机 C++ 采集节点

基于 **大恒图像官方 Galaxy SDK (`libgxiapi`)** 的 ROS2 双 GigE 彩色相机采集节点，
用 C++ 从零实现，不依赖旧的 Aravis/Python `DahengCam` 工程。

## 相机

| 项目 | 值 |
|------|------|
| 型号 | MER2-503-23GC-P (Daheng Imaging) |
| 分辨率 | 2448 × 2048 (5MP) |
| 接口 | GigE Vision |
| 像素格式 | BayerRG8 → RGB8 (SDK `DxRaw8toRGB24`) |
| cam1 IP | 192.168.19.10 |
| cam2 IP | 192.168.19.11 |

## 话题 (默认命名空间 `/Car/galaxy`)

| 话题 | 类型 | 说明 |
|------|------|------|
| `/Car/galaxy/Cam_Left/image_raw` | `sensor_msgs/Image` | BayerRG8 原始 (`bayer_rggr8`) |
| `/Car/galaxy/Cam_Left/image_raw/color` | `sensor_msgs/Image` | RGB8 彩色 (`rgb8`) |
| `/Car/galaxy/Cam_Left/camera_info` | `sensor_msgs/CameraInfo` | 占位内参 (首帧后发布) |
| `/Car/galaxy/Cam_Right/...` | 同上 | 第二台相机 |

> 兼容旧 Aravis 话题命名：启动时 `NAMESPACE=daheng ./start_galaxy2.sh`。

## 目录结构

```
Galaxy2/                          # colcon 工作空间
├── start_galaxy2.sh              # 一键启动
├── stop_galaxy2.sh               # 停止
└── src/galaxy2/                  # ROS2 包
    ├── package.xml
    ├── CMakeLists.txt
    ├── include/galaxy2/galaxy_camera.h   # Galaxy SDK 封装
    ├── src/galaxy_camera.cpp             # 实现 (开设备/采集/解Bayer)
    ├── src/galaxy_node.cpp               # ROS2 节点
    ├── launch/galaxy2_dual.launch.py
    └── config/camera_info_mer2_503.yaml  # 标定参考 (占位)
```

## 编译

```bash
cd ~/program/Galaxy2
source /opt/ros/humble/setup.bash
colcon build --packages-select galaxy2
```

## 运行

```bash
./start_galaxy2.sh                 # 默认参数
./start_galaxy2.sh --check         # 仅检查环境
NAMESPACE=daheng ./start_galaxy2.sh   # 用旧话题命名空间
./stop_galaxy2.sh
```

## 实现要点

- **打开方式**：`GXOpenDevice` + `GX_OPEN_IP`，按相机 IP 直接打开（SDK 自带传输层
  发现，无需手动配置 `GENICAM_GENTL64_PATH`）。
- **采集**：`GXStreamOn` + 独立线程 `GXDQBuf`/`GXQBuf` 循环取帧。
- **彩色解码**：`DxRaw8toRGB24`（BayerRG8 → RGB24，邻域插值）。
- **包大小**：`GXGetOptimalPacketSize` 自动适配链路 MTU。
- **MTU 9000**：建议在 `eno1` 上启用 Jumbo Frame 以提高帧率
  (`sudo ./set_mtu9000.sh` 位于旧工程目录)。

## 采集模式 (触发)

通过 `trigger_mode` 参数切换采集方式，支持三种模式：

| 模式 | 说明 | 参数 |
|------|------|------|
| `continuous` | 自由采集（默认），按 `frame_rate` 持续输出 | 无需额外参数 |
| `external` | 硬件外部触发，每个触发边沿采集一帧 | `trigger_source`, `trigger_activation` |
| `software` | 软件触发，通过 ROS2 服务按需采集 | ROS2 服务 `~/trigger` |

### 外部触发使用

```bash
# Line0 上升沿触发
ros2 launch galaxy2 galaxy2_dual.launch.py \
  trigger_mode:=external trigger_source:=Line0 trigger_activation:=RisingEdge

# Line1 下降沿触发
ros2 launch galaxy2 galaxy2_dual.launch.py \
  trigger_mode:=external trigger_source:=Line1 trigger_activation:=FallingEdge
```

**触发源说明**：
- `Line0` / `Line1` / `Line2` — 相机 GPIO 输入线，具体可用线路请参考相机型号手册
- `trigger_activation` — `RisingEdge` (上升沿) 或 `FallingEdge` (下降沿)

### 软件触发使用

```bash
# 启动为软件触发模式
ros2 launch galaxy2 galaxy2_dual.launch.py trigger_mode:=software

# 按需采集一帧（触发两台相机同时采集）
ros2 service call /galaxy/trigger std_srvs/srv/Trigger
```

### 连续采集（默认）

```bash
ros2 launch galaxy2 galaxy2_dual.launch.py  # trigger_mode 默认为 continuous
```

### 触发模式与帧率

- 外部触发/软件触发模式下，`frame_rate` 参数不控制实际输出帧率，实际帧率由触发信号频率决定
- `exposure_time` 仍有效，需确保曝光时间 < 触发信号周期
- 采集线程在触发模式下使用 5s 超时等待帧，避免频繁空转

# 环境与依赖

## 已验证环境

本模块在 Ubuntu 22.04／ROS 2 Humble 的 P3 远程机器上编译和测试。默认学习视觉配置使用 CUDA；需要与机器驱动匹配的 PyTorch。仓库不包含 UE、原 P3 主工程、模型权重、bag 或构建目录。

## 新目录准备

在已安装 ROS 2 Humble、编译工具、PCL、OpenCV、Eigen、Ceres、Python NumPy/PyYAML、兼容 PyTorch 的 Linux 环境中：

```bash
git clone https://github.com/HandsomeLizh/p3-lidar-visual-fusion.git lidar_visual_fusion
cd lidar_visual_fusion
./scripts/fetch_sources.sh
/usr/bin/python3 scripts/prepare_learned_assets.py
./build.sh
```

`fetch_sources.sh` 下载固定版本的 VINS-Fusion 和 VoxelMap，应用随仓库保存的 VINS 补丁。`prepare_learned_assets.py` 下载固定版本的 XFeat、LightGlue、模型与 Kornia 依赖，不升级现有 PyTorch 或 NumPy。模型加载前校验权重摘要。

ROS 依赖包括 `robot_localization`、`cv_bridge`、`pcl_ros`、`image_transport`、`tf2_ros`、`rmw_cyclonedds_cpp` 和 RViz；标准 GridMap 与 T3 消息源码已包含。缺少依赖时根据构建提示安装对应 Humble 包。`scripts/install_local_deps.py` 是原 P3 私有依赖目录的维护工具，需要原机器的 apt 源配置，不作为全新机器安装入口。

Ceres 配置优先查找本目录 `deps/root/usr` 中的原 P3 依赖，再查系统安装。当前远程机器已完成构建；全新机器的端到端部署仍需验证。

## 原 P3 集成依赖

`start_simulation_sources.sh` 调用原 P3 的 `envx_runtime/start_capture.sh`、`start_car.sh` 和 `start_gui.sh`。默认原项目位于同级 `roma_t3_algorithm_bundle_20260825` 目录；可用 `P3_RUNTIME_ROOT` 指向另一处 `envx_runtime`。

定制可视化通过 `scripts/p3_visuals.sh` 和 `scripts/p3_visual_monitor.py` 复用原 P3 的 `integration_demo`、RViz 插件及已编译窗口；当前默认位置是 `/home/yanfa/P3/roma_t3_algorithm_bundle_20260825`。更换机器时需部署该外部工程并调整这两个文件中的路径。算法源码及标准 ROS 输出在本仓库内，原 P3 控制与可视化工程不重复打包。

原机器上的 `start_short_bag.sh`、`start_long_bag.sh` 和测试套件包含本地数据路径。自己的 bag 可使用：

```bash
./start_bag.sh /path/to/bag --profile config/selected_bag_voxelmap.yaml
```

运行前按真实输入修改标定、图像尺寸、消息话题、点云坐标和时间字段。未部署定制 P3 可视化时先不加 `--rviz`，可另开标准 RViz 订阅地图与轨迹。**仿真标定不能直接用于实车。**

## 测试与报告

纯算法测试位于 `tests/`，含 ROS 的集成测试需要构建并加载 `scripts/env.sh`。部分历史回归依赖未随仓库上传的 bag／`test_data`。文字结果在 `docs/`；报告里的 `results/` 截图及原始测量路径指向远程留存数据，不是仓库内附件。

当前默认不使用 Point-LIO；已退役代码和二进制不随此仓库发布。

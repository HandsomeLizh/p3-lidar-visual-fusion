# 定位跳变回归（2026-09-17）

测试只在 `ROS_LOCALHOST_ONLY=1` 的独立 ROS domain 中运行，不发送车辆控制或目标点。

- `replay_visual_motion.py`：从已有 bag 的左右图像计算独立视觉增量；不发布 ROS 数据。
- `replay_voxel_jump.py`：原始 LiDAR 和独立视觉增量送入指定的 VoxelMap 可执行文件。检查未收敛结果不被接纳、失败帧不污染配准地图、末段异常位移消失以及后续恢复。
- `replay_fusion_guard.py`：把上一步估计和原始图像、点云送入实际 EKF、Guard、高程 mapper，检查正式定位连续性。该测试不是在线延迟或绝对精度测量。
- `test_rejected_lidar_anchor_ros.py`：正式输出拒绝 LiDAR/EKF 跳变后，视觉不得跟着未经认可的参考重新对齐。
- `test_degenerate_projection.cpp`：平地及倾斜平面的弱方向约束，保留可观测高度、姿态，结构充分时不裁减更新方向。
- `test_independent_visual_motion.cpp`：视觉相对坐标消除、LiDAR 安装偏移、不同视觉 epoch、无效视觉、缓存上限和初始化。
- `test_tracking_guard.cpp`：按车体中心检查运动；原地旋转时考虑前置 LiDAR 的安装偏移。
- `test_vehicle_returns_ros.py`：车身候选回波只显示当前帧，不进入定位输入、持久地图或 P4 高程图。

正式输出的跳变保护保持启用。未经标定的仿真速度不作为真实 IMU。没有可靠运动参考时，弱几何配准保持不可用；停发不能解释成已经确认车辆静止。

测试报告需同时检查：正式位姿是否持续输出、地图最后更新的采集时间、拒绝原因及不确定性。累计帧数增加或匹配率很高，不能单独证明定位正确。

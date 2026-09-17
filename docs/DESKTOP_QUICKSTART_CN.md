# P3 融合建图使用说明（216 / Orin）

更新：2026-09-17。项目：`/home/yanfa/P3/lidar_visual_fusion`。

**日常只需要两个终端：一个启动采集和人工控制，一个启动建图和显示。UE 需先运行。** 窗口显示在 216 的桌面，SSH 终端不显示图形窗口。

本文命令采用当前使用的模式：**视觉＋完整 LiDAR 共同定位，视觉范围内的 LiDAR 点云建立高程图**。仿真速度反馈、IMU 和 ToF 当前不参与融合或补图；实车标准 IMU 接口仍保留。

## 1. 启动采集和车辆控制

在 216 的终端 A 执行：

```bash
cd /home/yanfa/P3/lidar_visual_fusion
./start_simulation_sources.sh
```

- 启动 capture、车辆控制桥接和人工控制窗口，复用配置一致的已有节点。
- 脚本不会自动开车；由人在控制窗口操作。
- 保持终端打开。若三项已经运行，脚本检查后退出是正常的。
- UE 地址为 `192.168.10.22`；采集端口 6665，控制与反馈端口 6668。

## 2. 启动建图和显示

在终端 B 执行：

```bash
cd /home/yanfa/P3/lidar_visual_fusion
./start_live.sh --profile config/simulation_lidar_camera_fov.yaml
```

等待图像、有效定位和地图出现，再人工驾驶。已有 capture 和控制时，只启动这一步即可。

需要同时录制输入，便于之后排查问题时，改用下面这一条（不要与上一条同时运行）：

```bash
./start_live.sh --profile config/simulation_lidar_camera_fov.yaml --record-input
```

录制会增加磁盘占用。**不要省略 `--profile ...` 来启动当前模式**：不带参数的 `./start_live.sh` 默认是仅双目点云建图。

## 3. 窗口怎么看，怎样接 P4

| 位置 | 内容 |
|---|---|
| 左侧 | 累计点云预览，Points 样式、2 像素 |
| 右侧 | 按高度着色的高程栅格，默认跟车显示 32×32 米 |
| “全图” / “跟车 32 m” | 切换累计地图和跟车视图 |
| 右侧紫线 / 橙线 | P4 局部路径 / 已行驶轨迹；左右两侧均不显示全局规划线 |

给 P4 的局部地图仍是 **64×64 米、0.2 米分辨率**；右侧显示范围不改变发布尺寸。P3 提供高程、方差和观测次数，未观测区域为 NaN；障碍和通行性由 P4 判断。

局部、全局地图发布目标周期均为 **2 秒**，接纳新数据还需要有效点云及对应时间的可信位姿。2 秒刷新不等于一定有新的观测。

**需要选点导航时**，由 P4 操作者确认地图与当前 P3 会话一致后，启动 P4：

```bash
bash /home/yanfa/P4/scripts/navigation.sh start
```

P4 就绪后，在右侧点“设置目标”，在已观测、可通行的栅格上按下并拖动指定朝向，松开发送；Esc 取消。发送目标可能让车辆开始行驶。仅启动 P3 不会自动导航。

## 4. 保存、结束、重启

运行中手动保存地图检查点：

```bash
cd /home/yanfa/P3/lidar_visual_fusion
./save_map.sh
```

结束时：**先在控制窗口停车；使用了 P4 时，由操作者先结束其导航任务；再在终端 B 按 Ctrl+C**。等待本次地图保存、建图和显示关闭。

必要时可用 `./stop.sh` 停止本套建图，但它不是车辆停车命令。终端 A 可继续保留；在 A 按 Ctrl+C 只停止该脚本新启动的节点，复用节点仍由原入口管理。

**重启顺序：停车 → P4 操作者清理旧任务和地图 → 停 P3 → 按第 2 节重新启动 P3 → P4 重新对齐。** P3 重启会建立新的地图原点，不能让 P4 继续沿用旧地图坐标。

地图完全停止后，可导出点云和高程：

```bash
./export_map.sh results/latest
```

结果位于 `results/latest/` 指向的运行目录。主要文件为 `lidar_map.pcd`（导出的点云）、`global_grid_map.sqlite3`（高程数据）、`trajectory_map.tum`（轨迹）。新会话会改变 `latest` 指向；导出旧会话时请填写其实际目录。

## 5. 没数据或定位不可用时

```bash
cd /home/yanfa/P3/lidar_visual_fusion
./status.sh
./start_simulation_sources.sh --check
```

需要查看定位健康状态：

```bash
source scripts/env.sh
export ROS_DOMAIN_ID=57 ROS_LOCALHOST_ONLY=0
ros2 topic echo --once /fusion/status
```

- **没有图像、点云**：检查 UE 是否运行、capture 是否连接。采集请求上限 2 Hz，实际频率受 UE 和传输耗时影响。
- **“定位暂不可用”**：整体没有通过检查的正式位姿，不只表示 LiDAR 失败。视觉有匹配也可能尚未接回旧地图坐标。先停车，查看 `localization_valid` 和具体原因。
- **地图不再增加**：查看本次运行的 `fusion_status.json`、`map_statistics.json`，区分断流、定位无效和高程冲突。
- **右侧路径错位**：确认 P4 和 P3 使用同一地图会话；重新启动 P3 后，需要由 P4 操作者重新对齐。

常用接口：正式位姿 `/Car/T3/localization/odometry`；P4 局部高程 `/Car/T3/mapping/grid_map`；全局高程 `/T3/mapping/global_grid_map`；点云 `/T3/mapping/lidar_map`。建图域为 57，采集和控制域为 10。

## 6. 当前关键帧恢复怎么用

**随上述配置自动启用，不需要额外启动命令。** 系统保留当前、备用参考和最多 4 个历史关键帧；从最后一次可信跟踪起，最多保留 120 秒用于尝试恢复。

短暂黑暗或跟踪失败后，有共同可见内容且几何检查通过时，经过连续两次确认再接回原坐标段。恢复期间不会把旧位置伪装成新的有效定位。没有共同视野、恢复检查不过或超过保留时间，仍可能恢复失败。

**目前关键帧只在内存中；保存地图不等于保存可重定位的关键帧库。** 重启后加载关键帧、跨会话重定位、完整回环纠正历史轨迹与高程图，属于下一步功能。方案见桌面另一份《P3关键帧保存与回环_后续方案.md》。

完整文档：[GitHub 文档索引](https://github.com/HandsomeLizh/p3-lidar-visual-fusion/blob/main/docs/INDEX_CN.md)；改动和问题复盘：[项目演进记录](https://github.com/HandsomeLizh/p3-lidar-visual-fusion/blob/main/docs/PROJECT_EVOLUTION_CN.md)。

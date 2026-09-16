# 目录与常用入口

项目：`/home/yanfa/P3/lidar_visual_fusion`

## 日常只需这几个命令

| 任务 | 命令 |
|---|---|
| UE 已开，启动采集和人工控制窗口 | `./start_simulation_sources.sh` |
| 实时建图和 P3 窗口 | `./start_live.sh` |
| 开启 P4 选点导航 | `bash /home/yanfa/P4/scripts/navigation.sh start` |
| 查看当前运行 | `./status.sh` |
| 保存地图 | `./save_map.sh` |
| 停止自己的建图 | `./stop.sh` |
| 历史 bag | `./start_short_bag.sh fusion` / `./start_long_bag.sh` |

完整步骤见 [使用说明](QUICKSTART_CN.md)。根目录的其他启动脚本是不同算法和旧实验的兼容入口，日常使用以上命令即可。

同一设备有人使用时先协调，不要直接停止或重启。`./stop.sh` 只停建图，不是车辆停车命令；停车和 P4 结束顺序见使用说明。

## 文件放在哪里

| 目录／文件 | 内容 |
|---|---|
| `src/` | 自己的定位、融合、建图源码 |
| `config/` | 标定、运行参数和 RViz 设置 |
| `visual/` | 自己的显示界面代码 |
| `scripts/` | 启动管理、评估、整理等工具 |
| `tests/` | 回归测试 |
| `docs/` | 使用说明、接口和测试报告 |
| `results/latest/` | 最近一次启动的快捷入口；下一次启动会更新，停止后仍保留 |
| `results/history/日期/` | 已结束、未被程序或报告引用的旧测试结果 |
| `results/organization/` | 目录归档清单，记录每项原路径和新路径 |
| `run_state.json` | 当前运行的真实目录和进程信息 |
| `build/`、`install/`、`log/` | 编译产物、实际运行程序、构建日志 |
| `vendor/`、`deps/` | 算法和 ROS 依赖 |
| `test_data/` | 测试索引；原始 bag 仍放在原项目和采集目录 |

旧测试仅移动归档，不删除。报告或测试程序明确引用的目录保留原路径；正在使用或最近 10 分钟更新的目录也保留。归档清单可用于按原路径还原。归档不等于释放磁盘空间。

预览／执行后续整理：

```bash
python3 scripts/archive_old_results.py
python3 scripts/archive_old_results.py --apply
```

整理范围仅限本项目的 `results/`，不移动其他 P3/P4 项目、原始 bag、模型权重或运行依赖。

#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser();parser.add_argument('suite',type=Path);args=parser.parse_args()
    root=args.suite.resolve();comparison=json.loads((root/'comparison.json').read_text())
    cases={c['name']:c for c in comparison['cases']}
    labels={'repair_clean':'原修复段／正常光照','short_low_contrast':'short／低对比度',
            'next_day_route':'次日／平坦路线','heldout_terrain':'后续起伏段','repair_perturbed':'原修复段／图像干扰'}
    order=['repair_clean','short_low_contrast','next_day_route','heldout_terrain']
    table=['| 路线 | 评分区间路程 | LiDAR ATE | 视觉 ATE | 融合 ATE | 融合 ATE／路程 | 融合最大误差 | 入图帧 |',
           '|---|---:|---:|---:|---:|---:|---:|---:|']
    def error(case,method):
        value=case['methods'].get(method,{})
        return f"{100*value['ate_m']:.2f} cm" if value.get('full_scored_interval') else '未形成完整轨迹'
    for name in order:
        case=cases[name];fused=case['methods'].get('fused',{})
        if not case['passed']:
            table.append(f"| {labels[name]} | — | — | — | 未完成，见失败记录 | — | — | {case['map_statistics'].get('mapped_scans',0)}/{case['frames']} |");continue
        table.append(f"| {labels[name]} | {case['travel_m']:.2f} m | {error(case,'lidar')} | {error(case,'visual')} | {error(case,'fused')} | {fused['ate_over_travel_percent']:.4f}% | {100*fused['max_error_m']:.2f} cm | {case['map_statistics']['mapped_scans']}/{case['frames']} |")
    performance=['| 路线 | LiDAR 中位数 | 视觉中位数 | 最后一次同帧位姿中位数 | 峰值 RSS（含 RViz） |', '|---|---:|---:|---:|---:|']
    for name in order+['repair_perturbed']:
        case=cases[name]
        if not case.get('timing'):continue
        timing=case['timing']
        def ms(key):return f"{1000*timing[key]['p50']:.0f} ms" if timing.get(key) else '—'
        performance.append(f"| {labels[name]} | {ms('lidar')} | {ms('visual')} | {ms('last_same_stamp_pose_receipt_sec')} | {case['peak_process_tree_rss_mib']/1024:.2f} GiB |")
    fault_names={'blackout':'黑屏','overexposure':'过曝','diagonal_glare':'斜向强光','low_light':'低照度','grayscale_control':'灰度对照'}
    faults=['| 干扰 | 输入帧 | 视觉运动约束帧 | 可靠位姿帧 | 恢复后首次使用视觉的等待时间 |','|---|---:|---:|---:|---:|']
    for row in comparison.get('image_fault_blocks',[]):
        delay=row.get('recovery_sensor_seconds')
        text=f'{delay:.2f} s' if delay is not None else ('不适用' if row['kind']=='grayscale_control' else '下一干扰前未恢复')
        faults.append(f"| {fault_names[row['kind']]} | {row['input_frames']} | {row['visual_motion_frames']} | {row['qualified_pose_frames']} | {text} |")
    clean,fault=cases['repair_clean'],cases['repair_perturbed']
    if clean['passed'] and fault['passed']:
        fault_summary=(f"正常图像融合 ATE 为 **{100*clean['methods']['fused']['ate_m']:.2f} cm**，干扰图像为 **{100*fault['methods']['fused']['ate_m']:.2f} cm**。"
                       f"干扰回放中 **{fault['map_statistics']['mapped_scans']}/{fault['frames']} 帧入图**，丢帧 {fault['map_statistics']['dropped_scans']}，收尾待处理队列 {fault['map_statistics']['pending']}。"
                       "下表的恢复等待包含原 bag 每约 3～4 秒才有一帧的采样间隔，不是单帧计算耗时。")
    else:fault_summary='该对照未完整通过，保留逐例失败记录，不能报告完整成功。'
    release_path=root/'release_verification.json'
    if release_path.exists():
        release=json.loads(release_path.read_text())
        release_text=("多轨迹表对应每个结果目录保存的冻结版本。最终交付在该版本上追加了地图背压和按实际到达周期判断失联的处理，随后重新编译、核对安装代码，并做针对性 ROS 回归和短段完整回放；没有用新增的回归用例重新计算或覆盖上表。\n\n"
                      f"最终源码指纹：`{release['released_manifest']['source_sha256']}`。具体检查和冻结版本差异见 `results/body_motion_integrated_20260916/release_verification.json`。")
    else:release_text='最终交付检查尚未结束；当前结果对应各目录中的冻结代码和配置。'
    text=(ROOT/'test_data/fusion_multitrajectory_report.md.in').read_text()
    for key,value in dict(accuracy_table='\n'.join(table),performance_table='\n'.join(performance),fault_summary=fault_summary,
                          fault_table='\n'.join(faults),release_validation=release_text).items():
        text=text.replace('{{'+key+'}}',value)
    if '{{' in text:raise ValueError('Unfilled report template')
    (ROOT/'docs/fusion_multitrajectory_report.md').write_text(text)
    print('Report rendered',flush=True)


if __name__=='__main__':main()

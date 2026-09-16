#!/usr/bin/env python3
"""Finalize documentation and artifact evidence after completed remote checks."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path('/home/yanfa/P3/lidar_visual_fusion')
SUITE = ROOT / 'results/body_motion_integrated_20260916'
sys.path.insert(0, str(ROOT / 'scripts'))
import source_manifest
from control import alive


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


comparison = read_json(SUITE / 'comparison.json')
release = read_json(SUITE / 'release_verification.json')
completion = read_json(SUITE / 'suite_complete.json')
artifacts = read_json(SUITE / 'artifact_verification.json')
assert comparison['all_passed'] and completion['all_passed'] and release['passed']
cases = {c['name']: c for c in comparison['cases']}
assert sum(c['frames'] for c in cases.values()) == 754
assert all(c['map_statistics']['mapped_scans'] == c['frames'] and
           c['map_statistics']['pending'] == c['map_statistics']['dropped_scans'] == 0 and
           all(c['checks'].values()) for c in cases.values())
assert len(artifacts['exports']) == 5 and all(c['pcd_size_verified'] for c in artifacts['exports'])

labels = {'repair_clean': '原修复段／正常光照', 'short_low_contrast': 'short／低对比度',
          'next_day_route': '次日／平坦路线', 'heldout_terrain': '后续起伏段'}
table = ['| 路线 | 评分区间路程 | 融合 ATE | ATE／路程 |', '|---|---:|---:|---:|']
for name, label in labels.items():
    case = cases[name]
    fused = case['methods']['fused']
    table.append(f"| {label} | {case['travel_m']:.2f} m | {100*fused['ate_m']:.2f} cm | {fused['ate_over_travel_percent']:.4f}% |")
memory = [c['peak_process_tree_rss_mib'] / 1024 for c in cases.values()]
fault = cases['repair_perturbed']
text = (ROOT / 'test_data/README.body_motion_final.md.in').read_text(encoding='utf-8')
for key, value in {
    '__RESULT_TABLE__': '\n'.join(table),
    '__FAULT_ATE__': f"{100*fault['methods']['fused']['ate_m']:.2f}",
    '__FAULT_MAP__': f"{fault['map_statistics']['mapped_scans']}/{fault['frames']}",
    '__MEMORY_RANGE__': f'{min(memory):.2f}～{max(memory):.2f}',
}.items():
    text = text.replace(key, value)
assert '__RESULT_' not in text and '__FAULT_' not in text and '__MEMORY_' not in text
(ROOT / 'README.md').write_text(text, encoding='utf-8')

template_path = ROOT / 'test_data/fusion_multitrajectory_report.md.in'
template = template_path.read_text(encoding='utf-8')
view_section = '''## RViz 与保存的地图

每轮回放都使用 P3 原有 RViz 窗口。实时高程显示采用车辆附近的 64 米窗口，完整地图持续写入分块存储；窗口外没有显示不等于地图被删除。点云使用有上限的显示预览。

下面是回放结束后，从最终第 174 次地图更新重建的完整高程图和点云快照；这是已保存地图的查看界面，截图中的处理时间是录下的历史值。原修复段导出完整点云 735,877 点，RViz 最多显示约 50,000 点。

![P3 RViz：保存的完整点云、高程与轨迹](../results/body_motion_integrated_20260916/rviz_saved_complete.png)

可独立查看这份地图，查看结束后停止对应窗口：

```bash
cd /home/yanfa/P3/lidar_visual_fusion
source scripts/env.sh
python3 scripts/p3_comparison_view.py start --output results/body_motion_integrated_20260916/repair_clean/verified_view
python3 scripts/p3_comparison_view.py stop --output results/body_motion_integrated_20260916/repair_clean/verified_view
```

'''
if '## RViz 与保存的地图' not in template:
    assert '## 使用\n' in template
    template = template.replace('## 使用\n', view_section + '## 使用\n')
    template_path.write_text(template, encoding='utf-8')
view = SUITE / 'repair_clean/verified_view'
assert (view / 'rviz_saved_complete.png').stat().st_size > 10000
shutil.copy2(view / 'rviz_saved_complete.png', SUITE / 'rviz_saved_complete.png')
subprocess.run([sys.executable, str(ROOT / 'scripts/render_fusion_suite_report.py'), str(SUITE)], check=True)

notice = '> **阶段记录**：本文保留当时的实现与测量。当前默认配置、修复结果和多轨迹对照以 [最终多轨迹报告](fusion_multitrajectory_report.md) 为准。\n\n'
historical = ['fusion_repair_report.md', 'multi_trajectory_report.md', 'implementation_status.md',
              '实现状态.md', 'short_bag_test_20260915.md', 'candidate_status.md', '轻量视觉候选.md',
              'p3_comparison_report.md', '视觉选择与可靠性.md', 'voxelmap_short_report.md',
              'voxelmap_long_report.md', 'visual_only_comparison_report.md', '待测清单.md', 'fusion_followup.md']
for name in historical:
    path = ROOT / 'docs' / name
    old = path.read_text(encoding='utf-8')
    if not old.startswith('> **阶段记录**'):
        path.write_text(notice + old, encoding='utf-8')

interface_path = ROOT / 'docs/接口与资源约定.md'
interface = interface_path.read_text(encoding='utf-8')
if '## 当前视觉融合内部消息' not in interface:
    interface += '''
## 当前视觉融合内部消息

当前默认 `visual_constraint_mode: body_twist`。内部 `/fusion/vision_odom_guarded` 仍用 `nav_msgs/msg/Odometry`，EKF 仅使用其 `twist` 的六个车体运动分量及协方差，`child_frame_id` 为 `base_link`。`pose` 保留视觉自身的 `learned_epoch_N` 坐标和大协方差，不参与 EKF；不能把这条内部消息的 pose 当作正式全局位姿。

正式 `/T3/semantic/current_pose`、增量地图和 TF 的坐标与消息契约不变。光照或几何失效、重置、过期时关闭视觉约束，恢复后只计算同一 epoch 内的运动。细节和验证见 [最终多轨迹报告](fusion_multitrajectory_report.md)。
'''
    interface_path.write_text(interface, encoding='utf-8')

inventory_path = ROOT / 'results/project_audit_sources_20260916.json'
inventory = read_json(inventory_path)
changed_external = []
for entry in inventory['files']:
    path = Path(entry['path'])
    if not path.is_file():
        entry['missing_at_finalization'] = True
        continue
    current = sha(path)
    if path.is_relative_to(ROOT):
        if current != entry['sha256']:
            entry.setdefault('previous_sha256', entry['sha256'])
        entry['sha256'] = current
        entry['lines'] = len(path.read_text(encoding='utf-8').splitlines())
    elif current != entry['sha256']:
        entry['current_sha256'] = current
        entry['changed_since_review'] = True
        changed_external.append(str(path))
body = ROOT / 'src/t3_lidar_visual_fusion/t3_lidar_visual_fusion/body_motion.py'
if str(body) not in {entry['path'] for entry in inventory['files']}:
    inventory['files'].append({'path': str(body), 'sha256': sha(body),
                               'lines': len(body.read_text().splitlines()),
                               'verification': 'test_body_motion and actual ROS body-motion guard regression'})
inventory['finalized_utc'] = datetime.now(timezone.utc).isoformat()
inventory['external_files_changed_since_review'] = changed_external
inventory['qualification'] += ' Final owned-source hashes refreshed after the body-motion release; changes in external P3 files are marked rather than represented as newly reviewed.'
write_json(inventory_path, inventory)

wrapper = ROOT / 'test_fusion_bag.sh'
wrapper.write_text(wrapper.read_text().replace('# One fixed 174-frame trajectory; save and verify before stopping owned nodes.',
                                              '# Fixed multi-trajectory suite or one 174-frame case; verify before stopping owned nodes.'))
shells = ['start.sh', 'start_live.sh', 'test_fusion_bag.sh', 'status.sh', 'save_map.sh', 'stop.sh']
for name in shells:
    subprocess.run(['bash', '-n', str(ROOT / name)], check=True)
manifest = source_manifest.check()
assert manifest['source_sha256'] == release['released_manifest']['source_sha256']

subprocess.run([sys.executable, str(ROOT / 'scripts/p3_comparison_view.py'), 'stop', '--output', str(view)], check=True)
assert not any(alive(p) for p in read_json(view / 'view_state.json')['services'].values())
assert not any(alive(p) for p in read_json(ROOT / 'run_state.json').get('processes', {}).values())

required = [ROOT / 'README.md', ROOT / 'docs/fusion_multitrajectory_report.md', ROOT / 'docs/project_audit.md',
            SUITE / 'comparison.json', SUITE / 'trajectory_comparison.png', SUITE / 'image_fault_comparison.png',
            SUITE / 'rviz_saved_complete.png', SUITE / 'release_verification.json', SUITE / 'artifact_verification.json']
assert all(p.is_file() and p.stat().st_size > 0 for p in required)
error = SUITE / 'finalization_error.json'
if error.exists():
    old_error = read_json(error)
    assert old_error.get('error') == 'Release check failed: suite_comparison'
    destination = SUITE / 'resolved_report_plotting_error.json'
    if not destination.exists():
        error.rename(destination)
    else:
        assert read_json(destination) == old_error
        error.unlink()
write_json(SUITE / 'finalization_complete.json', {
    'completed': True, 'utc': datetime.now(timezone.utc).isoformat(),
    'suite_cases_passed': 5, 'input_frames': 754, 'mapped_frames': 754,
    'runtime_source_sha256': manifest['source_sha256'],
    'release_verification_passed': True, 'artifact_export_verified': True,
    'saved_rviz_image_captured_and_inspected': True, 'owned_test_processes_stopped': True,
    'resolved_report_error': 'Matplotlib 3.5 requires height_ratios inside gridspec_kw. Plot generation was corrected and both comparison figures were generated and visually inspected; estimator results were unchanged.',
    'shell_syntax_checked': shells,
    'artifacts_sha256': {str(p.relative_to(ROOT)): sha(p) for p in required},
})
print('DOCUMENTS FINALIZED; 754/754 mapped; release and artifacts verified; owned test processes stopped.', flush=True)

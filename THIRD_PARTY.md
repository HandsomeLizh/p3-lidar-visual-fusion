# Third-party sources

This repository preserves per-component license declarations. It does not replace upstream notices with a single blanket license.

| Component | Origin / pinned version | Included material |
|---|---|---|
| VoxelMap | https://github.com/hku-mars/VoxelMap, `d787ee8ccfb0e509a36adb2c52bd5da97b29c39a` | Adapted headers and ROS 2 backend in `src/t3_voxelmap`; upstream README declares GPLv2. Original declaration is retained in `licenses/VoxelMap-notice.md`; GPLv2 text is in `licenses/GPL-2.0.txt`. |
| VINS-Fusion ROS 2 | https://github.com/leohaijunli/VINS-Fusion-ROS2-humble-arm, `ff167b7933a7389ee0b57659c21475599bdc92a5` | Downloaded into ignored `vendor/`; changes supplied in `docs/VINS-Fusion.patch`. Upstream GPLv3 text is retained in `licenses/VINS-Fusion-LICENCE`. |
| XFeat / LighterGlue | https://github.com/verlab/accelerated_features, `e92685f57f8318b18725c5c8c0bd28c7fe188d9a` | Downloaded source and weights; integration patch in `docs/XFeat.patch`. Upstream Apache-2.0 text is in `licenses/XFeat-LICENSE`. |
| LightGlue | https://github.com/cvg/LightGlue, `eb42fee2d71449efb0aa5c10549752b5d75384d8` | Downloaded optional frontend source; upstream Apache-2.0 text in `licenses/LightGlue-LICENSE`. |
| grid_map_msgs | https://github.com/ANYbotics/grid_map | Standard message/service package, BSD-3-Clause; notice retained in `src/grid_map_msgs/LICENSE`. |
| P3 mapping utilities | Original P3 project | Reused modules in `t3_lidar_visual_fusion/legacy`; provenance and source digests in `docs/reused_mapping_sources.json`. |

The fusion package's existing ROS metadata declares GPL-3.0-or-later; the VoxelMap adapter declares GPL-2.0. Their metadata is preserved. The corresponding GPL texts are included in `licenses/`.

Optional SuperPoint/ALIKED weights and Kornia packages are fetched separately; URLs, versions and checksums are recorded in `docs/learned_assets.json`. Their original terms continue to apply. No model weights, third-party Git histories, binary dependencies or datasets are re-uploaded here.

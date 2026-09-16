from pathlib import Path
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src/t3_lidar_visual_fusion'))
from t3_lidar_visual_fusion.output_continuity import OutputContinuity
from t3_lidar_visual_fusion.continuous_frame import ContinuousFrame

def pose(x):
    p=np.eye(4);p[0,3]=x;return p

g=OutputContinuity();g.accept(1.,pose(0.))
assert g.check(1.03,pose(.006))=='qualified'
g.accept(1.03,pose(.006))
assert g.check(10.499,pose(21.6648))=='output_discontinuity'
assert g.check(1000.,pose(21.6648))=='output_discontinuity'
assert g.check(2.33,pose(.266))=='qualified'
g.accept(2.33,pose(.266))
assert g.check(2.33,pose(.466))=='qualified'
g.accept(2.33,pose(.466))
assert g.check(2.33,pose(.866))=='same_stamp_discontinuity'
assert g.check(1.0,pose(0.))=='out_of_order'
source=ContinuousFrame(recovery_frames=1,recovery_max_step=6.,recovery_max_angle=1.)
source.begin_epoch('odom');source.alignment=np.eye(4)
assert source.accept(1.,pose(0.)).transform is not None
source.close('failed_registration')
assert source.accept(20.,pose(21.)).reason=='recovery_discontinuity'
assert source.accept(21.,pose(.5)).transform is not None
print('PASS: long-gap output/source jump rejected; same-stamp cumulative bound; nearby recovery')

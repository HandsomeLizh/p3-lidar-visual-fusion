"""Photometric quality of mono images; color is never a validity requirement."""
from dataclasses import dataclass,asdict
import cv2
import numpy as np


@dataclass(frozen=True)
class ImageQuality:
    valid: bool
    reason: str
    mean: float
    contrast: float
    dark_fraction: float
    saturated_fraction: float
    feature_count: int
    feature_coverage: float
    laplacian_variance: float
    score: float

    def dictionary(self):return asdict(self)


def assess_image(image,*,min_brightness=8.,min_features=25,min_contrast=18.,
                 min_coverage=.20,min_sharpness=8.,texture_required=True):
    a=np.asarray(image)
    if a.ndim!=2 or a.dtype!=np.uint8 or a.size==0:
        raise ValueError("Quality input must be nonempty mono8")
    a=cv2.resize(a,(320,240),interpolation=cv2.INTER_AREA)
    mean=float(a.mean());low,high=np.percentile(a,[5,95]);contrast=float(high-low)
    dark=float(np.mean(a<=5));sat=float(np.mean(a>=250))
    corners=cv2.goodFeaturesToTrack(a,maxCorners=250,qualityLevel=.01,minDistance=5)
    count=0 if corners is None else len(corners)
    coverage=0.
    if count:
        xy=corners.reshape(-1,2);cells=(xy/[40.,40.]).astype(int)
        coverage=len(np.unique(cells,axis=0))/48.
    sharp=float(cv2.Laplacian(a,cv2.CV_32F).var())
    reason="healthy"
    if mean<min_brightness or dark>.95:reason="underexposed"
    elif mean>248 or sat>.92:reason="overexposed"
    elif texture_required and sat>.75 and coverage<.30:reason="overexposed"
    elif texture_required and dark>.30 and sat>.25 and coverage<.40:reason="mixed_extreme_lighting"
    elif texture_required and sat>.40 and coverage<.30:reason="localized_glare"
    elif texture_required and contrast<min_contrast:reason="low_contrast"
    elif texture_required and count<min_features:reason="low_image_features"
    elif texture_required and coverage<min_coverage:reason="concentrated_features"
    elif texture_required and sharp<min_sharpness:reason="blurred"
    # Clipping reduces confidence even when enough spatially distributed
    # structure remains for tracking. A black-and-white image can still pass.
    # Learned RoMa matches are not measured by a classical corner detector.
    # Its own frontend + LiDAR motion agreement decide geometric usability.
    texture_score=min(1.,count/100.)*min(1.,coverage/.5) if texture_required else 1.
    score=float(np.clip(texture_score*
                        max(.25,1.-.7*sat-.4*dark),.1,1.))
    return ImageQuality(reason=="healthy",reason,mean,contrast,dark,sat,count,coverage,sharp,score)

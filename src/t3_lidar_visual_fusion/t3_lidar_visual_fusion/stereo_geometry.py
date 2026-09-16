"""Calibrated sparse stereo and robust 3D-to-2D motion; no learned depth or map."""
from dataclasses import dataclass
import cv2
import numpy as np
from .core import rigid, inverse


class TrackingFailure(ValueError):
    pass


def coverage(points,size):
    points=np.asarray(points).reshape(-1,2)
    if not len(points):return 0.
    xy=np.floor(points/np.asarray(size)*[8,6]).astype(int)
    valid=(xy>=0).all(axis=1)&(xy<[8,6]).all(axis=1)
    return len(np.unique(xy[valid,1]*8+xy[valid,0]))/48.


def project(projection,points):
    h=np.column_stack([points,np.ones(len(points))])@projection.T
    z=h[:,2]
    with np.errstate(divide="ignore",invalid="ignore"):
        uv=h[:,:2]/z[:,None]
    return uv,z


@dataclass
class GeometryResult:
    current_from_reference: np.ndarray
    metrics: dict
    current_inliers: np.ndarray


class StereoGeometry:
    def __init__(self,profile):
        self.cfg=profile["learned_visual"]
        self.size=tuple(map(int,profile["output_image_size"]))
        limit=int(self.cfg.get("max_image_side",640))
        if min(self.size)<64 or max(self.size)>limit or limit>1024:
            raise ValueError("Learned image size exceeds its configured resource budget")
        scale=np.asarray(self.size)/np.asarray(profile["input_image_size"])
        k0=np.asarray(profile["camera_k"],dtype=float).reshape(3,3).copy()
        k1=np.asarray(profile.get("camera_k_right",profile["camera_k"]),dtype=float).reshape(3,3).copy()
        k0[:2]*=scale[:,None];k1[:2]*=scale[:,None]
        d0=np.asarray(profile.get("distortion",[0.,0.,0.,0.]),dtype=float)
        d1=np.asarray(profile.get("distortion_right",d0),dtype=float)
        self.base_from_left=rigid(profile["base_from_camera_left"])
        base_from_right=rigid(profile["base_from_camera_right"])
        right_from_left=inverse(base_from_right)@self.base_from_left
        if not np.isfinite(k0).all() or not np.isfinite(k1).all() or min(k0[0,0],k0[1,1],k1[0,0],k1[1,1])<=0:
            raise ValueError("Invalid stereo intrinsics")
        if np.linalg.norm(right_from_left[:3,3])<.01:
            raise ValueError("Stereo baseline must be measured and nonzero")
        r0,r1,self.p0,self.p1,_,_,_=cv2.stereoRectify(
            k0,d0,k1,d1,self.size,right_from_left[:3,:3],right_from_left[:3,3],
            flags=cv2.CALIB_ZERO_DISPARITY,alpha=0,newImageSize=self.size)
        if abs(self.p1[1,3])>1e-5 or abs(self.p1[0,3])<1e-5:
            raise ValueError("This frontend requires horizontal rectified stereo")
        self.k=self.p0[:,:3].copy()
        self.disparity_sign=-np.sign(self.p1[0,3])
        self.map0=cv2.initUndistortRectifyMap(k0,d0,r0,self.p0,self.size,cv2.CV_32FC1)
        self.map1=cv2.initUndistortRectifyMap(k1,d1,r1,self.p1,self.size,cv2.CV_32FC1)
        rect_from_left=np.eye(4);rect_from_left[:3,:3]=r0
        self.base_from_rect=self.base_from_left@inverse(rect_from_left)

    def rectify(self,left,right):
        expected=self.size[::-1]
        if left.shape!=expected or right.shape!=expected:
            raise TrackingFailure("uncalibrated_image_dimensions")
        return (cv2.remap(left,*self.map0,cv2.INTER_LINEAR),
                cv2.remap(right,*self.map1,cv2.INTER_LINEAR))

    def triangulate(self,left,right,matches):
        left=np.asarray(left,dtype=float).reshape(-1,2)
        right=np.asarray(right,dtype=float).reshape(-1,2)
        pairs=np.asarray(matches,dtype=int).reshape(-1,2)
        points=np.full((len(left),3),np.nan)
        if not len(pairs):return points,{"stereo_matches":0,"stereo_points":0,"stereo_coverage":0.}
        a,b=left[pairs[:,0]],right[pairs[:,1]]
        disp=(a[:,0]-b[:,0])*self.disparity_sign
        keep=(np.abs(a[:,1]-b[:,1])<=self.cfg.get("epipolar_error_px",1.5))
        keep &= disp>=self.cfg.get("min_disparity_px",1.5)
        pairs,a,b=pairs[keep],a[keep],b[keep]
        if len(pairs):
            h=cv2.triangulatePoints(self.p0,self.p1,a.T,b.T).T
            with np.errstate(divide="ignore",invalid="ignore"):
                xyz=h[:,:3]/h[:,3,None]
            uv0,z0=project(self.p0,xyz);uv1,z1=project(self.p1,xyz)
            error=np.maximum(np.linalg.norm(uv0-a,axis=1),np.linalg.norm(uv1-b,axis=1))
            good=np.isfinite(xyz).all(axis=1)&(z0>self.cfg.get("min_depth_m",.4))
            good &= (z1>0)&(z0<self.cfg.get("max_depth_m",60.))
            good &= error<=self.cfg.get("stereo_reprojection_px",1.5)
            points[pairs[good,0]]=xyz[good]
        valid=np.isfinite(points).all(axis=1)
        return points,dict(stereo_matches=int(len(matches)),stereo_points=int(valid.sum()),
            stereo_coverage=coverage(left[valid],self.size))

    def estimate(self,reference_points,current_points,reference_pixels,current_pixels,matches):
        pairs=np.asarray(matches,dtype=int).reshape(-1,2)
        if not len(pairs):raise TrackingFailure("no_temporal_matches")
        objects=np.asarray(reference_points)[pairs[:,0]]
        image=np.asarray(current_pixels)[pairs[:,1]]
        target=np.asarray(current_points)[pairs[:,1]]
        good=np.isfinite(objects).all(axis=1)&np.isfinite(image).all(axis=1)
        current_ids=pairs[good,1]
        objects,image,target=objects[good],image[good],target[good]
        minimum=int(self.cfg.get("min_pnp_inliers",25))
        if len(objects)<minimum:raise TrackingFailure("insufficient_temporal_depth")
        ok,rvec,tvec,inliers=cv2.solvePnPRansac(
            np.ascontiguousarray(objects,dtype=np.float64),
            np.ascontiguousarray(image,dtype=np.float64),self.k,None,
            iterationsCount=int(self.cfg.get("pnp_iterations",150)),
            reprojectionError=float(self.cfg.get("pnp_reprojection_px",2.5)),
            confidence=.999,flags=cv2.SOLVEPNP_EPNP)
        if not ok or inliers is None or len(inliers)<minimum:
            raise TrackingFailure("pnp_ransac_failed")
        indices=inliers.reshape(-1)
        rvec,tvec=cv2.solvePnPRefineLM(objects[indices],image[indices],self.k,None,rvec,tvec)
        transform=np.eye(4);transform[:3,:3]=cv2.Rodrigues(rvec)[0];transform[:3,3]=tvec.reshape(3)
        transform=rigid(transform)
        predicted=objects@transform[:3,:3].T+transform[:3,3]
        uv,z=project(self.p0,predicted)
        error=np.linalg.norm(uv-image,axis=1)
        accepted=(z>0)&np.isfinite(error)&(error<=self.cfg.get("pnp_reprojection_px",2.5))
        ratio=float(accepted.mean());n=int(accepted.sum())
        if n<minimum or ratio<self.cfg.get("min_pnp_ratio",.45):
            raise TrackingFailure("pnp_low_inlier_ratio")
        spread=coverage(image[accepted],self.size)
        if spread<self.cfg.get("min_pnp_coverage",.15):
            raise TrackingFailure("pnp_poor_spatial_coverage")
        common=accepted&np.isfinite(target).all(axis=1)
        if common.sum()<self.cfg.get("min_stereo_checks",12):
            raise TrackingFailure("insufficient_current_stereo_checks")
        distance=np.linalg.norm(predicted[common]-target[common],axis=1)
        tolerance=self.cfg.get("stereo_motion_floor_m",.15)+self.cfg.get("stereo_motion_depth_ratio",.03)*target[common,2]
        metric_ratio=float(np.mean(distance<=tolerance))
        if metric_ratio<self.cfg.get("min_stereo_motion_ratio",.6):
            raise TrackingFailure("stereo_motion_disagreement")
        return GeometryResult(transform,dict(temporal_with_depth=len(objects),pnp_inliers=n,
            pnp_ratio=ratio,pnp_coverage=spread,
            reprojection_median_px=float(np.median(error[accepted])),
            reprojection_p95_px=float(np.percentile(error[accepted],95)),
            stereo_checks=int(common.sum()),stereo_motion_ratio=metric_ratio,
            stereo_motion_median_m=float(np.median(distance))),
            np.unique(current_ids[common][distance<=tolerance]))

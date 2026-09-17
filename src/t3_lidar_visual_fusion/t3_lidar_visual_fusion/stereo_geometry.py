"""Calibrated sparse stereo and robust 3D-to-2D motion; no learned depth or map."""
from dataclasses import dataclass
import cv2
import numpy as np
from .core import rigid, inverse


class TrackingFailure(ValueError):
    def __init__(self,reason,metrics=None):
        super().__init__(reason)
        self.metrics=dict(metrics or {})


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


def spatial_density_mask(world_points,weights,cell_size):
    """Stable thinning by world XY cell; repeated scans cannot fill a fade band.

    Every height in a column gets the same spatial sample, so thinning does not
    preferentially discard a high or low surface. No random state or new points.
    """
    points=np.asarray(world_points,dtype=float).reshape(-1,3)
    weights=np.asarray(weights,dtype=float).reshape(-1)
    if (len(points)!=len(weights) or not np.isfinite(cell_size) or cell_size<=0 or
            not np.isfinite(weights).all() or np.any((weights<0)|(weights>1))):
        raise ValueError('Invalid spatial density sampling configuration')
    valid=np.isfinite(points).all(axis=1)
    keep=valid&(weights>=1.)
    ids=np.flatnonzero(valid&(weights>0.)&(weights<1.))
    if not len(ids):return keep
    cells=np.floor(points[ids,:2]/cell_size).astype(np.int64).astype(np.uint64)
    hashed=cells[:,0]*np.uint64(0x9E3779B97F4A7C15)
    hashed ^= (cells[:,1]+np.uint64(0xD1B54A32D192ED03))*np.uint64(0xBF58476D1CE4E5B9)
    hashed ^= hashed>>np.uint64(30)
    hashed *= np.uint64(0xBF58476D1CE4E5B9)
    hashed ^= hashed>>np.uint64(27)
    hashed *= np.uint64(0x94D049BB133111EB)
    hashed ^= hashed>>np.uint64(31)
    sample=(hashed>>np.uint64(11)).astype(np.float64)*(1./2**53)
    keep[ids]=sample<weights[ids]
    return keep


def radial_continuity_weights(base_points,*,sector_deg=2.,max_gap_m=.25,
                              feather_m=.6,seed_range_m=5.):
    """Keep the near connected returns before the first radial measurement gap.

    Run on the measured scan BEFORE cosmetic thinning. Each angular sector
    starts at its first nearby return, so the sensor's initial blind zone is
    not a gap. A sector containing only remote returns is not a near seed.
    Heights and measured coordinates are never changed or interpolated.
    """
    options=np.asarray([sector_deg,max_gap_m,feather_m,seed_range_m],dtype=float)
    if (not np.isfinite(options).all() or not .25<=sector_deg<=30. or
            max_gap_m<=0. or feather_m<0. or seed_range_m<=0.):
        raise ValueError('Invalid cloud preview continuity configuration')
    points=np.asarray(base_points,dtype=float).reshape(-1,3)
    weights=np.zeros(len(points),dtype=float)
    ranges=np.hypot(points[:,0],points[:,1])
    ids=np.flatnonzero(np.isfinite(points).all(axis=1)&np.isfinite(ranges)&(ranges>0.))
    if not len(ids):return weights
    angles=np.mod(np.arctan2(points[ids,1],points[ids,0])+np.pi,2*np.pi)
    sectors=np.floor(angles/np.deg2rad(sector_deg)).astype(np.int64)
    order=np.lexsort((ranges[ids],sectors))
    ids=ids[order];sectors=sectors[order];distance=ranges[ids]
    new_sector=np.r_[True,sectors[1:]!=sectors[:-1]]
    starts=np.flatnonzero(new_sector);group=np.cumsum(new_sector)-1
    gap=(~new_sector[1:])&(np.diff(distance)>max_gap_m+1e-6)
    cutoff=np.full(len(starts),np.inf)
    np.minimum.at(cutoff,group[1:][gap],distance[:-1][gap])
    seeded=distance[starts]<=seed_range_m
    if feather_m>0.:
        fade=np.clip((cutoff[group]-distance)/feather_m,0.,1.)
        weights[ids]=fade*fade*(3.-2.*fade)
    else:
        weights[ids]=(distance<=cutoff[group]).astype(float)
    weights[ids[~seeded[group]]]=0.
    return weights


@dataclass
class GeometryResult:
    current_from_reference: np.ndarray
    metrics: dict


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
        self.inverse_k=np.linalg.inv(self.k)
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
            # PnP observes the LEFT pixel. Symmetric DLT also fits right-image
            # vertical noise, moving the point off that left viewing ray. That
            # inconsistency creates motion even when an image repeats exactly.
            # Retain stereo depth, but anchor its bearing to the left observation.
            rays=np.column_stack((a,np.ones(len(a))))@self.inverse_k.T
            xyz=rays*xyz[:,2,None]
            uv0,z0=project(self.p0,xyz);uv1,z1=project(self.p1,xyz)
            error=np.maximum(np.linalg.norm(uv0-a,axis=1),np.linalg.norm(uv1-b,axis=1))
            good=np.isfinite(xyz).all(axis=1)&(z0>self.cfg.get("min_depth_m",.4))
            good &= (z1>0)&(z0<self.cfg.get("max_depth_m",60.))
            good &= error<=self.cfg.get("stereo_reprojection_px",1.5)
            points[pairs[good,0]]=xyz[good]
        valid=np.isfinite(points).all(axis=1)
        return points,dict(stereo_matches=int(len(matches)),stereo_points=int(valid.sum()),
            stereo_coverage=coverage(left[valid],self.size))

    def mapping_points(self,points):
        """Conservative near-field stereo evidence, expressed once in base_link.

        sigma_z is a screening estimate from disparity precision, not a calibrated
        posterior uncertainty. Localization still uses the full stereo range.
        """
        points=np.asarray(points,dtype=float).reshape(-1,3)
        points=points[np.isfinite(points).all(axis=1)]
        depth=points[:,2]
        sigma=depth**2*float(self.cfg.get("mapping_disparity_sigma_px",.5))/abs(self.p1[0,3])
        keep=(depth>self.cfg.get("min_depth_m",.4))&(depth<=self.cfg.get("mapping_max_depth_m",12.))
        keep &= sigma<=self.cfg.get("mapping_max_depth_std_m",.08)
        points=points[keep]
        return points@self.base_from_rect[:3,:3].T+self.base_from_rect[:3,3]

    def mapping_frustum_mask(self,base_points,*,image_margin_px=0.,match_stereo_depth_limit=True):
        """Select body-frame returns inside both rectified camera images.

        This is a geometric field-of-view crop, not a camera occlusion test.
        It never changes a point's measured coordinates or the localization scan.
        """
        return self.mapping_frustum_weights(base_points,image_margin_px=image_margin_px,
            match_stereo_depth_limit=match_stereo_depth_limit)>0.

    def mapping_frustum_weights(self,base_points,*,image_margin_px=0.,match_stereo_depth_limit=True,
                                image_feather_fraction=0.,depth_feather_m=0.,
                                full_density_range_m=None,range_feather_m=0.):
        """Full density in the original view, smooth falloff just outside it."""
        base=np.asarray(base_points,dtype=float).reshape(-1,3)
        margin=float(image_margin_px)
        if not np.isfinite(margin) or margin<0 or 2*margin>=min(self.size):
            raise ValueError('Camera crop margin leaves no valid image area')
        feather=float(image_feather_fraction);depth_feather=float(depth_feather_m)
        range_feather=float(range_feather_m)
        if not np.isfinite([feather,depth_feather,range_feather]).all() or min(feather,depth_feather,range_feather)<0:
            raise ValueError('Camera feather widths must be finite and nonnegative')
        if full_density_range_m is not None:
            full_density_range_m=float(full_density_range_m)
            if (not np.isfinite(full_density_range_m) or full_density_range_m<=0 or
                    match_stereo_depth_limit or depth_feather>0):
                raise ValueError('Explicit LiDAR range requires positive range and disabled stereo depth limit')
        elif range_feather>0:raise ValueError('Range feather requires a full density range')
        rect=(base-self.base_from_rect[:3,3])@self.base_from_rect[:3,:3]
        a,za=project(self.p0,rect);b,zb=project(self.p1,rect)
        minimum=float(self.cfg.get('min_depth_m',.4))
        maximum=float(self.cfg.get('mapping_max_depth_m',12.))
        if match_stereo_depth_limit:
            sigma=float(self.cfg.get('mapping_disparity_sigma_px',.5))
            allowed=float(self.cfg.get('mapping_max_depth_std_m',.08))
            if not np.isfinite([sigma,allowed]).all() or min(sigma,allowed)<=0:
                raise ValueError('Invalid stereo depth limit for camera crop')
            maximum=min(maximum,np.sqrt(allowed*abs(self.p1[0,3])/sigma))
        if not np.isfinite([minimum,maximum]).all() or minimum<0 or maximum<=minimum:
            raise ValueError('Invalid camera crop depth interval')
        keep=np.isfinite(rect).all(axis=1)&(za>minimum)&(zb>minimum)
        weights=keep.astype(float)
        def falloff(x):
            t=np.clip(x,0.,1.)
            return (1.-t)**2*(1.+2.*t)
        if full_density_range_m is not None:
            distance=np.linalg.norm(base[:,:2],axis=1)
            weights *= (falloff((distance-full_density_range_m)/range_feather)
                        if range_feather>0 else distance<=full_density_range_m)
        elif depth_feather>0:
            weights *= falloff((za-maximum)/depth_feather)
        else:weights *= za<=maximum
        outside=np.zeros((len(base),2))
        for uv in (a,b):
            weights *= np.isfinite(uv).all(axis=1)
            if feather>0:
                excess=np.maximum(np.maximum(margin-uv,uv-(np.asarray(self.size)-margin)),0.)
                outside=np.maximum(outside,excess/(np.asarray(self.size)*feather))
            else:
                weights *= (uv[:,0]>=margin)&(uv[:,1]>=margin)
                weights *= (uv[:,0]<self.size[0]-margin)&(uv[:,1]<self.size[1]-margin)
        if feather>0:
            # Euclidean excess rounds the image corners instead of introducing
            # another expanded rectangular boundary.
            weights *= falloff(np.linalg.norm(outside,axis=1))
        return np.nan_to_num(weights,nan=0.,posinf=0.,neginf=0.)

    def estimate(self,reference_points,current_points,reference_pixels,current_pixels,matches):
        pairs=np.asarray(matches,dtype=int).reshape(-1,2)
        if not len(pairs):raise TrackingFailure("no_temporal_matches")
        objects=np.asarray(reference_points)[pairs[:,0]]
        image=np.asarray(current_pixels)[pairs[:,1]]
        target=np.asarray(current_points)[pairs[:,1]]
        good=np.isfinite(objects).all(axis=1)&np.isfinite(image).all(axis=1)
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
        metrics=dict(temporal_with_depth=len(objects),pnp_inliers=n,
            pnp_ratio=ratio,pnp_coverage=spread,
            reprojection_median_px=float(np.median(error[accepted])),
            reprojection_p95_px=float(np.percentile(error[accepted],95)),
            stereo_checks=int(common.sum()),stereo_motion_ratio=metric_ratio,
            stereo_motion_median_m=float(np.median(distance)),
            stereo_motion_tolerance_median_m=float(np.median(tolerance)))
        if metric_ratio<self.cfg.get("min_stereo_motion_ratio",.6):
            raise TrackingFailure("stereo_motion_disagreement",metrics)
        return GeometryResult(transform,metrics)

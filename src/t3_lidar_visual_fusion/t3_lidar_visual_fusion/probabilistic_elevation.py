"""Bounded probabilistic height updates, inspired by ANYbotics elevation_mapping.

One surface observation per cell/scan; pose uncertainty is a correlated floor.
This is an independent implementation, not a port of the complete ROS1 mapper.
"""
import numpy as np
from .surface_grid import SurfaceGrid


FUSION_MODEL = 'probabilistic_v2'
FILTER_FIELDS = (("height_filter_variance", np.float32), ("height_pose_variance", np.float32),
                 ("height_last_stamp", np.float64), ("height_source_mask", np.uint8),
                 ("height_candidate", np.float32), ("height_candidate_variance", np.float32),
                 ("height_candidate_pose_variance", np.float32),
                 ("height_candidate_stamp", np.float64), ("height_candidate_count", np.uint8))


def correlated_height_update(old_height, old_variance, old_pose_variance,
                             height, variance, pose_variance, minimum):
    """Preserve shared pose error while averaging independent sensor noise.

    A scalar common component approximates temporal pose correlation; this is
    not a full cross-pose covariance model. A noisier pose gets less weight.
    """
    shared = min(old_pose_variance, pose_variance, old_variance, variance)
    gain = max(0., min(1., (old_variance-shared) / max(1e-12,old_variance+variance-2*shared)))
    pose_variance = ((1-gain)**2*old_pose_variance + gain**2*pose_variance +
                     2*gain*(1-gain)*shared)
    return (old_height+gain*(height-old_height),
            max(minimum,pose_variance,old_variance-gain*(old_variance-shared)),pose_variance)


def height_variances(base, pose, covariance, source, profile, config):
    """Sensor noise and fixed-world-axis ROS pose covariance -> vertical variance.

    Returned pose variance is kept as a floor; repeating a scan must not average
    away the common uncertainty of its body pose. Noise settings are models,
    not claims of calibrated sensor accuracy.
    """
    base = np.asarray(base, dtype=float)
    covariance = np.asarray(covariance, dtype=float).reshape(6, 6)
    if (not np.isfinite(covariance).all() or
            not np.allclose(covariance, covariance.T, atol=1e-8) or
            np.linalg.eigvalsh(covariance).min() < -1e-8):
        raise ValueError("Invalid pose covariance for elevation fusion")
    lever = base @ pose[:3, :3].T
    jacobian = np.zeros((len(base), 6))
    jacobian[:, 2] = 1.; jacobian[:, 3] = lever[:, 1]; jacobian[:, 4] = -lever[:, 0]
    pose_variance = np.maximum(0., np.einsum('ni,ij,nj->n', jacobian, covariance, jacobian))
    floor = float(config.get('sensor_floor_std_m', .01)) ** 2
    if source == 'stereo':
        mount = np.asarray(profile['base_from_camera_left'])
        camera = (base - mount[:3, 3]) @ mount[:3, :3]
        depth = camera[:, 2]
        focal = float(np.asarray(profile['camera_k'])[0, 0])
        focal *= profile['output_image_size'][0] / profile['input_image_size'][0]
        baseline = np.linalg.norm(mount[:3, 3] - np.asarray(profile['base_from_camera_right'])[:3, 3])
        if focal <= 0. or baseline <= 0.: raise ValueError('Invalid stereo noise calibration')
        sigma = depth ** 2 * config.get('stereo_disparity_std_px', .5) / (focal * baseline)
        world_camera = pose[:3, :3] @ mount[:3, :3]
        bearing_z = (camera @ world_camera[2]) / np.maximum(depth, 1e-6)
        pixel_variance = (depth * config.get('stereo_pixel_std_px', .5) / focal) ** 2
        sensor_variance = floor + (bearing_z * sigma) ** 2 + pixel_variance * np.sum(world_camera[2, :2] ** 2)
        sensor_variance = np.where(depth > 0., sensor_variance, np.inf)
    else:
        mount = np.asarray(profile.get('base_from_lidar', np.eye(4))) if source == 'lidar' else np.eye(4)
        vector = (base - mount[:3, 3]) @ pose[:3, :3].T
        distance = np.linalg.norm(vector, axis=1)
        vertical = vector[:, 2] / np.maximum(distance, 1e-6)
        sigma = config.get('range_std_m', .015) + config.get('range_std_per_m', .0002) * distance
        angular = config.get('beam_angle_std_rad', .001)
        sensor_variance = floor + (vertical * sigma) ** 2 + np.sum(vector[:, :2] ** 2, axis=1) * angular ** 2
    return sensor_variance, pose_variance


def scan_observations(points, sensor_variance, pose_variance, resolution, maximum,
                      *, priority_center=None, priority_radius=8., selection_phase=0):
    """Reduce each scan to an upper surface mode, without averaging two surfaces.

    Trim only the top/bottom 10% for dense cells. Sparse high returns remain
    candidates and require temporal confirmation before replacing old ground.
    Repeated points in one scan do not reduce the observation variance. Prefer
    the body neighborhood; rotate the remaining bounded sample between scans
    so a stationary sensor cannot permanently exclude the same cells.
    """
    if not np.isfinite(resolution) or resolution <= 0. or int(maximum) < 1:
        raise ValueError('Elevation observations require positive resolution and cell budget')
    points = np.asarray(points, dtype=float).reshape(-1, 3)
    sensor_variance = np.broadcast_to(sensor_variance, (len(points),))
    pose_variance = np.broadcast_to(pose_variance, (len(points),))
    valid = (np.isfinite(points).all(axis=1) & np.isfinite(sensor_variance) &
             np.isfinite(pose_variance) & (sensor_variance > 0.) & (pose_variance >= 0.))
    points, sensor_variance, pose_variance = points[valid], sensor_variance[valid], pose_variance[valid]
    if not len(points): return dict(cells=np.empty((0, 2), dtype=np.int64), height=np.empty(0),
                                   variance=np.empty(0), pose_variance=np.empty(0), relief=np.empty(0))
    cells = np.floor(points[:, :2] / resolution).astype(np.int64)
    order = np.lexsort((points[:, 2], cells[:, 1], cells[:, 0]))
    cells, z = cells[order], points[order, 2]
    sensor_variance, pose_variance = sensor_variance[order], pose_variance[order]
    starts = np.r_[0, np.flatnonzero(np.any(np.diff(cells, axis=0), axis=1)) + 1]
    counts = np.diff(np.r_[starts, len(z)])
    upper = starts + np.ceil((counts - 1) * .9).astype(int)
    lower = starts + np.floor((counts - 1) * .1).astype(int)
    group = np.repeat(np.arange(len(starts)), counts)
    width = np.clip(3. * np.sqrt(sensor_variance[upper]), .03, .08)
    keep = np.abs(z - z[upper[group]]) <= width[group]
    groups = group[keep]; weights = 1. / sensor_variance[keep]
    total = np.bincount(groups, weights, minlength=len(starts))
    def average(values):
        return np.bincount(groups, weights * values[keep], minlength=len(starts)) / total
    result = dict(cells=cells[starts], height=average(z),
                  variance=average(sensor_variance + pose_variance),
                  pose_variance=average(pose_variance), relief=z[upper] - z[lower])
    if len(starts) > maximum:
        maximum = int(maximum)
        indices = np.arange(len(starts))
        def sample(pool, count):
            if count == 0: return np.empty(0,dtype=int)
            return pool[(np.arange(count)*len(pool)//count + int(selection_phase)) % len(pool)]
        if priority_center is not None:
            center = np.asarray(priority_center,dtype=float).reshape(2)
            if not np.isfinite(center).all() or not np.isfinite(priority_radius) or priority_radius < 0.:
                raise ValueError('Invalid elevation sampling neighborhood')
            close = np.linalg.norm((result['cells']+.5)*resolution-center,axis=1) <= priority_radius
            nearby = indices[close]
            if len(nearby) >= maximum:
                selected = sample(nearby,maximum)
            else:
                selected = np.r_[nearby,sample(indices[~close],maximum-len(nearby))]
        else:
            selected = sample(indices,maximum)
        selected.sort()
        result = {key: value[selected] for key, value in result.items()}
    return result


class ProbabilisticSurfaceGrid(SurfaceGrid):
    def __init__(self, *, fusion_config, **kwargs):
        super().__init__(**kwargs)
        self.fusion_config = fusion_config
        shape = self.elevation_count.shape
        for name, dtype in FILTER_FIELDS:
            setattr(self, name, np.zeros(shape, dtype=dtype))
        self.height_last_stamp.fill(-np.inf)
        self.height_candidate_stamp.fill(-np.inf)

    def fuse_height(self, row, col, height, variance, pose_variance, relief, stamp, source):
        cfg = self.fusion_config
        if (not np.isfinite([height, variance, pose_variance, relief, stamp]).all() or
                variance <= 0. or pose_variance < 0. or relief < 0.):
            return 'invalid'
        index = (row, col)
        tolerance = cfg.get('same_scan_tolerance_sec', .02)
        last = self.height_last_stamp[index]
        if stamp < last - tolerance: return 'out_of_order'
        source_mask = {'lidar': 1, 'stereo': 2}.get(source, 4)
        if stamp <= last + tolerance:
            if self.height_source_mask[index] & source_mask: return 'duplicate'
            self.height_source_mask[index] |= source_mask
        else:
            self.height_source_mask[index] = source_mask
        self.height_last_stamp[index] = max(stamp, last)
        minimum = cfg.get('minimum_variance', .0001)
        old_count = int(self.elevation_count[index])
        old_height = float(self.elevation_mean[index])
        old_variance = float(self.height_filter_variance[index])
        result = 'initialized'
        if old_count:
            # Bounded process noise lets new independent evidence remain useful.
            elapsed = max(0., min(stamp-last, cfg.get('aging_max_sec', 10.)))
            old_variance += elapsed * cfg.get('process_variance_per_sec', .00001)
            threshold = min(cfg.get('maximum_innovation_m', .10),
                            cfg.get('mahalanobis_threshold', 3.) * np.sqrt(old_variance + variance))
            threshold = max(cfg.get('minimum_innovation_m', .03), threshold)
            if abs(height - old_height) > threshold:
                candidate_count = int(self.height_candidate_count[index])
                gap = stamp - self.height_candidate_stamp[index]
                coherent = (candidate_count > 0 and 0. < gap <= cfg.get('candidate_max_gap_sec', 6.) and
                            abs(height-self.height_candidate[index]) <= cfg.get('candidate_tolerance_m', .05))
                if gap <= tolerance and candidate_count: return 'candidate_duplicate'
                if coherent:
                    candidate_count = min(255, candidate_count+1)
                    candidate_height = float(self.height_candidate[index])
                    candidate_variance = float(self.height_candidate_variance[index])
                    height,variance,pose_variance = correlated_height_update(candidate_height,candidate_variance,
                        float(self.height_candidate_pose_variance[index]),height,variance,pose_variance,minimum)
                else:
                    candidate_count = 1
                self.height_candidate[index] = height
                self.height_candidate_variance[index] = variance
                self.height_candidate_pose_variance[index] = pose_variance
                self.height_candidate_stamp[index] = stamp
                self.height_candidate_count[index] = candidate_count
                if candidate_count < cfg.get('change_confirmations', 3): return 'candidate'
                # A lower return can be a different face/under an overhang.
                # Confirmed ray cleanup owns removal; point averaging cannot.
                if height < old_height and not cfg.get('allow_confirmed_lowering', False):
                    return 'lower_surface_unconfirmed_by_rays'
                result = 'surface_replaced'
            else:
                height,variance,pose_variance = correlated_height_update(old_height,old_variance,
                    float(self.height_pose_variance[index]),height,variance,pose_variance,minimum)
                result = 'fused'
        retain_upper_candidate = (result == 'fused' and self.height_candidate_count[index] > 0 and
            self.height_candidate[index] > height + cfg.get('minimum_innovation_m', .03) and
            stamp-self.height_candidate_stamp[index] <= cfg.get('candidate_max_gap_sec', 6.))
        if not retain_upper_candidate:
            self.height_candidate_count[index] = 0
            self.height_candidate_stamp[index] = -np.inf
        count = min(np.iinfo(np.uint32).max, old_count+1)
        self.elevation_count[index] = count
        self.elevation_mean[index] = height
        self.height_filter_variance[index] = max(minimum, pose_variance, variance)
        self.height_pose_variance[index] = pose_variance
        # Keep old snapshot consumers approximately compatible; live exports use
        # height_filter_variance directly, including the first observation.
        self.elevation_M2[index] = self.height_filter_variance[index] * max(0, count-1)
        self.elevation_min[index] = min(self.elevation_min[index], height)
        self.elevation_max[index] = max(self.elevation_max[index], height)
        self.surface_height_range[index] = max(self.surface_height_range[index], relief)
        return result

    def _update_elevation_cell(self, row, column, z_values):
        # Compatibility for direct grid updates and verified ray-clear rebuilds.
        values = np.asarray(z_values)
        values = values[np.isfinite(values)]
        if not len(values): return 0
        points = np.column_stack((np.zeros(len(values)), np.zeros(len(values)), values))
        batch = scan_observations(points, .0009, 0., self.geometry.resolution, 1)
        last = self.height_last_stamp[row, column]
        stamp = last+1. if np.isfinite(last) else 0.
        result = self.fuse_height(row, column, float(batch['height'][0]), float(batch['variance'][0]),
                                  0., float(batch['relief'][0]), stamp, 'direct')
        return int(result in ('initialized', 'fused', 'surface_replaced'))

    def elevation_variance_layer(self):
        return np.where(self.elevation_count > 0, self.height_filter_variance, np.nan).astype(np.float32)

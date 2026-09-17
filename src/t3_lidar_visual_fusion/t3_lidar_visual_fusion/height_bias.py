"""Conservative scan-to-map vertical bias; never changes odometry or map history."""
import numpy as np


class HeightBiasCompensator:
    def __init__(self, config):
        self.config = config
        self.offset = 0.
        self.last_stamp = -np.inf
        self.last_pose = None
        self.updates = 0
        self.reference_allowed = True
        self.stats = dict(reason='initializing', offset_m=0., updates=0)

    def state(self):
        return dict(offset_m=self.offset, last_stamp=self.last_stamp if np.isfinite(self.last_stamp) else None,
                    last_pose=self.last_pose.tolist() if self.last_pose is not None else None,
                    updates=self.updates, reference_allowed=self.reference_allowed)

    def restore(self, state):
        offset = float(state['offset_m'])
        stamp = state.get('last_stamp')
        pose = np.asarray(state['last_pose'], dtype=float) if state.get('last_pose') is not None else None
        if (not np.isfinite(offset) or abs(offset) > self.config.get('bias_max_offset_m', .5) or
                (stamp is not None and not np.isfinite(stamp)) or
                (pose is not None and (pose.shape != (4, 4) or not np.isfinite(pose).all()))):
            raise ValueError('Invalid persisted height bias')
        self.offset = offset; self.last_stamp = float(stamp) if stamp is not None else -np.inf
        self.last_pose = pose; self.updates = int(state.get('updates', 0))
        self.reference_allowed = bool(state.get('reference_allowed', True))
        self.stats = dict(reason='restored', offset_m=self.offset, updates=self.updates)

    def update(self, observations, priors, *, stamp, source, pose, resolution):
        cfg = self.config
        old_height, old_variance, old_relief = priors
        xy = (observations['cells']+.5)*resolution
        valid = (np.isfinite(old_height) & np.isfinite(old_variance) &
                 (old_variance <= cfg.get('bias_max_prior_variance', .25)) &
                 (old_relief <= cfg.get('bias_max_relief_m', .06)) &
                 (observations['relief'] <= cfg.get('bias_max_relief_m', .06)) &
                 (np.linalg.norm(xy-pose[:2, 3], axis=1) <= cfg.get('bias_max_range_m', 12.)))
        count = int(valid.sum())
        stats = dict(reason='insufficient_overlap', offset_m=self.offset,
                     matched_cells=count, updates=self.updates, map_update_allowed=True)
        # Other sources share the same map datum. They cannot independently
        # estimate a different world height offset from a narrow camera view.
        if source != cfg.get('bias_reference_source', 'lidar'):
            stats['reason'] = 'shared_height_datum'
            if not self.reference_allowed and stamp <= self.last_stamp + cfg.get('bias_source_hold_sec', 3.):
                stats.update(reason='reference_scan_height_conflict', map_update_allowed=False)
            self.stats = stats
            return self.offset, stats['map_update_allowed']
        if stamp <= self.last_stamp:
            stats['reason'] = 'out_of_order_reference_scan'
            stats['map_update_allowed'] = False
            self.stats = stats
            return self.offset, False
        self.last_stamp = stamp
        moved = self.last_pose is not None and (
            np.linalg.norm(pose[:3, 3]-self.last_pose[:3, 3]) >= cfg.get('bias_min_pose_translation_m', .02)
            or np.max(np.abs(pose[:3, :3]-self.last_pose[:3, :3])) >= .001)
        if self.last_pose is None: self.last_pose = pose.copy()
        minimum = cfg.get('bias_min_cells', 80)
        if count >= minimum:
            sample_xy = xy[valid]
            residual = observations['height'][valid]-old_height[valid]
            median = float(np.median(residual))
            inliers = abs(residual-median) <= cfg.get('bias_inlier_band_m', .04)
            support = sample_xy[inliers]
            centered = support - np.median(support, axis=0) if len(support) else support
            quadrants = (centered[:, 0] >= 0).astype(int) + 2*(centered[:, 1] >= 0)
            spatial = (len(support) >= minimum and np.ptp(support, axis=0).min() >= cfg.get('bias_min_span_m', 3.)
                       and min(np.bincount(quadrants, minlength=4)) >= 5)
            coherent = spatial and inliers.mean() >= cfg.get('bias_min_inlier_ratio', .75)
            stats.update(residual_median_m=median, inlier_ratio=float(inliers.mean()),
                         spatially_distributed=bool(spatial))
            if coherent:
                plane = np.linalg.lstsq(np.column_stack((centered, np.ones(len(centered)))),
                                       residual[inliers], rcond=None)[0]
                residual_slope = float(np.linalg.norm(plane[:2]))
                coherent = residual_slope <= cfg.get('bias_max_residual_slope', .008)
                stats['residual_slope'] = residual_slope
            change = median-self.offset
            bounded = (abs(median) <= cfg.get('bias_max_offset_m', .5) and
                       abs(change) <= cfg.get('bias_max_change_m', .25))
            if cfg.get('height_bias_enabled', True) and coherent and bounded and moved:
                self.offset = median
                self.last_pose = pose.copy()
                self.updates += int(abs(change) >= .001)
                stats['reason'] = 'overlap_height_bias_applied'
            elif abs(change) <= cfg.get('bias_deadband_m', .01):
                stats['reason'] = 'height_consistent'
            else:
                stats['reason'] = 'bias_not_qualified'
            # A widespread disagreement is not hundreds of independent new
            # obstacles. Quarantine the scan instead of confirming a tilted map.
            conflict_ratio = float(np.mean(abs(residual-self.offset) > cfg.get('maximum_innovation_m', .10)))
            stats['conflict_ratio'] = conflict_ratio
            if conflict_ratio >= cfg.get('scan_conflict_ratio', .35):
                stats.update(reason='widespread_height_conflict', map_update_allowed=False)
        stats.update(offset_m=self.offset, updates=self.updates)
        self.stats = stats
        self.reference_allowed = stats['map_update_allowed']
        return self.offset, stats['map_update_allowed']

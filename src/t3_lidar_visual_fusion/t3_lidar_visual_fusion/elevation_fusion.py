"""Per-scan height fusion with bounded confidence and confirmed surface changes.

This filters observations in the existing map frame. It does not correct pose
drift, assume flat ground, or merge different XY cells. A cell still represents
one surface; vertical spread is retained separately from height uncertainty.
"""
import math
import numpy as np
from .legacy.semantic_grid import LayeredSemanticGridMap


EXTRA_FIELDS = (
    ('elevation_uncertainty', np.float32),
    ('elevation_last_stamp', np.float64),
    ('elevation_candidate', np.float32),
    ('elevation_candidate_variance', np.float32),
    ('elevation_candidate_stamp', np.float64),
    ('elevation_candidate_since', np.float64),
    ('elevation_candidate_count', np.uint8),
)

DEFAULTS = dict(sensor_std_m=.03, min_std_m=.02, process_variance_per_sec=.0004,
                max_age_sec=10., max_gain=.25, gate_sigma=3., gate_floor_m=.08,
                gate_ceiling_m=.20, max_measurement_std_m=.25,
                change_confirmations=3, change_min_span_sec=.20, change_max_gap_sec=1.)


def surface_observations(points, resolution, covariance, pose_origin, options):
    """One robust measurement per XY cell, without dividing pose noise by N.

    Return x, y, height, variance, low, high, spatial variance, pose variance. ROS fixed-axis
    pose covariance is projected with dz/dpose = [0,0,1,ry,-rx,0]. The result is
    a quality estimate, not a calibrated guarantee of absolute map accuracy.
    """
    cfg = dict(DEFAULTS, **options)
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    points = points[np.isfinite(points).all(axis=1)]
    if not len(points):
        return np.empty((0, 8))
    if not math.isfinite(resolution) or resolution <= 0:
        raise ValueError('Elevation resolution must be positive')
    cov = np.asarray(covariance, dtype=np.float64).reshape(6, 6)
    if not np.isfinite(cov).all() or np.any(np.diag(cov) < 0):
        raise ValueError('Invalid pose covariance for elevation fusion')
    cov = (cov + cov.T) * .5
    if np.linalg.eigvalsh(cov).min() < -1.e-9:
        raise ValueError('Indefinite pose covariance for elevation fusion')
    keys = np.floor(points[:, :2] / resolution).astype(np.int64)
    order = np.lexsort((points[:, 2], keys[:, 1], keys[:, 0]))
    p, keys = points[order], keys[order]
    starts = np.r_[0, np.flatnonzero(np.any(np.diff(keys, axis=0), axis=1)) + 1]
    ends = np.r_[starts[1:], len(p)]
    sizes = ends - starts
    groups = np.repeat(np.arange(len(starts)), sizes)
    # Retain a supported upper surface after the separate ground-clearance
    # selection. A minority rock surface must not disappear into a ground median.
    # Dense isolated extremes are trimmed; sparse high returns still need the
    # temporal confirmation below before replacing an already observed surface.
    low = p[starts + np.floor((sizes - 1) * .1).astype(int), 2]
    upper_indices=starts + np.ceil((sizes - 1) * .9).astype(int)
    high = p[upper_indices, 2].copy()
    width = np.clip(3. * cfg['sensor_std_m'], .03, .08)
    keep = np.abs(p[:, 2] - high[groups]) <= width
    support = np.bincount(groups[keep], minlength=len(starts))
    # Two agreeing returns can reject one isolated high spike even in a sparse
    # cell; a genuinely supported minority upper surface is still preserved.
    second=p[np.maximum(starts,upper_indices-1),2]
    alternate=np.abs(p[:,2]-second[groups])<=width
    alternate_support=np.bincount(groups[alternate],minlength=len(starts))
    isolated=(sizes>=3)&(support==1)&(alternate_support>=2)&(high-second>width)
    high[isolated]=second[isolated]
    keep=np.abs(p[:,2]-high[groups])<=width
    support=np.bincount(groups[keep],minlength=len(starts))
    height = np.bincount(groups[keep], p[keep, 2], minlength=len(starts)) / support
    spread = np.bincount(groups[keep], (p[keep, 2] - height[groups[keep]])**2,
                         minlength=len(starts)) / support
    xy = (keys[starts] + .5) * resolution
    relative = xy - np.asarray(pose_origin)[:2]
    jac = np.zeros((len(xy), 6))
    jac[:, 2] = 1.; jac[:, 3] = relative[:, 1]; jac[:, 4] = -relative[:, 0]
    pose_variance = np.maximum(0., np.einsum('ni,ij,nj->n', jac, cov, jac))
    variance = cfg['sensor_std_m']**2 + spread + pose_variance
    return np.column_stack((xy, height, variance, low, high, spread, pose_variance))


class TemporalElevationTile(LayeredSemanticGridMap):
    def __init__(self, *, fusion, **kwargs):
        super().__init__(**kwargs)
        self.fusion = dict(DEFAULTS, **fusion)
        if (any(not math.isfinite(float(self.fusion[k])) for k in DEFAULTS)
                or not 0 < self.fusion['max_gain'] <= 1 or self.fusion['min_std_m'] <= 0
                or self.fusion['sensor_std_m'] <= 0 or self.fusion['process_variance_per_sec'] < 0
                or self.fusion['max_age_sec'] <= 0 or self.fusion['gate_sigma'] <= 0
                or self.fusion['max_measurement_std_m'] < self.fusion['min_std_m']
                or not 0 < self.fusion['gate_floor_m'] <= self.fusion['gate_ceiling_m']
                or not 2 <= self.fusion['change_confirmations'] <= 255
                or self.fusion['change_min_span_sec'] <= 0 or self.fusion['change_max_gap_sec'] <= 0):
            raise ValueError('Invalid temporal elevation configuration')
        for name, dtype in EXTRA_FIELDS:
            setattr(self, name, np.zeros(self.elevation_count.shape, dtype=dtype))

    def clone_elevation_for_update(self):
        candidate = super().clone_elevation_for_update()
        for name, _ in EXTRA_FIELDS:
            setattr(candidate, name, getattr(self, name).copy())
        return candidate

    def _update_elevation_cell(self, row, column, z_values):
        raise ValueError('Temporal elevation requires update_observations with acquisition time and uncertainty')

    def elevation_variance_layer(self):
        return np.where(self.elevation_count > 0, self.elevation_uncertainty, np.nan).astype(np.float32)

    def roughness_layer(self):
        # Spatial dispersion, separate from confidence in the estimated height.
        n = self.elevation_count
        return np.where(n > 0, np.sqrt(self.elevation_M2 / np.maximum(n.astype(float) - 1, 1)), np.nan).astype(np.float32)

    def update_observations(self, observations, stamp):
        if not math.isfinite(stamp):
            raise ValueError('Elevation fusion requires a finite acquisition stamp')
        observations = np.asarray(observations, dtype=float).reshape(-1, 8)
        rows, cols, valid = self.xy_to_indices(observations[:, 0], observations[:, 1])
        cfg = self.fusion
        floor = cfg['min_std_m']**2
        stats = dict(accepted=0, pending=0, replaced=0, rejected=0)
        for r, c, obs in zip(rows[valid], cols[valid], observations[valid]):
            _, _, z, noise, low, high, spatial, pose_variance = map(float, obs)
            count = int(self.elevation_count[r, c])
            if count and stamp <= self.elevation_last_stamp[r, c]:
                stats['rejected'] += 1
                continue
            if (not np.isfinite(obs).all() or noise < 0 or spatial < 0 or pose_variance < 0
                    or pose_variance > noise + 1e-9 or low > high
                    or noise > cfg['max_measurement_std_m']**2):
                self.elevation_candidate_count[r, c] = 0
                stats['rejected'] += 1
                continue
            noise = max(floor, noise)
            previous_stamp = self.elevation_last_stamp[r, c]
            self.elevation_last_stamp[r, c] = stamp
            old = float(self.elevation_mean[r, c])
            age = min(cfg['max_age_sec'], max(0., stamp - previous_stamp)) if count else 0.
            prior = max(floor, float(self.elevation_uncertainty[r, c])) + cfg['process_variance_per_sec'] * age
            gate = min(cfg['gate_ceiling_m'], max(cfg['gate_floor_m'], cfg['gate_sigma'] * math.sqrt(prior + noise)))
            reset = not count
            pending = int(self.elevation_candidate_count[r, c]) > 0
            if count and (abs(z - old) > gate or (pending and abs(z - old) > cfg['gate_floor_m'])):
                n = int(self.elevation_candidate_count[r, c])
                candidate = float(self.elevation_candidate[r, c])
                agreement = min(.12, max(.06, 2. * math.sqrt(noise)))
                if (not n or stamp - self.elevation_candidate_stamp[r, c] > cfg['change_max_gap_sec']
                        or abs(z - candidate) > agreement):
                    n = 1; candidate = z
                    self.elevation_candidate_since[r, c] = stamp
                    self.elevation_candidate_variance[r, c] = noise
                else:
                    n = min(255, n + 1)
                    candidate += (z - candidate) / n
                    # Correlated scans must not drive candidate variance to zero.
                    self.elevation_candidate_variance[r, c] = max(noise, self.elevation_candidate_variance[r, c])
                self.elevation_candidate[r, c] = candidate
                self.elevation_candidate_stamp[r, c] = stamp
                self.elevation_candidate_count[r, c] = n
                if n < cfg['change_confirmations'] or stamp - self.elevation_candidate_since[r, c] < cfg['change_min_span_sec'] - 1.e-6:
                    # A conflict reduces confidence while the published height stays stable.
                    self.elevation_uncertainty[r, c] = max(prior, pose_variance, min((z - old)**2, .01))
                    stats['pending'] += 1
                    continue
                shift = candidate - z
                z = candidate; low += shift; high += shift
                noise = max(noise, float(self.elevation_candidate_variance[r, c]))
                reset = True; stats['replaced'] += 1
            self.elevation_candidate_count[r, c] = 0
            if reset:
                mean = z; uncertainty = noise; count = 1
                new_low, new_high, new_spatial = low, high, spatial
            else:
                gain = min(cfg['max_gain'], prior / (prior + noise))
                mean = old + gain * (z - old)
                # This conservative form also accounts for the capped gain.
                uncertainty = (1. - gain)**2 * prior + gain**2 * noise
                old_spatial = self.elevation_M2[r, c] / max(count - 1, 1)
                new_spatial = (1. - gain) * old_spatial + gain * spatial
                new_low = mean + (1. - gain) * (self.elevation_min[r, c] - old) + gain * (low - z)
                new_high = mean + (1. - gain) * (self.elevation_max[r, c] - old) + gain * (high - z)
                count = min(count + 1, 2**32 - 1)
            self.elevation_count[r, c] = count
            self.elevation_mean[r, c] = mean
            # All points share their pose error. Repeated measurements must
            # not average that common uncertainty away (216's conservative floor).
            self.elevation_uncertainty[r, c] = max(floor, pose_variance, uncertainty)
            self.elevation_M2[r, c] = new_spatial * max(count - 1, 1)
            self.elevation_min[r, c], self.elevation_max[r, c] = new_low, new_high
            stats['accepted'] += 1
        return stats

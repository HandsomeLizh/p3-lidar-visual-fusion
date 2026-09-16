"""Describe usable guarded visual inputs without claiming EKF acceptance."""
import math


def summarize_visual_evidence(verification):
    rows = sorted((m for m in verification.get("measurements", [])
                   if m.get("source") == "vision_accepted"),
                  key=lambda m: m["stamp_sec"])
    published = int(verification.get("counts", {}).get("vision_accepted", len(rows)))
    covariance_known = bool(rows) and all("pose_covariance_diagonal" in row for row in rows)
    old = verification.get("visual_checks", {})
    modes={s.get("visual_constraint_mode") for s in verification.get("states", [])}
    body_twist = old.get("input_mode") == "body_twist" or "body_twist" in modes
    absolute = not body_twist and (old.get("input_mode") == "absolute" or "absolute" in modes or any(
        s.get("fusion_strategy") in ("quality_gated_absolute_poses", "persistent_epoch_full_covariance")
        and s.get("visual_constraint_mode", "absolute") == "absolute"
        for s in verification.get("states", [])))

    def usable(row):
        diagonal = row.get("pose_covariance_diagonal", [])
        return len(diagonal) == 6 and all(math.isfinite(x) and 0 < x < 1e5 for x in diagonal)

    poses = sum(usable(row) for row in rows) if covariance_known else (0 if published == 0 else None)
    twists = sum(len(row.get("twist_covariance_diagonal", [])) == 6 and all(
        math.isfinite(v) and 0 < v < 1e5 for v in row["twist_covariance_diagonal"])
        for row in rows) if body_twist else None
    intervals = None
    if not absolute and not body_twist:
        if published < 2:
            intervals = 0
        elif covariance_known:
            intervals = sum(usable(a) and usable(b) and b["stamp_sec"] > a["stamp_sec"]
                            for a, b in zip(rows, rows[1:]))
    available = twists if body_twist else poses if absolute else intervals
    return dict(
        requested_source=old.get("requested_source"),
        input_mode="body_twist" if body_twist else "absolute" if absolute else "differential",
        raw_pose_observed=old.get("raw_pose_observed", False),
        guard_topic_observed=published > 0,
        guard_messages_observed=published,
        covariance_observed=covariance_known,
        non_anchor_poses_available=poses,
        body_motion_constraints_available=twists,
        absolute_input_observed=(poses > 0 if poses is not None else None) if absolute else None,
        differential_intervals_available=intervals,
        differential_input_observed=(intervals > 0 if intervals is not None else None),
        usable_visual_input_observed=(available > 0 if available is not None else None),
        fusion_mode_reported=old.get("fusion_mode_reported", old.get("fusion_active_observed", False)),
        ekf_measurement_acceptance_instrumented=False,
        qualification=("Body motion inputs require finite positive twist covariance; the visual pose is inactive." if body_twist else
                       "Absolute inputs require a finite, non-anchor pose." if absolute else
                       "Differential inputs require two successive finite, non-anchor poses.") +
        " Topic activity and reported mode do not prove EKF acceptance.")

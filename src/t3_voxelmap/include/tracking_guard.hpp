#pragma once
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <algorithm>
#include <cmath>
#include <stdexcept>

// A low point-to-plane residual alone does not validate a recovered trajectory.
// Limits are vehicle/stream limits, not per-route registration tuning.
struct TrackingLimits {
    double max_speed = 4., max_angular_speed = 2.;
    double max_step = 6., max_angle = 1.;
    Eigen::Vector3d body_origin_in_sensor=Eigen::Vector3d::Zero();
    void validate() const {
        for (double v : {max_speed, max_angular_speed, max_step, max_angle})
            if (!std::isfinite(v) || v <= 0.)
                throw std::invalid_argument("Tracking limits must be finite and positive");
    }
    template<class State>
    const char *check(const State &reference, const State &candidate, double dt, bool using_imu) const {
        if (!std::isfinite(dt) || dt <= 0. || !candidate.pos_end.allFinite() ||
            !candidate.rot_end.allFinite() || !candidate.vel_end.allFinite() ||
            !candidate.bias_g.allFinite() || !candidate.cov.allFinite()) return "nonfinite_state";
        const double distance=((candidate.pos_end+candidate.rot_end*body_origin_in_sensor)-
                               (reference.pos_end+reference.rot_end*body_origin_in_sensor)).norm();
        const double angle=Eigen::AngleAxisd(reference.rot_end.transpose()*candidate.rot_end).angle();
        if (distance > std::min(max_step, max_speed*dt+.05)) return "translation_discontinuity";
        if (angle > std::min(max_angle, max_angular_speed*dt+.02)) return "rotation_discontinuity";
        const Eigen::Vector3d body_velocity=candidate.vel_end+
            (using_imu?Eigen::Vector3d::Zero().eval():
             (candidate.rot_end*candidate.bias_g.cross(body_origin_in_sensor)).eval());
        const double velocity_limit=max_speed+(using_imu?max_angular_speed*body_origin_in_sensor.norm():0.);
        if (body_velocity.norm() > velocity_limit) return "unbounded_velocity";
        // In CV mode this slot is angular velocity. In IMU mode it is gyro bias.
        if (!using_imu && candidate.bias_g.norm() > max_angular_speed) return "unbounded_angular_velocity";
        return nullptr;
    }
};

template<class State>
void hold_unobserved_cv(State &state, const State &accepted) {
    state=accepted;
    // This is a registration seed, NOT a stationary/zero-velocity measurement.
    // The caller publishes this frame as unavailable with inflated covariance.
    state.vel_end.setZero(); state.bias_g.setZero();
    // Remove correlations to the discarded CV motion estimate. Retain the
    // last measured pose covariance; do not widen association indefinitely.
    state.cov.template block<6,18>(6,0).setZero();
    state.cov.template block<18,6>(0,6).setZero();
    state.cov.template block<3,3>(6,6).setIdentity();
    state.cov.template block<3,3>(9,9)=.5*Eigen::Matrix3d::Identity();
}

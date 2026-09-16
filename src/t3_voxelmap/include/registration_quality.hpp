#pragma once
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <Eigen/Eigenvalues>
#include <algorithm>
#include <cmath>

// Coordinates: [translation in map axes, right rotation in LiDAR axes].
// Keep overlap/residual validity separate from directional observability.
struct RegistrationQuality {
    using M6 = Eigen::Matrix<double, 6, 6>;
    M6 information = M6::Zero(), weighted_information = M6::Zero();
    double residual_square = 0.;
    int matched = 0, candidates = 0;
    void reset() {
        information.setZero(); weighted_information.setZero();
        residual_square = 0.; matched = candidates = 0;
    }
    void add(const Eigen::MatrixXd &h, const Eigen::VectorXd &residual, int considered) {
        candidates += considered;
        if (h.rows() == 0) return;
        const M6 value = h.leftCols(6).transpose() * h.leftCols(6);
        information += value;
        weighted_information += value;
        residual_square += residual.squaredNorm();
        matched += h.rows();
    }
    void set_weighted_information(const M6 &value) {
        weighted_information = .5 * (value + value.transpose());
    }
    double match_ratio() const { return double(matched) / std::max(1, candidates); }
    double residual_rms() const { return matched ? std::sqrt(residual_square / matched) : 1e6; }
    double lever() const {
        return std::max(.1, std::sqrt(std::max(1e-12, information.bottomRightCorner<3,3>().trace()) /
                                      std::max(1e-12, information.topLeftCorner<3,3>().trace())));
    }
    M6 scaling() const {
        M6 d = M6::Identity(); d.bottomRightCorner<3,3>() /= lever(); return d;
    }
    double translation_ratio() const {
        if (!matched) return 0.;
        Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> solver(information.topLeftCorner<3,3>());
        if (solver.info() != Eigen::Success) return 0.;
        return std::max(0., solver.eigenvalues()[0]) / std::max(1e-12, solver.eigenvalues()[2]);
    }
    double pose_ratio() const {
        if (!matched) return 0.;
        const M6 d = scaling();
        Eigen::SelfAdjointEigenSolver<M6> solver(d * information * d);
        if (solver.info() != Eigen::Success) return 0.;
        return std::max(0., solver.eigenvalues()[0]) / std::max(1e-12, solver.eigenvalues()[5]);
    }
    bool registration_usable() const {
        return matched >= 50 && match_ratio() >= .15 && residual_rms() <= .25 &&
               weighted_information.allFinite();
    }
    bool reliable() const {
        // Retained as a diagnostic, never an all-six-axis rejection switch.
        return registration_usable() && translation_ratio() >= .02 && pose_ratio() >= .003;
    }
    M6 directional_covariance() const {
        const M6 d = scaling();
        Eigen::SelfAdjointEigenSolver<M6> solver(d * weighted_information * d);
        if (solver.info() != Eigen::Success) return 1e6 * M6::Identity();
        // A null direction has 100 m^2 equivalent displacement uncertainty.
        // Finite weak directions use their absolute noise-weighted information,
        // not their ratio to a densely sampled strong plane.
        Eigen::Matrix<double,6,1> variance;
        for (int i=0; i<6; ++i) variance[i] = 1. / std::max(.01, solver.eigenvalues()[i]);
        M6 covariance = d * solver.eigenvectors() * variance.asDiagonal() * solver.eigenvectors().transpose() * d;
        return .5 * (covariance + covariance.transpose());
    }
};

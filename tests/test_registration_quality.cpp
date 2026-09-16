#include "registration_quality.hpp"
#include <cassert>
#include <iostream>
#include <random>

int main() {
    std::mt19937 random(5);
    std::normal_distribution<double> normal;
    const int count = 600;
    Eigen::MatrixXd plane = Eigen::MatrixXd::Zero(count, 6);
    Eigen::MatrixXd room = Eigen::MatrixXd::Zero(count, 6);
    Eigen::VectorXd residual = Eigen::VectorXd::Constant(count, .02);
    for (int i = 0; i < count; ++i) {
        Eigen::Vector3d p(normal(random)*5, normal(random)*5, normal(random)*5);
        Eigen::Vector3d n = Eigen::Vector3d::Unit(i % 3);
        room.block<1,3>(i,0) = n.transpose();
        room.block<1,3>(i,3) = p.cross(n).transpose();
        n = Eigen::Vector3d::UnitZ();
        plane.block<1,3>(i,0) = n.transpose();
        plane.block<1,3>(i,3) = p.cross(n).transpose();
    }
    RegistrationQuality q;
    q.add(plane, residual, count);
    assert(!q.reliable()); // A flat, densely sampled surface still leaves x/y/yaw unobserved.
    assert(q.registration_usable()); // Retain height/roll/pitch observations.
    q.set_weighted_information(q.information / (.04*.04));
    auto flat_cov=q.directional_covariance();
    assert(flat_cov(0,0)>99. && flat_cov(1,1)>99.);
    assert(flat_cov(2,2)<1e-4 && flat_cov(3,3)<1e-4 && flat_cov(4,4)<1e-4);
    assert(flat_cov(5,5)>.1);
    Eigen::Matrix3d rotation=Eigen::AngleAxisd(.7,Eigen::Vector3d(1,2,3).normalized()).toRotationMatrix();
    RegistrationQuality::M6 axes=RegistrationQuality::M6::Zero();
    axes.topLeftCorner<3,3>()=rotation;axes.bottomRightCorner<3,3>()=rotation;
    RegistrationQuality rotated;
    rotated.add(plane*axes.transpose(),residual,count);
    rotated.set_weighted_information(rotated.information/(.04*.04));
    assert((rotated.directional_covariance()-axes*flat_cov*axes.transpose()).norm()<1e-5);
    assert(std::abs(rotated.directional_covariance()(0,1))>1.);
    std::cout << "flat ratios: " << q.translation_ratio() << " " << q.pose_ratio() << "\n";
    q.reset(); q.add(room, residual, count);
    assert(q.reliable());
    std::cout << "room ratios: " << q.translation_ratio() << " " << q.pose_ratio() << "\n";
    q.reset(); q.add(room, residual.array()+.5, count);
    assert(!q.reliable()); // Rich directions do not excuse bad registration residuals.
    q.reset(); q.add(room, residual, 10*count);
    assert(!q.reliable()); // Too little scan overlap is also invalid.
    q.reset(); assert(!q.reliable());
    std::cout << "registration quality checks passed\n";
}

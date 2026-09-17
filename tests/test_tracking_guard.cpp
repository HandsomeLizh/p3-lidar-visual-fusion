#include "tracking_guard.hpp"
#include <Eigen/Eigenvalues>
#include <iostream>
struct State {
    Eigen::Vector3d pos_end=Eigen::Vector3d::Zero(),vel_end=Eigen::Vector3d::Zero(),bias_g=Eigen::Vector3d::Zero();
    Eigen::Matrix3d rot_end=Eigen::Matrix3d::Identity();
    Eigen::Matrix<double,18,18> cov=Eigen::Matrix<double,18,18>::Identity();
};
void require(bool value,const char *why){if(!value)throw std::runtime_error(why);}
int main(){
    TrackingLimits limits;limits.validate();State measured,candidate;
    measured.pos_end.x()=1.;measured.vel_end.x()=.2;
    candidate=measured;candidate.pos_end.x()+=.26;
    require(!limits.check(measured,candidate,1.3,false),"Normal motion rejected");
    candidate.pos_end.x()+=21.6;
    require(limits.check(measured,candidate,9.469,false),"Observed 21 metre jump accepted");
    require(limits.check(measured,candidate,1000.,false),"Waiting bypassed recovery bound");
    candidate=measured;candidate.vel_end.x()=7.95;
    require(limits.check(measured,candidate,1.3,false),"Runaway CV velocity accepted");
    for(int i=0;i<300;++i){
        candidate.pos_end+=candidate.vel_end*1.3;
        hold_unobserved_cv(candidate,measured);
        require((candidate.pos_end-measured.pos_end).norm()<1e-12,"Rejected CV scan drifted");
        require(candidate.vel_end.isZero()&&candidate.bias_g.isZero(),"Unobserved velocity reused");
        require(Eigen::SelfAdjointEigenSolver<decltype(candidate.cov)>(candidate.cov).eigenvalues().minCoeff()>=0.,"Rollback covariance not PSD");
    }
    candidate.pos_end.x()+=.26;
    require(!limits.check(measured,candidate,1.3,false),"Nearby recovery rejected");
    candidate.bias_g.z()=3.;
    require(limits.check(measured,candidate,1.3,false),"CV angular runaway accepted");
    require(!limits.check(measured,candidate,1.3,true),"IMU gyro bias confused with CV angular rate");
    TrackingLimits slow;slow.max_speed=.3;slow.body_origin_in_sensor=Eigen::Vector3d(-.607,0.,0.);
    measured=State{};candidate=measured;
    candidate.rot_end=Eigen::AngleAxisd(.5,Eigen::Vector3d::UnitZ()).toRotationMatrix();
    candidate.pos_end=slow.body_origin_in_sensor-candidate.rot_end*slow.body_origin_in_sensor;
    candidate.bias_g.z()=.5;
    candidate.vel_end=-candidate.rot_end*candidate.bias_g.cross(slow.body_origin_in_sensor);
    require(!slow.check(measured,candidate,1.,false),"Off-centre LiDAR turn mistaken for body translation");
    candidate.pos_end.x()+=.9;
    require(slow.check(measured,candidate,1.4,false),"0.9m fake displacement accepted at simulation speed");
    std::cout<<"PASS: 21m recovery rejection, 300 failed CV scans held, nearby recovery, IMU state semantics\n";
}

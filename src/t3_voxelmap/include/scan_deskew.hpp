#pragma once
#include "optional_imu.hpp"
#include <map>

// The IMU prediction and all point timestamps share one clock. Points are
// transformed into the LiDAR frame at the END of this scan (including lever arm).
inline Eigen::Matrix4d pose_at(const std::vector<fusion_imu::PoseSample> &poses, double stamp) {
    if(poses.size()<2 || stamp<poses.front().stamp-1e-6 || stamp>poses.back().stamp+1e-6)
        throw std::runtime_error("IMU trajectory does not cover the complete scan");
    auto hi=std::lower_bound(poses.begin(),poses.end(),stamp,
        [](const auto &p,double t){return p.stamp<t;});
    if(hi==poses.end())hi=poses.end()-1;
    auto lo=hi==poses.begin()?hi:hi-1;
    const double dt=hi->stamp-lo->stamp;
    const double alpha=dt>0.?std::clamp((stamp-lo->stamp)/dt,0.,1.):0.;
    Eigen::Matrix4d t=Eigen::Matrix4d::Identity();
    t.topLeftCorner<3,3>()=Eigen::Quaterniond(lo->rotation).slerp(alpha,Eigen::Quaterniond(hi->rotation)).toRotationMatrix();
    t.topRightCorner<3,1>()=(1.-alpha)*lo->position+alpha*hi->position;
    return t;
}

inline void deskew_scan(PointCloudXYZI &cloud,const std::vector<double> &relative_times,
                        double scan_start,const std::vector<fusion_imu::PoseSample> &poses) {
    if(relative_times.size()!=cloud.size())throw std::runtime_error("Point timing size mismatch");
    if(poses.size()<2)throw std::runtime_error("IMU trajectory is empty");
    const auto end_inverse=pose_at(poses,poses.back().stamp).inverse().eval();
    std::map<double,Eigen::Matrix4d> transforms;
    for(size_t i=0;i<cloud.size();++i) {
        const double relative=relative_times[i];
        auto found=transforms.find(relative);
        if(found==transforms.end()) {
            const auto t=(end_inverse*pose_at(poses,scan_start+relative)).eval();
            found=transforms.emplace(relative,t).first;
        }
        const auto &t=found->second;
        const V3D p=t.topLeftCorner<3,3>()*V3D(cloud[i].x,cloud[i].y,cloud[i].z)+t.topRightCorner<3,1>();
        cloud[i].x=p.x();cloud[i].y=p.y();cloud[i].z=p.z();
    }
}

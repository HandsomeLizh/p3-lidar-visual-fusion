#pragma once
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <algorithm>
#include <deque>
#include <string>
#include <cmath>

// Unfused frontend increments only. No global visual/LiDAR frame alignment or
// filtered output is allowed here: that would validate LiDAR against itself.
class IndependentVisualMotion {
public:
    struct Sample {
        double stamp=0.;
        std::string epoch;
        Eigen::Matrix4d body=Eigen::Matrix4d::Identity();
        bool valid=false;
        // A frontend epoch origin is not a motion observation. It may be the
        // previous endpoint only after a subsequent valid tracked increment.
        bool epoch_origin=false;
    };
    struct Result {
        bool available=false,consistent=true;
        std::string reason="no_same_epoch_visual_pair";
        double translation_error=0.,rotation_error=0.;
        double translation_limit=0.,rotation_limit=0.;
    };
private:
    std::deque<Sample> samples_;
public:
    void observe(const Sample &sample) {
        if(!std::isfinite(sample.stamp) || sample.epoch.empty() ||
           sample.epoch=="odom" || sample.epoch=="map" ||
           (!samples_.empty() && sample.stamp<=samples_.back().stamp))return;
        samples_.push_back(sample);
        while(samples_.size()>64 || sample.stamp-samples_.front().stamp>20.)samples_.pop_front();
    }
    const Sample *at(double stamp) const {
        const Sample *best=nullptr;double distance=.08;
        for(const auto &s:samples_)if(std::abs(s.stamp-stamp)<=distance){distance=std::abs(s.stamp-stamp);best=&s;}
        return best;
    }
    bool prediction(double previous_stamp,double stamp,const Eigen::Matrix4d &previous_lidar,
                    const Eigen::Matrix4d &base_from_lidar,Eigen::Matrix4d &expected) const {
        const Sample *a=at(previous_stamp),*b=at(stamp);
        if(!a || !b || (!a->valid && !a->epoch_origin) || !b->valid || a->epoch!=b->epoch ||
           b->stamp<=a->stamp || b->stamp-a->stamp>6.)return false;
        expected=previous_lidar*base_from_lidar.inverse()*a->body.inverse()*b->body*base_from_lidar;
        return expected.allFinite();
    }
    Result check(const Eigen::Matrix4d &previous,const Eigen::Matrix4d &candidate,
                 const Eigen::Matrix4d &expected) const {
        Result result;result.available=true;
        result.translation_error=(candidate.topRightCorner<3,1>()-expected.topRightCorner<3,1>()).norm();
        result.rotation_error=Eigen::AngleAxisd(expected.topLeftCorner<3,3>().transpose()*candidate.topLeftCorner<3,3>()).angle();
        const double distance=std::max((candidate.topRightCorner<3,1>()-previous.topRightCorner<3,1>()).norm(),
                                       (expected.topRightCorner<3,1>()-previous.topRightCorner<3,1>()).norm());
        const double angle=std::max(Eigen::AngleAxisd(previous.topLeftCorner<3,3>().transpose()*candidate.topLeftCorner<3,3>()).angle(),
                                   Eigen::AngleAxisd(previous.topLeftCorner<3,3>().transpose()*expected.topLeftCorner<3,3>()).angle());
        // Same fixed model-error allowances used by the fusion motion check.
        // Applied only to weak-geometry LiDAR, after independent frontend gates.
        result.translation_limit=.15+.20*distance;
        result.rotation_limit=.14+.20*angle;
        result.consistent=result.translation_error<=result.translation_limit && result.rotation_error<=result.rotation_limit;
        result.reason=result.consistent?"independent_visual_motion_consistent":"weak_lidar_visual_motion_conflict";
        return result;
    }
};

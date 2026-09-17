#pragma once
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <algorithm>
#include <deque>
#include <string>
#include <cmath>
#include <optional>
#include <stdexcept>

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
        double position_variance=.001,rotation_variance=.0001;
        size_t segment=0;
        double path_length=0.;
    };
    struct Result {
        bool available=false,consistent=true;
        std::string reason="no_same_epoch_visual_pair";
        double translation_error=0.,rotation_error=0.;
        double translation_limit=0.,rotation_limit=0.;
    };
    struct PredictionQuality {
        std::string reason="no_same_epoch_visual_pair";
        double anchor_stamp=-1.,elapsed=0.,path_length=0.;
        double position_variance=0.,rotation_variance=0.;
        bool retained_anchor=false;
    };
private:
    std::deque<Sample> samples_;
    std::optional<Sample> anchor_,last_valid_;
    double anchor_request_=-1.;
    size_t segment_=0;
    mutable PredictionQuality prediction_quality_;
public:
    // The horizon applies between validated frontend poses, not between LiDAR
    // successes. A pinned co-timed anchor survives the small lookup queue.
    double max_gap=6.,max_speed=4.,max_angular_speed=2.,max_step=6.,max_angle=1.;
    double max_position_variance=.5,max_rotation_variance=.1;
    void validate() const {
        for(double v:{max_gap,max_speed,max_angular_speed,max_step,max_angle,
                      max_position_variance,max_rotation_variance})
            if(!std::isfinite(v) || v<=0.)throw std::invalid_argument("Invalid independent visual bounds");
        if(max_gap>120.)throw std::invalid_argument("Visual recovery horizon exceeds frontend bound");
    }
    const PredictionQuality &prediction_quality() const {return prediction_quality_;}
    void retain_anchor(double stamp) {
        anchor_request_=stamp;anchor_.reset();
        const Sample *sample=at(stamp);
        if(sample && (sample->valid || sample->epoch_origin))anchor_=*sample;
    }
    void observe(Sample sample) {
        if(!std::isfinite(sample.stamp) || sample.epoch.empty() ||
            sample.epoch=="odom" || sample.epoch=="map" ||
            (!samples_.empty() && sample.stamp<=samples_.back().stamp))return;
        if(!samples_.empty() && sample.epoch!=samples_.back().epoch) {
            ++segment_;last_valid_.reset();
        }
        const auto rotation=sample.body.topLeftCorner<3,3>();
        if(!sample.body.allFinite() ||
           !(rotation*rotation.transpose()).isApprox(Eigen::Matrix3d::Identity(),1e-6) ||
           std::abs(rotation.determinant()-1.)>1e-6 ||
           !std::isfinite(sample.position_variance) || !std::isfinite(sample.rotation_variance) ||
           sample.position_variance<0. || sample.rotation_variance<0.)
            sample.valid=sample.epoch_origin=false;
        // A held identity with invalid covariance is not a fresh epoch origin.
        // Otherwise repeated rejected images could extend a blind interval.
        if(sample.epoch_origin && last_valid_ && sample.epoch==last_valid_->epoch)
            sample.epoch_origin=false;
        if(sample.valid || sample.epoch_origin) {
            if(last_valid_) {
                const double dt=sample.stamp-last_valid_->stamp;
                const double distance=(sample.body.topRightCorner<3,1>()-last_valid_->body.topRightCorner<3,1>()).norm();
                const double angle=Eigen::AngleAxisd(last_valid_->body.topLeftCorner<3,3>().transpose()*rotation).angle();
                if(sample.epoch!=last_valid_->epoch || dt>max_gap)++segment_;
                else if(distance>std::min(max_step,max_speed*dt+.05) ||
                        angle>std::min(max_angle,max_angular_speed*dt+.02)) {
                    // A same-epoch pose jump breaks the chain, even if a later
                    // sample looks locally quiet. It cannot rejoin an old pin.
                    ++segment_;sample.valid=sample.epoch_origin=false;last_valid_.reset();
                } else sample.path_length=last_valid_->path_length+distance;
            }
            sample.segment=segment_;
            if(sample.valid || sample.epoch_origin)last_valid_=sample;
        } else sample.segment=segment_;
        samples_.push_back(sample);
        if((sample.valid || sample.epoch_origin) && std::abs(sample.stamp-anchor_request_)<=.08 &&
           (!anchor_ || std::abs(sample.stamp-anchor_request_)<std::abs(anchor_->stamp-anchor_request_)))anchor_=sample;
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
        prediction_quality_=PredictionQuality{};
        if(!a && anchor_ && std::abs(anchor_->stamp-previous_stamp)<=.08) {
            a=&*anchor_;prediction_quality_.retained_anchor=true;
        }
        if(!a || !b)return false;
        prediction_quality_.anchor_stamp=a->stamp;
        if((!a->valid && !a->epoch_origin) || !b->valid) {
            prediction_quality_.reason="invalid_visual_endpoint";return false;
        }
        if(a->epoch!=b->epoch || a->segment!=b->segment || b->stamp<=a->stamp) {
            prediction_quality_.reason="visual_chain_disconnected";return false;
        }
        const double elapsed=b->stamp-a->stamp;
        const double path=std::max(0.,b->path_length-a->path_length);
        const double distance=(b->body.topRightCorner<3,1>()-a->body.topRightCorner<3,1>()).norm();
        // Conservative endpoint, lever-arm and distance/time drift allowances.
        // Do not add the same keyframe measurement once per repeated scan.
        const double rv=3.*(a->rotation_variance+b->rotation_variance)+std::pow(.001*path,2)+.000001*elapsed;
        const double pv=3.*(a->position_variance+b->position_variance+
            distance*distance*(a->rotation_variance+b->rotation_variance))+std::pow(.01*path,2)+.00002*elapsed+
            base_from_lidar.topRightCorner<3,1>().squaredNorm()*rv;
        prediction_quality_.elapsed=elapsed;prediction_quality_.path_length=path;
        prediction_quality_.position_variance=pv;prediction_quality_.rotation_variance=rv;
        if(pv>max_position_variance || rv>max_rotation_variance) {
            prediction_quality_.reason="visual_bridge_uncertainty_exceeded";return false;
        }
        expected=previous_lidar*base_from_lidar.inverse()*a->body.inverse()*b->body*base_from_lidar;
        prediction_quality_.reason=expected.allFinite()?"qualified_visual_chain":"nonfinite_visual_prediction";
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

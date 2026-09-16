#pragma once
// Optional inertial prediction for the isolated ROS2 VoxelMap adapter.
// Measurement update/map remain VoxelMap. No fabricated IMU and no pose reset.
#include "common_lib.h"
#include <algorithm>
#include <deque>
#include <string>
#include <stdexcept>

namespace fusion_imu {
using Cov = Eigen::Matrix<double,18,18>;
using AuxCov = Eigen::Matrix<double,9,9>;
inline M3D cross_matrix(const V3D &v) { M3D m; m << SKEW_SYM_MATRX(v); return m; }

struct Sample {
    double stamp = 0.;
    V3D gyro = V3D::Zero(); // rad/s, expressed in LiDAR axes after push()
    V3D accel = V3D::Zero(); // m/s^2 specific force, includes gravity at rest
};

struct PoseSample {
    double stamp;
    M3D rotation;
    V3D position;
};

struct Config {
    bool enabled = true, calibrated = false;
    M3D lidar_from_imu_rotation = M3D::Identity();
    V3D imu_origin_in_lidar = V3D::Zero();
    double buffer_seconds = 8., max_gap = .05, end_tolerance = .015;
    size_t max_samples = 4000, init_min_samples = 50;
    double init_seconds = 1., gravity = 9.81;
    double stationary_gyro_std = .02, stationary_accel_std = .15;
    double stationary_max_rate = .08, stationary_max_speed = .08;
    double gravity_tolerance = .6, max_rate = 30., max_acceleration = 200.;
    // Continuous white-noise densities and bias random walks (SI / sqrt(Hz)).
    double gyro_noise = .01, accel_noise = .1;
    double gyro_bias_noise = .0001, accel_bias_noise = .001;
    double init_gyro_bias_std = .02, init_accel_bias_std = .1, init_gravity_std = .1;
    double cv_velocity_noise = 1., cv_omega_noise = .5;
    int recovery_scans = 3;
};

struct Report {
    bool using_imu = false, initialized = false;
    std::string reason = "no_imu";
    size_t samples_used = 0, buffered = 0, accepted = 0, rejected = 0, evicted = 0;
    size_t transitions = 0, imu_scans = 0, cv_scans = 0;
    double integrated_seconds = 0.;
};

class OptionalImu {
    Config cfg_;
    std::deque<Sample> samples_;
    bool initialized_ = false, active_ = false, ever_active_ = false;
    int recovery_ = 0;
    double last_received_ = -1., aux_stamp_ = -1.;
    V3D saved_bg_ = V3D::Zero(), saved_ba_ = V3D::Zero(), saved_g_ = V3D::Zero();
    V3D last_omega_ = V3D::Zero(), last_gyro_ = V3D::Zero();
    AuxCov saved_cov_ = AuxCov::Identity();
    Report report_;

    static Sample mix(const Sample &a, const Sample &b, double t) {
        const double w = std::clamp((t-a.stamp)/(b.stamp-a.stamp),0.,1.);
        return Sample{t, (1.-w)*a.gyro+w*b.gyro, (1.-w)*a.accel+w*b.accel};
    }

    bool interval(double start, double end, std::vector<Sample> &out) {
        if (samples_.empty()) { report_.reason="no_imu"; return false; }
        if (samples_.front().stamp > start+1e-9) {
            report_.reason="missing_interval_start"; return false;
        }
        if (end-samples_.back().stamp > cfg_.end_tolerance+1e-9) {
            report_.reason="imu_timeout"; return false;
        }
        auto it=std::upper_bound(samples_.begin(),samples_.end(),start,
             [](double t,const Sample &s){return t<s.stamp;});
        if(it==samples_.begin()){report_.reason="missing_interval_start";return false;}
        auto prev=it-1;
        if (it!=samples_.end() && it->stamp-prev->stamp>cfg_.max_gap+1e-9 &&
            start>prev->stamp+1e-9) {report_.reason="imu_gap";return false;}
        Sample head=*prev;head.stamp=start;
        if(it!=samples_.end())head=mix(*prev,*it,start);
        out.push_back(head);
        for(;it!=samples_.end() && it->stamp<end; ++it) {
            if(it->stamp-prev->stamp>cfg_.max_gap+1e-9){report_.reason="imu_gap";return false;}
            out.push_back(*it);prev=it;
        }
        Sample tail=*prev;tail.stamp=end;
        if(it!=samples_.end()) {
            if(it->stamp-prev->stamp>cfg_.max_gap+1e-9){report_.reason="imu_gap";return false;}
            tail=mix(*prev,*it,end);
        } else if(end-prev->stamp>cfg_.end_tolerance+1e-9) {
            report_.reason="imu_timeout";return false;
        }
        if(end-out.back().stamp>1e-9)out.push_back(tail);
        return out.size()>=2;
    }

    bool initialize(const StatesGroup &state, double stamp) {
        if(initialized_)return true;
        if(state.vel_end.norm()>cfg_.stationary_max_speed ||
           state.bias_g.norm()>cfg_.stationary_max_rate) {
            report_.reason="waiting_for_stationary_initialization";return false;
        }
        std::vector<Sample> window;
        if(!interval(stamp-cfg_.init_seconds,stamp,window)) {
            report_.reason="warming_up";return false;
        }
        if(window.size()<cfg_.init_min_samples){report_.reason="warming_up";return false;}
        V3D a=V3D::Zero(),w=V3D::Zero(),a2=V3D::Zero(),w2=V3D::Zero();
        for(const auto &s:window){a+=s.accel;w+=s.gyro;a2+=s.accel.cwiseProduct(s.accel);w2+=s.gyro.cwiseProduct(s.gyro);}
        const double n=window.size();a/=n;w/=n;
        const V3D av=(a2/n-a.cwiseProduct(a)).cwiseMax(0.);
        const V3D wv=(w2/n-w.cwiseProduct(w)).cwiseMax(0.);
        if(std::abs(a.norm()-cfg_.gravity)>cfg_.gravity_tolerance ||
           w.norm()>cfg_.stationary_max_rate ||
           av.maxCoeff()>cfg_.stationary_accel_std*cfg_.stationary_accel_std ||
           wv.maxCoeff()>cfg_.stationary_gyro_std*cfg_.stationary_gyro_std) {
            report_.reason="waiting_for_stationary_initialization";return false;
        }
        saved_bg_=w;saved_ba_.setZero();
        saved_g_=-state.rot_end*a.normalized()*cfg_.gravity;
        saved_cov_.setZero();
        saved_cov_.block<3,3>(0,0).diagonal().setConstant(cfg_.init_gyro_bias_std*cfg_.init_gyro_bias_std);
        saved_cov_.block<3,3>(3,3).diagonal().setConstant(cfg_.init_accel_bias_std*cfg_.init_accel_bias_std);
        saved_cov_.block<3,3>(6,6).diagonal().setConstant(cfg_.init_gravity_std*cfg_.init_gravity_std);
        aux_stamp_=stamp;initialized_=true;report_.initialized=true;
        return true;
    }

    void enter_imu(StatesGroup &s, double stamp) {
        // The upstream CV model stores angular RATE in bias_g. Preserve the
        // actual inertial biases separately; never reinterpret CV omega as bias.
        Cov covariance=Cov::Zero();
        covariance.topLeftCorner<9,9>()=2.*s.cov.topLeftCorner<9,9>();
        const double dt=std::max(0.,stamp-aux_stamp_);
        saved_cov_.block<3,3>(0,0).diagonal().array()+=cfg_.gyro_bias_noise*cfg_.gyro_bias_noise*dt;
        saved_cov_.block<3,3>(3,3).diagonal().array()+=cfg_.accel_bias_noise*cfg_.accel_bias_noise*dt;
        covariance.bottomRightCorner<9,9>()=2.*saved_cov_;
        // Doubling both retained diagonal blocks conservatively bounds the
        // discarded cross-covariance (2 blockdiag(P)-P is PSD).
        s.cov=covariance;s.bias_g=saved_bg_;s.bias_a=saved_ba_;s.gravity=saved_g_;
        active_=ever_active_=true;++report_.transitions;
    }

    void leave_imu(StatesGroup &s,double stamp) {
        saved_bg_=s.bias_g;saved_ba_=s.bias_a;saved_g_=s.gravity;
        saved_cov_=2.*s.cov.bottomRightCorner<9,9>();aux_stamp_=stamp;
        Cov covariance=Cov::Zero();
        covariance.topLeftCorner<9,9>()=2.*s.cov.topLeftCorner<9,9>();
        covariance.block<3,3>(9,9)=saved_cov_.topLeftCorner<3,3>();
        covariance.block<3,3>(9,9).diagonal().array()+=cfg_.gyro_noise*cfg_.gyro_noise/cfg_.max_gap;
        covariance.block<6,6>(12,12).diagonal().setConstant(INIT_COV);
        s.cov=covariance;s.bias_g=last_gyro_-saved_bg_;s.bias_a.setZero();s.gravity.setZero();
        active_=false;recovery_=0;++report_.transitions;
    }

    void integrate(StatesGroup &s,const Sample &a,const Sample &b) {
        const double dt=b.stamp-a.stamp;
        const V3D w0=a.gyro-s.bias_g,w1=b.gyro-s.bias_g,w=.5*(w0+w1);
        const V3D f=.5*(a.accel+b.accel)-s.bias_a;
        const V3D &lever=cfg_.imu_origin_in_lidar;
        const M3D r0=s.rot_end,rm=r0*Exp(w,.5*dt),r1=r0*Exp(w,dt);
        // Integrate translation at the IMU origin, then transform back to the
        // LiDAR origin. This includes tangential AND centripetal lever-arm motion.
        V3D pi=s.pos_end+r0*lever;
        V3D vi=s.vel_end+r0*w0.cross(lever);
        const V3D acceleration=rm*f+s.gravity;
        pi+=vi*dt+.5*acceleration*dt*dt;vi+=acceleration*dt;
        s.pos_end=pi-r1*lever;s.vel_end=vi-r1*w1.cross(lever);s.rot_end=r1;

        Cov jin=Cov::Identity(),jout=Cov::Identity(),transition=Cov::Identity();
        jin.block<3,3>(3,0)=-r0*cross_matrix(lever);
        jin.block<3,3>(6,0)=-r0*cross_matrix(w0.cross(lever));
        jin.block<3,3>(6,9)=r0*cross_matrix(lever);
        jout.block<3,3>(3,0)=r1*cross_matrix(lever);
        jout.block<3,3>(6,0)=r1*cross_matrix(w1.cross(lever));
        jout.block<3,3>(6,9)=-r1*cross_matrix(lever);
        transition.block<3,3>(0,0)=Exp(w,-dt);
        transition.block<3,3>(0,9)=-M3D::Identity()*dt;
        transition.block<3,3>(3,6)=M3D::Identity()*dt;
        const M3D af=-rm*cross_matrix(f);
        transition.block<3,3>(3,0)=.5*af*dt*dt;
        transition.block<3,3>(6,0)=af*dt;
        transition.block<3,3>(3,12)=-.5*rm*dt*dt;
        transition.block<3,3>(6,12)=-rm*dt;
        transition.block<3,3>(3,15)=M3D::Identity()*.5*dt*dt;
        transition.block<3,3>(6,15)=M3D::Identity()*dt;
        const Cov total=jout*transition*jin;
        Eigen::Matrix<double,18,12> noise=Eigen::Matrix<double,18,12>::Zero();
        noise.block<3,3>(0,0)=M3D::Identity()*cfg_.gyro_noise;
        noise.block<3,3>(3,3)=.5*rm*dt*cfg_.accel_noise;
        noise.block<3,3>(6,3)=rm*cfg_.accel_noise;
        noise.block<3,3>(9,6)=M3D::Identity()*cfg_.gyro_bias_noise;
        noise.block<3,3>(12,9)=M3D::Identity()*cfg_.accel_bias_noise;
        const auto transformed=(jout*noise).eval();
        Cov process=transformed*transformed.transpose()*dt;
        // Additional conservative endpoint-rate uncertainty for a nonzero lever
        // arm; samples correlate with integration, so use a factor-two bound.
        if(lever.squaredNorm()>1e-12) {
            Eigen::Matrix<double,18,6> e=Eigen::Matrix<double,18,6>::Zero();
            e.block<3,3>(3,0)=-r0*cross_matrix(lever)*dt;
            e.block<3,3>(6,0)=-r0*cross_matrix(lever);
            e.block<3,3>(6,3)=r1*cross_matrix(lever);
            process=2.*(process+e*e.transpose()*cfg_.gyro_noise*cfg_.gyro_noise/dt);
        }
        s.cov=total*s.cov*total.transpose()+process;
        s.cov=.5*(s.cov+s.cov.transpose()).eval();
        last_omega_=w1;last_gyro_=b.gyro;
    }

public:
    explicit OptionalImu(const Config &cfg):cfg_(cfg) {
        const double values[]={cfg.buffer_seconds,cfg.max_gap,cfg.end_tolerance,cfg.init_seconds,
            cfg.gravity,cfg.stationary_gyro_std,cfg.stationary_accel_std,cfg.stationary_max_rate,
            cfg.stationary_max_speed,cfg.gravity_tolerance,cfg.max_rate,cfg.max_acceleration,
            cfg.gyro_noise,cfg.accel_noise,cfg.gyro_bias_noise,cfg.accel_bias_noise,
            cfg.init_gyro_bias_std,cfg.init_accel_bias_std,cfg.init_gravity_std,
            cfg.cv_velocity_noise,cfg.cv_omega_noise};
        for(double v:values)if(!std::isfinite(v)||v<=0.)throw std::invalid_argument("IMU parameters must be finite and positive");
        if(cfg.max_samples<cfg.init_min_samples || cfg.init_min_samples<3 ||
           cfg.buffer_seconds<cfg.init_seconds || cfg.end_tolerance>cfg.max_gap ||cfg.recovery_scans<1)
            throw std::invalid_argument("Inconsistent IMU buffer/initialization/recovery parameters");
        const auto &r=cfg.lidar_from_imu_rotation;
        if(!r.allFinite()||!cfg.imu_origin_in_lidar.allFinite()||
           (r*r.transpose()-M3D::Identity()).norm()>1e-4||std::abs(r.determinant()-1.)>1e-4)
            throw std::invalid_argument("IMU extrinsic must be a rigid proper rotation");
    }

    bool push(Sample sample) {
        if(!cfg_.enabled)return false;
        if(!std::isfinite(sample.stamp)||sample.stamp<=last_received_||
           !sample.gyro.allFinite()||!sample.accel.allFinite()||
           sample.gyro.norm()>cfg_.max_rate||sample.accel.norm()>cfg_.max_acceleration) {
            ++report_.rejected;return false;
        }
        last_received_=sample.stamp;
        sample.gyro=cfg_.lidar_from_imu_rotation*sample.gyro;
        sample.accel=cfg_.lidar_from_imu_rotation*sample.accel;
        samples_.push_back(sample);++report_.accepted;
        while(samples_.size()>cfg_.max_samples ||
              (!samples_.empty() && sample.stamp-samples_.front().stamp>cfg_.buffer_seconds)) {
            samples_.pop_front();++report_.evicted;
        }
        report_.buffered=samples_.size();return true;
    }
    void reject(){++report_.rejected;}
    const Report &report()const{return report_;}
    bool active()const{return active_;}

    static void predict_cv(StatesGroup &s,double dt,double velocity_noise,double omega_noise) {
        // Exact pre-IMU VoxelMap model, retained for no-IMU regression equality.
        Cov f=Cov::Identity(),q=Cov::Zero();
        f.block<3,3>(0,0)=Exp(s.bias_g,-dt);
        f.block<3,3>(0,9)=M3D::Identity()*dt;
        f.block<3,3>(3,6)=M3D::Identity()*dt;
        q.block<3,3>(9,9).diagonal().setConstant(omega_noise*dt*dt);
        q.block<3,3>(6,6).diagonal().setConstant(velocity_noise*dt*dt);
        s.cov=f*s.cov*f.transpose()+q;
        s.rot_end=s.rot_end*Exp(s.bias_g,dt);s.pos_end+=s.vel_end*dt;
    }

    bool physically_valid(const StatesGroup &s) const {
        return s.rot_end.allFinite()&&s.pos_end.allFinite()&&s.vel_end.allFinite()&&s.cov.allFinite()&&
            (!active_||(s.bias_g.norm()<1. && s.bias_a.norm()<5. &&
                         std::abs(s.gravity.norm()-cfg_.gravity)<2.));
    }

    void predict(StatesGroup &s,double start,double end,bool first=false,
                 std::vector<PoseSample> *trajectory=nullptr) {
        if(trajectory)trajectory->clear();
        report_.samples_used=0;report_.integrated_seconds=0.;report_.using_imu=false;
        report_.reason=cfg_.enabled?"no_imu":"disabled";
        const double dt=first?.1:end-start;
        if(!std::isfinite(dt)||dt<=0.)throw std::invalid_argument("Prediction interval must be positive");
        std::vector<Sample> segment;
        bool use=cfg_.enabled&&cfg_.calibrated&&!first;
        if(cfg_.enabled&&!cfg_.calibrated)report_.reason="calibration_unconfirmed";
        if(use)use=interval(start,end,segment)&&initialize(s,start);
        if(use && !active_ && ever_active_ && ++recovery_<cfg_.recovery_scans) {
            use=false;report_.reason="recovering_imu";
        } else if(!use)recovery_=0;
        if(use) {
            if(!active_)enter_imu(s,start);
            StatesGroup before=s;
            if(trajectory)trajectory->push_back({start,s.rot_end,s.pos_end});
            for(size_t i=1;i<segment.size();++i) {
                integrate(s,segment[i-1],segment[i]);
                if(trajectory)trajectory->push_back({segment[i].stamp,s.rot_end,s.pos_end});
            }
            if(physically_valid(s)) {
                report_.using_imu=true;report_.reason="imu";report_.samples_used=segment.size();
                report_.integrated_seconds=end-start;++report_.imu_scans;return;
            }
            s=before;report_.reason="invalid_prediction";
        }
        if(trajectory)trajectory->clear();
        if(active_)leave_imu(s,start);
        predict_cv(s,dt,cfg_.cv_velocity_noise,cfg_.cv_omega_noise);++report_.cv_scans;
    }
};
} // namespace fusion_imu

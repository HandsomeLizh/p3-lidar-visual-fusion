// ROS2 adapter of hku-mars/VoxelMap's released constant-velocity estimator.
// Original map/plane uncertainty and association implementation are reused in
// generated upstream headers; provenance records pin their exact commit.
#include "voxel_map_util.hpp"
#include "registration_quality.hpp"
#include "optional_imu.hpp"
#include "tracking_guard.hpp"
#include <sensor_msgs/msg/imu.hpp>
#include <memory>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/point_cloud2_iterator.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <std_msgs/msg/string.hpp>
#include <pcl/filters/voxel_grid.h>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <sys/resource.h>

using Clock = std::chrono::steady_clock;
double seconds(Clock::time_point a, Clock::time_point b) {
    return std::chrono::duration<double>(b-a).count();
}
M3D skew(const V3D &p) { M3D m; m << SKEW_SYM_MATRX(p); return m; }

class VoxelMapNode : public rclcpp::Node {
    StatesGroup state_;
    StatesGroup accepted_state_;
    TrackingLimits tracking_limits_;
    double accepted_stamp_ = -1.;
    size_t consecutive_failures_ = 0;
    std::string tracking_reason_ = "initializing";
    std::unique_ptr<fusion_imu::OptionalImu> imu_;
    rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr imu_input_;
    double imu_time_offset_ = 0., imu_future_tolerance_ = .1;
    std::string imu_frame_;
    rclcpp::Publisher<std_msgs::msg::String>::SharedPtr imu_status_;
    std::unordered_map<VOXEL_LOC, OctoTree*> map_;
    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr input_;
    rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr odom_;
    rclcpp::Publisher<std_msgs::msg::String>::SharedPtr metrics_, quality_pub_;
    double last_stamp_ = -1.;
    double voxel_size_, leaf_size_, range_noise_, angle_noise_, plane_threshold_;
    double velocity_noise_, omega_noise_, local_radius_;
    double map_keyframe_translation_, map_keyframe_rotation_;
    Eigen::Matrix3d last_map_rotation_ = Eigen::Matrix3d::Identity();
    Eigen::Vector3d last_map_position_ = Eigen::Vector3d::Zero();
    size_t map_keyframes_ = 0;
    int max_layer_, max_points_, max_iterations_, max_roots_, max_input_;
    std::vector<int> layer_size_;
    Eigen::Matrix4d base_from_lidar_;
    std::ofstream timing_;
    size_t frames_ = 0, failures_ = 0, evicted_ = 0;
    RegistrationQuality quality_;
    bool solver_converged_ = false, solution_stable_ = false;
    double last_translation_correction_ = 0., last_rotation_correction_ = 0.;

    void free_tree(OctoTree *tree) {
        if (!tree) return;
        for (auto *child : tree->leaves_) free_tree(child);
        delete tree->plane_ptr_;
        delete tree;
    }
    void bound_map() {
        for (auto it=map_.begin(); it!=map_.end();) {
            V3D center((it->first.x+.5)*voxel_size_, (it->first.y+.5)*voxel_size_, (it->first.z+.5)*voxel_size_);
            if ((center-state_.pos_end).norm() > local_radius_) {
                free_tree(it->second); it=map_.erase(it); ++evicted_;
            } else ++it;
        }
        if (map_.size() > size_t(max_roots_)) {
            std::vector<std::pair<double, VOXEL_LOC>> distances;
            distances.reserve(map_.size());
            for (const auto &entry:map_) {
                V3D center((entry.first.x+.5)*voxel_size_, (entry.first.y+.5)*voxel_size_, (entry.first.z+.5)*voxel_size_);
                distances.emplace_back((center-state_.pos_end).squaredNorm(), entry.first);
            }
            std::sort(distances.begin(), distances.end(), [](const auto&a,const auto&b){return a.first>b.first;});
            size_t count=map_.size()-max_roots_;
            for (size_t i=0;i<count;++i) {
                auto it=map_.find(distances[i].second);
                free_tree(it->second); map_.erase(it); ++evicted_;
            }
        }
    }
    void imu_sample(const sensor_msgs::msg::Imu::ConstSharedPtr m) {
        if(m->header.frame_id!=imu_frame_ ||
           m->angular_velocity_covariance[0]<0. ||
           m->linear_acceleration_covariance[0]<0.) {imu_->reject();return;}
        fusion_imu::Sample sample;
        sample.stamp=rclcpp::Time(m->header.stamp).seconds()+imu_time_offset_;
        const double now=get_clock()->now().seconds();
        if(sample.stamp<0. || (now>0. && sample.stamp>now+imu_future_tolerance_)) {imu_->reject();return;}
        sample.gyro=V3D(m->angular_velocity.x,m->angular_velocity.y,m->angular_velocity.z);
        sample.accel=V3D(m->linear_acceleration.x,m->linear_acceleration.y,m->linear_acceleration.z);
        imu_->push(sample);
    }
    std::string imu_json(double stamp) const {
        const auto &v=imu_->report();
        std::ostringstream text;
        text << std::setprecision(12) << "{\"stamp_sec\":" << stamp
             << ",\"mode\":\"" << (v.using_imu?"imu":"constant_velocity")
             << "\",\"reason\":\"" << v.reason
             << "\",\"initialized\":" << (v.initialized?"true":"false")
             << ",\"samples_used\":" << v.samples_used
             << ",\"integrated_seconds\":" << v.integrated_seconds
             << ",\"buffered\":" << v.buffered << ",\"accepted\":" << v.accepted
             << ",\"rejected\":" << v.rejected << ",\"evicted\":" << v.evicted
             << ",\"transitions\":" << v.transitions << ",\"imu_scans\":" << v.imu_scans
             << ",\"cv_scans\":" << v.cv_scans << "}";
        return text.str();
    }
    std::vector<pointWithCov> points_with_cov(const PointCloudXYZI &cloud,
                   const std::vector<M3D> &body_cov, bool for_map) {
        std::vector<pointWithCov> result;
        result.reserve(cloud.size());
        for (size_t i=0;i<cloud.size();++i) {
            V3D body(cloud[i].x,cloud[i].y,cloud[i].z);
            M3D cross=skew(body);
            pointWithCov point;
            point.point_world=state_.rot_end*body+state_.pos_end;
            point.point=for_map ? point.point_world : body;
            // Retain the released VoxelMap uncertainty propagation for baseline.
            point.cov=state_.rot_end*body_cov[i]*state_.rot_end.transpose()+
                      (-cross)*state_.cov.block<3,3>(0,0)*(-cross).transpose()+state_.cov.block<3,3>(3,3);
            result.push_back(point);
        }
        return result;
    }
    bool update(const PointCloudXYZI &cloud,const std::vector<M3D> &body_cov,
                double &matching, double &solving, int &iterations) {
        StatesGroup prior=state_;
        Eigen::Matrix<double,18,18> posterior=prior.cov;
        int converged_updates=0;
        solver_converged_=solution_stable_=false;
        last_translation_correction_=last_rotation_correction_=0.;
        for (int iteration=0;iteration<max_iterations_;++iteration) {
            auto start=Clock::now();
            auto points=points_with_cov(cloud,body_cov,false);
            std::vector<ptpl> matches;
            std::vector<V3D> non_matches;
            BuildResidualListOMP(map_,voxel_size_,3.,max_layer_,points,matches,non_matches);
            matching+=seconds(start,Clock::now());
            start=Clock::now();
            if (matches.size()<6) return false;
            Eigen::MatrixXd h(matches.size(),6), htr(6,matches.size());
            Eigen::VectorXd residual(matches.size());
            for (size_t i=0;i<matches.size();++i) {
                const auto &match=matches[i];
                V3D body=match.point;
                if (std::abs(body.z())<1e-6) body.z()=1e-6;
                M3D covariance;
                calcBodyCov(body,range_noise_,angle_noise_,covariance);
                covariance=state_.rot_end*covariance*state_.rot_end.transpose();
                V3D world=state_.rot_end*body+state_.pos_end;
                Eigen::Matrix<double,1,6> j;
                j << (world-match.center).transpose(), -match.normal.transpose();
                double plane_variance=(j*match.plane_cov*j.transpose())(0,0);
                double variance=plane_variance+(match.normal.transpose()*covariance*match.normal)(0,0);
                if (!std::isfinite(variance)) return false;
                double inverse_variance=1./std::max(variance,1e-9);
                V3D a=skew(body)*state_.rot_end.transpose()*match.normal;
                h.row(i) << a.transpose(),match.normal.transpose();
                htr.col(i)=h.row(i).transpose()*inverse_variance;
                residual[i]=-(match.normal.dot(world)+match.d);
            }
            Eigen::Matrix<double,18,18> information=Eigen::Matrix<double,18,18>::Zero();
            information.topLeftCorner<6,6>()=htr*h;
            // State-sized solve, matching the released information-form update.
            Eigen::LDLT<Eigen::Matrix<double,18,18>> prior_solver(prior.cov);
            if (prior_solver.info()!=Eigen::Success) return false;
            information+=prior_solver.solve(Eigen::Matrix<double,18,18>::Identity());
            Eigen::LDLT<Eigen::Matrix<double,18,18>> solver(information);
            if (solver.info()!=Eigen::Success) return false;
            posterior=solver.solve(Eigen::Matrix<double,18,18>::Identity());
            Eigen::MatrixXd gain=posterior.leftCols<6>()*htr;
            Eigen::Matrix<double,18,1> difference=prior-state_;
            Eigen::Matrix<double,18,1> delta=gain*residual+difference-gain*h*difference.head<6>();
            if (!delta.allFinite()) return false;
            last_rotation_correction_=delta.head<3>().norm();
            last_translation_correction_=delta.segment<3>(3).norm();
            state_+=delta;
            if(!imu_->physically_valid(state_))return false;
            quality_.reset();
            Eigen::MatrixXd ordered(h.rows(),6);
            ordered.leftCols<3>()=h.rightCols<3>();ordered.rightCols<3>()=h.leftCols<3>();
            quality_.add(ordered,residual,cloud.size());
            RegistrationQuality::M6 permutation=RegistrationQuality::M6::Zero();
            permutation.topRightCorner<3,3>().setIdentity();
            permutation.bottomLeftCorner<3,3>().setIdentity();
            quality_.set_weighted_information(permutation*(htr*h)*permutation.transpose());
            solving+=seconds(start,Clock::now());
            iterations=iteration+1;
            bool converged=last_rotation_correction_*57.3<.01 && last_translation_correction_*100<.015;
            converged_updates=converged ? converged_updates+1 : 0;
            if (converged_updates>=2) {solver_converged_=true;break;}
        }
        // Reaching an iteration budget is not evidence of a solved registration.
        // A limited solve may be retained only once its remaining correction is
        // below the configured measurement noise in both translation and angle.
        solution_stable_=last_translation_correction_<=range_noise_ &&
                         last_rotation_correction_<=angle_noise_/57.29577951308232;
        if(!solution_stable_ || !quality_.registration_usable())return false;
        state_.cov=.5*(posterior+posterior.transpose());
        return true;
    }
    void publish(const sensor_msgs::msg::PointCloud2 &input, bool valid, double cost,
                 size_t raw,size_t down,double matching,double solving,double map_time,int iterations) {
        Eigen::Matrix4d lidar=Eigen::Matrix4d::Identity();
        lidar.topLeftCorner<3,3>()=state_.rot_end;lidar.topRightCorner<3,1>()=state_.pos_end;
        Eigen::Matrix4d body=base_from_lidar_*lidar*base_from_lidar_.inverse();
        nav_msgs::msg::Odometry odom;
        odom.header=input.header;odom.header.frame_id="odom";odom.child_frame_id="base_link";
        odom.pose.pose.position.x=body(0,3);odom.pose.pose.position.y=body(1,3);odom.pose.pose.position.z=body(2,3);
        Eigen::Quaterniond quaternion(body.topLeftCorner<3,3>());
        odom.pose.pose.orientation.x=quaternion.x();odom.pose.pose.orientation.y=quaternion.y();
        odom.pose.pose.orientation.z=quaternion.z();odom.pose.pose.orientation.w=quaternion.w();
        M3D r=base_from_lidar_.topLeftCorner<3,3>();
        V3D translation=r.transpose()*base_from_lidar_.topRightCorner<3,1>();
        Eigen::Matrix<double,6,6> j=Eigen::Matrix<double,6,6>::Zero();
        j.topLeftCorner<3,3>()=r*state_.rot_end*skew(translation);
        j.topRightCorner<3,3>()=r;j.bottomLeftCorner<3,3>()=r*state_.rot_end;
        Eigen::Matrix<double,6,6> cov=j*state_.cov.topLeftCorner<6,6>()*j.transpose();
        // The first scan fixes the arbitrary odometry gauge. Subsequent scans
        // retain noise-weighted weak directions rather than discard all six.
        bool reliable=valid && (frames_==1 || quality_.registration_usable());
        if (reliable && frames_>1) {
            RegistrationQuality::M6 permutation=RegistrationQuality::M6::Zero();
            permutation.topRightCorner<3,3>().setIdentity();
            permutation.bottomLeftCorner<3,3>().setIdentity();
            const auto geometry=quality_.directional_covariance();
            cov+=j*permutation*geometry*permutation.transpose()*j.transpose();
        }
        if (!reliable) cov.diagonal().array()+=1e6;
        cov=.5*(cov+cov.transpose()).eval();
        for (int i=0;i<6;++i) for(int k=0;k<6;++k) odom.pose.covariance[i*6+k]=cov(i,k);
        odom_->publish(odom);
        std::ostringstream health;
        health << std::setprecision(16) << "{\"stamp_sec\":" << rclcpp::Time(input.header.stamp).seconds()
            << ",\"backend\":\"voxelmap\",\"reliable\":" << (reliable?"true":"false")
            << ",\"matched\":" << quality_.matched << ",\"candidates\":" << quality_.candidates
            << ",\"match_ratio\":" << quality_.match_ratio() << ",\"residual_rms_m\":" << quality_.residual_rms()
            << ",\"initial_gauge\":" << (frames_==1?"true":"false")
            << ",\"solver_converged\":" << (solver_converged_?"true":"false")
            << ",\"solution_stable\":" << (solution_stable_?"true":"false")
            << ",\"last_translation_correction_m\":" << last_translation_correction_
            << ",\"last_rotation_correction_rad\":" << last_rotation_correction_
            << ",\"legacy_full_rank_test\":" << (quality_.reliable()?"true":"false")
            << ",\"covariance_basis\":\"prior_plus_noise_weighted_directional_geometry\""
            << ",\"translation_eigen_ratio\":" << quality_.translation_ratio()
            << ",\"pose_eigen_ratio\":" << quality_.pose_ratio() << "}";
        std_msgs::msg::String report;report.data=health.str();quality_pub_->publish(report);
        struct rusage usage;getrusage(RUSAGE_SELF,&usage);
        std::ostringstream text;
        text << std::setprecision(12) << "{\"backend\":\"voxelmap\",\"frame\":" << frames_
             << ",\"stamp_sec\":" << rclcpp::Time(input.header.stamp).seconds()
             << ",\"valid_update\":" << (valid?"true":"false") << ",\"processing_sec\":" << cost
             << ",\"raw_points\":" << raw << ",\"downsampled_points\":" << down
             << ",\"matching_sec\":" << matching << ",\"solve_sec\":" << solving
              << ",\"map_update_sec\":" << map_time << ",\"iterations\":" << iterations
              << ",\"map_keyframes\":" << map_keyframes_
              << ",\"solver_converged\":" << (solver_converged_?"true":"false")
              << ",\"solution_stable\":" << (solution_stable_?"true":"false")
              << ",\"last_translation_correction_m\":" << last_translation_correction_
             << ",\"last_rotation_correction_rad\":" << last_rotation_correction_
             << ",\"tracking_reason\":\"" << tracking_reason_ << "\""
             << ",\"consecutive_failures\":" << consecutive_failures_
             << ",\"estimated_speed_mps\":" << state_.vel_end.norm()
             << ",\"position\":[" << state_.pos_end.x() << "," << state_.pos_end.y() << "," << state_.pos_end.z() << "]"
             << ",\"roots\":" << map_.size() << ",\"evicted_roots\":" << evicted_
             << ",\"failures\":" << failures_ << ",\"peak_rss_mib\":" << usage.ru_maxrss/1024. << "}";
        std::string record=text.str();record.pop_back();
        record+=",\"imu\":"+imu_json(rclcpp::Time(input.header.stamp).seconds())+"}";
        report.data=record;metrics_->publish(report);
        std_msgs::msg::String imu_report;imu_report.data=imu_json(rclcpp::Time(input.header.stamp).seconds());
        imu_status_->publish(imu_report);
        if (timing_) timing_<<report.data<<std::endl;
    }
    void scan(const sensor_msgs::msg::PointCloud2::ConstSharedPtr msg) {
        auto start=Clock::now();
        double stamp=rclcpp::Time(msg->header.stamp).seconds();
        if (last_stamp_>=0 && stamp<=last_stamp_) return;
        if (size_t(msg->width)*msg->height>size_t(max_input_)) {
            RCLCPP_ERROR(get_logger(),"Input exceeds configured point capacity");return;
        }
        PointCloudXYZI::Ptr cloud(new PointCloudXYZI), down(new PointCloudXYZI);
        cloud->reserve(size_t(msg->width)*msg->height);
        sensor_msgs::PointCloud2ConstIterator<float> x(*msg,"x"), y(*msg,"y"), z(*msg,"z");
        for (;x!=x.end();++x,++y,++z) {
            V3D p(*x,*y,*z);if(!p.allFinite() || p.norm()<.5 || p.norm()>70.)continue;
            PointType point;point.x=*x;point.y=*y;point.z=*z;point.intensity=0.;point.curvature=0.;
            point.normal_x=point.normal_y=point.normal_z=0.;cloud->push_back(point);
        }
        if (cloud->size()<50) return;
        pcl::VoxelGrid<PointType> filter;
        filter.setLeafSize(leaf_size_,leaf_size_,leaf_size_);filter.setInputCloud(cloud);filter.filter(*down);
        bool first=last_stamp_<0;
        if(first)solver_converged_=solution_stable_=true;
        imu_->predict(state_,last_stamp_,stamp,first);last_stamp_=stamp;
        const bool using_imu=imu_->report().using_imu;
        // Never seed CV registration with an unbounded pose after a long gap.
        if (!first && !using_imu && tracking_limits_.check(accepted_state_,state_,stamp-accepted_stamp_,false))
            hold_unobserved_cv(state_,accepted_state_);
        PointCloudXYZI &used=first?*cloud:*down;
        std::vector<M3D> body_cov;body_cov.reserve(used.size());
        for(const auto &point:used){V3D p(point.x,point.y,point.z);if(std::abs(p.z())<1e-6)p.z()=1e-6;
            M3D c;calcBodyCov(p,range_noise_,angle_noise_,c);body_cov.push_back(c);}
        double matching=0.,solving=0.;int iterations=0;
        StatesGroup propagated=state_;
        quality_.reset();
        bool valid=first || update(used,body_cov,matching,solving,iterations);
        tracking_reason_=valid?"tracking":"registration_failed";
        if (valid && !first) {
            if (const char *reason=tracking_limits_.check(accepted_state_,state_,stamp-accepted_stamp_,using_imu)) {
                valid=false; tracking_reason_=reason;
            }
        }
        if (!valid) {
            ++failures_; ++consecutive_failures_;
            if (!using_imu) hold_unobserved_cv(state_,accepted_state_);
            else state_=propagated;  // Preserve calibrated real-IMU propagation/biases.
        } else {
            accepted_state_=state_; accepted_stamp_=stamp; consecutive_failures_=0;
        }
        auto map_start=Clock::now();
        // Repeated scans from the same viewpoint are correlated observations.
        // Integrating all of them fills and overweights plane cells while the
        // vehicle waits. Still register every scan; insert only new viewpoints.
        const bool new_viewpoint = first ||
            (state_.pos_end-last_map_position_).norm() >= map_keyframe_translation_ ||
            Eigen::AngleAxisd(last_map_rotation_.transpose()*state_.rot_end).angle() >= map_keyframe_rotation_;
        if(valid && new_viewpoint){
            auto world=points_with_cov(used,body_cov,true);
            std::sort(world.begin(),world.end(),[](const pointWithCov&a,const pointWithCov&b){return a.cov.trace()<b.cov.trace();});
            if(first)buildVoxelMap(world,voxel_size_,max_layer_,layer_size_,max_points_,max_points_,plane_threshold_,map_);
            else updateVoxelMap(world,voxel_size_,max_layer_,layer_size_,max_points_,max_points_,plane_threshold_,map_);
            bound_map();
            last_map_position_=state_.pos_end;last_map_rotation_=state_.rot_end;
            ++map_keyframes_;
        }
        double map_time=seconds(map_start,Clock::now());
        ++frames_;
        publish(*msg,valid,seconds(start,Clock::now()),cloud->size(),down->size(),matching,solving,map_time,iterations);
    }
public:
    VoxelMapNode():Node("fusion_voxelmap") {
        const bool instantaneous=declare_parameter<bool>("instantaneous_cloud",true);
        const bool deskewed=declare_parameter<bool>("cloud_motion_compensated",false);
        if(!instantaneous && !deskewed)
            throw std::runtime_error("VoxelMap requires instantaneous or upstream motion-compensated clouds");
        voxel_size_=declare_parameter<double>("voxel_size",3.);
        leaf_size_=declare_parameter<double>("downsample_size",.5);
        range_noise_=declare_parameter<double>("range_noise",.04);
        angle_noise_=declare_parameter<double>("angle_noise",.1);
        map_keyframe_translation_=declare_parameter<double>("map_keyframe_translation",.1);
        map_keyframe_rotation_=declare_parameter<double>("map_keyframe_rotation_deg",1.)/57.29577951308232;
        if(!std::isfinite(map_keyframe_translation_) || map_keyframe_translation_<0. ||
           !std::isfinite(map_keyframe_rotation_) || map_keyframe_rotation_<0.)
            throw std::runtime_error("Map keyframe thresholds must be finite and non-negative");
        plane_threshold_=declare_parameter<double>("plane_threshold",.01);
        velocity_noise_=declare_parameter<double>("velocity_noise",1.);
        omega_noise_=declare_parameter<double>("angular_velocity_noise",.5);
        tracking_limits_.max_speed=declare_parameter<double>("tracking_max_speed",4.);
        tracking_limits_.max_angular_speed=declare_parameter<double>("tracking_max_angular_speed",2.);
        tracking_limits_.max_step=declare_parameter<double>("tracking_max_step",6.);
        tracking_limits_.max_angle=declare_parameter<double>("tracking_max_angle",1.);
        tracking_limits_.validate();
        local_radius_=declare_parameter<double>("local_map_radius",80.);
        max_layer_=declare_parameter<int>("max_layer",4);
        max_points_=declare_parameter<int>("max_points_per_cell",1000);
        max_iterations_=declare_parameter<int>("max_iterations",12);
        if(max_iterations_<1 || max_iterations_>32)
            throw std::runtime_error("max_iterations must be in [1,32]");
        max_roots_=declare_parameter<int>("max_root_voxels",10000);
        max_input_=declare_parameter<int>("max_input_points",140000);
        layer_size_=std::vector<int>(max_layer_+1,5);
        std::vector<double> identity{1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1};
        auto transform=declare_parameter<std::vector<double>>("base_from_lidar",identity);
        if(transform.size()!=16)throw std::runtime_error("base_from_lidar requires 16 numbers");
        for(int i=0;i<4;++i)for(int j=0;j<4;++j)base_from_lidar_(i,j)=transform[i*4+j];
        fusion_imu::Config imu_cfg;
        const auto mode=declare_parameter<std::string>("imu_mode","auto");
        if(mode!="auto" && mode!="off")throw std::runtime_error("imu_mode must be auto or off");
        imu_cfg.enabled=mode=="auto";
        imu_cfg.calibrated=declare_parameter<bool>("imu_calibration_confirmed",false);
        auto imu_transform=declare_parameter<std::vector<double>>("base_from_imu",identity);
        if(imu_transform.size()!=16)throw std::runtime_error("base_from_imu requires 16 numbers");
        Eigen::Matrix4d base_from_imu;
        for(int i=0;i<4;++i)for(int j=0;j<4;++j)base_from_imu(i,j)=imu_transform[i*4+j];
        for(const auto &t:{base_from_lidar_,base_from_imu}) {
            if(!t.allFinite() || (t.row(3)-Eigen::RowVector4d(0,0,0,1)).norm()>1e-8 ||
               (t.topLeftCorner<3,3>()*t.topLeftCorner<3,3>().transpose()-M3D::Identity()).norm()>1e-4 ||
               std::abs(t.topLeftCorner<3,3>().determinant()-1.)>1e-4)
                throw std::runtime_error("Sensor extrinsics must be proper rigid transforms");
        }
        const Eigen::Matrix4d lidar_from_imu=base_from_lidar_.inverse()*base_from_imu;
        imu_cfg.lidar_from_imu_rotation=lidar_from_imu.topLeftCorner<3,3>();
        imu_cfg.imu_origin_in_lidar=lidar_from_imu.topRightCorner<3,1>();
        imu_cfg.cv_velocity_noise=velocity_noise_;imu_cfg.cv_omega_noise=omega_noise_;
        imu_cfg.buffer_seconds=declare_parameter<double>("imu_buffer_seconds",8.);
        const int capacity=declare_parameter<int>("imu_buffer_samples",4000);
        const int init_samples=declare_parameter<int>("imu_init_min_samples",50);
        if(capacity<3||init_samples<3)throw std::runtime_error("IMU sample limits must be >= 3");
        imu_cfg.max_samples=capacity;imu_cfg.init_min_samples=init_samples;
        imu_cfg.max_gap=declare_parameter<double>("imu_max_gap_sec",.05);
        imu_cfg.end_tolerance=declare_parameter<double>("imu_end_tolerance_sec",.015);
        imu_cfg.init_seconds=declare_parameter<double>("imu_init_seconds",1.);
        imu_cfg.gyro_noise=declare_parameter<double>("imu_gyro_noise",.01);
        imu_cfg.accel_noise=declare_parameter<double>("imu_accel_noise",.1);
        imu_cfg.gyro_bias_noise=declare_parameter<double>("imu_gyro_bias_noise",.0001);
        imu_cfg.accel_bias_noise=declare_parameter<double>("imu_accel_bias_noise",.001);
        imu_cfg.recovery_scans=declare_parameter<int>("imu_recovery_scans",3);
        imu_time_offset_=declare_parameter<double>("imu_time_offset_sec",0.);
        imu_future_tolerance_=declare_parameter<double>("imu_future_tolerance_sec",.1);
        if(!std::isfinite(imu_future_tolerance_)||imu_future_tolerance_<0.)throw std::runtime_error("Invalid IMU future tolerance");
        imu_frame_=declare_parameter<std::string>("imu_frame","imu");
        if(!std::isfinite(imu_time_offset_)||imu_frame_.empty())throw std::runtime_error("Invalid IMU stamp/frame configuration");
        imu_=std::make_unique<fusion_imu::OptionalImu>(imu_cfg);
        imu_status_=create_publisher<std_msgs::msg::String>("/fusion/imu_status",10);
        if(imu_cfg.enabled)imu_input_=create_subscription<sensor_msgs::msg::Imu>("/fusion/imu",
            rclcpp::SensorDataQoS().keep_last(capacity),std::bind(&VoxelMapNode::imu_sample,this,std::placeholders::_1));
        auto path=declare_parameter<std::string>("timing_path","");if(!path.empty())timing_.open(path);
        omp_set_num_threads(declare_parameter<int>("threads",2));
        odom_=create_publisher<nav_msgs::msg::Odometry>("/fusion/lio_raw",30);
        metrics_=create_publisher<std_msgs::msg::String>("/fusion/lidar_metrics",10);
        quality_pub_=create_publisher<std_msgs::msg::String>("/fusion/lidar_quality",10);
        input_=create_subscription<sensor_msgs::msg::PointCloud2>("/fusion/lidar",rclcpp::QoS(2).reliable(),
             std::bind(&VoxelMapNode::scan,this,std::placeholders::_1));
        RCLCPP_INFO(get_logger(),"VoxelMap optional IMU=%s, calibration=%s; no-IMU CV fallback; root cap %d",
            mode.c_str(),imu_cfg.calibrated?"confirmed":"unconfirmed",max_roots_);
    }
    ~VoxelMapNode(){for(auto &entry:map_)free_tree(entry.second);}
};
int main(int argc,char**argv){rclcpp::init(argc,argv);rclcpp::spin(std::make_shared<VoxelMapNode>());rclcpp::shutdown();}

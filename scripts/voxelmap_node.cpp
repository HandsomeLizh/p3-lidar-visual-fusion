// ROS2 adapter of hku-mars/VoxelMap's released constant-velocity estimator.
// Original map/plane uncertainty and association implementation are reused in
// generated upstream headers; provenance records pin their exact commit.
#include "voxel_map_util.hpp"
#include "registration_quality.hpp"
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
    std::unordered_map<VOXEL_LOC, OctoTree*> map_;
    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr input_;
    rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr odom_;
    rclcpp::Publisher<std_msgs::msg::String>::SharedPtr metrics_, quality_pub_;
    double last_stamp_ = -1.;
    double voxel_size_, leaf_size_, range_noise_, angle_noise_, plane_threshold_;
    double velocity_noise_, omega_noise_, local_radius_;
    int max_layer_, max_points_, max_iterations_, max_roots_, max_input_;
    std::vector<int> layer_size_;
    Eigen::Matrix4d base_from_lidar_;
    std::ofstream timing_;
    size_t frames_ = 0, failures_ = 0, evicted_ = 0;
    RegistrationQuality quality_;

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
    void predict(double dt) {
        // Same constant velocity/angular velocity model as upstream only_propag.
        Eigen::Matrix<double,18,18> f=Eigen::Matrix<double,18,18>::Identity();
        Eigen::Matrix<double,18,18> q=Eigen::Matrix<double,18,18>::Zero();
        f.block<3,3>(0,0)=Exp(state_.bias_g,-dt);
        f.block<3,3>(0,9)=M3D::Identity()*dt;
        f.block<3,3>(3,6)=M3D::Identity()*dt;
        q.block<3,3>(9,9).diagonal().setConstant(omega_noise_*dt*dt);
        q.block<3,3>(6,6).diagonal().setConstant(velocity_noise_*dt*dt);
        state_.cov=f*state_.cov*f.transpose()+q;
        state_.rot_end=state_.rot_end*Exp(state_.bias_g,dt);
        state_.pos_end+=state_.vel_end*dt;
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
        const StatesGroup prior=state_;
        Eigen::Matrix<double,18,18> posterior=prior.cov;
        int rematches=0;
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
            state_+=delta;
            quality_.reset();
            Eigen::MatrixXd ordered(h.rows(),6);
            ordered.leftCols<3>()=h.rightCols<3>();ordered.rightCols<3>()=h.leftCols<3>();
            quality_.add(ordered,residual,cloud.size());
            solving+=seconds(start,Clock::now());
            iterations=iteration+1;
            bool converged=delta.head<3>().norm()*57.3<.01 && delta.segment<3>(3).norm()*100<.015;
            if (converged || (rematches==0 && iteration==max_iterations_-2)) ++rematches;
            if (rematches>=2 || iteration==max_iterations_-1) break;
        }
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
        bool reliable=valid && quality_.reliable();
        if (!reliable) cov.diagonal().array()+=1000.;
        for (int i=0;i<6;++i) for(int k=0;k<6;++k) odom.pose.covariance[i*6+k]=cov(i,k);
        odom_->publish(odom);
        std::ostringstream health;
        health << std::setprecision(16) << "{\"stamp_sec\":" << rclcpp::Time(input.header.stamp).seconds()
            << ",\"backend\":\"voxelmap\",\"reliable\":" << (reliable?"true":"false")
            << ",\"matched\":" << quality_.matched << ",\"candidates\":" << quality_.candidates
            << ",\"match_ratio\":" << quality_.match_ratio() << ",\"residual_rms_m\":" << quality_.residual_rms()
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
             << ",\"roots\":" << map_.size() << ",\"evicted_roots\":" << evicted_
             << ",\"failures\":" << failures_ << ",\"peak_rss_mib\":" << usage.ru_maxrss/1024. << "}";
        report.data=text.str();metrics_->publish(report);
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
        predict(first?.1:stamp-last_stamp_);last_stamp_=stamp;
        PointCloudXYZI &used=first?*cloud:*down;
        std::vector<M3D> body_cov;body_cov.reserve(used.size());
        for(const auto &point:used){V3D p(point.x,point.y,point.z);if(std::abs(p.z())<1e-6)p.z()=1e-6;
            M3D c;calcBodyCov(p,range_noise_,angle_noise_,c);body_cov.push_back(c);}
        double matching=0.,solving=0.;int iterations=0;
        StatesGroup propagated=state_;
        bool valid=first || update(used,body_cov,matching,solving,iterations);
        if (!valid) {state_=propagated;++failures_;}
        auto map_start=Clock::now();
        if(valid){
            auto world=points_with_cov(used,body_cov,true);
            std::sort(world.begin(),world.end(),[](const pointWithCov&a,const pointWithCov&b){return a.cov.trace()<b.cov.trace();});
            if(first)buildVoxelMap(world,voxel_size_,max_layer_,layer_size_,max_points_,max_points_,plane_threshold_,map_);
            else updateVoxelMap(world,voxel_size_,max_layer_,layer_size_,max_points_,max_points_,plane_threshold_,map_);
            bound_map();
        }
        double map_time=seconds(map_start,Clock::now());
        ++frames_;
        publish(*msg,valid,seconds(start,Clock::now()),cloud->size(),down->size(),matching,solving,map_time,iterations);
    }
public:
    VoxelMapNode():Node("fusion_voxelmap") {
        if(!declare_parameter<bool>("instantaneous_cloud",true))
            throw std::runtime_error("Pure VoxelMap requires externally deskewed or instantaneous clouds");
        voxel_size_=declare_parameter<double>("voxel_size",3.);
        leaf_size_=declare_parameter<double>("downsample_size",.5);
        range_noise_=declare_parameter<double>("range_noise",.04);
        angle_noise_=declare_parameter<double>("angle_noise",.1);
        plane_threshold_=declare_parameter<double>("plane_threshold",.01);
        velocity_noise_=declare_parameter<double>("velocity_noise",1.);
        omega_noise_=declare_parameter<double>("angular_velocity_noise",.5);
        local_radius_=declare_parameter<double>("local_map_radius",80.);
        max_layer_=declare_parameter<int>("max_layer",4);
        max_points_=declare_parameter<int>("max_points_per_cell",1000);
        max_iterations_=declare_parameter<int>("max_iterations",3);
        max_roots_=declare_parameter<int>("max_root_voxels",10000);
        max_input_=declare_parameter<int>("max_input_points",140000);
        layer_size_=std::vector<int>(max_layer_+1,5);
        std::vector<double> identity{1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1};
        auto transform=declare_parameter<std::vector<double>>("base_from_lidar",identity);
        if(transform.size()!=16)throw std::runtime_error("base_from_lidar requires 16 numbers");
        for(int i=0;i<4;++i)for(int j=0;j<4;++j)base_from_lidar_(i,j)=transform[i*4+j];
        auto path=declare_parameter<std::string>("timing_path","");if(!path.empty())timing_.open(path);
        omp_set_num_threads(declare_parameter<int>("threads",2));
        odom_=create_publisher<nav_msgs::msg::Odometry>("/fusion/lio_raw",30);
        metrics_=create_publisher<std_msgs::msg::String>("/fusion/lidar_metrics",10);
        quality_pub_=create_publisher<std_msgs::msg::String>("/fusion/lidar_quality",10);
        input_=create_subscription<sensor_msgs::msg::PointCloud2>("/fusion/lidar",rclcpp::QoS(2).reliable(),
             std::bind(&VoxelMapNode::scan,this,std::placeholders::_1));
        RCLCPP_INFO(get_logger(),"VoxelMap constant-velocity mode; map root cap %d",max_roots_);
    }
    ~VoxelMapNode(){for(auto &entry:map_)free_tree(entry.second);}
};
int main(int argc,char**argv){rclcpp::init(argc,argv);rclcpp::spin(std::make_shared<VoxelMapNode>());rclcpp::shutdown();}

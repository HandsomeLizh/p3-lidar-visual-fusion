#include <cmath>
#include <iostream>
#include <stdexcept>
#include "../vendor/point_lio/include/output_motion.hpp"
struct State {
    Eigen::Vector3d vel{.4,-.2,.1},omg{.01,-.03,.02},acc{2.,-4.,8.},gravity{0.,0.,-9.81};
    Eigen::Quaterniond rot=Eigen::Quaterniond(Eigen::AngleAxisd(.3,Eigen::Vector3d(1,2,3).normalized()));
};
void require(bool ok,const char* what){if(!ok)throw std::runtime_error(what);}
int main(){
    State s;
    const double eps=1e-6;
    double worst=0.;
    for(bool imu:{false,true}) {
        auto f=point_lio_motion::derivative(s,imu);
        auto J=point_lio_motion::jacobian(s,imu);
        for(int j=0;j<30;++j) {
            State p=s,m=s;
            if(j>=3 && j<6) {
                Eigen::Vector3d axis=Eigen::Vector3d::Unit(j-3);
                p.rot=s.rot*Eigen::Quaterniond(Eigen::AngleAxisd(eps,axis));
                m.rot=s.rot*Eigen::Quaterniond(Eigen::AngleAxisd(-eps,axis));
            }
            for(int base:{12,15,18,21})if(j>=base&&j<base+3){
                auto edit=[&](State& q,double d){
                    if(base==12)q.vel[j-base]+=d;
                    if(base==15)q.omg[j-base]+=d;
                    if(base==18)q.acc[j-base]+=d;
                    if(base==21)q.gravity[j-base]+=d;
                };edit(p,eps);edit(m,-eps);
            }
            Eigen::Matrix<double,30,1> numeric=(point_lio_motion::derivative(p,imu)-point_lio_motion::derivative(m,imu))/(2*eps);
            worst=std::max(worst,(numeric-J.col(j)).norm());
        }
        if(!imu) require(f.segment<3>(12).isZero(0),"No-IMU acceleration leaked");
        else require((f.segment<3>(12)-(s.rot*s.acc+s.gravity)).norm()<1e-12,"IMU physics changed");
    }
    require(worst<1e-7,"Motion Jacobian failed finite differences");
    // Long, sparse-input prediction cannot invent acceleration from unused states.
    Eigen::Vector3d position=Eigen::Vector3d::Zero();
    State q=s;
    for(int k=0;k<100;k++){
        auto f=point_lio_motion::derivative(q,false);
        position+=f.segment<3>(0)*3.6;
        q.vel+=f.segment<3>(12)*3.6;
    }
    require((position-s.vel*360.).norm()<1e-9,"Constant-velocity extrapolation changed velocity");
    std::cout<<"{\"passed\":true,\"max_jacobian_error\":"<<worst<<",\"sparse_prediction_duration_sec\":360}\n";
}

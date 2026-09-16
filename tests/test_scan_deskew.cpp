#include "scan_deskew.hpp"
#include <iostream>

void require(bool ok,const char *message){if(!ok)throw std::runtime_error(message);}
int main(){
    std::vector<fusion_imu::PoseSample> poses;
    const V3D velocity(.3,-.1,.05),omega(0.,0.,.6),world(4.,2.,.7);
    for(int i=0;i<=10;++i){double t=i*.01;poses.push_back({10.+t,Exp(omega,t),velocity*t});}
    PointCloudXYZI cloud;std::vector<double> times;
    for(int i=0;i<=1000;++i){double t=i*.0001;
        V3D raw=Exp(omega,t).transpose()*(world-velocity*t);
        PointType p;p.x=raw.x();p.y=raw.y();p.z=raw.z();cloud.push_back(p);times.push_back(t);
    }
    deskew_scan(cloud,times,10.,poses);
    V3D expected=Exp(omega,.1).transpose()*(world-velocity*.1);
    double worst=0.;for(const auto&p:cloud)worst=std::max(worst,(V3D(p.x,p.y,p.z)-expected).norm());
    require(worst<2e-6,"combined translation/rotation deskew incorrect");
    bool blocked=false;try{pose_at(poses,9.99);}catch(const std::exception&){blocked=true;}
    require(blocked,"uncovered scan must be rejected");
    fusion_imu::Config cfg;cfg.calibrated=true;fusion_imu::OptionalImu imu(cfg);StatesGroup state;
    for(int i=0;i<=220;++i)imu.push({i*.01,V3D::Zero(),V3D(0,0,9.81)});
    imu.predict(state,2.,2.1,false,&poses);
    require(imu.report().using_imu&&poses.size()>=10,"real IMU trajectory not exposed");
    require(state.pos_end.norm()<1e-9,"stationary deskew prediction drifted");
    imu.predict(state,2.1,3.,false,&poses);
    require(poses.empty(),"missing IMU must not produce a deskew trajectory");
    std::cout<<"PASS combined motion, interval rejection, stationary initialization, missing IMU; worst error "<<worst<<" m\n";
}

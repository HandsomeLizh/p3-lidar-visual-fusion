#include "independent_visual_motion.hpp"
#include <iostream>
#include <stdexcept>

using M4=Eigen::Matrix4d;
void require(bool condition,const char *message){if(!condition)throw std::runtime_error(message);}
M4 pose(double x,double y,double yaw=0.){
    M4 p=M4::Identity();p(0,3)=x;p(1,3)=y;
    p.topLeftCorner<3,3>()=Eigen::AngleAxisd(yaw,Eigen::Vector3d::UnitZ()).toRotationMatrix();return p;
}
int main(){
    IndependentVisualMotion motion;
    M4 mount=pose(.6065,-.0023);mount(2,3)=.612;
    const M4 previous=pose(4.585,.560,.01),a=pose(10.,-20.,.7);
    const M4 b=a*pose(.006,0.,.016);
    motion.observe({100.,"learned_epoch_1",a,true});motion.observe({102.547,"learned_epoch_1",b,true});
    M4 expected;
    IndependentVisualMotion startup;
    startup.observe({1.,"learned_epoch_1",M4::Identity(),false,true});
    startup.observe({2.,"learned_epoch_1",pose(.1,0.),true});
    require(startup.prediction(1.,2.,M4::Identity(),mount,expected),"First tracked increment cannot follow the frontend epoch origin");
    require(!startup.prediction(1.,1.,M4::Identity(),mount,expected),"Unmeasured frontend origin was used as a current observation");
    require(motion.prediction(100.,102.547,previous,mount,expected),"Valid same-time pair missing");
    const M4 actual=previous*mount.inverse()*pose(.006,0.,.016)*mount;
    require((actual-expected).norm()<1e-10,"Camera origin or LiDAR lever arm corrupted relative motion");
    require(motion.check(previous,expected,expected).consistent,"Consistent motion rejected");
    M4 wrong=expected;wrong(0,3)+=.688;wrong(1,3)-=.616;
    require(!motion.check(previous,wrong,expected).consistent,"Observed 0.9m planar false match accepted");
    IndependentVisualMotion shifted;
    const M4 gauge=pose(-500.,900.,-1.2);
    shifted.observe({100.,"other_epoch",gauge*a,true});shifted.observe({102.547,"other_epoch",gauge*b,true});
    M4 same;require(shifted.prediction(100.,102.547,previous,mount,same),"Shifted epoch unavailable");
    require((same-expected).norm()<1e-10,"Independent epoch gauge leaked into LiDAR map");
    motion.observe({104.,"learned_epoch_2",b,true});
    require(!motion.prediction(102.547,104.,previous,mount,same),"Visual epoch reset bridged");
    motion.observe({105.,"learned_epoch_2",b,false});
    require(!motion.prediction(104.,105.,previous,mount,same),"Rejected/blackout visual sample used");
    motion.observe({112.,"learned_epoch_2",b,true});
    require(!motion.prediction(104.,112.,previous,mount,same),"Long unobserved visual interval bridged");
    motion.observe({113.,"odom",b,true});require(!motion.at(113.),"Fused odometry feedback accepted as independent vision");
    for(int i=0;i<100;++i)motion.observe({200.+i*.1,"learned_epoch_3",pose(i*.08,0.),true});
    require(!motion.at(200.),"Visual history is unbounded");
    std::cout<<"PASS: false-match rejection; moving/turning increments; frame/lever arm; invalid, missing and expired vision; bounded history\n";
}

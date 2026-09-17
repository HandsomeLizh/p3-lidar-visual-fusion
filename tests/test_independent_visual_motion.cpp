#include "independent_visual_motion.hpp"
#include <iostream>
#include <stdexcept>

using M4=Eigen::Matrix4d;
void require(bool condition,const char *message){if(!condition)throw std::runtime_error(message);}
M4 pose(double x=0.,double y=0.,double yaw=0.){
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
    IndependentVisualMotion recovered;
    recovered.max_gap=120.;recovered.max_speed=.3;recovered.max_step=2.;recovered.validate();
    recovered.observe({1.,"learned_epoch_1",pose(),true});
    recovered.retain_anchor(1.);
    recovered.observe({2.,"learned_epoch_1",pose(),false});
    recovered.observe({7.87,"learned_epoch_1",pose(1.37,0.),true});
    require(recovered.prediction(1.,7.87,M4::Identity(),mount,same),"Validated 6.87s visual recovery is permanently latched out");
    require((same-mount.inverse()*pose(1.37,0.)*mount).norm()<1e-10,"Recovered frame was silently rebased");
    // A finite history is only a lookup cache: never evict the last accepted
    // co-timed LiDAR/vision reference when LiDAR is temporarily unavailable.
    IndependentVisualMotion retained;retained.max_speed=.3;
    retained.observe({1.,"learned_epoch_1",pose(),true});retained.retain_anchor(1.);
    for(int i=1;i<=90;++i)retained.observe({1.+i,"learned_epoch_1",pose(i*.01,0.),true});
    require(!retained.at(1.),"Ordinary visual lookup history grew without bound");
    require(retained.prediction(1.,91.,M4::Identity(),mount,same),"Pinned anchor was evicted at 20 seconds");
    require(retained.prediction_quality().retained_anchor,"Recovery did not use its pinned reference");
    require((same-mount.inverse()*pose(.9,0.)*mount).norm()<1e-10,"Chained increments changed the map origin");
    IndependentVisualMotion jumped;jumped.max_speed=.3;
    jumped.observe({1.,"learned_epoch_1",pose(),true});jumped.retain_anchor(1.);
    jumped.observe({2.,"learned_epoch_1",pose(2.,0.),true});
    jumped.observe({3.,"learned_epoch_1",pose(2.01,0.),true});
    require(!jumped.prediction(1.,3.,M4::Identity(),mount,same),"Locally quiet motion hid an earlier visual jump");
    retained.observe({92.,"learned_epoch_2",pose(),true});
    require(!retained.prediction(1.,92.,M4::Identity(),mount,same),"Pinned reference bridged a new epoch");
    IndependentVisualMotion uncertain;uncertain.max_speed=.3;
    uncertain.observe({1.,"learned_epoch_1",pose(),true});uncertain.retain_anchor(1.);
    for(int i=1;i<=160;++i)uncertain.observe({1.+i,"learned_epoch_1",pose(i*.2,0.),true});
    require(!uncertain.prediction(1.,161.,M4::Identity(),mount,same),"Uncertainty was reset at every visual increment");
    require(uncertain.prediction_quality().reason=="visual_bridge_uncertainty_exceeded","Missing accumulated-uncertainty reason");
    IndependentVisualMotion delayed;delayed.retain_anchor(10.);
    delayed.observe({10.,"learned_epoch_1",pose(),true});delayed.observe({11.,"learned_epoch_1",pose(.1,0.),true});
    require(delayed.prediction(10.,11.,M4::Identity(),mount,same),"Late co-timed anchor was lost");
    IndependentVisualMotion blind;blind.max_gap=120.;
    blind.observe({1.,"learned_epoch_1",pose(),true});blind.retain_anchor(1.);
    for(int i=2;i<140;++i)blind.observe({double(i),"learned_epoch_1",pose(),false,true});
    blind.observe({140.,"learned_epoch_1",pose(.1,0.),true});
    require(!blind.prediction(1.,140.,M4::Identity(),mount,same),"Invalid held origins extended a blind interval");
    IndependentVisualMotion reset;
    reset.observe({1.,"learned_epoch_1",pose(),true});reset.retain_anchor(1.);
    reset.observe({2.,"learned_epoch_2",pose(),false});
    reset.observe({3.,"learned_epoch_1",pose(.1,0.),true});
    require(!reset.prediction(1.,3.,M4::Identity(),mount,same),"Invalid epoch transition was hidden by a later old-epoch pose");
    std::cout<<"PASS: false-match rejection; moving/turning increments; frame/lever arm; invalid, missing and expired vision; bounded history\n";
}

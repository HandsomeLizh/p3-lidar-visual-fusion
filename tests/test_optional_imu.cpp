#include "optional_imu.hpp"
#include <chrono>
#include <iostream>
#include <limits>

using namespace fusion_imu;
void require(bool condition,const std::string &name) {
    if(!condition)throw std::runtime_error("FAIL: "+name);
}
void near(double actual,double expected,double tolerance,const std::string &name) {
    require(std::isfinite(actual)&&std::abs(actual-expected)<tolerance,
            name+" actual="+std::to_string(actual)+" expected="+std::to_string(expected));
}
Config settings(){Config c;c.calibrated=true;return c;}
Sample sample(double t,V3D w=V3D::Zero(),V3D a=V3D(0,0,9.81)) {return {t,w,a};}
void stationary(OptionalImu &imu,double end=2.) {
    for(int k=0;k<=int(end*100+.1);++k)imu.push(sample(k*.01));
}
void positive_covariance(const StatesGroup &s) {
    require(s.cov.allFinite(),"finite covariance");
    near((s.cov-s.cov.transpose()).norm(),0.,1e-8,"symmetric covariance");
    Eigen::SelfAdjointEigenSolver<Cov> eig(s.cov);
    require(eig.eigenvalues().minCoeff()>-1e-9,"positive semidefinite covariance");
}

void absent_identical() {
    OptionalImu imu(settings());StatesGroup a,b;
    a.vel_end=V3D(.2,-.1,.04);a.bias_g=V3D(.01,.02,.04);b=a;
    for(int i=0;i<30;++i) {
        imu.predict(a,i*.1,(i+1)*.1);
        // Independent expression of the pre-change CV propagation.
        const double dt=(i+1)*.1-i*.1;
        Cov f=Cov::Identity(),q=Cov::Zero();
        f.block<3,3>(0,0)=Exp(b.bias_g,-dt);f.block<3,3>(0,9)=M3D::Identity()*dt;
        f.block<3,3>(3,6)=M3D::Identity()*dt;
        q.block<3,3>(9,9).diagonal().setConstant(.5*dt*dt);
        q.block<3,3>(6,6).diagonal().setConstant(dt*dt);
        b.cov=f*b.cov*f.transpose()+q;b.rot_end=b.rot_end*Exp(b.bias_g,dt);b.pos_end+=b.vel_end*dt;
    }
    near((a-b).norm(),0.,1e-14,"no-IMU trajectory unchanged");
    near((a.cov-b.cov).norm(),0.,1e-14,"no-IMU covariance unchanged");
    require(!imu.active()&&imu.report().cv_scans==30,"absent stream never blocks");
}

void translation_and_rotation() {
    OptionalImu imu(settings());StatesGroup s;stationary(imu);
    imu.predict(s,1.,2.);require(imu.active(),"stationary initialization");
    near(s.pos_end.norm(),0.,1e-12,"gravity removed at rest");
    // Jump acceleration at t=2.01: interpolate the one initial 10 ms interval.
    for(int k=201;k<=300;++k)imu.push(sample(k*.01,V3D::Zero(),V3D(1,0,9.81)));
    imu.predict(s,2.,3.);
    near(s.pos_end.x(),.495025,2e-5,"1 m/s^2 translation");
    near(s.vel_end.x(),.995,1e-8,"accelerometer changes velocity");
    near(s.pos_end.z(),0.,1e-9,"gravity does not create vertical drift");
    positive_covariance(s);
    OptionalImu turning(settings());StatesGroup t;stationary(turning);
    turning.predict(t,1.,2.);
    for(int k=201;k<=300;++k)turning.push(sample(k*.01,V3D(0,0,.5)));
    turning.predict(t,2.,3.);
    near(Log(t.rot_end).z(),.4975,1e-8,"gyro drives attitude");
    near(t.pos_end.norm(),0.,1e-9,"pure rotation at co-located origins");
    positive_covariance(t);
}

void extrinsic_and_lever() {
    Config c=settings();c.lidar_from_imu_rotation=Exp(V3D(1,0,0),.8);
    OptionalImu rotated(c);StatesGroup s;
    for(int k=0;k<=200;++k)rotated.push(sample(k*.01,V3D::Zero(),c.lidar_from_imu_rotation.transpose()*V3D(0,0,9.81)));
    rotated.predict(s,1.,2.);
    for(int k=201;k<=300;++k)rotated.push(sample(k*.01,c.lidar_from_imu_rotation.transpose()*V3D(0,0,.4),
                                                  c.lidar_from_imu_rotation.transpose()*V3D(0,0,9.81)));
    rotated.predict(s,2.,3.);
    near(Log(s.rot_end).z(),.398,1e-8,"IMU axes rotated to LiDAR");
    near(Log(s.rot_end).head<2>().norm(),0.,1e-8,"extrinsic does not invent roll/pitch");
    c=settings();c.imu_origin_in_lidar=V3D(.5,.2,0);
    OptionalImu imu(c);StatesGroup l;stationary(imu);imu.predict(l,1.,2.);
    // Rotate about the fixed LiDAR origin; the displaced IMU measures both
    // alpha x lever and omega x (omega x lever).
    const double step=.0025,alpha=.2;
    // Replace the last stationary endpoint's acceleration by a half-ramp
    // through interpolation; the integration error bound is checked below.
    for(int k=1;k<=400;++k) {
        double dt=k*step;V3D w(0,0,alpha*dt),wdot(0,0,alpha);
        V3D a=V3D(0,0,9.81)+wdot.cross(c.imu_origin_in_lidar)+w.cross(w.cross(c.imu_origin_in_lidar));
        imu.push(sample(2.+dt,w,a));
    }
    imu.predict(l,2.,3.);
    near(Log(l.rot_end).z(),.1,1e-6,"angular acceleration attitude");
    near(l.pos_end.norm(),0.,.0003,"lever-arm rotation does not invent translation");
    near(l.vel_end.norm(),0.,.0003,"lever-arm rotational velocity removed");
    positive_covariance(l);
}

void dropout_resume() {
    OptionalImu imu(settings());StatesGroup s;stationary(imu);imu.predict(s,1.,2.);
    for(int k=201;k<=300;++k)imu.push(sample(k*.01,V3D(0,0,.3)));
    imu.predict(s,2.,3.);s.vel_end=V3D(.2,0,0);
    const V3D p=s.pos_end;const M3D r=s.rot_end;
    imu.predict(s,3.,3.5);
    require(!imu.active()&&imu.report().reason=="imu_timeout","dropout returns to CV");
    near((s.pos_end-p-V3D(.1,0,0)).norm(),0.,1e-10,"dropout position continuity");
    near(Log(M3D(r.transpose()*s.rot_end)).z(),.15,1e-8,"dropout preserves last angular rate");
    for(int k=350;k<=400;++k)imu.push(sample(k*.01,V3D(0,0,.3)));
    imu.predict(s,3.5,3.6);require(!imu.active(),"recovery first scan held");
    imu.predict(s,3.6,3.7);require(!imu.active(),"recovery second scan held");
    const V3D p2=s.pos_end;
    imu.predict(s,3.7,3.8);require(imu.active(),"recovered stream reused");
    near((s.pos_end-p2-V3D(.02,0,0)).norm(),0.,1e-8,"resume no position reset");
    near(s.bias_g.norm(),0.,1e-10,"CV angular rate never reused as gyro bias");
    require(imu.report().transitions==3,"three mode transitions");positive_covariance(s);
}

void invalid_and_bounded() {
    auto c=settings();c.max_samples=150;c.buffer_seconds=2.;OptionalImu imu(c);
    require(imu.push(sample(1.)),"first sample valid");
    require(!imu.push(sample(1.)),"duplicate rejected");
    require(!imu.push(sample(.5)),"reversed stamp rejected");
    require(!imu.push(sample(2.,V3D(std::numeric_limits<double>::quiet_NaN(),0,0))),"NaN rejected");
    require(!imu.push(sample(2.,V3D(100,0,0))),"saturated gyro rejected");
    for(int k=101;k<4000;++k)imu.push(sample(k*.01));
    require(imu.report().buffered<=150&&imu.report().evicted>3000,"finite IMU queue");
    require(imu.report().rejected==4,"invalid sample count");
    StatesGroup s;imu.predict(s,1.,2.);require(!imu.active(),"old evicted interval not fabricated");
    OptionalImu hole(settings());stationary(hole);hole.predict(s,1.,2.);
    for(int k=201;k<=300;++k)if(k<220||k>240)hole.push(sample(k*.01));
    hole.predict(s,2.,3.);require(!hole.active()&&hole.report().reason=="imu_gap","interior dropout rejected");
    c=settings();c.calibrated=false;OptionalImu uncal(c);stationary(uncal);StatesGroup u;uncal.predict(u,1.,2.);
    require(!uncal.active()&&uncal.report().reason=="calibration_unconfirmed","uncalibrated IMU ignored");
    c=settings();c.enabled=false;OptionalImu off(c);require(!off.push(sample(1.)),"off ignores samples");
    off.predict(u,1.,2.);require(off.report().reason=="disabled","off explicit");
    OptionalImu moving(settings());StatesGroup v;
    for(int k=0;k<=200;++k)moving.push(sample(k*.01,V3D(0,0,.5)));
    moving.predict(v,1.,2.);
    require(!moving.active()&&moving.report().reason=="waiting_for_stationary_initialization", "moving initialization refused");
}

int main() {
    try {
        auto started=std::chrono::steady_clock::now();
        absent_identical();translation_and_rotation();extrinsic_and_lever();dropout_resume();invalid_and_bounded();
        std::cout << "PASS: optional IMU dynamics, gravity, extrinsics, lever arm, covariance, dropout/resume, bounded queue, invalid inputs, no-IMU equality\n";
        std::cout << "test_seconds=" << std::chrono::duration<double>(std::chrono::steady_clock::now()-started).count() << '\n';
    } catch(const std::exception &e){std::cerr<<e.what()<<'\n';return 1;}
}

#include "registration_quality.hpp"
#include <iostream>
#include <stdexcept>
void require(bool value,const char *reason){if(!value)throw std::runtime_error(reason);}
int main(){
    RegistrationQuality q;q.matched=q.candidates=1000;
    q.information.diagonal()<<1.,2.,1000.,600.,500.,.5;
    q.weighted_information=q.information;
    int weak=0;auto projection=q.observable_projection(weak);
    Eigen::Matrix<double,6,1> error;error<<.7,-.9,.12,.03,-.02,.4;
    Eigen::Matrix<double,6,1> expected;expected<<0.,0.,.12,.03,-.02,0.;
    require(weak==3 && (projection*error-expected).norm()<1e-10,"Flat ground did not preserve height/tilt and exclude tangent/yaw drift");
    auto r=Eigen::AngleAxisd(.4,Eigen::Vector3d::UnitX()).toRotationMatrix();
    RegistrationQuality::M6 frame=RegistrationQuality::M6::Zero();
    frame.topLeftCorner<3,3>()=r;frame.bottomRightCorner<3,3>()=r;
    q.information=frame*q.information*frame.transpose();
    projection=q.observable_projection(weak);
    require(weak==3 && (projection*frame*error-frame*expected).norm()<1e-8,"Projection assumes world XY ground rather than observed plane");
    q.information=1000.*RegistrationQuality::M6::Identity();
    projection=q.observable_projection(weak);
    require(weak==0 && (projection-RegistrationQuality::M6::Identity()).norm()<1e-10,"Strong geometry was constrained");
    std::cout<<"PASS: flat and sloped weak directions; observable height/tilt retained; strong geometry unchanged\n";
}

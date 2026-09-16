#include <Eigen/Dense>
#include <chrono>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <sys/resource.h>
#include "bounded_gain.hpp"

int main() {
  using S = Eigen::Matrix<double,30,30>;
  using D = Eigen::MatrixXd;
  S a=S::Random(), p=a*a.transpose()+.1*S::Identity(), posterior;
  Eigen::Matrix<double,30,Eigen::Dynamic> k;
  double worst_gain=0., worst_cov=0.;
  for (int m : {1,6,30,31,96,192}) {
    D h=D::Random(m,12);
    if (!esekfom::scalar_noise_gain<double,30>(p,h,.05,k,posterior))
      throw std::runtime_error("Valid system rejected");
    D pht=p.leftCols(12)*h.transpose();
    D innovation=h*pht.topRows(12)+.05*D::Identity(m,m);
    D reference=innovation.ldlt().solve(pht.transpose()).transpose();
    S reference_cov=p-reference*h*p.topRows(12);
    double gain_error=(k-reference).norm()/std::max(1.,reference.norm());
    double cov_error=(posterior-reference_cov).norm()/std::max(1.,reference_cov.norm());
    worst_gain=std::max(worst_gain,gain_error);worst_cov=std::max(worst_cov,cov_error);
    if (gain_error>1e-8 || cov_error>1e-8 || Eigen::LLT<S>(posterior).info()!=Eigen::Success)
      throw std::runtime_error("Dense reference or positive covariance mismatch");
  }
  D h=D::Zero(12000,12);
  if (!esekfom::scalar_noise_gain<double,30>(p,h,.05,k,posterior) ||
      k.norm()!=0. || (posterior-p).norm()>1e-9)
    throw std::runtime_error("Zero measurement must preserve prior");
  if (esekfom::scalar_noise_gain<double,30>(p,h,0.,k,posterior))
    throw std::runtime_error("Accepted zero measurement noise");
  h(0,0)=std::numeric_limits<double>::quiet_NaN();
  if (esekfom::scalar_noise_gain<double,30>(p,h,.05,k,posterior))
    throw std::runtime_error("Accepted nonfinite measurement");
  h=D::Random(12000,12);
  auto begin=std::chrono::steady_clock::now();
  if (!esekfom::scalar_noise_gain<double,30>(p,h,.05,k,posterior))
    throw std::runtime_error("Large simultaneous batch rejected");
  double seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-begin).count();
  struct rusage usage{};getrusage(RUSAGE_SELF,&usage);
  if (usage.ru_maxrss>128*1024) throw std::runtime_error("Memory exceeds 128 MiB");
  std::cout<<"{\"dense_gain_relative_error\":"<<worst_gain
    <<",\"dense_covariance_relative_error\":"<<worst_cov
    <<",\"large_batch_points\":12000,\"large_batch_sec\":"<<seconds
    <<",\"peak_rss_mib\":"<<usage.ru_maxrss/1024.<<",\"passed\":true}\n";
}

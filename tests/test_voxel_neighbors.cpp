#include "voxel_map_util.hpp"
#include <iostream>
#include <stdexcept>

using TestVoxelMap = std::unordered_map<VOXEL_LOC,OctoTree*>;
void require(bool ok,const char *why) {if(!ok)throw std::runtime_error(why);}
OctoTree *plane(double x,double y,double z) {
  auto *node=new OctoTree(0,0,{5},100,100,.01);
  node->plane_ptr_->is_plane=true;
  node->plane_ptr_->center=V3D(x,y,z);
  node->plane_ptr_->normal=V3D::UnitZ();
  node->plane_ptr_->x_normal=V3D::UnitX();node->plane_ptr_->y_normal=V3D::UnitY();
  node->plane_ptr_->d=-z;node->plane_ptr_->radius=.5;
  node->plane_ptr_->plane_cov.setIdentity();node->plane_ptr_->plane_cov*=.0001;
  node->voxel_center_[0]=std::floor(x/2)*2+1;
  node->voxel_center_[1]=std::floor(y/2)*2+1;
  node->voxel_center_[2]=std::floor(z/2)*2+1;node->quater_length_=.5;
  return node;
}
void clear(TestVoxelMap &map) {for(auto &p:map){delete p.second->plane_ptr_;delete p.second;}map.clear();}
pointWithCov point(double x,double y,double z) {
  pointWithCov p;p.point=p.point_world=V3D(x,y,z);p.cov=M3D::Identity()*.0001;return p;
}
int main() {
  TestVoxelMap map;std::vector<ptpl> matches,serial;std::vector<V3D> missing,serial_missing;
  auto p=point(3.99,.5,.01);
  map[VOXEL_LOC(2,0,0)]=plane(4.01,.5,.01);
  BuildResidualListOMP(map,2.,3.,0,{p},matches,missing);
  require(matches.size()==1,"empty containing voxel must search existing neighbor");
  map[VOXEL_LOC(1,0,0)]=plane(3.,.5,1.);
  BuildResidualListOMP(map,2.,3.,0,{p},matches,missing);
  require(matches.size()==1,"neighbor direction must use metres, not voxel indices");
  auto uncertain=p;uncertain.cov=M3D::Identity();
  BuildResidualListOMP(map,2.,3.,0,{uncertain},matches,missing);
  require(matches.size()==1 && std::abs(matches[0].d+.01)<1e-7,
          "a plausible containing plane must not hide a better neighboring plane");
  std::vector<pointWithCov> many(1024,p);
  BuildResidualListOMP(map,2.,3.,0,many,matches,missing);
  BuildResidualListNormal(map,2.,3.,0,many,serial,serial_missing);
  require(matches.size()==many.size() && serial.size()==matches.size(),"parallel and serial results differ");
  clear(map);
  auto on_ground=point(3.,.5,-1.);on_ground.cov=M3D::Identity();on_ground.cov(2,2)=9.;
  map[VOXEL_LOC(1,0,-1)]=plane(2.9,.5,-1.);
  auto *wall=plane(4.,.5,-1.);wall->plane_ptr_->normal=V3D::UnitX();wall->plane_ptr_->d=-4.;
  map[VOXEL_LOC(2,0,-1)]=wall;
  BuildResidualListOMP(map,2.,3.,0,{on_ground},matches,missing);
  require(matches.size()==1 && std::abs(matches[0].normal.z())>.99,
          "anisotropic uncertainty must not replace an exact ground match with a distant wall");
  clear(map);
  std::vector<pointWithCov> negative;
  for(int y=0;y<4;++y)for(int z=0;z<4;++z)negative.push_back(point(-2.,.2+y*.1,.2+z*.1));
  buildVoxelMap(negative,2.,0,{5},100,100,.01,map);
  require(map.count(VOXEL_LOC(-1,0,0))==1 && map.count(VOXEL_LOC(-2,0,0))==0,
          "negative exact boundary has an off-by-one voxel index");
  clear(map);
  updateVoxelMap(negative,2.,0,{5},100,100,.01,map);
  require(map.count(VOXEL_LOC(-1,0,0))==1,"map insert/update voxel index mismatch");
  clear(map);
  BuildResidualListOMP(map,2.,3.,0,many,matches,missing);
  require(matches.empty() && missing.size()==many.size(),"unmatched points not accounted for");
  std::cout<<"PASS: empty-cell neighbors, metric directions, negative boundaries, serial/parallel parity, missing points\n";
}

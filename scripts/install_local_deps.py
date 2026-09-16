from pathlib import Path
import subprocess,re
root=Path('/home/yanfa/P3/lidar_visual_fusion')
options=['-o','Dir::State::lists='+str(root/'deps/lists'),'-o','Dir::Etc::sourcelist='+str(root/'deps/sources.list'),'-o','Dir::Etc::sourceparts='+str(root/'deps/empty'),'-o','APT::Get::List-Cleanup=0']
subprocess.run(['apt-get',*options,'update'],check=True)
plan=subprocess.run(['apt-get',*options,'-s','install','--no-install-recommends','ros-humble-robot-localization','libceres-dev','ros-humble-grid-map-ros','ros-humble-pcl-ros'],capture_output=True,text=True,check=True).stdout
names=re.findall(r'^Inst (\S+)',plan,re.M)
print('DOWNLOAD',names,flush=True)
subprocess.run(['apt-get',*options,'download',*names],cwd=root/'deps/debs',check=True)
for p in (root/'deps/debs').glob('*.deb'):
 subprocess.run(['dpkg-deb','-x',str(p),str(root/'deps/root')],check=True)
print('DEPENDENCIES_READY',flush=True)

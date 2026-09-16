import faulthandler,runpy,sys
faulthandler.dump_traceback_later(8,repeat=True)
sys.argv=['assess_bag_run.py','/home/yanfa/P3/lidar_visual_fusion/results/short_xfeat_reliable_20260915_211742']
runpy.run_path('/home/yanfa/P3/lidar_visual_fusion/scripts/assess_bag_run.py',run_name='__main__')

from setuptools import setup, find_packages
setup(name="t3_lidar_visual_fusion", version="0.1.0", packages=find_packages(),
      data_files=[("share/ament_index/resource_index/packages", ["resource/t3_lidar_visual_fusion"]),
                  ("share/t3_lidar_visual_fusion", ["package.xml"])],
      install_requires=["setuptools"], zip_safe=True,
      maintainer="T3", maintainer_email="t3@example.com",
      description="VoxelMap with guarded lightweight vision and persistent terrain maps",
      license="GPL-3.0-or-later",
      entry_points={"console_scripts": [
          "sensor_adapter = t3_lidar_visual_fusion.sensor_adapter:main",
          "odometry_guard = t3_lidar_visual_fusion.odometry_guard:main",
            "adaptive_guard = t3_lidar_visual_fusion.adaptive_guard:main",
          "terrain_mapper = t3_lidar_visual_fusion.terrain_mapper:main",
          "learned_odometry = t3_lidar_visual_fusion.learned_odometry:main"]})

"""Launch file for the Galaxy2 dual Daheng GigE camera acquisition node.

Topic = topic_prefix + cam_name + "/" + topic_suffix
  e.g. "Car/T5/" + "Cam_Left" + "/" + "image_raw" → /Car/T5/Cam_Left/image_raw

Image resolution is auto-calculated from binning/decimation:
  binning=1, decimation=1 → 2448x2048
  binning=2, decimation=1 → 1224x1024

Acquisition mode (trigger_mode):
  continuous  — free-run at frame_rate (default)
  external    — hardware trigger on trigger_source (Line0/Line2/Line3)
                with trigger_activation edge (RisingEdge/FallingEdge)
  software   — software trigger via ROS2 topic /sensors_trigger (std_msgs/Header)

ros2 launch galaxy2 galaxy2_dual.launch.py

External trigger example:
ros2 launch galaxy2 galaxy2_dual.launch.py trigger_mode:=external trigger_source:=Line2 trigger_activation:=RisingEdge

Software trigger example:
ros2 launch galaxy2 galaxy2_dual.launch.py trigger_mode:=software
# then capture frames:
ros2 topic pub -r 1 /sensors_trigger std_msgs/msg/Header "{stamp: {sec: 0, nanosec: 0}}"
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
import os


def generate_launch_description():
    args = [
        DeclareLaunchArgument('cam1_ip', default_value='192.168.19.10'),
        DeclareLaunchArgument('cam2_ip', default_value='192.168.19.11'),
        # --- Topic 命名 ---
        DeclareLaunchArgument('topic_prefix', default_value='Car/T5/'),
        DeclareLaunchArgument('cam1_name', default_value='Cam_Left'),
        DeclareLaunchArgument('cam2_name', default_value='Cam_Right'),
        DeclareLaunchArgument('topic_image_raw', default_value='image_raw'),
        DeclareLaunchArgument('topic_image_color', default_value='image_raw/color'),
        DeclareLaunchArgument('topic_image_compressed', default_value='image_raw/color/compressed'),
        DeclareLaunchArgument('topic_image_light', default_value='image_raw/light'),
        DeclareLaunchArgument('topic_camera_info', default_value='camera_info'),
        # --- 采集参数 ---
        DeclareLaunchArgument('frame_rate', default_value='10.0'),
        DeclareLaunchArgument('exposure_time', default_value='100000.0'),
        # 默认只发布压缩图像，减少 WiFi 带宽压力 (raw 5MB/帧, color 15MB/帧, compressed ~0.5MB/帧)
        # 需要时通过命令行覆盖: ros2 launch galaxy2 galaxy2_dual.launch.py publish_raw:=true publish_color:=true
        DeclareLaunchArgument('publish_raw', default_value='false'),
        DeclareLaunchArgument('publish_color', default_value='true'),
        DeclareLaunchArgument('publish_mapping', default_value='false'),
        DeclareLaunchArgument('mapping_width', default_value='640'),
        DeclareLaunchArgument('mapping_height', default_value='536'),
        DeclareLaunchArgument('publish_color_compressed', default_value='true'),
        # 轻量化图像: 长宽各缩放 1/8, 用于远程 WiFi 查看 (2448x2048 -> 306x256, ~0.23MB/帧)
        DeclareLaunchArgument('publish_light', default_value='true'),
        DeclareLaunchArgument('light_scale', default_value='0.125'),
        DeclareLaunchArgument('jpeg_quality', default_value='80'),
        # --- 相机参数 (两相机统一) ---
        DeclareLaunchArgument('auto_white_balance', default_value='true'),
        DeclareLaunchArgument('auto_gain', default_value='true'),
        DeclareLaunchArgument('gain', default_value='0.0'),
        DeclareLaunchArgument('gamma_enable', default_value='true'),
        DeclareLaunchArgument('wb_red', default_value='1.0'),
        DeclareLaunchArgument('wb_green', default_value='1.0'),
        DeclareLaunchArgument('wb_blue', default_value='1.0'),
        DeclareLaunchArgument('binning_horizontal', default_value='1'),
        DeclareLaunchArgument('binning_vertical', default_value='1'),
        DeclareLaunchArgument('binning_mode', default_value='Average'),
        DeclareLaunchArgument('decimation_horizontal', default_value='1'),
        DeclareLaunchArgument('decimation_vertical', default_value='1'),
        # --- 触发模式 ---
        # trigger_mode: continuous (free-run) | external (hardware trigger) | software (software trigger)
        DeclareLaunchArgument('trigger_mode', default_value='continuous'),
        # trigger_source: Line0 | Line2 | Line3 (仅 external 模式生效)
        DeclareLaunchArgument('trigger_source', default_value='Line2'),
        # trigger_activation: RisingEdge | FallingEdge (仅 external 模式生效)
        DeclareLaunchArgument('trigger_activation', default_value='RisingEdge'),
        # trigger_topic: 触发 topic, 与 TOF2 共用, 消息类型 std_msgs/Header, stamp 作为帧时间戳
        DeclareLaunchArgument('trigger_topic', default_value='/sensors_trigger'),
        # --- 图像保存 ---
        # 是否保存图像到文件（默认关闭，启用后每帧保存 PNG）
        DeclareLaunchArgument('enable_save', default_value='false'),
        # 保存基目录, 节点自动创建 {save_dir}/Galaxy2_{时间戳}/ 子目录
        DeclareLaunchArgument('save_dir', default_value=os.path.expanduser('~/SaveImages/')),
    ]

    galaxy_node = Node(
        package='galaxy2',
        executable='galaxy2_node',
        name='galaxy_node',
        output='screen',
        emulate_tty=True,
        parameters=[{
            'cam1_ip': ParameterValue(LaunchConfiguration('cam1_ip'), value_type=str),
            'cam2_ip': ParameterValue(LaunchConfiguration('cam2_ip'), value_type=str),
            'topic_prefix': ParameterValue(LaunchConfiguration('topic_prefix'), value_type=str),
            'cam1_name': ParameterValue(LaunchConfiguration('cam1_name'), value_type=str),
            'cam2_name': ParameterValue(LaunchConfiguration('cam2_name'), value_type=str),
            'topic_image_raw': ParameterValue(LaunchConfiguration('topic_image_raw'), value_type=str),
            'topic_image_color': ParameterValue(LaunchConfiguration('topic_image_color'), value_type=str),
            'topic_image_compressed': ParameterValue(LaunchConfiguration('topic_image_compressed'), value_type=str),
            'topic_image_light': ParameterValue(LaunchConfiguration('topic_image_light'), value_type=str),
            'topic_camera_info': ParameterValue(LaunchConfiguration('topic_camera_info'), value_type=str),
            'frame_rate': ParameterValue(LaunchConfiguration('frame_rate'), value_type=float),
            'exposure_time': ParameterValue(LaunchConfiguration('exposure_time'), value_type=float),
            'publish_raw': ParameterValue(LaunchConfiguration('publish_raw'), value_type=bool),
            'publish_color': ParameterValue(LaunchConfiguration('publish_color'), value_type=bool),
            'publish_mapping': ParameterValue(LaunchConfiguration('publish_mapping'), value_type=bool),
            'mapping_width': ParameterValue(LaunchConfiguration('mapping_width'), value_type=int),
            'mapping_height': ParameterValue(LaunchConfiguration('mapping_height'), value_type=int),
            'publish_color_compressed': ParameterValue(LaunchConfiguration('publish_color_compressed'), value_type=bool),
            'publish_light': ParameterValue(LaunchConfiguration('publish_light'), value_type=bool),
            'light_scale': ParameterValue(LaunchConfiguration('light_scale'), value_type=float),
            'jpeg_quality': ParameterValue(LaunchConfiguration('jpeg_quality'), value_type=int),
            'auto_white_balance': ParameterValue(LaunchConfiguration('auto_white_balance'), value_type=bool),
            'auto_gain': ParameterValue(LaunchConfiguration('auto_gain'), value_type=bool),
            'gain': ParameterValue(LaunchConfiguration('gain'), value_type=float),
            'gamma_enable': ParameterValue(LaunchConfiguration('gamma_enable'), value_type=bool),
            'wb_red': ParameterValue(LaunchConfiguration('wb_red'), value_type=float),
            'wb_green': ParameterValue(LaunchConfiguration('wb_green'), value_type=float),
            'wb_blue': ParameterValue(LaunchConfiguration('wb_blue'), value_type=float),
            'binning_horizontal': ParameterValue(LaunchConfiguration('binning_horizontal'), value_type=int),
            'binning_vertical': ParameterValue(LaunchConfiguration('binning_vertical'), value_type=int),
            'binning_mode': ParameterValue(LaunchConfiguration('binning_mode'), value_type=str),
            'decimation_horizontal': ParameterValue(LaunchConfiguration('decimation_horizontal'), value_type=int),
            'decimation_vertical': ParameterValue(LaunchConfiguration('decimation_vertical'), value_type=int),
            'trigger_mode': ParameterValue(LaunchConfiguration('trigger_mode'), value_type=str),
            'trigger_source': ParameterValue(LaunchConfiguration('trigger_source'), value_type=str),
            'trigger_activation': ParameterValue(LaunchConfiguration('trigger_activation'), value_type=str),
            'trigger_topic': ParameterValue(LaunchConfiguration('trigger_topic'), value_type=str),
            'cam1_frame_id': 'galaxy_Cam_Left_frame',
            'cam2_frame_id': 'galaxy_Cam_Right_frame',
            # --- 图像保存 ---
            'enable_save': ParameterValue(LaunchConfiguration('enable_save'), value_type=bool),
            'save_dir': ParameterValue(LaunchConfiguration('save_dir'), value_type=str),
        }],
    )

    return LaunchDescription(args + [galaxy_node])

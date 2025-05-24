from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration

from os.path import expanduser

image_topic_arg = DeclareLaunchArgument("image_topic", default_value="/routecam/image_raw")
# tire_model_arg = DeclareLaunchArgument("tire_model_path", default_value="/home/ryank/ros2_ws/src/route_yolo_ros2/route_yolo_ros2/models/best-05142025.pt")
tire_model_arg = DeclareLaunchArgument("tire_model_path", default_value=f"{expanduser("~")}/ros2_ws/src/route_yolo_ros2/route_yolo_ros2/models/best-tire05212025.pt")
coco_model_arg = DeclareLaunchArgument("coco_model_path", default_value=f"{expanduser("~")}/ros2_ws/src/route_yolo_ros2/route_yolo_ros2/models/yolo12m.pt")

def detect():
    return Node(
        package='route_yolo_ros2',
        executable='detect',
        name='yolo_detector',
        output='screen',
        parameters=[
            {'image_topic': LaunchConfiguration("image_topic")},
            {'coco_model_path': LaunchConfiguration("coco_model_path")},
            {'tire_model_path': LaunchConfiguration("tire_model_path")},
            {'flip_image': False},
            {'image_resize': 640},
        ],
    )

def generate_launch_description():
    return LaunchDescription([
        image_topic_arg,
        coco_model_arg,
        tire_model_arg,
        detect(),
    ])

from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():

    perception = Node(
        package="perception",
        executable="perception_node.py",
        output="screen"
    )

    return LaunchDescription([perception])
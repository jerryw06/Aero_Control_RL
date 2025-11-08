from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='rl_with_cpp',
            executable='train_rl',
            name='train_rl',
            output='screen'
        )
    ])

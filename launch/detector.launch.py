"""Запускает bag play + detector_node + rviz2 вместе.

Закрытие окна RViz гасит весь launch (bag play и узел останавливаются тоже) -
через событие OnProcessExit на процессе rviz2.
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, EmitEvent, ExecuteProcess, RegisterEventHandler
from launch.event_handlers import OnProcessExit
from launch.events import Shutdown
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    default_rviz = os.path.join(
        get_package_share_directory('robot_b_detector'), 'rviz', 'livox_view.rviz')

    bag_path_arg = DeclareLaunchArgument(
        'bag_path',
        description='Путь к папке bag-файла (rosbag2, mcap)')
    rviz_config_arg = DeclareLaunchArgument(
        'rviz_config', default_value=default_rviz)

    bag_play = ExecuteProcess(
        cmd=['ros2', 'bag', 'play', LaunchConfiguration('bag_path'), '--loop', '--clock'],
        output='screen')

    detector_node = Node(
        package='robot_b_detector',
        executable='detector_node',
        output='screen')

    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        arguments=['-d', LaunchConfiguration('rviz_config')],
        output='screen')

    shutdown_on_rviz_exit = RegisterEventHandler(
        OnProcessExit(target_action=rviz_node, on_exit=[EmitEvent(event=Shutdown())]))

    return LaunchDescription([
        bag_path_arg,
        rviz_config_arg,
        bag_play,
        detector_node,
        rviz_node,
        shutdown_on_rviz_exit,
    ])

from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    rviz_config = os.path.join(
        get_package_share_directory('underwater_slam'),
        'config',
        'slam.rviz'
    )

    return LaunchDescription([
        Node(
    package='tf2_ros',
    executable='static_transform_publisher',
    arguments=[
        '--frame-id', 'base_link',
        '--child-frame-id', 'dvl_frame',
        '--x', '0', '--y', '0', '--z', '0',
        '--roll', '0', '--pitch', '0', '--yaw', '0'
    ],
    name='dvl_tf_publisher'
    ),

    Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        arguments=[
            '--frame-id', 'base_link',
            '--child-frame-id', 'sonar_frame',
            '--x', '0.5', '--y', '0', '--z', '-0.3',
            '--roll', '0', '--pitch', '0', '--yaw', '0'
        ],
        name='sonar_tf_publisher'
    ),

    Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        arguments=[
            '--frame-id', 'base_link',
            '--child-frame-id', 'camera_optical_frame',
            '--x', '0.1', '--y', '0', '--z', '0.05',
            '--roll', '0', '--pitch', '0', '--yaw', '0'
        ],
        name='camera_tf_publisher'
    ),
        Node(
            package='underwater_slam',
            executable='dvl_publisher',
            name='dvl_publisher'
        ),
        Node(
            package='underwater_slam',
            executable='sonar_publisher',
            name='sonar_publisher'
        ),
        Node(
            package='underwater_slam',
            executable='image_publisher',
            name='image_publisher'
        ),
        Node(
            package='underwater_slam',
            executable='slam',
            name='slam',
            parameters=[{
                'map_resolution': 0.1,
                'map_size': 200,  # Larger map for continuous building
                'num_particles': 500,
                'enable_visual_slam': True,
                'enable_sonar_mapping': True
            }]
        ),
        Node(
            package='rviz2',
            executable='rviz2',
            name='rviz2',
            arguments=['-d', rviz_config],
            parameters=[{'use_sim_time': False}]
        )
    ])
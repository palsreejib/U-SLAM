"""
mps_sim.launch.py
─────────────────
Launches the full MPS + FLS fusion simulation stack:

  1. Gazebo Classic with mps_world1.world (structured obstacles + the
     libgazebo_ros_state.so plugin that makes ground truth possible)
  2. robot_state_publisher       (URDF → TF)
  3. spawn_entity                 (drops vehicle into Gazebo)
  4. ros2_control joint position controller for mps_rotate_joint
  5. mps_driver_node              (MPS: rotating single-ray sonar step logic)
  6. ground_truth_node            (/gazebo/model_states → /ground_truth/odometry)
  7. dvl_node, pressure_node      (derived from ground truth — see README)
  8. sync_node                    (MPS + FLS fusion → /training/sample)

NOTE ON THE FLS ITSELF: it is NOT a node in this launch file. It's a
Gazebo sensor plugin baked directly into the URDF (see
urdf/mps_vehicle.urdf.xacro), so it comes up automatically when Gazebo
loads the robot — same as the MPS ray sensor. It requires
libnps_multibeam_sonar_ros_plugin.so to be on your Gazebo plugin path,
which means building forssea-robotics/nps_uw_multibeam_sonar in your
workspace FIRST. See README for install steps and the NVIDIA GPU / CUDA
requirement — if that plugin isn't built, Gazebo will print a plugin-
load error for the fls_sensor and /fls/image will simply never appear,
while everything else in this launch file keeps working normally.

Usage:
  ros2 launch mps_sim mps_sim.launch.py

Optional args:
  world:=mps_world1.world      or mps_test.world for the simpler box world
  n_steps:=200                 MPS bearing steps per sweep
  step_period_sec:=0.025       MPS seconds between steps (5.0s/sweep default)
  max_range:=30.0              MPS max range (m)
  fls_horizontal_fov_deg:=120.0
  fls_max_range:=10.0
  fls_update_rate:=15.0
  mps_buffer_size:=36          sync_node rolling MPS buffer size
  use_sim_time:=true
"""

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    IncludeLaunchDescription,
    ExecuteProcess,
    TimerAction,
)
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, Command, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():

    pkg = get_package_share_directory('mps_sim')
    gazebo_ros_pkg = get_package_share_directory('gazebo_ros')

    # ── Launch arguments ─────────────────────────────────────────────
    use_sim_time_arg = DeclareLaunchArgument(
        'use_sim_time', default_value='true',
        description='Use Gazebo simulation clock'
    )
    world_arg = DeclareLaunchArgument(
        'world', default_value='mps_world1.world',
        description='World file (in worlds/) — mps_world1.world (structured '
                     'obstacles, default) or mps_test.world (simple box world)'
    )
    n_steps_arg = DeclareLaunchArgument(
        'n_steps', default_value='200',
        description='Number of MPS bearing steps per 360° sweep'
    )
    step_period_arg = DeclareLaunchArgument(
        'step_period_sec', default_value='0.025',
        description='Time between MPS steps (seconds) — 200*0.025=5.0s/sweep'
    )
    max_range_arg = DeclareLaunchArgument(
        'max_range', default_value='30.0',
        description='MPS maximum range (metres)'
    )
    fls_fov_arg = DeclareLaunchArgument(
        'fls_horizontal_fov_deg', default_value='120.0',
        description='FLS horizontal field of view (degrees)'
    )
    fls_range_arg = DeclareLaunchArgument(
        'fls_max_range', default_value='10.0',
        description='FLS maximum range (metres)'
    )
    fls_rate_arg = DeclareLaunchArgument(
        'fls_update_rate', default_value='15.0',
        description='FLS frame rate (Hz)'
    )
    mps_buffer_arg = DeclareLaunchArgument(
        'mps_buffer_size', default_value='36',
        description='sync_node: number of MPS sectors kept in the rolling buffer'
    )

    use_sim_time   = LaunchConfiguration('use_sim_time')
    world_file_arg = LaunchConfiguration('world')
    n_steps        = LaunchConfiguration('n_steps')
    step_period    = LaunchConfiguration('step_period_sec')
    max_range      = LaunchConfiguration('max_range')
    fls_fov        = LaunchConfiguration('fls_horizontal_fov_deg')
    fls_range      = LaunchConfiguration('fls_max_range')
    fls_rate       = LaunchConfiguration('fls_update_rate')
    mps_buffer     = LaunchConfiguration('mps_buffer_size')

    # ── 1. Gazebo ────────────────────────────────────────────────────
    world_path = PathJoinSubstitution([pkg, 'worlds', world_file_arg])

    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(gazebo_ros_pkg, 'launch', 'gazebo.launch.py')
        ),
        launch_arguments={
            'world': world_path,
            'verbose': 'true',
        }.items()
    )

    # ── 2. robot_state_publisher ─────────────────────────────────────
    xacro_file = os.path.join(pkg, 'urdf', 'mps_vehicle.urdf.xacro')
    robot_desc = ParameterValue(Command([
        'xacro ', xacro_file,
        ' fls_horizontal_fov_deg:=', fls_fov,
        ' fls_max_range:=', fls_range,
        ' fls_update_rate:=', fls_rate,
    ]), value_type=str)

    rsp = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        output='screen',
        parameters=[{
            'robot_description': robot_desc,
            'use_sim_time': use_sim_time,
        }]
    )

    # ── 3. Spawn vehicle ─────────────────────────────────────────────
    # NOTE: -entity name here MUST match ground_truth_node's
    # "vehicle_name" parameter (default "mps_vehicle") — that's how it
    # finds us inside /gazebo/model_states.
    spawn = Node(
        package='gazebo_ros',
        executable='spawn_entity.py',
        name='spawn_mps_vehicle',
        arguments=[
            '-topic', 'robot_description',
            '-entity', 'mps_vehicle',
            '-x', '0', '-y', '0', '-z', '0.5',
        ],
        output='screen'
    )

    # ── 5. MPS driver node ───────────────────────────────────────────
    mps_driver = Node(
        package='mps_sim',
        executable='mps_driver_node.py',
        name='mps_driver',
        output='screen',
        parameters=[{
            'use_sim_time':      use_sim_time,
            'n_steps':           n_steps,
            'step_period_sec':   step_period,
            'max_range':         max_range,
            'min_range':         0.3,
            'noise_sigma_base':  0.01,
            'dropout_prob':      0.02,
            'specular_thresh':   0.10,
        }]
    )

    # ── 6. Ground truth logger ────────────────────────────────────────
    ground_truth = Node(
        package='mps_sim',
        executable='ground_truth_node.py',
        name='ground_truth_node',
        output='screen',
        parameters=[{
            'use_sim_time':  use_sim_time,
            'vehicle_name':  'mps_vehicle',
            'world_frame':   'world',
            'base_frame':    'base_link',
        }]
    )

    # ── 7. DVL + pressure (derived from ground truth) ────────────────
    dvl = Node(
        package='mps_sim',
        executable='dvl_node.py',
        name='dvl_node',
        output='screen',
        parameters=[{
            'use_sim_time':      use_sim_time,
            'publish_rate_hz':   7.0,
            'noise_percent':     0.5,
            'noise_floor':       0.002,
            'seafloor_z':        0.0,
            'max_lock_range':    30.0,
        }]
    )

    pressure = Node(
        package='mps_sim',
        executable='pressure_node.py',
        name='pressure_node',
        output='screen',
        parameters=[{
            'use_sim_time':      use_sim_time,
            'publish_rate_hz':   20.0,
            'water_surface_z':   0.0,
            'noise_stddev_pa':   50.0,
        }]
    )

    # ── 8. Sensor fusion / sync node ──────────────────────────────────
    sync = Node(
        package='mps_sim',
        executable='sync_node.py',
        name='sync_node',
        output='screen',
        parameters=[{
            'use_sim_time':             use_sim_time,
            'mps_buffer_size':          mps_buffer,
            'sync_warn_threshold_sec':  0.15,
        }]
    )

    # Stagger startup: controller → MPS driver → everything else.
    # Ground truth/DVL/pressure/sync don't depend on the controller,
    # but staggering keeps startup logs readable and avoids a burst of
    # "waiting for topic" warnings on slower machines.
    delayed_driver        = TimerAction(period=7.0, actions=[mps_driver])
    delayed_ground_truth  = TimerAction(period=3.0, actions=[ground_truth])
    delayed_dvl_pressure  = TimerAction(period=4.0, actions=[dvl, pressure])
    delayed_sync          = TimerAction(period=8.0, actions=[sync])

    return LaunchDescription([
        use_sim_time_arg,
        world_arg,
        n_steps_arg,
        step_period_arg,
        max_range_arg,
        fls_fov_arg,
        fls_range_arg,
        fls_rate_arg,
        mps_buffer_arg,
        gazebo,
        rsp,
        spawn,
        delayed_driver,
        delayed_ground_truth,
        delayed_dvl_pressure,
        delayed_sync,
    ])

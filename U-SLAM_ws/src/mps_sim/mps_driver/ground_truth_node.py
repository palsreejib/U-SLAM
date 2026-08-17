#!/usr/bin/env python3
"""
ground_truth_node.py — Ground truth pose/twist logger for training labels.

Subscribes to /gazebo/model_states (gazebo_msgs/ModelStates), which is
published by the libgazebo_ros_state.so world plugin (see worlds/mps_world1.world
— without that plugin loaded, this topic does not exist at all).

Finds our vehicle by name in the ModelStates arrays and republishes its
pose + twist as a properly-stamped nav_msgs/Odometry on /ground_truth/odometry.

Why this matters: ModelStates itself carries NO per-message timestamp
(it's just parallel arrays of name/pose/twist, refreshed continuously) —
this node is what attaches the current sim-time stamp, which every other
part of this pipeline (sync_node.py in particular) relies on to do
nearest-in-time matching against MPS/FLS/IMU/DVL/pressure.
"""

import math

import rclpy
from rclpy.node import Node

from gazebo_msgs.msg import ModelStates
from nav_msgs.msg import Odometry


class GroundTruthNode(Node):

    def __init__(self):
        super().__init__('ground_truth_node')

        self.declare_parameter('vehicle_name', 'mps_vehicle')
        self.declare_parameter('world_frame', 'world')
        self.declare_parameter('base_frame', 'base_link')

        self.vehicle_name = self.get_parameter('vehicle_name').value
        self.world_frame  = self.get_parameter('world_frame').value
        self.base_frame   = self.get_parameter('base_frame').value

        self.odom_pub = self.create_publisher(Odometry, '/ground_truth/odometry', 10)

        self.sub = self.create_subscription(
            ModelStates,
            '/gazebo/model_states',
            self._on_model_states,
            10
        )

        self._warned_missing = False
        self.get_logger().info(
            f'Ground truth node started, tracking model "{self.vehicle_name}". '
            f'Waiting for /gazebo/model_states '
            f'(requires libgazebo_ros_state.so to be loaded in the world file).'
        )

    def _on_model_states(self, msg: ModelStates):
        try:
            idx = msg.name.index(self.vehicle_name)
        except ValueError:
            if not self._warned_missing:
                self.get_logger().warn(
                    f'"{self.vehicle_name}" not found in /gazebo/model_states. '
                    f'Available models: {list(msg.name)}. '
                    f'Check the "vehicle_name" parameter matches the -entity '
                    f'name used in spawn_entity.py.'
                )
                self._warned_missing = True
            return
        self._warned_missing = False

        odom = Odometry()
        odom.header.stamp = self.get_clock().now().to_msg()
        odom.header.frame_id = self.world_frame
        odom.child_frame_id = self.base_frame

        odom.pose.pose = msg.pose[idx]
        odom.twist.twist = msg.twist[idx]

        # ModelStates carries no covariance; leave at zero (== "unknown")
        # rather than fabricating confident-looking numbers. This is
        # ground truth from the simulator, so "unknown covariance" is
        # honest — a learned model shouldn't be trained to expect
        # calibrated uncertainty on its own label.
        odom.pose.covariance = [0.0] * 36
        odom.twist.covariance = [0.0] * 36

        self.odom_pub.publish(odom)


def main(args=None):
    rclpy.init(args=args)
    node = GroundTruthNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

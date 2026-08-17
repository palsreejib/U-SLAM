#!/usr/bin/env python3
"""
dvl_node.py — Simulated DVL, derived from ground-truth velocity.

Why this exists instead of using Dave's real DVL plugin:
Dave's actual DVL implementation (whoi_teledyne_whn and friends) lives in
Field-Robotics-Lab/nps_uw_sensors_gazebo, which is ROS1/Noetic + Gazebo
Classic era code with no confirmed ROS 2 port. Rather than gambling the
whole build on porting that plugin, this node derives an equally useful
signal directly from Gazebo ground truth (/ground_truth/odometry) plus a
noise model shaped like real DVL vendor specs (percent-of-reading + a
fixed floor). If you specifically need the real WHN-family plugin later,
see the README for pointers on swapping this out.

Physics:
  1. Take world-frame linear velocity from ground-truth odometry.
  2. Rotate into the body frame using the vehicle's current orientation
     (a real DVL reports velocity in its own frame, not the world frame).
  3. Add Gaussian noise scaled as a percentage of speed + a fixed floor
     (this is how real DVL error is specified, e.g. "0.2% of reading
     ± 1 mm/s" for a Teledyne Workhorse-class instrument).
  4. Estimate altitude as vehicle z-height above the seafloor, ASSUMING
     A FLAT SEAFLOOR AT z=0 (true for the worlds shipped in this
     package). If you change the world, update `seafloor_z`.
  5. If altitude exceeds max_lock_range, mark the reading invalid
     (simulates losing bottom lock — a real DVL failure mode).
"""

import math
import random

import numpy as np
import rclpy
from rclpy.node import Node

from nav_msgs.msg import Odometry
from geometry_msgs.msg import Vector3

from mps_sim.msg import DvlReading


def quat_conjugate(q_xyzw):
    x, y, z, w = q_xyzw
    return np.array([-x, -y, -z, w])


def quat_rotate_vector(q_xyzw, v):
    """Rotate 3-vector v by unit quaternion q=(x,y,z,w)."""
    q_xyz = np.array(q_xyzw[:3])
    w = q_xyzw[3]
    t = 2.0 * np.cross(q_xyz, v)
    return v + w * t + np.cross(q_xyz, t)


class DvlNode(Node):

    def __init__(self):
        super().__init__('dvl_node')

        self.declare_parameter('publish_rate_hz', 7.0)   # typical WHN-class DVL rate
        self.declare_parameter('noise_percent', 0.5)      # % of reading, 1-sigma
        self.declare_parameter('noise_floor', 0.002)      # m/s, 1-sigma fixed floor
        self.declare_parameter('seafloor_z', 0.0)         # world-frame z of the seafloor
        self.declare_parameter('max_lock_range', 30.0)    # m, beyond this: dropout
        self.declare_parameter('frame_id', 'dvl_link')

        self.rate_hz         = self.get_parameter('publish_rate_hz').value
        self.noise_percent   = self.get_parameter('noise_percent').value
        self.noise_floor     = self.get_parameter('noise_floor').value
        self.seafloor_z      = self.get_parameter('seafloor_z').value
        self.max_lock_range  = self.get_parameter('max_lock_range').value
        self.frame_id        = self.get_parameter('frame_id').value

        self.latest_odom = None

        self.pub = self.create_publisher(DvlReading, '/dvl/data', 10)
        self.sub = self.create_subscription(
            Odometry, '/ground_truth/odometry', self._on_odom, 10
        )
        self.timer = self.create_timer(1.0 / self.rate_hz, self._step)

        self.get_logger().info(
            f'DVL node started at {self.rate_hz}Hz '
            f'(derived from ground truth, not a native Dave plugin — see README)'
        )

    def _on_odom(self, msg: Odometry):
        self.latest_odom = msg

    def _step(self):
        if self.latest_odom is None:
            return

        odom = self.latest_odom
        q = odom.pose.pose.orientation
        q_xyzw = (q.x, q.y, q.z, q.w)

        v_world = np.array([
            odom.twist.twist.linear.x,
            odom.twist.twist.linear.y,
            odom.twist.twist.linear.z,
        ])

        # World -> body frame: rotate by the conjugate of the orientation
        v_body = quat_rotate_vector(quat_conjugate(q_xyzw), v_world)

        speed = float(np.linalg.norm(v_body))
        sigma = self.noise_percent / 100.0 * speed + self.noise_floor
        v_body_noisy = v_body + np.random.normal(0.0, sigma, size=3)

        # Altitude above assumed-flat seafloor
        altitude = float(odom.pose.pose.position.z - self.seafloor_z)
        valid = 0.0 <= altitude <= self.max_lock_range

        reading = DvlReading()
        reading.header.stamp = self.get_clock().now().to_msg()
        reading.header.frame_id = self.frame_id
        reading.velocity = Vector3(
            x=float(v_body_noisy[0]),
            y=float(v_body_noisy[1]),
            z=float(v_body_noisy[2]),
        )
        reading.altitude = altitude
        reading.velocity_valid = bool(valid)
        reading.noise_percent = float(self.noise_percent)

        self.pub.publish(reading)


def main(args=None):
    rclpy.init(args=args)
    node = DvlNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

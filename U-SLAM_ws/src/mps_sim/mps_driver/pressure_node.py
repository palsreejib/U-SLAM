#!/usr/bin/env python3
"""
pressure_node.py — Simulated depth/pressure sensor, derived from ground truth.

Gazebo Classic has no built-in "underwater pressure sensor" plugin, and
this project isn't pulling in Dave's full hydrodynamics/buoyancy stack —
just sensors. So depth is computed directly and honestly from the
vehicle's ground-truth z-position against an assumed water surface
height, then converted to pressure via the standard hydrostatic formula:

    P = P_atm + rho_seawater * g * depth

IMPORTANT ASSUMPTION: this treats `water_surface_z` (a parameter, default
0.0) as sea level in the world frame, and depth = max(0, water_surface_z
- vehicle_z). The worlds shipped in this package don't render an actual
water volume (Gazebo Classic's plain sensors don't know about water
optically) — this parameter is what defines "underwater" for the
pressure model. If you change the vehicle's operating height in your
world file, update this parameter to match, or the depth numbers will
be meaningless.
"""

import rclpy
from rclpy.node import Node

from nav_msgs.msg import Odometry
from sensor_msgs.msg import FluidPressure

import numpy as np

P_ATM_PA = 101325.0        # standard atmospheric pressure, Pa
RHO_SEAWATER = 1025.0      # kg/m^3
G = 9.80665                # m/s^2


class PressureNode(Node):

    def __init__(self):
        super().__init__('pressure_node')

        self.declare_parameter('publish_rate_hz', 20.0)
        self.declare_parameter('water_surface_z', 0.0)
        self.declare_parameter('noise_stddev_pa', 50.0)  # ~5cm depth-equivalent
        self.declare_parameter('frame_id', 'pressure_link')

        self.rate_hz         = self.get_parameter('publish_rate_hz').value
        self.water_surface_z = self.get_parameter('water_surface_z').value
        self.noise_stddev    = self.get_parameter('noise_stddev_pa').value
        self.frame_id        = self.get_parameter('frame_id').value

        self.latest_odom = None

        self.pub = self.create_publisher(FluidPressure, '/pressure/data', 10)
        self.sub = self.create_subscription(
            Odometry, '/ground_truth/odometry', self._on_odom, 10
        )
        self.timer = self.create_timer(1.0 / self.rate_hz, self._step)

        self.get_logger().info(
            f'Pressure node started at {self.rate_hz}Hz '
            f'(water_surface_z={self.water_surface_z} — depth = '
            f'water_surface_z - vehicle_z, see module docstring)'
        )

    def _on_odom(self, msg: Odometry):
        self.latest_odom = msg

    def _step(self):
        if self.latest_odom is None:
            return

        z = self.latest_odom.pose.pose.position.z
        depth = max(0.0, self.water_surface_z - z)

        pressure_pa = P_ATM_PA + RHO_SEAWATER * G * depth
        pressure_pa += float(np.random.normal(0.0, self.noise_stddev))

        msg = FluidPressure()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.frame_id
        msg.fluid_pressure = float(pressure_pa)
        msg.variance = float(self.noise_stddev ** 2)

        self.pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = PressureNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

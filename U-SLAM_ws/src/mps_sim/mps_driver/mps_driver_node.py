#!/usr/bin/env python3
"""
mps_driver_node.py — Mechanical Profiling Sonar driver for Gazebo Classic 11 / ROS 2 Humble

What this node does
───────────────────
1.  Maintains the current bearing angle (starts at -π, steps by ANGLE_STEP_RAD each tick).
2.  On each tick it commands the mps_rotate_joint to the new bearing via a
    JointTrajectory command (ros_control position interface).
3.  It subscribes to /mps/raw_range (sensor_msgs/Range from the Gazebo ray plugin).
4.  When a new Range message arrives it applies the sonar noise model (range-dependent
    Gaussian sigma + dropout + specular clipping) and publishes:
      - /mps/sector       (MpsSector) — one message per bearing, stamped at *that moment*
      - /mps/scan         (sensor_msgs/LaserScan) — assembled after each full 360° sweep
5.  The key physics property: each sector message has its own header.stamp equal to the
    ROS clock time at which that ray was fired. Because the vehicle is moving, each bearing
    is measured from a slightly different vehicle pose. Sweep distortion is present in the
    data by construction.

Parameters (set via ROS 2 parameter server or launch file)
──────────────────────────────────────────────────────────
  n_steps          int     200       number of bearing steps per 360° sweep
  step_period_sec  float   0.025     seconds between steps (200 steps * 0.025s
                                      = 5.0s/sweep, matching project spec)
  max_range        float   30.0      maximum sonar range (m) — project spec
  min_range        float   0.3       minimum sonar range (m)
  noise_sigma_base float   0.01      Gaussian range noise at 1m (σ scales with √range)
  dropout_prob     float   0.02      probability of a ray returning no echo (0–1)
  specular_thresh  float   0.1       range jumps larger than this between adjacent
                                      bearings are flagged as possible specular (not filtered,
                                      just sets intensity low — your SLAM can decide)
"""

import math
import random
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.clock import Clock

from std_msgs.msg import Header
from sensor_msgs.msg import Range, LaserScan
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from builtin_interfaces.msg import Duration

# Custom message — built as part of this package
from mps_sim.msg import MpsSector


class MPSDriver(Node):

    def __init__(self):
        super().__init__('mps_driver')

        # ── Declare + fetch parameters ───────────────────────────────────
        # Defaults match project spec: 360° sweep, 30m range, ~5s/sweep
        # (200 steps * 0.025s = 5.0s).
        self.declare_parameter('n_steps',          200)
        self.declare_parameter('step_period_sec',  0.025)
        self.declare_parameter('max_range',        30.0)
        self.declare_parameter('min_range',        0.3)
        self.declare_parameter('noise_sigma_base', 0.01)
        self.declare_parameter('dropout_prob',     0.02)
        self.declare_parameter('specular_thresh',  0.1)

        self.n_steps          = self.get_parameter('n_steps').value
        self.step_period      = self.get_parameter('step_period_sec').value
        self.max_range        = self.get_parameter('max_range').value
        self.min_range        = self.get_parameter('min_range').value
        self.sigma_base       = self.get_parameter('noise_sigma_base').value
        self.dropout_prob     = self.get_parameter('dropout_prob').value
        self.specular_thresh  = self.get_parameter('specular_thresh').value

        # Derived geometry
        self.angle_step  = (2.0 * math.pi) / self.n_steps   # radians per step
        self.angles      = [
            -math.pi + i * self.angle_step for i in range(self.n_steps)
        ]

        # ── State ────────────────────────────────────────────────────────
        self.current_step     = 0
        self.current_bearing  = self.angles[0]
        self.sweep_ranges     = [float('nan')] * self.n_steps
        self.sweep_valid      = [False] * self.n_steps
        self.sweep_start_time = None
        self.last_range       = None   # most recent raw range from Gazebo
        self.last_range_time  = None
        self.prev_range       = None   # previous step's range, for specular detection

        # ── Publishers ───────────────────────────────────────────────────
        self.sector_pub = self.create_publisher(MpsSector,  '/mps/sector', 50)
        self.scan_pub   = self.create_publisher(LaserScan,  '/mps/scan',   10)

        # Joint controller — ros_control expects JointTrajectory on this topic
        self.joint_pub = self.create_publisher(
            JointTrajectory,
            '/set_joint_trajectory',
            10
        )

        # ── Subscriber ───────────────────────────────────────────────────
        self.range_sub = self.create_subscription(
            Range,
            '/mps/raw_range',
            self._on_raw_range,
            10
        )

        # ── Step timer ───────────────────────────────────────────────────
        self.step_timer = self.create_timer(self.step_period, self._step)

        self.get_logger().info(
            f'MPS driver started: {self.n_steps} steps/sweep, '
            f'step period {self.step_period:.3f}s, '
            f'max range {self.max_range}m'
        )

    # ────────────────────────────────────────────────────────────────────
    # Internal helpers
    # ────────────────────────────────────────────────────────────────────

    def _command_joint(self, angle_rad: float):
        """Send a position command to the rotating head joint."""
        traj = JointTrajectory()
        traj.header.stamp = self.get_clock().now().to_msg()
        traj.joint_names = ['mps_rotate_joint']

        pt = JointTrajectoryPoint()
        pt.positions = [angle_rad]
        pt.velocities = [0.0]
        pt.time_from_start = Duration(sec=0, nanosec=int(self.step_period * 0.5 * 1e9))

        traj.points = [pt]
        self.joint_pub.publish(traj)

    def _apply_noise_model(self, raw_range: float) -> tuple[float, float, bool]:
        """
        Apply the sonar noise model to a raw Gazebo range reading.

        Returns (noisy_range, intensity, valid)

        Noise model:
        ─────────────
        1. Dropout: with probability `dropout_prob`, the beam returns nothing.
           Real sonars lose returns on steep angles, soft sediment, etc.

        2. Range-dependent Gaussian noise:
           σ(r) = σ_base × √(r / 1.0)
           Acoustic travel-time uncertainty grows with range because of
           absorption, multipath, and timing jitter.

        3. Intensity model:
           I(r) = 1 / (r² + 1)   (inverse square, clamped to [0,1])
           This is a gross simplification; real intensity depends on
           target strength and beam angle, but it gives qualitatively
           correct falloff.

        4. Max-range clipping:
           If the Gazebo ray hit nothing it returns max_range exactly.
           We treat this as no-return (valid=False).
        """
        # No-return if Gazebo returned max_range (nothing in beam)
        if raw_range >= self.max_range - 0.01:
            return self.max_range, 0.0, False

        # Dropout
        if random.random() < self.dropout_prob:
            return self.max_range, 0.0, False

        # Range-dependent Gaussian noise
        sigma = self.sigma_base * math.sqrt(max(raw_range, 0.1))
        noisy = raw_range + random.gauss(0.0, sigma)
        noisy = max(self.min_range, min(self.max_range, noisy))

        # Intensity (inverse square law)
        intensity = float(1.0 / (noisy ** 2 + 1.0))
        intensity = min(1.0, max(0.0, intensity))

        return noisy, intensity, True

    def _check_specular(self, current_range: float) -> float:
        """
        Detect possible specular reflection artefacts.
        If the range jump from the previous bearing exceeds specular_thresh,
        return a reduced intensity weight (caller can use this as a flag).
        This does NOT discard the measurement — that is your SLAM's job.
        """
        if self.prev_range is None or not math.isfinite(self.prev_range):
            return 1.0
        jump = abs(current_range - self.prev_range)
        if jump > self.specular_thresh:
            return 0.3   # low confidence
        return 1.0

    # ────────────────────────────────────────────────────────────────────
    # Callbacks
    # ────────────────────────────────────────────────────────────────────

    def _on_raw_range(self, msg: Range):
        """Cache the most recent raw range reading from Gazebo."""
        self.last_range      = msg.range
        self.last_range_time = self.get_clock().now()

    def _step(self):
        """
        Called once per step_period. This is the heartbeat of the MPS.

        Sequence per call:
          1. Command joint to current bearing.
          2. Read cached raw range (best available; non-blocking).
          3. Apply noise model.
          4. Publish MpsSector with *this moment's* timestamp.
          5. Advance bearing. If full sweep complete, publish LaserScan.
        """
        now = self.get_clock().now()

        # ── 1. Move the joint ───────────────────────────────────────────
        self._command_joint(self.current_bearing)

        # ── 2. Read range ───────────────────────────────────────────────
        # Use the most recently received raw range. In sim this is fine;
        # on hardware you'd wait for an echo gate here.
        if self.last_range is None:
            # Sensor not yet publishing — skip this step
            return

        raw = self.last_range

        # ── 3. Noise model ──────────────────────────────────────────────
        noisy_range, intensity, valid = self._apply_noise_model(raw)

        # Specular confidence modifier
        if valid:
            conf = self._check_specular(noisy_range)
            intensity *= conf
            self.prev_range = noisy_range
        else:
            self.prev_range = None

        # ── 4. Publish MpsSector ────────────────────────────────────────
        sector = MpsSector()
        sector.header.stamp    = now.to_msg()        # ← THIS is the key line.
        sector.header.frame_id = 'mps_head_link'     #   Each bearing gets its own time.
        sector.bearing_rad     = float(self.current_bearing)
        sector.bearing_deg     = float(math.degrees(self.current_bearing))
        sector.range           = float(noisy_range)
        sector.intensity       = float(intensity)
        sector.valid           = valid
        self.sector_pub.publish(sector)

        # Store for LaserScan assembly
        self.sweep_ranges[self.current_step] = noisy_range if valid else float('nan')
        self.sweep_valid[self.current_step]  = valid

        # ── 5. Advance bearing ──────────────────────────────────────────
        self.current_step += 1

        if self.current_step >= self.n_steps:
            # Full sweep complete — publish assembled LaserScan
            self._publish_scan(now)
            # Reset for next sweep
            self.current_step    = 0
            self.sweep_ranges    = [float('nan')] * self.n_steps
            self.sweep_valid     = [False] * self.n_steps
            self.sweep_start_time = None

        self.current_bearing = self.angles[self.current_step]

        if self.sweep_start_time is None:
            self.sweep_start_time = now

    def _publish_scan(self, now):
        """
        Assemble and publish a sensor_msgs/LaserScan from the completed sweep.

        Note: the LaserScan stamp is set to the END of the sweep (now).
        The sweep_start_time is available if your consumer needs it.
        The individual sector messages already carry per-bearing stamps —
        prefer those for motion-compensated processing.

        The LaserScan is provided for convenience (e.g. rviz2 visualisation,
        nav2 costmap) but does NOT carry sweep-distortion information.
        """
        scan = LaserScan()
        scan.header.stamp    = now.to_msg()
        scan.header.frame_id = 'base_link'

        scan.angle_min       = -math.pi
        scan.angle_max       =  math.pi - self.angle_step
        scan.angle_increment = self.angle_step
        scan.time_increment  = float(self.step_period)   # time between bearings
        scan.scan_time       = float(self.n_steps * self.step_period)
        scan.range_min       = float(self.min_range)
        scan.range_max       = float(self.max_range)

        # Replace invalid returns with inf (LaserScan convention)
        scan.ranges = [
            r if (v and math.isfinite(r)) else float('inf')
            for r, v in zip(self.sweep_ranges, self.sweep_valid)
        ]

        # Intensities (0.0–1.0 scaled to arbitrary units for rviz)
        # We don't store per-step intensity above; fill with zeros here.
        # If you need intensity in the LaserScan, extend the sweep buffer.
        scan.intensities = []

        self.scan_pub.publish(scan)

        valid_count = sum(self.sweep_valid)
        self.get_logger().debug(
            f'Sweep complete: {valid_count}/{self.n_steps} valid returns'
        )


def main(args=None):
    rclpy.init(args=args)
    node = MPSDriver()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

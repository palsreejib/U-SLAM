#!/usr/bin/env python3
"""
sync_node.py — MPS + FLS sensor fusion / synchronization layer.

THE CORE DESIGN PROBLEM THIS SOLVES
────────────────────────────────────
The MPS takes ~5s to complete one 360° sweep (200 steps at 25ms/step).
The FLS delivers a full 2D frame at 15Hz (every ~67ms). At any given FLS
frame, only a ~4° sliver of a fresh MPS sweep exists
(step_period / sweep_period * 360° ≈ 4°/frame). A complete MPS sweep is
NEVER available at the FLS frame rate — waiting for one would mean
gating your transformer's input on an event that happens roughly once
every 75 FLS frames, throwing away 74 of every 75 FLS frames in the
process. That is the mistake this node exists to avoid.

THE FIX: a rolling buffer, not a completion gate
──────────────────────────────────────────────────
This node keeps the last N MPS sectors (default N=36 → 180° of angular
coverage, since 36 steps * 1.8°/step = 64.8°... see NOTE below) in a
deque. On every FLS frame arrival — and ONLY on FLS frame arrival, never
on MPS sweep completion — it snapshots whatever is currently in that
buffer, finds the nearest-in-time reading from IMU/pressure/DVL/ground
truth, and publishes all of it together as one SyncedSample. Early in a
run, or right after a large yaw turn, the buffer may hold fewer than N
sectors, or sectors whose bearings don't evenly cover 180° — that's
expected and is exactly why mps_sector_count and each sector's own
bearing/stamp are carried through rather than assumed.

NOTE on N=36: with this package's default MPS config (200 steps/360°
sweep = 1.8°/step), 36 sectors covers 36 * 1.8° = 64.8° of arc, not
180°. To get genuine 180° coverage as the project spec describes, you
need N = 100 (100 * 1.8° = 180°) OR change the MPS to 100 steps/sweep
(100 * 3.6° = 180° in 36 steps). mps_buffer_size defaults to 36 to match
the spec's literal number, but READ THIS: check the angular math against
your actual mps_driver n_steps setting and adjust mps_buffer_size to hit
the coverage you actually want — the two are only consistent for a
specific n_steps value, and the code will not warn you if you change one
without the other. See README.

WHAT THIS NODE IS NOT
──────────────────────
This is data plumbing, not state estimation. It does not fuse sensors
into a pose estimate — no GTSAM, no filtering. It guarantees a
downstream consumer (transformer, GTSAM factor graph, whatever comes
next) receives all modalities time-aligned and with honest skew
reporting, so mis-synced training data is a visible, measured quantity
rather than a silent bug.
"""

import math
from collections import deque

import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image, Imu, FluidPressure
from nav_msgs.msg import Odometry

from mps_sim.msg import MpsSector, DvlReading, SyncedSample


def stamp_to_sec(stamp) -> float:
    return stamp.sec + stamp.nanosec * 1e-9


class TimedBuffer:
    """Small deque of (stamp_sec, msg) pairs with nearest-in-time lookup."""

    def __init__(self, maxlen=50):
        self._buf = deque(maxlen=maxlen)

    def push(self, stamp_sec: float, msg):
        self._buf.append((stamp_sec, msg))

    def nearest(self, target_sec: float):
        """Returns (msg, skew_sec) for the closest entry, or (None, inf)."""
        if not self._buf:
            return None, float('inf')
        best_msg, best_skew = None, float('inf')
        for t, m in self._buf:
            skew = abs(t - target_sec)
            if skew < best_skew:
                best_skew, best_msg = skew, m
        return best_msg, best_skew


class SyncNode(Node):

    def __init__(self):
        super().__init__('sync_node')

        self.declare_parameter('mps_buffer_size', 36)
        self.declare_parameter('sync_warn_threshold_sec', 0.15)

        self.mps_buffer_size = self.get_parameter('mps_buffer_size').value
        self.warn_threshold  = self.get_parameter('sync_warn_threshold_sec').value

        self.mps_buffer = deque(maxlen=self.mps_buffer_size)
        self.imu_buf     = TimedBuffer()
        self.pressure_buf = TimedBuffer()
        self.dvl_buf     = TimedBuffer()
        self.gt_buf      = TimedBuffer()

        # ── Subscriptions ────────────────────────────────────────────
        self.create_subscription(MpsSector, '/mps/sector', self._on_mps, 50)
        self.create_subscription(Image, '/fls/image', self._on_fls, 10)
        self.create_subscription(Imu, '/imu/data', self._on_imu, 50)
        self.create_subscription(FluidPressure, '/pressure/data', self._on_pressure, 20)
        self.create_subscription(DvlReading, '/dvl/data', self._on_dvl, 10)
        self.create_subscription(Odometry, '/ground_truth/odometry', self._on_gt, 50)

        self.pub = self.create_publisher(SyncedSample, '/training/sample', 10)

        self._sample_count = 0
        self._warn_count = 0

        self.get_logger().info(
            f'Sync node started. mps_buffer_size={self.mps_buffer_size}, '
            f'sync_warn_threshold_sec={self.warn_threshold}. '
            f'Publishing /training/sample, gated on /fls/image arrival.'
        )

    # ── Buffer-filling callbacks (cheap; just append) ──────────────────

    def _on_mps(self, msg: MpsSector):
        self.mps_buffer.append(msg)

    def _on_imu(self, msg: Imu):
        self.imu_buf.push(stamp_to_sec(msg.header.stamp), msg)

    def _on_pressure(self, msg: FluidPressure):
        self.pressure_buf.push(stamp_to_sec(msg.header.stamp), msg)

    def _on_dvl(self, msg: DvlReading):
        self.dvl_buf.push(stamp_to_sec(msg.header.stamp), msg)

    def _on_gt(self, msg: Odometry):
        self.gt_buf.push(stamp_to_sec(msg.header.stamp), msg)

    # ── The gating event: FLS frame arrival ─────────────────────────────

    def _on_fls(self, fls_msg: Image):
        ref_sec = stamp_to_sec(fls_msg.header.stamp)
        # Fall back to node clock if the plugin ever publishes a zero
        # stamp (shouldn't happen once sim time is flowing, but this
        # keeps the node from producing a nonsense skew against t=0).
        if ref_sec <= 0.0:
            ref_sec = self.get_clock().now().nanoseconds * 1e-9

        imu_msg, imu_skew = self.imu_buf.nearest(ref_sec)
        pressure_msg, pressure_skew = self.pressure_buf.nearest(ref_sec)
        dvl_msg, dvl_skew = self.dvl_buf.nearest(ref_sec)
        gt_msg, gt_skew = self.gt_buf.nearest(ref_sec)

        # Don't publish a sample missing entire modalities — better to
        # wait one frame than hand downstream code a silently-empty field.
        if None in (imu_msg, pressure_msg, dvl_msg, gt_msg):
            return

        sectors = list(self.mps_buffer)
        oldest_age = (
            ref_sec - stamp_to_sec(sectors[0].header.stamp)
            if sectors else 0.0
        )

        sample = SyncedSample()
        sample.header = fls_msg.header
        sample.mps_sectors = sectors
        sample.mps_sector_count = len(sectors)
        sample.fls_image = fls_msg
        sample.imu = imu_msg
        sample.pressure = pressure_msg
        sample.dvl = dvl_msg
        sample.ground_truth = gt_msg

        sample.imu_skew_sec = float(imu_skew)
        sample.pressure_skew_sec = float(pressure_skew)
        sample.dvl_skew_sec = float(dvl_skew)
        sample.ground_truth_skew_sec = float(gt_skew)
        sample.oldest_mps_sector_age_sec = float(oldest_age)

        worst_skew = max(imu_skew, pressure_skew, dvl_skew, gt_skew)
        sample.sync_ok = bool(worst_skew <= self.warn_threshold)

        self.pub.publish(sample)
        self._sample_count += 1

        if not sample.sync_ok:
            self._warn_count += 1
            self.get_logger().warn(
                f'Sync skew above threshold ({self.warn_threshold}s): '
                f'imu={imu_skew:.3f}s pressure={pressure_skew:.3f}s '
                f'dvl={dvl_skew:.3f}s gt={gt_skew:.3f}s '
                f'(sample #{self._sample_count}, '
                f'{self._warn_count} warnings total)'
            )


def main(args=None):
    rclpy.init(args=args)
    node = SyncNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

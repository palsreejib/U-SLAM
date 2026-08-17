# SyncedSample.msg
#
# One synchronized multi-modal training tuple, emitted at the FLS frame
# rate (15Hz). This is the "sensor fusion" layer for MPS + FLS: it
# packages the rolling MPS sector buffer (partial-sweep aware, per the
# project's caveat about MPS taking ~5s/sweep vs FLS's 15Hz) together
# with the FLS frame and all other sensors, each resolved to its
# nearest-in-time reading relative to this sample's reference stamp
# (the FLS frame's own stamp — FLS arrival is what gates a sample,
# never MPS sweep completion).
#
# This message does NOT perform state estimation — no GTSAM factors,
# no pose fusion, no filtering. It only guarantees that everything a
# downstream SLAM/learning consumer needs is bundled together with
# honest, checked timestamp bookkeeping. Probabilistic fusion (the
# "Multi-modal GTSAM fusion" project task) is a separate, later stage
# that consumes this message's output.

std_msgs/Header header                  # stamp = FLS frame stamp (this sample's reference time)

# Rolling buffer of the last N MPS sectors (N=36 default = 180° coverage
# at 200 steps/sweep, ~4.5s of sweep history at the 5s/sweep spec).
# Each sector carries ITS OWN stamp + bearing — this is NOT a
# synchronous scan, do not treat it as one.
#
# The buffer capacity (N) is enforced by sync_node.py's internal deque
# (mps_buffer_size parameter, default 36), not by this message type.
# mps_sector_count is simply len(mps_sectors) at publish time — fewer
# than the configured capacity right after startup, before the buffer
# has filled once.
mps_sim/MpsSector[] mps_sectors
uint16 mps_sector_count

# FLS frame — native output of the FLS Gazebo plugin (encoding: bgr8)
sensor_msgs/Image fls_image

# Nearest-in-time auxiliary sensor readings
sensor_msgs/Imu imu
sensor_msgs/FluidPressure pressure
mps_sim/DvlReading dvl
nav_msgs/Odometry ground_truth

# ── Timestamp alignment diagnostics (seconds) ───────────────────────
# "Check timestamp alignment across FLS, MPS, IMU, ground truth —
# misalignment corrupts training" — this is that check, computed for
# every sample instead of discovered after the fact during training.
float32 imu_skew_sec             # |imu.header.stamp - header.stamp|
float32 pressure_skew_sec        # |pressure.header.stamp - header.stamp|
float32 dvl_skew_sec             # |dvl.header.stamp - header.stamp|
float32 ground_truth_skew_sec    # |ground_truth.header.stamp - header.stamp|
float32 oldest_mps_sector_age_sec  # header.stamp - mps_sectors[0].header.stamp

bool    sync_ok                  # false if ANY skew above exceeds the
                                  # node's sync_warn_threshold_sec param

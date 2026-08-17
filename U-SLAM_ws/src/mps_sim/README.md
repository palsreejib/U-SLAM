# mps_sim — MPS + FLS sensor fusion for Gazebo Classic 11 / ROS 2 Humble

Simulates a Mechanical Profiling Sonar (rotating single-beam, per-bearing
timestamped — sweep distortion is real and present in the data) fused
with a Forward-Looking Sonar (the real Dave-ecosystem plugin) into
synchronized multi-modal training tuples.

**Read the next section before building anything.** Three things in the
original project plan don't match reality, and it's much cheaper to
know that now than after an hour of failed builds.

---

## ⚠ Critical findings — read first

### 1. There is no "ROS 2 Humble branch" of `Field-Robotics-Lab/dave`

The `dave` repo has exactly two branches that matter, and neither is
Humble + Gazebo Classic:

| Branch | ROS | Gazebo |
|---|---|---|
| `master` | ROS 1 Noetic | Classic 11 |
| `ros2` | ROS 2 **Jazzy** | **Harmonic** (not Classic) |

Since this project is committed to Humble + Gazebo Classic 11 (matching
the MPS package already built), pulling the full `dave` stack doesn't
get you anything runnable. **This package does not depend on `dave`
directly at all.**

### 2. The actual usable FLS plugin is a different, smaller repo

The real, working ROS 2 + Gazebo Classic 11 FLS plugin is:

```
forssea-robotics/nps_uw_multibeam_sonar
```

This is a ROS 2 port (built/tested against Galactic, but written
against generic `rclcpp`/`gazebo_ros`/`gazebo_plugins` APIs with no
Galactic-specific code — it has a real chance of building cleanly on
Humble; see install steps below) of `Field-Robotics-Lab/nps_uw_multibeam_sonar`.
The `nps_uw_sensors_gazebo` repo named in the original plan is real too,
but it's ROS 1/Noetic-era (DVL, underwater lidar) with no confirmed
ROS 2 port — see the DVL section below for how this package routes
around that instead of gambling the build on it.

### 3. This FLS plugin requires an NVIDIA GPU + CUDA. Hard requirement.

Verified directly in the plugin's `CMakeLists.txt`:
`find_package(CUDA REQUIRED)`. Not optional, not a runtime-only
dependency — **the plugin will not build at all without CUDA**, on
either the raster or ray-based sonar variant. It also needs
`ros-humble-velodyne-simulator`-family packages for the ray variant, and
GPU memory (the README for the plugin specifies ≥4GB).

**If you don't have an NVIDIA GPU on this machine, stop here and tell
me — I'll build a CPU-only custom FLS simulator (same design pattern as
the MPS: a Gazebo depth/camera sensor + a Python node doing the
polar-image rendering ourselves) instead of the real plugin.** Everything
else in this package (MPS, ground truth, DVL, pressure, the sync node)
runs with zero GPU dependency and works either way.

### 4. Good news: the plugin already publishes `sensor_msgs/Image` natively

I pulled the actual plugin source (not just docs/wiki pages) to verify
this rather than guessing. `gazebo_multibeam_sonar_raster_based.cpp`
publishes **two** topics:

- `sonarImageRawTopicName` → `acoustic_msgs/msg/SonarImage` (full-precision
  polar data: per-beam ranges, azimuth angles, intensities — see
  "Later: raw polar data" below if you want this instead)
- `sonarImageTopicName` → **`sensor_msgs/msg/Image`, encoding `bgr8`**

The second one is exactly the project spec ("outputs 2D acoustic
intensity image as sensor_msgs/Image"), already true out of the box —
**no bridge/converter node was needed**, which simplified this package
versus my first draft plan. `mps_vehicle.urdf.xacro` sets
`sonarImageTopicName` to the absolute path `/fls/image`, so it publishes
exactly where `sync_node.py` expects it regardless of any Gazebo node
namespacing.

---

## Architecture

```
Gazebo Classic 11 (mps_world1.world)
  ├─ libgazebo_ros_state.so (world plugin) → /gazebo/model_states
  │
  └─ mps_vehicle (URDF)
       ├─ base_link
       ├─ mps_head_link (continuous joint, rotates per MPS step)
       │    └─ ray sensor → /mps/raw_range
       ├─ fls_link
       │    └─ depth-camera sensor + libnps_multibeam_sonar_ros_plugin.so
       │         → /fls/image            (sensor_msgs/Image, bgr8)   ← used by sync_node
       │         → /fls/sonar_image_raw  (acoustic_msgs/SonarImage)  ← available, unused for now
       ├─ imu_link
       │    └─ libgazebo_ros_imu_sensor.so → /imu/data
       ├─ dvl_link   (geometry only — see dvl_node.py)
       └─ pressure_link (geometry only — see pressure_node.py)

ros2_control → JointTrajectoryController → mps_rotate_joint

mps_driver_node   ──▶ /mps/sector  (MpsSector, one per bearing, own stamp)
ground_truth_node ──▶ /ground_truth/odometry  (from /gazebo/model_states)
dvl_node           ──▶ /dvl/data    (DvlReading, derived from ground truth)
pressure_node      ──▶ /pressure/data (sensor_msgs/FluidPressure, derived from ground truth)

sync_node:
  rolling deque of last N /mps/sector messages
  gated on /fls/image arrival (NEVER on MPS sweep completion) ──▶ /training/sample (SyncedSample)
```

---

## Install

### 1. System packages

```bash
sudo apt update
sudo apt install -y \
  ros-humble-desktop \
  ros-humble-gazebo-ros-pkgs \
  ros-humble-gazebo-ros2-control \
  ros-humble-ros2-control \
  ros-humble-ros2-controllers \
  ros-humble-joint-state-broadcaster \
  ros-humble-joint-trajectory-controller \
  ros-humble-robot-state-publisher \
  ros-humble-xacro \
  ros-humble-cv-bridge \
  ros-humble-velodyne-gazebo-plugins
```

If `ros-humble-velodyne-gazebo-plugins` doesn't resolve, you'll need to
build `velodyne_simulator` from source too — only actually required if
you switch the FLS to the ray-based (`gpu_ray`) variant instead of the
raster/depth-camera variant this package defaults to.

### 2. CUDA (only if you have an NVIDIA GPU — see finding #3 above)

Follow NVIDIA's install guide for your distro. Verify with `nvidia-smi`
and `nvcc --version` before proceeding — both must work.

### 3. Message package: `acoustic_msgs` (from `hydrographic_msgs`)

This is a small, dependency-free message package (`std_msgs` +
`geometry_msgs` only) — builds fine on Humble.

```bash
cd ~/mps_ws/src
git clone https://github.com/forssea-robotics/hydrographic_msgs.git
```

### 4. The FLS plugin itself

```bash
cd ~/mps_ws/src
git clone https://github.com/forssea-robotics/nps_uw_multibeam_sonar.git
```

### 5. This package

```bash
cd ~/mps_ws/src
# copy the extracted mps_sim/ folder here
```

### 6. Build everything together

```bash
cd ~/mps_ws
source /opt/ros/humble/setup.bash
rosdep install --from-paths src --ignore-src -r -y
colcon build --packages-select acoustic_msgs nps_uw_multibeam_sonar mps_sim
source install/setup.bash
```

Build `acoustic_msgs` and `nps_uw_multibeam_sonar` **before** or
**together with** `mps_sim` — `mps_sim` doesn't directly depend on
either at the package.xml level (it only ever subscribes to the plain
`sensor_msgs/Image` the plugin publishes), but Gazebo needs
`libnps_multibeam_sonar_ros_plugin.so` on its plugin path at launch
time, which only exists after that package builds successfully.

**If `nps_uw_multibeam_sonar` fails to build**, the single most likely
cause is CUDA (missing, wrong version, or no GPU). Check that first.
The second most likely cause is a genuine Galactic→Humble API
incompatibility somewhere in `rclcpp`/`gazebo_ros` — if you hit one,
paste me the exact compiler error and we'll patch it; I can't
pre-verify a full Humble compile from here without the toolchain, so
treat this as the most likely place we'll need a debug pass together.

---

## Launch

```bash
ros2 launch mps_sim mps_sim.launch.py
```

Optional arguments (defaults shown):

```bash
ros2 launch mps_sim mps_sim.launch.py \
  world:=mps_world1.world \
  n_steps:=200 step_period_sec:=0.025 max_range:=30.0 \
  fls_horizontal_fov_deg:=120.0 fls_max_range:=10.0 fls_update_rate:=15.0 \
  mps_buffer_size:=36
```

`world:=mps_test.world` gets you back the original simple box world if
`mps_world1.world`'s obstacle count is too heavy while debugging.

---

## Topics

| Topic | Type | Rate | Source |
|---|---|---|---|
| `/mps/sector` | `mps_sim/MpsSector` | ~40Hz (1/step_period) | mps_driver_node |
| `/mps/scan` | `sensor_msgs/LaserScan` | ~0.2Hz (1/sweep) | mps_driver_node, RViz convenience only |
| `/fls/image` | `sensor_msgs/Image` (bgr8) | 15Hz | Gazebo FLS plugin |
| `/fls/sonar_image_raw` | `acoustic_msgs/SonarImage` | 15Hz | Gazebo FLS plugin (unused by sync_node currently) |
| `/imu/data` | `sensor_msgs/Imu` | 100Hz | Gazebo IMU plugin |
| `/ground_truth/odometry` | `nav_msgs/Odometry` | 50Hz | ground_truth_node |
| `/dvl/data` | `mps_sim/DvlReading` | 7Hz | dvl_node |
| `/pressure/data` | `sensor_msgs/FluidPressure` | 20Hz | pressure_node |
| `/training/sample` | `mps_sim/SyncedSample` | 15Hz (= FLS rate) | sync_node |

---

## Verifying it's working

```bash
source ~/mps_ws/install/setup.bash

# All expected rates in one place
ros2 topic hz /mps/sector
ros2 topic hz /fls/image
ros2 topic hz /ground_truth/odometry
ros2 topic hz /training/sample

# In RViz2: add an Image display on /fls/image, and an Odometry
# display on /ground_truth/odometry, to see both sonars + ground
# truth moving simultaneously.
rviz2
```

If `/fls/image` never appears: check the Gazebo terminal output for a
plugin-load error naming `fls_sensor` or
`libnps_multibeam_sonar_ros_plugin.so` — that means the plugin didn't
build/isn't on the plugin path, not a bug in this package's URDF wiring.

---

## The sensor fusion layer (`sync_node.py`) — read this carefully

**The core problem:** MPS takes ~5s for a full 360° sweep; FLS delivers
a frame every ~67ms (15Hz). At any given FLS frame, only a few degrees
of a fresh MPS sweep exist. A complete MPS sweep is never available at
the FLS rate — gating on one would mean discarding the overwhelming
majority of FLS frames while waiting.

**The fix:** `sync_node.py` keeps a rolling deque of the last
`mps_buffer_size` (default 36) `/mps/sector` messages. On every
`/fls/image` arrival — and **only** on FLS arrival, never on MPS sweep
completion — it snapshots whatever's currently in that buffer, finds
the nearest-in-time IMU/pressure/DVL/ground-truth reading, and publishes
everything together as one `SyncedSample` on `/training/sample`.

**⚠ N=36 vs "180° coverage" — check this against your own numbers.**
The project caveat states N=36 gives 180° coverage. With this package's
default MPS config (200 steps / 360° sweep = 1.8°/step),
36 sectors covers 36 × 1.8° = **64.8°**, not 180°. To get literal 180°
coverage you need N=100 (100 × 1.8°=180°) — either raise
`mps_buffer_size` to 100, or change `n_steps` to 100 (100 steps × 3.6°/step,
then N=50 gives 180°). I did not silently pick one of these for you:
the caveat's own numbers ("~4° sector every 14ms" vs "~5s per 360°
sweep") aren't internally consistent with each other either
(4°-per-step implies ~90 steps/sweep → ~55ms/step at 5s/sweep, not
14ms), so there's no single unambiguous target to snap to. Pick the
n_steps/mps_buffer_size pair that matches what you actually want and
I'll help wire it through — this is exactly the kind of thing worth a
quick confirmation before it quietly ships into training data.

**Each `SyncedSample` carries its own sync diagnostics** so misalignment
is a measured field, not a silent assumption:

```python
sample.imu_skew_sec            # |imu.stamp - sample.stamp|
sample.pressure_skew_sec
sample.dvl_skew_sec
sample.ground_truth_skew_sec
sample.oldest_mps_sector_age_sec   # how stale the oldest buffered sector is
sample.sync_ok                     # False if any skew > sync_warn_threshold_sec (default 0.15s)
```

`sync_node.py` also logs a `ROS_WARN` every time `sync_ok` is `False`,
so skew problems show up in your terminal during a run, not just in
post-hoc inspection of a bag.

**What this node deliberately does NOT do:** no GTSAM factors, no pose
fusion, no filtering. It's data plumbing — it guarantees your downstream
consumer (transformer, or later, the GTSAM factor graph from the
"Multi-modal GTSAM fusion" project stage) gets clean, time-aligned,
skew-annotated multi-modal tuples. State estimation is a separate,
later problem that consumes this node's output.

---

## Why DVL and pressure are custom nodes, not Dave plugins

The project plan says "DVL plugin if available in Dave." I looked: Dave's
real DVL implementation (`whoi_teledyne_whn` and similar) lives in
`Field-Robotics-Lab/nps_uw_sensors_gazebo`, which is ROS 1/Noetic-era
code with no confirmed ROS 2 port. Rather than betting the whole build
on porting a second unverified plugin (on top of the FLS one), `dvl_node.py`
and `pressure_node.py` derive physically-reasonable readings directly
from `/ground_truth/odometry`:

- **DVL** (`dvl_node.py`): rotates world-frame ground-truth velocity into
  the body frame, adds noise shaped like real DVL specs (percent-of-
  reading + fixed floor), estimates altitude assuming a flat seafloor at
  `z=0`, and simulates bottom-lock dropout beyond `max_lock_range`.
- **Pressure** (`pressure_node.py`): standard hydrostatic formula
  (`P = P_atm + ρ·g·depth`) against a `water_surface_z` parameter, since
  Gazebo Classic's plain sensors don't model an actual water volume.

**IMU is the real thing** — `libgazebo_ros_imu_sensor.so` is a stock ROS 2
Humble + Gazebo Classic plugin (ships with `gazebo_plugins`), zero risk,
genuinely simulating IMU noise via Gazebo's own sensor model.

If you specifically need the real WHN-family DVL plugin later (e.g. for
water-tracking mode, which the ground-truth-derived version can't
simulate since there's no simulated water column), say so and I'll
attempt the `nps_uw_sensors_gazebo` ROS 2 port as a separate task —
I didn't want to gamble this delivery on a second unverified porting
effort on top of the FLS one.

---

## Later: switching to raw polar FLS data

`/fls/sonar_image_raw` (`acoustic_msgs/SonarImage`) is already being
published by the plugin — it's just not consumed anywhere yet. Verified
real fields (pulled directly from `hydrographic_msgs/acoustic_msgs/msg/SonarImage.msg`):

```
std_msgs/Header header
float32 frequency
float32 sound_speed
float32 azimuth_beamwidth
float32 elevation_beamwidth
float32[] azimuth_angles
float32[] elevation_angles
float32[] ranges
bool      is_bigendian
uint8     data_size
uint8[]   intensities   # azimuth-major: len = len(ranges)*len(azimuth_angles)*data_size
```

This is full-precision polar data (vs. the already-rendered `bgr8`
image), which may be worth switching to once you're training the actual
transformer — say the word and I'll wire `sync_node.py` to consume this
instead/as well.

---

## Tuning reference

| Parameter | Where | Default | Notes |
|---|---|---|---|
| MPS `n_steps` | launch arg | 200 | steps/360° sweep |
| MPS `step_period_sec` | launch arg | 0.025 | 200×0.025=5.0s/sweep |
| MPS `max_range` | launch arg | 30.0m | |
| FLS `fls_horizontal_fov_deg` | launch arg | 120.0° | |
| FLS `fls_max_range` | launch arg | 10.0m | |
| FLS `fls_update_rate` | launch arg | 15.0Hz | |
| sync `mps_buffer_size` | launch arg | 36 | see coverage-math warning above |
| sync `sync_warn_threshold_sec` | node param | 0.15s | |
| DVL `publish_rate_hz` | node param | 7.0Hz | typical WHN-class rate |
| DVL `noise_percent` / `noise_floor` | node param | 0.5% / 2mm/s | |
| Pressure `publish_rate_hz` | node param | 20.0Hz | |
| Pressure `water_surface_z` | node param | 0.0m | must match your world's vehicle operating height |

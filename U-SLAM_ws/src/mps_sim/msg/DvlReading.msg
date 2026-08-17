# DvlReading.msg
#
# Custom lightweight DVL reading, derived from Gazebo ground-truth
# velocity — NOT from a Dave-native DVL plugin. See README for why:
# short version is that Dave's real DVL plugin lives in the ROS1/
# Noetic-era nps_uw_sensors_gazebo repo with no confirmed ROS 2 port,
# so dvl_node.py derives an equally useful signal from ground truth
# instead of gambling the whole build on an unverified dependency.
#
# Body-frame velocity is computed by rotating the world-frame ground
# truth twist into the vehicle's current orientation, then adding
# realistic percentage-of-reading + fixed-floor noise, matching how
# real DVL specs are usually written (e.g. "0.2% of reading + 1mm/s").

std_msgs/Header header

geometry_msgs/Vector3 velocity   # body-frame velocity estimate (m/s)
float64 altitude                 # estimated altitude above seafloor (m)
                                  # NOTE: assumes a flat seafloor at z=0 —
                                  # see pressure_node.py / README for the
                                  # same assumption applied to depth.
bool    velocity_valid           # false on simulated bottom-lock dropout
                                  # (altitude beyond max_lock_range)
float32 noise_percent            # noise-percent-of-reading actually applied,
                                  # kept for reference/debugging

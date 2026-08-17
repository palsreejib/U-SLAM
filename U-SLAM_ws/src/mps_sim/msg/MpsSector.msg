# MpsSector.msg
# One bearing measurement from the Mechanical Profiling Sonar.
#
# The driver publishes one of these per simulation step — i.e. once per
# acoustic pulse. Each message has its own stamp, which is the simulation
# time at which that particular ray was fired. This means a full 360°
# scan is a time-series of 200 (or N_STEPS) individually-timestamped
# messages. Your SLAM front-end can use these stamps together with the
# vehicle odometry to undo sweep distortion.

std_msgs/Header header       # stamp = sim time this ray was fired
                             # frame_id = "mps_head_link"

float32 bearing_rad          # current bearing of the transducer (radians, 0 = forward)
float32 bearing_deg          # same in degrees for convenience

float32 range                # measured range (metres), after noise model applied
float32 intensity            # simulated echo intensity [0.0 – 1.0]
                             # proportional to 1/r^2 with dropout

bool    valid                # false if ray returned max_range (no return) or dropout

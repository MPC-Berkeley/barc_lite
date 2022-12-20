# This is a message to hold data with system state information
std_msgs/Header header
float64                                  t       # Time

mpclab_common/PositionMsg                x

mpclab_common/OrientationEulerMsg        e # Orientation in Euler angles
mpclab_common/OrientationQuaternionMsg   q # Orientation in quaternions
mpclab_common/BodyAngularVelocityMsg     w # Vector of angular velocities
mpclab_common/BodyAngularAccelerationMsg aa # Vector of angular accelerations
mpclab_common/BodyLinearVelocityMsg      v # local body velocities
mpclab_common/BodyLinearAccelerationMsg  a # local body accelerations

mpclab_common/ParametricPoseMsg          p
mpclab_common/ParametricVelocityMsg      pt

mpclab_common/VehicleActuationMsg        u
mpclab_common/DriveStateMsg              hw

float64                                  lap_num    # Lap number

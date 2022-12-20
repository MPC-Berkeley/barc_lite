#!/usr/bin/env python3

import rclpy
from rclpy.qos import qos_profile_sensor_data

from nav_msgs.msg import Odometry

import numpy as np
from scipy.spatial.transform import Rotation

from mpclab_simulation.optitrack_simulator import OptiTrackSimulator
from mpclab_simulation.sim_types import OptiTrackSimConfig

from mpclab_common.msg import VehicleStateMsg, VehicleActuationMsg
from mpclab_common.mpclab_base_nodes import MPClabNode
from mpclab_common.pytypes import NodeParamTemplate, VehicleState

class OptiTrackSimNodeParams(NodeParamTemplate):
    '''
    template that stores all parameters needed for the node as well as default values
    '''
    def __init__(self):
        self.dt = 0.01
        self.name = 'tracker'
        self.dropout_rate = 0.0
        self.config = OptiTrackSimConfig()

class OptiTrackSimNode(MPClabNode):
    '''
    The Vive simulation node subscribes to the 'sim_state' topic from a dynamics simulator and publishes
    the simulated output of the Vive tracking system
    '''

    def __init__(self):
        super().__init__('OptiTrack_simulator')

        namespace = self.get_namespace()
        param_template = OptiTrackSimNodeParams()
        self.autodeclare_parameters(param_template, namespace)
        self.autoload_parameters(param_template, namespace)

        # self.get_logger().info(str(self.config))
        self.sim = OptiTrackSimulator(self.config)

        self.update_timer = self.create_timer(self.dt, self.step)

        self.sim_sub = self.create_subscription(
            VehicleStateMsg,
            'sim_state',
            self.state_callback,
            qos_profile_sensor_data)

        self.pub = self.create_publisher(
            Odometry,
            '/'.join((self.name,'odom')),
            qos_profile_sensor_data)

        # Get handle to ROS clock
        self.clock = self.get_clock()
        self.t_start = self.clock.now().nanoseconds/1E9

        self.rng = np.random.default_rng()

        self.state = VehicleState()

        self.mode = 'init'

    def state_callback(self, msg):
        self.unpack_msg(msg, self.state)

    def step(self):
        t = self.clock.now().nanoseconds/1E9
        if self.mode == 'init':
            if self.state.t is not None:
                self.get_logger().info('===== OptiTrack simulator start =====') # TODO: Need to change to rclpy.logging
                self.mode = 'run'
        else:
            meas = self.sim.step(self.state)  # modifies sim_state by reference
            pose = meas['pose']

            # Get message time stamp
            # t_sec = int(np.floor(self.state.t))
            # t_nsec = int((self.state.t-t_sec)*1E9)
            t_sec = int(np.floor(t))
            t_nsec = int((t-t_sec)*1E9)

            # Print simulated vive output
            # self.get_logger().info('Dyn sim: X: %g, Y: %g, yaw: %g, yaw_dot: %g' % (self.state.x, self.state.y, self.state.psi*180/np.pi, self.state.psidot))
            # self.get_logger().info('Vive sim: X: %g, Y: %g, yaw: %g, yaw_dot: %g' % (pose.x, pose.y, pose.yaw*180/np.pi, pose.yaw_dot))
            rot = Rotation.from_euler('ZYX', [pose.yaw, pose.pitch, pose.roll])
            quat = rot.as_quat()

            # Populate the odometry message and publish
            msg = Odometry()
            msg.header.stamp.sec = t_sec
            msg.header.stamp.nanosec = t_nsec
            msg.pose.pose.position.x = pose.x
            msg.pose.pose.position.y = pose.y
            msg.pose.pose.position.z = pose.z
            msg.pose.pose.orientation.x = quat[0]
            msg.pose.pose.orientation.y = quat[1]
            msg.pose.pose.orientation.z = quat[2]
            msg.pose.pose.orientation.w = quat[3]
            msg.twist.twist.linear.x = pose.v_long
            msg.twist.twist.linear.y = pose.v_tran
            msg.twist.twist.linear.z = pose.v_vert
            msg.twist.twist.angular.x = pose.roll_dot
            msg.twist.twist.angular.y = pose.pitch_dot
            msg.twist.twist.angular.z = pose.yaw_dot

            # After 2 seconds simulated vive measurements will dropout at self.dropout_rate
            if t-self.t_start < 2.0 or self.rng.random() > self.dropout_rate:
                self.pub.publish(msg)

def main(args = None):
    rclpy.init(args = args)
    simulator = OptiTrackSimNode()
    rclpy.spin(simulator)

    simulator.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

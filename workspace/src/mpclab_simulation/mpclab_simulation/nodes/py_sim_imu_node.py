#!/usr/bin/env python3

import rclpy
from rclpy.qos import qos_profile_sensor_data

from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu

import numpy as np
from scipy.spatial.transform import Rotation

from mpclab_simulation.imu_simulator import IMUSimulator
from mpclab_simulation.sim_types import IMUSimConfig

from mpclab_common.msg import VehicleStateMsg, VehicleActuationMsg
from mpclab_common.mpclab_base_nodes import MPClabNode
from mpclab_common.pytypes import NodeParamTemplate, VehicleState

class IMUSimNodeParams(NodeParamTemplate):
    '''
    template that stores all parameters needed for the node as well as default values
    '''
    def __init__(self):
        self.dt = 0.01
        self.name = ''
        self.imu_config = IMUSimConfig()

class IMUSimNode(MPClabNode):
    '''
    The Vive simulation node subscribes to the 'sim_state' topic from a dynamics simulator and publishes
    the simulated output of the Vive tracking system
    '''

    def __init__(self):
        super().__init__('t265_simulation')

        namespace = self.get_namespace()
        param_template = IMUSimNodeParams()
        self.autodeclare_parameters(param_template, namespace)
        self.autoload_parameters(param_template, namespace)

        # self.get_logger().info(str(self.imu_config))
        self.imu_sim = IMUSimulator(self.imu_config)

        self.update_timer = self.create_timer(self.dt, self.step)

        self.sim_sub = self.create_subscription(
            VehicleStateMsg,
            'sim_state',
            self.state_callback,
            qos_profile_sensor_data)

        self.imu_pub = self.create_publisher(
            Imu,
            '/'.join((self.name, 'imu')),
            qos_profile_sensor_data)

        # Get handle to ROS clock
        self.clock = self.get_clock()
        self.t_start = self.clock.now().nanoseconds/1E9

        self.state = VehicleState()

        self.mode = 'init'

    def state_callback(self, msg):
        self.unpack_msg(msg, self.state)

    def step(self):
        t = self.clock.now().nanoseconds/1E9
        if self.mode == 'init':
            if self.state.t is not None:
                self.imu_sim.initialize()
                self.get_logger().info('===== IMU simulator start =====') # TODO: Need to change to rclpy.logging
                self.mode = 'run'
        else:
            imu_meas = self.imu_sim.step(self.state)['imu']

            # Get message time stamp
            t_sec = int(np.floor(t))
            t_nsec = int((t-t_sec)*1E9)

            # Populate the IMU message
            imu_msg = Imu()
            imu_msg.header.stamp.sec = t_sec
            imu_msg.header.stamp.nanosec = t_nsec
            imu_msg.orientation.x = imu_meas.orientation.x
            imu_msg.orientation.y = imu_meas.orientation.y
            imu_msg.orientation.z = imu_meas.orientation.z
            imu_msg.orientation.w = imu_meas.angular_velocity.w
            imu_msg.linear_acceleration.x = imu_meas.linear_acceleration.x
            imu_msg.linear_acceleration.y = imu_meas.linear_acceleration.y
            imu_msg.linear_acceleration.z = imu_meas.linear_acceleration.z
            imu_msg.angular_velocity.x = imu_meas.angular_velocity.x
            imu_msg.angular_velocity.y = imu_meas.angular_velocity.y
            imu_msg.angular_velocity.z = imu_meas.angular_velocity.x

            # self.get_logger().info(str(imu_msg))

            # Publish
            self.imu_pub.publish(imu_msg)

def main(args = None):
    rclpy.init(args = args)
    simulator = IMUSimNode()
    rclpy.spin(simulator)

    simulator.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

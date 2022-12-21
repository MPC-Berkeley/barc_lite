#!/usr/bin/env python3

import rclpy
from rclpy.qos import qos_profile_sensor_data

from mpclab_common.msg import VehicleStateMsg, VehicleActuationMsg
from mpclab_common.pytypes import VehicleActuation, VehicleState
from mpclab_common.mpclab_base_nodes import MPClabNode

import pdb

class OpenLoopControlNodeParams():
    '''
    template that stores all parameters needed for the node as well as default values
    '''
    def __init__(self):
        self.dt = 0.1
        return

class OpenLoopControlNode(MPClabNode):

    def __init__(self):
        super().__init__('ol_control')
        namespace = self.get_namespace()

        param_template = OpenLoopControlNodeParams()
        self.autodeclare_parameters(param_template, namespace)
        self.autoload_parameters(param_template, namespace)

        self.update_timer = self.create_timer(self.dt, self.step)

        self.state_sub = self.create_subscription(
            VehicleStateMsg,
            'est_state',
            self.state_callback,
            qos_profile_sensor_data)

        self.control_pub = self.create_publisher(
            VehicleActuationMsg,
            'ecu',
            qos_profile_sensor_data)

        # Publish closed loop state input trajectories for system ID
        self.logger_pub = self.create_publisher(
            VehicleStateMsg,
            'closed_loop_traj',
            qos_profile_sensor_data)

        # Get handle to ROS clock
        self.clock = self.get_clock()
        self.t_start = self.clock.now().nanoseconds/1E9

        self.state = VehicleState()
        self.input = VehicleActuation(u_a=0, u_steer=0)

        self.wait_time = 5.0

        self.get_logger().info('===== Open Loop Controller start =====')

    def state_callback(self, msg):
        self.unpack_msg(msg, self.state)

    def step(self):
        t = self.clock.now().nanoseconds/1E9
        if  t - self.t_start < self.wait_time:
            self.input.t, self.input.u_a, self.input.u_steer = t, 0, 0
            control_msg = self.populate_msg(VehicleActuationMsg(), self.input)
            self.control_pub.publish(control_msg)
            return

        self.input.t = t
        # self.input.u_a = 0.1*np.sin(2*np.pi*(t-(self.t_start+self.wait_time))/5.0)+0.5
        # self.input.u_steer = 0.3*np.sin(2*np.pi*(t-(self.t_start+self.wait_time))/5.0)
        if t - self.t_start < self.wait_time + 3:
            self.input.u_a = 0.5
            self.input.u_steer = 0.0
        else:
            self.input.u_a = 0.0
            self.input.u_steer = 0.0

        control_msg = self.populate_msg(VehicleActuationMsg(), self.input)
        self.control_pub.publish(control_msg)

        logger_state = self.state.copy()
        logger_state.t = t
        logger_state.u_a = self.input.u_a
        logger_state.u_steer = self.input.u_steer

        logger_msg = self.populate_msg(VehicleStateMsg(), logger_state)
        self.logger_pub.publish(logger_msg)

def main(args=None):
    rclpy.init(args=args)
    control_node = OpenLoopControlNode()
    rclpy.spin(control_node)

    control_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

#!/usr/bin/env python3

import rclpy
from rclpy.qos import qos_profile_sensor_data

from dataclasses import dataclass, field

from mpclab_controllers.PID import PIDLaneFollower
from mpclab_controllers.utils.controllerTypes import PIDParams

from mpclab_common.msg import VehicleStateMsg, VehicleActuationMsg
from mpclab_common.pytypes import VehicleActuation, VehicleState
from mpclab_common.mpclab_base_nodes import MPClabNode, ControllerScheduler

from mpclab_common.track import get_track


class PIDControlNodeParams():
    '''
    template that stores all parameters needed for the node as well as default values
    '''
    def __init__(self):
        self.dt = 0.1
        self.n_laps = 5
        self.track_name = 'LTrack_barc'
        self.pid_steer_params = PIDParams()
        self.pid_speed_params = PIDParams()

        self.pid_steer_params.default_steer_params()
        self.pid_speed_params.default_speed_params()
        return


class PIDControlNode(MPClabNode):

    def __init__(self):
        super().__init__('pid_control')

        self.namespace = self.get_namespace()
        param_template = PIDControlNodeParams()
        self.autodeclare_parameters(param_template, self.namespace)
        self.autoload_parameters(param_template, self.namespace)

        self.pid_controller = PIDLaneFollower(self.dt, self.pid_steer_params, self.pid_speed_params)

        self.track = get_track(self.track_name)
        self.track_length = self.track.track_length # used by check_lap_status()

        self.update_timer = self.create_timer(self.dt, self.step)

        self.state = VehicleState()
        self.input = VehicleActuation(u_a=0, u_steer=0)

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
            'state_input_log',
            qos_profile_sensor_data)

        # Get handle to ROS clock
        self.clock = self.get_clock()
        self.t_start = self.clock.now().nanoseconds/1E9

        self.controller_scheduler = ControllerScheduler(self)
        self.controller_scheduler.add_controller(self.pid_controller, self.n_laps)

        self.wait_time = 5.0

        return

    def state_callback(self, msg):
        self.unpack_msg(msg, self.state)
        return

    def step(self):
        nodes = self.get_node_names_and_namespaces()
        # self.get_logger().info(str(nodes))
        t = self.clock.now().nanoseconds/1E9
        # We hard code the controller to wait for a bit before starting
        if  t - self.t_start < self.wait_time:
            self.input.t, self.input.u_a, self.input.u_steer = t, 0, 0
            control_msg = self.populate_msg(VehicleActuationMsg(), self.input)
            self.control_pub.publish(control_msg)
            return

        # Wait for state estimate to start coming in
        if self.state.t is None:
            self.get_logger().info('Waiting for valid state estimate')
            return

        # self.get_logger().info('control time: %g, state time: %g, delta: %g' % (t, self.state.t, t-self.state.t))

        if self.track.global_to_local_typed(self.state):
            self.get_logger().info('Vehicle is outside of track, Disabling output')
            self.state.u.u_a = 0
            self.state.u.u_steer = 0
        else:
            self.check_lap_status(self.state)
            self.controller_scheduler.step(self.state, None)

        self.state.copy_control(self.input)

        self.input.t = t
        control_msg = self.populate_msg(VehicleActuationMsg(), self.input)
        self.control_pub.publish(control_msg)

        logger_state = self.state.copy()
        logger_state.t = t
        logger_msg = self.populate_msg(VehicleStateMsg(), logger_state)
        self.logger_pub.publish(logger_msg)
        return


def main(args=None):
    rclpy.init(args=args)
    control_node = PIDControlNode()
    rclpy.spin(control_node)

    control_node.destroy_node()
    rclpy.shutdown()
    return

if __name__ == '__main__':
    main()

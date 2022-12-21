#!/usr/bin/env python3

import rclpy
from rclpy.qos import qos_profile_sensor_data

from mpclab_controllers.PID import PIDLaneFollower
from mpclab_controllers.utils.controllerTypes import PIDParams

from mpclab_common.msg import VehicleStateMsg, VehicleActuationMsg
from mpclab_common.pytypes import VehicleActuation, VehicleState
from mpclab_common.mpclab_base_nodes import MPClabNode, ControllerScheduler

from mpclab_common.track import get_track

class PIDControlNode(MPClabNode):

    def __init__(self):
        super().__init__('pid_control')

        self.namespace = self.get_namespace()

        self.track_name = 'L_track_barc'
        self.dt = 0.1
        self.n_laps = 50
        pid_steer_params = PIDParams(dt=self.dt,
                                     Kp=1.0,
                                     Ki=0.005,
                                     Kd=0.0,
                                     u_min=-0.436,
                                     u_max=0.436,
                                     du_min=-4.5,
                                     du_max=4.5,
                                     x_ref=0.0)
        pid_speed_params = PIDParams(dt=self.dt,
                                     Kp=1.0,
                                     Ki=0.005,
                                     Kd=0.0,
                                     u_min=-2.0,
                                     u_max=2.0,
                                     du_min=-10.0,
                                     du_max=10.0,
                                     x_ref=1.0)

        self.pid_controller = PIDLaneFollower(self.dt, pid_steer_params, pid_speed_params)

        self.track = get_track(self.track_name)
        self.L = self.track.track_length # used by check_lap_status()

        self.update_timer = self.create_timer(self.dt, self.step)

        self.state = VehicleState()
        self.state_prev = VehicleState()
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

        self.lap_start = 0
        self.lap_times = []
        self.lap_number = 0

        self.wait_time = 5.0

        return

    def state_callback(self, msg):
        self.unpack_msg(msg, self.state)
        return

    def step(self):
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
            self.input.t, self.input.u_a, self.input.u_steer = t, 0, 0
            control_msg = self.populate_msg(VehicleActuationMsg(), self.input)
            self.control_pub.publish(control_msg)
            return
        else:
            # Check if lap number needs to be incremented
            if self.state.p.s - self.state_prev.p.s < -self.L/2:
                lap_end = t
                self.lap_times.append(lap_end-self.lap_start)
                self.lap_start = lap_end
                self._print(f'===== Lap {self.lap_number} finished, time {self.lap_times[-1]} s. =====')
                self.lap_number += 1

                if self.lap_number >= self.n_laps and self.controller_mode != 'finished':
                    self._print('===== Laps finished =====')
                    self.controller_mode = 'finished'

                self._print(f'===== Starting lap {self.lap_number} =====')

            self.pid_controller.step(self.state)
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

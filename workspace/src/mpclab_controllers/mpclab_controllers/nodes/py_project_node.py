#!/usr/bin/env python3

import rclpy
from rclpy.qos import qos_profile_sensor_data

import copy

from mpclab_controllers.project_controller import ProjectController

from mpclab_common.mpclab_base_nodes import MPClabNode
from mpclab_common.msg import VehicleStateMsg, VehicleActuationMsg, PredictionMsg
from mpclab_common.pytypes import VehicleState, VehicleActuation
from mpclab_common.track import get_track

class ProjectControlNode(MPClabNode):

    def __init__(self):
        super().__init__('ProjectControlNode')
        self.get_logger().info('Initializing controller node')

        # Get handle to ROS clock
        self.clock = self.get_clock()
        self.t_start = self.clock.now().nanoseconds/1E9

        self.dt = 0.1
        self.n_laps = 50
        
        self.track_name = 'L_track_barc'
        self.track = get_track(self.track_name)
        self.L = self.track.track_length
        self.H = self.track.half_width  

        self.controller = ProjectController(self.dt,
                                            print_method=self.get_logger().info)

        self.state = VehicleState()
        self.input = VehicleActuation(u_a=0, u_steer=0)
        self.state_prev = VehicleState()
        self.input_prev = VehicleActuation(u_a=0, u_steer=0)

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

        self.prediction_pub = self.create_publisher(
            PredictionMsg,
            'pred',
            qos_profile_sensor_data)

        self.ref_pub = self.create_publisher(
            PredictionMsg,
            'ref',
            qos_profile_sensor_data)

        # Publish closed loop state input trajectories for system ID
        self.logger_pub = self.create_publisher(
            VehicleStateMsg,
            'state_input_log',
            qos_profile_sensor_data)

        self.controller_mode = 'initialization'
        self.wait_time = 5.0
        self.lap_start = 0
        self.lap_times = []
        self.lap_number = 0

        return

    def state_callback(self, msg):
        self.unpack_msg(msg, self.state)

    def step(self):
        t = self.clock.now().nanoseconds/1E9
        if  t - self.t_start < self.wait_time:
            self.input.t, self.input.u_a, self.input.u_steer = t, 0, 0
            control_msg = self.populate_msg(VehicleActuationMsg(), self.input)
            self.control_pub.publish(control_msg)
            return

        # Wait for valid state estimate
        if self.state.t is None:
            self.get_logger().info('Waiting for valid state estimate')
            return

        if self.controller_mode == 'initialization':
            self.input.u_a, self.input.u_steer = 0, 0
            if self.state.t is not None:
                self.controller_mode = 'run'
                self.lap_start = self.get_ros_time()
                self.state_prev = copy.deepcopy(self.state)
                self.controller.initialize(self.state)
                self.get_logger().info(f'===== Starting lap {self.lap_number} with controller {self.controller_mode} =====')
        elif self.controller_mode == 'run':
            # Convert global to local coords, stop controller if car is outside track
            if self.track.global_to_local_typed(self.state):
                self.get_logger().info('===== Vehicle is outside of track, stopping node =====')
                self.destroy_node()
            else:
                # Check if lap number needs to be incremented
                if self.state.p.s - self.state_prev.p.s < -self.L/2:
                    lap_end = t
                    self.lap_times.append(lap_end-self.lap_start)
                    self.lap_start = lap_end
                    self.get_logger().info(f'===== Lap {self.lap_number} finished, time {self.lap_times[-1]} s. =====')
                    self.lap_number += 1
                    if self.lap_number > self.n_laps:
                        self.get_logger().info('===== Laps finished, stopping node =====')
                        self.destroy_node()
                    self.get_logger().info(f'===== Starting lap {self.lap_number} =====')

                self.state.lap_num = self.lap_number
        else:
            self.get_logger().info(f'===== Control mode {self.controller_mode} not recognized, stopping node =====')
            self.destroy_node()

        # Save previous state and input
        self.state_prev = copy.deepcopy(self.state)
        self.input_prev = copy.copy(self.input)

        # Step the controller
        self.controller.step(self.state)
        self.state.copy_control(self.input)

        # Publish control
        self.input.t = t
        control_msg = self.populate_msg(VehicleActuationMsg(), self.input)
        self.control_pub.publish(control_msg)

        # Publish prediction
        pred = self.controller.get_prediction()
        if pred is not None:
            pred.t = t
            pred_msg = self.populate_msg(PredictionMsg(), pred)
            self.prediction_pub.publish(pred_msg)
        
        # Publish reference
        ref = self.controller.get_reference()
        if ref is not None:
            ref.t = t
            ref_msg = self.populate_msg(PredictionMsg(), ref)
            self.ref_pub.publish(ref_msg)

        # Publish log state
        logger_state = self.state.copy()
        logger_state.t = t
        logger_msg = self.populate_msg(VehicleStateMsg(), logger_state)
        self.logger_pub.publish(logger_msg)

        return

def main(args=None):
    rclpy.init(args=args)
    control_node = ProjectControlNode()
    rclpy.spin(control_node)

    control_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

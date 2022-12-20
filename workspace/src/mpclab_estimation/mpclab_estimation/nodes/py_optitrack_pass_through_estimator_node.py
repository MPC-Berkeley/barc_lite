#!/usr/bin/env python3

import rclpy
from rclpy.qos import qos_profile_sensor_data

from nav_msgs.msg import Odometry
from geometry_msgs.msg import Vector3Stamped

from mpclab_estimation.optitrack_interface import OptiTrackInterface
from mpclab_estimation.utils.interfaceTypes import OptiTrackParams
from mpclab_estimation.pass_through_estimator import PassThroughEstimator
from mpclab_estimation.utils.estimatorTypes import PassThroughParams

from mpclab_common.mpclab_base_nodes import MPClabNode
from mpclab_common.msg import VehicleStateMsg, VehicleActuationMsg
from mpclab_common.pytypes import VehicleState, VehicleActuation, NodeParamTemplate
from mpclab_common.models.model_types import PoseVelMeasurement, AccelMeasurement
from mpclab_common.track import get_track

from scipy.spatial.transform import Rotation
import numpy as np
from collections import deque
import copy

SHOW_MSG_TRANSFER_WARNINGS = False

# Default parameters, will be overwritten by self.autoload_parameters
class OptiTrackPassThroughEstimatorParams(NodeParamTemplate):
    def __init__(self):
        self.dt = 0.05
        self.track_name = 'L_track_barc'
        self.estimate_accel = False

        self.ot_params = OptiTrackParams()

class OptiTrackPassThroughEstimatorNode(MPClabNode):

    def __init__(self):
        super().__init__('optitrack_pass_through_estimator')

        namespace = self.get_namespace()
        param_template = OptiTrackPassThroughEstimatorParams()
        self.autodeclare_parameters(param_template, namespace)
        self.autoload_parameters(param_template, namespace)

        self.ot_interface = OptiTrackInterface(self.ot_params)

        self.track = get_track(self.track_name)

        # Get handle to ROS clock
        self.clock = self.get_clock()
        self.t_start = self.clock.now().nanoseconds/1E9

        self.estimator_params = PassThroughParams(dt=self.dt)
        self.estimator = PassThroughEstimator(self.track, self.estimator_params)
        self.estimator.initialize()

        self.update_timer = self.create_timer(self.dt, self.step)

        self.odom_sub = self.create_subscription(
            Odometry,
            self.ot_params.odom_topic,
            self.odom_callback,
            qos_profile_sensor_data)

        self.ecu_sub = self.create_subscription(
            VehicleActuationMsg,
            'ecu',
            self.ecu_callback,
            qos_profile_sensor_data)

        if self.estimate_accel:
            self.lin_accel_meas = AccelMeasurement()
            self.lin_accel_sub = self.create_subscription(
                Vector3Stamped,
                self.ot_params.lin_accel_topic,
                self.get_accel_callback(self.lin_accel_meas),
                qos_profile_sensor_data)
            
            self.ang_accel_meas = AccelMeasurement()
            self.ang_accel_sub = self.create_subscription(
                Vector3Stamped,
                self.ot_params.ang_accel_topic,
                self.get_accel_callback(self.ang_accel_meas),
                qos_profile_sensor_data)
        else:
            self.lin_accel_meas = None
            self.ang_accel_meas = None

        self.estimate_pub = self.create_publisher(
            VehicleStateMsg,
            'est_state',
            qos_profile_sensor_data)

        self.posevel_meas = PoseVelMeasurement()

        self.est_state = VehicleState()
        self.ecu = VehicleActuation()

        self.mode = 'init'
        self.meas_stale_timeout = self.dt

    def odom_callback(self, msg):
        t_sec = msg.header.stamp.sec
        t_nsec = msg.header.stamp.nanosec
        self.posevel_meas.t = t_sec + float(t_nsec)/float(1E9)

        self.posevel_meas.x = msg.pose.pose.position.x
        self.posevel_meas.y = msg.pose.pose.position.y
        self.posevel_meas.z = msg.pose.pose.position.z

        quat_x = msg.pose.pose.orientation.x
        quat_y = msg.pose.pose.orientation.y
        quat_z = msg.pose.pose.orientation.z
        quat_w = msg.pose.pose.orientation.w
        rot = Rotation.from_quat([quat_x, quat_y, quat_z, quat_w])
        # r, p, y = rot.as_euler('xyz')
        y, p, r = rot.as_euler('ZYX') # Use intrinsic rotations along principle axes 1. yaw axis, 2. pitch axis, 3. roll axis

        self.posevel_meas.roll = r
        self.posevel_meas.pitch = p
        self.posevel_meas.yaw = y

        self.posevel_meas.v_long = msg.twist.twist.linear.x
        self.posevel_meas.v_tran = msg.twist.twist.linear.y
        self.posevel_meas.v_vert = msg.twist.twist.linear.z

        self.posevel_meas.roll_dot = msg.twist.twist.angular.x
        self.posevel_meas.pitch_dot = msg.twist.twist.angular.y
        self.posevel_meas.yaw_dot = msg.twist.twist.angular.z

    def get_accel_callback(self, d):
        def accel_callback(msg):
            t_sec = msg.header.stamp.sec
            t_nsec = msg.header.stamp.nanosec
            d.t = t_sec + float(t_nsec)/float(1E9)

            d.x = msg.vector.x
            d.y = msg.vector.y
            d.z = msg.vector.z
        return accel_callback

    def ecu_callback(self, msg):
        self.ecu.t = msg.t
        self.ecu.u_a = msg.u_a
        self.ecu.u_steer = msg.u_steer

    def step(self):
        t = self.clock.now().nanoseconds/1E9

        # Wait for measurements to start coming in
        if self.posevel_meas.t is None:
            return

        if t - self.posevel_meas.t > self.meas_stale_timeout:
            self.get_logger().warn('OptiTrack measurement has not been updated in %f s' % (t-self.posevel_meas.t))
        
        com_meas = self.ot_interface.get_com_meas(self.posevel_meas, self.lin_accel_meas, self.ang_accel_meas)
        self.est_state = self.estimator.update(*com_meas, self.ecu)
        self.est_state.t = t

        # self.get_logger().info(f'X: {self.posevel_meas.x}, Y: {self.posevel_meas.y}, yaw: {self.posevel_meas.yaw}, pitch: {self.posevel_meas.pitch}, roll: {self.posevel_meas.roll}')

        est_state_msg = self.populate_msg(VehicleStateMsg(), self.est_state)
        self.estimate_pub.publish(est_state_msg)

def main(args=None):
    rclpy.init(args=args)
    node = OptiTrackPassThroughEstimatorNode()
    rclpy.spin(node)

    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

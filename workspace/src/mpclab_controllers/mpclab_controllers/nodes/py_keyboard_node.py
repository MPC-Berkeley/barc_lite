#!/usr/bin/env python3
import rclpy
from rclpy.qos import qos_profile_sensor_data
from mpclab_controllers.keyboard import KeyboardController

from mpclab_common.msg import VehicleStateMsg, VehicleActuationMsg
from mpclab_common.mpclab_base_nodes import MPClabNode
from mpclab_common.pytypes import VehicleState, VehicleActuation, NodeParamTemplate, VehiclePrediction

class KeyboardNodeParams(NodeParamTemplate):

    def __init__(self):
        self.dt = 0.1

class KeyboardControlNode(MPClabNode):
    def __init__(self):
        super().__init__('keyboard_control')
        self.get_logger().info('Initializing Keyboard Node')
        namespace = self.get_namespace()

        param_template = KeyboardNodeParams()
        self.autodeclare_parameters(param_template, namespace)
        self.autoload_parameters(param_template, namespace)

        # Get handle to ROS clock
        self.clock = self.get_clock()
        self.t_start = self.clock.now().nanoseconds/1E9

        self.controller = KeyboardController()

        self.update_timer = self.create_timer(self.dt, self.step)

        self.state = VehicleState()
        self.input = VehicleActuation(u_a=0, u_steer=0)


        self.control_pub = self.create_publisher(
            VehicleActuationMsg,
            'ecu',
            qos_profile_sensor_data)


    def step(self):
        t = self.clock.now().nanoseconds/1E9

        self.controller.step(self.state, None)

        self.state.copy_control(self.input)

        self.input.t = t
        control_msg = self.populate_msg(VehicleActuationMsg(), self.input)
        self.control_pub.publish(control_msg)

        return

def main(args=None):
    rclpy.init(args=args)
    control_node = KeyboardControlNode()
    rclpy.spin(control_node)

    control_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

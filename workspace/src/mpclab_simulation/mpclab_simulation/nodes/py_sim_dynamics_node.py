#!/usr/bin/env python3

import rclpy
from rclpy.qos import qos_profile_sensor_data

from mpclab_simulation.dynamics_simulator import DynamicsSimulator

from mpclab_common.msg import VehicleStateMsg, VehicleActuationMsg
from mpclab_common.mpclab_base_nodes import MPClabNode
from mpclab_common.pytypes import VehicleState, NodeParamTemplate, Position, OrientationEuler, BodyLinearVelocity, BodyAngularVelocity, BodyLinearAcceleration
from mpclab_common.models.dynamics_models import get_dynamics_model_params, get_dynamics_model
from mpclab_common.track import get_track

import copy

class DynamicsSimNodeParams(NodeParamTemplate):
    def __init__(self, model_name):
        self.track_name = None
        self.dt = None
        self.dynamics_config = get_dynamics_model_params(model_name)
        self.initial_config = VehicleState()
        self.delay = None

class DynamicsSimNode(MPClabNode):

    def __init__(self):
        super().__init__('simulation')

        # Get handle to ROS clock
        self.clock = self.get_clock()
        t0 = self.clock.now().nanoseconds/1E9

        namespace = self.get_namespace()

        # Need to get the model name first to set up parameter template
        model_name = self.load_parameter('dynamics_config.model_name')
        param_template = DynamicsSimNodeParams(model_name)

        self.autodeclare_parameters(param_template, namespace)
        self.autoload_parameters(param_template, namespace)

        self.dynamics_config.dt = self.dt
        self.dynamics_config.track_name = self.track_name
        self.simulator = DynamicsSimulator(t0, self.dynamics_config, delay=self.delay)

        self.u = [0.0,0.0]

        self.ctrl_msg_type = VehicleActuationMsg
        self.ctrl_msg_topic = 'ecu'
        self.ctrl_msg_callback = self.update

        self.sim_msg_type = VehicleStateMsg
        self.sim_msg_topic = 'sim_state'

        self.update_timer = self.create_timer(self.dt, self.step)

        self.ctrl_sub = self.create_subscription(
            self.ctrl_msg_type,
            self.ctrl_msg_topic,
            self.ctrl_msg_callback,
            qos_profile_sensor_data)

        self.sim_pub = self.create_publisher(
            self.sim_msg_type,
            self.sim_msg_topic,
            qos_profile_sensor_data)

        # Default nested states are 0.0, thus we can copy over initial_config
        self.sim_state = copy.deepcopy(self.initial_config)
        self.sim_state.t = t0

        self.init = True

    def update(self,msg):
        self.u = [msg.u_a, msg.u_steer]

    def step(self):
        t = self.clock.now().nanoseconds/1E9
        if self.init:
            self.get_logger().info('===== Vehicle simulator start =====')
            self.init = False

        self.sim_state.u.u_a = self.u[0]
        self.sim_state.u.u_steer = self.u[1]

        self.simulator.step(self.sim_state)  # modifies sim_state by reference
        # self.get_logger().info(str(self.sim_state))

        self.sim_state.t = t          # simulator increments timestap so simulation time is kept
        sim_msg = self.populate_msg(self.sim_msg_type(), self.sim_state)
        self.sim_pub.publish(sim_msg)

def main(args = None):
    rclpy.init(args=args)
    simulator = DynamicsSimNode()
    rclpy.spin(simulator)

    simulator.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

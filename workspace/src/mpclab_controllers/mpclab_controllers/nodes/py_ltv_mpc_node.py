#!/usr/bin/env python3

import rclpy
from rclpy.qos import qos_profile_sensor_data

import numpy as np
import casadi as ca

import copy

from mpclab_controllers.PID import PIDLaneFollower
from mpclab_controllers.CA_LTV_MPC import CA_LTV_MPC
from mpclab_controllers.utils.controllerTypes import CALTVMPCParams, PIDParams

from mpclab_common.mpclab_base_nodes import MPClabNode
from mpclab_common.msg import VehicleStateMsg, VehicleActuationMsg, PredictionMsg
from mpclab_common.pytypes import NodeParamTemplate, VehicleState, VehicleActuation, VehiclePrediction, ParametricPose, BodyLinearVelocity, BodyAngularVelocity
from mpclab_common.models.dynamics_models import CasadiDynamicCLBicycle
from mpclab_common.models.model_types import DynamicBicycleConfig
from mpclab_common.track import get_track

class LTVMPCControlNodeParams(NodeParamTemplate):
    '''
    template that stores all parameters needed for the node as well as default values
    '''
    def __init__(self):
        self.dt = 0.1
        self.n_init_laps = 1
        self.n_laps = 5
        self.track_name = 'L_track_barc'

        self.simulation = True

        self.v_long_max = 10
        self.v_long_min = -10
        self.v_tran_max = 10
        self.v_tran_min = -10
        self.w_psi_max  = 10
        self.w_psi_min  = -10
        self.u_a_max    = 2
        self.u_a_min    = -2
        self.u_steer_max = 0.45
        self.u_steer_min = -0.45
        self.u_a_rate_max    = 5
        self.u_a_rate_min    = -5
        self.u_steer_rate_max = 10
        self.u_steer_rate_min = -10

        self.pid_steer_params = PIDParams()
        self.pid_speed_params = PIDParams()
        self.dynamics_config = DynamicBicycleConfig()
        self.mpc_params = CALTVMPCParams()

class LTVMPCControlNode(MPClabNode):

    def __init__(self):
        super().__init__('ltv_mpc_control')
        self.get_logger().info('Initializing LTV MPC node')
        namespace = self.get_namespace()

        param_template = LTVMPCControlNodeParams()
        self.autodeclare_parameters(param_template, namespace)
        self.autoload_parameters(param_template, namespace)

        # Get handle to ROS clock
        self.clock = self.get_clock()
        self.t_start = self.clock.now().nanoseconds/1E9

        self.track = get_track(self.track_name)
        self.L = self.track.track_length
        self.H = self.track.half_width  

        self.pid_speed_params.u_max = self.u_a_max
        self.pid_speed_params.u_min = self.u_a_min
        self.pid_speed_params.du_max = self.u_a_rate_max
        self.pid_speed_params.du_min = self.u_a_rate_min

        self.pid_steer_params.u_max = self.u_steer_max
        self.pid_steer_params.u_min = self.u_steer_min
        self.pid_steer_params.du_max = self.u_steer_rate_max
        self.pid_steer_params.du_min = self.u_steer_rate_min

        self.pid_controller = PIDLaneFollower(self.dt, self.pid_steer_params, self.pid_speed_params)

        self.dynamics_config.dt = self.dt
        self.dynamics = CasadiDynamicCLBicycle(self.t_start, self.dynamics_config, track=self.track)

        state_input_ub = VehicleState(p=ParametricPose(s=2*self.L, x_tran=(self.H-0.1), e_psi=100),
                                    v=BodyLinearVelocity(v_long=self.v_long_max, v_tran=self.v_tran_max),
                                    w=BodyAngularVelocity(w_psi=self.w_psi_max),
                                    u=VehicleActuation(u_a=self.u_a_max, u_steer=self.u_steer_max))
        state_input_lb = VehicleState(p=ParametricPose(s=-2*self.L, x_tran=-(self.H-0.1), e_psi=-100),
                                    v=BodyLinearVelocity(v_long=self.v_long_min, v_tran=self.v_tran_min),
                                    w=BodyAngularVelocity(w_psi=self.w_psi_min),
                                    u=VehicleActuation(u_a=self.u_a_min, u_steer=self.u_steer_min))
        input_rate_ub = VehicleState(u=VehicleActuation(u_a=self.u_a_rate_max, u_steer=self.u_steer_rate_max))
        input_rate_lb = VehicleState(u=VehicleActuation(u_a=self.u_a_rate_min, u_steer=self.u_steer_rate_min))

        # Define state reference
        D = (self.n_laps+1)*self.L
        v_ref = 1.2
        s_ref = np.linspace(0, D, int(D/(v_ref*self.dt)))
        ey_ref = 0.25*np.sin(2*np.pi*6*s_ref/(self.L-1))
        self.q_ref = []
        for s, ey in zip(s_ref, ey_ref):
            self.q_ref.append(np.array([v_ref, 0, 0, 0, s, ey]))
        self.q_ref = np.array(self.q_ref)

        # Symbolic placeholder variables
        sym_q = ca.MX.sym('q', self.dynamics.n_q)
        sym_u = ca.MX.sym('u', self.dynamics.n_u)
        sym_du = ca.MX.sym('du', self.dynamics.n_u)

        sym_q_ref = ca.MX.sym('q_ref', self.dynamics.n_q)
        
        s_idx = 4
        ey_idx = 5

        ua_idx = 0
        us_idx = 1

        if self.simulation:
            Q = np.diag([1, 0, 0, 0, 10, 10])
            sym_input_stage = 0.5*(1e-4*(sym_u[ua_idx])**2 + 1e-4*(sym_u[us_idx])**2)
            sym_input_term = 0.5*(1e-4*(sym_u[ua_idx])**2 + 1e-4*(sym_u[us_idx])**2)
            sym_rate_stage = 0.5*(0.01*(sym_du[ua_idx])**2 + 1*(sym_du[us_idx])**2)
        else:
            Q = np.diag([1, 0, 0, 0, 10, 10])
            sym_input_stage = 0.5*(0.1*(sym_u[ua_idx])**2 + 0.1*(sym_u[us_idx])**2)
            sym_input_term = 0.5*(0.1*(sym_u[ua_idx])**2 + 0.1*(sym_u[us_idx])**2)
            sym_rate_stage = 0.5*(1.0*(sym_du[ua_idx])**2 + 1.0*(sym_du[us_idx])**2)

        sym_state_stage = 0.5*ca.bilin(Q, sym_q-sym_q_ref, sym_q-sym_q_ref)
        sym_state_term = 0.5*ca.bilin(Q, sym_q-sym_q_ref, sym_q-sym_q_ref)

        sym_costs = {'state': [None for _ in range(self.mpc_params.N+1)], 'input': [None for _ in range(self.mpc_params.N+1)], 'rate': [None for _ in range(self.mpc_params.N)]}
        for k in range(self.mpc_params.N):
            sym_costs['state'][k] = ca.Function(f'state_stage_{k}', [sym_q, sym_q_ref], [sym_state_stage])
            sym_costs['input'][k] = ca.Function(f'input_stage_{k}', [sym_u], [sym_input_stage])
            sym_costs['rate'][k] = ca.Function(f'rate_stage_{k}', [sym_du], [sym_rate_stage])
        sym_costs['state'][self.mpc_params.N] = ca.Function('state_term', [sym_q, sym_q_ref], [sym_state_term])
        sym_costs['input'][self.mpc_params.N] = ca.Function('input_term', [sym_u], [sym_input_term])

        sym_constrs = {'state_input': [None for _ in range(self.mpc_params.N+1)], 
                        'rate': [None for _ in range(self.mpc_params.N)]}

        self.ltv_mpc_controller = CA_LTV_MPC(self.dynamics, 
                                            sym_costs, 
                                            sym_constrs, 
                                            {'qu_ub': state_input_ub, 'qu_lb': state_input_lb, 'du_ub': input_rate_ub, 'du_lb': input_rate_lb},
                                            self.mpc_params,
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

        self.controller_mode = 'controller_init'
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

        # Wait for state estimate to start coming in
        if self.state.t is None:
            self.get_logger().info('Waiting for valid state estimate')
            return

        ref = None

        # Check for initialization and termination modes
        if self.controller_mode == 'controller_init':
            self.input.u_a, self.input.u_steer = 0, 0
            if self.state.t is not None:
                self.controller_mode = 'pid'
                self.lap_start = self.get_ros_time()
                self.state_prev = copy.deepcopy(self.state)
                self.get_logger().info(f'===== Starting lap {self.lap_number} with controller {self.controller_mode} =====')
        elif self.controller_mode == 'finished':
            self.input.u_a, self.input.u_steer = 0, 0
        else:
            # Convert global to local coords, stop controller if car is outside track
            if self.track.global_to_local_typed(self.state):
                self.get_logger().info('Vehicle is outside of track, sending zero actuation command')
                self.input.u_a, self.input.u_steer = 0, 0
                self.controller_mode = 'finished'
            else:
                # Check if lap number needs to be incremented
                if self.state.p.s < self.state_prev.p.s and self.state.p.s < 1.0 and t - self.t_start >= self.wait_time+1.0:
                    lap_end = t
                    self.lap_times.append(lap_end-self.lap_start)
                    self.lap_start = lap_end
                    
                    self.get_logger().info(f'===== Lap {self.lap_number} finished, time {self.lap_times[-1]} s. =====')

                    self.lap_number += 1

                    if self.lap_number >= self.n_laps + self.n_init_laps and self.controller_mode != 'finished':
                        self.get_logger().info('===== Laps finished =====')
                        self.controller_mode = 'finished'
                    elif self.lap_number >= self.n_init_laps-1 and self.controller_mode != 'ltv_mpc':
                        self.get_logger().info('===== Initialization laps finished =====')
                        self.controller_mode = 'ltv_mpc'
                        ref_start = np.argmin(np.abs(self.q_ref[:,4]-self.state.p.s))
                        q_ref = copy.copy(self.q_ref[ref_start:ref_start+self.mpc_params.N+1])
                        u_ws = np.tile(self.dynamics.input2u(self.input), (self.mpc_params.N+1, 1))
                        du_ws = np.zeros((self.mpc_params.N, self.dynamics.n_u))
                        self.ltv_mpc_controller.set_warm_start(u_ws, du_ws, state=self.state, params=q_ref.ravel())

                    self.get_logger().info(f'===== Starting lap {self.lap_number} with controller {self.controller_mode} =====')

                # Solve for control action
                self.state.p.s = np.mod(self.state.p.s, self.L)
                if self.controller_mode == 'pid':
                    self.pid_controller.step(self.state)
                elif self.controller_mode == 'ltv_mpc':
                    s = self.state.p.s + self.L*(self.lap_number-1)
                    ref_start = np.argmin(np.abs(self.q_ref[:,4]-s))
                    q_ref = copy.copy(self.q_ref[ref_start:ref_start+self.mpc_params.N+1])
                    q_ref[:,4] = q_ref[:,4] - self.L*(self.lap_number-1)
                    self.ltv_mpc_controller.step(self.state, params=q_ref.ravel())
                    ref = VehiclePrediction()
                    self.dynamics.qu2prediction(ref, q=q_ref)
                self.state.copy_control(self.input)
                self.state.lap_num = self.lap_number

        self.state_prev = copy.deepcopy(self.state)
        self.input_prev = copy.copy(self.input)

        self.input.t = t
        control_msg = self.populate_msg(VehicleActuationMsg(), self.input)
        self.control_pub.publish(control_msg)

        pred = self.ltv_mpc_controller.get_prediction()
        if pred is not None:
            pred.t = t
            pred_msg = self.populate_msg(PredictionMsg(), pred)
            self.prediction_pub.publish(pred_msg)
        
        if ref is not None:
            ref.t = t
            ref_msg = self.populate_msg(PredictionMsg(), ref)
            self.ref_pub.publish(ref_msg)

        logger_state = self.state.copy()
        logger_state.t = t
        logger_msg = self.populate_msg(VehicleStateMsg(), logger_state)
        self.logger_pub.publish(logger_msg)

        return

def main(args=None):
    rclpy.init(args=args)
    control_node = LTVMPCControlNode()
    rclpy.spin(control_node)

    control_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

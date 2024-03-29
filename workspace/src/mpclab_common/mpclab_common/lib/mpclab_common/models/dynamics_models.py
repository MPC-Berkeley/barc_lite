#!/usr/bin python3

import numpy as np
import scipy.linalg as la
from scipy.integrate import solve_ivp

import casadi as ca

from abc import abstractmethod
import array
from typing import Tuple, List

from mpclab_common.pytypes import VehicleState, VehicleActuation, VehiclePrediction
from mpclab_common.models.model_types import *
from mpclab_common.models.abstract_model import AbstractModel
from mpclab_common.track import get_track


class CasadiDynamicsModel(AbstractModel):
    '''
    Base class for dynamics models that use casadi for their models.
    Implements common functions for linearizing models, integrating models, etc...
    '''

    # whether or not the dynamics depend on curvature
    # curvature is not a state variable as it must be calculated from the track definition
    # so any such model has a third input that is needed for calculations!
    curvature_model = False

    def __init__(self, t0: float, model_config: DynamicsConfig, track=None):
        super().__init__(model_config)

        self.t0 = t0
        # Try to Load track by name first, then by track object passed into model init
        if model_config.track_name is not None:
            self.track = get_track(model_config.track_name)
        else:
            self.track = track

        self.dt = model_config.dt
        self.M = model_config.M # RK4 integration steps
        self.h = self.dt/self.M # RK4 integration time intervals

    @abstractmethod
    def state2qu(self):
        pass

    @abstractmethod
    def state2q(self):
        pass

    @abstractmethod
    def input2u(self):
        pass

    @abstractmethod
    def q2state(self):
        pass

    def precompute_model(self):
        '''
        wraps up model initialization
        require the following fields to be initialized:
        self.sym_q:  ca.SX with elements of state vector q
        self.sym_u:  ca.SX with elements of control vector u
        self.sym_dq: ca.SX with time derivatives of q (dq/dt = sym_dq(q,u))
        '''

        dyn_inputs = [self.sym_q, self.sym_u]
        if type(self.dt) is ca.SX or type(self.dt) is ca.MX:
            dyn_inputs += [self.dt]

        # Continuous time dynamics function
        self.fc = ca.Function('fc', dyn_inputs, [self.sym_dq], self.options('fc'))

        # First derivatives
        self.sym_Ac = ca.jacobian(self.sym_dq, self.sym_q)
        self.sym_Bc = ca.jacobian(self.sym_dq, self.sym_u)
        self.sym_Cc = self.sym_dq

        self.fA = ca.Function('fA', dyn_inputs, [self.sym_Ac], self.options('fA'))
        self.fB = ca.Function('fB', dyn_inputs, [self.sym_Bc], self.options('fB'))
        self.fC = ca.Function('fC', dyn_inputs, [self.sym_Cc], self.options('fC'))

        # Discretization
        if self.model_config.discretization_method == 'euler':
            sym_q_kp1 = self.sym_q + self.dt * self.fc(*dyn_inputs)
        elif self.model_config.discretization_method == 'rk4':
            sym_q_kp1 = self.rk4(*dyn_inputs, self.fc, self.M, self.h)
        elif self.model_config.discretization_method == 'rk3':
            sym_q_kp1 = self.rk3(*dyn_inputs, self.fc, self.M, self.h)
        elif self.model_config.discretization_method == 'rk2':
            sym_q_kp1 = self.rk2(*dyn_inputs, self.fc, self.M, self.h)
        else:
            raise ValueError('Discretization method of %s not recognized' % self.model_config.discretization_method)

        if self.model_config.noise:
            if self.model_config.noise_cov is None:
                raise RuntimeError('Noise covariance matrix not provided to dynamics model')
            noise_cov = np.array(self.model_config.noise_cov)
            self.noise_cov = np.diag(noise_cov) if noise_cov.ndim == 1 else noise_cov

            # Symbolic variables for additive process noise components for each state
            self.n_m = self.n_q
            self.sym_m = ca.SX.sym('m', self.n_m)
            sym_q_kp1 += ca.mtimes(la.sqrtm(self.noise_cov), self.sym_m)
            dyn_inputs += [self.sym_m]

        # Discrete time dynamics function
        self.fd = ca.Function('fd', dyn_inputs, [sym_q_kp1], self.options('fd'))

        # First derivatives
        self.sym_Ad = ca.jacobian(sym_q_kp1, self.sym_q)
        self.sym_Bd = ca.jacobian(sym_q_kp1, self.sym_u)
        self.sym_Cd = sym_q_kp1

        self.fAd = ca.Function('fAd', dyn_inputs, [self.sym_Ad], self.options('fAd'))
        self.fBd = ca.Function('fBd', dyn_inputs, [self.sym_Bd], self.options('fBd'))
        self.fCd = ca.Function('fCd', dyn_inputs, [self.sym_Cd], self.options('fCd'))

        # Second derivatives
        if self.model_config.compute_hessians:
            self.sym_Ed = [ca.jacobian(ca.jacobian(sym_q_kp1[i], self.sym_q), self.sym_q) for i in range(self.n_q)]
            self.sym_Fd = [ca.jacobian(ca.jacobian(sym_q_kp1[i], self.sym_u), self.sym_u) for i in range(self.n_q)]
            self.sym_Gd = [ca.jacobian(ca.jacobian(sym_q_kp1[i], self.sym_u), self.sym_q) for i in range(self.n_q)]

            self.fEd = ca.Function('fEd', dyn_inputs, self.sym_Ed, self.options('fEd'))
            self.fFd = ca.Function('fFd', dyn_inputs, self.sym_Fd, self.options('fFd'))
            self.fGd = ca.Function('fGd', dyn_inputs, self.sym_Gd, self.options('fGd'))
        
        if self.model_config.noise:
            self.sym_Md = ca.jacobian(sym_q_kp1, self.sym_m)
            self.fMd = ca.Function('fMd', dyn_inputs, [self.sym_Md], self.options('fMd'))

        # Build shared object if not doing just-in-time compilation
        if self.code_gen and not self.jit:
            so_fns = [self.fc, self.fA, self.fB, self.fC, self.fd, self.fAd, self.fBd, self.fCd]
            if self.model_config.compute_hessians:
                so_fns += [self.fEd, self.fFd, self.fGd]
            if self.model_config.noise:
                so_fns += [self.fMd]
            self.install_dir = self.build_shared_object(so_fns)

        return

    def step(self, vehicle_state: VehicleState):
        '''
        steps noise-free model forward one time step (self.dt) using numerical integration
        '''
        q, u = self.state2qu(vehicle_state)
        t = vehicle_state.t - self.t0
        tf = t + self.dt

        # q_n = self.rk4(q, u, self.fc, self.M, self.h).toarray().squeeze()
        sol = solve_ivp(lambda t, z: np.array(self.fc(z, u)).squeeze(), (0, self.dt), q, t_eval=[self.dt])
        q_n = sol.y.squeeze()
        
        a_l, a_t = self.f_a(q_n, u)

        self.qu2state(vehicle_state, q_n, u)
        vehicle_state.t = tf + self.t0
        vehicle_state.a.a_long, vehicle_state.a.a_tran = float(a_l), float(a_t)

        if self.curvature_model:
            self.track.local_to_global_typed(vehicle_state)
        else:
            self.track.global_to_local_typed(vehicle_state)
        return

    def rk4(self, x, u, f, M, h):
        '''
        Discrete nonlinear dynamics (RK4 approx.)
        '''
        x_p = x
        for _ in range(M):
            a1 = f(x_p, u)
            a2 = f(x_p + (h / 2) * a1, u)
            a3 = f(x_p + (h / 2) * a2, u)
            a4 = f(x_p + h * a3, u)
            x_p += h * (a1 + 2 * a2 + 2 * a3 + a4) / 6
        return x_p

    def rk3(self, x, u, f, M, h):
        '''
        Discrete nonlinear dynamics (RK3 approx.)
        '''
        x_p = x
        for _ in range(M):
            a1 = h * f(x_p, u)
            a2 = h * f(x_p + a1/2, u)
            a3 = h * f(x_p - a1 + 2*a2, u)
            x_p += (a1 + 4*a2 + a3) / 6
        return x_p

    def rk2(self, x, u, f, M, h):
        x_p = x
        for _ in range(M):
            a1 = f(x_p, u)
            a2 = f(x_p + h*a1, u)
            x_p += h * (a1 + a2) / 2
        return x_p

    def ca_pos_abs(self, x, eps = 1e-3):
        '''
        smooth, positive apporoximation to abs(x)
        meant for tire slip ratios, where result must be nonzero
        '''
        return ca.sqrt(x**2 + eps**2)

    def ca_abs(self, x):
        '''
        absolute value in casadi
        do not use for tire slip ratios
        used for quadratic drag: c * v * abs(v)
        '''
        return ca.if_else(x > 0, x, -x)

    def ca_sign(self, x, eps = 1e-3):
        ''' smooth apporoximation to sign(x)'''
        return x / self.ca_pos_abs(x)

class CasadiDynamicBicycle(CasadiDynamicsModel):
    '''
    Global frame of reference dynamic bicycle model - Pacejka model tire forces

    Body frame velocities and global frame positions
    '''

    def __init__(self, t0: float,
                    model_config: DynamicBicycleConfig = DynamicBicycleConfig(), track=None):
        super().__init__(t0, model_config, track=track)

        self.curvature_model = False

        self.n_q = 6
        self.n_u = 2

        self.L_f            = self.model_config.wheel_dist_front
        self.L_r            = self.model_config.wheel_dist_rear

        self.m              = self.model_config.mass
        self.I_z            = self.model_config.yaw_inertia
        self.g              = self.model_config.gravity

        self.c_dr           = self.model_config.drag_coefficient
        self.c_da           = self.model_config.damping_coefficient
        self.c_r            = self.model_config.rolling_resistance
        self.p_r            = self.model_config.rolling_resistance_exponent

        self.mu             = self.model_config.wheel_friction
        self.tire_model     = self.model_config.tire_model

        self.pacejka_Bf     = self.model_config.pacejka_b_front
        self.pacejka_Br     = self.model_config.pacejka_b_rear
        self.pacejka_Cf     = self.model_config.pacejka_c_front
        self.pacejka_Cr     = self.model_config.pacejka_c_rear
        self.pacejka_Df     = self.model_config.pacejka_d_front
        self.pacejka_Dr     = self.model_config.pacejka_d_rear

        self.linear_Bf      = self.model_config.linear_bf
        self.linear_Br      = self.model_config.linear_br

        self.simple_slip    = self.model_config.simple_slip

        # symbolic variables
        self.sym_vx         = ca.SX.sym('vx')  # body fram vx, vy (vx>0 points in direction car points, vy>0 points to left hand side)
        self.sym_vy         = ca.SX.sym('vy')
        self.sym_psidot     = ca.SX.sym('psidot')
        self.sym_x          = ca.SX.sym('x')
        self.sym_y          = ca.SX.sym('y')
        self.sym_psi        = ca.SX.sym('psi')
        self.sym_u_s        = ca.SX.sym('gamma')
        self.sym_u_a        = ca.SX.sym('a')

        # Slip angles and Pacejka tire forces
        if self.simple_slip:
            self.sym_alpha_f = -ca.atan2(self.sym_vy + self.L_f * self.sym_psidot, self.sym_vx) + self.sym_u_s
        else:
            self.sym_alpha_f = -ca.atan2((self.sym_vy + self.L_f * self.sym_psidot) * ca.cos(self.sym_u_s) - self.sym_vx * ca.sin(self.sym_u_s),
                                         self.sym_vx * ca.cos(self.sym_u_s) + (self.sym_vy + self.L_f * self.sym_psidot) * ca.sin(self.sym_u_s))
        self.sym_alpha_r = -ca.atan2(self.sym_vy - self.L_r * self.sym_psidot, self.sym_vx)

        if self.tire_model == 'pacejka':
            self.sym_fyf = self.pacejka_Df * ca.sin(self.pacejka_Cf * ca.atan(self.pacejka_Bf * self.sym_alpha_f))
            self.sym_fyr = self.pacejka_Dr * ca.sin(self.pacejka_Cr * ca.atan(self.pacejka_Br * self.sym_alpha_r))
        elif self.tire_model == 'linear':
            self.sym_fyf = self.linear_Bf * self.m*self.g*self.L_r/(self.L_f+self.L_r) * self.sym_alpha_f
            self.sym_fyr = self.linear_Br * self.m*self.g*self.L_f/(self.L_f+self.L_r) * self.sym_alpha_r
        else:
            raise(ValueError("Tire model must be 'linear' or 'pacejka'"))

        # External forces
        F_ext = - self.c_da * self.sym_vx \
                - self.c_dr * self.sym_vx * self.ca_abs(self.sym_vx) \
                - self.c_r * self.ca_abs(self.sym_vx)**self.p_r * self.ca_sign(self.sym_vx)

        # instantaneous accelerations
        self.sym_ax = self.sym_u_a + (F_ext - self.sym_fyf * ca.sin(self.sym_u_s)) / self.m
        self.sym_ay = (self.sym_fyf * ca.cos(self.sym_u_s) + self.sym_fyr) / self.m
        self.sym_wz = (self.L_f * self.sym_fyf * ca.cos(self.sym_u_s) - self.L_r * self.sym_fyr) / self.I_z

        # time derivatives
        self.sym_dvx        = self.sym_ax + self.sym_psidot * self.sym_vy
        self.sym_dvy        = self.sym_ay - self.sym_psidot * self.sym_vx
        self.sym_dpsidot    = self.sym_wz
        self.sym_dx         = self.sym_vx * ca.cos(self.sym_psi) - self.sym_vy * ca.sin(self.sym_psi)
        self.sym_dy         = self.sym_vy * ca.cos(self.sym_psi) + self.sym_vx * ca.sin(self.sym_psi)
        self.sym_dpsi       = self.sym_psidot

        # state and state derivative functions
        self.sym_q  = ca.vertcat(self.sym_vx, self.sym_vy, self.sym_psidot, self.sym_x, self.sym_y, self.sym_psi)
        self.sym_u  = ca.vertcat(self.sym_u_a, self.sym_u_s)
        self.sym_dq = ca.vertcat(self.sym_dvx, self.sym_dvy, self.sym_dpsidot, self.sym_dx, self.sym_dy, self.sym_dpsi)
        
        # Auxilliary functions
        self.f_a = ca.Function('f_a', [self.sym_q, self.sym_u], [self.sym_ax, self.sym_ay], self.options('f_a'))
        self.f_alpha = ca.Function('f_alpha', [self.sym_q, self.sym_u], [self.sym_alpha_f, self.sym_alpha_r])

        self.precompute_model()
        return

    def get_slip_angles(self, state: VehicleState) -> np.ndarray:
        q, u = self.state2qu(state)
        return np.array(self.f_alpha(q, u)).squeeze()

    def state2qu(self, state: VehicleState) -> Tuple[np.ndarray, np.ndarray]:
        q = np.array([state.v.v_long, state.v.v_tran, state.w.w_psi, state.x.x, state.x.y, state.e.psi])
        u = np.array([state.u.u_a, state.u.u_steer])
        return q, u

    def state2q(self, state: VehicleState) -> np.ndarray:
        q = np.array([state.v.v_long, state.v.v_tran, state.w.w_psi, state.x.x, state.x.y, state.e.psi])
        return q

    def input2u(self, input: VehicleActuation) -> np.ndarray:
        u = np.array([input.u_a, input.u_steer])
        return u

    def q2state(self, state: VehicleState, q: np.ndarray):
        state.v.v_long  = q[0]
        state.v.v_tran  = q[1]
        state.w.w_psi   = q[2]
        state.x.x       = q[3]
        state.x.y       = q[4]
        state.e.psi     = q[5]
        return

    def u2input(self, input: VehicleActuation, u: np.ndarray):
        input.u_a       = u[0]
        input.u_steer   = u[1]
        return

    def qu2state(self, state: VehicleState, q: np.ndarray = None, u: np.ndarray = None):
        if q is not None:
            state.v.v_long  = q[0]
            state.v.v_tran  = q[1]
            state.w.w_psi   = q[2]
            state.x.x       = q[3]
            state.x.y       = q[4]
            state.e.psi     = q[5]
        if u is not None:
            state.u.u_a     = u[0]
            state.u.u_steer = u[1]
        return

    def qu2prediction(self, prediction: VehiclePrediction, q: np.ndarray = None, u: np.ndarray = None):
        if prediction is None:
            prediction = VehiclePrediction()
        if q is not None:
            prediction.v_long   = array.array('d', q[:, 0])
            prediction.v_tran   = array.array('d', q[:, 1])
            prediction.psidot   = array.array('d', q[:, 2])
            prediction.x        = array.array('d', q[:, 3])
            prediction.y        = array.array('d', q[:, 4])
            prediction.psi      = array.array('d', q[:, 5])
        if u is not None:
            prediction.u_a      = array.array('d', u[:, 0])
            prediction.u_steer  = array.array('d', u[:, 1])

        return prediction

def get_dynamics_model(t_start: float, model_config: DynamicsConfig, track=None) -> CasadiDynamicsModel:
    '''
    Helper function for getting a vehicle model class from a text string
    Should be used anywhere vehicle models may be changed by configuration
    '''
    if model_config.model_name == 'dynamic_bicycle':
        return CasadiDynamicBicycle(t_start, model_config, track=track)
    else:
        raise ValueError('Unrecognized vehicle model name: %s' % model_config.model_name)

def get_dynamics_model_params(model_name: str):
    if 'dynamic_bicycle' in model_name:
        return DynamicBicycleConfig()
    else:
        raise ValueError('Unrecognized vehicle type: %s' % model_name)

#!/usr/bin python3

from dataclasses import dataclass, field

from mpclab_common.pytypes import PythonMsg, VehicleState

@dataclass
class ControllerConfig(PythonMsg):
    dt: float = field(default=0.1)

@dataclass
class PIDParams(ControllerConfig):
    Kp: float = field(default=2.0)
    Ki: float = field(default=0.0)
    Kd: float = field(default=0.0)

    int_e_max: float = field(default=100)
    int_e_min: float = field(default=-100)
    u_max: float = field(default=None)
    u_min: float = field(default=None)
    du_max: float = field(default=None)
    du_min: float = field(default=None)

    u_ref: float = field(default=0.0)
    x_ref: float = field(default=0.0)

    noise: bool = field(default=False)
    noise_max: float = field(default=0.1)
    noise_min: float = field(default=-0.1)

    periodic_disturbance: bool = field(default=False)
    disturbance_amplitude: float = field(default=0.1)
    disturbance_period: float = field(default=1.0)

    def default_speed_params(self):
        self.Kp = 1
        self.Ki = 0
        self.Kd = 0
        self.u_min = -2
        self.u_max = 2
        self.du_min = -10 * self.dt
        self.du_max =  10 * self.dt
        self.noise = False
        return

    def default_steer_params(self):
        self.Kp = 1
        self.Ki = 0.0005 / self.dt
        self.Kd = 0
        self.u_min = -0.35
        self.u_max = 0.35
        self.du_min = -4 * self.dt
        self.du_max = 4 * self.dt
        self.noise = False
        return

@dataclass
class JoystickParams(ControllerConfig):
    dt: float                           = field(default = 0.1)

    u_steer_max: float                  = field(default = 0.436)
    u_steer_min: float                  = field(default = -0.436)
    u_steer_neutral: float              = field(default = 0.0)
    u_steer_rate_max: float             = field(default=None)
    u_steer_rate_min: float             = field(default=None)

    u_a_max: float                      = field(default = 2.0)
    u_a_min: float                      = field(default = -2.0)
    u_a_neutral: float                  = field(default = 0.0)
    u_a_rate_max: float                 = field(default=None)
    u_a_rate_min: float                 = field(default=None)
    
    throttle_pid: bool                  = field(default=False)
    steering_pid: bool                  = field(default=False)

    throttle_pid_params: PIDParams      = field(default=None)
    steering_pid_params: PIDParams      = field(default=None)


@dataclass
class CALTVMPCParams(ControllerConfig):
    N: int                              = field(default=10) # horizon length

    qp_interface: str                   = field(default='casadi')
    
    # Code gen options
    verbose: bool                       = field(default=False)
    code_gen: bool                      = field(default=False)
    jit: bool                           = field(default=False)
    opt_flag: str                       = field(default='O0')
    solver_name: str                    = field(default='LTV_MPC')
    debug_plot: bool                    = field(default=False)

    soft_state_bound_idxs: list         = field(default=None)
    soft_state_bound_quad: list         = field(default=None)
    soft_state_bound_lin: list          = field(default=None)

    soft_constraint_idxs: list         = field(default=None)
    soft_constraint_quad: list         = field(default=None)
    soft_constraint_lin: list          = field(default=None)

    wrapped_state_idxs: list            = field(default=None)
    wrapped_state_periods: list         = field(default=None)

    state_scaling: list                 = field(default=None)
    input_scaling: list                 = field(default=None)
    damping: float                      = field(default=0.75)
    qp_iters: int                       = field(default=2)

    delay: list                         = field(default=None)

@dataclass
class CAMPCCParams(ControllerConfig):
    N: int                              = field(default=10) # horizon length

    # Code gen options
    verbose: bool                       = field(default=False)
    code_gen: bool                      = field(default=False)
    jit: bool                           = field(default=False)
    opt_flag: str                       = field(default='O0')
    enable_jacobians: bool              = field(default=True)
    solver_name: str                    = field(default='CA_MPCC')
    solver_dir: str                     = field(default=None)
    debug_plot: bool                    = field(default=False)

    conv_approx: bool                   = field(default=False)
    soft_track: bool                    = field(default=False)
    track_tightening: float             = field(default=0)

    pos_idx: list                       = field(default_factory=lambda : [3, 4])
    state_scaling: list                 = field(default=None)
    input_scaling: list                 = field(default=None)
    damping: float                      = field(default=0.75)
    qp_iters: int                       = field(default=2)

    contouring_cost: float              = field(default=0.1)
    contouring_cost_N: float            = field(default=1.0)
    lag_cost: float                     = field(default=1000.0)
    lag_cost_N: float                   = field(default=1000.0)
    performance_cost: float             = field(default=0.02)
    vs_cost: float                      = field(default=1e-4)
    vs_rate_cost: float                 = field(default=1e-3)
    track_slack_quad: float             = field(default=100.0)
    track_slack_lin: float              = field(default=0.0)

    vs_max: float                       = field(default=5.0)
    vs_min: float                       = field(default=0.0)
    vs_rate_max: float                  = field(default=5.0)
    vs_rate_min: float                  = field(default=-5.0)

    delay: list                         = field(default=None)

if __name__ == "__main__":
    pass

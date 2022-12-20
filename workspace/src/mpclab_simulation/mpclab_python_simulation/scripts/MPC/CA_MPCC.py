#!/usr/bin/env python3

from mpclab_controllers.CA_MPCC import CA_MPCC
from mpclab_controllers.utils.controllerTypes import CAMPCCParams

from mpclab_common.pytypes import VehicleState, VehicleActuation, VehiclePrediction, Position, ParametricPose, BodyLinearVelocity, OrientationEuler, BodyAngularVelocity
from mpclab_common.models.dynamics_models import CasadiDynamicBicycle
from mpclab_common.models.model_types import DynamicBicycleConfig
from mpclab_common.models.dynamics_models import CasadiKinematicBicycle
from mpclab_common.models.model_types import KinematicBicycleConfig
from mpclab_common.track import get_track

from mpclab_visualizations.vis_types import GlobalPlotConfigs, VehiclePlotConfigs
from mpclab_visualizations.barc_plotter_qt import BarcFigure

import pdb

import numpy as np
import casadi as ca

import copy
import time

import matplotlib.pyplot as plt

save_fig = False

# Initial time
t = 0

# Import scenario
track_obj = get_track('Lab_Track_barc')
sim_state = VehicleState(t=0.0, 
                        x=Position(x=0.1, y=0.1),
                        e=OrientationEuler(psi=0.0), 
                        v=BodyLinearVelocity(v_long=1.0, v_tran=0.0),
                        w=BodyAngularVelocity(w_psi=0.0))
# track_obj.local_to_global_typed(sim_state)

half_width = track_obj.half_width

# =============================================
# Set up model
# =============================================
discretization_method = 'rk4'
dt = 0.1
dynamics_config = DynamicBicycleConfig(dt=dt,
                                        model_name='model',
                                        noise=False,
                                        discretization_method=discretization_method,
                                        simple_slip=True)
dyn_model = CasadiDynamicBicycle(t, dynamics_config, track=track_obj)
# dynamics_config = KinematicBicycleConfig(dt=dt,
#                                         model_name='model',
#                                         noise=False,
#                                         discretization_method=discretization_method)
# dyn_model = CasadiKinematicBicycle(t, dynamics_config, track=track_obj)

state_input_ub=VehicleState(x=Position(x=np.ceil(track_obj.track_extents['x_max']), y=np.ceil(track_obj.track_extents['y_max'])),
                            e=OrientationEuler(psi=10),
                            v=BodyLinearVelocity(v_long=4, v_tran=2),
                            w=BodyAngularVelocity(w_psi=7),
                            u=VehicleActuation(u_a=2.0, u_steer=0.35))
state_input_lb=VehicleState(x=Position(x=np.floor(track_obj.track_extents['x_min']), y=np.floor(track_obj.track_extents['y_min'])),
                            e=OrientationEuler(psi=-10),
                            v=BodyLinearVelocity(v_long=-0.1, v_tran=-2),
                            w=BodyAngularVelocity(w_psi=-7),
                            u=VehicleActuation(u_a=-0.1, u_steer=-0.35))
state_input_rate_max=VehicleState(u=VehicleActuation(u_a=10.0, u_steer=10.0))
state_input_rate_min=VehicleState(u=VehicleActuation(u_a=-10.0, u_steer=-10.0))

# =============================================
# MPC controller setup
# =============================================
N = 20
mpc_params = CAMPCCParams(dt=dt, N=N,
                            conv_approx=True,
                            damping=0.75,
                            qp_iters=2,
                            pos_idx=[3, 4],
                            state_scaling=[4, 2, 7, 3, 3, 2*np.pi],
                            input_scaling=[2, 0.35],
                            # state_scaling=None,
                            # input_scaling=None,
                            contouring_cost=0.1,
                            lag_cost=1000.0,
                            performance_cost=1.0,
                            vs_cost=1e-4,
                            vs_rate_cost=1e-3,
                            vs_max=4.0,
                            vs_min=-4.0,
                            vs_rate_max=10.0,
                            vs_rate_min=-10.0,
                            soft_track=False,
                            track_slack_quad=100.0,
                            track_slack_lin=0.0,
                            code_gen=False,
                            opt_flag='O3')

# Symbolic placeholder variables
sym_q = ca.MX.sym('q', dyn_model.n_q)
sym_u = ca.MX.sym('u', dyn_model.n_u)
sym_um = ca.MX.sym('um', dyn_model.n_u)

wz_idx = 2
ua_idx = 0
us_idx = 1

quad_state_cost = 0.5*(1e-5*sym_q[wz_idx]**2)
quad_input_cost = 0.5*(5*(sym_u[ua_idx])**2 + 5*(sym_u[us_idx])**2)

quad_input_rate_cost = 0.5*(10*(sym_u[ua_idx]-sym_um[ua_idx])**2 \
                         + 10*(sym_u[us_idx]-sym_um[us_idx])**2)

# Ego stage cost
sym_stage = quad_input_cost \
            + quad_input_rate_cost \
            + quad_state_cost

quad_state_cost = 0.5*(1e-4*sym_q[wz_idx]**2)

# Ego terminal cost
sym_term = quad_state_cost

# Ego cost functions
sym_costs = []
for k in range(N):
    sym_costs.append(ca.Function(f'stage_{k}', [sym_q, sym_u, sym_um], [sym_stage],
                                [f'q_{k}', f'u_{k}', f'u_{k-1}'], [f'stage_cost_{k}']))
sym_costs.append(ca.Function('term', [sym_q], [sym_term],
                            [f'q_{N}'], ['term_cost']))

# Build symbolic constraints g_i(x, u, um) <= 0
input_rate_constr = ca.vertcat((sym_u[ua_idx]-sym_um[ua_idx]) - dt*state_input_rate_max.u.u_a, 
                                dt*state_input_rate_min.u.u_a - (sym_u[ua_idx]-sym_um[ua_idx]),
                                (sym_u[us_idx]-sym_um[us_idx]) - dt*state_input_rate_max.u.u_steer, 
                                dt*state_input_rate_min.u.u_steer - (sym_u[us_idx]-sym_um[us_idx]))

constrs = []
for k in range(N):
    constrs.append(ca.Function('constrs_%i' % k, [sym_q, sym_u, sym_um], [input_rate_constr]))
    # constrs.append(None)
constrs.append(None)

mpc_controller = CA_MPCC(dyn_model, 
                            sym_costs, 
                            constrs, 
                            {'ub': state_input_ub, 'lb': state_input_lb},
                            mpc_params)

q_ws = [dyn_model.state2q(sim_state)]
u_ws = np.zeros((N, dyn_model.n_u))
s0, _, _ = track_obj.global_to_local((sim_state.x.x, sim_state.x.y, 0))
s_ws = [s0]
vs_ws = sim_state.v.v_long*np.ones(N)
for k in range(N):
    q_ws.append(dyn_model.fd(q_ws[k], u_ws[k]).toarray().squeeze())
    s_ws.append(float(mpc_controller.fs_d(s_ws[k], vs_ws[k])))
q_ws = np.array(q_ws[1:])
s_ws = np.array(s_ws[1:])
mpc_controller.set_warm_start(q_ws, u_ws, s_ws, vs_ws, state=sim_state)
# pdb.set_trace()

# =============================================
# Create visualizer
# =============================================
# Set up plotter
global_plot_params = GlobalPlotConfigs(track_name='Lab_Track_barc', 
                                       draw_period=0.05, 
                                       update_period=0.05, 
                                       state_data_fields=['p.s', 'p.x_tran', 'v.v_long'],
                                       state_units=['m', 'm', 'm/s'],
                                       input_data_fields=['u_a', 'u_steer'],
                                       input_units=['m/s^2', 'rad'],
                                       buffer_length=50)
vehicle_plot_params = VehiclePlotConfigs(name='ego', 
                                         color='b', 
                                         vehicle_draw_L=0.37, 
                                         vehicle_draw_W=0.195, 
                                         show_traces=True, 
                                         show_state=True,
                                         state_topics=['state'],
                                         state_trace_styles=['solid'],
                                         show_input=True,
                                         input_topics=['ecu'],
                                         input_trace_styles=['solid'],
                                         show_pred=True,
                                         pred_topics=['pred'],
                                         show_cov=False)
barc_fig = BarcFigure(t0=t, params=global_plot_params)
barc_fig.add_vehicle(vehicle_plot_params)
barc_fig.run()

# =============================================
# Run race
# =============================================
# Initialize inputs
t = 0.0
sim_state.u.u_a, sim_state.u.u_steer = 0.0, 0.0
pred = VehiclePrediction()
control = VehicleActuation(t=t, u_a=0, u_steer=0)

while True:
    state = copy.deepcopy(sim_state)

    # Solve for car 1 control
    t = time.time()
    mpc_controller.step(state)
    print(time.time() - t)
    pred = mpc_controller.get_prediction()
    pred.t = t

    state.copy_control(control)
    sim_state = copy.deepcopy(state)

    # Update plots
    barc_fig.update('ego', {'state': copy.deepcopy(sim_state), 
                            'ecu': copy.deepcopy(control), 
                            'pred': copy.deepcopy(pred)})

    # Apply control action and advance simulation
    # last_state = copy.deepcopy(sim_state)
    t += dt
    dyn_model.step(sim_state)

    time.sleep(dt)

    pdb.set_trace()

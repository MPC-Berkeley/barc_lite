#!/usr/bin/env python3

from mpclab_controllers.CA_NL_MPC_test import CA_NL_MPC
# from mpclab_controllers.CA_NL_MPC import CA_NL_MPC
from mpclab_controllers.utils.controllerTypes import CANLMPCParams

from mpclab_common.pytypes import VehicleState, VehicleActuation, VehiclePrediction, Position, ParametricPose, BodyLinearVelocity, OrientationEuler, BodyAngularVelocity
from mpclab_common.models.dynamics_models import CasadiCLPointMass, CasadiPointMassCombined
from mpclab_common.models.model_types import PointMassConfig
from mpclab_common.track import get_track

from mpclab_visualizations.vis_types import GlobalPlotConfigs, VehiclePlotConfigs
from mpclab_visualizations.barc_plotter_qt import BarcFigure

import pdb

import numpy as np
import casadi as ca

import multiprocessing as mp

import copy
import matplotlib.pyplot as plt

save_fig = False

# Initial time
t = 0

# Import scenario
track_obj = get_track('Lab_Track_barc')
sim_state = VehicleState(t=0.0, 
                        p=ParametricPose(s=0.5, x_tran=-0.1, e_psi=0), 
                        v=BodyLinearVelocity(v_long=0.1))
track_obj.local_to_global_typed(sim_state)

half_width = track_obj.half_width

# =============================================
# Set up model
# =============================================
discretization_method = 'euler'
dt = 0.1
dynamics_config = PointMassConfig(dt=dt,
                                    model_name='point_mass',
                                    noise=False,
                                    discretization_method=discretization_method,
                                    drag_coefficient=0,
                                    rolling_resistance=0)
# dyn_model = CasadiCLPointMass(t, dynamics_config, track=track_obj)
dyn_model = CasadiPointMassCombined(t, dynamics_config, track=track_obj)

state_input_ub=VehicleState(x=Position(x=np.inf, y=np.inf),
                            p=ParametricPose(s=np.inf, x_tran=track_obj.half_width, e_psi=np.inf),
                            e=OrientationEuler(psi=np.inf),
                            v=BodyLinearVelocity(v_long=np.inf, v_tran=np.inf),
                            w=BodyAngularVelocity(w_psi=np.inf),
                            u=VehicleActuation(u_a=np.inf, u_steer=np.inf))
state_input_lb=VehicleState(x=Position(x=-np.inf, y=-np.inf),
                            p=ParametricPose(s=-np.inf, x_tran=-track_obj.half_width, e_psi=-np.inf),
                            e=OrientationEuler(psi=-np.inf),
                            v=BodyLinearVelocity(v_long=-np.inf, v_tran=-np.inf),
                            w=BodyAngularVelocity(w_psi=-np.inf),
                            u=VehicleActuation(u_a=-np.inf, u_steer=-np.inf))
state_input_rate_max=VehicleState(u=VehicleActuation(u_a=1, u_steer=1))
state_input_rate_min=VehicleState(u=VehicleActuation(u_a=-1, u_steer=-1))

# =============================================
# MPC controller setup
# =============================================
N = 15
nlmpc_params = CANLMPCParams(dt=dt, N=N)

# Symbolic placeholder variables
sym_q = ca.MX.sym('q', dyn_model.n_q)
sym_u = ca.MX.sym('u', dyn_model.n_u)
sym_um = ca.MX.sym('um', dyn_model.n_u)

# s_idx = 2
# ey_idx = 3
s_idx = 4
ey_idx = 5

fx_idx = 0
fy_idx = 1

quad_input_cost = 0.5*(0.1*(sym_u[fx_idx])**2 \
                    + 0.1*(sym_u[fy_idx])**2)

quad_input_rate_cost = 1*(sym_u[fx_idx]-sym_um[fx_idx])**2 \
                         + 1*(sym_u[fy_idx]-sym_um[fy_idx])**2

# Ego stage cost
sym_stage = quad_input_cost #\
            #+ quad_input_rate_cost

comp_cost = -10*sym_q[s_idx]

# Ego terminal cost
sym_term = comp_cost

# Ego cost functions
sym_costs = []
for k in range(N):
    sym_costs.append(ca.Function(f'stage_{k}', [sym_q, sym_u, sym_um], [sym_stage],
                                [f'q_{k}', f'u_{k}', f'u_{k-1}'], [f'stage_cost_{k}']))
sym_costs.append(ca.Function('term', [sym_q], [sym_term],
                            [f'q_{N}'], ['term_cost']))

# Build symbolic constraints g_i(x, u, um) <= 0
input_rate_constr = ca.vertcat((sym_u[fx_idx]-sym_um[fx_idx]) - dt*state_input_rate_max.u.u_a, 
                                dt*state_input_rate_min.u.u_a - (sym_u[fx_idx]-sym_um[fx_idx]),
                                (sym_u[fy_idx]-sym_um[fy_idx]) - dt*state_input_rate_max.u.u_steer, 
                                dt*state_input_rate_min.u.u_steer - (sym_u[fy_idx]-sym_um[fy_idx]))

mu = 0.8
friction_limit = sym_u[fy_idx]**2 + sym_u[fx_idx]**2 - (mu*dyn_model.m*9.81)**2

constrs = []
for k in range(N):
    constrs.append(ca.Function('nl_constrs_%i' % k, [sym_q, sym_u, sym_um], [friction_limit]))
    # constrs.append(None)
constrs.append(None)

nlmpc_controller = CA_NL_MPC(dyn_model, 
                                sym_costs, 
                                constrs, 
                                {'ub': state_input_ub, 'lb': state_input_lb},
                                nlmpc_params)

q_ws = [dyn_model.state2q(sim_state)]
u_ws = np.zeros((N, dyn_model.n_u))
for i in range(N):
    q_ws.append(dyn_model.fd(q_ws[-1], u_ws[i]).toarray().squeeze())
q_ws = np.array(q_ws[1:])
nlmpc_controller.set_warm_start(q_ws, u_ws)
# nlmpc_controller.set_warm_start(u_ws)

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

p = mp.Process(target=barc_fig.run_plotter)
p.start()

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
    nlmpc_controller.step(state)
    pred = nlmpc_controller.get_prediction()
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

    pdb.set_trace()

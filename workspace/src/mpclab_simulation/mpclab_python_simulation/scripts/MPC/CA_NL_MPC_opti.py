#!/usr/bin/env python3

from mpclab_common.pytypes import VehicleState, VehicleActuation, VehiclePrediction, Position, ParametricPose, BodyLinearVelocity, OrientationEuler, BodyAngularVelocity
from mpclab_common.models.dynamics_models import CasadiCLPointMass
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
                        p=ParametricPose(s=0.5, x_tran=-0.2, e_psi=0), 
                        v=BodyLinearVelocity(v_long=0.5))
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
dyn_model = CasadiCLPointMass(t, dynamics_config, track=track_obj)

state_input_ub=VehicleState(x=Position(x=np.inf, y=np.inf),
                            p=ParametricPose(s=np.inf, x_tran=track_obj.half_width, e_psi=np.inf),
                            e=OrientationEuler(psi=np.inf),
                            v=BodyLinearVelocity(v_long=np.inf, v_tran=np.inf),
                            w=BodyAngularVelocity(w_psi=np.inf),
                            u=VehicleActuation(u_a=3, u_steer=3))
state_input_lb=VehicleState(x=Position(x=-np.inf, y=-np.inf),
                            p=ParametricPose(s=-np.inf, x_tran=-track_obj.half_width, e_psi=-np.inf),
                            e=OrientationEuler(psi=-np.inf),
                            v=BodyLinearVelocity(v_long=-np.inf, v_tran=-np.inf),
                            w=BodyAngularVelocity(w_psi=-np.inf),
                            u=VehicleActuation(u_a=-3, u_steer=-3))
state_input_rate_max=VehicleState(u=VehicleActuation(u_a=1, u_steer=1))
state_input_rate_min=VehicleState(u=VehicleActuation(u_a=-1, u_steer=-1))

state_ub, input_ub = dyn_model.state2qu(state_input_ub)
state_lb, input_lb = dyn_model.state2qu(state_input_lb)

# =============================================
# MPC controller setup
# =============================================
N = 15

opti = ca.Opti()
q = opti.variable(dyn_model.n_q, N+1)
u = opti.variable(dyn_model.n_u, N)
q_0 = opti.parameter(dyn_model.n_q)

J = 0
opti.subject_to(q[:,0] == q_0)
for k in range(N):
    J += 0.5*ca.bilin(0.1*np.eye(dyn_model.n_u), u[:,k], u[:,k])
    opti.subject_to(q[:,k+1] == dyn_model.fd(q[:,k], u[:,k]))
    opti.subject_to([q[:,k] <= state_ub, q[:,k] >= state_lb])
    opti.subject_to([u[:,k] <= input_ub, u[:,k] >= input_lb])

J += -10*q[2,-1]
opti.subject_to([q[:,-1] <= state_ub, q[:,-1] >= state_lb])

opti.minimize(J)

solver_opts = {
            "mu_strategy" : "adaptive",
            "mu_init" : 1e-5,
            "mu_min" : 1e-15,
            "barrier_tol_factor" : 1,
            "print_level" : 5,
            "linear_solver" : "ma27"
            }
plugin_opts = {"verbose" : False, "print_time" : False, "print_out" : False}
opti.solver('ipopt', plugin_opts, solver_opts)

# pdb.set_trace()

q_ws = [dyn_model.state2q(sim_state)]
u_ws = np.zeros((N, dyn_model.n_u))
for i in range(N):
    q_ws.append(dyn_model.fd(q_ws[-1], u_ws[i]).toarray().squeeze())
q_ws = np.array(q_ws)
opti.set_initial(q, q_ws.T)
opti.set_initial(u, u_ws.T)


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
                                         state_topics=['sim_state'],
                                         state_trace_styles=['solid'],
                                         show_input=True,
                                         input_topics=['ecu'],
                                         input_trace_styles=['solid'],
                                         show_pred=True,
                                         pred_topics=['pred_state'],
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
    opti.set_value(q_0, dyn_model.state2q(state))
    opti.solve()

    q_pred = opti.value(q)
    u_pred = opti.value(u)
    pred = dyn_model.qu2prediction(None, q_pred.T, u_pred.T)
    pred.t = t
    state.u.u_a, state.u.u_steer = u_pred[0,0], u_pred[1,0]

    state.copy_control(control)
    sim_state = copy.deepcopy(state)

    # print(control)
    # print(pred)

    # Update plots
    barc_fig.update('ego', {'sim_state': copy.deepcopy(sim_state), 'ecu': copy.deepcopy(control), 'pred_state': copy.deepcopy(pred)})

    # Apply control action and advance simulation
    # last_state = copy.deepcopy(sim_state)
    t += dt
    dyn_model.step(sim_state)

    pdb.set_trace()

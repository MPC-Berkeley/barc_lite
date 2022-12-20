#!/usr/bin/env python3

from mpclab_controllers.CA_MPCC_conv import CA_MPCC_conv
from mpclab_controllers.utils.controllerTypes import CAMPCCParams

from mpclab_simulation.dynamics_simulator import DynamicsSimulator

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

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.transforms import Affine2D

import copy
import time

# Initial time
t = 0

# Import scenario
# track_obj = get_track('Lab_Track_barc')
track_obj = get_track('L_track_barc')
sim_state = VehicleState(t=0.0, 
                        x=Position(x=0.5, y=0.0),
                        e=OrientationEuler(psi=0), 
                        v=BodyLinearVelocity(v_long=1, v_tran=0),
                        w=BodyAngularVelocity(w_psi=0))
# track_obj.local_to_global_typed(sim_state)

half_width = track_obj.half_width

# =============================================
# Set up model
# =============================================
discretization_method = 'rk4'
dt = 0.1
dynamics_config = KinematicBicycleConfig(dt=dt,
                                        model_name='kinematic_bicycle',
                                        noise=False,
                                        discretization_method=discretization_method,
                                        mass=2.2187)
dyn_model = CasadiKinematicBicycle(t, dynamics_config, track=track_obj)

sim_dynamics_config = KinematicBicycleConfig(dt=0.01,
                                        model_name='kinematic_bicycle',
                                        noise=False,
                                        discretization_method=discretization_method,
                                        mass=2.2187)
dynamics_simulator = DynamicsSimulator(t, sim_dynamics_config, delay=[0.1, 0.1], track=track_obj)

state_input_ub=VehicleState(x=Position(x=6, y=6),
                            e=OrientationEuler(psi=10),
                            v=BodyLinearVelocity(v_long=4),
                            u=VehicleActuation(u_a=2.0, u_steer=0.45))
state_input_lb=VehicleState(x=Position(x=-3, y=-3),
                            e=OrientationEuler(psi=-10),
                            v=BodyLinearVelocity(v_long=-0.1),
                            u=VehicleActuation(u_a=-2.0, u_steer=-0.45))
input_rate_max=VehicleState(u=VehicleActuation(u_a=20.0, u_steer=4.5))
input_rate_min=VehicleState(u=VehicleActuation(u_a=-20.0, u_steer=-4.5))

# =============================================
# MPC controller setup
# =============================================
N = 20
mpc_params = CAMPCCParams(dt=dt, N=N,
                            verbose=True,
                            debug_plot=False,
                            damping=0.75,
                            qp_iters=2,
                            pos_idx=[0, 1],
                            state_scaling=[6, 6, 4, 2*np.pi],
                            input_scaling=[2, 0.45],
                            delay=[1, 1],
                            # delay=None,
                            contouring_cost=0.1,
                            contouring_cost_N=1.0,
                            lag_cost=1000.0,
                            lag_cost_N=1000.0,
                            performance_cost=0.02,
                            vs_cost=1e-4,
                            vs_rate_cost=1e-3,
                            vs_max=5.0,
                            vs_min=0.0,
                            vs_rate_max=5.0,
                            vs_rate_min=-5.0,
                            soft_track=True,
                            track_slack_quad=100.0,
                            track_slack_lin=25.0,
                            code_gen=False,
                            opt_flag='O3',
                            solver_name='MPCC_conv')

# Symbolic placeholder variables
sym_q = ca.MX.sym('q', dyn_model.n_q)
sym_u = ca.MX.sym('u', dyn_model.n_u)
sym_du = ca.MX.sym('du', dyn_model.n_u)

# wz_idx = 2
ua_idx = 0
us_idx = 1

# sym_state_stage = 0.5*(1e-5*sym_q[wz_idx]**2)
# sym_state_term = 0.5*(1e-4*sym_q[wz_idx]**2)

sym_input_stage = 0.5*(1e-4*(sym_u[ua_idx])**2 + 1e-4*(sym_u[us_idx])**2)
sym_input_term = 0.5*(1e-4*(sym_u[ua_idx])**2 + 1e-4*(sym_u[us_idx])**2)

sym_rate_stage = 0.5*(0.01*(sym_du[ua_idx])**2 + 1*(sym_du[us_idx])**2)

sym_costs = {'state': [None for _ in range(N+1)], 'input': [None for _ in range(N+1)], 'rate': [None for _ in range(N)]}
for k in range(N):
    # sym_costs['state'][k] = ca.Function(f'state_stage_{k}', [sym_q], [sym_state_stage])
    sym_costs['input'][k] = ca.Function(f'input_stage_{k}', [sym_u], [sym_input_stage])
    sym_costs['rate'][k] = ca.Function(f'rate_stage_{k}', [sym_du], [sym_rate_stage])
# sym_costs['state'][N] = ca.Function('state_term', [sym_q], [sym_state_term])
sym_costs['input'][N] = ca.Function('input_term', [sym_u], [sym_input_term])

# a_max = dynamics_config.gravity*dynamics_config.wheel_friction
# sym_ax, sym_ay = dyn_model.f_a(sym_q, sym_u)
# friction_circle_constraint = ca.Function('friction_circle', [sym_q, sym_u], [sym_ax**2 + sym_ay**2 - a_max**2])

sym_constrs = {'state_input': [None for _ in range(N+1)],
                'rate': [None for _ in range(N)]}

# qp_interface options: hpipm, casadi, cvxpy
mpc_controller = CA_MPCC_conv(dyn_model, 
                            sym_costs, 
                            sym_constrs, 
                            {'qu_ub': state_input_ub, 'qu_lb': state_input_lb, 'du_ub': input_rate_max, 'du_lb': input_rate_min},
                            mpc_params,
                            qp_interface='hpipm')

u_ws = np.zeros((N+1, dyn_model.n_u))
# u_ws = 1e-3*np.ones((N+1, dyn_model.n_u))
vs_ws = np.zeros(N+1)
du_ws = np.zeros((N, dyn_model.n_u))
dvs_ws = np.zeros(N)
mpc_controller.set_warm_start(u_ws, vs_ws, du_ws, dvs_ws, state=sim_state)
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
# barc_fig = BarcFigure(t0=t, params=global_plot_params)
# barc_fig.add_vehicle(vehicle_plot_params)
# barc_fig.run()

plot = True

if plot:
    from collections import deque
    plt.ion()
    fig = plt.figure()
    ax_xy = fig.add_subplot(1,2,1)
    ax_a = fig.add_subplot(1,2,2)
    track_obj.plot_map(ax_xy)
    ax_xy.set_aspect('equal')
    l_pred = ax_xy.plot([], [], 'b-o', markersize=4)[0]
    l_ref = ax_xy.plot([], [], 'ko', markersize=4)[0]
    VL = vehicle_plot_params.vehicle_draw_L
    VW = vehicle_plot_params.vehicle_draw_W
    rect = patches.Rectangle((-0.5*VL, -0.5*VW), VL, VW, linestyle='solid', color='b', alpha=0.5)
    ax_xy.add_patch(rect)
    
    # fc_x = a_max*np.cos(np.linspace(0, 2*np.pi, 100))
    # fc_y = a_max*np.sin(np.linspace(0, 2*np.pi, 100))
    # ax_a.plot(fc_x, fc_y, 'r')
    # s_a = ax_a.scatter([], [], s=5, c=[], cmap='plasma')
    # ax_hist = deque(maxlen=10)
    # ay_hist = deque(maxlen=10)
    # ax_a.set_aspect('equal')
    fig.canvas.draw()
    fig.canvas.flush_events()

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
    st = time.time()
    mpc_controller.step(state)
    print('Controller solve time: ' + str(time.time()-st))
    pred = mpc_controller.get_prediction()
    pred.t = t

    state.copy_control(control)
    sim_state = copy.deepcopy(state)

    # Update plots
    # barc_fig.update('ego', {'state': copy.deepcopy(sim_state), 
    #                         'ecu': copy.deepcopy(control), 
    #                         'pred': copy.deepcopy(pred)})

    if plot:
        x, y, psi = track_obj.local_to_global((state.p.s, state.p.x_tran, state.p.e_psi))
        b_left = x - VL/2
        b_bot  = y - VW/2
        r = Affine2D().rotate_around(x, y, psi) + ax_xy.transData
        rect.set_xy((b_left,b_bot))
        rect.set_transform(r)
        pred_x, pred_y, ref_x, ref_y = [], [], [], []
        for i in range(len(pred.x)):
            pred_x.append(pred.x[i])
            pred_y.append(pred.y[i])
        l_pred.set_data(pred_x, pred_y)
        l_ref.set_data(ref_x, ref_y)
        # ax, ay = dyn_model.f_a(*dyn_model.state2qu(sim_state))
        # ax_hist.append(float(ax))
        # ay_hist.append(float(ay))
        # s_a.set_offsets(np.array([ax_hist, ay_hist]).T)
        # s_a.set_array(np.linspace(1, 0, 10))
        # ax_a.relim()
        # ax_a.autoscale_view()
        fig.canvas.draw()
        fig.canvas.flush_events()

    # Apply control action and advance simulation
    # last_state = copy.deepcopy(sim_state)
    t += dt
    # dyn_model.step(sim_state)
    dynamics_simulator.step(sim_state, T=dt)

    # time.sleep(dt)

    pdb.set_trace()

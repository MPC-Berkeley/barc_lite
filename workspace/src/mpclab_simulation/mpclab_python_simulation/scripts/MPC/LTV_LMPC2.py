#!/usr/bin/env python3

from mpclab_controllers.PID import PIDLaneFollower
from mpclab_controllers.LTV_LMPC2 import LTV_LMPC
from mpclab_controllers.utils.controllerTypes import LTVLMPC2Params, PIDParams

from mpclab_simulation.dynamics_simulator import DynamicsSimulator

from mpclab_common.models.dynamics_models import CasadiDynamicCLBicycle
from mpclab_common.models.model_types import DynamicBicycleConfig
from mpclab_common.pytypes import VehicleState, VehicleActuation, VehiclePrediction, Position, ParametricPose, BodyLinearVelocity, OrientationEuler, BodyAngularVelocity
from mpclab_common.track import get_track
# from mpclab_common.rosbag_utils import rosbagData

from mpclab_visualizations.barc_plotter_qt import BarcFigure
from mpclab_visualizations.vis_types import GlobalPlotConfigs, VehiclePlotConfigs

import pdb

import numpy as np
import casadi as ca

import time
import copy

# import matplotlib
# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.transforms import Affine2D

track_obj = get_track('L_track_barc')
L = track_obj.track_length
H = track_obj.half_width
sim_state = VehicleState(t=0.0, 
                        p=ParametricPose(s=0.1, x_tran=0),
                        e=OrientationEuler(psi=0), 
                        v=BodyLinearVelocity(v_long=0.5, v_tran=0),
                        w=BodyAngularVelocity(w_psi=0))
track_obj.local_to_global_typed(sim_state)


# =============================================
# Set up model
# =============================================
t = 0.0
discretization_method = 'rk4'
dt = 0.1
dynamics_config = DynamicBicycleConfig(dt=dt,
                                        model_name='dynamic_bicycle',
                                        noise=False,
                                        discretization_method=discretization_method,
                                        simple_slip=False,
                                        tire_model='pacejka',
                                        mass=2.2187,
                                        yaw_inertia=0.02723,
                                        wheel_friction=0.5,
                                        pacejka_b_front=20,
                                        pacejka_b_rear=20,
                                        pacejka_c_front=1,
                                        pacejka_c_rear=1)
dyn_model = CasadiDynamicCLBicycle(t, dynamics_config, track=track_obj)

sim_dynamics_config = DynamicBicycleConfig(dt=0.01,
                                        model_name='dynamic_bicycle',
                                        noise=False,
                                        discretization_method=discretization_method,
                                        simple_slip=False,
                                        tire_model='pacejka',
                                        mass=2.2187,
                                        yaw_inertia=0.02723,
                                        wheel_friction=0.5,
                                        pacejka_b_front=20,
                                        pacejka_b_rear=20,
                                        pacejka_c_front=1,
                                        pacejka_c_rear=1)
# dynamics_simulator = DynamicsSimulator(t, sim_dynamics_config, delay=[0.1, 0.1], track=track_obj)
dynamics_simulator = DynamicsSimulator(t, sim_dynamics_config, delay=None, track=track_obj)

state_input_ub = VehicleState(p=ParametricPose(s=2*L, x_tran=H-0.17, e_psi=100),
                              v=BodyLinearVelocity(v_long=10, v_tran=10),
                              w=BodyAngularVelocity(w_psi=10),
                              u=VehicleActuation(u_a=2.0, u_steer=0.436))
state_input_lb = VehicleState(p=ParametricPose(s=-2*L, x_tran=-(H-0.17), e_psi=-100),
                              v=BodyLinearVelocity(v_long=-10, v_tran=-10),
                              w=BodyAngularVelocity(w_psi=-10),
                              u=VehicleActuation(u_a=-2.0, u_steer=-0.436))
input_rate_ub = VehicleState(u=VehicleActuation(u_a=20.0, u_steer=4.5))
input_rate_lb = VehicleState(u=VehicleActuation(u_a=-20.0, u_steer=-4.5))

# Set up PID controller
noise = True
pid_steer_params = PIDParams(dt=dt, 
                             Kp=0.5, 
                             u_max=state_input_ub.u.u_steer, 
                             u_min=state_input_lb.u.u_steer, 
                             du_max=input_rate_ub.u.u_steer, 
                             du_min=input_rate_lb.u.u_steer, 
                             x_ref=0.0,
                             noise=noise, 
                             noise_max=0.2, 
                             noise_min=-0.2)
pid_speed_params = PIDParams(dt=dt, 
                             Kp=1.5, 
                             u_max=state_input_ub.u.u_a, 
                             u_min=state_input_lb.u.u_a, 
                             du_max=input_rate_ub.u.u_a, 
                             du_min=input_rate_lb.u.u_a, 
                             x_ref=1.0,
                             noise=noise, 
                             noise_max=0.9, 
                             noise_min=-0.9)
pid_controller = PIDLaneFollower(dt, pid_steer_params, pid_speed_params)

# Set up LMPC controller
N = 15
n_ss_pts = 48
n_ss_its = 4
mpc_params = LTVLMPC2Params(dt=dt,
                            N=N,
                            state_scaling=None,
                            input_scaling=None,
                            delay=[1, 1],
                            # convex_hull_slack_quad=[100, 1, 10, 1, 100, 10],
                            convex_hull_slack_quad=[500, 500, 500, 500, 500, 500],
                            convex_hull_slack_lin=[0, 0, 0, 0, 0, 0],
                            soft_state_bound_idxs=[5],
                            # soft_state_bound_idxs=None,
                            soft_state_bound_quad=[5],
                            soft_state_bound_lin=[25],
                            n_ss_pts=n_ss_pts, 
                            n_ss_its=n_ss_its,
                            regression_regularization=0.0,
                            regression_state_out_idxs=[[0], [1], [2]],
                            regression_state_in_idxs=[[0, 1, 2], [0, 1, 2], [0, 1, 2]],
                            regression_input_in_idxs=[[0], [1], [1]],
                            nearest_neighbor_weights=[[1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1]],
                            nearest_neighbor_bw=5.0,
                            nearest_neighbor_max_points=25,
                            wrapped_state_idxs=[4],
                            wrapped_state_periods=[L],
                            debug_plot=False,
                            verbose=True,
                            keep_init_safe_set=True)

# Symbolic placeholder variables
sym_q = ca.MX.sym('q', dyn_model.n_q)
sym_u = ca.MX.sym('u', dyn_model.n_u)
sym_du = ca.MX.sym('du', dyn_model.n_u)

ua_idx = 0
us_idx = 1

sym_input_stage = 0.5*(1*(sym_u[ua_idx])**2 + 1*(sym_u[us_idx])**2)
sym_input_term = 0.5*(1*(sym_u[ua_idx])**2 + 1*(sym_u[us_idx])**2)

sym_rate_stage = 0.5*(5*(sym_du[ua_idx])**2 + 5*(sym_du[us_idx])**2)

sym_costs = {'state': [None for _ in range(N+1)], 'input': [None for _ in range(N+1)], 'rate': [None for _ in range(N)]}
for k in range(N):
    sym_costs['input'][k] = ca.Function(f'input_stage_{k}', [sym_u], [sym_input_stage])
    sym_costs['rate'][k] = ca.Function(f'rate_stage_{k}', [sym_du], [sym_rate_stage])
sym_costs['input'][N] = ca.Function('input_term', [sym_u], [sym_input_term])

sym_constrs = {'state_input': [None for _ in range(N+1)], 
                'rate': [None for _ in range(N)]}

lmpc_controller = LTV_LMPC(dyn_model, 
                            sym_costs,
                            sym_constrs,
                            {'qu_ub': state_input_ub, 'qu_lb': state_input_lb, 'du_ub': input_rate_ub, 'du_lb': input_rate_lb},
                            control_params=mpc_params,
                            qp_interface='casadi')

# =============================================
# Create visualizer
# =============================================
# Set up plotter
global_plot_params = GlobalPlotConfigs(track_name='L_track_barc', 
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
                                         show_cov=False,
                                         show_point_set=True,
                                         point_set_topics=['ss'])
# barc_fig = BarcFigure(t0=t, params=global_plot_params)
# barc_fig.add_vehicle(vehicle_plot_params)
# barc_fig.run()

sim_state.u.u_a, sim_state.u.u_steer = 0.0, 0.0
pred = VehiclePrediction()
ss = VehiclePrediction()
control = VehicleActuation(t=t, u_a=0, u_steer=0)

# pdb.set_trace()

# Run initialization laps with PID controller
# n_init = 1
n_init = n_ss_its
lap_no = 0
lap_data = []
last_lap_data = None
last_lap_value = None
last_state = copy.deepcopy(sim_state)
while True:
    state = copy.deepcopy(sim_state)
    
    # Compute control action
    # track_obj.global_to_local_typed(state)
    pid_controller.step(state)
    state.copy_control(control)
    lap_data.append(copy.deepcopy(state))
                            
    # Apply control action and advance simulation
    t += dt
    sim_state = copy.deepcopy(state)
    dynamics_simulator.step(sim_state, T=dt)
    track_obj.global_to_local_typed(sim_state)
    sim_state.p.s = np.mod(sim_state.p.s, L)
    
    if sim_state.p.s - last_state.p.s < 0:
        print(f'Initialization lap {lap_no} completed')
        lmpc_controller.add_iter_data(lap_data)
        lap_c2g = np.flip(np.arange(len(lap_data)))
        lmpc_controller.add_safe_set_data(lap_data, lap_c2g)
        if last_lap_data:
            for i, d in enumerate(copy.deepcopy(lap_data)):
                d.p.s += track_obj.track_length
                last_lap_data.append(d)
                last_lap_c2g = np.append(last_lap_c2g, -(i+1))
            lmpc_controller.add_safe_set_data(last_lap_data, last_lap_c2g, iter_idx=lap_no-1)
        last_lap_data = copy.deepcopy(lap_data)
        last_lap_c2g = copy.deepcopy(lap_c2g)
        lap_data = []
        lap_no += 1

    if lap_no >= n_init:
        print('Initialization laps complete')
        break

    last_state = copy.deepcopy(sim_state)

u_ws = np.zeros((N+1, dyn_model.n_u))
# u_ws = np.tile(dyn_model.input2u(control), (N+1, 1))
du_ws = np.zeros((N, dyn_model.n_u))
lmpc_controller.set_warm_start(u_ws, du_ws)

plot = True
if plot:
    plt.ion()
    fig = plt.figure()
    ax_xy = fig.add_subplot(1,2,1)
    ax_a = fig.add_subplot(2,2,2)
    ax_d = fig.add_subplot(2,2,4)
    track_obj.plot_map(ax_xy)
    ax_xy.set_aspect('equal')
    l_pred = ax_xy.plot([], [], 'b-o', markersize=4)[0]
    l_ss = ax_xy.plot([], [], 'rs', markersize=4, markerfacecolor='None')[0]
    l_a = ax_a.plot([], [], '-bo')[0]
    l_d = ax_d.plot([], [], '-bo')[0]
    VL = vehicle_plot_params.vehicle_draw_L
    VW = vehicle_plot_params.vehicle_draw_W
    rect = patches.Rectangle((-0.5*VL, -0.5*VW), VL, VW, linestyle='solid', color='b', alpha=0.5)
    ax_xy.add_patch(rect)
    fig.canvas.draw()
    fig.canvas.flush_events()

# Run LMPC
lap_data = []
lap_speed = []
n_laps = 100
lap_start = t
while True:
    state = copy.deepcopy(sim_state)
    last_state = copy.copy(sim_state)

    # Compute control action
    # track_obj.global_to_local_typed(state)
    # st = time.time()
    lmpc_controller.step(state)
    # print('LMPC solve time: ' + str(time.time()-st))
    state.copy_control(control)

    lap_data.append(copy.copy(state))

    pred = lmpc_controller.get_prediction()
    ss = lmpc_controller.get_safe_set()

    if plot:
        x, y, psi = track_obj.local_to_global((state.p.s, state.p.x_tran, state.p.e_psi))
        b_left = x - VL/2
        b_bot  = y - VW/2
        r = Affine2D().rotate_around(x, y, psi) + ax_xy.transData
        rect.set_xy((b_left,b_bot))
        rect.set_transform(r)
        pred_x, pred_y, ss_x, ss_y = [], [], [], []
        for i in range(len(pred.s)):
            x, y, psi = track_obj.local_to_global((pred.s[i], pred.x_tran[i], pred.e_psi[i]))
            pred_x.append(x)
            pred_y.append(y)
        for i in range(len(ss.s)):
            x, y, psi = track_obj.local_to_global((ss.s[i], ss.x_tran[i], ss.e_psi[i]))
            ss_x.append(x)
            ss_y.append(y)
        l_pred.set_data(pred_x, pred_y)
        l_ss.set_data(ss_x, ss_y)
        l_a.set_data(np.arange(N), pred.u_a)
        l_d.set_data(np.arange(N), pred.u_steer)
        ax_a.relim()
        ax_a.autoscale_view()
        ax_d.relim()
        ax_d.autoscale_view()
        fig.canvas.draw()
        fig.canvas.flush_events()

    # Update plots
    # barc_fig.update('ego', {'state':    copy.deepcopy(sim_state), 
    #                         'ecu':      copy.deepcopy(control), 
    #                         'pred':     copy.deepcopy(pred),
    #                         'ss':       copy.deepcopy(ss)})

    # Apply control action and advance simulation
    t += dt
    sim_state = copy.deepcopy(state)
    lap_speed.append(np.sqrt(sim_state.v.v_long**2+sim_state.v.v_tran**2))
    sim_state.p.s = np.mod(sim_state.p.s, L)
    dynamics_simulator.step(sim_state, T=dt)

    # print(lmpc_controller.controller.uPred)
    # if sim_state.p.s < 0.5 or sim_state.p.s > L - 0.5:
    #     pdb.set_trace()

    if sim_state.p.s - last_state.p.s < 0 and t - lap_start > 3:
        print(f'LMPC lap {lap_no} finished in {(t-lap_start):.2f} s, avg v: {np.mean(lap_speed):.2f} m/s, max v {np.amax(lap_speed):.2f} m/s')
        lmpc_controller.add_iter_data(lap_data)
        lap_c2g = np.flip(np.arange(len(lap_data)))
        lmpc_controller.add_safe_set_data(lap_data, lap_c2g)
        lap_data = []
        lap_speed = []
        lap_no += 1
        lap_start = t

    if lap_no >= n_laps:
        break

    # time.sleep(0.01)

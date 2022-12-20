#!/usr/bin/env python3

from mpclab_controllers.PID import PIDLaneFollower
from mpclab_controllers.LTV_LMPC import LTV_LMPC
from mpclab_controllers.utils.controllerTypes import LTVLMPCParams, PIDParams

from mpclab_common.models.dynamics_models import CasadiDynamicCLBicycle
from mpclab_common.models.model_types import DynamicBicycleConfig

from mpclab_visualizations.barc_plotter_qt import BarcFigure
from mpclab_visualizations.vis_types import GlobalPlotConfigs, VehiclePlotConfigs

from mpclab_common.pytypes import VehicleState, VehicleActuation, VehiclePrediction, Position, ParametricPose, BodyLinearVelocity, OrientationEuler, BodyAngularVelocity
from mpclab_common.track import get_track
# from mpclab_common.rosbag_utils import rosbagData

import pdb

import numpy as np

import time
import copy

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.transforms import Affine2D

track_obj = get_track('L_track_barc')
L = track_obj.track_length
H = track_obj.half_width
sim_state = VehicleState(t=0.0, 
                        p=ParametricPose(s=0.1, x_tran=0),
                        e=OrientationEuler(psi=0), 
                        v=BodyLinearVelocity(v_long=0.1, v_tran=0),
                        w=BodyAngularVelocity(w_psi=0))
# track_obj.global_to_local_typed(sim_state)


# =============================================
# Set up model
# =============================================
t = 0.0
discretization_method = 'rk4'
dt = 0.1
dynamics_config = DynamicBicycleConfig(dt=dt,
                                        model_name='model',
                                        noise=False,
                                        discretization_method=discretization_method,
                                        simple_slip=True)
dyn_model = CasadiDynamicCLBicycle(t, dynamics_config, track=track_obj)

state_input_ub = VehicleState(p=ParametricPose(s=2*L, x_tran=H, e_psi=100),
                              v=BodyLinearVelocity(v_long=10, v_tran=10),
                              w=BodyAngularVelocity(w_psi=10),
                              u=VehicleActuation(u_a=2.0, u_steer=0.45))
state_input_lb = VehicleState(p=ParametricPose(s=-2*L, x_tran=-H, e_psi=-100),
                              v=BodyLinearVelocity(v_long=-10, v_tran=-10),
                              w=BodyAngularVelocity(w_psi=-10),
                              u=VehicleActuation(u_a=-2.0, u_steer=-0.45))
input_rate_ub = VehicleState(u=VehicleActuation(u_a=2.0, u_steer=1.0))
input_rate_lb = VehicleState(u=VehicleActuation(u_a=-2.0, u_steer=-1.0))

# Set up PID controller
noise = False
pid_steer_params = PIDParams(dt=dt, 
                             Kp=0.5, 
                             u_max=state_input_ub.u.u_steer, 
                             u_min=state_input_lb.u.u_steer, 
                             du_max=input_rate_ub.u.u_steer, 
                             du_min=input_rate_lb.u.u_steer, 
                             noise=noise, 
                             noise_max=0.2, 
                             noise_min=-0.2)
pid_speed_params = PIDParams(dt=dt, 
                             Kp=1.5, 
                             u_max=state_input_ub.u.u_a, 
                             u_min=state_input_lb.u.u_a, 
                             du_max=input_rate_ub.u.u_a, 
                             du_min=input_rate_lb.u.u_a, 
                             noise=noise, 
                             noise_max=0.9, 
                             noise_min=-0.9)
v_ref = 1.0
x_ref = 0
pid_controller = PIDLaneFollower(v_ref, x_ref, dt, pid_steer_params, pid_speed_params)

# Set up LMPC controller
N = 15
n_ss_pts = 48
n_ss_its = 4
mpc_params = LTVLMPCParams(dt=dt, n=dyn_model.n_q, d=dyn_model.n_u,
                            N=N,
                            n_ss_pts=n_ss_pts, n_ss_its=n_ss_its,
                            Q=[0, 0, 0, 0, 0, 0],
                            Q_f=[0, 0, 0, 0, 0, 0],
                            R=[1, 1],
                            R_d=[5, 5],
                            Q_slack=[500, 500, 500, 500, 500, 500],
                            Q_lane=[5, 25],
                            u_steer_max=0.436,
                            u_steer_min=-0.436,
                            u_a_max=2.0,
                            u_a_min=-2.0,
                            time_varying=True,
                            regression_regularization=0.0,
                            safe_set_init_data_file=None,
                            safe_set_topic='/experiment/barc_1/closed_loop_traj')
mpc_params.vectorize_constraints()

# Load closed-loop data to initialize safe set
# state_names = ['v_long', 'v_tran', 'psidot', 'e_psi', 's', 'x_tran']
# input_names = ['u_a', 'u_steer']
# if mpc_params.safe_set_init_data_file:
#     rb_data = rosbagData(mpc_params.safe_set_init_data_file)
#     attributes = state_names + input_names + ['lap_num']
#     ss_data = rb_data.read_from_topic_to_numpy_array(mpc_params.safe_set_topic, attributes)

#     first_nonzero_idx = np.where(np.any(ss_data[:,:mpc_params.n], axis=1))[0][0]
#     ss_data = ss_data[first_nonzero_idx:]

#     lap = 0
#     init_ss_pts, lap_ss_pts = [], []
#     # Skip the last lap because the data from that is typically incomplete
#     for i in range(ss_data.shape[0]):
#         if ss_data[i,-1] > lap:
#             # self.get_logger().info('===== Lap with %i data points added to LMPC safe set =====' % (len(lap_ss_pts)))
#             init_ss_pts.append(lap_ss_pts)
#             lap_ss_pts = []
#             lap += 1

#             if lap == ss_data[-1,-1]:
#                 break

#         x = ss_data[i,:mpc_params.n]
#         u = ss_data[i,mpc_params.n:mpc_params.n+mpc_params.d]
#         state = VehicleState()
#         dyn_model.qu2state(state, x, u)
#         lap_ss_pts.append(state)

lmpc_controller = LTV_LMPC(dyn_model, track_obj, control_params=mpc_params)

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
barc_fig = BarcFigure(t0=t, params=global_plot_params)
barc_fig.add_vehicle(vehicle_plot_params)
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
init_laps = []
while True:
    state = copy.deepcopy(sim_state)
    last_state = copy.deepcopy(sim_state)

    # Compute control action
    # track_obj.global_to_local_typed(state)
    pid_controller.step(state)
    state.copy_control(control)
    lap_data.append(copy.deepcopy(state))
    
    # Apply control action and advance simulation
    t += dt
    sim_state = copy.deepcopy(state)
    sim_state.p.s = np.mod(sim_state.p.s, track_obj.track_length)
    dyn_model.step(sim_state)

    if sim_state.p.s - last_state.p.s < 0:
        print(f'Initialization lap {lap_no} completed')
        init_laps.append(lap_data)
        lap_data = []
        lap_no += 1

    if lap_no >= n_init:
        print('Initialization laps complete')
        break
    
    # pdb.set_trace()
    # time.sleep(0.1)
# pdb.set_trace()

# Initialize safe set with loaded data
lmpc_controller.initialize(init_laps)
# Add PID laps to safe set
# for l in init_laps:
#     lmpc_controller.add_trajectory(l)

# Initialize safe set with copies of first lap
# lmpc_controller.initialize([init_laps[0] for _ in range(n_ss_its)])

# with open('pid_traj.pkl', 'rb') as f:
#     init_lap = pickle.load(f)
# lmpc_controller.initialize([init_lap for _ in range(n_ss_its)])
# x_N, u_N = dyn_model.state2qu(init_lap[mpc_params.N])
# lmpc_controller.set_terminal_set_sample_point(x_N, np.flip(u_N))

# Warm start linearization with first N points from last PID lap
u_seq = np.zeros((N, dyn_model.n_u))
x_seq = [dyn_model.state2q(sim_state)]
for k in range(u_seq.shape[0]):
    x_seq.append(dyn_model.fd(x_seq[k], u_seq[k]).toarray().squeeze())
x_seq = np.array(x_seq)
lmpc_controller.set_warm_start(x_seq, u_seq)
lmpc_controller.set_terminal_set_sample_point(x_seq[-1], u_seq[-1])

# pdb.set_trace()
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
n_laps = 100
lap_start = t
while True:
    state = copy.deepcopy(sim_state)
    last_state = copy.copy(sim_state)

    # Compute control action
    # track_obj.global_to_local_typed(state)
    st = time.time()
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
    sim_state.p.s = np.mod(sim_state.p.s, track_obj.track_length)
    dyn_model.step(sim_state)

    # print(lmpc_controller.controller.uPred)
    # pdb.set_trace()

    if sim_state.p.s - last_state.p.s < 0 and t - lap_start > 3:
        print(f'LMPC lap {lap_no} finished in {t-lap_start} s')
        lmpc_controller.add_trajectory(lap_data)
        lap_data = []
        lap_no += 1
        lap_start = t

    if lap_no >= n_laps:
        break

    # time.sleep(0.01)

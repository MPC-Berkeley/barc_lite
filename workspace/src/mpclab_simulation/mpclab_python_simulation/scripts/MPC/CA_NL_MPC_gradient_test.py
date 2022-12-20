#!/usr/bin/env python3

# from mpclab_controllers.CA_NL_MPC import CA_NL_MPC
from mpclab_controllers.CA_NL_MPC_test_batch import CA_NL_MPC

from mpclab_common.pytypes import VehicleState, VehicleActuation, Position, ParametricPose, OrientationEuler, BodyLinearVelocity, BodyAngularVelocity
from mpclab_common.models.dynamics_models import CasadiKinematicBicycleCombined
from mpclab_common.models.model_types import KinematicBicycleConfig

from mpclab_visualizations.vis_types import GlobalPlotConfigs, VehiclePlotConfigs
from mpclab_visualizations.barc_plotter_qt import BarcFigure

import pdb

import numpy as np
import scipy as sp
import casadi as ca

import multiprocessing as mp

import copy
import matplotlib.pyplot as plt

save_fig = False

# Initial time
t = 0

# Import scenario
from CA_NL_MPC_scenarios import scenario_1 as scenario

track_obj = scenario.track
half_width = track_obj.half_width

# =============================================
# Set up joint model
# =============================================
discretization_method = 'rk4'
dt = scenario.control_params.dt
dynamics_config = KinematicBicycleConfig(dt=dt,
                                         model_name='kinematic_bicycle',
                                         noise=False,
                                         discretization_method=discretization_method,
                                         wheel_dist_front=0.13,
                                         wheel_dist_rear=0.13,
                                         code_gen=False)
ego_dyn_model = CasadiKinematicBicycleCombined(t, dynamics_config, track=track_obj)
tar_dyn_model = CasadiKinematicBicycleCombined(t, dynamics_config, track=track_obj)

state_input_ub=VehicleState(x=Position(x=np.inf, y=np.inf),
                            p=ParametricPose(s=np.inf, x_tran=track_obj.half_width, e_psi=np.inf),
                            e=OrientationEuler(psi=np.inf),
                            v=BodyLinearVelocity(v_long=np.inf, v_tran=np.inf),
                            w=BodyAngularVelocity(w_psi=np.inf),
                            u=VehicleActuation(u_a=2.0, u_steer=0.436))
state_input_lb=VehicleState(x=Position(x=-np.inf, y=-np.inf),
                            p=ParametricPose(s=-np.inf, x_tran=-track_obj.half_width, e_psi=-np.inf),
                            e=OrientationEuler(psi=-np.inf),
                            v=BodyLinearVelocity(v_long=-np.inf, v_tran=-np.inf),
                            w=BodyAngularVelocity(w_psi=-np.inf),
                            u=VehicleActuation(u_a=-2.0, u_steer=-0.436))
state_input_rate_max=VehicleState(u=VehicleActuation(u_a=10.0, u_steer=np.pi))
state_input_rate_min=VehicleState(u=VehicleActuation(u_a=-10.0, u_steer=-np.pi))

# =============================================
# Helper functions
# =============================================
# Saturation cost fuction
sym_signed_u = ca.SX.sym('u', 1)
saturation_cost = ca.Function('saturation_cost', [sym_signed_u], [ca.fmax(ca.DM.zeros(1), sym_signed_u)])

# Symbolic log barrier function
sym_x = ca.SX.sym('x', 1)
sym_c = ca.SX.sym('c', 1)
sym_xu = ca.SX.sym('x_u', 1) # Upper bound
sym_xl = ca.SX.sym('x_l', 1) # Lower bound
sym_xb = 2*sym_xu*sym_xl/(sym_xu+sym_xl)
log_bar_u = ca.Function('log_bar_u', [sym_x, sym_xu], [-ca.log(sym_xu - sym_x)])
log_bar_l = ca.Function('log_bar_l', [sym_x, sym_xl], [-ca.log(sym_x - sym_xl)])
log_bar_u_norm = ca.Function('log_bar_u_norm', [sym_x, sym_xu], [-ca.log(1 - sym_x/sym_xu)])
log_bar_l_norm = ca.Function('log_bar_l_norm', [sym_x, sym_xl], [-ca.log(-sym_x/sym_xl + 1)])
# Double ended log barrier functions
log_bar_ul_norm_symm = ca.Function('log_bar_ul_norm_symm', [sym_x, sym_c, sym_xu], \
                            [log_bar_u_norm(sym_x+sym_c, sym_xu) \
                                + log_bar_l_norm(sym_x-sym_c, -sym_xu) \
                                - (log_bar_u_norm(sym_c, sym_xu) \
                                 + log_bar_l_norm(-sym_c, -sym_xu))])

# =============================================
# iLQR controller setup
# =============================================
nlmpc_params = copy.deepcopy(scenario.control_params)

# Symbolic placeholder variables
sym_q = ca.MX.sym('q', ego_dyn_model.n_q)
sym_u = ca.MX.sym('u', ego_dyn_model.n_u)
sym_um = ca.MX.sym('um', ego_dyn_model.n_u)

sym_tar_pos = ca.MX.sym('tar_pos', 2)
sym_tar_s = ca.MX.sym('tar_s', 1)

x_idx = 0
y_idx = 1
s_idx = 4
ey_idx = 5

ua_idx = 0
us_idx = 1

pos = sym_q[[x_idx, y_idx]]

quad_input_cost = scenario.cost_params['input_weight'][0]*(sym_u[ua_idx])**2 \
                    + scenario.cost_params['input_weight'][1]*(sym_u[us_idx])**2

quad_input_rate_cost = scenario.cost_params['input_rate_weight'][0]*(sym_u[ua_idx]-sym_um[ua_idx])**2 \
                         + scenario.cost_params['input_rate_weight'][1]*(sym_u[us_idx]-sym_um[us_idx])**2


prog_cost = -scenario.cost_params['comp_weights'][0]*sym_q[s_idx]
comp_cost = scenario.cost_params['comp_weights'][1]*sym_tar_s

# Ego stage cost
ego_sym_stage = quad_input_cost \
                + quad_input_rate_cost

# Ego terminal cost
ego_sym_term = prog_cost + comp_cost

# Ego cost functions
ego_sym_costs = []
for k in range(nlmpc_params.N):
    ego_sym_costs.append(ca.Function(f'ego_stage_{k}', [sym_q, sym_u, sym_um], [ego_sym_stage]))
ego_sym_costs.append(ca.Function('ego_term', [sym_q, sym_tar_s], [ego_sym_term]))

# Build symbolic constraints g_i(x, u, um) <= 0
input_rate_constr = ca.vertcat((sym_u[ua_idx]-sym_um[ua_idx]) - dt*state_input_rate_max.u.u_a, 
                                dt*state_input_rate_min.u.u_a - (sym_u[ua_idx]-sym_um[ua_idx]),
                                (sym_u[us_idx]-sym_um[us_idx]) - dt*state_input_rate_max.u.u_steer, 
                                dt*state_input_rate_min.u.u_steer - (sym_u[us_idx]-sym_um[us_idx]))
obs_avoid_constr = (0.4)**2 - ca.bilin(ca.DM.eye(2), pos - sym_tar_pos,  pos - sym_tar_pos)

ego_constrs = []
for k in range(nlmpc_params.N):
    # ego_constrs.append(ca.Function('constrs_%i' % k, [sym_q, sym_u, sym_um, sym_tar_pos], [ca.vertcat(obs_avoid_constr, input_rate_constr)]))
    if k >= 1:
        ego_constrs.append(ca.Function('constrs_%i' % k, [sym_q, sym_u, sym_um, sym_tar_pos], [obs_avoid_constr]))
    else:
        ego_constrs.append(None)
ego_constrs.append(ca.Function('constrs_%i' % k, [sym_q, sym_tar_pos], [obs_avoid_constr]))

ego_controller = CA_NL_MPC(ego_dyn_model, 
                            ego_sym_costs, 
                            ego_constrs, 
                            {'ub': state_input_ub, 'lb': state_input_lb},
                            nlmpc_params)

# Target stage cost
tar_sym_stage = quad_input_cost \
                + quad_input_rate_cost
                
# Target terminal cost
tar_sym_term = prog_cost

# Target cost functions
tar_sym_costs = []
for k in range(nlmpc_params.N):
    tar_sym_costs.append(ca.Function(f'tar_stage_{k}', [sym_q, sym_u, sym_um], [tar_sym_stage]))
tar_sym_costs.append(ca.Function('tar_term', [sym_q], [tar_sym_term]))

tar_constrs = []
for k in range(nlmpc_params.N):
    # tar_constrs.append(ca.Function('constrs_%i' % k, [sym_q, sym_u, sym_um], [input_rate_constr]))
    tar_constrs.append(None)
tar_constrs.append(None)

tar_controller = CA_NL_MPC(tar_dyn_model, 
                            tar_sym_costs, 
                            tar_constrs, 
                            {'ub': state_input_ub, 'lb': state_input_lb},
                            nlmpc_params)

# u_0, ..., u_N-1, u_-1
u_ph = [ca.MX.sym(f'u_ph_{k}', tar_dyn_model.n_u) for k in range(nlmpc_params.N)] # Inputs
# q_0, ..., q_N
q_ph = [ca.MX.sym(f'q_ph_0', tar_dyn_model.n_q)] # State
tar_pos_ph = []
tar_s_ph = []
for k in range(nlmpc_params.N):
    q_ph.append(tar_dyn_model.fd(q_ph[k], u_ph[k]))
    tar_pos_ph.append(q_ph[k+1][[x_idx, y_idx]])
tar_s_ph.append(q_ph[-1][s_idx])
Du_p = ca.jacobian(ca.vertcat(*tar_s_ph, *tar_pos_ph), ca.vertcat(*u_ph))
f_p = ca.Function('f_p', [ca.vertcat(*u_ph), q_ph[0]], [ca.vertcat(*tar_s_ph, *tar_pos_ph)])
f_Du_p = ca.Function('f_Du_p', [ca.vertcat(*u_ph), q_ph[0]], [Du_p])

# =============================================
# Create visualizer
# =============================================
# Set up plotter
global_plot_params = GlobalPlotConfigs(track_name='Lab_Track_barc', 
                                       draw_period=0.05, 
                                       update_period=0.05, 
                                       show_subplots=True, 
                                       buffer_length=50)
ego_plot_params = VehiclePlotConfigs(name='ego', 
                                         color='b', 
                                         vehicle_draw_L=0.37, 
                                         vehicle_draw_W=0.195, 
                                         show_traces=True, 
                                         show_pred=True, 
                                         show_est=False, 
                                         show_sim=True, 
                                         show_ecu=True, 
                                         show_cov=False)
tar_plot_params = VehiclePlotConfigs(name='tar', 
                                         color='g', 
                                         vehicle_draw_L=0.37, 
                                         vehicle_draw_W=0.195, 
                                         show_traces=True, 
                                         show_pred=True, 
                                         show_est=False, 
                                         show_sim=True, 
                                         show_ecu=True, 
                                         show_cov=False)
barc_fig = BarcFigure(t0=t, params=global_plot_params)
barc_fig.add_vehicle(ego_plot_params)
barc_fig.add_vehicle(tar_plot_params)

p = mp.Process(target=barc_fig.run_plotter)
p.start()

# =============================================
# Run race
# =============================================
ego_sim_state = scenario.ego_init_state
track_obj.local_to_global_typed(ego_sim_state)

tar_sim_state = scenario.tar_init_state
track_obj.local_to_global_typed(tar_sim_state)

# Initialize inputs
t = 0.0
ego_sim_state.u.u_a, ego_sim_state.u.u_steer = 0.0, 0.0
tar_sim_state.u.u_a, tar_sim_state.u.u_steer = 0.0, 0.0
ego_control = VehicleActuation(t=t, u_a=0, u_steer=0)
tar_control = VehicleActuation(t=t, u_a=0, u_steer=0)

rng = np.random.default_rng()
N = scenario.control_params.N
n_q = ego_dyn_model.n_q
n_u = ego_dyn_model.n_u

while True:
    tar_state = copy.deepcopy(tar_sim_state)
    ego_state = copy.deepcopy(ego_sim_state)

    tar_x0 = ego_dyn_model.state2q(tar_state)
    ego_x0 = ego_dyn_model.state2q(ego_state)
    
    tar_up = tar_controller.u_prev
    ego_up = ego_controller.u_prev

    # Solve for target vehicle
    tar_controller.step(tar_state)
    tar_u = tar_controller.u_pred.ravel()
    tar_sim_state.copy_control(tar_control)
    tar_sim_state = copy.deepcopy(tar_state)

    # Solve for ego vehicle
    p = f_p(tar_u, tar_x0).toarray().squeeze()
    Du_p = f_Du_p(tar_u, tar_x0).toarray()
    tar_s, tar_pos = np.expand_dims(p[0], 0), p[1:]

    eps = 1e-4
    tar_u_e = 2*rng.random(len(tar_u))-1
    tar_u_e = eps*tar_u_e/np.linalg.norm(tar_u_e)
    p_pert = f_p(tar_u+tar_u_e, tar_x0).toarray().squeeze()
    tar_s_pert, tar_pos_pert = np.expand_dims(p_pert[0], 0), p_pert[1:]

    info_pert = ego_controller.solve(ego_state, 
                                    params=dict(cost=tar_s_pert, constr=tar_pos_pert))
    ego_u_pert = ego_controller.u_pred.ravel()

    info = ego_controller.step(ego_state, 
                                solver_params=dict(cost=tar_s, constr=tar_pos))
    ego_u = ego_controller.u_pred.ravel()
    ego_sim_state.copy_control(ego_control)
    ego_sim_state = copy.deepcopy(ego_state)

    # Update plots
    barc_fig.update('ego', 
                    sim_data=copy.copy(ego_sim_state), 
                    ecu_data=copy.copy(ego_control), 
                    pred_data=copy.copy(ego_controller.get_prediction()))
    barc_fig.update('tar', 
                    sim_data=copy.copy(tar_sim_state), 
                    ecu_data=copy.copy(tar_control), 
                    pred_data=copy.copy(tar_controller.get_prediction()))

    ego_C = ca.vertcat(*ego_controller.f_C(ego_u, ego_x0, ego_up, tar_pos)).toarray().squeeze()
    ego_l = info['sol']['lam_g'].toarray().squeeze()

    # active_idxs = np.where(np.abs(C_val) <= 1e-4)[0]
    active_idxs = np.where(ego_l >= 1e-7)[0]
    # inactive_idxs = np.setdiff1d(np.arange(len(ego_l)), active_idxs)
    # ego_l[inactive_idxs] = 0
    # ego_C[active_idxs] = 0

    Dul_F = ego_controller.f_Dul_F(ego_u, ego_l, ego_x0, ego_up, tar_s, tar_pos).toarray()
    Dp_F = ego_controller.f_Dp_F(ego_u, ego_l, ego_x0, ego_up, tar_s, tar_pos).toarray()
    try:
        Dp_ul = -sp.linalg.solve(Dul_F, Dp_F)
        policy_grad = Dp_ul[:N*n_u] @ Du_p # Gradient of ego policy w.r.t. tar input sequence

        delta_u = (ego_u_pert - ego_u)/eps
        # d_u = Dp_ul[:N*n_u] @ (p_pert - p)/eps
        d_u = policy_grad @ tar_u_e/eps

        print(active_idxs)
        print(delta_u)
        print(d_u)
    except Exception as e:
        print(e)

    pdb.set_trace()

    # Apply control action and advance simulation
    # last_state = copy.deepcopy(sim_state)
    t += dt
    ego_dyn_model.step(ego_sim_state)
    tar_dyn_model.step(tar_sim_state)

    

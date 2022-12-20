#!/usr/bin/env python3

from mpclab_common.pytypes import VehicleState, VehicleActuation, Position, ParametricPose, OrientationEuler, BodyLinearVelocity, BodyAngularVelocity
from mpclab_common.models.dynamics_models import CasadiKinematicBicycleCombined
from mpclab_common.models.model_types import KinematicBicycleConfig
from mpclab_common.track import get_track

from mpclab_controllers.PID import PIDLaneFollower
from mpclab_controllers.utils.controllerTypes import PIDParams

from mpclab_simulation.dynamics_simulator import DynamicsSimulator

from mpclab_visualizations.barc_plotter_qt import BarcFigure
from mpclab_visualizations.vis_types import GlobalPlotConfigs, VehiclePlotConfigs

import pdb

import time
import numpy as np
import copy
import multiprocessing as mp

dt = 0.1
t = 0

#high level parameters
track = 'L_track_barc'
track_obj = get_track(track)

discretization_method='rk4'
dynamics_config = KinematicBicycleConfig(dt=dt,
                                            model_name='kinematic_bicycle_cl',
                                            noise=False,
                                            discretization_method=discretization_method,
                                            wheel_dist_front=0.13,
                                            wheel_dist_rear=0.13,
                                            drag_coefficient=0.1,
                                            slip_coefficient=0.1,
                                            code_gen=False)
dyn_model = CasadiKinematicBicycleCombined(t, dynamics_config, track=track_obj)
dynamics_simulator = DynamicsSimulator(t, dynamics_config, track=track_obj)

state_input_max=VehicleState(x=Position(x=np.inf, y=np.inf),
                            p=ParametricPose(s=np.inf, x_tran=track_obj.half_width, e_psi=np.inf),
                            e=OrientationEuler(psi=np.inf),
                            v=BodyLinearVelocity(v_long=np.inf, v_tran=np.inf),
                            w=BodyAngularVelocity(w_psi=np.inf),
                            u=VehicleActuation(u_a=2.1, u_steer=0.436))
state_input_min=VehicleState(x=Position(x=-np.inf, y=-np.inf),
                            p=ParametricPose(s=-np.inf, x_tran=-track_obj.half_width, e_psi=-np.inf),
                            e=OrientationEuler(psi=-np.inf),
                            v=BodyLinearVelocity(v_long=-np.inf, v_tran=-np.inf),
                            w=BodyAngularVelocity(w_psi=-np.inf),
                            u=VehicleActuation(u_a=-2.1, u_steer=-0.436))
state_input_rate_max=VehicleState(u=VehicleActuation(u_a=10.0, u_steer=np.pi))
state_input_rate_min=VehicleState(u=VehicleActuation(u_a=-10.0, u_steer=-np.pi))

# Set up PID controllers for warm start
steer_params = PIDParams(dt=dt, Kp=1.0, Ki=0.005,
                            x_ref=0.0,
                            u_max=state_input_max.u.u_steer, 
                            u_min=state_input_min.u.u_steer, 
                            du_max=state_input_rate_max.u.u_steer, 
                            du_min=state_input_rate_min.u.u_steer)
speed_params = PIDParams(dt=dt, Kp=1.0, 
                            x_ref=1.0,
                            u_max=state_input_max.u.u_a, 
                            u_min=state_input_min.u.u_a, 
                            du_max=state_input_rate_max.u.u_a, 
                            du_min=state_input_rate_min.u.u_a)
pid_controller = PIDLaneFollower(dt, steer_params, speed_params)

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
                                         show_cov=False)
barc_fig = BarcFigure(t0=t, params=global_plot_params)
barc_fig.add_vehicle(vehicle_plot_params)
barc_fig.run()

sim_state = VehicleState(t=0.0, 
                        p=ParametricPose(s=0.5, x_tran=0.0, e_psi=0), 
                        v=BodyLinearVelocity(v_long=0.0))
track_obj.local_to_global_typed(sim_state)
sim_input = VehicleActuation(t=t, u_a=0, u_steer=0)

while True:
    pid_controller.step(sim_state)
    sim_state.copy_control(sim_input)
    
    dynamics_simulator.step(sim_state)

    # Update figure
    barc_fig.update('ego', {'state': copy.deepcopy(sim_state), 
                            'ecu': copy.deepcopy(sim_input)})
    
    time.sleep(0.1)
    t += dt

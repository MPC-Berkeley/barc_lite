#!/usr/bin/env python3

import copy

import numpy as np
import scipy as sp
from filterpy.kalman import ExtendedKalmanFilter
from datetime import datetime
import pdb

from mpclab_estimation.abstractEstimator import abstractEstimator
from mpclab_estimation.utils.estimatorTypes import EKFParams

from mpclab_common.pytypes import VehicleState, VehicleActuation, OrientationEuler, BodyLinearVelocity, BodyAngularVelocity
from mpclab_common.models.model_types import PoseVelMeasurement
from mpclab_common.models.dynamics_models import CasadiDynamicsModel
from mpclab_common.models.observation_models import CasadiObservationModel

class EKFEstimator(abstractEstimator):
    def __init__(self, track,
        dynamics_model: CasadiDynamicsModel,
        observation_model: CasadiObservationModel,
        params: EKFParams = EKFParams()):

        self.track = track
        self.dynamics_model = dynamics_model
        self.observation_model = observation_model
        self.params = params

        self.n = self.dynamics_model.n_q
        self.d = self.dynamics_model.n_u
        self.z = self.observation_model.n_z

        self.dt = self.params.dt

        self.ekf = ExtendedKalmanFilter(self.n, self.z, self.d)
        self.ekf.Q = self.dynamics_model.noise_cov
        self.ekf.R = self.observation_model.noise_cov
        self.ekf.P = np.diag(self.params.init_state_cov)

        # We are supplying our own nominal dynamics model
        def f(u):
            f = lambda t, q: self.dynamics_model.fc(q, u).toarray().squeeze()
            self.ekf.x = sp.integrate.solve_ivp(f, [0,self.dt], self.ekf.x).y[:,-1]
        self.ekf.predict_x = f

        self.x_prior = copy.copy(self.ekf.x)
        self.P_prior = copy.copy(self.ekf.P)
        self.x_post = copy.copy(self.ekf.x)
        self.P_post = copy.copy(self.ekf.P)

        self.triu_idxs = np.triu_indices(self.n) # Numpy is row-major indexing by default

        self.est_state = VehicleState()

        self.initialized = False
        self.integration_time = 0

    def initialize(self, state: VehicleState):
        if self.dynamics_model.curvature_model:
            self.track.local_to_global_typed(state)
        else:
            self.track.global_to_local_typed(state)

        self.est_state = copy.copy(state)
        self.ekf.x = copy.copy(self.dynamics_model.state2q(state))
        self.initialized = True

    def update(self, z: PoseVelMeasurement, u: VehicleActuation) -> VehicleState:
        if not self.initialized:
            raise RuntimeError('Estimator is not initialized with state')

        x_bar = copy.copy(self.ekf.x)
        u_bar = self.dynamics_model.input2u(copy.copy(u))
        z_bar = self.observation_model.meas2z(copy.copy(z))

        # Update process and measurement noise covariance matrices
        self.ekf.Q = self.dynamics_model.noise_cov
        self.ekf.R = self.observation_model.noise_cov
        
        # Get process model jacobians
        self.ekf.F = self.dynamics_model.fAd(x_bar, u_bar, np.zeros(self.dynamics_model.n_m)).toarray()
        self.ekf.B = self.dynamics_model.fBd(x_bar, u_bar, np.zeros(self.dynamics_model.n_m)).toarray()

        # Do process update
        self.ekf.predict(u_bar)

        # Do measurement update
        obs_args = [x_bar, u_bar]
        if self.observation_model.noise:
            obs_args += [np.zeros(self.observation_model.n_n)]
        H = lambda x_bar: self.observation_model.fH(*obs_args).toarray()
        h = lambda x_bar: self.observation_model.h(*obs_args).toarray().squeeze()
        self.ekf.update(z_bar, H, h)

        self.x_prior = copy.copy(self.ekf.x_prior)
        self.P_prior = copy.copy(self.ekf.P_prior)
        self.x_post = copy.copy(self.ekf.x_post)
        self.P_post = copy.copy(self.ekf.P_post)

        self.dynamics_model.qu2state(self.est_state, copy.copy(self.x_post), copy.copy(u_bar))

        if self.dynamics_model.curvature_model:
            self.track.local_to_global_typed(self.est_state)
            self.est_state.local_state_covariance = copy.copy(self.P_post[self.triu_idxs]) # Only store upper triangular part of covariance matrix
        else:
            self.track.global_to_local_typed(self.est_state)
            self.est_state.global_state_covariance = copy.copy(self.P_post[self.triu_idxs]) # Only store upper triangular part of covariance matrix

        return self.est_state

    def reset(self, state: VehicleState):
        if self.dynamics_model.curvature_model:
            self.track.local_to_global_typed(state)
        else:
            self.track.global_to_local_typed(state)

        self.est_state = copy.copy(state)

        self.ekf.x = copy.copy(self.dynamics_model.state2q(state))
        self.ekf.P = np.diag(self.params.init_state_cov)

def main():
    from mpclab_common.models.dynamics_models import CasadiDynamicBicycle
    from mpclab_common.track import get_track
    from mpclab_common.pytypes import VehicleState, Position, VehicleActuation
    from mpclab_common.models.model_types import DynamicBicycleConfig, PoseVelMeasurement
    from mpclab_common.models.observation_models import CasadiDynamicBicycleFullStateObserver

    import pdb

    rng = np.random.default_rng()

    vehicle_config = DynamicBicycleConfig()
    dynamics = CasadiDynamicBicycle(0, vehicle_config)
    track = get_track('LTrack_barc')
    observer = CasadiDynamicBicycleFullStateObserver()

    params = EKFParams(process_noise_cov=np.sqrt(0.1)*np.ones(6),
                       measurement_noise_cov=np.sqrt(0.1)*np.ones(6),
                       init_state_cov=np.sqrt(0.1)*np.ones(6))
    ekf_estimator = EKFEstimator(track, dynamics, observer, params)
    pos = Position(x=0, y=0)
    x_k = VehicleState(t=0, x=pos, e = OrientationEuler(psi = 0), v = BodyLinearVelocity(v_long=0.5, v_tran=0), w = BodyAngularVelocity(w_psi = 0))
    ekf_estimator.initialize(x_k)

    u_k = VehicleActuation(u_a=0.5, u_steer=0.1)

    x_kp1 = copy.copy(x_k)
    x_kp1.u_a = u_k.u_a
    x_kp1.u_steer = u_k.u_steer
    dynamics.step(x_kp1)

    z_kp1 = PoseVelMeasurement(x=x_kp1.x+rng.normal(0,0.1),
                               y=x_kp1.y+rng.normal(0,0.1),
                               yaw=x_kp1.psi+rng.normal(0,0.1),
                               v_long=x_kp1.v_long+rng.normal(0,0.1),
                               v_tran=x_kp1.v_tran+rng.normal(0,0.1),
                               yaw_dot=x_kp1.psidot+rng.normal(0,0.1))

    x_est = ekf_estimator.update(z_kp1, u_k)
    pdb.set_trace()

if __name__ == '__main__':
    main()

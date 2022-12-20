#!/usr/bin/env python3

import copy

import numpy as np
from scipy.spatial.transform import Rotation

from mpclab_estimation.abstractEstimator import abstractEstimator
from mpclab_estimation.utils.estimatorTypes import PassThroughParams

from mpclab_common.pytypes import VehicleState, VehicleActuation
from mpclab_common.models.model_types import PoseVelMeasurement, AccelMeasurement

class PassThroughEstimator(abstractEstimator):
    def __init__(self, track,
        params: PassThroughParams = PassThroughParams()):

        self.track = track

        self.state_est = VehicleState()

        self.initialized = False

    def initialize(self):
        self.initialized = True

    def update(self, z: PoseVelMeasurement, a: AccelMeasurement, aa: AccelMeasurement, u: VehicleActuation) -> VehicleState:
        if not self.initialized:
            raise(RuntimeError('Estimator is not initialized'))

        self.state_est.t            = copy.copy(z.t)

        self.state_est.x.x          = copy.copy(z.x)
        self.state_est.x.y          = copy.copy(z.y)
        self.state_est.x.z          = copy.copy(z.z)

        self.state_est.e.phi        = copy.copy(z.roll)
        self.state_est.e.theta      = copy.copy(z.pitch)
        self.state_est.e.psi        = copy.copy(z.yaw)

        self.state_est.v.v_long     = copy.copy(z.v_long)
        self.state_est.v.v_tran     = copy.copy(z.v_tran)
        self.state_est.v.v_n        = copy.copy(z.v_vert)

        self.state_est.w.w_phi      = copy.copy(z.roll_dot)
        self.state_est.w.w_theta    = copy.copy(z.pitch_dot)
        self.state_est.w.w_psi      = copy.copy(z.yaw_dot)
        
        rot = Rotation.from_euler('ZYX', [z.yaw, z.pitch, z.roll])
        qi, qj, qk, qr = rot.as_quat()
        self.state_est.q.qi         = qi
        self.state_est.q.qj         = qj
        self.state_est.q.qk         = qk
        self.state_est.q.qr         = qr

        if a is not None:
            self.state_est.a.a_long     = copy.copy(a.x)
            self.state_est.a.a_tran     = copy.copy(a.y)
            self.state_est.a.a_n        = copy.copy(a.z)
        
        if aa is not None:
            self.state_est.aa.a_phi     = copy.copy(aa.x)
            self.state_est.aa.a_theta   = copy.copy(aa.y)
            self.state_est.aa.a_psi     = copy.copy(aa.z)
            
        self.state_est.u.u_a        = copy.copy(u.u_a)
        self.state_est.u.u_steer    = copy.copy(u.u_steer)

        # self.track.global_to_local_typed(self.state_est)

        return self.state_est

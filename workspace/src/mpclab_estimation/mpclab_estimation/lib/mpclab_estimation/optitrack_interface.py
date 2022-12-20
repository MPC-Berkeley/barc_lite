#!/usr/bin/env python3

import numpy as np
from scipy.spatial.transform import Rotation

from mpclab_estimation.abstractInterface import abstractInterface
from mpclab_estimation.utils.interfaceTypes import OptiTrackParams

from mpclab_common.models.model_types import PoseVelMeasurement, AccelMeasurement

import time
import pdb

'''
Class which preprocesses the measurements from the OptiTrack system
go: Frame attached to the OptiTrack origin (global origin)
to: Frame attached to track origin
rf: Frame attached to the OptiTrack rigid body
cf: Frame attached to car CoM and with +x aligned with its longitudinal axis
'''
class OptiTrackInterface(abstractInterface):
    def __init__(self, params: OptiTrackParams = OptiTrackParams()):
        # Offset of optitrack rigid body frame with respect to com aligned body frame of car
        self.cf2rf_p = np.array([params.offset_long, params.offset_tran, params.offset_vert])

        # Transformation from car body frame to Optitrack rigid body frame
        self.cf2rf_R = Rotation.from_euler('ZYX', [params.offset_yaw, params.offset_pitch, params.offset_roll])

        # Location and orientation of the origin of the track frame with respect to the OptiTrack frame
        self.go2to_p = np.array([params.origin_x, params.origin_y, params.origin_z])

        # Transformation from Optitrack global origin frame to track origin frame
        self.go2to_R = Rotation.from_euler('ZYX', [params.origin_yaw, params.origin_pitch, params.origin_roll])

        self.posevel_com = PoseVelMeasurement()

    def get_com_meas(self, posevel_meas: PoseVelMeasurement, lin_accel_meas: AccelMeasurement, ang_accel_meas: AccelMeasurement) -> PoseVelMeasurement:
        # Unpack pose and velocity measurements
        rf_p_go = np.array([posevel_meas.x, posevel_meas.y, posevel_meas.z])
        rf_e_go = np.array([posevel_meas.yaw, posevel_meas.pitch, posevel_meas.roll])
        rf_v = np.array([posevel_meas.v_long, posevel_meas.v_tran, posevel_meas.v_vert])
        rf_w = np.array([posevel_meas.roll_dot, posevel_meas.pitch_dot, posevel_meas.yaw_dot])

        # Correct for attitude misalignment between Optitrack rigid body and car frames
        cf_e_go = rf_e_go - self.cf2rf_R.as_euler('ZYX')

        # Transformation from global origin frame to car body frame
        go2cf_R = Rotation.from_euler('ZYX', cf_e_go)
        
        # Angular velocities in car body frame
        cf_w = self.cf2rf_R.apply(rf_w)

        # Position of the car frame in the global frame
        cf_p_go = rf_p_go - go2cf_R.apply(self.cf2rf_p)

        # Get the position of the car frame in the track frame
        cf_p_to = self.go2to_R.inv().apply(cf_p_go - self.go2to_p)

        # Attitude of the car in the track frame
        cf_e_to = cf_e_go - self.go2to_R.as_euler('ZYX')

        # Linear velocities in car body frame
        C = np.cross(cf_w, -self.cf2rf_p)
        cf_v = self.cf2rf_R.apply(rf_v) + C

        # Correct for pitch and roll of car so that velocities are in the global x-y plane
        cf2to_R = Rotation.from_euler('ZYX', [0, cf_e_to[1], cf_e_to[2]])
        cf_v = cf2to_R.apply(cf_v)

        self.posevel_com.t = posevel_meas.t

        self.posevel_com.x = cf_p_to[0]
        self.posevel_com.y = cf_p_to[1]
        self.posevel_com.z = cf_p_to[2]

        self.posevel_com.yaw = cf_e_to[0]
        self.posevel_com.pitch = cf_e_to[1]
        self.posevel_com.roll = cf_e_to[2]
        
        self.posevel_com.v_long = cf_v[0]
        self.posevel_com.v_tran = cf_v[1]
        self.posevel_com.v_vert = cf_v[2]

        self.posevel_com.roll_dot =  cf_w[0]
        self.posevel_com.pitch_dot =  cf_w[1]
        self.posevel_com.yaw_dot =  cf_w[2]

        if lin_accel_meas is not None:
            rf_a = np.array([lin_accel_meas.x, lin_accel_meas.y, lin_accel_meas.z])
            # Linear accelerations in car body frame
            cf_a = self.cf2rf_R.apply(rf_a) + np.cross(cf_w, C)
        
        if ang_accel_meas is not None:
            rf_aa = np.array([ang_accel_meas.x, ang_accel_meas.y, ang_accel_meas.z])
            # Angular accelerations in car body frame
            cf_aa = self.cf2rf_R.apply(rf_aa)
            cf_a += np.cross(cf_aa, -self.cf2rf_p)

        if lin_accel_meas is not None:
            lin_accel_com = AccelMeasurement(x=cf_a[0], y=cf_a[1], z=cf_a[2])
        else:
            lin_accel_com = None

        if ang_accel_meas is not None:
            ang_accel_com = AccelMeasurement(x=cf_aa[0], y=cf_aa[1], z=cf_aa[2])
        else:
            ang_accel_com = None

        return self.posevel_com, lin_accel_com, ang_accel_com

if __name__ == '__main__':
    # params = OptiTrackParams(origin_x=0, origin_y=0, origin_z=0, 
    #                      origin_roll=0, origin_pitch=0, origin_yaw=0, 
    #                      offset_long=0.5, offset_tran=0.2, offset_vert=0,
    #                      offset_roll=0, offset_pitch=-10*np.pi/180, offset_yaw=0.0)
    params = OptiTrackParams(origin_x=0, origin_y=0, origin_z=0, 
                         origin_roll=0, origin_pitch=0, origin_yaw=0, 
                         offset_long=0.5, offset_tran=0.2, offset_vert=0,
                         offset_roll=0.0, offset_pitch=0.0, offset_yaw=0.0)

    interface = OptiTrackInterface(params)

    measurement = PoseVelMeasurement(x=10.0, y=7.0, z=0.5,
                                    yaw=np.pi/4, pitch=10*np.pi/180, roll=10*np.pi/180,
                                    v_long=1.0, v_tran=1.0, v_vert=0.0,
                                    yaw_dot=0.5, pitch_dot=0.0, roll_dot=0.0)

    out = interface.get_com_meas(measurement)

    print(out)
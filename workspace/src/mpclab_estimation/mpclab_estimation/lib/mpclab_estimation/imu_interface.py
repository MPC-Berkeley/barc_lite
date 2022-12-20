#!/usr/bin/env python3

import numpy as np
from scipy.spatial.transform import Rotation

from mpclab_estimation.abstractInterface import abstractInterface
from mpclab_estimation.utils.interfaceTypes import IMUParams

from mpclab_common.models.model_types import IMUMeasurement

'''
Class which preprocesses the measurements from an IMU
'''
class IMUInterface(abstractInterface):
    def __init__(self, params: IMUParams = IMUParams()):
        # Offset of vive tracker with respect to com of car
        self.imu_offset_long = params.offset_long
        self.imu_offset_tran = params.offset_tran
        self.imu_offset_vert = params.offset_vert
        self.imu_offset_yaw = params.offset_yaw
        self.imu_offset_pitch = params.offset_pitch
        self.imu_offset_roll = params.offset_roll

        self.accel_transformation = np.vstack((params.accel_x_transformation, params.accel_y_transformation, params.accel_z_transformation))
        self.gyro_transformation = np.vstack((params.gyro_x_transformation, params.gyro_y_transformation, params.gyro_z_transformation))

        self.imu_meas = IMUMeasurement()

    def get_com_meas(self, imu_meas: IMUMeasurement) -> IMUMeasurement:
        quat = [imu_meas.orientation.x, imu_meas.orientation.y, imu_meas.orientation.z, imu_meas.orientation.w]
        if np.linalg.norm(quat) <= 1e-8:
            imu_yaw, imu_pitch, imu_roll = 0, 0, 0
        else:
            rot = Rotation.from_quat([imu_meas.orientation.x, imu_meas.orientation.y, imu_meas.orientation.z, imu_meas.orientation.w])
            imu_yaw, imu_pitch, imu_roll = rot.as_euler('ZYX') # Use intrinsic rotations along principle axes 1. yaw axis, 2. pitch axis, 3. roll axis

        imu_roll_dot, imu_pitch_dot, imu_yaw_dot = self.gyro_transformation @ np.array([imu_meas.angular_velocity.x, imu_meas.angular_velocity.y, imu_meas.angular_velocity.z])
        imu_a_long, imu_a_tran, imu_a_vert = self.accel_transformation @ np.array([imu_meas.linear_acceleration.x, imu_meas.linear_acceleration.y, imu_meas.linear_acceleration.z])

        com_roll = imu_roll
        com_pitch = imu_pitch
        com_yaw = imu_yaw

        com_roll_dot = imu_roll_dot
        com_pitch_dot = imu_pitch_dot
        com_yaw_dot = imu_yaw_dot

        com_a_long = imu_a_long
        com_a_tran = imu_a_tran
        com_a_vert = imu_a_vert

        rot = Rotation.from_euler('ZYX', [com_yaw, com_pitch, com_roll])
        quat = rot.as_quat()

        self.imu_meas.t = imu_meas.t

        self.imu_meas.orientation.x = quat[0]
        self.imu_meas.orientation.y = quat[1]
        self.imu_meas.orientation.z = quat[2]
        self.imu_meas.orientation.w = quat[3]

        self.imu_meas.angular_velocity.x =  com_roll_dot # TODO: Need to account for misalignment between tracker frame and car body frame
        self.imu_meas.angular_velocity.y =  com_pitch_dot
        self.imu_meas.angular_velocity.z =  com_yaw_dot

        self.imu_meas.linear_acceleration.x = com_a_long
        self.imu_meas.linear_acceleration.y = com_a_tran
        self.imu_meas.linear_acceleration.z = com_a_vert

        return self.imu_meas

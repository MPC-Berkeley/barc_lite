#!/usr/bin python3

import numpy as np
from dataclasses import dataclass, field

from mpclab_common.pytypes import PythonMsg

@dataclass
class ViveParams(PythonMsg):
    topic: str          = field(default = 'tracker/odom')
    origin_x: float     = field(default = 0)
    origin_y: float     = field(default = 0)
    origin_z: float     = field(default = 0)
    origin_yaw: float   = field(default = 0)
    offset_long: float  = field(default = 0)
    offset_tran: float  = field(default = 0)
    offset_vert: float  = field(default = 0)
    offset_yaw: float   = field(default = 0)

@dataclass
class OptiTrackParams(PythonMsg):
    odom_topic: str     = field(default = 'optitrack/odom')
    lin_accel_topic: str    = field(default = 'optitrack/lin_accel')
    ang_accel_topic: str    = field(default = 'optitrack/ang_accel')
    
    origin_x: float     = field(default = 0)
    origin_y: float     = field(default = 0)
    origin_z: float     = field(default = 0)

    origin_roll: float  = field(default = 0)
    origin_pitch: float = field(default = 0)
    origin_yaw: float   = field(default = 0)

    offset_long: float  = field(default = 0)
    offset_tran: float  = field(default = 0)
    offset_vert: float  = field(default = 0)

    offset_roll: float  = field(default = 0)
    offset_pitch: float = field(default = 0)
    offset_yaw: float   = field(default = 0)

@dataclass
class T265Params(PythonMsg):
    odom_topic: str     = field(default = 'camera/odom/sample')
    imu_topic: str      = field(default = 'camera/imu')
    offset_long: float  = field(default = 0)
    offset_tran: float  = field(default = 0)
    offset_vert: float  = field(default = 0)
    offset_yaw: float   = field(default = 0)

@dataclass
class IMUParams(PythonMsg):
    topic: str                          = field(default = 'camera/imu')
    offset_long: float                  = field(default = 0)
    offset_tran: float                  = field(default = 0)
    offset_vert: float                  = field(default = 0)
    offset_yaw: float                   = field(default = 0)
    offset_pitch: float                 = field(default = 0)
    offset_roll: float                  = field(default = 0)

    accel_x_transformation: np.ndarray  = field(default = None)
    accel_y_transformation: np.ndarray  = field(default = None)
    accel_z_transformation: np.ndarray  = field(default = None)
    gyro_x_transformation: np.ndarray   = field(default = None)
    gyro_y_transformation: np.ndarray   = field(default = None)
    gyro_z_transformation: np.ndarray   = field(default = None)

    def __post_init__(self):
        if self.accel_x_transformation is None:
            self.accel_x_transformation = np.array([1, 0, 0])
        if self.accel_y_transformation is None:
            self.accel_y_transformation = np.array([0, 1, 0])
        if self.accel_z_transformation is None:
            self.accel_z_transformation = np.array([0, 0, 1])
        if self.gyro_x_transformation is None:
            self.gyro_x_transformation = np.array([1, 0, 0])
        if self.gyro_y_transformation is None:
            self.gyro_y_transformation = np.array([0, 1, 0])
        if self.gyro_z_transformation is None:
            self.gyro_z_transformation = np.array([0, 0, 1])

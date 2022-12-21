#!/usr/bin python3

from dataclasses import dataclass, field
from mpclab_common.pytypes import PythonMsg

@dataclass
class OptiTrackSimConfig(PythonMsg):
    n_bound: float          = field(default = 0.5)

    x_std: float            = field(default = None)
    y_std: float            = field(default = None)
    z_std: float            = field(default = None)

    yaw_std: float          = field(default = None)
    pitch_std: float        = field(default = None)
    roll_std: float         = field(default = None)

    v_long_std: float       = field(default = None)
    v_tran_std: float       = field(default = None)
    v_vert_std: float       = field(default = None)

    yaw_dot_std:float       = field(default = None)
    roll_dot_std:float      = field(default = None)
    pitch_dot_std:float     = field(default = None)

    origin_x: float         = field(default = 0)
    origin_y: float         = field(default = 0)
    origin_z: float         = field(default = 0)

    origin_roll: float      = field(default = 0)
    origin_pitch: float     = field(default = 0)
    origin_yaw: float       = field(default = 0)

    offset_long: float      = field(default = 0)
    offset_tran: float      = field(default = 0)
    offset_vert: float      = field(default = 0)

    offset_roll: float      = field(default = 0)
    offset_pitch: float     = field(default = 0)
    offset_yaw: float       = field(default = 0)
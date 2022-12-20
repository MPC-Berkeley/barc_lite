from dataclasses import dataclass, field
import time

from mpclab_common.pytypes import VehicleState
from typing import List

import numpy as np

from mpclab_common.pytypes import PythonMsg

@dataclass
class GlobalPlotConfigs(PythonMsg):
    # Global visualization params
    track_name: str         = field(default = 'LTrack_barc')
    circuit: bool           = field(default = True)
    show_meter_markers: bool = field(default = False)

    figure_size: list       = field(default_factory = lambda : [1500, 750])
    figure_title: str       = field(default = 'BARC Visualizer')

    show_lineplots: bool    = field(default = True)

    state_data_fields: list = field(default_factory = list)
    state_units: list       = field(default_factory = list)

    input_data_fields: list = field(default_factory = list)
    input_units: list       = field(default_factory = list)

    buffer_length: int      = field(default = 100)
    keep_history: bool      = field(default = False)

    draw_period: float      = field(default = 0.1)
    update_period: float    = field(default = 0.001)

@dataclass
class VehiclePlotConfigs(PythonMsg):
    # Vehicle specific params
    name: str               = field(default = 'barc_1')
    color: str              = field(default = 'b')

    show_traces: bool       = field(default = True)

    show_state: bool        = field(default = True)
    state_topics: list      = field(default_factory = lambda : ['est_state'])
    state_trace_styles: list = field(default_factory = lambda : ['solid']) 

    show_input: bool        = field(default = True)
    input_topics: str       = field(default_factory = lambda : ['ecu'])
    input_trace_styles: list = field(default_factory = lambda : ['solid']) 

    show_pred: bool         = field(default = False)
    pred_topics: list       = field(default_factory = lambda : ['pred'])
    pred_styles: list       = field(default_factory = lambda : ['solid']) 
    
    show_point_set: bool    = field(default = False)
    point_set_topics: list  = field(default_factory = lambda : ['ss'])
    point_set_modes: list    = field(default_factory = lambda : ['points'])
    
    show_cov: bool          = field(default = False)
    cov_topics: str         = field(default_factory = lambda : ['est_state'])

    show_full_traj: bool    = field(default = False)
    state_list: List[VehicleState] = field(default = List[VehicleState])
    
    show_full_vehicle_bodies: bool          = field(default = False)

    vehicle_draw_L: float   = field(default = 0.25)
    vehicle_draw_W: float   = field(default = 0.1)

    simulated: bool         = field(default = False)

    raceline_file: str      = field(default = None)

@dataclass
class ObstaclePlotConfigs(PythonMsg):
    name: str               = field(default = 'obs_1')
    color: str              = field(default = 'b')

    alpha: float            = field(default = 1)

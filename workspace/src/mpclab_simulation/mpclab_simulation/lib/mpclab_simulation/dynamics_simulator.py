#!/usr/bin/env python3

import numpy as np

from mpclab_common.models.dynamics_models import get_dynamics_model
from mpclab_common.pytypes import VehicleState

from collections import deque
import copy
import pdb

class DynamicsSimulator():
    '''
    Class for simulating vehicle dynamics possibly with delay
    '''
    def __init__(self, t0: float, dynamics_config, delay=None, track=None):
        # delay: delay time in seconds for each input channel
        self.model = get_dynamics_model(t0, dynamics_config, track=track)
        if delay is not None and delay > 0:
            self.delay_steps = int(delay/self.model.dt)
            self.delay_buffer = [deque([0 for _ in range(self.delay_steps)], maxlen=self.delay_steps) for _ in range(self.model.n_u)]
        else:
            self.delay_steps = 0
            self.delay_buffer = None
        return

    def step(self, state: VehicleState, T=None):
        # T: simulation duration in seconds
        if T is None:
            sim_steps = 1
        else:
            sim_steps = int(T/self.model.dt)
        
        u_new = copy.copy(self.model.input2u(state.u))
        for _ in range(sim_steps):
            if self.delay_buffer is not None:
                u_delay = np.array([self.delay_buffer[i][0] for i in range(self.model.n_u)])
                self.model.u2input(state.u, u_delay)
                for i in range(self.model.n_u):
                    self.delay_buffer[i].append(u_new[i])
            self.model.step(state)


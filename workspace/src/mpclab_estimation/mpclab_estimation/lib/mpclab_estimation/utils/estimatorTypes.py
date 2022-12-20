#!/usr/bin python3

from dataclasses import dataclass, field
import numpy as np

from mpclab_common.pytypes import PythonMsg

@dataclass
class EKFParams(PythonMsg):
    dt: float                           = field(default=0.1)

    init_state_cov: np.ndarray          = field(default=None)

@dataclass
class PassThroughParams(PythonMsg):
    dt: float = field(default=0.1)

#!/usr/bin/env python
# coding: utf-8

"""
Exploration of the Generic2DOscillator phase portrait. 


Author: Paula Sanz-Leon, QIMRB, 2020
"""

import tvb.simulator.models
from tvb.simulator.plot.phase_plane_interactive import PhasePlaneInteractive
import numpy as np


parameter_sets = {'spiegler-jirsa-2013': 
                                   {'tau': np.array([1]), 'a': np.array([0.23]),
                                    'b': np.array([-0.3333]), 'c': np.array([0.0]),
                                    'I': np.array([0.0]), 'd': np.array([0.1]),
                                    'e': np.array([0.0]), 'f': np.array([1.0]),
                                    'g': np.array([3.0]), 'alpha': np.array([3.0]),
                                    'beta': np.array([0.27]), 'gamma': np.array([-1.0])} 
                   }                    





this_set = 'spiegler-jirsa-2013'
params = parameter_sets[this_set]

MODEL = tvb.simulator.models.Generic2dOscillator(**params)
import tvb.simulator.integrators
INTEGRATOR = tvb.simulator.integrators.HeunStochastic(dt=2**-4, 
	                                                  noise=tvb.simulator.noise.Additive(nsig=np.array([0.0])))
ppi_fig = PhasePlaneInteractive(model=MODEL, integrator=INTEGRATOR)
ppi_fig.show()
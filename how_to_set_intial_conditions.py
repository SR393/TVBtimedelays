#!/usr/bin/env python
# coding: utf-8

"""
This script shows how to obtain constant initial conditions

Set initial conditions to a constant value, in general this constant value is 
the fixed point  of the model in the absence of coupling and delays.


Author: Paula Sanz-Leon, QIMRB, 2020
"""

from tvb.simulator.lab import *
import numpy as np

# Large scale structure
white_matter = connectivity.Connectivity.from_file()
white_matter.speed = np.array([2.0])
white_matter_coupling = coupling.Linear(a=np.array([0.0]), b=np.array([0.0]))

# Integrator
heunint = integrators.HeunDeterministic(dt=2**-6)

# Monitors
mon_tavg = monitors.TemporalAverage(period=2**-2)
what_to_watch = (mon_tavg, )

#Model with parameters from Spiegler and Jirsa, 2013 - Fixed point at roughly V=1.2 and W=-0.6
oscillator = models.Generic2dOscillator(tau = np.array([1.0]), a = np.array([0.23]),
                                        b = np.array([-0.3333]), c = np.array([0.0]),
                                        I = np.array([0.0]), d = np.array([0.1]),
                                        e = np.array([0.0]), f = np.array([1.0]),
                                        g = np.array([3.0]), alpha = np.array([3.0]),
                                        beta = np.array([0.27]), gamma = np.array([-1.0]))
        




################################################################################
###                              CASE A - set constant values                ###
################################################################################

sim = simulator.Simulator(model = oscillator, 
                          connectivity = white_matter,
                          coupling = white_matter_coupling,
                          integrator = heunint, 
                          monitors = what_to_watch,
                          simulation_length=500.0).configure()



# Expose the underbelly of the simulator's memory/history 
print(sim.history.buffer.shape) # shape is -> [max_delay_steps, cvar, nodes, modes]

# Replace the history by a constant value of V -- the only coupling variable of this model
v_fixed = 1.1967187 # Approximately this value, should calculate the value analytically to be 100% sure.
sim.history.buffer = v_fixed * np.ones(sim.history.buffer.shape)

w_fixed = -0.62543087 # Approximately this value, should calculate the value analytically to be 100% sure.
# Replace the initial state of the simulator with constant values of V and W
# V
sim.current_state[0, ...] = v_fixed
# W
sim.current_state[1, ...] = w_fixed


# Now run the simulation 
(time_a, data_a), = sim.run()

# Plot results -- Showing some transient activity until it vanishes after 250 milliseconds
import matplotlib.pyplot as plt
fig_a, ax = plt.subplots(2, 1, figsize = (15, 10))
# Data
ax[0].plot(time_a, data_a[:, 0, :, 0], 'k', alpha=0.4)
ax[0].set_ylim((1.19,1.20))
# After running the simulation 
ax[1].plot(sim.history.buffer[:, 0, :, 0], 'b', alpha=0.4)


# We can now run the actual simulation of interest
# 1. Change coupling strength cause we had set it to zero
sim.coupling.a = np.array(0.025)

# 2. Change simulation length to the actual length we want 
sim.simulation_length = 2000.0

(time_b, data_b), = sim.run()

fig_b, ax = plt.subplots(1, 1, figsize = (15, 10))
# Data
ax.plot(time_b, data_b[:, 0, :, 0], 'k', alpha=0.2)

plt.show()
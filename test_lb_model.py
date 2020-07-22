#!/usr/bin/env python
# coding: utf-8

"""
This runs short simulations  with the Larter-Breakspear model,
also known as the chaotic oscillator used for metastable brainwaves


Author: Paula Sanz-Leon, QIMRB, 2020
"""

from tvb.simulator.lab import *
import numpy as np

# Parameters Roberts et al 2019
# Global coupling strength - Changes between 0.0 and 0.6
gcs_uncoupled = np.array([0.0]) # start with single node dynamics
QV_max = np.array([1.0])
VT = np.array([0.0])
delta_V = np.array([0.65])
oscillator = models.LarterBreakspear(gCa = np.array([1.0]), gNa = np.array([6.7]),
                                     gK= np.array([2.0]), gL = np.array([0.5]),
                                     VCa = np.array([1.0]), VNa = np.array([0.53]),
                                     VK = np.array([-0.7]), VL = np.array([-0.5]),
                                     TCa = np.array([-0.01]), TNa = np.array([0.3]),
                                     TK = np.array([0.0]), d_Ca = np.array([0.15]),
                                     d_Na = np.array([0.15]), d_K = np.array([0.3]),
                                     VT = VT, ZT = np.array([0.0]),
                                     d_V = delta_V, d_Z = np.array([0.65]),
                                     QV_max = QV_max, QZ_max = np.array([1.0]),
                                     b = np.array([0.1]), phi = np.array([0.7]),
                                     rNMDA = np.array([0.25]), aee = np.array([0.36]),
                                     aie = np.array([2.0]), ane = np.array([1.0]),
                                     aei = np.array([2.0]), ani = np.array([0.4]),
                                     Iext = np.array([0.3]), tau_K = np.array([1.0]),
                                     C = gcs_uncoupled)  

# Connectivity Matrix
filename_conn = 'geombinsurr_partial_000_Gs.zip'
white_matter = connectivity.Connectivity.from_file(filename_conn)

# in-strength 
in_strength = white_matter.weights.sum(axis=1)

#Scale weights by in-strength 
white_matter.weights /= in_strength
white_matter.speed = np.array([5000.0])

# Long-range coupling function is a nonlinear interaction
coupling_function_par_a = 0.5* QV_max
coupling_function_par_b = np.array([1.0])
coupling_function_par_midpoint = VT
coupling_function_par_sigma = delta_V

# Use nonlinear coupling as in Roberts et al 2019
white_matter_coupling = coupling.HyperbolicTangent(a=coupling_function_par_a, 
	                                               b=coupling_function_par_b,
	                                               midpoint= coupling_function_par_midpoint,
	                                               sigma=coupling_function_par_sigma)

# Integration scheme
heunint = integrators.HeunDeterministic(dt=2**-5)


# Monitors
mon_raw = monitors.Raw()
what_to_watch = (mon_raw, )

# Create simulator object
sim = simulator.Simulator(model = oscillator, 
                          connectivity = white_matter,
                          coupling = white_matter_coupling,
                          integrator = heunint, 
                          monitors = what_to_watch,
                          simulation_length=200.0).configure()


# Set initial conditions for all the nodes
print(sim.history.buffer.shape) # shape is -> [max_delay_steps, cvar, nodes, modes]

# Replace the history by a constant value of V -- the only coupling variable of this model
v_fixed = 0.0 # Arbitrary
sim.history.buffer = v_fixed * np.ones(sim.history.buffer.shape)

w_fixed = 0.0 # Arbitrary 
z_fixed = 0.0 # Arbitrary 

# Replace the initial state of the simulator with constant values of V and W
# V
sim.current_state[0, ...] = v_fixed
# W
sim.current_state[1, ...] = w_fixed
# Z
sim.current_state[2, ...] = z_fixed


(time_a, data_a), = sim.run()

# Plot results -- Showing some transient activity until it settles 
import matplotlib.pyplot as plt
fig_a, ax_a = plt.subplots(2, 1, figsize = (15, 10))
# Data
ax_a[0].plot(time_a, data_a[:, 0, 0:256, 0], 'k', alpha=0.4)
ax_a[1].plot(time_a, data_a[:, 0, 256:-1, 0], 'k', alpha=0.4)

# We can now run the actual simulation of interest
# 1. Change coupling strength cause we had set it to zero
gcs_coupled = np.array([0.1]) # enable long-range coupling
sim.model.C = gcs_coupled

# 2. Change simulation length to the actual length we want 
sim.simulation_length = 500.0
(time_b, data_b), = sim.run()

fig_b, ax_b = plt.subplots(2, 1, figsize = (15, 10))
# Data
ax_b[0].plot(time_b, data_b[:, 0, 0:256, 0], 'k', alpha=0.2)
ax_b[1].plot(time_b, data_b[:, 0, 256:-1, 0], 'k', alpha=0.2)

plt.show()
#!/usr/bin/env python
# coding: utf-8

"""
This script shows a couple of ways to set initial conditions

(a) set them to a constant value, in general the fixed point 
    of the model, in the absence of coupling and delays.

(b) using continuation and the absence of coupling.

Author: Paula Sanz-Leon, QIMRB, 2020
"""

from tvb.simulator.lab import *


# Large scale structure
white_matter = connectivity.Connectivity.from_file()
white_matter.speed = numpy.array([2.0])
white_matter_coupling = coupling.Linear(a=numpy.array([0.025]))

# Integrator
heunint = integrators.HeunDeterministic(dt=2**-6)

# Monitors
mon_tavg = monitors.TemporalAverage(period=2**-2)
what_to_watch = (mon_tavg, )

#Model with parameters from Spiegler and Jirsa, 2013 - Fixed point at roughly V=1.2 and W=-0.6
oscillator = models.Generic2dOscillator(tau = numpy.array([1]), a = numpy.array([0.23]),
                                        b = numpy.array([-0.3333]), c = numpy.array([0]),
                                        I = numpy.array([0]), d = numpy.array([0.1]),
                                        e = numpy.array([0]), f = numpy.array([1]),
                                        g = numpy.array([3]), alpha = numpy.array([3]),
                                        beta = numpy.array([0.27]), gamma = numpy.array([-1]))
        




def sim_run(white_matter):
    """
    Define simple function to run simulations
    Returns confifured Simulator object.

    """

    sim = simulator.Simulator(model = oscillator, 
                              connectivity = white_matter,
                              coupling = white_matter_coupling,
                              integrator = heunint, 
                              monitors = what_to_watch,
                              simulation_length=500.0).configure()


        

    # Run the simulation for a  bit ~ 500 ms
    (time, data), = sim.run()

    # Update sim length and run for a bit longer 
    sim.simulation_length = 1000.0
    (time, data), = sim.run()

    return time, data, sim

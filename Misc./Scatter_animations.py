import numpy as np
import scipy.signal as scp
from tvb.simulator.lab import *
from tvb.datatypes import time_series
from tvb.basic.config import settings
import matplotlib as mpl
from matplotlib import cm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation


#John Doe, 76 nodes
white_matter = connectivity.Connectivity.from_file()
weights = white_matter.weights
white_matter.speed = np.array([4.0])
positions = white_matter.centres.transpose()

x = positions[0]
y = positions[1]
z = positions[2]

heunint = integrators.HeunDeterministic(dt=2**-6)

mon_tavg = monitors.TemporalAverage(period=2**-2)

what_to_watch = (mon_tavg, )

oscillator = models.Generic2dOscillator(tau = np.array([1]), a = np.array([0.23]),
                                        b = np.array([-0.3333]), c = np.array([0]),
                                        I = np.array([0]), d = np.array([0.1]),
                                        e = np.array([0]), f = np.array([1]),
                                        g = np.array([3]), alpha = np.array([3]),
                                        beta = np.array([0.27]), gamma = np.array([-1]))

white_matter_coupling = coupling.Linear(a=np.array([0.025]))

sim = simulator.Simulator(model = oscillator, connectivity = white_matter,
                          coupling = white_matter_coupling,
                          integrator = heunint, monitors = what_to_watch)

sim.configure()

sim.coupling.a = np.array([0.025])

sim.simulation_length = 1000.0

(time, TAVG), = sim.run()
TAVG = TAVG[:, 0, :, 0]

TAVG1 = TAVG[2000:-1,:]
time1 = time[2000:-1]

#stdT = np.std(TAVG1, axis = 0)

#std_TAVG1 = 

fig = plt.figure()
ax = fig.add_subplot(111, projection = '3d')


def init_plot():
    
    global grph
    grph = ax.scatter(x, y, z, c = TAVG1[0, :], cmap = 'coolwarm', s = 1000)
    
    return grph, 

#use np.where instead of ifs
def update(n):
    
    grph.set_array(TAVG1[n, :])
    
    return grph

TS = FuncAnimation(fig, update, frames = np.arange(0, 13999), init_func = init_plot, interval = 2)
plt.show()


    








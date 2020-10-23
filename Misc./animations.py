import numpy as np
import scipy.signal as scp
import os as os
from tvb.simulator.lab import *
from tvb.datatypes import time_series
from tvb.basic.config import settings
import matplotlib as mpl
from matplotlib import colors
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation

mpl.rcParams['animation.ffmpeg_path'] = r'C:\Users\sebastiR\Miniconda3\envs\TVBEnv\ffmpeg-20200623-ce297b4-win64-static\bin\ffmpeg.exe'
writer = animation.FFMpegWriter(fps=10)

folder1 = r'L:\Lab_JamesR\sebastianR\Data\DelaysVariance\Spiegler and Jirsa Parameters\Animations\MovieWriterTests'

if not os.path.exists(folder1):
    os.makedirs(folder1)

fivetwelve_node = 'geombinsurr_partial_000_Gs.zip'
seventysix_node = 'connectivity_76.zip'
ninetysix_node = 'connectivity_96.zip'
oneninetytwo_node = 'connectivity_192.zip'

network = fivetwelve_node

white_matter = connectivity.Connectivity.from_file(network)
weights = white_matter.weights
white_matter.speed = np.array([2.0])
positions = white_matter.centres.transpose()
simlength = 1000
numnodes = weights.shape[0]

inter_hem_lens = np.full((int(numnodes/2), int(numnodes/2)), 110)

x = positions[0]
y = positions[1]
z = positions[2]

#To split network into quarters, find midpoints
x_range = [np.amin(x), np.amax(x)]
x_mid = 0.5*(x_range[1] + x_range[0])
y_range = [np.amin(y), np.amax(y)]
y_mid = 0.5*(y_range[1] + y_range[0])
z_range = [np.amin(z), np.amax(z)]
z_mid = 0.5*(z_range[1] + z_range[0])

nodes = np.arange(0, numnodes)

left = np.arange(0, int(numnodes/2))
right = np.arange(int(numnodes/2), int(numnodes))
#ant-post vs left-righ axes may vary as x or y axes

if network == seventysix_node or network == oneninetytwo_node:
    
    back = np.where(x > x_mid)[0]
    ant_post_axis_x = True
    
elif network == ninetysix_node:

    back = np.where(y > y_mid)[0]
    ant_post_axis_x = False

elif network == fivetwelve_node:
    
    back = np.where(y < y_mid)[0]
    ant_post_axis_x = False
    
front = np.delete(nodes, back)
under = np.where(z < z_mid)[0]
over = np.delete(nodes, under)


heunint = integrators.HeunDeterministic(dt=2**-4)

mon_tavg = monitors.TemporalAverage(period=2**-2)

what_to_watch = (mon_tavg, )

#Parameters from Spiegler and Jirsa, 2013
oscillator = models.Generic2dOscillator(tau = np.array([1]), a = np.array([0.23]),
                                        b = np.array([-0.3333]), c = np.array([0]),
                                        I = np.array([0]), d = np.array([0.1]),
                                        e = np.array([0]), f = np.array([1]),
                                        g = np.array([3]), alpha = np.array([3]),
                                        beta = np.array([0.27]), gamma = np.array([-1]))

couplings = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]

for c in couplings:

    white_matter_coupling = coupling.Linear(a=np.array([c]))
    sim = simulator.Simulator(model = oscillator, connectivity = white_matter,
                              coupling = white_matter_coupling,
                              integrator = heunint, monitors = what_to_watch)
    sim.configure()

    sim.simulation_length = simlength

    (time, TAVG), = sim.run()
    TAVG = TAVG[:, 0, :, 0]

    TAVG1 = TAVG[2000:-1, :]
    time1 = time[2000:-1]

    fig1 = plt.figure(figsize = (12, 8))
    grid = plt.GridSpec(3, 4, wspace = 0, hspace = 0)
    ax2_frontview = plt.subplot(grid[0, 1])
    ax2_frontview.set_xticks([])
    ax2_frontview.set_yticks([])
    ax2_frontview.set_title('Front')
    ax2_leftview = plt.subplot(grid[1, 0])
    ax2_leftview.set_xticks([])
    ax2_leftview.set_yticks([])
    ax2_leftview.set_title('Left')
    ax2_overview = plt.subplot(grid[1, 1])
    ax2_overview.set_xticks([])
    ax2_overview.set_yticks([])
    ax2_rightview = plt.subplot(grid[1, 2])
    ax2_rightview.set_xticks([])
    ax2_rightview.set_yticks([])
    ax2_rightview.set_title('Right')
    ax2_backview = plt.subplot(grid[2, 1])
    ax2_backview.set_xticks([])
    ax2_backview.set_yticks([])
    ax2_backview.set_xlabel('Back')
    ax2_underview = plt.subplot(grid[1, 3])
    ax2_underview.set_xticks([])
    ax2_underview.set_yticks([])
    ax2_underview.set_title('Bottom')

    if ant_post_axis_x:
                    
        frontview = ax2_frontview.scatter(y[front], z[front], c = TAVG1[0, front], cmap = 'coolwarm', s = 150)
        backview = ax2_backview.scatter(-y[back], z[back], c = TAVG1[0, back], cmap = 'coolwarm', s = 150)
        leftview = ax2_leftview.scatter(x[left], z[left], c = TAVG1[0, left], cmap = 'coolwarm', s = 150)
        rightview = ax2_rightview.scatter(-x[right], z[right], c = TAVG1[0, right], cmap = 'coolwarm', s = 150)
        overview = ax2_overview.scatter(-y[over], -x[over], c = TAVG1[0, over], cmap = 'coolwarm', s = 150)
        underview = ax2_underview.scatter(x[under], y[under], c = TAVG1[0, under], cmap = 'coolwarm', s = 150)
        
    else:
        
        frontview = ax2_frontview.scatter(-x[front], z[front], c = TAVG1[0, front], cmap = 'coolwarm', s = 150)
        backview = ax2_backview.scatter(x[back], z[back], c = TAVG1[0, back], cmap = 'coolwarm', s = 150)
        leftview = ax2_leftview.scatter(-y[left], z[left], c = TAVG1[0, left], cmap = 'coolwarm', s = 150)
        rightview = ax2_rightview.scatter(y[right], z[right], c = TAVG1[0, right], cmap = 'coolwarm', s = 150)
        overview = ax2_overview.scatter(x[over], -y[over], c = TAVG1[0, over], cmap = 'coolwarm', s = 150)
        underview = ax2_underview.scatter(x[under], y[under], c = TAVG1[0, under], cmap = 'coolwarm', s = 150)


    with writer.saving(fig1, folder1+r'\Coup'+str(c)+'NET.mp4', 100):
        for n in range(TAVG1.shape[0]):
            TAVG_n = TAVG1[n, :]
            frontview.set_array(TAVG_n[front])
            backview.set_array(TAVG_n[back])
            overview.set_array(TAVG_n[over])
            underview.set_array(TAVG_n[under])
            leftview.set_array(TAVG_n[left])
            rightview.set_array(TAVG_n[right])
            writer.grab_frame()

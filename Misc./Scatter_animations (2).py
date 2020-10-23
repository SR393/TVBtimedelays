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

mpl.rcParams['animation.ffmpeg_path'] = r'C:\Users\sebastiR\.conda\envs\TVBEnv\ffmpeg-20200623-ce297b4-win64-static\bin\ffmpeg.exe'

folder1 = r'L:\Lab_JamesR\sebastianR\Data\DelaysVariance\Spiegler and Jirsa Parameters\ScatterAnimations'

if not os.path.exists(folder1):
    os.makedirs(folder1)


#John Doe, 76 nodes
white_matter = connectivity.Connectivity.from_file()
weights = white_matter.weights
white_matter.speed = np.array([4.0])
positions = white_matter.centres.transpose()
simlength = 3000

inter_hem_lens = np.full((38, 38), 110)

x = positions[0]
y = positions[1]
z = positions[2]

nodes = np.arange(0, 76)
left = np.arange(0, 37)
right = np.arange(38, 76)
back_left = np.array([2, 3, 6, 7, 8, 10, 11, 12, 17, 18, 19, 20, 21, 22, 23, 25, 26, 27, 29, 32])
back_right = back_left + 38
back = np.concatenate((back_left, back_right))
top_left = np.array([1, 3, 4, 7, 12 ,13, 14, 15, 16, 17, 18, 19, 25, 26, 27, 28, 29, 35, 36])
top_right = top_left + 38
top = np.concatenate((top_left, top_right))
front = np.delete(nodes, back)
bottom = np.delete(nodes, top)

heunint = integrators.HeunDeterministic(dt=2**-6)

mon_tavg = monitors.TemporalAverage(period=2**-2)

what_to_watch = (mon_tavg, )

#Parameters from Spiegler and Jirsa, 2013
oscillator = models.Generic2dOscillator(tau = np.array([1]), a = np.array([0.23]),
                                        b = np.array([-0.3333]), c = np.array([0]),
                                        I = np.array([0]), d = np.array([0.1]),
                                        e = np.array([0]), f = np.array([1]),
                                        g = np.array([3]), alpha = np.array([3]),
                                        beta = np.array([0.27]), gamma = np.array([-1]))


    
index = np.arange(1, 6)
index_delay_adjustment = 2.5

for l in index:

    left_mean = index_delay_adjustment*l

    savefile = folder1 + '\Mean Delay Left = ' +str(left_mean)
    if not os.path.exists(savefile):
        os.makedirs(savefile)
    
    for r in index:

        right_mean = index_delay_adjustment*r

        left_intra_hem_lens = np.random.gamma(5*l**2, 2/l, (38, 38))
        np.fill_diagonal(left_intra_hem_lens, 0)
        right_intra_hem_lens = np.random.gamma(5*r**2, 2/r, (38, 38))
        np.fill_diagonal(right_intra_hem_lens, 0)

        Q1and3 = np.concatenate((left_intra_hem_lens, inter_hem_lens))
        Q2and4 = np.concatenate((inter_hem_lens, right_intra_hem_lens))

        tract_lens = np.concatenate((Q1and3, Q2and4), axis=1)

        white_matter.tract_lengths = tract_lens
            
        white_matter_coupling = coupling.Linear(a=np.array([0.025]))

        sim = simulator.Simulator(model = oscillator, connectivity = white_matter,
                                  coupling = white_matter_coupling,
                                  integrator = heunint, monitors = what_to_watch)

        sim.configure()

        sim.simulation_length = simlength

        (time, TAVG), = sim.run()
        TAVG = TAVG[:, 0, :, 0]

        TAVG1 = TAVG[2000:-1,:]
        time1 = time[2000:-1]

        fig1 = plt.figure()
        ax1 = fig1.add_subplot(111, projection = '3d')

        def init_plot1():
            
            global grph
            
            grph = ax1.scatter(x, y, z, c = TAVG1[0, :], cmap = 'coolwarm', s = 400)

            return grph, 

        def update1(n):

            grph.set_array(TAVG1[n, :])
            
            return grph, 

        fig2 = plt.figure()
        grid = plt.GridSpec(3, 3, wspace = 0, hspace = 0)
        ax2_frontview = plt.subplot(grid[0, 1])
        ax2_frontview.set_xticks([])
        ax2_frontview.set_yticks([])
        ax2_leftview = plt.subplot(grid[1, 0])
        ax2_leftview.set_xticks([])
        ax2_leftview.set_yticks([])
        ax2_topview = plt.subplot(grid[1, 1])
        ax2_topview.set_xticks([])
        ax2_topview.set_yticks([])
        ax2_rightview = plt.subplot(grid[1, 2])
        ax2_rightview.set_xticks([])
        ax2_rightview.set_yticks([])
        ax2_backview = plt.subplot(grid[2, 1])
        ax2_backview.set_xticks([])
        ax2_backview.set_yticks([])
        
        def init_plot2():

            global frontview
            
            frontview = ax2_frontview.scatter(y[front], z[front], c = TAVG1[0, front], cmap = 'coolwarm', s = 200)
            
            global backview

            backview = ax2_backview.scatter(y[back], z[back], c = TAVG1[0, back], cmap = 'coolwarm', s = 200)
            
            global topview

            topview = ax2_topview.scatter(x[top], y[top], c = TAVG1[0, top], cmap = 'coolwarm', s = 200)
            
            global leftview

            leftview = ax2_leftview.scatter(x[left], z[left], c = TAVG1[0, left], cmap = 'coolwarm', s = 200)
            
            global rightview

            rightview = ax2_rightview.scatter(-x[right], z[right], c = TAVG1[0, right], cmap = 'coolwarm', s = 200)

            return frontview, backview, topview, leftview, rightview
            

        def update2(n):

            frontview.set_array(TAVG1[n, front])
            backview.set_array(TAVG1[n, back])
            topview.set_array(TAVG1[n, top])
            leftview.set_array(TAVG1[n, left])
            rightview.set_array(TAVG1[n, right])

            return frontview, backview, topview, leftview, rightview
        
        writervideo = animation.FFMpegWriter(fps=60)
        
        NET = animation.FuncAnimation(fig2, update2, frames = np.arange(0, simlength*4 - 2001), init_func = init_plot2, interval = 40)
        NET.save(savefile+r'\Net'+str(right_mean)+'RH.mp4', writer=writervideo)

        TS = animation.FuncAnimation(fig1, update1, frames = np.arange(0, simlength*4 - 2001), init_func = init_plot1, interval = 2)
        TS.save(savefile+'\ThreeD'+str(right_mean)+'RH.mp4', writer=writervideo)
        plt.close('all')


    








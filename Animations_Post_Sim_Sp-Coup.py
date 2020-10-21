#Makes animations of data from 1500 - 1600ms for all sims in K-s space

from tvb.simulator.lab import *
import os as os
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.cm as mcm
from matplotlib import colors
import numpy as np
import scipy.io as io

folder = r'D:\Honours Results\LarterBreakspear\LastSims'

wm = connectivity.Connectivity.from_file('geombinsurr_partial_000_Gs.zip')
numnodes = 512
animation_length = 100 #(ms)
simlength = 5000
dt = 2**-4
simsteps = simlength/dt

mpl.rcParams['animation.ffmpeg_path'] = r'C:\Users\Sebastian\miniconda3\envs\TVBEnv\ffmpeg-20200807-fab00b0-win64-static\bin\ffmpeg.exe'
writer = animation.FFMpegWriter(fps=30)

positions = wm.centres.copy().transpose()
x = positions[0]
y = positions[1]
z = positions[2]

x_range = [np.amin(x), np.amax(x)]
x_mid = 0.5*(x_range[1] + x_range[0])
y_range = [np.amin(y), np.amax(y)]
y_mid = 0.5*(y_range[1] + y_range[0])
z_range = [np.amin(z), np.amax(z)]
z_mid = 0.5*(z_range[1] + z_range[0])

nodes = np.arange(0, numnodes)

left = np.arange(0, 256)
right = np.arange(256, 512)
back = np.where(y < y_mid)[0]
front = np.delete(nodes, back)
under = np.where(z < z_mid)[0]
over = np.delete(nodes, under)

couplings = np.array([0.5, 0.6, 0.7, 0.8])
speeds = np.array([40, 60, 80, 100, 120, 140])

for C in couplings:

    coupfolder = folder + r'\Coupling = '+str(C)
    datafolder = coupfolder + r'\Matlab Files'

    for sp in speeds:

        spfolder = coupfolder + '\Speed = '+str(sp)
        data = io.loadmat(datafolder + r'\Sp'+str(sp)+'Coup'+str(C)+'.mat')['data']

        TAVG = np.mean(data.reshape(int(dt*4*simsteps), int(1/(dt*4)), numnodes), axis = 1) #use coarse grained data
        TAVG = 0.1*np.tanh(10*(TAVG+0.15))

        #Set up figure axes for 'brain net' 
        fig = plt.figure(figsize = (12, 8))
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
        ax2_rightview.set_xlabel('Right')
        ax2_backview = plt.subplot(grid[2, 1])
        ax2_backview.set_xticks([])
        ax2_backview.set_yticks([])
        ax2_backview.set_xlabel('Back')
        ax2_underview = plt.subplot(grid[1, 3])
        ax2_underview.set_xticks([])
        ax2_underview.set_yticks([])
        ax2_underview.set_title('Bottom')
        timestamp = plt.subplot(grid[0, 2])
        timest = timestamp.text(0.5, 0.5, '0 ms', transform = timestamp.transAxes, fontsize=15)
        timestamp.set_xticks([])
        timestamp.set_yticks([])

        cmap = mcm.cividis
        norm = colors.Normalize(vmin = np.min(TAVG), vmax = np.max(TAVG))

        #Initial frames
        frontview = ax2_frontview.scatter(-x[front], z[front], c = TAVG[0, front], cmap = 'cividis', s = 150)
        backview = ax2_backview.scatter(x[back], z[back], c = TAVG[0, back], cmap = 'cividis', s = 150)
        leftview = ax2_leftview.scatter(-y[left], z[left], c = TAVG[0, left], cmap = 'cividis', s = 150)
        rightview = ax2_rightview.scatter(y[right], z[right], c = TAVG[0, right], cmap = 'cividis', s = 150)
        overview = ax2_overview.scatter(x[over], -y[over], c = TAVG[0, over], cmap = 'cividis', s = 150)
        underview = ax2_underview.scatter(x[under], y[under], c = TAVG[0, under], cmap = 'cividis', s = 150)

        #make and save animations frame-by-frame
        with writer.saving(fig, spfolder+r'\AnimationSp'+str(sp)+'Coup'+str(C)+'NewNET.mp4', 100):
            for n in range(int(np.rint(animation_length*4))):
                TAVG_n = TAVG[6000+n, :]
                frontview.set_color(cmap(norm(TAVG_n[front])))
                backview.set_color(cmap(norm(TAVG_n[back])))
                overview.set_color(cmap(norm(TAVG_n[over])))
                underview.set_color(cmap(norm(TAVG_n[under])))
                leftview.set_color(cmap(norm(TAVG_n[left])))
                rightview.set_color(cmap(norm(TAVG_n[right])))
                timest.set_text(str((6000+n)/4)+' ms')
                writer.grab_frame()
        plt.close('all')
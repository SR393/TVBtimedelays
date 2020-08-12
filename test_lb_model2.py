#!/usr/bin/env python
# coding: utf-8

"""
This runs short simulations  with the Larter-Breakspear model,
also known as the chaotic oscillator used for metastable brainwaves


Author: Paula Sanz-Leon, QIMRB, 2020
"""

import numpy as np
import scipy.signal as scp
from numpy.linalg import norm
import os as os
from tvb.simulator.lab import *
import matplotlib as mpl
from matplotlib import colors
import matplotlib.cm as mcm
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pycohcorrgram as ihcc

mpl.rcParams['animation.ffmpeg_path'] = r'C:\Users\Sebastian\miniconda3\envs\TVBEnv\ffmpeg-20200807-fab00b0-win64-static\bin\ffmpeg.exe'
writer = animation.FFMpegWriter(fps=30)

folder = r'C:\Users\Sebastian\miniconda3\envs\TVBEnv\Results\LarterBreakspear\brainwaves'

if not os.path.exists(folder):
    os.makedirs(folder)

#used to loop through C and Iext as well
Cs = [0.8]
Is = [0.3]
speeds = [140.0, 120.0, 100.0, 80.0, 60.0]
#speeds = speeds[::-1]

numnodes = 512 #defined prior to selecting connectivity - don't know what might cause brainwaves to disappear...
dt = 2**-4
simlength = 1000
simsteps = simlength/dt

#For interhemispheric cross coherences
left = np.arange(0, int(numnodes/2))
right = np.arange(int(numnodes/2), numnodes)
opts = {"dt":dt, "maxlag":100, "winlen":50, "overlapfrac":0.8, "inputphases":False}
window_steps = opts["winlen"]/dt
window_gap = opts["winlen"]*opts["overlapfrac"]/dt
n_windows = int(np.ceil((simsteps - window_steps)/(window_steps - window_gap))) + 1

#For plotting each IHCC
xticklocs = np.floor(np.linspace(0, n_windows, 11))
for i in range(10):
    xticklocs[i] = int(xticklocs[i])

yticklocs = np.floor(np.linspace(0, 2*opts["maxlag"]/dt + 1, 5))
for i in range(5):
    yticklocs[i] = int(yticklocs[i])
    
# Connectivity Matrix
fivetwelve_node = 'geombinsurr_partial_000_Gs.zip'
seventysix_node = 'connectivity_76.zip'
ninetysix_node = 'connectivity_96.zip'
oneninetytwo_node = 'connectivity_192.zip'
network = fivetwelve_node
white_matter = connectivity.Connectivity.from_file(network)
labels = white_matter.region_labels

#this part randomises delays when required, but this does not reproduce (strong) brain waves
#newtracts = 149.85*np.random.weibull(3.85, (512, 512))
#mag_newtracts = np.sqrt(np.sum(newtracts**2))
#mag_tracts = np.sqrt(np.sum(white_matter.tract_lengths**2))
#FN = 1/(mag_newtracts*mag_tracts)*np.sum(newtracts*white_matter.tract_lengths)
#print(FN)
#white_matter.tract_lengths = newtracts


for C in Cs:
    
    folder1 = folder+r'\c = '+str(C)
    if not os.path.exists(folder1):
        os.makedirs(folder1)
        
    for sp in speeds:

        folder2 = folder1+r'\speed = '+str(sp)
        
        if not os.path.exists(folder2):
            os.makedirs(folder2)
        
        for I in Is:

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
                                                 Iext = np.array([I]), tau_K = np.array([1.0]),
                                                 C = gcs_uncoupled)  

            # in-strength 
            in_strength = white_matter.weights.sum(axis=1)

            #Scale weights by in-strength
            white_matter.weights /= in_strength
            white_matter.speed = np.array([sp])

            ###This section sets up for animations###

            positions = white_matter.centres.transpose()
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

            left = np.arange(0, int(numnodes/2))
            right = np.arange(int(numnodes/2), int(numnodes))

            #ant-post vs left-righ axes may vary as x or y axes depending on network chosen

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

            ##animation set up done##

            # Long-range coupling function is a nonlinear interaction
            coupling_function_par_a = 0.5*QV_max
            coupling_function_par_b = np.array([1.0])
            coupling_function_par_midpoint = VT
            coupling_function_par_sigma = delta_V

            # Use nonlinear coupling as in Roberts et al 2019
            white_matter_coupling = coupling.HyperbolicTangent(a=coupling_function_par_a, 
                                                                   b=coupling_function_par_b,
                                                                   midpoint= coupling_function_par_midpoint,
                                                                   sigma=coupling_function_par_sigma)

            # Integration scheme
            heunint = integrators.HeunDeterministic(dt=dt)

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

            print('Setting History')
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
            data_a = data_a[:, 0, :, 0]

            # Plot results -- two spikes, same thing is produced by 'New_LarterBreakspear_Tests' - difference must be in coupling parameters
            plt.plot(time_a, data_a, 'k')
            plt.savefig(folder2+'\History.png')
            plt.close('all')

            # We can now run the actual simulation of interest
            # 1. Change coupling strength cause we had set it to zero
            gcs_coupled = np.array([C]) # enable long-range coupling
            sim.model.C = gcs_coupled
            print('Simulation Started')
            # 2. Change simulation length to the actual length we want 
            sim.simulation_length = simlength
            simsteps = simlength/dt
            (time, RAW), = sim.run()
            print('Simulation complete')
            RAW = RAW[:, 0, :, 0]

            #filters out some firing activity for lower speeds
            sos = scp.butter(10, 90, 'hp', output = 'sos', fs = 1000/dt)
            FILT = scp.sosfilt(sos, RAW, axis = 0)

            #plot power spectrum of average signal
            freq, power = scp.periodogram(np.mean(FILT, axis = 1), fs = 1000/dt, nfft = 4000/dt)
            freq = freq[0:1600]
            power = power[0:1600]
            plt.plot(freq, power)
            plt.title('Power Spectrum of Average Time Series')
            plt.savefig(folder2+'\C='+str(C)+'I='+str(I)+'Sp='+str(sp)+'AVGPower.png')
            plt.close('all')

            #plot spectrogram of average signal - this does not work, but it is not important at the moment. Will fix later
            freq, spec_times, spec = scp.spectrogram(np.mean(FILT, axis = 1), fs = 1000/dt, window = 'boxcar', nperseg = int(simsteps/50))
            print(spec.shape)
            freq = freq[0:400]

            plt.imshow(spec, aspect = 'auto', cmap = 'cividis')
            plt.savefig(folder2+'\C='+str(C)+'I='+str(I)+'Sp='+str(sp)+'Spectrogram.png')
            plt.close('all')

            folder3 = folder2+r'\RandomPowerSpectra'

            if not os.path.exists(folder3):
                os.makedirs(folder3)

            #random selection of oscillators (but consistent across trials/reruns of script) to look at power spectra
            oscillators_left = np.array([0, 10, 25, 40, 50, 88, 92, 103, 140, 154, 160, 170, 184, 193, 204, 212, 235, 247])
            oscillators_right = 2*oscillators_left

            for i in range(len(oscillators_left)):
                
                oscl = oscillators_left[i]
                oscr = oscillators_right[i]
                
                freq, power = scp.periodogram(FILT[:, oscl], fs = 1000/dt, nfft = 4000/dt)
                freq = freq[0:1600]
                power = power[0:1600]
                plt.plot(freq, power)
                plt.title('Power Spectrum of '+labels[oscl]+' Region')
                plt.savefig(folder3+'\Oscillator'+str(oscl)+'PS.png')
                plt.close('all')

                freq, power = scp.periodogram(FILT[:, oscr], fs = 1000/dt, nfft = 4000/dt)
                freq = freq[0:1600]
                power = power[0:1600]
                plt.plot(freq, power)
                plt.title('Power Spectrum of '+labels[oscr]+' Region')
                plt.savefig(folder3+'\Oscillator'+str(oscr)+'PS.png')
                plt.close('all')
                
            #This is defined out of place, but still don't want to move anything - it is used in the animations
            TAVG = np.mean(FILT.reshape(int(dt*4*simsteps), int(1/(dt*4)), numnodes), axis = 1)
            TAVG = TAVG[200:, :]
            timeavg = np.linspace(50, 2500, 9800)

            #Get interhemispheric cross coherence using pycohcorrgram

            [IHCC, ord1, ord2, ct, cl]  = ihcc.pycohcorrgram(RAW, left, right, opts, False)
            print(IHCC.shape)
            IHCC = IHCC[::-1, :]

            fig = plt.figure(figsize = (12, 10))
            IHCC_plot = plt.imshow(IHCC, aspect = 'auto', cmap = 'coolwarm')
            plt.xticks(xticklocs, np.linspace(0, simlength, 11)/1000)
            plt.yticks(yticklocs, -np.linspace(-opts["maxlag"], opts["maxlag"], 5))
            plt.ylabel('Lag (ms)')
            plt.xlabel('Time (s)')
            plt.title('Interhemispheric Cross Coherence \n Fibre Lengths: Tractography Data   Conduction Speed = '+str(sp)+'mm/ms')
            plt.savefig(folder2+'\C='+str(C)+'I='+str(I)+'Sp='+str(sp)+'IHCC.png')
            plt.close('all')

            #Animation
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

            #This picks from two options depending on the network
            if ant_post_axis_x:
                            
                frontview = ax2_frontview.scatter(y[front], z[front], c = TAVG[0, front], cmap = 'coolwarm', s = 150)
                backview = ax2_backview.scatter(-y[back], z[back], c = TAVG[0, back], cmap = 'coolwarm', s = 150)
                leftview = ax2_leftview.scatter(x[left], z[left], c = TAVG[0, left], cmap = 'coolwarm', s = 150)
                rightview = ax2_rightview.scatter(-x[right], z[right], c = TAVG[0, right], cmap = 'coolwarm', s = 150)
                overview = ax2_overview.scatter(-y[over], -x[over], c = TAVG[0, over], cmap = 'coolwarm', s = 150)
                underview = ax2_underview.scatter(x[under], y[under], c = TAVG[0, under], cmap = 'coolwarm', s = 150)
                
            else:
                
                frontview = ax2_frontview.scatter(-x[front], z[front], c = TAVG[0, front], cmap = 'coolwarm', s = 150)
                backview = ax2_backview.scatter(x[back], z[back], c = TAVG[0, back], cmap = 'coolwarm', s = 150)
                leftview = ax2_leftview.scatter(-y[left], z[left], c = TAVG[0, left], cmap = 'coolwarm', s = 150)
                rightview = ax2_rightview.scatter(y[right], z[right], c = TAVG[0, right], cmap = 'coolwarm', s = 150)
                overview = ax2_overview.scatter(x[over], -y[over], c = TAVG[0, over], cmap = 'coolwarm', s = 150)
                underview = ax2_underview.scatter(x[under], y[under], c = TAVG[0, under], cmap = 'coolwarm', s = 150)

            #makes image, saves to mp4, deletes and repeats
            with writer.saving(fig, folder2+r'\C = '+str(C)+' speed = '+str(sp)+' I0 = '+str(I)+'NET.mp4', 100):
                for n in range(int(np.rint(TAVG.shape[0]/2))):
                    TAVG_n = TAVG[n, :]
                    frontview.set_color(cmap(norm(TAVG_n[front])))
                    backview.set_color(cmap(norm(TAVG_n[back])))
                    overview.set_color(cmap(norm(TAVG_n[over])))
                    underview.set_color(cmap(norm(TAVG_n[under])))
                    leftview.set_color(cmap(norm(TAVG_n[left])))
                    rightview.set_color(cmap(norm(TAVG_n[right])))
                    timest.set_text(str(n/4)+' ms')
                    writer.grab_frame()
            plt.close('all')

#Run simulations changing fibre length topological structure
#with parameters lambda and I

from tvb.simulator.lab import *
import pycohcorrgram as pyccg
import os as os
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.cm as mcm
from matplotlib import colors
import numpy as np
import numpy.linalg as linalg
import scipy.signal as scp
import scipy.io as io
import scipy.special as special

folder = r'D:\Honours Results\LarterBreakspear\InversionTests\ThirdSims(C0.8Sp100)'

if not os.path.exists(folder):
    os.makedirs(folder)

#set up connectivity
wm = connectivity.Connectivity.from_file('geombinsurr_partial_000_Gs.zip')
numnodes = 512
positions = wm.centres
tracts = wm.tract_lengths

#Some oscillator parameters, coupling = 0.8, cond. speed = 100
uncoupled_C = np.array([0.0])
coupled_C = np.array([0.8])
QV_Max = np.array([1.0])
VT = np.array([0.0])
delta_V = np.array([0.65])
wm.speed = np.array([100])

#oscillator model and parameters
oscillator = models.LarterBreakspear(gCa = np.array([1.0]), gNa = np.array([6.7]),
                                     gK= np.array([2.0]), gL = np.array([0.5]),
                                     VCa = np.array([1.0]), VNa = np.array([0.53]),
                                     VK = np.array([-0.7]), VL = np.array([-0.5]),
                                     TCa = np.array([-0.01]), TNa = np.array([0.3]),
                                     TK = np.array([0.0]), d_Ca = np.array([0.15]),
                                     d_Na = np.array([0.15]), d_K = np.array([0.3]),
                                     VT = VT, ZT = np.array([0.0]),
                                     d_V = delta_V, d_Z = np.array([0.65]),
                                     QV_max = QV_Max, QZ_max = np.array([1.0]),
                                     b = np.array([0.1]), phi = np.array([0.7]),
                                     rNMDA = np.array([0.25]), aee = np.array([0.36]),
                                     aie = np.array([2.0]), ane = np.array([1.0]),
                                     aei = np.array([2.0]), ani = np.array([0.4]),
                                     Iext = np.array([0.3]), tau_K = np.array([1.0]),
                                     C = uncoupled_C)  
#simulation parameters
dt = 2**-4
integrator = integrators.HeunDeterministic(dt = dt)
simlength = 5000.0
simsteps = simlength/dt
mon_raw = monitors.Raw()
what_to_watch = (mon_raw, )

#coupling parameters
a = 0.5*QV_Max
b = np.array([1.0])
midpoint = VT
sigma = delta_V
coupling = coupling.HyperbolicTangent(a = a, midpoint = VT,
                                      b = b, sigma = sigma)
in_strengths = wm.weights.sum(axis=1) 
wm.weights /= in_strengths

#variable parameters
ls = np.array([50, 100, 150, 200, 250, 300])
Is = np.array([-1, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1])
#Set up for fibre length sequences
whole_tracts = tracts.flatten()
whole_tracts = whole_tracts[whole_tracts != 0]
f_sequence = np.argsort(np.argsort(whole_tracts))
x = np.arange(0, len(whole_tracts))/len(whole_tracts)

fits = np.zeros((len(ls), len(Is), 512, 512))

#Create fibre length matrices
for i in range(len(ls)):
    folder1 = folder+r'\lambda = '+str(ls[i])
    maxlen = ls[i]*6 #max length when max(p_n) = inf
    if not os.path.exists(folder1):
        os.makedirs(folder1)
    for k in range(len(Is)):
        f_i = ls[i]*np.abs(np.log(1-x))**Is[k]
        f_i[f_i > maxlen] = np.full((len(f_i[f_i > maxlen])), maxlen)
        f_i = f_i[f_sequence]
        f_i = np.reshape(f_i, (512, 511))
        for j in range(512):
            fits[i, k, j, 0:j] = f_i[j, 0:j]
            fits[i, k, j, j+1:] = f_i[j, j:]
    
#set up information for net animations
mpl.rcParams['animation.ffmpeg_path'] = r'C:\Users\Sebastian\miniconda3\envs\TVBEnv\ffmpeg-20200807-fab00b0-win64-static\bin\ffmpeg.exe'
writer = animation.FFMpegWriter(fps=30)

animation_length = 1500 #(ms)
positions = positions.transpose()
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

back = np.where(y < y_mid)[0]
front = np.delete(nodes, back)
under = np.where(z < z_mid)[0]
over = np.delete(nodes, under)
left = np.arange(0, 256)
right = np.arange(256, 512)

counter = 0

for i in range(len(ls)):

    l = ls[i]

    folder1 = folder+r'\lambda = '+str(l)
    if not os.path.exists(folder1):
        os.makedirs(folder1)

    #store data here
    matlab_files = folder1+r'\Matlab Files'
    if not os.path.exists(matlab_files):
            os.makedirs(matlab_files)

    for j in range(len(Is)):

        I = Is[j]

        folder2 = folder1 + r'\k = '+str(k)
        if not os.path.exists(folder2):
            os.makedirs(folder2)

        wm.tract_lengths = fits[i, j] #this sets tract length matrix 

        sim = simulator.Simulator(model = oscillator, connectivity = wm,
                                coupling = coupling, integrator = integrator, 
                                monitors = what_to_watch, simulation_length = 50).configure()

        #Set initial conditions and histories by running short uncoupled simulation
        sim.model.C = uncoupled_C

        #Set initial conditions
        #Replace history 
        v_fixed = 0.0
        sim.history.buffer = v_fixed * np.ones(sim.history.buffer.shape)
        w_fixed = 0.0
        z_fixed = 0.0 

        sim.current_state[0, ...] = v_fixed
        sim.current_state[1, ...] = w_fixed
        sim.current_state[2, ...] = z_fixed
        (time_trans, data_trans), = sim.run()
        data_trans = data_trans[:, 0, :, 0]

        #plot history to check
        plt.plot(time_trans, data_trans, 'k')
        plt.title('History of All Oscillators, Variable V')
        plt.xlabel('Time (ms)')
        plt.savefig(folder2+r'\History.png')
        plt.close('all')

        #start simulation proper
        print('Started Simulation')
        sim.model.C = coupled_C
        sim.simulation_length = simlength+600

        (time, data), = sim.run()
        time = time[0:int(simlength/dt)]
        data = data[int(600/dt):, 0, :, 0]

        io.savemat(matlab_files+r'\lambda'+str(l)+'k'+str(k)+'.mat', {'data':data, 'locs':positions})

        #Make animation of first 1500ms of activity
        print('Making Animation')
        TAVG = np.mean(data.reshape(int(dt*4*simsteps), int(1/(dt*4)), numnodes), axis = 1)#use coarse grained data

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
        with writer.saving(fig, folder2+r'\Animationlambda'+str(l)+'k'+str(k)+'NET.mp4', 100):
            for n in range(int(np.rint(animation_length*4))):
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

        counter += 1
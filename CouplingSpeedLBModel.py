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
    
def trunc_n(x, n):
    return np.trunc(x*10**n)/(10**n)

#For power in find_n_peaks
def normalize(v):
    norm = linalg.norm(v)
    if norm == 0: 
       return v
    return v / norm, norm

#Finds n peaks in given power spectrum x
def find_n_peaks(x, f, n):

    x, norm = normalize(x)

    peaks, D = scp.find_peaks(x, height = 0.005, distance = 10)
    n_peaks = f[peaks]
    n_peak_heights = x[peaks]
    n_peaks = n_peaks[np.argsort(n_peak_heights)]
    n_peaks = n_peaks[::-1]
    n_peaks = n_peaks[0:n]
    n_peak_heights = np.sort(n_peak_heights)
    n_peak_heights = n_peak_heights[::-1]
    n_peak_heights = n_peak_heights[0:n]
    
    n_peak_heights = n_peak_heights[np.argsort(n_peaks)]
    n_peaks = np.sort(n_peaks)

    if len(n_peaks) < n:

        diff = n - len(peaks)
        n_peaks = np.concatenate((n_peaks, np.full(diff, n_peaks[0])))
        n_peak_heights = np.concatenate((n_peak_heights, np.full(diff, n_peak_heights[0])))

    return n_peaks, n_peak_heights

mpl.rcParams['animation.ffmpeg_path'] = r'C:\Users\Sebastian\miniconda3\envs\TVBEnv\ffmpeg-20200807-fab00b0-win64-static\bin\ffmpeg.exe'
writer = animation.FFMpegWriter(fps=30)

folder = r'C:\Users\Sebastian\miniconda3\envs\TVBEnv\Results\LarterBreakspear\CouplingSpeed\HomogeneousDelays'

if not os.path.exists(folder):
    os.makedirs(folder)

#set up connectivity
wm = connectivity.Connectivity.from_file('geombinsurr_partial_000_Gs.zip')
numnodes = 512
positions = wm.centres.transpose()
uncoupled_C = np.array([0.0])
QV_Max = np.array([1.0])
VT = np.array([0.0])
delta_V = np.array([0.65])

#oscillator model and parameters (Spiegler and Jirsa, 2013)
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
#cp = 0.5
mon_raw = monitors.Raw()
what_to_watch = (mon_raw, )
a = 0.5*QV_Max
b = np.array([1.0])
midpoint = VT
sigma = delta_V
coupling = coupling.HyperbolicTangent(a = a, midpoint = VT,
                                      b = b, sigma = sigma)
in_strengths = wm.weights.sum(axis=1) 
wm.weights /= in_strengths

#variable parameters
speeds = np.array([500, 140, 130, 120, 110, 100, 90, 80, 70, 60])
couplings = np.array([ 0.6, 0.7, 0.8, 0.9])

#For interhemispheric cross coherences
left = np.arange(0, int(numnodes/2))
right = np.arange(int(numnodes/2), numnodes)
opts = {"dt":dt, "maxlag":100, "winlen":20, "overlapfrac":0.8, "inputphases":False}
maxlag = opts['maxlag']
window_steps = opts["winlen"]/dt
window_gap = opts["winlen"]*opts["overlapfrac"]/dt
n_windows = int(np.ceil((simsteps - window_steps)/(window_steps - window_gap))) + 1
IHCCs = np.empty((int(2*opts["maxlag"]/dt) + 1, n_windows, len(speeds)*len(couplings)))

#For plotting each IHCC
xticklocs = np.floor(np.linspace(0, n_windows, 11))
for i in range(10):
    xticklocs[i] = int(xticklocs[i])

yticklocs = np.floor(np.linspace(0, 2*opts["maxlag"]/dt + 1, 5))
for i in range(5):
    yticklocs[i] = int(yticklocs[i])
    
#set up information for net animations
animation_length = 1500 #(ms)
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

back = np.where(y < y_mid)[0]
front = np.delete(nodes, back)
under = np.where(z < z_mid)[0]
over = np.delete(nodes, under)

counter = 0

for C in couplings:

    folder1 = folder + r'\Coupling = '+str(C)
    if not os.path.exists(folder1):
        os.makedirs(folder1)

    matlab_files = folder1+'\Matlab Files'
    if not os.path.exists(matlab_files):
            os.makedirs(matlab_files)

    for sp in speeds:

        wm.speed = np.array([sp])
        
        folder2 = folder1 + '\Speed = '+str(sp)
        if not os.path.exists(folder2):
            os.makedirs(folder2)

        sim = simulator.Simulator(model = oscillator, connectivity = wm,
                                coupling = coupling, integrator = integrator, 
                                monitors = what_to_watch, simulation_length = 530).configure()

        sim.model.C = uncoupled_C

        #Set initial conditions
        print(sim.history.buffer.shape)
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

        plt.plot(time_trans, data_trans, 'k')
        plt.title('History of All Oscillators, Variable V')
        plt.xlabel('Time (ms)')
        plt.savefig(folder2+'\History.png')
        plt.close('all')

        print('Started Simulation')
        coupled_C = np.array([C])
        sim.model.C = coupled_C
        sim.simulation_length = simlength+600

        (time, data), = sim.run()
        time = time[0:int(simlength/dt)]
        data = data[int(600/dt):, 0, :, 0]

        sos = scp.butter(10, 90, 'hp', output = 'sos', fs = 1000/dt)
        filtered_data = scp.sosfilt(sos, data, axis = 0)

        io.savemat(matlab_files+'\Sp'+str(sp)+'Coup'+str(C)+'.mat', {'data':filtered_data, 'locs':positions})

        windowlen = 512
        ovlp = 256
        spec, freqs, t, im = plt.specgram(np.mean(filtered_data, axis = 1), NFFT = windowlen, noverlap = ovlp, Fs = int(1000/dt),
                                          aspect = 'auto', cmap = 'cividis')
        #plt.axis(ymin = 0, ymax = 200)
        plt.xlabel('Time (ms)')
        plt.savefig(folder2 + '\SpectrogramSp'+str(sp)+'Coup'+str(C)+'.png')
        plt.close('all')
        
        left = np.arange(0, 256)
        right = np.arange(256, 512)

        [IHCC, ord1, ord2, ct, cl] = pyccg.pycohcorrgram(filtered_data, left, right, opts, False)
        IHCCs[:, :, counter] = IHCC
        plt.imshow(IHCC, aspect = 'auto', cmap = 'coolwarm')
        plt.xticks(xticklocs, trunc_n(np.linspace(0, simlength, len(xticklocs)), 2))
        plt.yticks(yticklocs, trunc_n(np.linspace(0, simlength, len(yticklocs)), 2))
        plt.title('IHCC for Speed = '+str(sp)+', Coupling = '+str(C))
        plt.savefig(folder2+'\IHCCSp'+str(sp)+'Coup'+str(C)+'.png')

        print('Making Animation')

        TAVG = np.mean(filtered_data.reshape(int(dt*4*simsteps), int(1/(dt*4)), numnodes), axis = 1)#use coarse grained data

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
        with writer.saving(fig, folder2+r'\AnimationSpeed'+str(sp)+'Coup'+str(C)+'NET.mp4', 100):
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

fig, axs = plt.subplots(len(speeds), len(couplings), sharex = True, sharey = True, figsize = (15, 15))
d = len(speeds)*len(couplings)
c = 0
s = 0
for idx, ax in enumerate(axs.flat):
    ax.imshow(IHCCs[:, :, idx], cmap = 'coolwarm', aspect = 'auto')
    ax.set_xticks([])
    ax.set_yticks([])
    if idx >= d - len(speeds):
        ax.set_xlabel(str(speeds[s]))
        s += 1
    if idx == 0 % len(speeds):
        ax.set_ylabel(str(couplings[c]))
        c += 1

fig.text(0.04, 0.5, 'Coupling', va = 'center', rotation = 'vertical')
fig.text(0.5, 0.04, 'Conduction Speed', ha = 'center')
plt.suptitle('IHCCs Over Coupling and Conduction Speeds')
plt.savefig(folder+r'\IHCCs.png')
plt.close('all')

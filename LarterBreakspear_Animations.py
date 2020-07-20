import numpy as np
from numpy.linalg import norm
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

#Makes net animations for a 512 node network of Larter-Breakspear oscillators, looping
#through values of global conduction speed. Consistently encounters overflow warnings
#and outputs NANs at the moment.

#For power in find_n_peaks
def normalize(v):
    
    norml = norm(v)
    
    if norml == 0: 
       return v
    
    return v / norml, norml

#Finds n peaks in given power spectrum x
def find_n_peaks(x, n):

    x, norm = normalize(x)

    peaks, D = scp.find_peaks(x, height = 0.01)
    peakheights = D['peak_heights']

    if len(peaks) > 1:

        peaks_indices = np.argpartition(-peakheights, n-1)
        peaks_indices = peaks_indices[0:n]
        n_peaks = peaks[peaks_indices]
        n_peak_heights = peakheights[peaks_indices]

    else:
        
        n_peaks = np.full(n, peaks[0])
        n_peak_heights = np.full(n, peakheights[0])

    n_peak_heights = n_peak_heights[np.argsort(n_peaks)]
    n_peaks = np.sort(n_peaks)

    return n_peaks, n_peak_heights

mpl.rcParams['animation.ffmpeg_path'] = r'C:\Users\sebastiR\Miniconda3\envs\TVBEnv\ffmpeg-20200623-ce297b4-win64-static\bin\ffmpeg.exe'
writer = animation.FFMpegWriter(fps=30)

folder = r'L:\Lab_JamesR\sebastianR\Data\DelaysVariance\Larter Breakspear Model\Animations'

if not os.path.exists(folder):
    os.makedirs(folder)

fivetwelve_node = 'geombinsurr_partial_000_Gs.zip'
seventysix_node = 'connectivity_76.zip'
ninetysix_node = 'connectivity_96.zip'
oneninetytwo_node = 'connectivity_192.zip'

network = fivetwelve_node

white_matter = connectivity.Connectivity.from_file(network)
weights = white_matter.weights
numnodes = weights.shape[0]
white_matter_coupling = coupling.Linear(a=np.array([0.5]))
simlength = 1000


#For animations
positions = white_matter.centres.transpose()
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


heunint = integrators.HeunDeterministic(dt=2**-4)#First error seems to be encountered
#in the models\larter_breakspear.py file, generally line 472 (RuntimeWarning: overflow encountered in multiply),
#but decreasing timestep can cause this to change. Tried as low as dt = 10**-3, which gave an error instead in
#simulator\integrators.py, line 201 (RuntimeWarning: overflow encountered in add).
#Also tried RungaKutta 4th order, dt = 2**-5, still no luck.

mon_raw = monitors.Raw()

what_to_watch = (mon_raw, )

#Parameters from Spiegler and Jirsa, 2013
oscillator = models.LarterBreakspear(gCa = np.array([1]), gNa = np.array([6.7]),
                                     gK= np.array([2]), gL = np.array([0.5]),
                                     VCa = np.array([1]), VNa = np.array([0.53]),
                                     VK = np.array([-0.7]), VL = np.array([-0.5]),
                                     TCa = np.array([-0.01]), TNa = np.array([0.3]),
                                     TK = np.array([0]), d_Ca = np.array([0.15]),
                                     d_Na = np.array([0.15]), d_K = np.array([0.3]),
                                     VT = np.array([0]), ZT = np.array([0]),
                                     d_V = np.array([0.65]), d_Z = np.array([0.65]),
                                     QV_max = np.array([1]), QZ_max = np.array([1]),
                                     b = np.array([0.1]), phi = np.array([0.7]),
                                     rNMDA = np.array([0.25]), aee = np.array([0.36]),
                                     aie = np.array([2]), ane = np.array([1]),
                                     aei = np.array([2]), ani = np.array([0.4]),
                                     Iext = np.array([0]), tau_K = np.array([1]))  

speed = [1, 2, 3, 4, 5, 6]

for s_p in speed:

    white_matter.speed = np.array([s_p])
    
    sim = simulator.Simulator(model = oscillator, connectivity = white_matter,
                              coupling = white_matter_coupling,
                              integrator = heunint, monitors = what_to_watch)
    sim.configure()

    sim.simulation_length = simlength
    simsteps = simlength*32

    (time, RAW), = sim.run()
    RAW = RAW[:, 0, :, 0]
    RAW = RAW[15999:-1, :]
    time = time[0:-16000]

    print('Sim Complete')

    TAVG = np.mean(RAW.reshape(int(simsteps/8), 8, numnodes), axis = 1)
    timeavg = np.linspace(0, simlength-500, 4*(simlength-500))

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


    with writer.saving(fig, folder+r'\Speed'+str(s_p)+'NET.mp4', 100):
        for n in range(int(np.rint(TAVG.shape[0]))):
            TAVG_n = TAVG[n, :]
            frontview.set_array(TAVG_n[front])
            backview.set_array(TAVG_n[back])
            overview.set_array(TAVG_n[over])
            underview.set_array(TAVG_n[under])
            leftview.set_array(TAVG_n[left])
            rightview.set_array(TAVG_n[right])
            timest.set_text(str(n/4)+' ms')
            writer.grab_frame()
    plt.close('all')

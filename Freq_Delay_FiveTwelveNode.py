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

    peaks, D = scp.find_peaks(x, height = 0.005, distance = 100)
    n_peaks = f[peaks]
    n_peak_heights = x[peaks]*norm
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

mpl.rcParams['animation.ffmpeg_path'] = r'C:\Users\sebastiR\Miniconda3\envs\TVBEnv\ffmpeg-20200802-b48397e-win64-static\bin\ffmpeg.exe'
writer = animation.FFMpegWriter(fps=30)

folder = r'L:\Lab_JamesR\sebastianR\Data\DelaysVariance\Generic2d\512Node\SpieglerandJirsaPars\FrequencyDelay'

#set up connectivity
wm = connectivity.Connectivity.from_file('geombinsurr_partial_000_Gs.zip')
numnodes = 512
positions = wm.centres.transpose()
labels = wm.region_labels.transpose()
nodes = np.arange(1, 513)
nodes = np.array([''.join(item) for item in nodes.astype(str)])
nodes = nodes.reshape(512, 1)

#oscillator model and parameters (Spiegler and Jirsa, 2013)
oscillator = models.Generic2dOscillator(tau = np.array([1]), a = np.array([0.23]),
                                        b = np.array([-0.3333]), c = np.array([0]),
                                        I = np.array([0]), d = np.array([0.1]),
                                        e = np.array([0]), f = np.array([1]),
                                        g = np.array([3]), alpha = np.array([3]),
                                        beta = np.array([0.27]), gamma = np.array([-1]))
#simulation parameters
dt = 2**-4
integrator = integrators.HeunDeterministic(dt = dt)
simlength = 5000
simsteps = simlength/dt
cp = 0.5
mon_raw = monitors.Raw()
what_to_watch = (mon_raw, )
coupling = coupling.Linear(a = np.array([cp]))

#For interhemispheric cross coherences
left = np.arange(0, int(numnodes/2))
right = np.arange(int(numnodes/2), numnodes)
opts = {"dt":dt, "maxlag":100, "winlen":100, "overlapfrac":0.8, "inputphases":False}
maxlag = opts['maxlag']
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
    
#set up information for net animations
animation_length = 500 #(ms)
positions = wm.centres.transpose()
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

#choose from experiments:
# 1. Constant delays (speed varies)
# 2. Distributed delays (Weibull dist. scale parameter (lambda) varies)
# 3. Distributed delays (Weibull dist. shape parameter (k) varies)

One = True
Two = False
Three = False
Four = False

if One:

    folderOne1 = folder + '\Exp1(ConstantDelays)'

    if not os.path.exists(folderOne1):
        os.makedirs(folderOne1)
        
    Datafolder = folderOne1 + '\FrequencyData'
    
    if not os.path.exists(Datafolder):
        os.makedirs(Datafolder)
        
    speeds = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17])
    delays = np.full(len(speeds), 40)/speeds #to plot against delay instead of speed
    tract_lens = np.full((numnodes, numnodes), 40)
    np.fill_diagonal(tract_lens, 0)
    wm.tract_lengths = tract_lens
    numtrials = 4

    #Keep track of everything in text file
    with open(folderOne1+'\info.txt', 'w') as info:
        info.write('Description: \n Defines fibre tract lengths as uniform with length 40mm and runs simulations for different values of the global conduction speed. Outputs graphs of the relationship between oscillation frequency \n'+
                   'and time delay for the average time series and for frequencies averaged over each node. Also outputs net animations of the first '+str(animation_length)+'ms of activity and the interhemispheric cross\n'+
                   'coherence for each simulation. \n'+
                   'Simulation Parameters: \n dt = '+str(dt)+'ms \n Simulation Length (without transient) = '+str(simlength)+'ms \n Integrator - HeunDeterministic \n Monitor - RawData \n'+
                   'Constant Parameters: \n Oscillator - Spiegler and Jirsa, 2013 \n Global Coupling Strength - '+str(cp)+' \n Network - 512 node \n Weight Matrix - unchanged \n Tract Length Matrix - Uniform (40mm)\n'+
                   'Variable Parameters: \n Global Conduction Speed = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] ms \n')
        
    #Store frequency data from each trial
    numexpectedpeaks = 2
    overall_average_timeseries_frequencies = np.empty((numtrials, numexpectedpeaks, len(speeds)))
    overall_average_timeseries_powers = np.empty((numtrials, numexpectedpeaks, len(speeds)))
    
    overall_average_frequency_components = np.empty((numtrials, numexpectedpeaks, len(speeds)))
    overall_average_component_powers = np.empty((numtrials, numexpectedpeaks, len(speeds)))
    
    for trial in range(numtrials):

        folderOne2 = folderOne1 + '\Trial '+str(trial)

        if not os.path.exists(folderOne2):
            os.makedirs(folderOne2)
            
        matlab_files = folderOne2 + '\Matlab Files'
        
        if not os.path.exists(matlab_files):
            os.makedirs(matlab_files)
        
        counter = 0
        #Collect IHCCs in this to make comparison diagram at the end of each simulation
        IHCCplots = np.empty((int(2*opts["maxlag"]/dt) + 1, n_windows, len(speeds)))

        #Store frequency components and powers of average time signal
        average_timeseries_frequencies = np.empty((numexpectedpeaks, len(speeds)))
        average_timeseries_powers = np.empty((numexpectedpeaks, len(speeds)))

        #Store frequency components and powers averaged over individual nodes
        average_frequency_components = np.empty((numexpectedpeaks, len(speeds)))
        average_component_powers = np.empty((numexpectedpeaks, len(speeds)))
    
        for sp in speeds:

            folderOne3 = folderOne2 + '\Speed = '+str(sp)

            if not os.path.exists(folderOne3):
                os.makedirs(folderOne3)
                
            wm.speed = np.array([sp])
            sim = simulator.Simulator(model = oscillator, connectivity = wm,
                                      coupling = coupling, integrator = integrator, 
                                      monitors = what_to_watch, simulation_length = simlength + 500).configure()
            
            #Run simulation
            print('Simulation Started')
            (time, RAW), = sim.run()
            RAW = RAW[int(500/dt):, 0, :, 0]
            time = time[int(500/dt):]
            print('Simulation Complete')

            #Save files to be analysed in neural-flows
            matlab_data = np.mean(RAW.reshape(int(simlength), int(1/(dt)), numnodes), axis = 1)
            matlab_data = matlab_data[0:200]
            io.savemat(matlab_files + '\Trial '+str(trial)+', Speed = '+str(sp)+'.mat', {'data':matlab_data, 'locs':positions, 'nodes_str_lbl':labels, 'nodes_num_lbls':nodes})

            #Find and plot average time signal (over nodes) and corresponding power spectrum
            avg_signal = np.mean(RAW, axis = 1)

            freq, power = scp.periodogram(avg_signal, fs = 1000/dt)

            freq_comps, comp_powers = find_n_peaks(power, freq, numexpectedpeaks)
            comp_powers = np.argsort(freq_comps)
            freq_comps = np.sort(freq_comps)
            average_timeseries_frequencies[:, counter] = freq_comps
            average_timeseries_powers[:, counter] = comp_powers

            freq = freq[0:750]
            power = power[0:750]

            fig, axs = plt.subplots(2, 1, figsize = (15, 15))
            axs[0].plot(time, avg_signal)
            axs[0].set_ylabel('Signal Amplitude')
            axs[0].set_xlabel('Time (ms)')
            axs[1].plot(freq, power)
            axs[1].set_ylabel('Power (V**2/Hz)')
            axs[1].set_xlabel('Frequency (Hz)')
            plt.suptitle('Time Course and Power Spectrum of Average Time Signal')
            plt.savefig(folderOne3+'\TSandPSSp = '+str(sp)+'SV.png')
            plt.close('all')
            
            #interhemispheric cross-coherence - store for composite plot
            [IHCC, ord1, ord2, ct, cl]  = pyccg.pycohcorrgram(RAW, left, right, opts, False)
            IHCC = IHCC[::-1, :]
            IHCCplots[:, :, counter] = IHCC

            #measure instanteneous synchronisation by averaging L and R hemisphere order parameters
            sync_signal = 0.5*(ord1+ord2)

            fig = plt.figure(figsize = (12, 12))
            plt.plot(time, sync_signal)
            plt.ylabel('Phase Order Parameter')
            plt.xlabel('Time (ms)')
            plt.title('Instanteneous Phase Order Parameter, Conduction Speed = '+str(sp))
            plt.savefig(folderOne3+'\IPOP CS = '+str(sp))
            plt.close('all')

            #Find frequencies of individual oscillators and average
            freqs_to_be_averaged = np.empty((numexpectedpeaks, numnodes))
            powers_to_be_averaged = np.empty((numexpectedpeaks, numnodes))
            
            for i in range(numnodes):

                freq, power = scp.periodogram(RAW[:, i], fs = 1000/dt)
                freq_comps, comp_powers = find_n_peaks(power, freq, numexpectedpeaks)
                comp_powers = np.argsort(freq_comps)
                freq_comps = np.sort(freq_comps)
                freqs_to_be_averaged[:, i] = freq_comps
                powers_to_be_averaged[:, i] = comp_powers
                
            average_freqs = np.mean(freqs_to_be_averaged, axis = 1)
            average_powers = np.mean(powers_to_be_averaged, axis = 1)
            average_frequency_components[:, counter] = average_freqs
            average_component_powers[:, counter] = average_powers

            requested = True

            counter += 1
            
            if requested and trial == 0:

                #Makes IHCC plots for first trial only
                fig = plt.figure(figsize = (12, 12))
                plt.imshow(IHCC, aspect = 'auto', cmap = 'coolwarm')
                plt.xlabel('Time')
                plt.ylabel('Lag')
                plt.xticks(xticklocs, np.linspace(0, simlength, len(xticklocs)))
                plt.yticks(yticklocs, np.linspace(-maxlag, maxlag, len(yticklocs)))
                plt.title('Interhemispheric Cross Coherence, Speed = '+str(sp)+' mm/ms')
                plt.savefig(folderOne3+r'\Trial = '+str(trial)+' speed = '+str(sp)+'IHCC.png')
                plt.close('all')

                print('Making Animation')
                #This section makes animations for the first trial only

                TAVG = np.mean(RAW.reshape(int(dt*4*simsteps), int(1/(dt*4)), numnodes), axis = 1)#use coarse grained data

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
                with writer.saving(fig, folderOne3+r'\Trial = '+str(trial)+' speed = '+str(sp)+'NET.mp4', 100):
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
        
        #store trial data plot average time series frequency-delay
        overall_average_timeseries_frequencies[trial] = average_timeseries_frequencies
        overall_average_timeseries_powers[trial] = average_timeseries_powers
        overall_average_frequency_components[trial] = average_frequency_components
        overall_average_component_powers[trial] = average_component_powers
        
        #plot average time series frequency-delay
        fig = plt.figure(figsize = (12, 12))
        grid = plt.GridSpec(4, 1)
        ax1 = plt.subplot(grid[0])
        ax2 = plt.subplot(grid[1:])
        ax1.plot(delays, average_timeseries_powers[0])
        ax1.plot(delays, average_timeseries_powers[1])
        ax1.set_ylabel('Power (V**2/Hz)')
        ax2.plot(delays, average_timeseries_frequencies[0], label = 'Lower Band')
        ax2.plot(delays, average_timeseries_frequencies[1], label = 'Upper Band')
        ax2.set_ylabel('Frequency (Hz)')
        ax2.set_xlabel('Delay (ms)')
        ax2.legend(loc = 'best')
        plt.suptitle('Frequency Components of Average Time Signal Over Delay')
        plt.savefig(folderOne2+'\AvTSFreqsTrial'+str(trial)+'SV.png')
        plt.close('all')

        print(average_component_powers)
        print(average_frequency_components)
        #plot average frequency-delay from individual nodes
        fig = plt.figure(figsize = (12, 12))
        grid = plt.GridSpec(4, 1)
        ax1 = plt.subplot(grid[0])
        ax2 = plt.subplot(grid[1:])
        ax1.plot(delays, average_component_powers[0])
        ax1.plot(delays, average_component_powers[1])
        ax1.set_ylabel('Power (V**2/Hz)')
        ax2.plot(delays, average_frequency_components[0], label = 'Lower Band')
        ax2.plot(delays, average_frequency_components[1], label = 'Upper Band')
        ax2.set_ylabel('Frequency (Hz)')
        ax2.set_xlabel('Delay (ms)')
        ax2.legend(loc = 'best')
        plt.suptitle('Average of Frequency Components of Individual Nodes Over Delay')
        plt.savefig(folderOne2+'\AvFreqsTrial'+str(trial)+'SV.png')
        plt.close('all')
   
    #Get standard deviation and means of frequency information to plot and save
    overall_average_timeseries_frequencies_error = np.std(overall_average_timeseries_frequencies, axis = 0)
    overall_average_timeseries_frequencies = np.mean(overall_average_timeseries_frequencies, axis = 0)
    overall_average_timeseries_powers_error = np.std(overall_average_timeseries_powers, axis = 0)
    overall_average_timeseries_powers = np.mean(overall_average_timeseries_powers, axis = 0)

    overall_average_frequency_components_error = np.std(overall_average_frequency_components, axis = 0)
    overall_average_frequency_components = np.mean(overall_average_frequency_components, axis = 0)
    overall_average_component_powers_error = np.std(overall_average_component_powers, axis = 0)
    overall_average_component_powers = np.mean(overall_average_component_powers, axis = 0)

    #Save
    np.savez(Datafolder+'\AVGTime_Series_Freqs_Means_And_STDs.npz', delays, overall_average_timeseries_frequencies, overall_average_timeseries_frequencies_error)
    np.savez(Datafolder+'\AVGTime_Series_Powers_Means_And_STDs.npz', delays, overall_average_timeseries_powers, overall_average_timeseries_powers_error)
    np.savez(Datafolder+'\AVGFreqs_Means_And_STDs.npz', delays, overall_average_frequency_components, overall_average_frequency_components_error)
    np.savez(Datafolder+'\AVGFreqPowers_Means_And_STDs.npz', delays, overall_average_component_powers, overall_average_component_powers_error)

    #Plot
    #average time series
    fig = plt.figure(figsize = (12, 12))
    grid = plt.GridSpec(4, 1)
    ax1 = plt.subplot(grid[0])
    ax2 = plt.subplot(grid[1:])
    ax1.errorbar(delays, overall_average_timeseries_powers[0], overall_average_timeseries_powers_error[0])
    ax1.errorbar(delays, overall_average_timeseries_powers[1], overall_average_timeseries_powers_error[1])
    ax1.set_ylabel('Power (V**2/Hz)')
    ax2.errorbar(delays, overall_average_timeseries_frequencies[0], overall_average_timeseries_frequencies_error[0], label = 'Lower Band')
    ax2.errorbar(delays, overall_average_timeseries_frequencies[1], overall_average_timeseries_frequencies_error[1], label = 'Upper Band')
    ax2.set_ylabel('Frequency (Hz)')
    ax2.set_xlabel('Delay (ms)')
    ax2.legend(loc = 'best')
    plt.suptitle('Frequency Components of Average Time Signal Over Delay')
    plt.savefig(folderOne1+'\OverallAvTSFreqs.png')
    plt.close('all')

    #from individual nodes
    fig = plt.figure(figsize = (12, 12))
    grid = plt.GridSpec(4, 1)
    ax1 = plt.subplot(grid[0])
    ax2 = plt.subplot(grid[1:])
    ax1.errorbar(delays, overall_average_component_powers[0], overall_average_component_powers_error[0])
    ax1.errorbar(delays, overall_average_component_powers[1], overall_average_component_powers_error[1])
    ax1.set_ylabel('Power (V**2/Hz)')
    ax2.errorbar(delays, overall_average_frequency_components[0], overall_average_frequency_components_error[0], label = 'Lower Band')
    ax2.errorbar(delays, overall_average_frequency_components[1], overall_average_frequency_components_error[1], label = 'Upper Band')
    ax2.set_ylabel('Frequency (Hz)')
    ax2.set_xlabel('Delay (ms)')
    ax2.legend(loc = 'best')
    plt.suptitle('Average of Frequency Components of Individual Nodes Over Delay')
    plt.savefig(folderOne1+'\OverallAvFreqs.png')
    plt.close('all')
    
if Two:

    folderTwo1 = folder + '\Exp2(VaryScale)'

    if not os.path.exists(folderTwo1):
        os.makedirs(folderTwo1)
        
    Datafolder = folderTwo1 + '\FrequencyData'
    
    if not os.path.exists(Datafolder):
        os.makedirs(Datafolder)

    speed = np.array([4.0])
    numtrials = 2
    lambdas = np.array([25, 50, 75, 100, 125, 150, 175, 200])
    
    #Keep track of everything in text file
    with open(folderTwo1+'\info.txt', 'w') as info:
        info.write('Description: \n Runs simulations for different values of the mean of the fibre length distribution, redistributing for each simulation. Outputs graphs of the relationship between oscillation frequency \n'+
                   'and time delay for the average time series and for frequencies averaged over each node. Also outputs net animations of the first '+str(animation_length)+'ms of activity and the interhemispheric cross\n'+
                   'coherence for each simulation. \n'+
                   'Simulation Parameters: \n dt = '+str(dt)+'ms \n Simulation Length (without transient) = '+str(simlength)+'ms \n Integrator - HeunDeterministic \n Monitor - RawData \n'+
                   'Constant Parameters: \n Oscillator - Spiegler and Jirsa, 2013 \n Global Conduction Speed - '+str(speed[0])+'\n Global Coupling Strength - '+str(cp)+' \n Network - 512 node \n Weight Matrix - unchanged\n'+
                   'Variable Parameters: \n Mean Fibre Lengths = '+str())
        
    #Store frequency data from each trial
    numexpectedpeaks = 2
    overall_average_timeseries_frequencies = np.empty((numtrials, numexpectedpeaks, len(speeds)))
    overall_average_timeseries_powers = np.empty((numtrials, numexpectedpeaks, len(speeds)))
    
    overall_average_frequency_components = np.empty((numtrials, numexpectedpeaks, len(speeds)))
    overall_average_component_powers = np.empty((numtrials, numexpectedpeaks, len(speeds)))

    for trial in range(len(numtrials)):
        
        folderTwo2 = folderTwo1 + '\Trial '+str(trial)

        if not os.path.exists(folderTwo2):
            os.makedirs(folderTwo2)
            
        matlab_files = folderTwo2 + '\Matlab Files'
        
        if not os.path.exists(matlab_files):
            os.makedirs(matlab_files)
        
        counter = 0
        #Collect IHCCs in this to make comparison diagram at the end of each simulation
        IHCCplots = np.empty((int(2*opts["maxlag"]/dt) + 1, n_windows, len(lambdas)))

        #Store frequency components and powers of average time signal
        average_timeseries_frequencies = np.empty((numexpectedpeaks, len(lambdas)))
        average_timeseries_powers = np.empty((numexpectedpeaks, len(lambdas)))

        #Store frequency components and powers averaged over individual nodes
        average_frequency_components = np.empty((numexpectedpeaks, len(lambdas)))
        average_component_powers = np.empty((numexpectedpeaks, len(lambdas)))
        
        #compare data against mean, variance, and skewness
        stats = np.empty((3, len(lambdas)))

        for lmda in lambdas:

            folderTwo3 = folderTwo2 + '\lambda = '+str(lmda)

            if not os.path.exists(folderTwo3):
                os.makedirs(folderTwo3)

            tracts = lmda*np.random.weibull(3.85, (512, 512))
            wm.tract_lengths = tracts
            tracts = tracts.flatten()
            tracts = tracts[tracts != 0]
            mu = np.mean(tracts)
            std = np.std(tracts)
            var = std**2
            standardised = (tracts-mu)/std
            skew = np.mean(standardised**3)

            stats[:, counter] = np.array([mu, var, skew])

            sim = simulator.Simulator(model = oscillator, connectivity = wm,
                                      coupling = coupling, integrator = integrator, 
                                      monitors = what_to_watch, simulation_length = simlength + 500).configure()
            #Run simulation
            print('Simulation Started')
            (time, RAW), = sim.run()
            RAW = RAW[int(500/dt):, 0, :, 0]
            time = time[int(500/dt):]
            print('Simulation Complete')

            #Save files to be analysed in neural-flows
            matlab_data = np.mean(RAW.reshape(int(simlength), int(1/(dt)), numnodes), axis = 1)
            matlab_data = matlab_data[0:200]
            io.savemat(matlab_files + '\Trial '+str(trial)+', Speed = '+str(sp)+'.mat', {'data':matlab_data, 'locs':positions, 'nodes_str_lbl':labels, 'nodes_num_lbls':nodes})
            
            
        
    

    
        






        

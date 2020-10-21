import numpy as numpy
import numpy.linalg as nplinalg
import scipy.signal as scp

#Runs simulations changing mean time delay in the right hemisphere (RH) according to
#a uniform distribution. From each, collects time-windowed power spectra and (coarse-grained) time series
#for every oscillator, and saves averages over power spectra time courses of two peaks (frequencies
#and powers). Plots these averages against mean RH delay:
#1. For each oscillator
#2. For right and left hemispheres, using average time series
#3. For right and left hemispheres, using average power spectra
#4. Also plots 3. assuming 1 peak frequency only
#Runs numtrials2 of this process, and averages over them at the end.

def trunc_n(x, n):
    return numpy.trunc(x*10**n)/(10**n)

#For power in find_n_peaks
def normalize(v):
    
    norm = nplinalg.norm(v)
    
    if norm == 0:
       return v
    
    return v / norm, norm

#Finds n peaks in given power spectrum x
def find_n_peaks(x, n):

    x, norm = normalize(x)

    peaks, D = scp.find_peaks(x, height = 0.01)
    peakheights = D['peak_heights']

    if len(peaks) > 1:

        peaks_indices = numpy.argpartition(-peakheights, n-1)
        peaks_indices = peaks_indices[0:n]
        n_peaks = peaks[peaks_indices]
        n_peak_heights = peakheights[peaks_indices]

    else:
        n_peaks = numpy.array([peaks[0], peaks[0]])
        n_peak_heights = numpy.array([peakheights[0], peakheights[0]])

    return [n_peaks, n_peak_heights]


from tvb.simulator.lab import *
from tvb.datatypes import time_series
from tvb.basic.config import settings
import matplotlib.pyplot as plt
import os as os
from pycorrgram import pybuffer

#John Doe, 76 nodes
white_matter = connectivity.Connectivity.from_file()
speed = 2.5
white_matter.speed = numpy.array([speed])
white_matter_coupling = coupling.Linear(a=numpy.array([0.022]))
weights = white_matter.weights

inter_hem_len = numpy.full((38, 38), 120)

heunint = integrators.HeunDeterministic(dt=2**-6)

mon_tavg = monitors.TemporalAverage(period=2**-2)

what_to_watch = (mon_tavg, )

nodes = numpy.arange(0, 74)

#Parameters from Spiegler and Jirsa, 2013
oscillator = models.Generic2dOscillator(tau = numpy.array([1]), a = numpy.array([0.23]),
                                        b = numpy.array([-0.3333]), c = numpy.array([0]),
                                        I = numpy.array([0]), d = numpy.array([0.1]),
                                        e = numpy.array([0]), f = numpy.array([1]),
                                        g = numpy.array([3]), alpha = numpy.array([3]),
                                        beta = numpy.array([0.27]), gamma = numpy.array([-1]))

n = [2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7, 7.5, 8, 8.5, 9, 9.5, 10, 10.5, 11, 11.5, 12, 12.5] #mean fibre length = 10*n 
numtrials = len(n)
numtrials2 = 8
delay_points = numpy.asarray(n)*(10/speed)
print(delay_points)

upto = 1

folder = r'C:\Users\sebra\miniconda3\envs\TVBS2\DelaysVariance\Ghosh et al Parameters\Freq_Delay_Runs\Group5(Random ICsandHistories)(CP0.022 InterDels120 Sp2.5 Var 10)(Uniform Dist)'
if not os.path.exists(folder):
    os.makedirs(folder)

folder1 = folder+'\FreqDelay'
if not os.path.exists(folder1):
    os.makedirs(folder1)

#Set up arrays to store various data for each trial
frequencies_for_plot_right_average = numpy.empty((numtrials2, numtrials)) #RH average time series frequencies
frequencies_for_plot_left_average = numpy.empty((numtrials2, numtrials)) #LH average time series frequencies
osc_frequencies_for_plot_average = numpy.empty((numtrials2, 74, 2, numtrials)) #All oscillator peak frequencies 
osc_powers_for_plot_average = numpy.empty((numtrials2, 74, 2, numtrials)) #All oscillator peak powers 
one_freq_plot_average = numpy.empty((numtrials2, 74, numtrials)) #All oscillator peak frequencies assuming one peak 
Ratios_per_trial = numpy.empty((numtrials2, 74, numtrials)) #All oscillator upper/lower band frequency ratios
Max_delays = numpy.empty((numtrials2, numtrials)) #Max delays RH 

for trial in range(numtrials2):

    upto_in_trial = 0 

    folder2 = folder1 + '\Trial' + str(trial+1)

    folder3 = folder2 + '\IndividualOsc'
    if not os.path.exists(folder3):
        os.makedirs(folder3)

    ##Set up arrays to store various data for each RH mean delay
    frequencies_for_plot_right = []
    frequencies_for_plot_left = []
    osc_frequencies_for_plot = numpy.empty((74, 2, numtrials))
    osc_powers_for_plot = numpy.empty((74, 2, numtrials))
    one_freq_plot = numpy.empty((74, numtrials))
    Ratios_per_delay = numpy.empty((74, numtrials))
    max_delays = []
    
    for intr in n:
        
        print(str(trial)+': '+str(upto_in_trial+1))

        savestring = 'Trial = '+str(trial + 1)+'Right_Mean_Length = '+str(10*intr)
        
        folder4 = folder+'\Trial = '+str(trial + 1)+'Right Mean Length = '+str(10*intr)
        if not os.path.exists(folder4):
            os.makedirs(folder4)

        #LH parameters constant, but fibre lengths randomised every simulation
        left_intra_hem_len = numpy.random.uniform(30 - numpy.sqrt(30), 30 + numpy.sqrt(30), (38, 38))
        numpy.fill_diagonal(left_intra_hem_len, 0)

        #RH parameters vary mean fibre length as 10*intr, keep variance = 10
        right_intra_hem_len = numpy.random.uniform(10*intr - numpy.sqrt(30), 10*intr + numpy.sqrt(30), (38, 38))
        numpy.fill_diagonal(right_intra_hem_len, 0)
        
        Q1and3 = numpy.concatenate((left_intra_hem_len, inter_hem_len))
        Q2and4 = numpy.concatenate((inter_hem_len, right_intra_hem_len))

        tract_len = numpy.concatenate((Q1and3, Q2and4), axis=1)

        white_matter.tract_lengths = tract_len
        wm_delays = tract_len/speed

        rmaxdelay = numpy.max(right_intra_hem_len)/speed
        max_delays.append(rmaxdelay)

        sim = simulator.Simulator(model = oscillator, connectivity = white_matter,
                                  coupling = white_matter_coupling,
                                  integrator = heunint, monitors = what_to_watch)

        sim.configure()

        #Run simulation

        sim.simulation_length = 5000.0

        (time, TAVG), = sim.run()
        TAVG = TAVG[:, 0, :, 0]

        #Remove transient (first 500ms)
        TAVG1 = TAVG[2000:-1,:]
        TAVG1 = numpy.concatenate((TAVG1[:, 0:37], TAVG1[:, 37:-2]), axis = 1)
        time1 = time[2000:-1]

        freq_for_osc_plots = numpy.empty((74, 2))
        powers_for_osc_plots = numpy.empty((74, 2))
        one_freq_plot_this_trial = numpy.empty(74)
        Ratios = numpy.empty(74) #To determine harmonicity
        
        #Create a separate plot for each node, containing power spectrum, time series,
        #and delay/weight inputs from RH nodes
        for l in nodes:

            #Time-windowed power spectra over sim
            nodewinds = pybuffer(TAVG1[:, l], 800, 760)
            nodewinds = nodewinds.transpose()
            nodepowermap = numpy.zeros((1750, 431))
            nodepowermapplot = numpy.zeros((1750, 431)) #Plot needs to be 'upside-down' and contraction mapped (for visualisation)
            
            for k in range(nodewinds.shape[1]):
                
                f, P = scp.periodogram(nodewinds[:,k], fs = 4000, nfft = 80000)
                nodePower = P[0:1750]
                nodepowermap[:, k] = nodePower
                nodePower = nodePower[::-1]
                #index 2/3 contraction maps for now; log(<1) will give off results, don't want upper lim (so no tanh, etc.)
                nodepowermapplot[:, k] = nodePower**(2/3)
            
            oscavgfreq = numpy.mean(nodepowermap, axis = 1)
            
            one_band = find_n_peaks(oscavgfreq, 1)[0]/25 #Extract only one peak to observe 'jump'
            one_freq_plot_this_trial[l] = one_band[0]
            
            [osc_bands, osc_peak_powers] = find_n_peaks(oscavgfreq, 2) #Main event
            osc_peak_powers = osc_peak_powers[numpy.argsort(osc_bands)] #Make sure powers line up with freqs
            osc_bands = numpy.sort(osc_bands)
                
            oscmaxfreqs = osc_bands/25 #Adjust for no. power spectrum data points

            ratio = osc_bands[1]/osc_bands[0]
            ratio = trunc_n(ratio, 2)

            Ratios[l] = ratio

            freq_for_osc_plots[l, :] = oscmaxfreqs
            powers_for_osc_plots[l, :] = osc_peak_powers
            
            #For plotting delay/wight inputs from RH nodes
            delay_input = wm_delays[l, 38:76].reshape(38, 1)
            delay_input = delay_input.transpose()
            weights_input = weights[l, 38:76].reshape(38, 1)
            weights_input = weights_input.transpose()
    
            fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize = (14, 10), gridspec_kw = {'hspace':0.4})
            ax1.imshow(nodepowermapplot, cmap = 'viridis', aspect='auto')
            ax1.set_ylabel("Frequency (Hz)")
            ax1.set_xticks([])
            yticks = numpy.arange(0, 1750, 250)
            ax1.set_yticks(yticks)
            ax1.set_yticklabels(["70","60","50", "40", "30", "20", "10"])
            ax2.plot(time1, TAVG1[:, l])
            ax2.set_xlabel("Time (ms)")
            ax2.set_ylabel("Signal Amplitude")
            del_im = ax3.imshow(delay_input, cmap = 'viridis', aspect='auto')
            ax3.set_yticks([])
            ax3.set_ylabel('Delay Inputs')
            ax3.set_xticks([])
            fig.colorbar(mappable = del_im, ax = ax3)
            ax3.set_xticks([])
            weight_im = ax4.imshow(weights_input, cmap = 'viridis', aspect='auto')
            ax4.set_yticks([])
            ax4.set_ylabel('Weight Inputs')
            ax4.set_xlabel('Oscillators')
            fig.colorbar(mappable = weight_im, ax = ax4)
            ax4.set_xticks(numpy.arange(0, 38, 2))
            ax4.set_xticklabels(numpy.arange(38, 76, 2))
            title = 'Power Spectrum, Time Series, Weight and Delay Inputs, Node ' +str(l)+ ' for '+savestring
            fig.suptitle(title)
            plt.savefig(folder4 +'\Osc'+str(l)+'SV.png', bbox_inches = "tight")
            plt.close("all")
            
        #Store for averaging
        osc_frequencies_for_plot[:, :, upto_in_trial] = freq_for_osc_plots
        osc_powers_for_plot[:, :, upto_in_trial] = powers_for_osc_plots
        one_freq_plot[:, upto_in_trial] = one_freq_plot_this_trial
        Ratios_per_delay[:, upto_in_trial] = Ratios

        #This section finds average time series for L and R hems, then frequencies
        leftavg = numpy.mean(TAVG1[:, 0:37], axis = 1)
        rightavg = numpy.mean(TAVG1[:, 38:75], axis = 1)

        leftwinds = pybuffer(leftavg, 475, 450)
        leftwinds = leftwinds.transpose()
        rightwinds = pybuffer(rightavg, 475, 450)
        rightwinds = rightwinds.transpose()

        leftpowermap = numpy.zeros((3500, 702))
        rightpowermap = numpy.zeros((3500, 702))
        leftpowermapplots = numpy.zeros((3500, 702))
        rightpowermapplots = numpy.zeros((3500, 702))
        
        for j in range(leftwinds.shape[1]):

            lfreq, LeftPower = scp.periodogram(leftwinds[:, j], fs = 4000, nfft = 160000)
            LeftPower = LeftPower[0:3500]
            leftpowermap[:, j] = LeftPower
            LeftPower = LeftPower[::-1]
            leftpowermapplots[:, j] = LeftPower**(2/3)

            rfreq, RightPower = scp.periodogram(rightwinds[:, j], fs = 4000, nfft = 160000)
            RightPower = RightPower[0:3500]
            rightpowermap[:, j] = RightPower
            RightPower = RightPower[::-1]
            rightpowermapplots[:, j] = RightPower**(2/3)

        #method from old sim, will change to find_n_peaks for consistency
        leftavgfreq = numpy.mean(leftpowermap, axis = 1)
        tupl1 = numpy.where(leftavgfreq == numpy.amax(leftavgfreq)) 
        maxleftfreq = tupl1[0]/50
        rightavgfreq = numpy.mean(rightpowermap, axis = 1)
        tupl2 = numpy.where(rightavgfreq == numpy.amax(rightavgfreq))
        maxrightfreq = tupl2[0]/50

        frequencies_for_plot_right.append(maxrightfreq[0])
        frequencies_for_plot_left.append(maxleftfreq[0])

        upto_in_trial = upto_in_trial + 1

    osc_frequencies_for_plot_average[trial, :, :, :] = osc_frequencies_for_plot
    osc_powers_for_plot_average[trial, :, :, :] = osc_powers_for_plot
    one_freq_plot_average[trial, :, :] = one_freq_plot
    
    Harmonic_test = numpy.mean(Ratios_per_delay, axis = 1)
    Ratios_per_trial[trial, :, :] = Ratios_per_delay
    
    max_delays = numpy.array(max_delays)
    max_delays = trunc_n(max_delays, 2)
    Max_delays[trial] = max_delays

    #Plot two frequency bands against RH mean delays for this trial, plus
    #table of ratios to test harmonicity
    for i in nodes:
        
        fig = plt.figure(figsize = (12, 10))
        grid = plt.GridSpec(8, 1, hspace = 0.05)
        powerplt = plt.subplot(grid[0:2])
        freqplt = plt.subplot(grid[2:5])
        tableplt = plt.subplot(grid[6:7])
        
        powerplt.plot(delay_points, osc_powers_for_plot[i,0,:], label = 'Lower Band')
        powerplt.plot(delay_points, osc_powers_for_plot[i,1,:], label = 'Upper Band')
        powerplt.set_xticks([])
        powerplt.set_ylabel('Norm Power')
        
        freqplt.plot(delay_points, osc_frequencies_for_plot[i,0,:], label = 'Lower Band')
        freqplt.plot(delay_points, osc_frequencies_for_plot[i,1,:], label = 'Upper Band')
        freqplt.legend(loc = 'best')
        freqplt.set_xlabel('Delay Mean (ms)')
        freqplt.set_ylabel('Frequency (Hz)')

        tableplt.table(cellText = [delay_points, Ratios_per_delay[i, :]], rowLabels = ['Mean RH Delay', 'Ratio (U/L)'], loc = 'lower center')
        tableplt.set_xticks([])
        tableplt.set_yticks([])
        tableplt.set_frame_on(False)
        
        fig.suptitle('Frequency Band-Delay Relationship, Oscillator '+str(i))
        plt.savefig(folder3+'\Osc '+str(i)+'Save.png')
        plt.close('all')

    #Plot assuming one frequency band
    one_freq_left = numpy.mean(one_freq_plot[0:37], axis = 0)
    one_freq_right = numpy.mean(one_freq_plot[37:74], axis = 0)
    
    #Left Hemisphere
    fig = plt.figure(figsize = (12, 10))
    plt.plot(delay_points, one_freq_left)
    plt.xlabel('Right Hem Delay Mean (ms)')
    plt.ylabel('Frequency (Hz)')
    plt.title('Left Frequency - Right Delay, (One) Frequency Averaged')
    plt.savefig(folder2+'\OneFreqLeftAvFreq.png')
    plt.close('all')

    #Right Hemisphere
    fig = plt.figure(figsize = (12, 10))
    plt.plot(delay_points, one_freq_right)
    plt.xlabel('Delay Mean (ms)')
    plt.ylabel('Frequency (Hz)')
    plt.title('Right Hem Frequency-Delay, (One) Frequency Averaged')
    plt.savefig(folder2+'\OneFreqRightAvFreq.png')
    plt.close('all')
    
    #Plot mean frequencies over each hemisphere from oscillator frequencies
    frequencymeansleft = numpy.mean(osc_frequencies_for_plot[0:37], axis = 0)
    frequencymeansright = numpy.mean(osc_frequencies_for_plot[37:74], axis = 0)

    powermeansleft = numpy.mean(osc_powers_for_plot[0:37], axis = 0)
    powermeansright = numpy.mean(osc_powers_for_plot[37:74], axis = 0)

    harmonic_ratio_left = numpy.mean(Ratios_per_delay[0:37, :], axis = 0) 
    harmonic_ratio_left = trunc_n(harmonic_ratio_left, 2)
    harmonic_ratio_right = numpy.mean(Ratios_per_delay[38:74, :], axis = 0)
    harmonic_ratio_right = trunc_n(harmonic_ratio_right, 2)

    #Left Hemisphere
    fig = plt.figure(figsize = (12, 10))
    grid = plt.GridSpec(8, 1, hspace = 0.05)
    powerplt = plt.subplot(grid[0:2])
    freqplt = plt.subplot(grid[2:5])
    tableplt = plt.subplot(grid[6:7])
    
    powerplt.plot(delay_points, powermeansleft[0,:], label = 'Lower Band')
    powerplt.plot(delay_points, powermeansleft[1,:], label = 'Upper Band')
    powerplt.set_xticks([])
    powerplt.set_ylabel('Norm Power')
    
    freqplt.plot(delay_points, frequencymeansleft[0, :], label = 'Lower Band')
    freqplt.plot(delay_points, frequencymeansleft[1, :], label = 'Upper Band')
    freqplt.set_xlabel('Right Hemisphere Delay Mean (ms)')
    freqplt.set_ylabel('Frequency (Hz)')
    freqplt.legend(loc = 'best')

    tableplt.table(cellText = [delay_points, harmonic_ratio_left], rowLabels = ['Mean RH Delay (ms)', 'Ratio (U/L)'], loc = 'lower center')
    tableplt.set_xticks([])
    tableplt.set_yticks([])
    tableplt.set_frame_on(False)
    
    fig.suptitle('Left Frequency - Right Delay Curve, Frequency Averaged')
    plt.savefig(folder2+'\LeftAvFreq.png')
    plt.close('all')

    #Right Hemisphere, includes max delay for each delay mean in table
    fig = plt.figure(figsize = (12, 10))
    grid = plt.GridSpec(8, 1, hspace = 0.05)
    powerplt = plt.subplot(grid[0:2])
    freqplt = plt.subplot(grid[2:5])
    tableplt = plt.subplot(grid[6:7])
    
    powerplt.plot(delay_points, powermeansright[0,:], label = 'Lower Band')
    powerplt.plot(delay_points, powermeansright[1,:], label = 'Upper Band')
    powerplt.set_xticks([])
    powerplt.set_ylabel('Norm Power')
    
    freqplt.plot(delay_points, frequencymeansright[0, :], label = 'Lower Band')
    freqplt.plot(delay_points, frequencymeansright[1, :], label = 'Upper Band')
    freqplt.set_xlabel('Delay Mean (ms)')
    freqplt.set_ylabel('Frequency (Hz)')
    freqplt.legend(loc = 'best')

    tableplt.table(cellText = [delay_points, harmonic_ratio_right, max_delays], rowLabels = ['Mean Delay (ms)', 'Ratio (U/L)', 'Max Delay (ms)'], loc = 'lower center')
    tableplt.set_xticks([])
    tableplt.set_yticks([])
    tableplt.set_frame_on(False)
    
    fig.suptitle('Right Hem Frequency-Delay Relationship, Frequency Averaged')
    plt.savefig(folder2+'\RightAvFreq.png')
    plt.close('all')

    #Plot mean frequencies over each hemisphere from average time series

    frequencies_for_plot_right = numpy.array(frequencies_for_plot_right)
    frequencies_for_plot_left = numpy.array(frequencies_for_plot_left)

    frequencies_for_plot_right_average[trial, :] = frequencies_for_plot_right
    frequencies_for_plot_left_average[trial, :] = frequencies_for_plot_left

    fig = plt.figure(figsize = (12, 10))
    plt.plot(delay_points, frequencies_for_plot_right)
    plt.title('Right Hem Frequency-Delay Relationship, Time Series Average')
    plt.xlabel('Delay Mean (ms)')
    plt.ylabel('Frequency (Hz)')
    plt.savefig(folder2+'\RSave.png')
    plt.close('all')

    fig = plt.figure(figsize = (12, 10))
    plt.plot(numpy.arange(0, numtrials), frequencies_for_plot_left)
    plt.title('Left Frequency - Right Delay, Time Series Average')
    plt.xlabel('Right Delay Mean (ms)')
    plt.ylabel('Frequency (Hz)')
    plt.savefig(folder2+'\LSave.png')
    plt.close('all')

folder4 = folder1 + '\Averages'
if not os.path.exists(folder4):
    os.makedirs(folder4)

folder5 = folder4 + '\IndividualOsc'
if not os.path.exists(folder5):
    os.makedirs(folder5)

ind_oscillators_averages = numpy.mean(osc_frequencies_for_plot_average, axis = 0)
ind_oscillators_std = numpy.std(osc_frequencies_for_plot_average, axis = 0)
ind_oscillators_power_averages = numpy.mean(osc_powers_for_plot_average, axis = 0)
ind_oscillators_power_std = numpy.std(osc_powers_for_plot_average, axis = 0)

harmonic_ratios = numpy.mean(Ratios_per_trial, axis = 0)
harmonic_ratios = trunc_n(harmonic_ratios, 2)

#Plot oscillator frequency bands against RH mean delay, averaged over all trials
for i in nodes:

    fig = plt.figure(figsize = (12, 10))
    grid = plt.GridSpec(8, 1, hspace = 0.05)
    powerplt = plt.subplot(grid[0:2])
    freqplt = plt.subplot(grid[2:5])
    tableplt = plt.subplot(grid[6:7])
    powerplt.errorbar(delay_points, ind_oscillators_power_averages[i, 0, :], yerr = ind_oscillators_power_std[i, 0, :], label = 'Lower Band', capsize = 5)
    powerplt.errorbar(delay_points, ind_oscillators_power_averages[i, 1, :], yerr = ind_oscillators_power_std[i, 1, :], label = 'Upper Band', capsize = 5)
    powerplt.set_xticks([])
    powerplt.set_ylabel('Norm Power')
    powerplt.legend(loc = 'best')
    
    freqplt.errorbar(delay_points, ind_oscillators_averages[i, 0, :], yerr = ind_oscillators_std[i, 0, :], label = 'Lower Band', capsize = 5)
    freqplt.errorbar(delay_points, ind_oscillators_averages[i, 1, :], yerr = ind_oscillators_std[i, 1, :], label = 'Upper Band', capsize = 5)
    freqplt.set_xlabel('Delay Mean (ms)')
    freqplt.set_ylabel('Frequency (Hz)')
    freqplt.legend(loc = 'best')
    fig.suptitle('Average Frequency-Delay Curve for Oscillator '+str(i))

    tableplt.table(cellText = [delay_points, harmonic_ratios[i, :]], rowLabels = ['Mean RH Delay (ms)', 'Ratio (U/L)'], loc = 'upper center')
    tableplt.set_xticks([])
    tableplt.set_yticks([])
    tableplt.set_frame_on(False)
    
    plt.savefig(folder5 + '\Oscillator ' + str(i) + 'Save.png')
    plt.close('all')
    
#Plot averages
#Assuming one freq, averaged
one_freq_plot_left_average = numpy.mean(one_freq_plot_average[:, 0:37, :], axis = 1)
one_freq_plot_left_std = numpy.std(one_freq_plot_left_average, axis = 0)
one_freq_plot_left_average = numpy.mean(one_freq_plot_left_average, axis = 0)
one_freq_plot_right_average = numpy.mean(one_freq_plot_average[:, 37:74, :], axis = 1)
one_freq_plot_right_std = numpy.std(one_freq_plot_right_average, axis = 0)
one_freq_plot_right_average = numpy.mean(one_freq_plot_right_average, axis = 0)

fig = plt.figure(figsize = (12, 10))
plt.errorbar(delay_points, one_freq_plot_right_average, one_freq_plot_right_std, label = 'Right Hemisphere', capsize = 5)
plt.errorbar(delay_points, one_freq_plot_left_average, one_freq_plot_left_std, label = 'Left Hemisphere', capsize = 5)
plt.xlabel('Right Hem Delay Mean (ms)')
plt.ylabel('Frequency (Hz)')
plt.title('Frequency - Right Hem Delay, Both Hemispheres, (One) Frequency Averaged')
plt.legend(loc = 'best')
plt.text(37.5, 10.5, ' MTD Left Hem = 7.5 ms \n MTD Inter Hem = 30 ms') #Annotate with interhemispheric delays and LH mean delay
plt.savefig(folder4+'\BHOneFreq.png')
plt.close('all')

#Plot Hemispheres separately, averaged over oscillators and trials
oscillator_averages_left = numpy.mean(osc_frequencies_for_plot_average[:, 0:37, :, :], axis = 1) #average over oscillators
oscillator_averages_left_averages = numpy.mean(oscillator_averages_left, axis = 0) #average over trials
oscillator_averages_left_std = numpy.std(oscillator_averages_left, axis = 0)

oscillator_power_averages_left = numpy.mean(osc_powers_for_plot_average[:, 0:37, :, :], axis = 1)
oscillator_power_averages_left_averages = numpy.mean(oscillator_power_averages_left, axis = 0)
oscillator_power_averages_left_std = numpy.std(oscillator_power_averages_left, axis = 0)

oscillator_averages_right = numpy.mean(osc_frequencies_for_plot_average[:, 38:74, :, :], axis = 1)
oscillator_averages_right_averages = numpy.mean(oscillator_averages_right, axis = 0)
oscillator_averages_right_std = numpy.std(oscillator_averages_right, axis = 0)

oscillator_power_averages_right = numpy.mean(osc_powers_for_plot_average[:, 38:74, :, :], axis = 1)
oscillator_power_averages_right_averages = numpy.mean(oscillator_power_averages_right, axis = 0)
oscillator_power_averages_right_std = numpy.std(oscillator_power_averages_right, axis = 0)

harmonic_ratios_left = numpy.mean(harmonic_ratios[0:37, :], axis = 0)
harmonic_ratios_left = trunc_n(harmonic_ratios_left, 2)
harmonic_ratios_right = numpy.mean(harmonic_ratios[38:74, :], axis = 0)
harmonic_ratios_right = trunc_n(harmonic_ratios_right, 2)

#Left Hemisphere, band frequency, power, and ratio against RH mean delay
fig = plt.figure(figsize = (12, 10))
grid = plt.GridSpec(8, 1, hspace = 0.05)
powerplt = plt.subplot(grid[0:2])
freqplt = plt.subplot(grid[2:5])
tableplt = plt.subplot(grid[6:7])

powerplt.errorbar(delay_points, oscillator_power_averages_left_averages[0], oscillator_power_averages_left_std[0], label = 'Lower Band', capsize = 5)
powerplt.errorbar(delay_points, oscillator_power_averages_left_averages[1], oscillator_power_averages_left_std[1], label = 'Upper Band', capsize = 5)
powerplt.set_xticks([])
powerplt.set_ylabel('Norm Power')
powerplt.legend(loc = 'best')

freqplt.errorbar(delay_points, oscillator_averages_left_averages[0], oscillator_averages_left_std[0], label = 'Lower Band', capsize = 5)
freqplt.errorbar(delay_points, oscillator_averages_left_averages[1], oscillator_averages_left_std[1], label = 'Upper Band', capsize = 5)
freqplt.legend(loc = 'best')
freqplt.set_xlabel('Right Hem Delay Mean (ms)')
freqplt.set_ylabel('Frequency (Hz)')

tableplt.table(cellText = [delay_points, harmonic_ratios_left], rowLabels = ['Delay (ms)', 'Ratio (U/L)'], loc = 'upper center')
tableplt.set_xticks([])
tableplt.set_yticks([])
tableplt.set_frame_on(False)

fig.suptitle('Left Frequency - Right Delay Curve, Frequency Averaged')
plt.savefig(folder4+'\LHAvSave.png')
plt.close('all')

#Right Hemisphere, band frequency, power, and ratio against mean delay, average maximum delay for each trial
Max_delays = numpy.mean(Max_delays, axis = 0)
Max_delays = trunc_n(Max_delays, 2)

fig = plt.figure(figsize = (12, 10))
grid = plt.GridSpec(8, 1, hspace = 0.05)
powerplt = plt.subplot(grid[0:2])
freqplt = plt.subplot(grid[2:5])
tableplt = plt.subplot(grid[6:7])

powerplt.errorbar(delay_points, oscillator_power_averages_right_averages[0], oscillator_power_averages_right_std[0], label = 'Lower Band', capsize = 5)
powerplt.errorbar(delay_points, oscillator_power_averages_right_averages[1], oscillator_power_averages_right_std[1], label = 'Upper Band', capsize = 5)
powerplt.set_xticks([])
powerplt.set_ylabel('Norm Power')
powerplt.legend(loc = 'best')

freqplt.errorbar(delay_points, oscillator_averages_right_averages[0], oscillator_averages_right_std[0], label = 'Lower Band', capsize = 5)
freqplt.errorbar(delay_points, oscillator_averages_right_averages[1], oscillator_averages_right_std[1], label = 'Upper Band', capsize = 5)
freqplt.legend(loc = 'best')
freqplt.set_xlabel('Delay Mean (ms)')
freqplt.set_ylabel('Frequency (Hz)')

tableplt.table(cellText = [delay_points, harmonic_ratios_right, Max_delays], rowLabels = ['Delay (ms)', 'Ratio (U/L)', 'Max Delay (ms)'], loc = 'upper center')
tableplt.set_xticks([])
tableplt.set_yticks([])
tableplt.set_frame_on(False)

fig.suptitle('Right Hem Frequency-Delay Relationship, Frequency Averaged')
plt.savefig(folder4+'\RHAvSave.png')
plt.close('all')

#For hemisphere average time series frequencies
frequencies_for_plot_right_average_average = numpy.mean(frequencies_for_plot_right_average, axis = 0)
frequencies_for_plot_left_average_average = numpy.mean(frequencies_for_plot_left_average, axis = 0)

frequencies_for_plot_right_average_std = numpy.std(frequencies_for_plot_right_average, axis = 0)
frequencies_for_plot_left_average_std = numpy.std(frequencies_for_plot_left_average, axis = 0)

fig = plt.figure(figsize = (12, 10))
plt.errorbar(delay_points, frequencies_for_plot_left_average_average, frequencies_for_plot_left_average_std)
plt.title('Left Frequency - Right Delay, Time Series Average')
plt.xlabel('Right Hem Delay Mean (ms)')
plt.ylabel('Frequency (Hz)')
plt.savefig(folder4+'\LSave.png')
plt.close('all')

fig = plt.figure(figsize = (12, 10))
plt.errorbar(delay_points, frequencies_for_plot_right_average_average, frequencies_for_plot_right_average_std)
plt.title('Right Hem Frequency-Delay, Time Series Average')
plt.xlabel('Delay Mean (ms)')
plt.ylabel('Frequency (Hz)')
plt.savefig(folder4+'\RSave.png')
plt.close('all')



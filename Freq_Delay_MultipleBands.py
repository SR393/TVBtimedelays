import numpy as numpy
import numpy.linalg as nplinalg
import scipy.signal as scp

#From internet somewhere

def normalize(v):
    
    norm = nplinalg.norm(v)
    
    if norm == 0: 
       return v
    
    return v / norm, norm

def find_n_peaks(x, n):

    x, norm = normalize(x)

    peaks, D = scp.find_peaks(x, height = 0.005)
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

#Analytic phases for a particular parameter configuration
def pyphase_nodes(y):

    phases = numpy.zeros(y.shape)

    for i in range(y.shape[1]):

        cent = y[:, i] - numpy.mean(y[:, i])
        ansig = scp.hilbert(cent)
        phase1 = numpy.angle(ansig)
        phase2 = numpy.unwrap(phase1)
        phases[:, i] = phase2

    return phases

from tvb.simulator.lab import *
from tvb.datatypes import time_series
from tvb.basic.config import settings
import matplotlib.pyplot as plt
import os as os
from pycohcorrgram import pycohcorrgram
from pycorrgram import pybuffer
from pycorrgram import pycorrgram

#John Doe, 76 nodes
white_matter = connectivity.Connectivity.from_file()
white_matter.speed = numpy.array([4.0])
white_matter_coupling = coupling.Linear(a=numpy.array([0.025]))
weights = white_matter.weights

inter_hem_len = numpy.full((38, 38), 120)

heunint = integrators.HeunDeterministic(dt=2**-6)

mon_tavg = monitors.TemporalAverage(period=2**-2)

what_to_watch = (mon_tavg, )

nodes = numpy.arange(0, 74)
#Variable parameters are intra- and interhemispheric tract length distribution variance, parametrised here by integer n
n = [2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7, 7.5, 8, 8.5, 9]
numtrials = len(n)
numtrials2 = 12
delay_points = numpy.asarray(n)*2.5

#Plot variance of variance and metastability over parameter variations
cnt = 0

upto = 1

folder = r'L:\Lab_JamesR\sebastianR\Data\DelaysVariance\Spiegler and Jirsa Parameters\Freq_Delay_Runs\Group5(Random ICsandHistories)(CP0.025 InterDels120 Sp4.0 Var 10)(Gamma Dist)'
if not os.path.exists(folder):
    os.makedirs(folder)

folder6 = folder+'\FreqDelay'
if not os.path.exists(folder6):
    os.makedirs(folder6)

frequencies_for_plot_right_average = numpy.empty((numtrials2, numtrials))
frequencies_for_plot_left_average = numpy.empty((numtrials2, numtrials))
osc_frequencies_for_plot_average = numpy.empty((numtrials2, 74, 2, numtrials))
osc_powers_for_plot_average = numpy.empty((numtrials2, 74, 2, numtrials))
one_freq_plot_average = numpy.empty((numtrials2, 74, numtrials))

for trial in range(numtrials2):

    upto_in_trial = 0

    folder8 = folder6 + '\Trial' + str(trial+1)

    folder7 = folder8 + '\IndividualOsc'
    if not os.path.exists(folder7):
        os.makedirs(folder7)
        
    frequencies_for_plot_right = []
    frequencies_for_plot_left = []
    osc_frequencies_for_plot = numpy.empty((74, 2, numtrials))
    osc_powers_for_plot = numpy.empty((74, 2, numtrials))
    one_freq_plot = numpy.empty((74, numtrials))
    
    for intr in n:
        
        print(upto)
        print(upto_in_trial+1)
        
        intrvar = 30
        rlmean = intr*10

        left_intra_hem_len = numpy.random.gamma(10*3**2, 1/3, (38, 38))
        numpy.fill_diagonal(left_intra_hem_len, 0)

        right_intra_hem_len = numpy.random.gamma(10*intr**2, 1/intr, (38, 38))
        numpy.fill_diagonal(right_intra_hem_len, 0)
        
        savestring = 'Trial = '+str(trial + 1)+'Right_Mean_Length = '+str(rlmean)
        
        folder4 = folder+'\Trial = '+str(trial + 1)+'Right Mean Length = '+str(rlmean)
        if not os.path.exists(folder4):
            os.makedirs(folder4)

        #Parameters from Spiegler and Jirsa, 2013
        oscillator = models.Generic2dOscillator(tau = numpy.array([1]), a = numpy.array([0.23]),
                                                b = numpy.array([-0.3333]), c = numpy.array([0]),
                                                I = numpy.array([0]), d = numpy.array([0.1]),
                                                e = numpy.array([0]), f = numpy.array([1]),
                                                g = numpy.array([3]), alpha = numpy.array([3]),
                                                beta = numpy.array([0.27]), gamma = numpy.array([-1]))
        
        Q1and3 = numpy.concatenate((left_intra_hem_len, inter_hem_len))
        Q2and4 = numpy.concatenate((inter_hem_len, right_intra_hem_len))

        tract_len = numpy.concatenate((Q1and3, Q2and4), axis=1)

        white_matter.tract_lengths = tract_len
        wm_delays = tract_len/4

        sim = simulator.Simulator(model = oscillator, connectivity = white_matter,
                                  coupling = white_matter_coupling,
                                  integrator = heunint, monitors = what_to_watch)

        sim.configure()

        #Run simulation

        sim.simulation_length = 2000.0

        (time, TAVG), = sim.run()
        TAVG = TAVG[:, 0, :, 0]
        
        TAVG1 = TAVG[2000:-1,:]
        TAVG1 = numpy.concatenate((TAVG1[:, 0:37], TAVG1[:, 37:-2]), axis = 1)
        time1 = time[2000:-1]

        freq_for_osc_plots = numpy.empty((74, 2))
        powers_for_osc_plots = numpy.empty((74, 2))
        one_freq_plot_this_trial = numpy.empty(74)
        #Create a separate plot for each node, containing power spectrum and time series
        for l in nodes:

            nodewinds = pybuffer(TAVG1[:, l], 800, 760)
            nodewinds = nodewinds.transpose()
            nodepowermap = numpy.zeros((1750, 431))
            nodepowermapplot = numpy.zeros((1750, 431))
            
            for k in range(nodewinds.shape[1]):
                
                f, P = scp.periodogram(nodewinds[:,k], fs = 4000, nfft = 80000)
                nodePower = P[0:1750]
                nodepowermap[:, k] = nodePower
                nodePower = nodePower[::-1]
                nodepowermapplot[:, k] = nodePower**(2/3)
            
            oscavgfreq = numpy.mean(nodepowermap, axis = 1)
            
            one_band = find_n_peaks(oscavgfreq, 1)[0]/25
            one_freq_plot_this_trial[l] = one_band[0]
            
            [osc_bands, osc_peak_powers] = find_n_peaks(oscavgfreq, 2)
            osc_peak_powers = osc_peak_powers[numpy.argsort(osc_bands)]
            osc_bands = numpy.sort(osc_bands)
                
            oscmaxfreqs = osc_bands/25

            freq_for_osc_plots[l, :] = oscmaxfreqs
            powers_for_osc_plots[l, :] = osc_peak_powers

            delay_input = wm_delays[l, 38:76].reshape(38, 1)
            delay_input = delay_input.transpose()
            weights_input = weights[l, 38:76].reshape(38, 1)
            weights_input = weights_input.transpose()
            
            fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(14, 10.0))
            ax1.imshow(nodepowermapplot, cmap = 'viridis', aspect='auto')
            ax1.set_ylabel("Frequency")
            ax1.set_xticklabels("")
            yticks = numpy.arange(0, 1750, 250)
            ax1.set_yticks(yticks)
            ax1.set_yticklabels(["70","60","50", "40", "30", "20", "10"])
            ax2.plot(time1, TAVG1[:, l])
            ax2.set_xlabel("Time (ms)")
            ax2.set_ylabel("Signal Amplitude")
            ax3.imshow(delay_input, cmap = 'viridis', aspect='auto')
            ax3.set_yticks([])
            ax3.set_xticks(numpy.arange(0, 38, 2))
            ax3.set_xticklabels(numpy.arange(38, 76, 2))
            ax4.imshow(weights_input, cmap = 'viridis', aspect='auto')
            ax4.set_yticks([])
            ax4.set_xticks(numpy.arange(0, 38, 2))
            ax4.set_xticklabels(numpy.arange(38, 76, 2))
            title = 'Temporal Average - Power Spectrum and Time Series of Node ' +str(l)+ ' for '+savestring
            fig.suptitle(title)
            plt.savefig(folder4 +'\Osc'+str(l)+'SV.png', bbox_inches = "tight")
            plt.close("all")
        
        osc_frequencies_for_plot[:, :, upto_in_trial] = freq_for_osc_plots
        osc_powers_for_plot[:, :, upto_in_trial] = powers_for_osc_plots
        one_freq_plot[:, upto_in_trial] = one_freq_plot_this_trial

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

        
        leftavgfreq = numpy.mean(leftpowermap, axis = 1)
        tupl1 = numpy.where(leftavgfreq == numpy.amax(leftavgfreq))
        maxleftfreq = tupl1[0]/50
        rightavgfreq = numpy.mean(rightpowermap, axis = 1)
        tupl2 = numpy.where(rightavgfreq == numpy.amax(rightavgfreq))
        maxrightfreq = tupl2[0]/50

        frequencies_for_plot_right.append(maxrightfreq[0])
        frequencies_for_plot_left.append(maxleftfreq[0])

        upto = upto + 1
        upto_in_trial = upto_in_trial + 1

    osc_frequencies_for_plot_average[trial, :, :, :] = osc_frequencies_for_plot
    osc_powers_for_plot_average[trial, :, :, :] = osc_powers_for_plot
    one_freq_plot_average[trial, :, :] = one_freq_plot
    
    for i in nodes:
        
        fig = plt.figure()
        grid = plt.GridSpec(3, 1, hspace = 0.05)
        powerplt = plt.subplot(grid[0])
        freqplt = plt.subplot(grid[1:])
        powerplt.plot(delay_points, osc_powers_for_plot[i,0,:], label = 'Lower Band')
        powerplt.plot(delay_points, osc_powers_for_plot[i,1,:], label = 'Upper Band')
        powerplt.legend(loc = 'best')
        powerplt.set_xticks([])
        powerplt.set_ylabel('Norm Power')
        
        freqplt.plot(delay_points, osc_frequencies_for_plot[i,0,:], label = 'Lower Band')
        freqplt.plot(delay_points, osc_frequencies_for_plot[i,1,:], label = 'Upper Band')
        freqplt.legend(loc = 'best')
        freqplt.set_xlabel('Delay Mean (ms)')
        freqplt.set_ylabel('Frequency (Hz)')
        
        fig.suptitle('Frequency Band-Delay Relationship, Oscillator '+str(i))
        plt.savefig(folder7+'\Osc '+str(i)+'Save.png')
        plt.close('all')

    #Plot assuming one frequency band
    one_freq_left = numpy.mean(one_freq_plot[0:37], axis = 0)
    one_freq_right = numpy.mean(one_freq_plot[37:74], axis = 0)
    
    #Left Hemisphere
    plt.plot(delay_points, one_freq_left)
    plt.xlabel('Right Hem Delay Mean (ms)')
    plt.ylabel('Frequency (Hz)')
    plt.title('Left Frequency - Right Delay, (One) Frequency Averaged')
    plt.savefig(folder8+'\OneFreqLeftAvFreq.png')
    plt.close('all')

    #Right Hemisphere
    plt.plot(delay_points, one_freq_right)
    plt.xlabel('Delay Mean (ms)')
    plt.ylabel('Frequency (Hz)')
    plt.title('Right Hem Frequency-Delay, (One) Frequency Averaged')
    plt.savefig(folder8+'\OneFreqRightAvFreq.png')
    plt.close('all')
    
    #Plot mean frequencies over each hemisphere from oscillator frequencies
    thisisgettingwayoutofhand = numpy.mean(osc_frequencies_for_plot[0:37], axis = 0)
    thisisgettingoutofhand = numpy.mean(osc_frequencies_for_plot[37:74], axis = 0)

    powermeansleft = numpy.mean(osc_powers_for_plot[0:37], axis = 0)
    powermeansright = numpy.mean(osc_powers_for_plot[37:74], axis = 0)

    #Left Hemisphere
    fig = plt.figure()
    grid = plt.GridSpec(3, 1, hspace = 0.05)
    powerplt = plt.subplot(grid[0])
    freqplt = plt.subplot(grid[1:])
    powerplt.plot(delay_points, powermeansleft[0,:], label = 'Lower Band')
    powerplt.plot(delay_points, powermeansleft[1,:], label = 'Upper Band')
    powerplt.legend(loc = 'best')
    powerplt.set_xticks([])
    powerplt.set_ylabel('Norm Power')
    
    freqplt.plot(delay_points, thisisgettingwayoutofhand[0, :], label = 'Lower Band')
    freqplt.plot(delay_points, thisisgettingwayoutofhand[1, :], label = 'Upper Band')
    freqplt.set_xlabel('Right Hemisphere Delay Mean (ms)')
    freqplt.set_ylabel('Frequency (Hz)')
    freqplt.legend(loc = 'best')
    
    fig.suptitle('Left Frequency - Right Delay Curve, Frequency Averaged')
    plt.savefig(folder8+'\LeftAvFreq.png')
    plt.close('all')

    #Right Hemisphere
    fig = plt.figure()
    grid = plt.GridSpec(3, 1, hspace = 0.05)
    powerplt = plt.subplot(grid[0])
    freqplt = plt.subplot(grid[1:])
    powerplt.plot(delay_points, powermeansright[0,:], label = 'Lower Band')
    powerplt.plot(delay_points, powermeansright[1,:], label = 'Upper Band')
    powerplt.legend(loc = 'best')
    powerplt.set_xticks([])
    powerplt.set_ylabel('Norm Power')
    
    freqplt.plot(delay_points, thisisgettingoutofhand[0, :], label = 'Lower Band')
    freqplt.plot(delay_points, thisisgettingoutofhand[1, :], label = 'Upper Band')
    freqplt.set_xlabel('Delay Mean (ms)')
    freqplt.set_ylabel('Frequency (Hz)')
    freqplt.legend(loc = 'best')
    
    fig.suptitle('Right Hem Frequency-Delay Relationship, Frequency Averaged')
    plt.savefig(folder8+'\RightAvFreq.png')
    plt.close('all')

    #Plot mean frequencies over each hemisphere from average time series

    frequencies_for_plot_right = numpy.array(frequencies_for_plot_right)
    frequencies_for_plot_left = numpy.array(frequencies_for_plot_left)

    frequencies_for_plot_right_average[trial, :] = frequencies_for_plot_right
    frequencies_for_plot_left_average[trial, :] = frequencies_for_plot_left

    plt.plot(delay_points, frequencies_for_plot_right)
    plt.title('Right Hem Frequency-Delay Relationship, Time Series Average')
    plt.xlabel('Delay Mean (ms)')
    plt.ylabel('Frequency (Hz)')
    plt.savefig(folder8+'\RSave.png')
    plt.close('all')

    plt.plot(numpy.arange(0, numtrials), frequencies_for_plot_left)
    plt.title('Left Frequency - Right Delay, Time Series Average')
    plt.xlabel('Right Delay Mean (ms)')
    plt.ylabel('Frequency (Hz)')
    plt.savefig(folder8+'\LSave.png')
    plt.close('all')

folder9 = folder6 + '\Averages'
if not os.path.exists(folder9):
    os.makedirs(folder9)

folder10 = folder9 + '\IndividualOsc'
if not os.path.exists(folder10):
    os.makedirs(folder10)

ind_oscillators_averages = numpy.mean(osc_frequencies_for_plot_average, axis = 0)
ind_oscillators_std = numpy.std(osc_frequencies_for_plot_average, axis = 0)
ind_oscillators_power_averages = numpy.mean(osc_powers_for_plot_average, axis = 0)
ind_oscillators_power_std = numpy.std(osc_powers_for_plot_average, axis = 0)

for i in nodes:

    fig = plt.figure()
    grid = plt.GridSpec(3, 1, hspace = 0.05)
    powerplt = plt.subplot(grid[0])
    freqplt = plt.subplot(grid[1:])
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
    
    plt.savefig(folder10 + '\Oscillator ' + str(i) + 'Save.png')
    plt.close('all')

#Assuming one freq, averaged
one_freq_plot_left_average = numpy.mean(one_freq_plot_average[:, 0:37, :], axis = 1)
one_freq_plot_left_std = numpy.std(one_freq_plot_left_average, axis = 0)
one_freq_plot_left_average = numpy.mean(one_freq_plot_left_average, axis = 0)
one_freq_plot_right_average = numpy.mean(one_freq_plot_average[:, 37:74, :], axis = 1)
one_freq_plot_right_std = numpy.std(one_freq_plot_right_average, axis = 0)
one_freq_plot_right_average = numpy.mean(one_freq_plot_right_average, axis = 0)

plt.errorbar(delay_points, one_freq_plot_right_average, one_freq_plot_right_std, label = 'Right Hemisphere', capsize = 5)
plt.errorbar(delay_points, one_freq_plot_left_average, one_freq_plot_left_std, label = 'Left Hemisphere', capsize = 5)
plt.xlabel('Right Hem Delay Mean (ms)')
plt.ylabel('Frequency (Hz)')
plt.title('Frequency - Right Hem Delay, Both Hemispheres, (One) Frequency Averaged')
plt.legend(loc = 'best')
plt.savefig(folder9+'\BHOneFreq.png')
plt.close('all')

#Plot Hemispheres separately, averaged over oscillators and trials
oscillator_averages_left = numpy.mean(osc_frequencies_for_plot_average[:, 0:37, :, :], axis = 1)
oscillator_averages_left_averages = numpy.mean(oscillator_averages_left, axis = 0)
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

#Left Hemisphere
fig = plt.figure()
grid = plt.GridSpec(3, 1, hspace = 0.05)
powerplt = plt.subplot(grid[0])
freqplt = plt.subplot(grid[1:])
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

fig.suptitle('Left Frequency - Right Delay Curve, Frequency Averaged')
plt.savefig(folder9+'\LHAvSave.png')
plt.close('all')

#Right Hemisphere
powerplt = plt.subplot(grid[0])
freqplt = plt.subplot(grid[1:])
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

fig.suptitle('Right Hem Frequency-Delay Relationship, Frequency Averaged')
plt.savefig(folder9+'\RHAvSave.png')
plt.close('all')

#For average time series

frequencies_for_plot_right_average_average = numpy.mean(frequencies_for_plot_right_average, axis = 0)
frequencies_for_plot_left_average_average = numpy.mean(frequencies_for_plot_left_average, axis = 0)

frequencies_for_plot_right_average_std = numpy.std(frequencies_for_plot_right_average, axis = 0)
frequencies_for_plot_left_average_std = numpy.std(frequencies_for_plot_left_average, axis = 0)

plt.errorbar(delay_points, frequencies_for_plot_left_average_average, frequencies_for_plot_left_average_std)
plt.title('Left Frequency - Right Delay, Time Series Average')
plt.xlabel('Right Hem Delay Mean (ms)')
plt.ylabel('Frequency (Hz)')
plt.savefig(folder9+'\LSave.png')
plt.close('all')

plt.errorbar(delay_points, frequencies_for_plot_right_average_average, frequencies_for_plot_right_average_std)
plt.title('Right Hem Frequency-Delay, Time Series Average')
plt.xlabel('Delay Mean (ms)')
plt.ylabel('Frequency (Hz)')
plt.savefig(folder9+'\RSave.png')
plt.close('all')



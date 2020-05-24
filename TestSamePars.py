import numpy as numpy
import scipy.signal as scp

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

#Analytic magnitudes
def analyticmag(y):

    mags = numpy.zeros(y.shape)
    print(y.shape)
    
    if len(y.shape) > 1:
        for i in range(y.shape[1]):

            ansig = scp.hilbert(y[:,i])
            mag1 = numpy.absolute(ansig)
            mags[:, i] = mag1
    
    if len(y.shape) == 1:
        ansig = scp.hilbert(y)
        mags = numpy.absolute(ansig)

    return mags

#As per Sanz-Leon et al 2015, no averaging over modes
def Metastability(A):

    mean = numpy.mean(A, axis = 1)
    shifted = numpy.zeros(A.shape)

    for i in range(A.shape[1]):

        y = numpy.absolute(A[:, i] - mean)
        shifted[:,i] = y
    
    V = numpy.mean(shifted, axis = 1)
    ExV = numpy.mean(V)
    ExVsquare = numpy.mean(V**2)
    M = numpy.sqrt(ExVsquare - ExV**2)

    return M
def trunc(values, decs=0):
    return numpy.trunc(values*10**decs)/(10**decs)

from tvb.simulator.lab import *
from tvb.datatypes import time_series
from tvb.basic.config import settings
import matplotlib.pyplot as plt
import os as os
from pycohcorrgram import pycohcorrgram
from pycorrgram import pybuffer

#John Doe, 76 nodes
white_matter = connectivity.Connectivity.from_file()
white_matter.speed = numpy.array([2.0])
white_matter_coupling = coupling.Linear(a=numpy.array([0.025]))

heunint = integrators.HeunDeterministic(dt=2**-6)

mon_tavg = monitors.TemporalAverage(period=2**-2)

what_to_watch = (mon_tavg, )

#Variable parameters are intra- and interhemispheric tract length distribution variance, parametrised here by integer n
n = numpy.arange(1, 4)

#Plot variance of variance and metastability over parameter variations
Varovars = numpy.zeros((3, 3))
cnt = 0

upto = 1

folder2 = r'C:\Users\sebra\miniconda3\envs\TVBS2\DelaysVariance\Spiegler and Jirsa Parameters\HeterogeneousDelays3\Group3(Measure2)\AveragedMeasures'
if not os.path.exists(folder2):
    os.makedirs(folder2)

simlength = 2000

opts = {"dt":(2**-2)*(10**-3), "maxlag":0.07, "winlen":0.125, "overlapfrac":0.9,
        "inputphases":True}
maxlag = int(numpy.rint(opts["maxlag"]/opts["dt"]))
winlen = int(numpy.rint(opts["winlen"]/opts["dt"]))
IHCC_lagaxis = 2*maxlag + 1

IHCC_Plotarray = numpy.zeros((IHCC_lagaxis, 111, 9))
StructMeas1 = numpy.zeros(9)
StructMeas2 = numpy.zeros(9)

for o in n:
    InterVars = []
    for p in n:
        
        print(upto)
        
        folder1 = r'C:\Users\sebra\miniconda3\envs\TVBS2\DelaysVariance\Spiegler and Jirsa Parameters\HeterogeneousDelays3\Group3(Measure2)\Test '+str(o)+', '+str(p)
        if not os.path.exists(folder1):
            os.makedirs(folder1)

        #Parameters from Spiegler and Jirsa, 2013
        oscillator = models.Generic2dOscillator(tau = numpy.array([1]), a = numpy.array([0.23]),
                                                b = numpy.array([-0.3333]), c = numpy.array([0]),
                                                I = numpy.array([0]), d = numpy.array([0.1]),
                                                e = numpy.array([0]), f = numpy.array([1]),
                                                g = numpy.array([3]), alpha = numpy.array([3]),
                                                beta = numpy.array([0.27]), gamma = numpy.array([-1]))
        
        #Homogeneous time delays within intra- and interhemispheric populations

        #intra_asym = numpy.random.uniform(0, 5, (38, 38))
        #inter_asym = numpy.random.uniform(0, 15, (38, 38))
        
        intra_hem_len = numpy.random.gamma(125/3, 0.6, (38, 38))
        numpy.fill_diagonal(intra_hem_len, 0)

        inter_hem_len = numpy.random.gamma(200, 0.5, (38, 38))

        Q1and3 = numpy.concatenate((intra_hem_len, inter_hem_len))
        Q2and4 = numpy.concatenate((inter_hem_len, intra_hem_len))

        tract_len = numpy.concatenate((Q1and3, Q2and4), axis=1)

        avgconn = numpy.mean(white_matter.weights, axis = 0)
        avgdelay = numpy.mean(tract_len, axis = 0)
        consqueezed = avgconn/(numpy.mean(avgconn))
        delsqueezed = avgdelay/(numpy.mean(avgdelay))
        ratio = consqueezed/delsqueezed
        Structsim1 = numpy.mean(ratio)
        Structsim1 = trunc(Structsim1, decs = 6)
        print(Structsim1)

        StructMeas1[upto-1] = Structsim1

        connorm = white_matter.weights/(numpy.linalg.norm(white_matter.weights))
        tractnorm = tract_len/(numpy.linalg.norm(tract_len))
        Structsim2 = numpy.sum(connorm*tractnorm)
        Structsim2 = trunc(Structsim2, decs = 3)
        StructMeas2[upto-1] = Structsim2
        print(Structsim2)
        
        white_matter.tract_lengths = tract_len

        sim = simulator.Simulator(model = oscillator, connectivity = white_matter,
                                  coupling = white_matter_coupling,
                                  integrator = heunint, monitors = what_to_watch)

        
        sim.configure()

        #Perform the simulation
        data = []
        time = []
        
        for tavg in sim(simulation_length=simlength):              
            if not tavg is None:
                time.append(tavg[0][0])
                data.append(tavg[0][1][0,:,0])

        #Make the list a numpy.array for easier use
        TAVG = numpy.array(data)
        TAVG1 = TAVG[2000:-1,:]
        time1 = time[2000:-1]

        #Find phases of analytic signal for each time series
        phases = pyphase_nodes(TAVG1)
        ind = 1j*phases
        ph = numpy.exp(ind)
        coh = numpy.absolute(numpy.mean(ph, axis = 1)) #Phase coherence order parameter

        #Average time series and analytic signal magnitude
        directavg = numpy.mean(TAVG1, axis = 1)
        
        nodes = numpy.arange(0, 76)
        
        winds = pybuffer(directavg, 700, 650)
        winds = winds.transpose()
        powermap = numpy.zeros((500, 107))
        
        for j in range(winds.shape[1]):

            freq, Power = scp.periodogram(winds[:, j], fs = 4000, nfft = 40000)
            Power = Power[0:500]
            Power = Power[::-1]
            contractedpower = Power**(2/3)
            powermap[:, j] = contractedpower
           
        
        #Plot phase coherence + mean time series and analytic signal
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 6.0))
        ax1.imshow(powermap, cmap = 'hot', aspect = 'auto')
        ax1.set_ylabel("Frequency")
        ax1.set_xticklabels("")
        yticks = numpy.arange(0, 500, 100)
        ax1.set_yticks(yticks)
        ax1.set_yticklabels(["50", "40", "30", "20", "10"])
        ax2.plot(time1, coh)
        ax2.set_ylabel("Phase Coherence")
        ax2.set_xticklabels("")
        ax3.plot(time1, directavg)
        ax3.set_ylabel("Average Signal Amplitude")
        ax3.set_xlabel("Time")
        fig.suptitle('Phase Coherence, Average Time Series and Power Map for Test '+str(o)+', '+str(p))
        plt.savefig(folder2 +'\Test '+str(o)+', '+str(p)+' Averages.png', bbox_inches = "tight")
        plt.close("all")
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 6.0))
        f1 = ax1.imshow(white_matter.weights)
        ax1.set_title("Weights Matrix")
        ax1.set_yticklabels("")
        ax1.set_xticklabels("")
        plt.colorbar(f1,ax=ax1)
        f2 = ax2.imshow(tract_len)
        ax2.set_title("Tract Lengths")
        ax2.set_yticklabels("")
        ax2.set_xticklabels("")
        plt.colorbar(f2,ax=ax2)
        plt.savefig(folder2 +'\Test '+str(o)+', '+str(p)+' Weights and Tracts.png', bbox_inches = "tight")
        plt.close("all")

        #Create a separate plot for each node, containing power spectrum and time series
        for l in nodes:

            f, P = scp.periodogram(TAVG1[:,l], fs = 4000, nfft = 160000)
            fig = plt.figure()
            ax1 = fig.add_subplot(121, xlim=(0, 60), xlabel = "Frequency", ylabel = "Power")
            ax1.plot(f, P)
            ax2 = fig.add_subplot(122, xlabel = "Time")
            ax2.plot(time1, TAVG1[:, l])
            title = "Temporal Average - Power Spectrum and Time Series of Node " +str(l)+ " for Test "+str(o)+', '+str(p)
            fig.suptitle(title)
            plt.savefig(folder1+'\Oscillator '+str(l)+'.png', bbox_inches = "tight")
            plt.close("all")

        TSVars = numpy.std(TAVG1, axis = 1)**2
        NodeVars = numpy.std(TSVars)**2
        InterVars.append(NodeVars)

        #Interhemispheric Cross-Coherence
        n1 = numpy.arange(38)
        n2 = numpy.arange(38, 76)
        
        Coh = pycohcorrgram(phases, n1, n2, opts, False)
        print(Coh["cc"][0].shape)
        IHCC_Plotarray[:, :, upto-1] = Coh["cc"][0]
        upto = upto+1
        
        tmarkers = numpy.linspace(0, Coh["cc"][0].shape[1], 9)
        lag = numpy.linspace(0, 2*maxlag + 1, 9)
        
        T = numpy.linspace(0, time1[-1], 9)
        L = numpy.linspace(-maxlag, maxlag, 9)
        
        plt.matshow(Coh["cc"][0], cmap = 'hot', aspect="auto")
        plt.xticks(tmarkers, T)
        plt.yticks(lag, L)
        plt.xlabel('Time')
        plt.ylabel('Lag')
        plt.colorbar()
        plt.title("Interhemispheric Cross-Coherence, Spiegler and Jirsa Parameters, Test "+str(o)+', '+str(p))
        plt.savefig(folder2 + '\Test '+str(o)+', '+str(p)+' InterhemisphericCross-Coherence.png')

    AnotherVars = numpy.array([InterVars])
    Varovars[cnt] = AnotherVars
    cnt = cnt+1

fig, axs = plt.subplots(3, 3, sharex = True, sharey = True)

for idx, ax in enumerate(axs.flat):
    ax.imshow(IHCC_Plotarray[:, :, idx], aspect = "auto")
    ax.set_title(str(StructMeas1[idx])+ ', ' + str(StructMeas1[idx]))
        
plt.tight_layout()
plt.savefig(r'C:\Users\sebra\miniconda3\envs\TVBS2\DelaysVariance\Spiegler and Jirsa Parameters\HeterogeneousDelays3\Group3(Measure2)\IHCCPlot.png')

plot_markers = numpy.linspace(0, 2, 3)

#Plot Variance of Variances    
plt.matshow(Varovars, aspect = "auto")
plt.xlabel('Intrahemispheric Delay VAR')
plt.ylabel('Interhemispheric Delay VAR')
plt.xticks(plot_markers, n)
plt.yticks(plot_markers, n)
plt.colorbar()
plt.title("Variance of Variances for Spiegler and Jirsa Parameters, Heterogeneous Delays")
plt.savefig(r'C:\Users\sebra\miniconda3\envs\TVBS2\DelaysVariance\Spiegler and Jirsa Parameters\HeterogeneousDelays3\Group3(Measure2)\VarianceofVariance.png')
plt.close("all")

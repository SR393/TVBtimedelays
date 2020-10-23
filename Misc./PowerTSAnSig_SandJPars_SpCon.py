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
def AnalyticMag(y):

    mags = numpy.zeros(y.shape)
    print(y.shape)

    for i in range(y.shape[1]):

        ansig = scp.hilbert(y[:,i])
        mag1 = numpy.absolute(ansig)
        mags[:, i] = mag1

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

from tvb.simulator.lab import *
from tvb.datatypes import time_series
from tvb.basic.config import settings
import matplotlib.pyplot as plt
import os as os
from pycohcorrgram import pycohcorrgram

#John Doe, 76 nodes
white_matter = connectivity.Connectivity.from_file()

heunint = integrators.HeunDeterministic(dt=2**-6)

mon_tavg = monitors.TemporalAverage(period=2**-2)

what_to_watch = (mon_tavg, )

#Variable parameters are conduction speed and global coupling
Speeds = numpy.linspace(2, 5, 7)
coup = numpy.linspace(0.015, 0.025, 5)
#Plot variance of variance and metastability over paramter variations
Varovars = numpy.zeros((7, 5))
MetaSt = numpy.zeros((7, 5))
cnt = 0

upto = 1 #Because it takes a while on my machine, not a necessity

folder2 = r'C:\Users\sebra\miniconda3\envs\TVBS2\DelaysVariance\Spiegler and Jirsa Parameters\ConstantDelays\SmallCoupling2\AveragedMeasures'
if not os.path.exists(folder2):
    os.makedirs(folder2)

for k in Speeds:
    CoupVars = []
    CoupMet = []
    for c in coup:
        
        print(upto)
        upto = upto+1
        
        folder1 = r'C:\Users\sebra\miniconda3\envs\TVBS2\DelaysVariance\Spiegler and Jirsa Parameters\ConstantDelays\SmallCoupling2\Coupling = '+str(c)+'Speed = '+str(k)
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
        intra_hem_len = numpy.full((38, 38), 25)
        numpy.fill_diagonal(intra_hem_len, 0)

        inter_hem_len = numpy.full((38, 38), 100)

        Q1and3 = numpy.concatenate((intra_hem_len, inter_hem_len))
        Q2and4 = numpy.concatenate((inter_hem_len, intra_hem_len))

        tract_len = numpy.concatenate((Q1and3, Q2and4), axis=1)

        white_matter.tract_lengths = tract_len
        white_matter.speed = numpy.array([k])
        white_matter_coupling = coupling.Linear(a=numpy.array([c]))

        sim = simulator.Simulator(model = oscillator, connectivity = white_matter,
                                  coupling = white_matter_coupling,
                                  integrator = heunint, monitors = what_to_watch)

        sim.configure()

        #Perform the simulation
        data = []
        time = []
        
        for tavg in sim(simulation_length=3000):              
            if not tavg is None:
                time.append(tavg[0][0])
                data.append(tavg[0][1][0,:,0])

        #Make the list a numpy.array for easier use
        TAVG = numpy.array(data)
        TAVG1 = TAVG[4095:-1,:]
        time1 = time[4095:-1]

        #Find phases and magnitudes of analytic signal for each time series
        phases = pyphase_nodes(TAVG1)
        ind = 1j*phases
        p = numpy.exp(ind)
        coh = numpy.absolute(numpy.mean(p, axis = 1))
        mags = AnalyticMag(TAVG1)
        avmag = numpy.absolute(numpy.mean(mags, axis = 1))

        directavg = numpy.mean(TAVG1, axis = 1) #Average time series over nodes

        magseries = numpy.array([avmag, directavg]).transpose()

        #Plot analytic signal mean phase, and analytic signal mean magnitude together with mean time series
        fig = plt.figure()
        ax1 = fig.add_subplot(211, xlabel = "Time", ylabel = "Phase Order Parameter")
        ax1.plot(time1, coh)
        ax2 = fig.add_subplot(212, xlabel = "Time")
        ax2.plot(time1, magseries)
        fig.suptitle('Mean Phase Coherence, Analytic Signal Magnitude and Time series for Coupling = '+str(c)+', Speed = '+str(k))
        plt.savefig(folder2 +'\Coupling = '+str(c)+', Speed = '+str(k)+' Averages.png', bbox_inches = "tight")
        plt.close("all")
        
        nodes = numpy.arange(0, 76)

        #Create a separate plot for each node, containing power spectrum and time series
        for l in nodes:

            f, P = scp.periodogram(TAVG1[:,l], fs = 2**12, nfft = 2**18)
            fig = plt.figure()
            ax1 = fig.add_subplot(121, xlim=(0, 60), xlabel = "Frequency", ylabel = "Power")
            ax1.plot(f, P)
            ax2 = fig.add_subplot(122, xlabel = "Time")
            ax2.plot(time1, TAVG1[:, l])
            title = "Temporal Average - Power Spectrum and Time Series of Node " +str(l)+ " for Speed = " +str(k)+', Coupling = ' +str(c)
            fig.suptitle(title)
            plt.savefig(folder1+'\Oscillator '+str(l)+'.png', bbox_inches = "tight")
            plt.close("all")

        TSVars = numpy.std(TAVG1, axis = 1)**2
        NodeVars = numpy.std(TSVars)**2
        CoupVars.append(NodeVars)

        Met = Metastability(TAVG1)
        CoupMet.append(Met)

        #Interhemispheric Cross-Coherence
        n1 = numpy.arange(38)
        n2 = numpy.arange(38, 76)

        opts = {"dt":(2**-2)*(10**-3), "maxlag":0.1, "winlen":0.125, "overlapfrac":0.9,
                    "inputphases":True}

        maxlag = int(numpy.rint(opts["maxlag"]/opts["dt"]))
        
        Coh = pycohcorrgram(phases, n1, n2, opts, False)
        
        tmarkers = numpy.linspace(0, Coh["cc"][0].shape[1], 9)
        lag = numpy.linspace(0, 2*maxlag + 1, 9)
        
        T = numpy.linspace(0, time1[-1], 9)
        L = numpy.linspace(-maxlag, maxlag, 9)
        
        plt.matshow(Coh["cc"][0], aspect="auto")
        plt.xticks(tmarkers, T)
        plt.yticks(lag, L)
        plt.xlabel('Time')
        plt.ylabel('Lag')
        plt.colorbar()
        plt.title("Interhemispheric Cross-Coherence, Spiegler and Jirsa Parameters, Coupling = "+str(c)+", Speed = "+str(k))
        plt.savefig(folder2 + '\Coupling = '+str(c)+', Speed = '+str(k)+' InterhemisphericCross-Coherence.png')

    AnotherVars = numpy.array([CoupVars])
    Varovars[cnt] = AnotherVars
    MetArray = numpy.array(CoupMet)
    MetaSt[cnt] = MetArray
    cnt = cnt+1


coup2 = numpy.linspace(0, 4, 5)
sp2 = numpy.linspace(0, 6, 7)

#Plot Variance of Variances    
plt.matshow(Varovars, aspect = "auto")
plt.xlabel('Coupling')
plt.ylabel('Conduction Speed')
plt.xticks(coup2, coup)
plt.yticks(sp2, Speeds)
plt.colorbar()
plt.title("Variance of Variances for Spiegler and Jirsa Parameters, Small Coupling")
plt.savefig(r'C:\Users\sebra\miniconda3\envs\TVBS2\DelaysVariance\Spiegler and Jirsa Parameters\ConstantDelays\SmallCoupling2\VarianceofVariance.png')
plt.close("all")

#Plot Metastability
plt.matshow(MetaSt, aspect = "auto")
plt.xlabel('Coupling')
plt.ylabel('Conduction Speed')
plt.xticks(coup2, coup)
plt.yticks(sp2, Speeds)
plt.colorbar()
plt.title("Metastability for Spiegler and Jirsa Parameters, Small Coupling")
plt.savefig(r'C:\Users\sebra\miniconda3\envs\TVBS2\DelaysVariance\Spiegler and Jirsa Parameters\ConstantDelays\SmallCoupling2\Metastability.png')


        

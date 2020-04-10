from tvb.simulator.lab import *
from tvb.datatypes import time_series
from tvb.basic.config import settings
import matplotlib.pyplot as plt
import numpy as numpy

intra_hem_len1 = numpy.random.normal(25, 15, (38, 38))
numpy.fill_diagonal(intra_hem_len1, 0)
intra_hem_len = numpy.maximum(intra_hem_len1, intra_hem_len1.transpose())

inter_hem_len1 = numpy.random.normal(100, 50, (38, 38))
inter_hem_len = numpy.maximum(inter_hem_len1, inter_hem_len1.transpose())

Q1and3 = numpy.concatenate((intra_hem_len, inter_hem_len))
Q2and4 = numpy.concatenate((inter_hem_len, intra_hem_len))

tract_len = numpy.concatenate((Q1and3, Q2and4), axis=1)

white_matter = connectivity.Connectivity.from_file()
white_matter.tract_lengths = tract_len
white_matter.speed = numpy.array([4.0])

white_matter_coupling = coupling.Linear(a=numpy.array([0.0154]))

heunint = integrators.HeunDeterministic(dt=2**-6)

#Initialise some Monitors with period in physical time
mon_raw = monitors.Raw()
mon_tavg = monitors.TemporalAverage(period=2**-2)

#Bundle them
what_to_watch = (mon_raw, mon_tavg)

t = numpy.linspace(-1.0, 0.5, 3)
alp = numpy.linspace(0.5, 0.5, 3)
bet = numpy.linspace(0.5, 1.5, 3)

for i in t:
    for k in alp:
        for l in bet:
                
            oscillator = models.Generic2dOscillator(tau=numpy.array([1.25]), I = numpy.array([1.0]), gamma = numpy.array([1.0]), a = numpy.array([i]), g = numpy.array([k]), f = numpy.array([l]), e=numpy.array([0]))

            #Initialise a Simulator -- Model, Connectivity, Integrator, and Monitors.
            sim = simulator.Simulator(model = oscillator, connectivity = white_matter,
                                        coupling = white_matter_coupling, 
                                        integrator = heunint, monitors = what_to_watch)

            sim.configure()

            #Perform the simulation
            raw_data = []
            raw_time = []
            tavg_data = []
            tavg_time = []

            for raw, tavg in sim(simulation_length=2**10):
                if not raw is None:
                    raw_time.append(raw[0])
                    raw_data.append(raw[1])
                        
                if not tavg is None:
                    tavg_time.append(tavg[0])
                    tavg_data.append(tavg[1])

            #Make the lists numpy.arrays for easier use.
            RAW = numpy.array(raw_data)
            TAVG = numpy.array(tavg_data)

            numpy.savez_compressed(r'C:\Users\sebra\miniconda3\envs\TVBscripting\Prelim Data Normal 2\norm2data a = ' + str(i) + ', g = '+ str(k) + ', f = ' + str(l), Normal2Raw = RAW, Normal2TAvg = TAVG)

            #Plot raw time series
            plt.figure(1)
            plt.plot(raw_time, RAW[:, 0, :, 0])
            plt.title("Raw -- State variable 0, a = " + str(i) + ", g = " + str(k) + ", f = " + str(l))
            plt.savefig(r'C:\Users\sebra\miniconda3\envs\TVBscripting\Prelim Figures Normal 2\norm2rawdata a = ' + str(i) + ', g = '+ str(k) + ', f = ' + str(l) + '.png')

            #Plot temporally averaged time series
            plt.figure(2)
            plt.plot(tavg_time, TAVG[:, 0, :, 0])
            plt.title("Temporal average, a = " + str(i) + ", g = " + str(k) + ", f = " + str(l))
            plt.savefig(r'C:\Users\sebra\miniconda3\envs\TVBscripting\Prelim Figures Normal 2\norm2avgdata a = ' + str(i) + ', g = '+ str(k) + ', f = ' + str(l) + '.png')





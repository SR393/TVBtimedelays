#Makes R_local(r) plots using order parameter data collected in OrderPars_SpCoup
#(For empirical fibre lengths, K-s space)

import os as os
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as io

folder = r'D:\Honours Results\LarterBreakspear\LastSims'

couplings = np.array([0.8, 0.7, 0.6, 0.5])
speeds = np.array([40, 60, 80, 100, 120, 140])
#radii of inclusion
thrshs = np.linspace(0, 25, 16)

data_for_plot = np.empty((len(couplings)*len(speeds), 16))
lines = np.empty((len(couplings)*len(speeds), 16))
cnt = 0
for C in couplings:

    folder1 = folder+r'\Coupling = '+str(C)
    OrdParData = folder1 + r'\OrdParData'

    for sp in speeds:

        #load data
        ordpardata = io.loadmat(OrdParData+r'\Speed'+str(sp)+'.mat')['rlocals'].transpose()
        ordpardata[0] = 1 #first data point otherwise NAN
        ordpardata = ordpardata[:, 0]
        line = np.linspace(ordpardata[0], ordpardata[-1], 16) #Straight line connecting curve ends

        data_for_plot[cnt] = ordpardata
        lines[cnt] = line

        #Make plot for each (K, s)
        plt.plot(thrshs, ordpardata)
        plt.plot(thrshs, line, 'k--')
        plt.ylabel(r'$R_{local}$', size = 15)
        plt.ylim(0.1, 1.1)
        plt.xlabel('Radius of Inclusion (mm)', size = 15)
        plt.title('Local Order Parameter vs Radius of Local Group', size = 17)
        plt.savefig(folder1 + r'\Speed = '+str(sp)+'\OrderParvsWavelengthWithLine.png')
        plt.close('all')

        cnt += 1

#Make composite plot
fig, axs = plt.subplots(len(couplings), len(speeds), sharex = True, sharey = True, figsize = (13, 10), gridspec_kw={'wspace':0, 'hspace':0})
d = len(speeds)*len(couplings)
c = 0
s = 0
for idx, ax in enumerate(axs.flat):
    ax.plot(thrshs, data_for_plot[idx, :])
    ax.plot(thrshs, lines[idx, :], 'k--')
    ax.set_ylim(0.18, 1.1)
    ax.set_xticks([])
    ax.set_yticks([])
    if idx >= d - len(speeds):
        ax.set_xlabel(str(speeds[s]), size = 17)
        s += 1
    if idx % len(speeds) == 0:
        ax.set_ylabel(str(couplings[c]), size = 17)
        c += 1

fig.text(0.04, 0.5, r'Global Coupling Strength', va = 'center', rotation = 'vertical', size = 18)
fig.text(0.5, 0.04, 'Conduction Speed (mm/ms)', ha = 'center', size = 18)
plt.suptitle(r'$R_{local}$(r) Curves Over Global Coupling Strength and Conduction Speed', size = 18)
plt.savefig(folder+r'\rlocals.png')
plt.close('all')

#Make heatmap
classes = np.array([[1, 2, 2, 2, 2, 2], [0, 1, 2, 2, 2, 2],
                    [0, 0, 1, 1, 2, 2], [0, 0, 0, 0, 0, 1]])
plt.imshow(classes, aspect = 'auto', cmap = 'cividis', vmin = 0, vmax = 3)
plt.xticks(np.arange(0, 6), speeds)
plt.yticks(np.arange(0, 4), couplings)
plt.xlabel('Conduction Speed (mm/ms)', size = 15)
plt.ylabel('Coupling Strength', size = 15)
plt.savefig(folder+r'\categoryheatmap.png')
plt.close('all')
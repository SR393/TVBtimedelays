#Creates individual interhemispheric cross-correlations (IHCC) for each simulation in (K, s) space

from tvb.simulator.lab import *
import pycohcorrgram as pyccg
import os as os
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as io

def trunc_n(x, n):
    return np.trunc(x*10**n)/(10**n)

folder = r'D:\Honours Results\LarterBreakspear\LastSims'

speeds = [140, 120, 100, 80, 60, 40]
speeds = speeds[::-1]
couplings = [0.8, 0.7, 0.6, 0.5]

numnodes = 512
dt = 2**-4
simlength = 5000
simsteps = simlength/dt

#Cross correlation of left and right hemispheres
left = np.arange(0, int(numnodes/2))
right = np.arange(int(numnodes/2), numnodes)
#Inputs to pycohcorrgram
opts = {"dt":dt, "maxlag":40, "winlen":20, "overlapfrac":0.5, "inputphases":False}
maxlag = opts['maxlag']
window_steps = opts["winlen"]/dt
window_gap = opts["winlen"]*opts["overlapfrac"]/dt
n_windows = int(np.ceil((simsteps - window_steps)/(window_steps - window_gap))) + 1
#For compilation plot
IHCCs = np.empty((int(2*opts["maxlag"]/dt) + 1, n_windows, len(speeds)*len(couplings)))
counter = 0

#For IHCC tick markers
xticklocs = np.floor(np.linspace(0, n_windows, 6))
for i in range(6):
    xticklocs[i] = int(xticklocs[i])

yticklocs = np.floor(np.linspace(0, 2*opts["maxlag"]/dt + 1, 5))
for i in range(5):
    yticklocs[i] = int(yticklocs[i])

counter = 0
for coup in couplings:

    folder1 = folder + r'\Coupling = '+str(coup)
    matlabfiles = folder1 + '\Matlab Files\\'
    cnt = 1
    for sp in speeds:

        folder2 = folder1 + r'\Speed = '+str(sp)
        data = io.loadmat(matlabfiles + str(cnt)+'Sp'+str(sp)+'Coup'+str(coup))['data']

        [IHCC, ord1, ord2, ct, cl] = pyccg.pycohcorrgram(data, left, right, opts, False) #Makes IHCC
        IHCCs[:, :, counter] = IHCC
        #Plot IHCC for each (K, s)
        plt.imshow(IHCC, aspect = 'auto', cmap = 'coolwarm')
        plt.vlines(150, 0, 1280, colors = 'k', linestyles = 'dotted')
        plt.vlines(160, 0, 1280, colors = 'k', linestyles = 'dotted')
        plt.xticks(xticklocs, trunc_n(np.linspace(0, simlength, len(xticklocs)), 2))
        plt.yticks(yticklocs, trunc_n(np.linspace(-maxlag, maxlag, len(yticklocs)), 2))
        plt.xlabel('Time (ms)', size = 13)
        plt.ylabel('Lag (ms)', size = 13)
        col = plt.colorbar()
        col.set_label('Cross Corr.', rotation = 270)
        plt.title('IHCC of Simulation (K, s) = ('+str(coup)+', '+str(sp)+')')
        plt.savefig(folder2+'\IHCCWMSp'+str(sp)+'Coup'+str(coup)+'New.png')
        plt.close('all')
        
        counter += 1
        cnt += 1

#Make compilation plot
fig, axs = plt.subplots(len(couplings), len(speeds), sharex = True, sharey = True, figsize = (18, 12), gridspec_kw={'wspace':0, 'hspace':0})
d = len(couplings)*len(speeds)
l = 0
i = 0
for idx, ax in enumerate(axs.flat):
    ax.imshow(IHCCs[:, :, idx], cmap = 'coolwarm', aspect = 'auto')
    ax.set_xticks([])
    ax.set_yticks([])
    if idx >= d - len(speeds):
        ax.set_xlabel(str(speeds[i]), size = 16)
        i += 1
    if idx % len(speeds) == 0:
        ax.set_ylabel(str(couplings[l]), size = 16)
        l += 1

fig.text(0.04, 0.5, 'Connectivity Strength', va = 'center', rotation = 'vertical', size = 17)
fig.text(0.5, 0.04, 'Conduction Speed', ha = 'center', size = 17)
plt.suptitle('IHCCs Over Coupling and Speed', size = 17)
plt.savefig(folder+r'\IHCCs.png')
plt.show()
plt.close('all')

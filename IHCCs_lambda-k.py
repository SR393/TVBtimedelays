#Creates individual interhemispheric cross-correlations (IHCC) for each simulation in (lambda, k) space

from tvb.simulator.lab import *
import pycohcorrgram as pyccg
import os as os
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as io

def trunc_n(x, n):
    return np.trunc(x*10**n)/(10**n)

folder = r'D:\Honours Results\LarterBreakspear\RandomSimsFinal'

lambdas = [50, 100, 150, 200, 250, 300]
lambdas = lambdas[::-1]
ks = [4/3, 2.0, 4.0, 24.0]

#Simulation parameters
numnodes = 512
dt = 2**-4
simlength = 5000
simsteps = simlength/dt

#Cross correlation on left and right hemispheres
left = np.arange(0, int(numnodes/2))
right = np.arange(int(numnodes/2), numnodes)
#inputs to pycohcorrgram
opts = {"dt":dt, "maxlag":40, "winlen":20, "overlapfrac":0.5, "inputphases":False}
maxlag = opts['maxlag']
window_steps = opts["winlen"]/dt
window_gap = opts["winlen"]*opts["overlapfrac"]/dt
n_windows = int(np.ceil((simsteps - window_steps)/(window_steps - window_gap))) + 1
#For compilation plot
IHCCs = np.empty((int(2*opts["maxlag"]/dt) + 1, n_windows, len(ks)*len(lambdas)))

#For IHCC tick markers
xticklocs = np.floor(np.linspace(0, n_windows, 6))
for i in range(6):
    xticklocs[i] = int(xticklocs[i])

yticklocs = np.floor(np.linspace(0, 2*opts["maxlag"]/dt + 1, 5))
for i in range(5):
    yticklocs[i] = int(yticklocs[i])

counter = 0
for lam in lambdas:

    folder1 = folder + r'\lambda = '+str(lam)
    matlabfiles = folder1 + r'\Matlab Files'
    cnt = 1

    for k in ks:

        folder2 = folder1 + r'\k = '+str(k)
        data = io.loadmat(matlabfiles + r'\s'+str(cnt)+'k'+str(k)+'lambda'+str(lam))['data']

        [IHCC, ord1, ord2, ct, cl] = pyccg.pycohcorrgram(data, left, right, opts, False) #makes IHCC
        IHCCs[:, :, counter] = IHCC
        #Plot IHCC for each (lambda, k)
        plt.imshow(IHCC, aspect = 'auto', cmap = 'coolwarm')
        plt.xticks(xticklocs, trunc_n(np.linspace(0, simlength, len(xticklocs)), 2))
        plt.yticks(yticklocs, trunc_n(np.linspace(-maxlag, maxlag, len(yticklocs)), 2))
        plt.xlabel('Time (ms)')
        plt.ylabel('Lag (ms)')
        col = plt.colorbar()
        col.set_label('Cross Corr.', rotation = 270)
        plt.savefig(folder2+'\IHCCk'+str(k)+'Lambda'+str(lam)+'New.png')
        plt.close('all')

        counter += 1
        cnt += 1

#Make compilation plot
fig, axs = plt.subplots(len(lambdas), len(ks), sharex = True, sharey = True, figsize = (17, 13), gridspec_kw={'wspace':0, 'hspace':0})
d = len(lambdas)*len(ks)
l = 0
i = 0
for idx, ax in enumerate(axs.flat):
    ax.imshow(IHCCs[:, :, idx], cmap = 'coolwarm', aspect = 'auto')
    ax.set_xticks([])
    ax.set_yticks([])
    if idx >= d - len(ks):
        ax.set_xlabel(str(trunc_n(ks[i], 2)), size = 20)
        i += 1
    if idx % len(ks) == 0:
        ax.set_ylabel(str(lambdas[l]), size = 20)
        l += 1

fig.text(0.04, 0.5, 'Lambda', va = 'center', rotation = 'vertical', size = 17)
fig.text(0.5, 0.04, 'k', ha = 'center', size = 17)
plt.suptitle('IHCCs Over Lambda and ks', size = 18)
plt.savefig(folder+r'\IHCCsNew.png')
plt.show()
plt.close('all')

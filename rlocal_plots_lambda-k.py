#Makes R_local(r) plots using order parameter data collected in OrderPars_lambda-k
#(For Weibull distributed fibre lengths, lambda-k space)

import os as os
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as io

def trunc_n(x, n):
    return np.trunc(x*10**n)/(10**n)

folder = r'D:\Honours Results\LarterBreakspear\RandomSimsFinal'

lambdas = np.array([50, 100, 150, 200, 250, 300])
lambdas = lambdas[::-1]
ks = np.array([4/3, 2, 4, 24])
#radii of inclusion
thrshs = np.linspace(0, 30, 13)

#store data for composite plot
data_for_plot = np.empty((len(lambdas)*len(ks), 13))
lines = np.empty((len(lambdas)*len(ks), 13))
cnt = 0
for lam in lambdas:

    folder1 = folder+r'\lambda = '+str(lam)
    OrdParData = folder1 + r'\OrdParData'

    for k in ks:

        #load data
        ordpardata = io.loadmat(OrdParData+r'\k'+str(k)+'.mat')['rlocals'].transpose()
        ordpardata[0] = 1 #first elements are NANs
        ordpardata = ordpardata[:, 0]
        line = np.linspace(ordpardata[0], ordpardata[-1], 13) #straight line connecting ends of curve

        data_for_plot[cnt] = ordpardata
        lines[cnt] = line

        #plot individual R_local(r) curves for each (lambda, k)
        plt.plot(thrshs, ordpardata)
        plt.plot(thrshs, line, 'k--')
        plt.ylabel(r'$R_{local}$', size = 15)
        plt.xlabel('Radius of Inclusion (mm)', size = 15)
        plt.ylim(0.1, 1.1)
        plt.title('Local Order Parameter vs Radius of Local Group', size = 17)
        plt.savefig(folder1 + r'\k = '+str(k)+'\OrderParvsWavelengthWithLine.png')
        plt.close('all')

        cnt += 1

#Make composite plot
fig, axs = plt.subplots(len(lambdas), len(ks), sharex = True, sharey = True, figsize = (12, 12), gridspec_kw={'wspace':0, 'hspace':0})
d = len(ks)*len(lambdas)
c = 0
s = 0
for idx, ax in enumerate(axs.flat):
    ax.plot(thrshs, data_for_plot[idx, :])
    ax.plot(thrshs, lines[idx, :], 'k--')
    ax.set_ylim(0.1, 1.1)
    ax.set_xticks([])
    ax.set_yticks([])
    if idx >= d - len(ks):
        ax.set_xlabel(str(trunc_n(ks[s], 2)), size = 15)
        s += 1
    if idx % len(ks) == 0:
        ax.set_ylabel(str(lambdas[c]), size = 15)
        c += 1

fig.text(0.04, 0.5, r'$\lambda$', va = 'center', size = 18)
fig.text(0.5, 0.04, 'k', ha = 'center', size = 18)
plt.suptitle(r'$R_{local}$(r) Curves Over Scaling Parameter and Shape Parameter', size = 16)
plt.savefig(folder+r'\rlocals.png')
plt.close('all')

#Make heatmap 
classes = np.array([[0, 0, 0, 0], [0, 0, 0, 0], 
                    [0, 0, 0, 0], [1, 1, 1, 1],
                    [3, 3, 3, 3], [2, 2, 2, 2]])
plt.imshow(classes, aspect = 'auto', cmap = 'cividis')
plt.xticks(np.arange(0, 4), trunc_n(ks, 2))
plt.yticks(np.arange(0, 6), lambdas)
plt.xlabel('k', size = 16)
plt.ylabel('$\lambda$', size = 16)
plt.savefig(folder+r'\catgoryheatmap.png')
plt.close('all')
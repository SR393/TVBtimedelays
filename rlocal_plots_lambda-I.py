#Makes R_local(r) plots using order parameter data collected in OrderPars_lambda-I
#(For topological changes, lambda-I space)

import os as os
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as io

folder = r'D:\Honours Results\LarterBreakspear\InversionTests\ThirdSims(C0.8Sp100)'

lambdas = np.array([50, 100, 150, 200, 250, 300])
lambdas = lambdas[::-1]
Is = np.array([-1.0, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0])
#radii of inclusion
thrshs = np.linspace(0, 30, 13)

data_for_plot = np.empty((len(lambdas)*len(Is), 13))
lines = np.empty((len(lambdas)*len(Is), 13))
cnt = 0
for lam in lambdas:

    folder1 = folder+r'\lambda = '+str(lam)
    OrdParData = folder1 + r'\OrdParData'

    for I in Is:

        #load data
        ordpardata = io.loadmat(OrdParData+r'\I'+str(I)+'.mat')['rlocals'].transpose()
        ordpardata[0] = 1 #First point otherwise NAN
        ordpardata = ordpardata[:, 0]
        line = np.linspace(ordpardata[0], ordpardata[-1], 13) #Line connecting curve ends

        data_for_plot[cnt] = ordpardata
        lines[cnt] = line

        #make plot for each R_local(r) curve
        plt.plot(thrshs, ordpardata)
        plt.plot(thrshs, line, 'k--')
        plt.ylabel(r'$R_{local}$', size = 15)
        plt.xlabel('Radius of Inclusion (mm)', size = 15)
        plt.ylim(0.1, 1.1)
        plt.title('Local Order Parameter vs Radius of Local Group', size = 17)
        plt.savefig(folder1 + r'\k = '+str(I)+'\OrderParvsWavelengthWithLine.png')
        plt.close('all')

        cnt += 1

#Make composite plot
fig, axs = plt.subplots(len(lambdas), len(Is), sharex = True, sharey = True, figsize = (13, 10), gridspec_kw={'wspace':0, 'hspace':0})
d = len(Is)*len(lambdas)
c = 0
s = 0
for idx, ax in enumerate(axs.flat):
    ax.plot(thrshs, data_for_plot[idx, :])
    ax.plot(thrshs, lines[idx, :], 'k--')
    ax.set_ylim(0.1, 1.1)
    ax.set_xticks([])
    ax.set_yticks([])
    if idx >= d - len(Is):
        ax.set_xlabel(str(Is[s]), size = 15)
        s += 1
    if idx % len(Is) == 0:
        ax.set_ylabel(str(lambdas[c]), size = 15)
        c += 1

fig.text(0.04, 0.5, r'$\lambda$', va = 'center', size = 19)
fig.text(0.5, 0.04, 'I', ha = 'center', size = 19)
plt.suptitle(r'$R_{local}$(r) Curves Over Scaling Parameter and Structural Parameter', size = 17)
plt.savefig(folder+r'\rlocals.png')
plt.close('all')

#Make heatmap
classes = np.array([[2, 1, 0, 2, 0, 1, 2, 2, 2], [2, 2, 1, 1, 0, 1, 2, 2, 2], 
                    [0, 0, 0, 1, 0, 2, 2, 2, 2], [2, 2, 1, 0, 1, 3, 2, 2, 2],
                    [2, 2, 1, 0, 3, 2, 2, 2, 2], [0, 0, 0, 3, 2, 2, 2, 2, 2]])
plt.imshow(classes, aspect = 'auto', cmap = 'cividis')
plt.xticks(np.arange(0, 9), Is)
plt.yticks(np.arange(0, 6), lambdas)
plt.xlabel('I', size = 15)
plt.ylabel('$\lambda$', size = 15)
plt.savefig(folder+r'\catgoryheatmap.png')
plt.close('all')
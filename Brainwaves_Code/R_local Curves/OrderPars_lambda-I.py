#Makes data to create R_local(r) plots for lambda-I space

from tvb.simulator.lab import *
import pycohcorrgram as pyccg
import os as os
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as scp
import scipy.io as io
    
#Extracts phases using Hilbert transform to calc.
#analytic signal
def pyphase_nodes(y):

    phases = np.zeros(y.shape)

    for i in range(y.shape[1]):

        cent = y[:, i] - np.mean(y[:, i])
        phases[:, i] = np.unwrap(np.angle(scp.hilbert(cent)))

    return phases

folder = r'D:\Honours Results\LarterBreakspear\InversionTests\ThirdSims(C0.8Sp100)'

#set up connectivity
wm = connectivity.Connectivity.from_file('geombinsurr_partial_000_Gs.zip')
numnodes = 512
positions = wm.centres

#define euclidean distances between nodes
#to determine which fall inside radius of inclusion 
Euc_dist = np.empty((numnodes, numnodes))
for i in range(numnodes):
    node_i = positions[i]
    for j in range(numnodes):
        if i != j:
            node_j = positions[j]
            diff = node_i - node_j
            dist = np.abs(np.sqrt(np.sum(diff**2)))
            Euc_dist[i, j] = dist
        else:
            Euc_dist[i, j] = 0

lambdas = np.array([50, 100, 150, 250, 300])
#radii of inclusion
thrshs = np.linspace(0, 30, 13)
Is = np.array([-1.0, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0])

for lam in lambdas:

    folder1 = folder+r'\lambda = '+str(lam)
    OrdParData = folder1 + r'\OrdParData'
    if not os.path.exists(OrdParData):
        os.makedirs(OrdParData)
    Rglobals = np.empty(len(Is))
    count = 0

    for I in Is:

        datafile = folder1 + '\Matlab Files\lambda'+str(lam)+'k'+str(I)
        data = io.loadmat(datafile)['data']

        simsteps = data.shape[0] #for pycohcorrgram

        #store data to be saved
        Rlocals = np.empty(len(thrshs))
        mean_n_nodes = np.empty(len(thrshs)) #mean no. nodes in radii of inclusion for given radius
        cnt = 0

        #extract data phases
        ind = 1j*pyphase_nodes(data)
        p = np.exp(ind)
        coh = np.absolute(np.mean(p, axis = 1))

        #calculate R_global
        Rglobal = np.mean(coh)
        Rglobals[count] = Rglobal

        for r in thrshs:
            print(r)
            groups = []

            #group defined by nodes within r of node i
            for i in range(numnodes):
                groups.append(np.nonzero(Euc_dist[i] < r)[0])

            lengths = [len(j) for j in groups]
            mean_n_nodes[cnt] = np.mean(lengths)
            
            orderparameters = np.empty((simsteps, numnodes))

            #calculate R for every group
            for k in range(len(groups)):

                p = np.exp(1j*pyphase_nodes(data[:, groups[k]]))
                coh = np.absolute(np.mean(p, axis = 1))
                orderparameters[:, k] = coh
            
            #calc. R_local
            R_local = np.mean(np.mean(orderparameters, axis = 1))

            Rlocals[cnt] = R_local
            cnt += 1

        #save to plot later
        io.savemat(OrdParData+r'\I'+str(I)+'.mat', {'rlocals':Rlocals, 'grpszs':mean_n_nodes})

        count += 1

    io.savemat(OrdParData+r'\global.mat', {'rglobals':Rglobals})

    #plot R_globals
    plt.plot(Is, Rglobals)
    plt.xlabel('Structure Parameter I')
    plt.title('Global Order Parameter over Structural Changes')
    plt.savefig(folder1 + r'\OrderParvsI.png')
    plt.close('all')



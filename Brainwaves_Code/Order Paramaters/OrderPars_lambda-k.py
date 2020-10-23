#Makes data to create R_local(r) plots for lambda-k space

from tvb.simulator.lab import *
import pycohcorrgram as pyccg
import os as os
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as scp
import scipy.io as io
    
def pyphase_nodes(y):

    phases = np.zeros(y.shape)

    for i in range(y.shape[1]):

        cent = y[:, i] - np.mean(y[:, i])
        phases[:, i] = np.unwrap(np.angle(scp.hilbert(cent)))

    return phases

folder = r'D:\Honours Results\LarterBreakspear\RandomSimsFinal'

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

lambdas = np.array([50, 100, 150, 200, 250, 300])
ks = np.array([4/3, 2, 4, 24])
#radii of inclusion
thrshs = np.linspace(0, 30, 13)

for lam in lambdas:

    folder1 = folder+r'\lambda = '+str(lam)
    OrdParData = folder1 + r'\OrdParData'

    if not os.path.exists(OrdParData):
        os.makedirs(OrdParData)

    Rglobals = np.empty(len(ks))
    count = 0

    for k in ks:
        
        datafile = folder1 + '\Matlab Files\s'+str(count+1)+'k'+str(k)+'lambda'+str(lam)
        data = io.loadmat(datafile)['data']
        simsteps = data.shape[0] #for pycohcorrgram

        Rlocals = np.empty(len(thrshs))
        mean_n_nodes = np.empty(len(thrshs))
        cnt = 0

        ind = 1j*pyphase_nodes(data)
        p = np.exp(ind)
        coh = np.absolute(np.mean(p, axis = 1))
        Rglobal = np.mean(coh)
        Rglobals[count] = Rglobal

        for r in thrshs:
            print(r)
            groups = []

            #groups defined by nodes within r of node i
            for i in range(numnodes):
                groups.append(np.nonzero(Euc_dist[i] < r)[0])

            lengths = [len(j) for j in groups]
            mean_n_nodes[cnt] = np.mean(lengths)
            
            orderparameters = np.empty((simsteps, numnodes))

            #calculate R for each group
            for g in range(len(groups)):

                p = np.exp(1j*pyphase_nodes(data[:, groups[g]]))
                coh = np.absolute(np.mean(p, axis = 1))
                orderparameters[:, g] = coh

            #calc. R_local
            R_local = np.mean(np.mean(orderparameters, axis = 1))

            Rlocals[cnt] = R_local
            cnt += 1

        #save to plot later
        io.savemat(OrdParData+r'\k'+str(k)+'.mat', {'rlocals':Rlocals, 'grpszs':mean_n_nodes})

        count += 1

    io.savemat(OrdParData+r'\global.mat', {'rglobals':Rglobals})
    
    #plot R_globals
    plt.plot(ks, Rglobals)
    plt.xlabel('Shape Parameter k')
    plt.title('Global Order Parameter Over Parameter k')
    plt.savefig(folder1 + r'\OrderParvsk.png')
    plt.close('all')



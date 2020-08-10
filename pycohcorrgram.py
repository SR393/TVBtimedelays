import numpy as np
import pycorrgram as cg
import scipy.signal as scp
import matplotlib.pyplot as plt

#I don't know who wrote this one
def pyphase_nodes(y):

    phases = np.zeros(y.shape)

    for i in range(y.shape[0]):

        cent = y[i] - np.mean(y[i])
        ansig = scp.hilbert(cent)
        phase1 = np.angle(ansig)
        phase2 = np.unwrap(phase1)
        phases[i] = phase2

    return phases

#correlation of coherence between sets of nodes n1 and n2 in y. y is a
#2D array of time series, dimensions time series length X nodes (or
#vice-versa). Optional arguments include opts dictionary containing
#physical timestep, maximum lag, window length, overlap fraction (of window
#length) and "inputphases", a Boolean object False by default, True if inputs
#are phase signals. Also a Boolean object (True by default) indicating whether
#outputs should be figure or dictionary of outputs.Originally written in MATLAB
#by JA Roberts, QIMR Berghofer, 2018.
def pycohcorrgram(*args):

    y = args[0]
    n1 = args[1]
    n2 = args[2]
    l = True

    if y.shape[0] > y.shape[1]:

        y = y.transpose() #Assumes more data points per time series than nodes
    
    if len(args) < 4:

        opts = {"dt":1, "maxlag":30, "winlen":20, "overlapfrac":0.8,
                "inputphases":False}
    else:

        opts = args[3]

    if len(args) == 5:

        l = args[4]

    dt = opts["dt"]
    maxlag = int(np.rint(opts["maxlag"]/dt))
    winlen = int(np.rint(opts["winlen"]/dt))
    overlapfrac = opts["overlapfrac"]
    
    #Check if signals are already phases
    if opts["inputphases"]:
        yphase = y
    else:
        yphase = pyphase_nodes(y)

    #Coherences
    ind1 = 1j*yphase[n1]
    p1 = np.exp(ind1)
    ind2 = 1j*yphase[n2]
    p2 = np.exp(ind2)
    coh1 = np.absolute(np.mean(p1, axis = 0))
    coh2 = np.absolute(np.mean(p2, axis = 0))
    
    #Cross correlation
    ovlp = np.floor(overlapfrac*winlen)
    Coh = np.asarray(cg.pycorrgram(coh1, coh2, maxlag, winlen, ovlp))
    
    #Outputs
    ct = Coh[2]*dt
    cl = Coh[1]*dt
    if l:

        time = np.linspace(0, Coh[0].shape[1], 9)
        lag = np.linspace(0, 2*maxlag + 1, 9)
        
        T = np.linspace(0, ct[-1], 9)
        L = np.linspace(-maxlag, maxlag, 9)
        
        fig = plt.matshow(Coh[0], aspect="auto")
        plt.xticks(time, T)
        plt.yticks(lag, L)
        plt.xlabel('Time')
        plt.ylabel('Lag')
        plt.colorbar()
        plt.show()

    else:

        out = [Coh[0], coh1, coh2, ct, cl]

        return out
        

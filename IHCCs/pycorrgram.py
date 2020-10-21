#Translation of corrgram function in MATLAB for windowed time lagged
#cross-correlation of two time series, originally written by Norbert Marwan,
#Potsdam University, Germany

#Input time series y1 and y2 for which the cross-correlation is to be calculated
#as first two args, assumes ndarray object.

import numpy as np
import matplotlib.pyplot as plt

#define function to standardise windows later on
def normalise(A):

    N = np.zeros(A.shape)

    mu = np.mean(A, axis = 1)
    std = np.std(A, axis = 1, dtype=np.float64, ddof = 1)

    for i in range(A.shape[0]):
        
        cent = A[i] - mu[i]
        norm = cent/std[i]
        N[i] = norm

    return N

#define (simpler) function to stand in for MATLAB's 'buffer' with no delay
def pybuffer(A, n, o):

    nw = int(np.ceil((len(A)-o)/(n-o)))
    B = np.zeros((nw, n))

    i = 0

    while i < nw - 1:
        
        j = i*(n - o)
        B[i] = A[j:j+n]
        i = i + 1
        
    j = (i)*(n-o)
    r = len(A) - (i-1)*(n-o) - n
    
    B[nw-1] = np.append(A[j:j+o+r], [0]*(n-o-r))

    return B

def pycorrgram(*args):

    x = args[0]
    y = args[1]

    nx = int(len(x))
    ny = int(len(y))

    if nx < ny:

        x.append(x, np.zeros(ny - nx))

    if ny < nx:

        np.append(y, np.zeros(nx - ny))

    maxlag = int(np.floor(nx/10))
    window = int(np.floor(nx/10))
    overlap = 0

    if len(args) > 2:

        maxlag = int(args[2])
        
        if maxlag < 0:
            raise ValueError("Requires positive integer value for maximum lag")

        if len(args) > 3:

            window = int(args[3])

            if window < 0:
                raise ValueError("Requires positive integer value for window length")

            if len(args) > 4:

                overlap = int(args[4])

                if overlap < 0:
                    raise ValueError("Requires positive integer value for overlap")
                if overlap >= window:
                    raise ValueError("Requires overlap to be strictly less than window length")
        
    #Create time delayed signals
    X = np.zeros((maxlag + 1, nx))
    Y = np.zeros((maxlag + 1, ny))

    i = 0

    while i <= maxlag:
        
        LSX = np.append(np.zeros(maxlag - i), x[:nx-maxlag+i])
        LSY = np.append(np.zeros(i), y[:ny-i])
        
        X[i] = LSX
        Y[i] = LSY
        i = i+1
    

    #Divide unshifted signals into windows and normalise
    Xw = pybuffer(x, window, overlap)
    Yw = pybuffer(y, window, overlap)

    Xw = normalise(Xw)
    Yw = normalise(Yw)
    
    #Perform cross-correlation
    C = np.zeros((2*maxlag + 1, int(np.ceil((nx - overlap)/(window - overlap)))))
    cnt = 0
    
    for i in range(X.shape[0]):
        
        XiS = X[i]
        Xi = pybuffer(XiS, window, overlap)
        Xi = normalise(Xi)
        R = np.mean(Yw*Xi, axis = 1)
        C[cnt] = R
        cnt = cnt + 1


    for i in range(1, Y.shape[0]):
    
        YiS = Y[i]
        Yi = pybuffer(YiS, window, overlap)
        Yi = normalise(Yi)
        R = np.mean(Xw*Yi, axis = 1)
        C[cnt] = R
        cnt = cnt + 1

    t = np.arange(0, nx+(window-overlap), window - overlap)
    lag = np.linspace(-maxlag, maxlag, 2*maxlag + 1)
    
    #Outputs
    if args == 2:
        
        time = np.arange(np.ceil((nx - overlap)/(window - overlap)))
        T = np.linspace(0, nx - window, Xw.shape[0])
        l = np.linspace(0, 2*maxlag+1, 9)
        L = np.linspace(-maxlag, maxlag, 9)
            
        fig = plt.matshow(C, aspect="auto")
        plt.xticks(time, T)
        plt.yticks(l, L)
        plt.xlabel('Time')
        plt.ylabel('Lag')
        plt.colorbar()
        plt.show()
        return fig

    return [C, lag, t]
    
    

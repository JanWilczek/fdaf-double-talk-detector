import numpy as np
import math

import scipy.signal as sig
import scipy.fft as fft

from utils import * 

def BFDF(X,H,S):
    """
    A Block-Frequency-domain filter implementation.

    Parameters
    ----------
    X : ndarray (in STFT-domain)
        shape: (N_block x N_DFT)
        Input signal.
    H : ndarray (in STFT-domain)
        shape: (1 x N_DFT)
        Impulse response.
    S : int
        Shift sample size (block length / # of shifts)

    Returns (yields)
    ----------
    Y : filter output (in STFT-domain)
    """
    #define variables for matrix / vector lengths
    #M : block-size (e.g. 2048)
    #N : # of shifts (e.g. 2)
    #S : shift sample size (e.g. M/N = 1024)
    #L : input / output size (e.g. 320.000)
    #Nb : # of blocks
   
    Y = np.zeros_like(X)

    for i in range(X.shape[0]): #a better way to write this?
        Yi = np.sum(H*X[i,:],axis=0)
        Y[i,:] = Yi
    y = fft.ifft(Y)
    y = y[:,S:].ravel()[:,None]

    return y.real;

def FDAF_OS(x, d, L=2048, M=1024, S=512, alpha=0.5, delta=1e-6, mu=0.7, freeze_index=None):
    """
    A Frequency-domain adaptive filter based on overlap-add method.

    Parameters
    ----------
    X : ndarray (in STFT-domain)
        Far end signal a.k.a. the sound played from the speaker
    D : ndarray (in STFT-domain)
        Near end signal a.k.a. microphone signal
    L : int
        Number of filter coefficients.
    M : int
        Number of blocks.
    S : int
        Number of shifts.
    alpha: number
        The forgetting factor
    delta: number
        Regularization parameter
    freeze: boolean
        Freezes adaptation of the filter for certain blocks.

    Returns (yields)
    ----------
    W : filter
    E : filter output (error)
    """
    h = np.zeros((L,1))
    M = nearest_pow_2(M)
    S = nearest_pow_2(S)
    N = M//S
    NL = L//S

    x_ = get_shifted_blocks(x,M,S)
    h_ = get_shifted_blocks(h,M,S)

    X = fft.fft(x_)
    H = fft.fft(h_)

    y = np.zeros_like(x)
    e = np.zeros_like(y)
    p = np.zeros((NL,M))
    p[0,:] = 1

    for i in range(len(X)//NL-1): #per block
        freeze_cond = freeze_index is not None \
        and (freeze_index[:,0]<=i*L).any() \
        and (i*L<freeze_index[:,1]).any()
        
        Xm = X[NL*i:NL*(i+1),:]
        y[L*i:L*(i+1)] = BFDF(Xm,H,S)
        e[L*i:L*(i+1)] = d[L*i:L*(i+1)] - y[L*i:L*(i+1)]
        
        if ( freeze_cond ):
            continue #do not perform adaptation

        #adaptation
        E = fft.fft(get_shifted_blocks(e[L*i:L*(i+1)],M,S))

        #stepsize computation
            #PSD estimate
        p = (1-alpha)*p + alpha*(abs(Xm)**2)
        mu_a = mu * np.reciprocal(p)
        H = H + 2 * mu_a * np.conj(Xm) * E


    return e, y, H, p

#to be removed
def NLMS(x, d, N, alpha, delta, freeze_index=None):
    error = 0
    h_hat = 0
    return error, h_hat


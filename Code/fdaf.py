import math
import numpy as np
import ipdb
import scipy.signal as sig
import scipy.fft as fft

from utils import * 
from coherence_double_talk_detector import CoherenceDoubleTalkDetector


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
    y : filter output
    """
   
    Y = np.zeros_like(X)

    for i in range(X.shape[0]): #a better way to write this?
        Yi = np.sum(H*X[i,:],axis=0)
        Y[i,:] = Yi
    y = fft.ifft(Y)
    y = y[:,S:].ravel()[:,None]

    return y.real

def FDAF_OS(x, d, M=2400, S=1200, alpha=0.85, delta=1e-8, mu=0.3, double_talk_threshold=0.5, freeze_index=None):
    """
    A Frequency-domain adaptive filter based on overlap-add method.

    Parameters
    ----------
    x : ndarray
        Far end signal a.k.a. the sound played from the speaker
    d : ndarray
        Near end signal a.k.a. microphone signal
    M : int
        Block size
    S : int
        Number of shifts.
    alpha: number
        The forgetting factor
    delta: number
        Regularization parameter
    double_talk_threshold: number
        Determines the threshold for the xi value below which double-talk is detected

    Returns (yields)
    ----------
    W : filter
    E : filter output (error)
    """

    x_ = get_shifted_blocks(x,M,S)
    X = fft.fft(x_,n=M)
    
    H = np.zeros((1,M))

    y = np.zeros_like(x)
    e = np.zeros_like(y)
    p = np.zeros((1,M))

    k = np.zeros((S,S))
    kp = np.diagflat(np.ones(S))
    k = np.concatenate((k,kp)).T
    kp = np.zeros((1,M))
    kp[:,:S] = 1
    g = np.diagflat(kp)

    dtd = CoherenceDoubleTalkDetector(block_length=S)

    for i in range(len(X)-3): #per block

        Xm = np.diagflat(X[i,:])

        Y = H@Xm
        yk = (k@(fft.ifft(Y).T)).real
        y[S*(i+1):S*(i+2)] = yk
        e[S*(i+1):S*(i+2)] = d[S*(i+1):S*(i+2)] - yk

        if freeze_index is None:
            do_not_adapt = dtd.is_double_talk(x[S*(i+1):S*(i+2)], d[S*(i+1):S*(i+2)], y[S*(i+1):S*(i+2)])
        else:
            do_not_adapt = (freeze_index[:,0]<=i*M).any() \
                            and (i*M<freeze_index[:,1]).any()
        
        if do_not_adapt:
            continue

        #adaptation
        e_ = k.T@e[S*(i+1):S*(i+2)]

        E = fft.fft(e_,axis=0,n=M) #check 1 more
        #stepsize computation
            #PSD estimate

        p = (1-alpha)*p + alpha*(np.abs(np.diag(Xm))**2)

        mu_a = mu * np.diagflat(np.reciprocal(p+delta))
        
        #filter update
        H_upd = 2 * fft.fft(\
            g @ fft.ifft(\
                mu_a @ (np.conj(Xm).T @ E),axis=0 ,n=M),axis=0,n=M)

        H = H + H_upd.T

        print('Block: ', i)
    return e, y, H, p

#to be removed
def NLMS(x, d, N, alpha, delta, freeze_index=None):
    error = 0
    h_hat = 0
    return error, h_hat


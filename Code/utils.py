from os.path import join, abspath
from scipy.io import wavfile
import scipy.signal as sig
import numpy as np
import matplotlib.pyplot as plt
import math

def nearest_pow_2(N):
    """
    Given the integer N, computes the lowest
    power of 2 that is higher than it.
    """
    if ~math.log2(N).is_integer():
        N = 2**math.ceil(math.log2(N))
        print('Fix N to be power of 2:', N)
    return N

def pad_N(x,n0):
    x = np.pad(x,((n0,0),(0,0)), 'constant')
    return x

def get_shifted_blocks(x,M,S):
    """
    1D array to 2D array with shifts and blocks

    Parameters
    ----------
    x : ndarray (in time-domain)
        shape: (1 x N_samples) (Not necessary pow2)
        Input signal.
    M : int
        Block size
    S : int
        Shift size (block length / # of shifts)

    Returns (yields)
    ----------
    X_ : ndarray
        shape: (L/(M/S) x M)
    """

    L = len(x)
    N = M//S

    #number of missing 0s from X
    n0 = ((( (L//M + 1) * M) ) - L) % M
    #pad with n0 0's

    x = np.pad(x,((0,n0),(0,0)), 'constant')

    L = len(x)
    Nb = L//M

    x = np.reshape(x,(Nb*N,M//N))

    x_ = np.zeros((N*Nb,M))
    for i in range(N*Nb-1):
        x_[i,:] = x[i:i+N,:].ravel()

    return x_

def reshaped_to_1d(array):
    return array.reshape((array.shape[0], 1))

def read_wav_file(path):
    rate, data = wavfile.read(path)
    data = data.astype(np.float32)
    data /= np.max(np.abs(data))
    data = reshaped_to_1d(data)
    return data, rate

def generate_microphone_signal(signal_loudspeaker, signal_noise, impulse_response, noise_start_samples):
    signal_microphone = sig.convolve(signal_loudspeaker.reshape(-1),impulse_response.reshape(-1),mode='full',method='direct')
    signal_microphone = reshaped_to_1d(signal_microphone)
    signal_microphone = signal_microphone[0:len(signal_loudspeaker)]

    signal_microphone[noise_start_samples:] += signal_noise[:-noise_start_samples]
    #signal_microphone /= np.max(np.abs(signal_microphone))

    return signal_microphone

def generate_signals(h=None):
    """
    Returns
    -------
    signal_microphone
    signal_loudspeaker
    impulse_response
    sample_rate
    """
    folder = r'../Data/'

    def process_file(filename): return read_wav_file(abspath(join(folder, filename)))

    signal_male, rate = process_file(r'male.wav')
    signal_female, rate = process_file(r'female.wav')
    impulse_response, rate = process_file(r'h.wav')
    if h is not None:
        impulse_response = h
    signal_microphone = generate_microphone_signal(signal_female, signal_male, impulse_response, noise_start_samples=int(4.5 * rate))

    return signal_microphone, signal_female, impulse_response, rate

def plot_signals(signal_microphone, signal_loudspeaker, impulse_response, signal_error, estimated_impulse_response, N):
    plt.figure()
    ax = plt.subplot(4, 1, 1)
    plt.plot(signal_microphone)
    plt.title('signal_microphone')
    plt.ylim([-1, 1])
    plt.subplot(4, 1, 2, sharex=ax)
    plt.plot(signal_loudspeaker)
    plt.title('signal_loudspeaker')
    plt.ylim([-1, 1])
    plt.subplot(4, 1, 3, sharex=ax)
    plt.plot(signal_error)
    plt.title('signal_error')
    plt.subplot(4, 1, 4)
    plt.plot(impulse_response)
    plt.plot(estimated_impulse_response)
    plt.legend(['h', '$\hat{h}$'])
    plt.title('h, h_hat - MSE: ' + str(np.sum(np.square(impulse_response[:N] - estimated_impulse_response))))
    plt.show()

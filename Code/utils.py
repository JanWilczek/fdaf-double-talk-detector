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
    nb_blocks = L//M + 1
    n0 = (nb_blocks * M - L) % M
    #pad with n0 0's

    x = np.pad(x,((0,n0),(0,0)), 'constant')

    Nb = L//M

    x = np.reshape(x,(nb_blocks*N,M//N))

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

    near_end = np.zeros_like(signal_microphone)
    near_end[noise_start_samples:min(noise_start_samples+len(signal_noise),len(near_end))] = signal_noise[:min(len(signal_noise), len(near_end) - noise_start_samples)]

    signal_microphone += near_end
    signal_microphone /= np.max(np.abs(signal_microphone))

    return signal_microphone, near_end

def generate_signals(h=None, noise_start_in_seconds=4.5, length_in_seconds=10):
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
    
    signal_microphone, near_end = generate_microphone_signal(signal_female, signal_male, impulse_response, noise_start_samples=int(noise_start_in_seconds * rate))
    length_in_samples = int(length_in_seconds * rate)

    return signal_microphone[:length_in_samples], signal_female[:length_in_samples], impulse_response, rate, near_end[:length_in_samples]

def plot_signals(signal_microphone, signal_loudspeaker, impulse_response):
    plt.figure()
    ax = plt.subplot(3, 1, 1)
    plt.plot(signal_microphone)
    plt.title('signal_microphone')
    plt.ylim([-1, 1])
    plt.subplot(3, 1, 2, sharex=ax)
    plt.plot(signal_loudspeaker)
    plt.title('signal_loudspeaker')
    plt.ylim([-1, 1])
    plt.subplot(3, 1, 3)
    plt.plot(impulse_response)
    plt.title('original impulse response')
    plt.show()

def dft_matrix(size):
    F = np.zeros((size, size), dtype=complex)
    for nu in range(0, size):
        for n in range(0, size):
            F[nu, n] = np.exp(- 1j * 2 * np.pi * nu * n / size)
    return F

def hermitian(matrix):
    return np.conj(np.transpose(matrix))

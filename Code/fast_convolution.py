from numpy import convolve as np_convolve

def convolve(signal1, signal2):
    return np_convolve(signal1, signal2, mode='same')

from os.path import join, abspath
from scipy.io import wavfile
import numpy as np
import matplotlib.pyplot as plt


def read_wav_file(path):
    rate, data = wavfile.read(path)
    data = data.astype(np.float32)
    data /= np.max(np.abs(data))
    return data, rate

def generate_signal(filename):
    signal, rate = read_wav_file(filename)
    return signal, rate

def reshape_to_1d(array):
    return array.reshape((array.shape[0], 1))

def generate_signals():
    folder = r'../Data/'
    path_male = abspath(join(folder, r'male.wav'))
    path_female = abspath(join(folder, r'female.wav'))
    path_impulse_response = abspath(join(folder, r'h.wav'))

    signal_male, rate = generate_signal(path_male)
    signal_female, rate = generate_signal(path_female)
    impulse_response, rate = generate_signal(path_impulse_response)

    signal_microphone = np.convolve(signal_female, impulse_response, mode='same')

    seconds_in = 4.5
    samples_in = int(seconds_in * rate)

    signal_microphone[samples_in:] += signal_male[:-samples_in]

    signal_microphone = reshape_to_1d(signal_microphone)
    signal_female = reshape_to_1d(signal_female)
    impulse_response = reshape_to_1d(impulse_response)

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

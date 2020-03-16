from os.path import join, abspath
from scipy.io import wavfile
import numpy as np
import matplotlib.pyplot as plt


def reshaped_to_1d(array):
    return array.reshape((array.shape[0], 1))

def read_wav_file(path):
    rate, data = wavfile.read(path)
    data = data.astype(np.float32)
    data /= np.max(np.abs(data))
    data = reshaped_to_1d(data)
    return data, rate

def generate_microphone_signal(signal_loudspeaker, signal_noise, impulse_response, noise_start_samples):
    signal_microphone = np.convolve(signal_loudspeaker.reshape(-1), impulse_response.reshape(-1), mode='same')
    signal_microphone = reshaped_to_1d(signal_microphone)

    near_end = np.zeros_like(signal_microphone)
    near_end[noise_start_samples:min(noise_start_samples+len(signal_noise),len(near_end))] = signal_noise[:min(len(signal_noise), len(near_end) - noise_start_samples)]

    signal_microphone += near_end
    signal_microphone /= np.max(np.abs(signal_microphone))

    return signal_microphone, near_end

def generate_signals(noise_start_in_seconds=4.5, length_in_seconds=10):
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

    signal_microphone, near_end = generate_microphone_signal(signal_female, signal_male, impulse_response, noise_start_samples=int(noise_start_in_seconds * rate))

    length_in_samples = int(length_in_seconds * rate)

    return signal_microphone[:length_in_samples], signal_female[:length_in_samples], impulse_response, rate, near_end

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

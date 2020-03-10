from os.path import join, abspath
from scipy.io import wavfile
import numpy as np
from fast_convolution import convolve
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

    signal_microphone = convolve(signal_female, impulse_response)

    seconds_in = 4.5
    samples_in = int(seconds_in * rate)

    signal_microphone[samples_in:] += signal_male[:-samples_in]

    signal_microphone = reshape_to_1d(signal_microphone)
    signal_female = reshape_to_1d(signal_female)
    impulse_response = reshape_to_1d(impulse_response)

    return signal_microphone, signal_female, impulse_response, rate

def NLMS(x, d, N, alpha, delta, freeze_index=None):
    h_hat = np.zeros((N, 1))
    error = np.zeros((len(x), 1))
    
    error[:N] = x[:N]

    max_iterations = len(x) - N
    if freeze_index is None:
        # adapt continuously
        freeze_index = max_iterations
        
    ## ------------- Insert Code Here ---------------
    for i in range(freeze_index):
        x_frame = x[i:(i + N)]
        y_hat = np.dot(np.conj(h_hat.T), x_frame)
        error[i + N] = d[i + N] - y_hat 

        x_norm2sq = np.dot(np.conj(x_frame.T), x_frame)
        mu_error = np.divide(alpha * np.conj(error[i + N]), x_norm2sq + delta)
            
        step_update = mu_error * x_frame
        
        h_hat = h_hat + step_update
        
    for i in range(freeze_index, max_iterations):
        x_frame = x[i:(i + N)]
        y_hat = np.dot(np.conj(h_hat.T), x_frame)
        error[i + N] = d[i + N] - y_hat 

    return error, h_hat

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

def main():
    signal_microphone, signal_loudspeaker, impulse_response, rate = generate_signals()

    N = 1000

    finish_index = 4 * rate # stop after 4 seconds
    final_nlms_signal_error, estimated_impulse_response = NLMS(signal_loudspeaker, signal_microphone, N, 0.5, 0.000001, freeze_index=finish_index)
    signal_error = final_nlms_signal_error

    plot_signals(signal_microphone, signal_loudspeaker, impulse_response, signal_error, estimated_impulse_response, N)

if __name__ == '__main__':
    main()

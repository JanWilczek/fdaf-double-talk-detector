from utils import generate_signals
from double_talk_detector import DoubleTalkDetector
import numpy as np
import matplotlib.pyplot as plt
import time


def plot_results(signal_microphone, signal_noise, detector_output, detector_benchmark):
    plt.figure()
    plt.subplot(3,1,1)
    plt.plot(signal_microphone)
    plt.legend(['microphone signal'])
    plt.subplot(3,1,2)
    plt.plot(signal_noise)
    plt.legend(['noise (double-talk) signal'])
    plt.subplot(3,1,3)
    plt.plot(0.5 * detector_output)
    plt.plot(detector_benchmark)
    plt.legend(['double-talk detection', 'double-talk active'])
    plt.show()


def main():
    signal_microphone, signal_loudspeaker, impulse_response, rate, noise_signal = generate_signals(noise_start_in_seconds=1, length_in_seconds=3)
    N = 64
    K = 50
    L = K * N
    lambd = 0.9
    lambd_b = 0.8   # the forgetting factor of the background filter should be smaller than that of the foreground filter
    double_talk_threshold = 0.7
    dtd = DoubleTalkDetector(N, L, lambd, lambd_b, double_talk_threshold)

    noise_power_threshold = 0.0015    # power of noise block to account as active (for benchmark purposes only)

    detector_output = np.zeros((len(signal_loudspeaker),))
    detector_benchmark = np.zeros_like(detector_output)

    time_accumulator = 0.0

    nb_iterations = len(signal_loudspeaker) // N
    for i in range(0, nb_iterations):
        print(f'Iteration {i} out of {nb_iterations-1}')

        start = time.time()

        mic_block = signal_microphone[i*N:(i+1)*N]
        speaker_block = signal_loudspeaker[i*N:(i+1)*N]
        noise_block = noise_signal[i*N:(i+1)*N]
        
        noise_block_power = np.linalg.norm(noise_block, 2) / len(noise_block)
        if noise_block_power > noise_power_threshold:
            detector_benchmark[i*N:(i+1)*N] = np.ones((N,))

        if dtd.is_double_talk(speaker_block, mic_block):
            detector_output[i*N:(i+1)*N] = np.ones((N,))

        end = time.time()
        time_accumulator += end - start
        print(f'Average iteration time: {time_accumulator / (i+1)}')

    plot_results(signal_microphone, noise_signal, detector_output, detector_benchmark)

if __name__ == "__main__":
    main()

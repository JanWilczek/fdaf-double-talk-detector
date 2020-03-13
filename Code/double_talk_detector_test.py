from utils import generate_signals
from double_talk_detector import DoubleTalkDetector
import numpy as np
import matplotlib.pyplot as plt


def plot_results(signal_microphone, signal_loudspeaker, detector_output):
    plt.figure()
    plt.subplot(3,1,1)
    plt.plot(signal_loudspeaker)
    plt.subplot(3,1,2)
    plt.plot(signal_microphone)
    plt.subplot(3,1,3)
    plt.plot(detector_output)
    plt.show()


def main():
    signal_microphone, signal_loudspeaker, impulse_response, rate = generate_signals()
    N = 256
    K = 8
    L = K * N
    lambd = 0.9
    lambd_b = 0.8   # the forgetting factor of the background filter should be smaller than that of the foreground filter
    double_talk_threshold = 0.9
    dtd = DoubleTalkDetector(N, L, lambd, lambd_b, double_talk_threshold)

    detector_output = np.zeros((len(signal_loudspeaker),))

    nb_iterations = len(signal_loudspeaker) // N // 10
    for i in range(0, nb_iterations):
        print(f'Iteration {i} out of {nb_iterations-1}')

        mic_block = signal_microphone[i*N:(i+1)*N]
        speaker_block = signal_loudspeaker[i*N:(i+1)*N]

        if dtd.is_double_talk(speaker_block, mic_block):
            detector_output[i*N:(i+1)*N] = np.ones((N,))

    plot_results(signal_microphone[:nb_iterations*N], signal_loudspeaker[:nb_iterations*N], detector_output[:nb_iterations*N])

if __name__ == "__main__":
    main()

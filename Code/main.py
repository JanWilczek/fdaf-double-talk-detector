import numpy as np
from utils import *
from fdaf import NLMS


def main():
    signal_microphone, signal_loudspeaker, impulse_response, rate = generate_signals()

    N = 1000

    finish_index = 4 * rate # stop after 4 seconds
    nlms_signal_error, estimated_impulse_response = NLMS(signal_loudspeaker, signal_microphone, N, 0.5, 0.000001, freeze_index=finish_index)
    signal_error = nlms_signal_error

    plot_signals(signal_microphone, signal_loudspeaker, impulse_response, signal_error, estimated_impulse_response, N)

if __name__ == '__main__':
    main()

from numpy.fft import fft
from coherence_calculator import CoherenceCalculator


class CoherenceDoubleTalkDetector:
    def __init__(self, block_length, lambda_coherence=0.68, closed_loop_threshold=0.95, open_loop_threshold=0.8):
        self.block_length = block_length
        self.lambda_coherence = lambda_coherence
        self.closed_loop_threshold = closed_loop_threshold
        self.open_loop_threshold = open_loop_threshold
        self.open_loop_coherence = CoherenceCalculator(block_length, lambda_coherence)
        self.closed_loop_coherence = CoherenceCalculator(block_length, lambda_coherence)
   
    def is_double_talk(self, loudspeaker_samples_block, microphone_samples_block, microphone_samples_estimate):
        D_b = fft(microphone_samples_block, axis=0)
        X_b = fft(loudspeaker_samples_block, axis=0)
        Y_hat = fft(microphone_samples_estimate, axis=0)

        open_loop_rho = self.open_loop_coherence.calculate_rho(X_b, D_b)
        closed_loop_rho = self.closed_loop_coherence.calculate_rho(Y_hat, D_b)

        return open_loop_rho < self.open_loop_threshold and \
            closed_loop_rho < self.closed_loop_threshold
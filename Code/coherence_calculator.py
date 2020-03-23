import numpy as np


class CoherenceCalculator:
    def __init__(self, block_length, lambda_coherence):
        self.block_length = block_length
        self.lambda_coherence = lambda_coherence
        self.coherence_12 = np.zeros(block_length)
        self.coherence_11 = np.ones(block_length)
        self.coherence_22 = np.ones(block_length)

    def calculate_rho(self, estimate, observation):
        self.coherence_12 = self.lambda_coherence * self.coherence_12 + (1 - self.lambda_coherence) * np.multiply(estimate, np.conjugate(observation))
        self.coherence_11 = self.lambda_coherence * self.coherence_11 + (1 - self.lambda_coherence) * np.square(np.absolute(estimate))
        self.coherence_22 = self.lambda_coherence * self.coherence_22 + (1 - self.lambda_coherence) * np.square(np.absolute(observation))
        coherence_b = np.divide(np.square(np.absolute(self.coherence_12)), np.multiply(self.coherence_11, self.coherence_22))
        weights = (1.0 / np.sum(self.coherence_11)) * self.coherence_11
        rho = np.sum(np.multiply(weights, coherence_b))
        return rho

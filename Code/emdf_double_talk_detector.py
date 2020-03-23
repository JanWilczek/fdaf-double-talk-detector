from collections import deque
import numpy as np
from numpy.fft import fft
from utils import dft_matrix, hermitian
import matplotlib.pyplot as plt


class EMDFDoubleTalkDetector:
    """
        Double-talk detector based on Robust Extended Multidelay Filtering.
        
        Not working. Fast Kalman gain computation fails and the algorithm itself is unstable.
    """
    def __init__(self, block_length, nb_blocks_per_filter, forgetting_factor, background_filter_forgetting_factor, fast_kalman=True):
        self.lambd = forgetting_factor
        self.lambd_b = background_filter_forgetting_factor
        self.fast_kalman = fast_kalman
        self.N = block_length
        self.F_2N = dft_matrix(2 * self.N)
        self.K = nb_blocks_per_filter
        self.L = self.K * self.N
        self.previous_loudspeaker_samples = np.zeros((self.N, 1))
        self.X_k = deque()   # buffered DFTs of sample blocks as matrices' diagonals
        for k in range(0, self.K):
            self.X_k.append(np.zeros((2 * self.N, 2 * self.N), dtype=complex))
        self.S_prim = 0.0015 * np.eye(2*self.N * self.K, dtype=complex) # regularized to make S_prim invertible in the first iteration
        self.W_1 = np.zeros((2*self.N, 2*self.N))
        self.W_1[self.N:2*self.N, self.N:2*self.N] = np.eye(self.N)
        self.W_2 = np.zeros((2*self.N, 2*self.N))
        self.W_2[0:self.N, 0:self.N] = np.eye(self.N)
        self.G_1 = self.F_2N @ self.W_1 @ np.linalg.inv(self.F_2N)
        self.G_2_tilde = self.F_2N @ self.W_2 @ np.linalg.inv(self.F_2N)
        self.G_2 = np.zeros((2*self.L, 2*self.L), dtype=complex)
        for k in range(0, self.K):
            self.G_2[k*2*self.N:(k+1)*2*self.N,k*2*self.N:(k+1)*2*self.N] = self.G_2_tilde # place G_2_tilde along the diagonal of G_2
        self.h_b = np.zeros((2*self.L, 1), dtype=complex)
        self.s_k = []
        for k in range(0, self.K):
            self.s_k.append(np.zeros((2 * self.N, 1)))
        self.var2_y = 0.0

        # Fast Kalman gain computation variables
        self.a = np.zeros((self.K, 2 * self.N), dtype=complex)
        self.phi = np.ones((2 * self.N,), dtype=complex)
        self.E_a = np.ones((2 * self.N,), dtype=complex)
        self.E_b = np.zeros((2 * self.N,), dtype=complex)
        self.K_1 = np.zeros((self.K, 2 * self.N), dtype=complex)
        self.b = np.zeros((self.K, 2 * self.N), dtype=complex)

    def enqueue_loudspeaker_block(self, new_samples_block):
        assert new_samples_block.shape == (self.N, 1)

        x_2N = np.vstack((self.previous_loudspeaker_samples, new_samples_block))
        assert x_2N.shape == (2 * self.N, 1)

        X_0_vec = fft(x_2N, axis=0)
        X_0 = np.diag(X_0_vec.reshape(-1))
        assert X_0.shape == (2*self.N, 2*self.N)

        self.X_k.appendleft(X_0)
        self.X_k.pop()

        self.previous_loudspeaker_samples = new_samples_block

    def X(self):
        X = np.zeros((2 * self.N, 2 * self.N * self.K), dtype=complex)
        for k in range(0, self.K):
            X [:,k * 2 * self.N:(k+1) * 2 * self.N] = self.X_k[k]
        assert X.shape == (2 * self.N, 2 * self.L)
        return X

    def kalman_gain(self, X):
        K = np.zeros((2 * self.L, 2 * self.N), dtype=complex)
        K_ni = np.zeros((self.K, 2 * self.N), dtype=complex)
        for ni in range(0, 2 * self.N):
            X_ni = X[ni, list(range(ni,2*self.L,2*self.N))].reshape((1, self.K))    # row vector
            assert X_ni.shape == (1, self.K)

            e_a_ni = np.conj(X_ni[0, 0]) - np.dot(hermitian(self.a[:, ni].reshape((self.K, 1))), hermitian(X_ni))
            assert e_a_ni.shape == (1, 1)
            e_a_ni = e_a_ni.item()
            assert np.isscalar(e_a_ni)
            e_a_ni_sq = np.power(np.abs(e_a_ni), 2)

            phi_1_ni = self.phi[ni] + e_a_ni_sq / self.E_a[ni]

            e_coeff = e_a_ni / self.E_a[ni]
            t_ni = np.zeros((self.K, 1), dtype=complex)
            t_ni[0, 0] = e_coeff
            t_ni[1:, 0] = self.K_1[:-1, ni] - self.a[:-1, ni] * e_coeff
            M_ni = self.K_1[-1, ni] - self.a[-1, ni] * e_coeff
            assert np.isscalar(M_ni)

            self.E_a[ni] = self.lambd * (self.E_a[ni] + e_a_ni_sq / self.phi[ni])
            assert np.isscalar(self.E_a[ni])

            a_ni_update = self.K_1[:, ni] * np.conj(e_a_ni) / self.phi[ni]
            assert a_ni_update.shape == (self.K,)
            self.a[:, ni] += a_ni_update

            e_b_ni = self.E_b[ni] * M_ni

            K_1_ni_update = t_ni.reshape(-1) + self.b[:, ni] * M_ni
            assert K_1_ni_update.shape == (self.K,)
            self.K_1[:, ni] += K_1_ni_update

            self.phi[ni] = phi_1_ni - np.conj(e_b_ni) * M_ni

            self.E_b[ni] = self.lambd * (self.E_b[ni] + np.power(np.abs(e_b_ni), 2) / self.phi[ni])
            assert np.isscalar(self.E_b[ni])

            self.b[:, ni] = self.b[:, ni] + self.K_1[:, ni] * np.conj(e_b_ni) / self.phi[ni]

            K_ni[:, ni] = self.K_1[:, ni] / self.phi[ni]
            assert K_ni[:, ni].shape == (self.K,)
            assert K_ni.shape == (self.K, 2 * self.N)

        for k in range(0, self.K):
            K[k * 2 * self.N:(k+1) * 2 * self.N, :] = np.diag(K_ni[k, :])

        return K

    def is_double_talk(self, loudspeaker_samples_block, microphone_samples_block, show_debug_plot=False):
        """
        Returns
        -------
        xi       decision variable in range [0, 1]. If close to 1: no double-talk. Else: double-talk active.
        """
        self.enqueue_loudspeaker_block(loudspeaker_samples_block)

        X = self.X()

        if self.fast_kalman:
            kalman_gain = self.kalman_gain(X)   # Fast implementation
        else:
            self.S_prim = self.lambd * self.S_prim  + (1 - self.lambd) * hermitian(X) @ X
            assert self.S_prim.shape == (2 * self.L, 2 * self.L)
            
            kalman_gain = np.linalg.inv(self.S_prim) @ hermitian(X)   # Original implementation

        assert kalman_gain.shape == (2 * self.L, 2 * self.N)

        zeros_y = np.vstack((np.zeros((self.N, 1)), microphone_samples_block))
        assert zeros_y.shape == (2 * self.N, 1)

        y_ = fft(zeros_y, axis=0)
        assert y_.shape == (2 * self.N, 1)

        background_y_estimate = self.G_1 @ (X @ self.h_b)
        error_b = np.subtract(y_, background_y_estimate)
        assert error_b.shape == (2 * self.N, 1)

        self.h_b = self.h_b + 2 * (1 - self.lambd_b) * self.G_2 @ (kalman_gain @ error_b)
        assert self.h_b.shape == (2 * self.L, 1)

        for k in range(0, self.K):
            self.s_k[k] = self.lambd_b * self.s_k[k] + (1 - self.lambd_b) * np.conj(self.X_k[k]) @ y_
            assert self.s_k[k].shape == (2 * self.N, 1)

        self.var2_y = self.lambd_b * self.var2_y + (1 - self.lambd_b) * (hermitian(y_) @ y_).item()
        assert np.isscalar(self.var2_y)
        assert np.imag(self.var2_y) == 0

        partial_sums = [np.abs((hermitian(self.h_b[2*self.N*k:2*self.N*(k+1), 0]) @ self.s_k[k]).item()) for k in range(0, self.K)]
        xi_sq = sum(partial_sums) / np.real(self.var2_y)
        assert np.isscalar(xi_sq)
        assert np.imag(xi_sq) == 0

        xi = np.sqrt(xi_sq)

        if show_debug_plot:
            plt.matshow(np.abs(kalman_gain))
            plt.show()

        return xi

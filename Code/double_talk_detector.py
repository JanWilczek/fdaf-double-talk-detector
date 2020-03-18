from collections import deque
import numpy as np
from numpy.fft import fft, ifft
from utils import dft_matrix, hermitian


class DoubleTalkDetector:
    def __init__(self, N, L, lambd, lambd_b):
        self.lambd = lambd
        self.lambd_b = lambd_b
        self.N = N
        self.F_2N = dft_matrix(2 * N)
        self.L = L
        assert L // N == L / N, "Filter's length is not divisible by block's length!"
        self.K = L // N
        self.x_k = deque()   # buffered sample blocks
        self.X_k = deque()   # buffered DFTs of sample blocks as matrices' diagonals
        for k in range(0, self.K):
            self.x_k.append(np.zeros((N, )))
            self.X_k.append(np.zeros((2 * self.N, 2 * self.N), dtype=complex))
        self.S_prim = np.eye(2*N * self.K, dtype=complex) # regularized to make S_prim invertible in the first iteration
        self.W_1 = np.zeros((2*N, 2*N))
        self.W_1[N:2*N, N:2*N] = np.eye(N)
        self.W_2 = np.zeros((2*N, 2*N))
        self.W_2[0:N, 0:N] = np.eye(N)
        self.G_1 = self.F_2N @ self.W_1 @ np.linalg.inv(self.F_2N)
        self.G_2_tilde = self.F_2N @ self.W_2 @ np.linalg.inv(self.F_2N)
        self.G_2 = np.zeros((2*L, 2*L), dtype=complex)
        for k in range(0, self.K):
            self.G_2[k*2*N:(k+1)*2*N,k*2*N:(k+1)*2*N] = self.G_2_tilde # place G_2_tilde along the diagonal of G_2
        self.h_b = np.zeros((2*L, ))
        self.s_k = []
        for k in range(0, self.K):
            self.s_k.append(np.zeros((2 * N,)))
        self.var2_y = 0.0

        # Fast Kalman gain computation variables
        delta = 1.0
        self.a = np.zeros((self.K, 2 * self.N), dtype=complex)
        self.phi = delta * np.ones((2 * self.N,), dtype=complex)
        self.E_a = delta * np.ones((2 * self.N,), dtype=complex)
        self.E_b = np.zeros((2 * self.N,), dtype=complex)
        self.K_1 = np.zeros((self.K, 2 * self.N), dtype=complex)
        self.b = np.zeros((self.K, 2 * self.N), dtype=complex)

    def enqueue_loudspeaker_block(self, new_samples_block):
        self.x_k.appendleft(new_samples_block.reshape((self.N,)))
        self.x_k.pop()
        assert len(self.x_k) == self.K, f"Inappriopriate number of buffered blocks! Is {len(self.x_k)}, should be {self.K}."

        x_2N = np.hstack((self.x_k[1], self.x_k[0]))
        assert len(x_2N) == 2 * self.N

        X_0_vec = fft(x_2N, axis=0)
        X_0 = np.diag(X_0_vec.reshape(-1))
        assert X_0.shape == (2*self.N, 2*self.N)
        self.X_k.appendleft(X_0)
        self.X_k.pop()

    def X(self):
        X = np.zeros((2 * self.N, 2 * self.N * self.K), dtype=complex)
        for k in range(0, self.K):
            X[:, k * 2 * self.N:(k+1) * 2 * self.N] = self.X_k[k]
        assert X.shape == (2 * self.N, 2 * self.L)
        return X

    def kalman_gain(self, X):
        K = np.zeros((2 * self.L, 2 * self.N), dtype=complex)
        K_ni = np.zeros((self.K, 2 * self.N), dtype=complex)
        for ni in range(0, 2 * self.N):
            X_ni = X[ni, list(range(ni,2*self.L,2*self.N))]
            assert X_ni.shape == (self.K,)

            e_a_ni = np.conj(X_ni[0]) - np.dot(hermitian(self.a[:, ni]), hermitian(X_ni))
            assert np.isscalar(e_a_ni)
            e_a_ni_sq = np.power(np.abs(e_a_ni), 2)

            phi_1_ni = self.phi[ni] + e_a_ni_sq / self.E_a[ni]

            e_coeff = e_a_ni / self.E_a[ni]
            t_ni = np.zeros_like(self.K_1[:, ni])
            t_ni[0] = e_coeff
            t_ni[1:] = self.K_1[:-1, ni] - self.a[:-1, ni] * e_coeff
            M_ni = self.K_1[-1, ni] - self.a[-1, ni] * e_coeff
            assert np.isscalar(M_ni)

            self.E_a[ni] = self.lambd * (self.E_a[ni] + e_a_ni_sq / self.phi[ni])
            assert np.isscalar(self.E_a[ni])

            self.a[:, ni] = self.a[:, ni] + self.K_1[:, ni] * np.conj(e_a_ni) / self.phi[ni]
            assert self.a[:, ni].shape == (self.K,)

            e_b_ni = self.E_b[ni] * M_ni

            self.K_1[:, ni] = t_ni + self.b[:, ni] * M_ni
            assert self.K_1[:, ni].shape == (self.K, )

            self.phi[ni] = phi_1_ni - np.conj(e_b_ni) * M_ni

            self.E_b[ni] = self.lambd * (self.E_b[ni] + np.power(np.abs(e_b_ni), 2) / self.phi[ni])
            assert np.isscalar(self.E_b[ni])

            self.b[:, ni] = self.b[:, ni] + self.K_1[:, ni] * np.conj(e_b_ni) / self.phi[ni]

            K_ni[:, ni] = self.K_1[:, ni] / self.phi[ni]
            assert K_ni[:, ni].shape == (self.K, )
    
        for k in range(0, self.K):
            K[k * 2 * self.N:(k+1) * 2 * self.N, :] = np.diag(K_ni[k, :])

        return K

    def is_double_talk(self, loudspeaker_samples_block, microphone_samples_block):
        """
        Returns
        -------
        xi       decision variable in range [0, 1]. If close to 1: no double-talk. Else: double-talk active.
        """
        self.enqueue_loudspeaker_block(loudspeaker_samples_block)

        X = self.X()
        self.S_prim = self.lambd * self.S_prim  + (1 - self.lambd) * hermitian(X) @ X
        assert self.S_prim.shape == (2 * self.L, 2 * self.L)

        # kalman_gain = np.linalg.inv(self.S_prim) @ hermitian(X)
        kalman_gain = self.kalman_gain(X)
        assert kalman_gain.shape == (2 * self.L, 2 * self.N)

        zeros_y = np.vstack((np.zeros((self.N, 1)), microphone_samples_block))
        assert zeros_y.shape == (2 * self.N, 1)

        y_ = fft(zeros_y, axis=0).reshape(-1)
        assert y_.shape == (2 * self.N,)

        background_y_estimate = self.G_1 @ (X @ self.h_b)
        error_b = np.subtract(y_, background_y_estimate)
        assert error_b.shape == (2 * self.N,)

        self.h_b = self.h_b + 2 * (1 - self.lambd_b) * self.G_2 @ (kalman_gain @ error_b)
        assert self.h_b.shape == (2 * self.L,)

        # Filter coefficients constraint (to avoid circular convolution artifacts)
        for k in range(0, self.K):
            h = ifft(self.h_b[k * 2 * self.N:(k+1) * 2 * self.N])
            h[self.N:] = 0.0
            self.h_b[k * 2 * self.N:(k+1) * 2 * self.N] = fft(h)

        for k in range(0, self.K):
            self.s_k[k] = self.lambd_b * self.s_k[k] + (1 - self.lambd_b) * np.conj(self.X_k[k]) @ y_
            assert self.s_k[k].shape == (2 * self.N,)

        self.var2_y = (self.lambd_b * self.var2_y + (1 - self.lambd_b) * np.dot(hermitian(y_), y_)).item()
        assert np.isscalar(self.var2_y)

        partial_sums = [np.dot(hermitian(self.h_b[2*self.N*k:2*self.N*(k+1)]), self.s_k[k]) for k in range(0, self.K)]
        xi_sq = sum(partial_sums) / self.var2_y
        assert np.isscalar(xi_sq)

        # assert np.imag(xi_sq) == 0 # Not true, unfortunately

        xi_sq = np.abs(xi_sq) # WARNING: Absolute value is added by me.
        xi = np.sqrt(xi_sq)

        return xi

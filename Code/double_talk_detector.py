from collections import deque
import numpy as np

def dft_matrix(size):
    F = np.zeros((size, size), dtype=complex)
    for nu in range(0, size):
        for n in range(0, size):
            F[nu, n] = np.exp(- 1j * 2 * np.pi * nu * n / size)
    return F

def hermitian(matrix):
    return np.conj(np.transpose(matrix))

class DoubleTalkDetector:
    def __init__(self, N, L, lambd, lambd_b, threshold):
        self.lambd = lambd
        self.lambd_b = lambd_b
        self.threshold = threshold
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
        self.S_prim = 0.0005 * np.eye(2*N * self.K, dtype=complex) # regularized to make S_prim invertible in the first iteration
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
            self.s_k = np.zeros((2 * N,))
        self.var2_y = 0.0

    def enqueue_loudspeaker_block(self, new_samples_block):
        self.x_k.appendleft(new_samples_block)
        self.x_k.pop()
        assert len(self.x_k) == self.K, f"Inappriopriate number of buffered blocks! Is {len(self.x_k)}, should be {self.K}."

        x_2N = np.vstack((self.x_k[0], new_samples_block))
        assert len(x_2N) == 2 * self.N
        self.X_k.appendleft(np.diag(self.F_2N @ x_2N))
        self.X_k.pop()

    def X(self):
        X = np.zeros((2 * self.N, 2 * self.N * self.K), dtype=complex)
        for k in range(0, self.K):
            X[:, k * 2 * self.N:(k+1) * 2 * self.N] = self.X_k[k]
        return X

    def is_double_talk(self, loudspeaker_samples_block, microphone_samples_block):
        self.enqueue_loudspeaker_block(loudspeaker_samples_block)

        X = self.X()
        self.S_prim = self.lambd * self.S_prim  + (1 - self.lambd) * hermitian(X) @ X
        kalman_gain = np.linalg.inv(self.S_prim) @ hermitian(X)

        zeros_y = np.vstack((np.zeros((self.N, 1)), microphone_samples_block))
        y_ = self.F_2N @ zeros_y

        background_y_estimate = self.G_1 @ X @ self.h_b
        error_b = np.subtract(y_.reshape(-1), background_y_estimate)

        self.h_b = self.h_b + 2 * (1 - self.lambd_b) * self.G_2 @ kalman_gain @ error_b

        for k in range(0, self.K):
            self.s_k[k] = self.lambd_b * self.s_k[k] + (1 - self.lambd_b) * np.conj(self.X_k[k]) @ y_

        self.var2_y = self.lambd_b * self.var2_y + (1 - self.lambd_b) * np.dot(hermitian(y_), y_)

        dzeta_sq = sum([np.dot(hermitian(self.h_b[self.N*k:self.N*(k+1) - 1]), self.s_k[k]) for k in range(0, self.K)]) / self.var2_y
        dzeta = np.sqrt(dzeta_sq)

        if dzeta < self.threshold:
            return True
        else:
            return False








from collections import deque
import numpy as np

def dft_matrix(size):
    F = np.zeros((size, size))
    for nu in range(0, size):
        for n in range(0, size):
            F[nu, n] = np.exp(- 1j * 2 * np.pi * nu * n / size)
    return F

def hermitian(matrix):
    return np.conj(np.transpose(matrix))

class DoubleTalkDetector:
    def __init__(self, N, L, lambd):
        self.lambd = lambd
        self.N = N
        self.F_2N = dft_matrix(2 * N)
        self.L = L
        assert L // N == L / N % "Filter's length is not divisible by block's length!"
        self.K = L / N
        self.x_k = deque()   # buffered sample blocks
        self.X_k = deque()   # buffered DFTs of sample blocks as matrices' diagonals
        for k in range(0, self.K):
            self.x_k.append(np.zeros((N, )))
            self.X_k.append(np.zeros((self.K, self.K)))
        self.S_prim = np.zeros((2*N, 2*N))
        self.W_1 = np.zeros((2*N, 2*N))
        self.W_1[N:2*N-1, N:2*N-1] = np.eye(N)
        self.W_2 = np.zeros((2*N, 2*N))
        self.W_2[0:N-1, 0:N-1] = np.eye(N)
        self.G_1 = self.F_2N @ self.W_1 @ np.linalg.inv(self.F_2N)
        self.G_2_tilde = self.F_2N @ self.W_2 @ np.linalg.inv(self.F_2N)
        self.G_2 = np.diag(self.G_2_tilde) # How many G_2_tildes???


    def enqueue_new_block(self, new_samples_block):
        self.x_k.appendleft(new_samples_block)
        self.x_k.pop()
        assert len(self.x_k) == self.K % f"Inappriopriate number of buffered blocks! Is {len(self.x_k)}, should be {self.K}."

        self.X_k.appendleft(np.diag(self.F_2N @ new_samples_block))
        self.X_k.pop()

    def X(self):
        X = np.zeros((2 * self.N, 2 * self.N * self.K))
        for k in range(0, self.K):
            X[:, k * 2 * self.N:(k+1) * 2 * self.N] = self.X_k[k]
        return X

    def is_double_talk(self, samples_block):
        self.enqueue_new_block(samples_block)

        X = self.X()
        self.S_prim = self.lambd * self.S_prim  + (1 - self.lambd) * hermitian(X) @ X
        kalman_gain = np.linalg.inv(self.S_prim) @ hermitian(X)

        y_ = G_01_2N_2N @ X @ h_2L + v

        e_b = y_ - G_1 @ X @ self.h_b








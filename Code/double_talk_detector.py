def dft_matrix(size):
    F = np.zeros((size, size))
    for nu in range(0, size):
        for n in range(0, size):
            F[i, j] = np.exp(- 1j * 2 * np.pi * nu * n / size)
    return F

def is_double_talk(samples_block, K):
    N = len(samples_block)
    F_2N = dft_matrix(2 * N)
    

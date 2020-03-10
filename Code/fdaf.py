import numpy as np


def NLMS(x, d, N, alpha, delta, freeze_index=None):
    h_hat = np.zeros((N, 1))
    error = np.zeros((len(x), 1))
    
    error[:N] = x[:N]

    max_iterations = len(x) - N
    if freeze_index is None:
        # adapt continuously
        freeze_index = max_iterations
        
    ## ------------- Insert Code Here ---------------
    for i in range(freeze_index):
        x_frame = x[i:(i + N)]
        y_hat = np.dot(np.conj(h_hat.T), x_frame)
        error[i + N] = d[i + N] - y_hat 

        x_norm2sq = np.dot(np.conj(x_frame.T), x_frame)
        mu_error = np.divide(alpha * np.conj(error[i + N]), x_norm2sq + delta)
            
        step_update = mu_error * x_frame
        
        h_hat = h_hat + step_update
        
    for i in range(freeze_index, max_iterations):
        x_frame = x[i:(i + N)]
        y_hat = np.dot(np.conj(h_hat.T), x_frame)
        error[i + N] = d[i + N] - y_hat 

    return error, h_hat


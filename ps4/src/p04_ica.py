"""PS4 Problem 4: Independent Component Analysis (blind source separation).

Given a mixed multi-channel signal X = S W^{-1}, ICA recovers an unmixing matrix
W (so that S = X W^T) by maximum likelihood under a Laplace source prior.  The
stochastic gradient-ascent update for a single sample x is

    W <- W + lr * ( (W^T)^{-1} - sign(W x) x^T ).

Learning rate is annealed over several passes.
"""
import os

import numpy as np
import scipy.io.wavfile

Fs = 11025  # sampling rate of the provided mixtures


def update_W(W, x, learning_rate):
    """One stochastic gradient-ascent step on the ICA log-likelihood."""
    return W + learning_rate * (np.linalg.inv(W.T) - np.outer(np.sign(W @ x), x))


def unmix(X, W):
    """Apply the unmixing matrix: S = X W^T."""
    return X @ W.T


def unmixer(X):
    """Learn W by annealed stochastic gradient ascent over the samples."""
    m, n = X.shape
    W = np.eye(n)
    anneal = [0.1, 0.1, 0.1, 0.05, 0.05, 0.05, 0.02, 0.02,
              0.01, 0.01, 0.005, 0.005, 0.002, 0.002, 0.001, 0.001]
    print('[p04] separating tracks ...')
    for lr in anneal:
        for i in np.random.permutation(m):
            W = update_W(W, X[i], lr)
    return W


def normalize(dat):
    return 0.99 * dat / np.max(np.abs(dat))


def main(data_dir='../../data/ps4'):
    os.makedirs('output', exist_ok=True)
    np.random.seed(0)

    X = normalize(np.loadtxt(f'{data_dir}/mix.dat'))
    print(f'[p04] mixed signal shape: {X.shape}')
    for i in range(X.shape[1]):
        scipy.io.wavfile.write(f'output/mixed_{i}.wav', Fs, X[:, i].astype(np.float32))

    W = unmixer(X)
    np.savetxt('output/W.txt', W)

    S = normalize(unmix(X, W))
    assert S.shape[1] == 5
    for i in range(S.shape[1]):
        scipy.io.wavfile.write(f'output/split_{i}.wav', Fs, S[:, i].astype(np.float32))

    print('[p04] recovered unmixing matrix W:')
    print(W)
    print(f'[p04] wrote {S.shape[1]} separated sources to output/split_*.wav')
    return W


if __name__ == '__main__':
    main()

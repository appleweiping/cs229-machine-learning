"""Kernelized SVM trained with averaged sub-gradient descent (Pegasos-style).

This mirrors the SVM helper shipped with the CS229 spam problem: an RBF-kernel
SVM on binary word-presence features, optimised by stochastic sub-gradient
descent with Polyak (running-average) iterate averaging.  The spam driver uses
it only through ``train_and_predict_svm``.
"""
import numpy as np

np.random.seed(123)


def _rbf_gram(a_sq, b_sq, gram, radius):
    """RBF kernel matrix from squared norms and the dot-product Gram matrix."""
    dist = a_sq[:, None] + b_sq[None, :] - 2 * gram
    return np.exp(-dist / (2 * radius ** 2))


def svm_train(matrix, category, radius):
    """Fit an RBF-kernel SVM; return the state needed for prediction."""
    m = matrix.shape[0]
    y = 2 * category - 1                     # map {0,1} -> {-1,+1}
    x = (matrix > 0).astype(float)           # binary word-presence features
    sq = np.sum(x * x, axis=1)
    k = _rbf_gram(sq, sq, x @ x.T, radius)

    alpha = np.zeros(m)
    alpha_avg = np.zeros(m)
    reg = 1.0 / (64 * m)
    steps = 10 * m

    for t in range(steps):
        i = int(np.random.rand() * m)
        margin = y[i] * k[i, :] @ alpha
        grad = m * reg * k[:, i] * alpha[i]
        if margin < 1:
            grad -= y[i] * k[:, i]
        alpha -= grad / np.sqrt(t + 1)
        alpha_avg += alpha

    alpha_avg /= steps
    return {'alpha_avg': alpha_avg, 'x_train': x, 'sq_train': sq}


def svm_predict(state, matrix, radius):
    """Predict {0,1} labels for ``matrix`` under a trained SVM ``state``."""
    x = (matrix > 0).astype(float)
    sq = np.sum(x * x, axis=1)
    k = _rbf_gram(sq, state['sq_train'], x @ state['x_train'].T, radius)
    scores = k @ state['alpha_avg']
    return (1 + np.sign(scores)) // 2


def train_and_predict_svm(train_matrix, train_labels, test_matrix, radius):
    """Convenience wrapper: train on the train set, predict the test set."""
    state = svm_train(train_matrix, train_labels, radius)
    return svm_predict(state, test_matrix, radius)

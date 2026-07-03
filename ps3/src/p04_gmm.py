"""PS3 Problem 4: EM for a Gaussian Mixture Model (unsupervised + semi-supervised).

Unsupervised EM fits K Gaussians to unlabeled points.  The semi-supervised
variant additionally uses a handful of labeled points, weighted by ``alpha``,
which both breaks the label-permutation symmetry and speeds convergence.  The
log-likelihood is printed each run and is monotonically increasing, as required
by part (a).
"""
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

PLOT_COLORS = ['red', 'green', 'blue', 'orange']
K = 4            # number of mixture components
NUM_TRIALS = 3   # random restarts
UNLABELED = -1


def _gaussian_pdf(x, mu, sigma):
    """N(x; mu, sigma) for every row of x, returned as shape (m,)."""
    n = x.shape[1]
    diff = x - mu
    sigma_inv = np.linalg.inv(sigma)
    exponent = -0.5 * np.sum((diff @ sigma_inv) * diff, axis=1)
    norm = 1.0 / (np.power(2 * np.pi, n / 2) * np.sqrt(np.linalg.det(sigma)))
    return norm * np.exp(exponent)


def _weighted_pdf_matrix(x, phi, mu, sigma):
    """Matrix P where P[i, j] = phi_j * N(x_i; mu_j, sigma_j)."""
    p = np.zeros((x.shape[0], K))
    for j in range(K):
        p[:, j] = phi[j] * _gaussian_pdf(x, mu[j], sigma[j])
    return p


def run_em(x, w, phi, mu, sigma, eps=1e-3, max_iter=1000):
    """Standard (unsupervised) EM."""
    ll = prev_ll = None
    it = 0
    while it < max_iter and (prev_ll is None or abs(ll - prev_ll) >= eps):
        # E-step: posterior responsibilities.
        p = _weighted_pdf_matrix(x, phi, mu, sigma)
        w = p / p.sum(axis=1, keepdims=True)

        # M-step.
        phi = w.mean(axis=0)
        for j in range(K):
            wj = w[:, j]
            mu[j] = (x.T @ wj) / wj.sum()
            diff = x - mu[j]
            sigma[j] = (diff * wj[:, None]).T @ diff / wj.sum()

        prev_ll = ll
        ll = np.sum(np.log(_weighted_pdf_matrix(x, phi, mu, sigma).sum(axis=1)))
        it += 1
    print(f'[p04] unsupervised EM: {it} iterations, final log-likelihood={ll:.2f}')
    return w, ll


def run_semi_supervised_em(x, x_tilde, z, w, phi, mu, sigma,
                           alpha=20.0, eps=1e-3, max_iter=1000):
    """Semi-supervised EM: labeled points contribute with weight ``alpha``."""
    z = z.reshape(-1).astype(int)
    m_tilde = x_tilde.shape[0]
    ll = prev_ll = None
    it = 0
    while it < max_iter and (prev_ll is None or abs(ll - prev_ll) >= eps):
        # E-step (unlabeled only).
        p = _weighted_pdf_matrix(x, phi, mu, sigma)
        w = p / p.sum(axis=1, keepdims=True)

        # M-step, combining soft (unlabeled) and hard (labeled) assignments.
        for j in range(K):
            wj = w[:, j]
            xt_j = x_tilde[z == j]
            n_lab = xt_j.shape[0]
            denom = wj.sum() + alpha * n_lab

            phi[j] = denom / (x.shape[0] + alpha * m_tilde)
            mu[j] = (x.T @ wj + alpha * xt_j.sum(axis=0)) / denom
            diff_u = x - mu[j]
            diff_l = xt_j - mu[j]
            sigma[j] = ((diff_u * wj[:, None]).T @ diff_u
                        + alpha * diff_l.T @ diff_l) / denom

        prev_ll = ll
        ll_u = np.sum(np.log(_weighted_pdf_matrix(x, phi, mu, sigma).sum(axis=1)))
        ll_l = 0.0
        for j in range(K):
            xt_j = x_tilde[z == j]
            if xt_j.shape[0]:
                ll_l += alpha * np.sum(np.log(phi[j] * _gaussian_pdf(xt_j, mu[j], sigma[j])))
        ll = ll_u + ll_l
        it += 1
    print(f'[p04] semi-supervised EM: {it} iterations, final log-likelihood={ll:.2f}')
    return w, ll


def _initialize(x):
    """Split points uniformly at random into K groups for the initial params."""
    m = x.shape[0]
    idx = np.random.permutation(m)
    groups = np.array_split(idx, K)
    mu = [x[g].mean(axis=0) for g in groups]
    sigma = [np.cov(x[g], rowvar=False) for g in groups]
    phi = np.ones(K) / K
    w = np.ones((m, K)) / K
    return w, phi, mu, sigma


def main(is_semi_supervised, trial_num, data_dir='../../data/ps3'):
    print(f'[p04] running {"semi-supervised" if is_semi_supervised else "unsupervised"} '
          f'EM, trial {trial_num}')
    x_all, z_all = load_gmm_dataset(f'{data_dir}/ds4_train.csv')

    x, x_tilde, z = x_all, None, None
    if is_semi_supervised:
        labeled = (z_all != UNLABELED).squeeze()
        x_tilde = x_all[labeled, :]
        z = z_all[labeled, :]
        x = x_all[~labeled, :]

    w, phi, mu, sigma = _initialize(x)
    if is_semi_supervised:
        w, ll = run_semi_supervised_em(x, x_tilde, z, w, phi, mu, sigma)
    else:
        w, ll = run_em(x, w, phi, mu, sigma)

    z_pred = np.argmax(w, axis=1)
    plot_gmm_preds(x, z_pred, is_semi_supervised, trial_num)
    return ll


def plot_gmm_preds(x, z, with_supervision, plot_id):
    plt.figure(figsize=(12, 8))
    plt.title(f'{"Semi-supervised" if with_supervision else "Unsupervised"} GMM predictions')
    plt.xlabel('x_1')
    plt.ylabel('x_2')
    for x1, x2, zi in zip(x[:, 0], x[:, 1], z):
        color = 'gray' if zi < 0 else PLOT_COLORS[int(zi)]
        plt.scatter(x1, x2, marker='.', c=color, alpha=0.75)
    suffix = '_ss' if with_supervision else ''
    plt.savefig(os.path.join('output', f'p04_pred{suffix}_{plot_id}.png'))
    plt.close()


def load_gmm_dataset(csv_path):
    with open(csv_path, 'r') as fh:
        headers = fh.readline().strip().split(',')
    x_cols = [i for i, h in enumerate(headers) if h.startswith('x')]
    z_cols = [i for i, h in enumerate(headers) if h == 'z']
    x = np.loadtxt(csv_path, delimiter=',', skiprows=1, usecols=x_cols, dtype=float)
    z = np.loadtxt(csv_path, delimiter=',', skiprows=1, usecols=z_cols, dtype=float)
    if z.ndim == 1:
        z = np.expand_dims(z, axis=-1)
    return x, z


if __name__ == '__main__':
    os.makedirs('output', exist_ok=True)
    np.random.seed(229)
    for t in range(NUM_TRIALS):
        main(is_semi_supervised=False, trial_num=t)
        main(is_semi_supervised=True, trial_num=t)

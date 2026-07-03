"""Shared utilities for CS229 PS1: dataset loading and 2-D plotting."""
import matplotlib
matplotlib.use('Agg')  # headless backend, no windows
import matplotlib.pyplot as plt
import numpy as np


def add_intercept(x):
    """Prepend a column of ones (the intercept term) to a design matrix.

    Args:
        x: 2-D array of shape (m, n).

    Returns:
        Array of shape (m, n + 1) whose first column is all ones.
    """
    new_x = np.zeros((x.shape[0], x.shape[1] + 1), dtype=x.dtype)
    new_x[:, 0] = 1
    new_x[:, 1:] = x
    return new_x


def load_dataset(csv_path, label_col='y', intercept=False):
    """Load a CSV dataset whose feature columns start with 'x'.

    Args:
        csv_path: Path to the CSV file.
        label_col: Column name to use as the label ('y' or 't').
        intercept: If True, prepend an intercept column to the features.

    Returns:
        inputs: Feature matrix of shape (m, n) (or (m, n + 1) with intercept).
        labels: Label vector of shape (m,).
    """
    allowed = ('y', 't')
    if label_col not in allowed:
        raise ValueError(f'Invalid label_col: {label_col} (expected {allowed})')

    with open(csv_path, 'r') as fh:
        headers = fh.readline().strip().split(',')

    x_cols = [i for i, h in enumerate(headers) if h.startswith('x')]
    l_cols = [i for i, h in enumerate(headers) if h == label_col]

    inputs = np.loadtxt(csv_path, delimiter=',', skiprows=1, usecols=x_cols)
    labels = np.loadtxt(csv_path, delimiter=',', skiprows=1, usecols=l_cols)

    if inputs.ndim == 1:
        inputs = np.expand_dims(inputs, -1)
    if intercept:
        inputs = add_intercept(inputs)
    return inputs, labels


def plot(x, y, theta, save_path, correction=1.0):
    """Scatter a 2-D binary dataset and draw a linear decision boundary.

    The boundary is theta^T x = 0, solved for x2 as a function of x1.

    Args:
        x: Design matrix with an intercept column (m, 3).
        y: Labels in {0, 1}, shape (m,).
        theta: Model parameters, shape (3,).
        save_path: Output PNG path.
        correction: Multiplicative correction on the intercept (PS1 2e only).
    """
    plt.figure()
    plt.plot(x[y == 1, -2], x[y == 1, -1], 'bx', linewidth=2)
    plt.plot(x[y == 0, -2], x[y == 0, -1], 'go', linewidth=2)

    m1 = (x[:, -2].max() - x[:, -2].min()) * 0.2
    m2 = (x[:, -1].max() - x[:, -1].min()) * 0.2
    x1 = np.arange(x[:, -2].min() - m1, x[:, -2].max() + m1, 0.01)
    x2 = -(theta[0] / theta[2] * correction + theta[1] / theta[2] * x1)
    plt.plot(x1, x2, c='red', linewidth=2)
    plt.xlim(x[:, -2].min() - m1, x[:, -2].max() + m1)
    plt.ylim(x[:, -1].min() - m2, x[:, -1].max() + m2)

    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.savefig(save_path)
    plt.close()

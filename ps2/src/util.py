"""Shared utilities for CS229 PS2."""
import csv
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def add_intercept(x):
    new_x = np.zeros((x.shape[0], x.shape[1] + 1), dtype=x.dtype)
    new_x[:, 0] = 1
    new_x[:, 1:] = x
    return new_x


def load_csv(csv_path, label_col='y', intercept=False):
    """Load a CSV whose feature columns start with 'x'."""
    with open(csv_path, 'r', newline='') as fh:
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


def load_spam_dataset(tsv_path):
    """Load the SMS spam TSV as (messages, binary labels) where 1 == spam."""
    messages, labels = [], []
    with open(tsv_path, 'r', newline='', encoding='utf8') as fh:
        for label, message in csv.reader(fh, delimiter='\t'):
            messages.append(message)
            labels.append(1 if label == 'spam' else 0)
    return messages, np.array(labels)


def plot_contour(predict_fn):
    """Fill a 2-D decision region using the sign of predict_fn on a grid."""
    x, y = np.meshgrid(np.linspace(-10, 10, num=20), np.linspace(-10, 10, num=20))
    z = np.zeros(x.shape)
    for i in range(x.shape[0]):
        for j in range(y.shape[1]):
            z[i, j] = predict_fn([x[i, j], y[i, j]])
    plt.contourf(x, y, z, levels=[-float('inf'), 0, float('inf')],
                 colors=['orange', 'cyan'])


def plot_points(x, y):
    """Scatter 2-D points coloured by binary label."""
    x0, x1 = x[y == 0, :], x[y == 1, :]
    plt.scatter(x0[:, 0], x0[:, 1], marker='x', color='red')
    plt.scatter(x1[:, 0], x1[:, 1], marker='o', color='blue')


def write_json(filename, value):
    with open(filename, 'w') as f:
        json.dump(value, f)

"""PS2 Problem 5: Kernelized perceptron with dot-product and RBF kernels.

The perceptron keeps its hypothesis as a list of (beta, x) support terms.  A
prediction is sign(sum_j beta_j K(x_j, x)); after each example we append a new
term with beta = lr * (y - prediction).  With a nonlinear kernel this learns a
nonlinear decision boundary that the plain dot-product kernel cannot.
"""
import math

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

import util


def initial_state():
    return []


def sign(a):
    return 1 if a >= 0 else 0


def dot_kernel(a, b):
    return np.dot(a, b)


def rbf_kernel(a, b, sigma=1):
    a, b = np.asarray(a), np.asarray(b)
    return math.exp(-(a - b) @ (a - b) / (2 * sigma ** 2))


def predict(state, kernel, x_i):
    total = 0.0
    for beta, x in state:
        total += beta * kernel(x, x_i)
    return sign(total)


def update_state(state, kernel, learning_rate, x_i, y_i):
    beta = learning_rate * (y_i - predict(state, kernel, x_i))
    state.append((beta, x_i))


def train_perceptron(kernel_name, kernel, learning_rate, data_dir):
    train_x, train_y = util.load_csv(f'{data_dir}/ds5_train.csv')
    state = initial_state()
    for x_i, y_i in zip(train_x, train_y):
        update_state(state, kernel, learning_rate, x_i, y_i)

    test_x, test_y = util.load_csv(f'{data_dir}/ds5_test.csv')

    plt.figure(figsize=(12, 8))
    util.plot_contour(lambda a: predict(state, kernel, a))
    util.plot_points(test_x, test_y)
    plt.title(f'Kernel perceptron decision region ({kernel_name})')
    plt.savefig(f'output/p05_{kernel_name}_output.png')
    plt.close()

    pred_y = np.array([predict(state, kernel, test_x[i]) for i in range(test_y.shape[0])])
    np.savetxt(f'output/p05_{kernel_name}_predictions.txt', pred_y, fmt='%d')
    acc = np.mean(pred_y == test_y)
    print(f'[p05] {kernel_name} kernel: test accuracy = {acc:.4f}')
    return acc


def main(data_dir='../../data/ps2'):
    return {
        'dot': train_perceptron('dot', dot_kernel, 0.5, data_dir),
        'rbf': train_perceptron('rbf', rbf_kernel, 0.5, data_dir),
    }


if __name__ == '__main__':
    import os
    os.makedirs('output', exist_ok=True)
    main()

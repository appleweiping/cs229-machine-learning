"""PS1 Problem 5(b): Locally weighted linear regression (LWR)."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

import util
from linear_model import LinearModel


def main(tau, train_path, eval_path):
    x_train, y_train = util.load_dataset(train_path, intercept=True)

    model = LocallyWeightedLinearRegression(tau=tau)
    model.fit(x_train, y_train)

    x_eval, y_eval = util.load_dataset(eval_path, intercept=True)
    y_pred = model.predict(x_eval)
    mse = np.mean((y_pred - y_eval) ** 2)
    print(f'[p05b] tau={tau}: validation MSE = {mse:.6f}')

    plt.figure()
    plt.plot(x_train[:, 1], y_train, 'bx', linewidth=2, label='train')
    plt.plot(x_eval[:, 1], y_pred, 'ro', linewidth=2, label='LWR prediction')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.title(f'Locally weighted regression (tau={tau})')
    plt.savefig('output/p05b.png')
    plt.close()
    return mse


class LocallyWeightedLinearRegression(LinearModel):
    """Locally weighted linear regression.

    Non-parametric: the whole training set is retained and, for each query
    point, a weighted normal-equations fit is solved with weights

        w^(i) = exp(-||x - x^(i)||^2 / (2 tau^2)).
    """

    def __init__(self, tau):
        super().__init__()
        self.tau = tau
        self.x = None
        self.y = None

    def fit(self, x, y):
        self.x = x
        self.y = y

    def predict(self, x):
        m = x.shape[0]
        y_pred = np.zeros(m)
        for i in range(m):
            dists = np.sum((self.x - x[i]) ** 2, axis=1)
            w = np.exp(-dists / (2 * self.tau ** 2))
            xtw = self.x.T * w  # (n, m_train)
            theta = np.linalg.solve(xtw @ self.x, xtw @ self.y)
            y_pred[i] = x[i] @ theta
        return y_pred

"""PS1 Problem 1(e): Gaussian Discriminant Analysis (GDA)."""
import numpy as np

import util
from linear_model import LinearModel


def main(train_path, eval_path, pred_path):
    """Fit GDA, plot the boundary, and save predictions."""
    x_train, y_train = util.load_dataset(train_path, intercept=False)

    model = GDA()
    model.fit(x_train, y_train)

    idx = pred_path[-5]
    x_train_i = util.add_intercept(x_train)
    util.plot(x_train_i, y_train, model.theta, f'output/p01e_{idx}.png')

    x_eval, y_eval = util.load_dataset(eval_path, intercept=True)
    y_pred = model.predict(x_eval)
    np.savetxt(pred_path, y_pred > 0.5, fmt='%d')

    acc = np.mean((y_pred > 0.5) == y_eval)
    print(f'[p01e] {train_path}: validation accuracy = {acc:.4f}')
    return acc


class GDA(LinearModel):
    """Gaussian Discriminant Analysis with a shared covariance matrix.

    The class-conditional densities p(x | y) are Gaussian with a common
    covariance Sigma, so the resulting posterior p(y = 1 | x) is a logistic
    function of a linear score theta^T x.  We fit phi, mu_0, mu_1, Sigma by
    maximum likelihood and convert them to the logistic parameters theta.
    """

    def fit(self, x, y):
        m, n = x.shape
        self.theta = np.zeros(n + 1)

        n1 = np.sum(y == 1)
        phi = n1 / m
        mu_0 = x[y == 0].mean(axis=0)
        mu_1 = x[y == 1].mean(axis=0)

        x_centered = x.copy()
        x_centered[y == 0] -= mu_0
        x_centered[y == 1] -= mu_1
        sigma = x_centered.T @ x_centered / m

        sigma_inv = np.linalg.inv(sigma)
        self.theta[0] = (0.5 * (mu_0 + mu_1) @ sigma_inv @ (mu_0 - mu_1)
                         - np.log((1 - phi) / phi))
        self.theta[1:] = sigma_inv @ (mu_1 - mu_0)
        return self.theta

    def predict(self, x):
        return 1.0 / (1.0 + np.exp(-x @ self.theta))

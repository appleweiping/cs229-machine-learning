"""PS1 Problem 1(b): Logistic regression fitted with Newton's method."""
import numpy as np

import util
from linear_model import LinearModel


def main(train_path, eval_path, pred_path):
    """Train logistic regression, plot the boundary, and save predictions."""
    x_train, y_train = util.load_dataset(train_path, intercept=True)

    model = LogisticRegression(eps=1e-5)
    model.fit(x_train, y_train)

    idx = pred_path[-5]  # '1' or '2' -> which dataset
    util.plot(x_train, y_train, model.theta, f'output/p01b_{idx}.png')

    x_eval, y_eval = util.load_dataset(eval_path, intercept=True)
    y_pred = model.predict(x_eval)
    np.savetxt(pred_path, y_pred > 0.5, fmt='%d')

    acc = np.mean((y_pred > 0.5) == y_eval)
    print(f'[p01b] {train_path}: validation accuracy = {acc:.4f}')
    return acc


class LogisticRegression(LinearModel):
    """Binary logistic regression trained with Newton's method.

    Newton's method updates theta by theta <- theta - H^{-1} grad, where the
    gradient and Hessian of the averaged negative log-likelihood are

        grad = (1/m) X^T (h - y)
        H    = (1/m) X^T diag(h (1 - h)) X       (h = sigmoid(X theta)).
    """

    def fit(self, x, y):
        m, n = x.shape
        self.theta = np.zeros(n)

        for _ in range(int(1e5)):
            theta_old = self.theta.copy()
            h = _sigmoid(x @ self.theta)
            grad = x.T @ (h - y) / m
            hess = (x.T * (h * (1 - h))) @ x / m
            self.theta = self.theta - np.linalg.solve(hess, grad)
            if np.linalg.norm(self.theta - theta_old, ord=1) < self.eps:
                break
        return self.theta

    def predict(self, x):
        return _sigmoid(x @ self.theta)


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

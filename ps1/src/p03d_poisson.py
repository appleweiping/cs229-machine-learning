"""PS1 Problem 3(d): Poisson regression via gradient ascent (a GLM)."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

import util
from linear_model import LinearModel


def main(lr, train_path, eval_path, pred_path):
    x_train, y_train = util.load_dataset(train_path, intercept=True)

    model = PoissonRegression(step_size=lr, eps=1e-5)
    model.fit(x_train, y_train)

    x_eval, y_eval = util.load_dataset(eval_path, intercept=True)
    y_pred = model.predict(x_eval)
    np.savetxt(pred_path, y_pred)

    plt.figure()
    plt.plot(y_eval, y_pred, 'bx')
    plt.xlabel('true counts')
    plt.ylabel('predicted counts')
    plt.title('Poisson regression: predicted vs. true')
    plt.savefig('output/p03d.png')
    plt.close()

    corr = np.corrcoef(y_eval, y_pred)[0, 1]
    print(f'[p03d] validation corr(true, pred) = {corr:.4f}')
    return corr


class PoissonRegression(LinearModel):
    """Poisson GLM with canonical (log) link, fit by batch gradient ascent.

    The response y is modelled as Poisson with mean exp(theta^T x).  The
    log-likelihood gradient for the canonical link is simply

        grad = (1/m) X^T (y - exp(X theta)).
    """

    def fit(self, x, y):
        m, n = x.shape
        self.theta = np.zeros(n)

        for _ in range(int(1e7)):
            theta_old = self.theta.copy()
            grad = x.T @ (y - np.exp(x @ self.theta)) / m
            self.theta = self.theta + self.step_size * grad
            if np.linalg.norm(self.theta - theta_old, ord=1) < self.eps:
                break
        return self.theta

    def predict(self, x):
        return np.exp(x @ self.theta)

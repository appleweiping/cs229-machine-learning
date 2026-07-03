"""PS1 Problem 5(c): Tune the LWR bandwidth tau on the validation set."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

import util
from p05b_lwr import LocallyWeightedLinearRegression


def main(tau_values, train_path, valid_path, test_path, pred_path):
    x_train, y_train = util.load_dataset(train_path, intercept=True)
    x_valid, y_valid = util.load_dataset(valid_path, intercept=True)
    x_test, y_test = util.load_dataset(test_path, intercept=True)

    model = LocallyWeightedLinearRegression(tau=tau_values[0])
    model.fit(x_train, y_train)

    mses = []
    for tau in tau_values:
        model.tau = tau
        y_pred = model.predict(x_valid)
        mse = np.mean((y_pred - y_valid) ** 2)
        mses.append(mse)
        print(f'[p05c] tau={tau}: validation MSE = {mse:.6f}')

        plt.figure()
        plt.title(f'tau = {tau}')
        plt.plot(x_train[:, 1], y_train, 'bx', linewidth=2)
        plt.plot(x_valid[:, 1], y_pred, 'ro', linewidth=2)
        plt.xlabel('x')
        plt.ylabel('y')
        plt.savefig(f'output/p05c_tau_{tau}.png')
        plt.close()

    tau_opt = tau_values[int(np.argmin(mses))]
    print(f'[p05c] best validation MSE={min(mses):.6f} at tau={tau_opt}')

    model.tau = tau_opt
    y_pred = model.predict(x_test)
    np.savetxt(pred_path, y_pred)
    test_mse = np.mean((y_pred - y_test) ** 2)
    print(f'[p05c] test MSE at tau={tau_opt}: {test_mse:.6f}')
    return dict(tau_opt=tau_opt, test_mse=test_mse)

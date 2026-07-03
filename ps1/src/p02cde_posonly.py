"""PS1 Problem 2: Logistic regression with positive-only (partial) labels.

Each example has a hidden true label t in {0, 1} and an observed label y.
Positives are labelled (y = 1) only with probability alpha, and negatives are
never labelled, so y = 1 => t = 1 but y = 0 is ambiguous.  We compare:

    (c) an "oracle" model trained on the true labels t,
    (d) a naive model trained on the observed labels y,
    (e) the naive model rescaled by the estimated alpha to recover t.
"""
import numpy as np

import util
from p01b_logreg import LogisticRegression

WILDCARD = 'X'


def main(train_path, valid_path, test_path, pred_path):
    pred_c = pred_path.replace(WILDCARD, 'c')
    pred_d = pred_path.replace(WILDCARD, 'd')
    pred_e = pred_path.replace(WILDCARD, 'e')

    # (c) Oracle: train and evaluate on the true label t.
    x_train, t_train = util.load_dataset(train_path, label_col='t', intercept=True)
    x_test, t_test = util.load_dataset(test_path, label_col='t', intercept=True)
    model_t = LogisticRegression()
    model_t.fit(x_train, t_train)
    util.plot(x_test, t_test, model_t.theta, 'output/p02c.png')
    np.savetxt(pred_c, model_t.predict(x_test) > 0.5, fmt='%d')
    acc_c = np.mean((model_t.predict(x_test) > 0.5) == t_test)

    # (d) Naive: train on the observed label y, evaluate against true t.
    x_train, y_train = util.load_dataset(train_path, label_col='y', intercept=True)
    model_y = LogisticRegression()
    model_y.fit(x_train, y_train)
    util.plot(x_test, t_test, model_y.theta, 'output/p02d.png')
    y_pred = model_y.predict(x_test)
    np.savetxt(pred_d, y_pred > 0.5, fmt='%d')
    acc_d = np.mean((y_pred > 0.5) == t_test)

    # (e) Correction: estimate alpha = E[p(y=1|x)] over the labelled (y=1)
    # validation examples, then divide the naive scores by alpha.
    x_valid, y_valid = util.load_dataset(valid_path, label_col='y', intercept=True)
    alpha = np.mean(model_y.predict(x_valid[y_valid == 1]))
    correction = 1 + np.log(2 / alpha - 1) / model_y.theta[0]
    util.plot(x_test, t_test, model_y.theta, 'output/p02e.png', correction)
    t_pred = y_pred / alpha
    np.savetxt(pred_e, t_pred > 0.5, fmt='%d')
    acc_e = np.mean((t_pred > 0.5) == t_test)

    print(f'[p02] alpha estimate = {alpha:.4f}')
    print(f'[p02] test accuracy vs true t:  (c) oracle={acc_c:.4f}  '
          f'(d) naive={acc_d:.4f}  (e) corrected={acc_e:.4f}')
    return dict(alpha=alpha, acc_c=acc_c, acc_d=acc_d, acc_e=acc_e)

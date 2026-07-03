"""Driver for PS1. Run from the ps1/src directory:  python run.py [problem]"""
import argparse
import os

from p01b_logreg import main as p01b
from p01e_gda import main as p01e
from p02cde_posonly import main as p02
from p03d_poisson import main as p03
from p05b_lwr import main as p05b
from p05c_tau import main as p05c

DATA = os.path.join('..', '..', 'data', 'ps1')


def d(name):
    return os.path.join(DATA, name)


def run(p_num):
    os.makedirs('output', exist_ok=True)

    if p_num in (0, 1):
        p01b(d('ds1_train.csv'), d('ds1_valid.csv'), 'output/p01b_pred_1.txt')
        p01b(d('ds2_train.csv'), d('ds2_valid.csv'), 'output/p01b_pred_2.txt')
        p01e(d('ds1_train.csv'), d('ds1_valid.csv'), 'output/p01e_pred_1.txt')
        p01e(d('ds2_train.csv'), d('ds2_valid.csv'), 'output/p01e_pred_2.txt')

    if p_num in (0, 2):
        p02(d('ds3_train.csv'), d('ds3_valid.csv'), d('ds3_test.csv'),
            'output/p02X_pred.txt')

    if p_num in (0, 3):
        p03(1e-7, d('ds4_train.csv'), d('ds4_valid.csv'), 'output/p03d_pred.txt')

    if p_num in (0, 5):
        p05b(5e-1, d('ds5_train.csv'), d('ds5_valid.csv'))
        p05c([3e-2, 5e-2, 1e-1, 5e-1, 1e0, 1e1],
             d('ds5_train.csv'), d('ds5_valid.csv'), d('ds5_test.csv'),
             'output/p05c_pred.txt')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('p_num', nargs='?', type=int, default=0,
                        help='Problem number to run (0 = all).')
    run(parser.parse_args().p_num)

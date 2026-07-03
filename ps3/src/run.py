"""Driver for PS3. Run from ps3/src:  python run.py [problem]"""
import argparse
import os

import numpy as np

import p04_gmm
import p05_kmeans

DATA = os.path.join('..', '..', 'data', 'ps3')


def run(p_num):
    os.makedirs('output', exist_ok=True)

    if p_num in (0, 4):
        np.random.seed(229)
        for t in range(p04_gmm.NUM_TRIALS):
            p04_gmm.main(is_semi_supervised=False, trial_num=t, data_dir=DATA)
            p04_gmm.main(is_semi_supervised=True, trial_num=t, data_dir=DATA)

    if p_num in (0, 5):
        p05_kmeans.main(p05_kmeans.build_args(DATA))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('p_num', nargs='?', type=int, default=0)
    run(parser.parse_args().p_num)

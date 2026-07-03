"""Driver for PS4. Run from ps4/src:  python run.py [problem]"""
import argparse
import os

import p04_ica
import p06_cartpole

DATA = os.path.join('..', '..', 'data', 'ps4')


def run(p_num):
    os.makedirs('output', exist_ok=True)
    if p_num in (0, 4):
        p04_ica.main(DATA)
    if p_num in (0, 6):
        p06_cartpole.main(plot=True, seed=0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('p_num', nargs='?', type=int, default=0)
    run(parser.parse_args().p_num)

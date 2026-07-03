"""Driver for PS2. Run from ps2/src:  python run.py [problem]"""
import argparse
import os

from p05_percept import main as p05
from p06_spam import main as p06

DATA = os.path.join('..', '..', 'data', 'ps2')


def run(p_num):
    os.makedirs('output', exist_ok=True)
    if p_num in (0, 5):
        p05(DATA)
    if p_num in (0, 6):
        p06(DATA)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('p_num', nargs='?', type=int, default=0)
    run(parser.parse_args().p_num)

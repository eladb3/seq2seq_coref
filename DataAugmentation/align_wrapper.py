from utils import get_alignment
import pickle
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file1",type=str)
    parser.add_argument("--file2",type=str)
    parser.add_argument("--out",type=str)
    return parser.parse_args()


def main(args):
    with open(args.file1, 'rb') as f: seq1 = pickle.load(f)
    with open(args.file2, 'rb') as f: seq2 = pickle.load(f)
    res = get_alignment(seq1, seq2)
    with open(args.out, 'wb') as f: pickle.dump(res, f)
    return 1

if __name__ == '__main__':
    args = parse_args()
    main(args)
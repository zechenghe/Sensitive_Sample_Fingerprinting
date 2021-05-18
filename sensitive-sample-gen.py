import sys
import os
import re

import six.moves.cPickle as pickle
import numpy as np
import torch
import os


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--input_dir', type = str, default = 'data/VGGFace-Clean', help='Directory of input data')
    parser.add_argument('--input_dir', type = str, default = 'data/VGGFace-Clean', help='Directory of trojaned data')
    parser.add_argument('--clean_model', type = str, default = 'model/VGGFace-Clean.ckpt', help='Clean model')
    parser.add_argument('--trojaned_model', type = str, default = 'model/VGGFace-Trojaned.ckpt', help='Trojaned model')
    args = parser.parse_args()

    


if __name__ == '__main__':
    main()

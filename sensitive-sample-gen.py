import sys
import os
import re
import argparse

import six.moves.cPickle as pickle
import numpy as np
import torch
import os
import torchvision

import net
import utils

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument('--input_dir_clean', type = str, default = 'data/VGGFace-Clean', help='Directory of input data')
    parser.add_argument('--input_dir_trojaned', type = str, default = 'data/VGGFace-Clean', help='Directory of trojaned data')
    parser.add_argument('--model_clean', type = str, default = 'model/VGG-face-clean.pt', help='Clean model')
    parser.add_argument('--model_trojaned', type = str, default = 'model/VGG-face-trojaned.pt', help='Trojaned model')
    parser.add_argument('--gpu', type = str, default = 'model/VGG-face-trojaned.pt', help='Trojaned model')
    args = parser.parse_args()


    model = net.VGG16FaceNet()
    model.load_state_dict(torch.load(args.model_clean))

    model_trojaned = net.VGG16FaceNet()
    model_trojaned.load_state_dict(torch.load(args.model_trojaned))

    if args.gpu:
        model.cuda

    for file_name in os.listdir(args.input_dir_clean):
        if file_name.startswith('.'):
            continue

        img = utils.read_img(os.path.join(args.input_dir_clean, file_name))
        img = torch.unsqueeze(img, 0)

        logits = torch.squeeze(model(img))
        print(file_name, torch.argmax(logits))

if __name__ == '__main__':
    main()

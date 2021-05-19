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
    parser.add_argument('--label_file', type = str, default = 'data/names.txt', help='Labels')
    parser.add_argument('--model_clean', type = str, default = 'model/VGG-face-clean.pt', help='Clean model')
    parser.add_argument('--model_trojaned', type = str, default = 'model/VGG-face-trojaned.pt', help='Trojaned model')
    parser.add_argument('--gpu', dest='gpu', action='store_true', help='Use gpu')
    parser.set_defaults(gpu=False)

    args = parser.parse_args()


    model = net.VGG16FaceNet()
    model.load_state_dict(torch.load(args.model_clean))

    #model_trojaned = net.VGG16FaceNet()
    #model_trojaned.load_state_dict(torch.load(args.model_trojaned))

    if args.gpu:
        model.cuda()
    #    model_trojaned.cuda()

    name_to_label, label_to_name = utils.get_label(args.label_file)


    for file_name in os.listdir(args.input_dir_clean):
        if file_name.startswith('.'):
            continue

        name = file_name[:re.search(r"\d", file_name).start()-1]
        label = name_to_label[name]

        img = utils.read_img(os.path.join(args.input_dir_clean, file_name))
        img = torch.unsqueeze(img, 0)

        if args.gpu:
            img.cuda()

        logits = torch.squeeze(model(img))
        pred_label = torch.argmax(logits)
        print(file_name, torch.argmax(logits), label)



if __name__ == '__main__':
    main()

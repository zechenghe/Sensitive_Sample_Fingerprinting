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


def eval(input_dir, label_file, model, gpu=False, attack_target=0, model2=None):
    """
        Evaluate model accuracy (and attack success rate on a trojaned dataset).
    """

    name_to_label, label_to_name = utils.get_label(label_file)
    pred_labels = []
    if model2 is not None:
        pred_labels2 = []
    ground_truth = []

    for file_name in os.listdir(input_dir):
        if file_name.startswith('.'):
            continue

        name = file_name[:re.search(r"\d", file_name).start()-1]
        label = name_to_label[name]

        img = utils.read_img(os.path.join(input_dir, file_name))
        img = torch.unsqueeze(img, 0)

        if gpu:
            img = img.cuda()

        logits = torch.squeeze(model(img))
        pred_label = torch.argmax(logits)

        pred_labels.append(pred_label.detach().cpu().numpy())
        ground_truth.append(label)

        if model2 is not None:
            logits2 = torch.squeeze(model2(img))
            pred_label2 = torch.argmax(logits2)
            pred_labels2.append(pred_label2.detach().cpu().numpy())

    pred_labels = np.array(pred_labels)
    ground_truth = np.array(ground_truth)
    acc = np.mean(pred_labels == ground_truth)
    attack_success_rate = np.mean(pred_labels == np.array([attack_target]*len(pred_labels)))

    if model2 is None:
        return acc, attack_success_rate
    else:
        pred_labels2 = np.array(pred_labels2)
        model_diff = np.mean(pred_labels != pred_labels2)
        return model_diff


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument('--input_dir_clean', type = str, default = 'data/VGGFace-Clean', help='Directory of input data')
    parser.add_argument('--input_dir_trojaned', type = str, default = 'data/VGGFace-Trojaned', help='Directory of trojaned data')
    parser.add_argument('--label_file', type = str, default = 'data/names.txt', help='Labels')
    parser.add_argument('--model_clean', type = str, default = 'model/VGG-face-clean.pt', help='Clean model')
    parser.add_argument('--model_trojaned', type = str, default = 'model/VGG-face-trojaned.pt', help='Trojaned model')
    parser.add_argument('--gpu', dest='gpu', action='store_true', help='Use gpu')
    parser.set_defaults(gpu=False)

    parser.add_argument('--nosanity_check', dest='sanity_check', action='store_false', help='Sanity check. Evaluate models before sample generation')
    parser.set_defaults(sanity_check=True)

    args = parser.parse_args()

    model = net.VGG16FaceNet()
    model.load_state_dict(torch.load(args.model_clean))

    model_trojaned = net.VGG16FaceNet()
    model_trojaned.load_state_dict(torch.load(args.model_trojaned))


    if args.gpu:
        model.cuda()
        model_trojaned.cuda()

    if args.sanity_check:
        model_diff = eval(
            input_dir=args.input_dir_clean,
            label_file=args.label_file,
            model=model,
            gpu=args.gpu,
            model2=model_trojaned
        )
        print(f"model_diff: {model_diff}")

        accuracy, _ = eval(
            input_dir=args.input_dir_clean,
            label_file=args.label_file,
            model=model,
            gpu=args.gpu
        )
        print(f"Clean model, clean data accuracy: {accuracy}")

        accuracy, _ = eval(
            input_dir=args.input_dir_clean,
            label_file=args.label_file,
            model=model_trojaned,
            gpu=args.gpu
        )
        print(f"Trojaned model, clean data accuracy: {accuracy}")

        accuracy, _ = eval(
            input_dir=args.input_dir_trojaned,
            label_file=args.label_file,
            model=model,
            gpu=args.gpu
        )
        print(f"Clean model, trojaned data accuracy: {accuracy}")

        accuracy, attack_success_rate = eval(
            input_dir=args.input_dir_trojaned,
            label_file=args.label_file,
            model=model_trojaned,
            gpu=args.gpu
        )
        print(f"Trojaned model, trojaned data accuracy: {accuracy}, attack_success_rate: {attack_success_rate}")




if __name__ == '__main__':
    main()

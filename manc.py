"""
Maximum Active Neural Cover (MANC) selection for sensitive samples.
"""

import sys
import os
import re
import argparse

import six.moves.cPickle as pickle
import numpy as np
import torch
import torch.nn.functional as F
import torchvision

import net
import utils
import glob


def manc(candidates, model, n_samples):
    """
        Maximum Activated Neuron Cover (MANC) for sensitive-samples selection
        Args:
            candidates: a bag of candidate sensitive samples, [b, c, h, w]
            model: clean model
            n_samples: number of selected samples
        Returns:
            Selected sensitive samples [n_samples, c, h, w]
    """

    data_loader = torch.utils.data.DataLoader(candidates, batch_size=8)

    activation_maps = []
    for batch_idx, batch in enumerate(data_loader):
        activation = model.forward(batch, end_layer_name='pool5') > 0
        activation_maps.append(activation)

    candidates_activation = torch.cat(activation_maps, axis=0)
    current_union_map = torch.zeros(size=candidates_activation[0].size(), dtype=torch.bool)
    if candidates_activation.is_cuda:
        current_union_map = current_union_map.cuda()
    #for act in candidates_activation:
    #    print(act.size(), torch.sum(act))

    selected = set()
    remaining = set(range(len(candidates_activation)))

    for idx in range(n_samples):
        n_joint_activated = [(torch.sum(current_union_map | candidates_activation[i]), i) for i in remaining]

        print(n_joint_activated)

        current_selected_idx = max(n_joint_activated)[1]

        print('selected', selected)
        print('remaining', remaining)
        print('current_selected_idx', current_selected_idx)

        selected.add(current_selected_idx)
        remaining.remove(current_selected_idx)
        current_union_map = current_union_map | candidates_activation[current_selected_idx]

    return candidates[selected]



def main():

    parser = argparse.ArgumentParser()

    parser.add_argument('--candidate_dir', type = str, default = 'generate/', help='Directory of candidate sensitive samples')
    parser.add_argument('--manc_save_dir', type = str, default = 'generate/manc/', help='Directory to save MANC images')
    parser.add_argument('--model_clean', type = str, default = 'model/VGG-face-clean.pt', help='Clean model')

    parser.add_argument('--label_file', type = str, default = 'data/names.txt', help='Labels')
    parser.add_argument('--gpu', dest='gpu', action='store_true', help='Use gpu')
    parser.set_defaults(gpu=False)

    # Trojaned model and trigger images only for sanity check
    parser.add_argument('--model_trojaned', type = str, default = 'model/VGG-face-trojaned.pt', help='Trojaned model')
    parser.add_argument('--input_dir_clean', type = str, default = 'data/VGGFace-Clean', help='Directory of input data')
    parser.add_argument('--input_dir_trojaned', type = str, default = 'data/VGGFace-Trojaned', help='Directory of trojaned data')

    parser.add_argument('--nosanity_check', dest='sanity_check', action='store_false', help='Sanity check. Evaluate models before sample generation')
    parser.set_defaults(sanity_check=True)

    args = parser.parse_args()

    model = net.VGG16FaceNet()
    model.load_state_dict(torch.load(args.model_clean))

    model_trojaned = net.VGG16FaceNet()
    model_trojaned.load_state_dict(torch.load(args.model_trojaned))

    if args.gpu:
        torch.cuda.empty_cache()
        model.cuda()
        model_trojaned.cuda()

    if args.sanity_check:
        # Accuracy of clean and trojaned models
        # Percentage of inputs that are predicted differently
        accuracy_model_clean, accuracy_model_trojaned, model_diff = utils.eval(
            input_dir=args.input_dir_clean,
            label_file=args.label_file,
            model=model,
            gpu=args.gpu,
            model2=model_trojaned
        )
        print(f"Clean model accuracy on clean inputs: {accuracy_model_clean}")
        print(f"Trojaned model accuracy on clean inputs: {accuracy_model_trojaned}")
        print(f"model_diff on clean inputs: {model_diff}")

        # Attack success rate of the trojaned model
        _, attack_success_rate = utils.eval(
            input_dir=args.input_dir_trojaned,
            label_file=args.label_file,
            model=model_trojaned,
            gpu=args.gpu
        )
        print(f"Trojaned model on trojaned inputs attack_success_rate: {attack_success_rate}")

    candidates = []
    for file in glob.glob(os.path.join(args.candidate_dir, '*.npy')):
        img = utils.read_img(file)
        candidates.append(img)

    candidates = torch.cat(candidates, dim=0)
    print("candidates.size()", candidates.size())

    if args.gpu:
        candidates = candidates.cuda()

    diff = utils.pred_diff(candidates, model, model_trojaned, verbose=False)
    print(f"Without MANK {diff} candidates cause different outputs.")

    candidates_selected = manc(candidates, model, n_samples=10)
    diff = utils.pred_diff(candidates_selected, model, model_trojaned, verbose=False)
    print(f"MANK {diff} candidates cause different outputs.")

if __name__ == '__main__':
    main()

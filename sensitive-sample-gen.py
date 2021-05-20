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
        acc2 = np.mean(pred_labels2 == ground_truth)
        model_diff = np.mean(pred_labels != pred_labels2)
        return acc, acc2, model_diff



def sensitive_sample_gen(x, model, similarity_constraint=True, eps=1.0, feasibility_constraint=True, n_iter=500, lr=1.0, gpu=False, early_stop=False, early_stop_th=1.0):

    x.requires_grad = True
    x_origin = x.detach().cpu().numpy()

    optimizer = torch.optim.Adam(
        params=[x],
        lr=lr,
    )

    for i in range(n_iter):

        logits = torch.squeeze(model(x))
        w = dict(model.named_parameters())['fc8.weight']

        max_i = torch.argmax(logits)
        df_dw = torch.autograd.grad(logits[max_i], w, create_graph=True)
        loss = -torch.sum(df_dw[0].pow(2))

        sensitivity_per_weight = -loss.detach().cpu().numpy() / torch.numel(w)
        print(f"Iter {i}, Sensitivity per weight {sensitivity_per_weight}")

        if early_stop and sensitivity_per_weight > early_stop_th:
            return x

        loss.backward()
        optimizer.step()

        x_new = x.detach().cpu().numpy()
        if similarity_constraint:
            x_new = utils.similarity_projection(x_origin, x_new, eps)

        if feasibility_constraint:
            x_new = utils.feasibility_projection(x_new)

        if gpu:
            x.data = torch.tensor(x_new).cuda()
        else:
            x.data = torch.tensor(x_new)

    return x


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument('--input_dir_clean', type = str, default = 'data/VGGFace-Clean', help='Directory of input data')
    parser.add_argument('--input_dir_trojaned', type = str, default = 'data/VGGFace-Trojaned', help='Directory of trojaned data')
    parser.add_argument('--image_save_dir', type = str, default = 'generate/', help='Directory to save images')
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

    #for m in [model, model_trojaned]:
    #    for p in m.parameters():
    #        print(p.requires_grad)
    #        p.requires_grad = True

    if args.gpu:
        torch.cuda.empty_cache()
        model.cuda()
        model_trojaned.cuda()

    if args.sanity_check:
        # Accuracy of clean and trojaned models
        # Percentage of inputs that are predicted differently
        accuracy_model_clean, accuracy_model_trojaned, model_diff = eval(
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
        _, attack_success_rate = eval(
            input_dir=args.input_dir_trojaned,
            label_file=args.label_file,
            model=model_trojaned,
            gpu=args.gpu
        )
        print(f"Trojaned model on trojaned inputs attack_success_rate: {attack_success_rate}")


    results_diff = []
    results_same = []
    for file_name in os.listdir(args.input_dir_clean):
        x = utils.read_img(os.path.join(args.input_dir_clean, file_name))
        if args.gpu:
            x = x.cuda()

        x_ss = sensitive_sample_gen(
            x,
            model,
            gpu=args.gpu,
            eps=3.0,
            similarity_constraint=True,
            feasibility_constraint=True,
        )

        logits_clean = model(x_ss)
        logits_trojaned = model_trojaned(x_ss)

        diff = utils.is_diff(logits_clean, logits_trojaned, mode='topk', k=1)

        if diff:
            results_diff.append(file_name)
        else:
            results_same.append(file_name)

        person_name = file_name[:re.search(r"\d", file_name).start()-1]
        utils.save_img(torch.squeeze(x_ss), dir=args.image_save_dir, fname=f"{person_name}_sensitive_sample.png")

        print("#############")
        n_total = len(results_diff)+len(results_same)
        success_rate = float(len(results_diff)) / float(n_total)
        print(f"Total {len(results_diff)+len(results_same)} sensitive samples generated. Success rate {success_rate}.")
        print("#############")

if __name__ == '__main__':
    main()

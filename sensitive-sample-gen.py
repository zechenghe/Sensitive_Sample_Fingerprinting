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

def sensitive_sample_gen(
        x, model,
        similarity_constraint=True, similarity_mode='l2', eps=1.0,
        feasibility_constraint=True,
        n_iter=500, lr=1.0, gpu=False,
        early_stop=False, early_stop_th=1.0,
        w_tv=0.0,
        model_trojaned=None   # model_trojaned for evalutaion only
    ):

    x.requires_grad = True
    x_origin = x.detach().cpu().numpy()

    optimizer = torch.optim.Adam(
        params=[x],
        lr=lr,
    )

    for i in range(n_iter):

        logits = torch.squeeze(model(x))
        softmax_out = F.softmax(logits, dim=-1)
        w = dict(model.named_parameters())['fc8.weight']

        # max_i = torch.argmax(softmax_out)
        # Ideally we can sum over all output dimensions -- but that is computational costly.
        # In practice, we observe that summing over a subset of output dimensions is enough
        f = 0
        for sindex in range(40):
            f = f + torch.log(softmax_out[sindex] + 1e-8)

        df_dw = torch.autograd.grad(f, w, create_graph=True)
        loss_sensitivity = -torch.mean(df_dw[0]**2)

        loss_TV = utils.TV(x)
        loss = loss_sensitivity + w_tv * loss_TV

        sensitivity_per_weight = -float(loss_sensitivity)
        x_TV = float(loss_TV)
        if w_tv != 0.0:
            print(f"Iter {i}, Sensitivity per weight {sensitivity_per_weight}, TV loss {loss_TV}")
        else:
            print(f"Iter {i}, Sensitivity per weight {sensitivity_per_weight}")


        if early_stop and sensitivity_per_weight > early_stop_th:
            return x, sensitivity_per_weight

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

        if i % 50 == 0 and model_trojaned is not None:
            print(f'Iter {i}: clean model predict, {torch.argmax(torch.squeeze(model(x)))}, trojaned model predict {torch.argmax(torch.squeeze(model_trojaned(x)))}')

    return x, sensitivity_per_weight


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument('--input_dir_clean', type = str, default = 'data/VGGFace-Clean', help='Directory of input data')
    parser.add_argument('--input_dir_trojaned', type = str, default = 'data/VGGFace-Trojaned', help='Directory of trojaned data')
    parser.add_argument('--image_save_dir', type = str, default = 'generate/', help='Directory to save images')
    parser.add_argument('--label_file', type = str, default = 'data/names.txt', help='Labels')
    parser.add_argument('--model_clean', type = str, default = 'model/VGG-face-clean.pt', help='Clean model')
    parser.add_argument('--model_trojaned', type = str, default = 'model/VGG-face-trojaned.pt', help='Trojaned model')
    parser.add_argument('--lr', type=float, default = 1.0, help='learning rate')
    parser.add_argument('--w_tv', type=float, default = 0.0, help='TV weight')
    parser.add_argument('--early_stop', dest='early_stop', action='store_true', help='Early stop')
    parser.set_defaults(early_stop=False)
    parser.add_argument('--sensitivity_per_weight_th', type=float, default = 5e-4, help='Threshold to determine if the generation is successful')
    parser.add_argument('--gpu', dest='gpu', action='store_true', help='Use gpu')
    parser.add_argument('--n_sensitive_samples', type=int, default = 100, help='Number of sensitive samples to generate')
    parser.set_defaults(gpu=False)

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


    results_diff = []
    sensitivity_per_weight_diff = []
    activated_neurons_diff = []

    results_same = []
    sensitivity_per_weight_same = []
    activated_neurons_same = []

    n_generated = 0
    for file_name in os.listdir(args.input_dir_clean):
        x = utils.read_img(os.path.join(args.input_dir_clean, file_name))
        x_origin = x.clone().detach().cpu().numpy()
        if args.gpu:
            x = x.cuda()

        x_ss, sensitivity_per_weight = sensitive_sample_gen(
            x,
            model,
            gpu=args.gpu,
            similarity_constraint=True,
            feasibility_constraint=True,
            early_stop=args.early_stop,
            early_stop_th=args.sensitivity_per_weight_th,
            lr=args.lr,
            n_iter=1000,
            similarity_mode='l2',
            eps=10.0,
            w_tv=args.w_tv,
            model_trojaned=model_trojaned,
        )

        logits_clean = model(x_ss)
        logits_trojaned = model_trojaned(x_ss)

        snr = utils.snr(x_origin, x_ss.detach().cpu().numpy())
        n_activated_neurons = int(torch.sum(logits_clean > 0))
        print(snr)

        if (sensitivity_per_weight > args.sensitivity_per_weight_th) and snr > 0:
            n_generated += 1
            diff = utils.is_diff(logits_clean, logits_trojaned, mode='topk', k=1)

            if diff:
                results_diff.append(file_name)
                sensitivity_per_weight_diff.append(sensitivity_per_weight)
                activated_neurons_diff.append(n_activated_neurons)
            else:
                results_same.append(file_name)
                sensitivity_per_weight_same.append(sensitivity_per_weight)
                activated_neurons_same.append(n_activated_neurons)

            person_name = file_name[:re.search(r"\d", file_name).start()-1]
            utils.save_img(torch.squeeze(x_ss), dir=args.image_save_dir, fname=f"{person_name}_sensitive_sample")

        print("#############")
        n_total = len(results_diff)+len(results_same)
        success_rate = float(len(results_diff)) / float(n_total + 1e-8)
        print(f"Total {len(results_diff)+len(results_same)} sensitive samples generated. Success rate {success_rate}.")
        print(f"Sensitivity per weight, diff {np.mean(sensitivity_per_weight_diff) if len(sensitivity_per_weight_diff) > 0 else 0}, same {np.mean(sensitivity_per_weight_same)if len(sensitivity_per_weight_same) > 0 else 0}")
        print(f"Number of activated neurons, diff {np.mean(activated_neurons_diff) if len(activated_neurons_diff) > 0 else 0}, same {np.mean(activated_neurons_same) if len(activated_neurons_same) else 0}")
        print("#############")

        if n_generated >= args.n_sensitive_samples:
            print(f"Success: {n_generated} sensitive samples generated.")
            return

if __name__ == '__main__':
    main()

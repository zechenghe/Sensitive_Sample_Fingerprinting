import os
import re

import numpy as np
import pathlib
import torch
import matplotlib.pyplot
import torch.nn.functional as F

averageImage = np.array([93.5940, 104.7624, 129.1863])


def preprocess(img):
    return img - averageImage

def deprocess(img):
    return img + averageImage

#def similarity_projection(ref, data, eps, mode='l2'):
#    """
#        Project data to the eps ball of reference.
#    """#

#    if mode == 'l2':
#        # Eps for L2 is per-pixel, convert it to per-image
#        eps = np.sqrt(ref.size) * eps#

#    diff = data - ref#

#    if mode == 'l1':
#        r = np.max(np.abs(diff))
#    elif mode == 'l2':
#        r = np.sqrt(np.sum(diff**2))#

#    else:
#        raise NotImplmentedError(f'Mode {mode} is not supported')#

#    if r > eps:
#        if mode == 'l1':
#            new_diff = diff.copy()
#            new_diff[new_diff > eps] = eps
#            new_diff[new_diff < -eps] = -eps
#        elif mode == 'l2':
#            new_diff = diff / r * eps
#        new_data = ref + new_diff
#    else:
#         new_data = data#

#    return np.float32(new_data)

def similarity_projection(ref, data, eps):
    """
        Project data to the eps ball of reference.
    """

    eps = np.sqrt(ref.size) * eps

    diff = data - ref
    r = np.sqrt(np.sum(diff**2))

    if r > eps:
        new_diff = diff / r * eps
        new_data = ref + new_diff
    else:
         new_data = data

    return np.float32(new_data)


def _tensor_size(t):
    return t.size()[1]*t.size()[2]*t.size()[3]

def TV(x):

    """
        Total variation of an image.
    """

    batch_size = x.size()[0]
    h_x = x.size()[2]
    w_x = x.size()[3]
    count_h = _tensor_size(x[:,:,1:,:])
    count_w = _tensor_size(x[:,:,:,1:])
    h_tv = torch.pow(x[:,:,1:,:]-x[:,:,:h_x-1,:], 2).sum()
    w_tv = torch.pow(x[:,:,:,1:]-x[:,:,:,:w_x-1], 2).sum()
    return (h_tv / count_h + w_tv / count_w) / batch_size


def feasibility_projection(data):

    if data.ndim == 4:
        batch_dim = True
        data = np.squeeze(data)
    else:
        batch_dim = False

    data = np.moveaxis(data, 0, -1)
    data = deprocess(data)
    data[data > 255.0] = 255.0
    data[data < 0.0] = 0.0
    data = preprocess(data)
    data = np.moveaxis(data, -1, 0)

    if batch_dim:
        data = np.expand_dims(data, axis=0)

    return np.float32(data)

def read_img(fname):
    """
        Args:
            fname: image file name
        Returns:
            A 4-d tensor [b, c, h, w]
    """

    img = matplotlib.pyplot.imread(fname)[:, :, :3]     # matplotlib.pyplot.imsave saves an extra alpha channel
    img = preprocess(img)
    img = np.moveaxis(img, -1, 0)
    return torch.unsqueeze(torch.tensor(img.astype(np.float32)), 0)


def save_img(img, dir, fname):
    """
        Args:
            img: a 3-d tensor [c, h, w]
            fname: image file name to write
    """

    img = img.detach().cpu().numpy()
    img = np.moveaxis(img, 0, -1)
    img = deprocess(img)

    pathlib.Path(dir).mkdir(parents=True, exist_ok=True)
    matplotlib.pyplot.imsave(os.path.join(dir, fname), np.uint8(img))
    return


def get_label(names_file):
    name_to_index = {}
    index_to_name = {}
    with open(names_file, 'r') as f:
        for i, name in enumerate(f):
            name = name.strip('\n')
            name_to_index[name] = i
            index_to_name[i] = name
    return name_to_index, index_to_name


def is_diff(logits1, logits2, mode='topk', k=None, n_digits=None):
    """
        Determine if outputs from two models are different from the view of mode.
        Args:
            logits1: logits of model 1
            logits2: logits of model 2
            mode: "topk", "topk_prob"
            k: top-k
        Returns:
            Boolean if logits1 and logits2 are different from the view of mode.
    """

    assert k is not None

    s1 = F.softmax(torch.squeeze(logits1), dim=-1)
    s2 = F.softmax(torch.squeeze(logits2), dim=-1)

    values1, idx1 = torch.topk(s1, k)
    values2, idx2 = torch.topk(s2, k)

    if mode == 'topk':
        print(idx1, idx2)
        return not torch.equal(idx1, idx2)
    elif mode == 'topk_prob':
        assert n_digits is not None
        r1 = torch.round(values1 * 10^n_digits) / (10^n_digits)
        r2 = torch.round(values2 * 10^n_digits) / (10^n_digits)
        print(r1, r2)
        return not torch.equal(r1, r2)
    else:
        raise NotImplmentedError('Mode is not supported in is_diff()')


def snr(x_origin, x):

    MSE = np.mean((x_origin - x)**2)
    S = np.mean(x_origin**2)
    SNR = 10*np.log10(S/MSE)

    return SNR


def eval(input_dir, label_file, model, gpu=False, attack_target=0, model2=None):
    """
        Evaluate model accuracy (and attack success rate on a trojaned dataset).
    """

    name_to_label, label_to_name = get_label(label_file)
    pred_labels = []
    if model2 is not None:
        pred_labels2 = []
    ground_truth = []

    for file_name in os.listdir(input_dir):
        if file_name.startswith('.'):
            continue

        name = file_name[:re.search(r"\d", file_name).start()-1]
        label = name_to_label[name]

        img = read_img(os.path.join(input_dir, file_name))

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


def pred_diff(candidates, model_clean, model_trojaned, verbose=False):

    """
        Evaluate the difference of model predictions
        Args:
            candidates: input samples,
            model_clean: clean model,
            model_trojaned: trojaned model,
        Returns:
            Portion of input data whose output are different.
    """

    def eval_model(data, model):
        logits = torch.squeeze(model(data))
        predicts = torch.argmax(logits, dim=-1)
        return predicts.detach().cpu().numpy()

    n_total = 0.0
    n_diff = 0.0

    data_loader = torch.utils.data.DataLoader(candidates, batch_size=32)

    for batch_idx, candidate in enumerate(data_loader):

        pred_clean = eval_model(candidate, model_clean)
        pred_trojaned = eval_model(candidate, model_trojaned)

        n_total += len(pred_clean)
        n_diff += np.sum(pred_clean != pred_trojaned)

        if verbose:
            print(f"Batch {batch_idx}, {len(pred_clean)} samples, current total diff rate {n_diff/n_total}", )

    return n_diff / n_total

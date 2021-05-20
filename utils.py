import os
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

def similarity_projection(ref, data, eps):
    """
        Project data to the eps ball of reference.
    """

    diff = data - ref
    r = np.sqrt(np.sum(diff**2))

    if r > eps:
        new_diff = diff / r * eps
        new_data = ref + new_diff
        print(np.sqrt(np.sum(new_diff**2)))
    else:
         new_data = data


    print(f"Eps: {eps}", r, np.sqrt(np.sum((new_data-ref)**2)))
    return np.float32(new_data)


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
    img = matplotlib.pyplot.imread(fname)
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

    print(logits1.shape, logits2.shape)
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

import numpy as np
import torch
import matplotlib.pyplot

averageImage = np.array([104.7624, 129.1863, 93.5940])

def read_img(fname):

    img = matplotlib.pyplot.imread(fname)
    img = img - averageImage
    img = np.moveaxis(img, -1, 0)
    return torch.tensor(img.astype(np.float32))

def get_label(names_file):
    name_to_index = {}
    index_to_name = {}
    with open(names_file, 'r') as f:
        for i, name in enumerate(f):
            name = name.strip('\n')
            name_to_index[name] = i
            index_to_name[i] = name
    return name_to_index, index_to_name

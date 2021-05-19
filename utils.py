import numpy as np
import torch
import matplotlib.pyplot

averageImage = np.array([93.5940, 104.7624, 129.1863])

def read_img(fname):

    img = matplotlib.pyplot.imread(fname)
    img = img - averageImage
    img = np.moveaxis(img, -1, 0)
    return torch.tensor(img.astype(np.float32))

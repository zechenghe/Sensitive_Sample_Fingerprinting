import collections

import torch
import torch.nn as nn
import torch.nn.functional as F

class VGG16FaceNet(nn.Module):

    def __init__(self):
        super(VGG16FaceNet, self).__init__()

        self.conv1_1 = nn.Conv2d(3, 64, 3)
        self.relu1_1 = nn.ReLU()
        self.conv1_2 = nn.Conv2d(64, 64, 3)
        self.relu1_2 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2, 2)

        self.conv2_1 = nn.Conv2d(64, 128, 3)
        self.relu2_1 = nn.ReLU()
        self.conv2_2 = nn.Conv2d(128, 128, 3)
        self.relu2_2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2, 2)

        self.conv3_1 = nn.Conv2d(128, 256, 3)
        self.relu3_1 = nn.ReLU()
        self.conv3_2 = nn.Conv2d(256, 256, 3)
        self.relu3_2 = nn.ReLU()
        self.conv3_3 = nn.Conv2d(256, 256, 3)
        self.relu3_3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(2, 2)

        self.conv4_1 = nn.Conv2d(256, 512, 3)
        self.relu4_1 = nn.ReLU()
        self.conv4_2 = nn.Conv2d(512, 512, 3)
        self.relu4_2 = nn.ReLU()
        self.conv4_3 = nn.Conv2d(512, 512, 3)
        self.relu4_3 = nn.ReLU()
        self.pool4 = nn.MaxPool2d(2, 2)

        self.conv5_1 = nn.Conv2d(512, 512, 3)
        self.relu5_1 = nn.ReLU()
        self.conv5_2 = nn.Conv2d(512, 512, 3)
        self.relu5_2 = nn.ReLU()
        self.conv5_3 = nn.Conv2d(512, 512, 3)
        self.relu5_3 = nn.ReLU()
        self.pool5 = nn.MaxPool2d(2, 2)

        self.features = collections.OrderedDict([
            ('conv1_1', self.conv1_1),
            ('relu1_1', self.relu1_1),
            ('conv1_2', self.conv1_2),
            ('relu1_2', self.relu1_2),
            ('pool1', self.pool1),
            ('conv2_1', self.conv2_1),
            ('relu2_1', self.relu2_1),
            ('conv2_2', self.conv2_2),
            ('relu2_2', self.relu2_2),
            ('pool2', self.pool2),
            ('conv3_1', self.conv3_1),
            ('relu3_1', self.relu3_1),
            ('conv3_2', self.conv3_2),
            ('relu3_2', self.relu3_2),
            ('conv3_3', self.conv3_3),
            ('relu3_3', self.relu3_3),
            ('pool3', self.pool3),
            ('conv4_1', self.conv4_1),
            ('relu4_1', self.relu4_1),
            ('conv4_2', self.conv4_2),
            ('relu4_2', self.relu4_2),
            ('conv4_3', self.conv4_3),
            ('relu4_3', self.relu4_3),
            ('pool4', self.pool4),
            ('conv5_1', self.conv5_1),
            ('relu5_1', self.relu5_1),
            ('conv5_2', self.conv5_2),
            ('relu5_2', self.relu5_2),
            ('conv5_3', self.conv5_3),
            ('relu5_3', self.relu5_3),
            ('pool5', self.pool5),
        ])

        self.fc6 = nn.Linear(7*7*512, 4096)
        self.relu6 = nn.ReLU()
        self.fc7 = nn.Linear(4096, 4096)
        self.relu7 = nn.ReLU()
        self.fc8 = nn.Linear(4096, 2622)

        self.classifier = collections.OrderedDict([
            ('fc6', self.fc6),
            ('relu6', self.relu6),
            ('fc7', self.fc7),
            ('relu7', self.relu7),
            ('fc8', self.fc8),
        ])

    def forward(self, x):

        for name, layer in self.features.items():
            # Kernel size = 3, pad input to simulate 'same size' convolution
            if 'conv' in name:
                x = F.pad(x, (1, 1, 1, 1))
            x = layer(x)

        x = torch.flatten(x, 1)
        for name, layer in self.classifier.items():
            x = layer(x)
        return x

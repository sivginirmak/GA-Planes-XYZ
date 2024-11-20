import torch
from torch import nn
import numpy as np

class Nerf(nn.Module):
    def __init__(self, in_features, hid_features, out_features=1):
        super(Nerf, self).__init__()

        self.fc1 = nn.Linear(in_features, hid_features)
        self.fc2 = nn.Linear(hid_features, hid_features)
        self.fc3 = nn.Linear(hid_features, out_features)

    def forward(self, x):

        x = nn.functional.relu(self.fc1(x))
        x = nn.functional.relu(self.fc2(x))
        x = self.fc3(x)
        return x


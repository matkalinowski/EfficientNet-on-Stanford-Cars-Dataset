import torch.nn as nn


class HardSigmoid(nn.Module):
    def __init__(self, inplace=True):
        super().__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6


class HardSwish(nn.Module):
    def __init__(self, inplace=True):
        super().__init__()
        self.sigmoid = HardSigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)


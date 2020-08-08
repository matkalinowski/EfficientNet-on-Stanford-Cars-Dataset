import torch.nn as nn
import torch

from models.mobile_nets.mobile_net_utils import scale_channels, parameter_generator_v3
from models.mobile_nets.activation_layers import HardSigmoid, HardSwish


class SELayer(nn.Module):
    def __init__(self, channel, reduction=4):
        super().__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, int(channel//reduction)),
            nn.ReLU(inplace=True),
            nn.Linear(int(channel//reduction), channel),
            HardSigmoid()
        )

    def forward(self, input):
        batch, channels, _, _ = input.size()
        y = self.avg_pool(input).view(batch, channels)
        y = self.fc(y).view(batch, channels, 1, 1)
        return input * y


class BottleNeckBlockV3(nn.Module):
    def __init__(self, in_channels, out_channels, expansion, kernel_size, stride,
                 squeeze_excitation, hard_swish):
        super().__init__()

        self.identity = stride == 1 and in_channels == out_channels
        self.hidden_channels = int(in_channels * expansion)

        self.block = \
            nn.Sequential(
                nn.Conv2d(in_channels, self.hidden_channels, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(self.hidden_channels),
                HardSwish() if hard_swish else nn.ReLU(inplace=True),
                nn.Conv2d(self.hidden_channels, self.hidden_channels, kernel_size, stride,
                          padding=(kernel_size - 1) // 2, groups=self.hidden_channels, bias=False),
                nn.BatchNorm2d(self.hidden_channels),
                SELayer(self.hidden_channels, reduction=4) if squeeze_excitation else nn.Identity(),
                HardSwish() if hard_swish else nn.ReLU(inplace=True),
                nn.Conv2d(self.hidden_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        if self.identity:
            return x + self.block(x)
        else:
            return self.block(x)


class MobileNetV3(nn.Module):
    def __init__(self, parameters, mode, num_classes, scaling_parameter=1):
        super().__init__()

        self.num_classes = num_classes
        self.mode = mode
        self.parameters = parameters
        self.parameters["out_channels"] = scale_channels(self.parameters["out_channels"], scaling_parameter)

        self.layers = []

        # First Convolutional layer
        output_channels = int(16*scaling_parameter)
        self.layers.append(
            nn.Sequential(
                nn.Conv2d(3, output_channels, kernel_size=3, stride=2),
                nn.BatchNorm2d(output_channels),
                HardSwish()
            )
        )

        # Bottleneck blocks
        self.bottleneck_params_generator = parameter_generator_v3(self.parameters, output_channels)
        self.layers.extend([BottleNeckBlockV3(**params) for params in self.bottleneck_params_generator])

        # Last layers before classification. Before creation we need to get all needed parameters from previous layer
        in_channels = self.parameters["out_channels"][-1]
        hidden_channels = int(in_channels * self.parameters["expansion"][-1])

        self.layers.append(
            nn.Sequential(
                nn.Conv2d(in_channels, hidden_channels, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_channels),
                HardSwish(),
                nn.AdaptiveAvgPool2d((1, 1))
            )
        )

        output_channel = {'large': 1280, 'small': 1024}
        output_channel = int(output_channel[self.mode] * scaling_parameter) if scaling_parameter > 1.0 else \
            output_channel[self.mode]

        self.model = nn.Sequential(*self.layers)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_channels, output_channel),
            HardSwish(),
            nn.Dropout(0.2),
            nn.Linear(output_channel, num_classes),
        )

    def forward(self, x):
        x = self.model(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)

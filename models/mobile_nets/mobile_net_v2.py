import torch.nn as nn
import torch

from models.mobile_nets.mobile_net_parameters import mobile2_params
from models.mobile_nets import parameter_generator_v2, scale_channels


class BottleNeckBlockV2(nn.Module):
    """
    This Neural Net block was introduce in march 2019 by Google research team.
    Main idea of this approach is to use depthwise 1x1 convolutions to expand
    the number of filters significantly before performing classical convolution
    that not alter filters number. After that, 1x1 convolution is used once
    more to stretch it into relatively small number again. This approach
    decrease needed computation without cost of lower accuracy.

    As proposed in the original paper, we will use :expansion: parameter, that
    will multiply number of filters before classic convolution.

    Layer-based it will look like this:

    Convolution 1x1 to increase channel number by :expansion:
    BatchNormalization
    ReLU
    Convolution 3x3 without changing channel number
    BatchNormalization
    ReLU
    Convolution 1x1 decreasing channels to the out_channels number
    BatchNormalization
    """

    def __init__(self, in_channels, out_channels, stride=1, expansion=1, n=1):
        super().__init__()

        expanded_channels = in_channels * expansion

        # If possible, during forward method there should be a residual connection. This flag will be used in the
        # forward method to indicate that.
        self.identity = stride == 1 and in_channels == out_channels

        self.block = nn.Sequential(
            nn.Conv2d(in_channels, expanded_channels, 1, bias=False),
            nn.BatchNorm2d(expanded_channels),
            nn.ReLU6(inplace=True),
            nn.Conv2d(expanded_channels, expanded_channels, 3, stride=stride, padding=1,
                      groups=expanded_channels, bias=False),
            nn.BatchNorm2d(expanded_channels),
            nn.ReLU6(inplace=True),
            nn.Conv2d(expanded_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels))

        # In first bottleneck, expansion rate is equal to 1. When this occur, 1x1 depthwise convolution is not doing any
        # transformations
        if expansion == 1:
            self.block = self.block[3:]

    def forward(self, input):
        if self.identity:
            return input + self.block(input)
        else:
            return self.block(input)


class MobileNetV2(nn.Module):
    """
    In this MobileNetV2 implementation we will use scaling parameter that will
    change number of output_channels.
    """

    def __init__(self, params, n_classes, scaling_parameter):
        super().__init__()

        self.n_classes = n_classes

        self.parameters = params
        self.parameters["out_channels"] = scale_channels(self.parameters["out_channels"], scaling_parameter)
        self.layer_params_generator = parameter_generator_v2(self.parameters)

        self.layers = []

        # First convolutional layer followed by BatchNormalization and ReLU activation.
        args_first_layer = next(self.layer_params_generator)
        self.layers.append(
            nn.Sequential(
                nn.Conv2d(
                    in_channels=args_first_layer["in_channels"],
                    out_channels=args_first_layer["out_channels"],
                    stride=args_first_layer["stride"],
                    kernel_size=3,
                    padding=1,
                    bias=False),
                nn.BatchNorm2d(args_first_layer["out_channels"]),
                nn.ReLU6(inplace=True),
            ))

        self.layers.extend([BottleNeckBlockV2(**params) for params in self.layer_params_generator])

        # Last part of MobileNet2 contains 1x1 convolution that significantly increase number of channels, transform the
        # tensor using average pooling, and then do the prediction using classifier which is also 1x1 convolution.
        self.layers.append(nn.Sequential(
            nn.Conv2d(in_channels=self.parameters["out_channels"][-1],
                      out_channels=int(1280 * scaling_parameter),
                      kernel_size=1,
                      stride=1,
                      bias=False),
            nn.AvgPool2d(7, stride=1),
            nn.Conv2d(int(1280 * scaling_parameter), self.n_classes, kernel_size=1, stride=1, bias=False)))

        self.model = nn.Sequential(*self.layers)

    def forward(self, input):
        x = self.model(input)
        return x.view(x.size(0), -1)

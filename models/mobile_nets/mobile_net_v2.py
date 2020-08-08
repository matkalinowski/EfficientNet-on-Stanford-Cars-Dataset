import torch.nn as nn
import torch

from models.mobile_nets import parameter_generator


class BottleNeckBlock(nn.Module):
    """
    This Neural Net block was introduce in march 2019 by Google research team.
    Main idea of this approach is to use depthwise 1x1 convolutions to expand
    the number of filters significantly before performing classical convolution
    that not alter filters number. After that, we will use 1x1 convolution once
    more to stretch them into relatively small number again. This approach
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
        print(input.shape)

        if self.identity:
            return input + self.block(input)
        else:
            return self.block(input)


class MobileNetV2(nn.Module):
    """
    In this MobileNetV2 implementation we will use scaling parameter that will
    change number of output_channels.
    """

    def __init__(self, n_classes, scaling_parameter):
        super().__init__()

        self.n_classes = n_classes
        self.scaling_parameter = scaling_parameter

        # Original parameters presented in the paper.
        self.layers_params = {
            'out_channels': self.scale_channels([32, 16, 24, 32, 64, 96, 160, 320]),
            'strides': [2, 1, 2, 2, 2, 1, 2, 1],
            'n': [1, 1, 2, 3, 4, 3, 3, 1],
            'expansion': [1, 1, 6, 6, 6, 6, 6, 6]}

        self.layer_params_generator = parameter_generator(self.layers_params)

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

        # BottleNeck convolutions
        for parameters in self.layer_params_generator:
            self.layers.append(BottleNeckBlock(parameters["in_channels"],
                                               parameters['out_channels'],
                                               parameters["stride"],
                                               parameters["expansion"]).block)

        # Last part of MobileNet2 contains 1x1 convolution that significantly increase number of channels, transform the
        # tensor using average pooling, and then do the prediction using classifier which is also 1x1 convolution.
        self.layers.append(nn.Sequential(
            nn.Conv2d(in_channels=self.layers_params["out_channels"][-1],
                      out_channels=1280 * scaling_parameter,
                      kernel_size=1,
                      stride=1,
                      bias=False),
            nn.AvgPool2d(7, stride=1),
            nn.Conv2d(1280 * scaling_parameter, self.n_classes, kernel_size=1, stride=1, bias=False)))

        self.model = nn.Sequential(*self.layers)

    def forward(self, input):
        x = self.model(input)
        return x.view(x.size(0), -1)

    def scale_channels(self, list_of_channels):
        return list(map(lambda ch: int(ch * self.scaling_parameter), list_of_channels))


if __name__ == "__main__":
    model = MobileNetV2(100, 1)
    input = torch.zeros(1, 3, 224, 224)
    print(model(input))
    # print(model)
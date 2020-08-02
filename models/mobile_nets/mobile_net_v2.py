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

    def __init__(self, in_channels, out_channels, n, stride=1,  expansion=1, first=False):
        super().__init__()

        expanded_channels = in_channels * expansion

        self.first = first

        self.layers = [nn.Conv2d(in_channels, expanded_channels, 1, bias=False),
                       nn.BatchNorm2d(expanded_channels),
                       nn.ReLU6(),
                       nn.Conv2d(expanded_channels, expanded_channels, 3, stride=stride, padding=1,
                                 groups=expanded_channels, bias=False),
                       nn.BatchNorm2d(expanded_channels),
                       nn.ReLU6(),
                       nn.Conv2d(expanded_channels, out_channels, 1, bias=False),
                       nn.BatchNorm2d(out_channels)]

        self.block = nn.Sequential(
            *[nn.Sequential(*self.layers) for _ in range(n)]
        )

    def forward(self, input):
        residual = input if not self.first else 0
        x = self.block(input)
        return x + residual


class MobileNetV2(nn.Module):
    """
    In this MobileNetV2 implementation we will use scaling parameter that will
    change number of output_channels.
    """

    def __init__(self, n_classes, scaling_parameter):
        super().__init__()

        self.n_classes = n_classes

        # Original parameters presented in the paper.
        self.layers_params = {
            'out_channels': list(map(lambda x: int(x * scaling_parameter), [32, 16, 24, 32, 64, 96, 160, 320])),
            'strides': [2, 1, 2, 2, 2, 1, 2, 1],
            'n': [1, 1, 2, 3, 4, 3, 3, 1]}

        self.layer_params_generator = parameter_generator(self.layers_params)

        self.layers = []

        # First convolutional layer followed by BatchNormalization
        args_first_layer = next(self.layer_params_generator)
        self.layers.append(
            nn.Sequential(
                nn.Conv2d(
                    in_channels=args_first_layer["in_channels"],
                    out_channels=args_first_layer["out_channels"],
                    stride=args_first_layer["stride"],
                    kernel_size=3,
                    padding=1),
                nn.BatchNorm2d(args_first_layer["out_channels"])))

        # BottleNeck convolutions
        for parameter in self.layer_params_generator:
            self.layers.append(BottleNeckBlock(expansion=6, first=False, **parameter))

        # Last part of MobileNet2 contains 1x1 convolution that significantly increase number of channels, transform the
        # tensor using average pooling, and then do the prediction using classifier which is also 1x1 convolution.
        self.layers.append(nn.Sequential(
            nn.Conv2d(in_channels=self.layers_params["out_channels"][-1],
                      out_channels=1280*scaling_parameter,
                      kernel_size=1,
                      stride=1,
                      bias=False),
            nn.AvgPool2d(7, stride=1),
            nn.Conv2d(1280*scaling_parameter, self.n_classes, kernel_size=1, stride=1, bias=False)))

        self.model = nn.Sequential(*self.layers)

    def forward(self, input):
        x = self.model(input)
        return x.view(x.size(0), -1)


if __name__ == "__main__":
    model = MobileNetV2(100, 1)
    input = torch.zeros(1, 3, 224, 224)
    print(model(input))

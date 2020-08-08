import torch.nn as nn

from models.mobile_nets.mobile_net_utils import scale_channels
from models.mobile_nets.activation_layers import HardSigmoid, HardSwish
from models.mobile_nets.mobile_net_parameters import mobile3_large, mobile3_small


class MobileNetV3(nn.Module):
    def __init__(self, parameters, mode, num_classes, scaling_parameter=1):
        super().__init__()
        assert mode in ["large", "small"]


        self.num_classes = num_classes
        self.mode = mode
        self.parameters = parameters
        self.scaling_parameter = scaling_parameter
        self.parameters["out_channels"] = scale_channels(self.parameters["out_channels"], scaling_parameter)

        # self.layer_params_generator = parameter_generator_v2(self.parameters)

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
        # self.layers.expand([BottleNeckBlockV3(**params) for params in self.])




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
    def __init__(self, input_channels, hidden_dim, output_channels, kernel, stride, use_se, use_hs):
        super().__init__()

        self.identity = stride == 1 and input_channels == output_channels


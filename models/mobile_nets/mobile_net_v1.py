import torch.nn as nn


from models.mobile_nets.mobile_net_utils import create_next_layer_calculator
from training.cars_dataset_lightning_module import StanfordCarsDatasetLightningModule


class ConvBlock(nn.Module):
    """
    Classical Convolution block that will be used in MobileNet architecture.
    It will contain Conv2d layer, followed by Batch Normalization and ReLU
    activation.
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, stride=2, kernel_size=(3, 3),
                      padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, input):
        return self.block(input)


class DepthwiseConvBlock(nn.Module):
    """
  Depthwise Separable Convolution was intruduced by Google research team in
  their MobileNet1 implementation.

  This technique is significantly reducing overall parameter number while
  maintaining almost as good accuracy as classical convolution.

  Such block will contain three different convolution layers, each for every
  dimension of the tensor. For example if we want to achieve 32 filters from
  RGB image with kernel 3x3 it will look as follows:

   - Conv2d(3,3, kernel_size=(1,3))
   - Conv2d(3,3, kernel_size=(3,1)),
   - Conv2d(3,32, kernel_size=(1,1)).

  Every block will also include two ReLU activation layers and two batch
  normalization layers.
  """

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        self.block = \
            nn.Sequential(
                nn.Conv2d(in_channels, in_channels, 3, stride, 1),
                nn.BatchNorm2d(in_channels),
                nn.ReLU(),
                nn.Conv2d(in_channels, out_channels, (1, 1), stride=1, padding=0),
                nn.BatchNorm2d(out_channels),
                nn.ReLU()
            )

    def forward(self, input):
        return self.block(input)


class Classifier(nn.Module):
    """
  This block will be used for final classification. It will contain
  FullyConnected layer with given output neurons (number of classes) followed by
  SoftMax activation.
  """

    def __init__(self, in_channels, n_classes):
        super().__init__()

        self.block = nn.Sequential(
            nn.Linear(in_channels, n_classes)
        )

    def forward(self, input):
        return self.block(input)


class MobileNetV1(StanfordCarsDatasetLightningModule):
    """
    Implementation of MobileNet. Architecture that was intruduced by Google's
    research team in april 2017. Biggest advantage of this architecture is by far
    its latency/accuracy tradeoff.

    To achieve some flexibility about number of parameters we will implement
    depth_parameter that will modify number of filters in each layer. According to
    original paper, depthwise convolutions are most computation consuming.
    """

    def __init__(self, n_classes, scaling_parameter=1, batch_size=128, image_size=224):
        super().__init__(batch_size, image_size)

        self.n_classes = n_classes

        # Preparing closure that will held information about number of channels
        self.scaling_parameter = scaling_parameter
        self.channel_calculator = create_next_layer_calculator(3, 32, self.scaling_parameter)
        n_channels = self.channel_calculator(as_is=True)

        self.layers = []

        # Starting convolution
        self.layers.append(ConvBlock(**n_channels))

        # Depthwise convolutions
        stride_out_channel_params_list = [
            # (<stride>, <out_channels>)
            # (2, 32)  First layer, already appended
            (1, 64), (2, 128), (1, 128), (2, 256), (1, 256), (2, 512),
            (1, 512), (1, 512), (1, 512), (1, 512), (1, 512), (2, 1024),
            (1, 1024)]

        for stride, out_channels in stride_out_channel_params_list:
            self.layers.append(DepthwiseConvBlock(stride=stride,
                                                  **self.channel_calculator(out_channels)))

        # Average pooling before Linear classifier
        self.layers.append(nn.AvgPool2d(7))

        self.model = nn.Sequential(*self.layers)

        self.classifier = nn.Linear(
            self.channel_calculator(as_is=True)["in_channels"], self.n_classes)

    def forward(self, input):
        x = self.model(input)
        x = x.view(-1, self.channel_calculator(as_is=True)["in_channels"])
        return self.classifier(x)

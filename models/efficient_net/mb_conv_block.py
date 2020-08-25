from copy import deepcopy

import torch
from torch import nn
from torch.nn import functional as F

from models.efficient_net.Swish import Swish
from models.efficient_net.block_params import BlockParams
from models.efficient_net.conv_2d import get_same_padding_conv2d
from models.efficient_net.network_params import GlobalParams


class MBConvBlock(nn.Module):
    """
    Mobile Inverted Residual Bottleneck Block

    Args:
        block_args (namedtuple): BlockArgs, see above
        global_params (namedtuple): GlobalParam, see above

    Attributes:
        has_se (bool): Whether the block contains a Squeeze and Excitation layer.
    """

    def __init__(self, block_args: BlockParams, global_params: GlobalParams, image_size: int):

        super().__init__()
        self._block_args = deepcopy(block_args)
        self.has_se = (self._block_args.se_ratio is not None) and (0 < self._block_args.se_ratio <= 1)
        self.id_skip = self._block_args.id_skip  # skip connection and drop connect

        output_channels = self._block_args.input_filters * self._block_args.expand_ratio

        # Get static or dynamic convolution depending on image size
        Conv2d = get_same_padding_conv2d(image_size=image_size)

        # Expansion phase
        if self._block_args.expand_ratio != 1:
            self._expand_conv = Conv2d(in_channels=self._block_args.input_filters, out_channels=output_channels,
                                       kernel_size=1, bias=False)
            self._bn0 = nn.BatchNorm2d(num_features=output_channels, momentum=global_params.batch_norm_momentum,
                                       eps=global_params.batch_norm_epsilon)

        # Depthwise convolution phase
        self._depthwise_conv = Conv2d(in_channels=output_channels, out_channels=output_channels, groups=output_channels,
                                      # groups makes it depthwise
                                      kernel_size=self._block_args.kernel_size, stride=self._block_args.stride,
                                      bias=False)
        self._bn1 = nn.BatchNorm2d(num_features=output_channels, momentum=global_params.batch_norm_momentum,
                                   eps=global_params.batch_norm_epsilon)

        # Squeeze and Excitation layer, if desired
        if self.has_se:
            num_squeezed_channels = max(1, int(self._block_args.input_filters * self._block_args.se_ratio))
            self._se_reduce = Conv2d(in_channels=output_channels, out_channels=num_squeezed_channels, kernel_size=1)
            self._se_expand = Conv2d(in_channels=num_squeezed_channels, out_channels=output_channels, kernel_size=1)

        # Output phase
        final_oup = self._block_args.output_filters
        self._project_conv = Conv2d(in_channels=output_channels, out_channels=final_oup, kernel_size=1, bias=False)
        self._bn2 = nn.BatchNorm2d(num_features=final_oup, momentum=global_params.batch_norm_momentum,
                                   eps=global_params.batch_norm_epsilon)
        self._swish = Swish()

    def forward(self, inputs, drop_connect_rate=None):
        """
        :param inputs: input tensor
        :param drop_connect_rate: drop connect rate (float, between 0 and 1)
        :return: output of block
        """

        # Expansion and Depthwise Convolution
        x = inputs
        if self._block_args.expand_ratio != 1:
            x = self._swish(self._bn0(self._expand_conv(inputs)))
        x = self._swish(self._bn1(self._depthwise_conv(x)))

        # Squeeze and Excitation
        if self.has_se:
            x_squeezed = F.adaptive_avg_pool2d(x, 1)
            x_squeezed = self._se_expand(self._swish(self._se_reduce(x_squeezed)))
            x = torch.sigmoid(x_squeezed) * x

        x = self._bn2(self._project_conv(x))

        # Skip connection and drop connect
        input_filters, output_filters = self._block_args.input_filters, self._block_args.output_filters
        if self.id_skip and self._block_args.stride == 1 and input_filters == output_filters:
            if drop_connect_rate:
                x = self._drop_connect(x, p=drop_connect_rate)
            x = x + inputs  # skip connection
        return x

    def _drop_connect(self, inputs, p):
        """ Drop connect. """
        if not self.training: return inputs
        batch_size = inputs.shape[0]
        keep_prob = 1 - p
        random_tensor = keep_prob
        random_tensor += torch.rand([batch_size, 1, 1, 1], dtype=inputs.dtype, device=inputs.device)
        binary_tensor = torch.floor(random_tensor)
        output = inputs / keep_prob * binary_tensor
        return output

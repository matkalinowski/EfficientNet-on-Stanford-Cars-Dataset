from torch import nn

from config.structure import EfficientNetInfoContainer
from model.Swish import MemoryEfficientSwish
from model.mb_conv_block import MBConvBlock
from structure.conv_2d import get_same_padding_conv2d
from structure.efficient_net_info import EfficientNetInfo
from structure.block_decoder import BlockDecoder
from structure.block_params import round_filters
from torch.utils import model_zoo


class EfficientNet(nn.Module):

    def __init__(self, net_info: EfficientNetInfo):
        super().__init__()
        self.net_info = net_info

        global_params = net_info.network_params.global_params
        out_channels = round_filters(1280, net_info.network_params)

        self._conv_stem, self._bn0 = self.build_batch_norm_conv(in_channels=global_params.in_channels,
                                                                out_channels=round_filters(32, net_info.network_params),
                                                                kernel_size=3,
                                                                stride=2)
        self._blocks = self.build_blocks()

        self._conv_head, self._bn1 = self.build_batch_norm_conv(in_channels=self._blocks[-1]._project_conv.out_channels,
                                                                out_channels=out_channels,
                                                                kernel_size=1,
                                                                stride=1)

        self._avg_pooling = nn.AdaptiveAvgPool2d(1)
        self._dropout = nn.Dropout(global_params.dropout_rate)
        self._fc = nn.Linear(out_channels, global_params.num_classes)
        self._swish = MemoryEfficientSwish()

    def build_batch_norm_conv(self, in_channels, out_channels, kernel_size, stride):
        global_params = self.net_info.network_params.global_params
        Conv2d = get_same_padding_conv2d(image_size=global_params.image_size)

        conv = Conv2d(in_channels,
                      out_channels,
                      kernel_size=kernel_size,
                      stride=stride,
                      bias=False)
        bn = nn.BatchNorm2d(num_features=out_channels,
                            momentum=global_params.batch_norm_momentum,
                            eps=global_params.batch_norm_epsilon)
        return conv, bn

    def build_blocks(self):
        global_params = self.net_info.network_params.global_params

        blocks = nn.ModuleList([])
        for block_args in BlockDecoder.decode(self.net_info.block_args):
            block_args = block_args.round_block(network_params=self.net_info.network_params)

            # The first block needs to take care of stride and filter size increase.
            blocks.append(MBConvBlock(block_args, global_params))
            if block_args.num_repeat > 1:
                block_args = block_args.update(input_filters=block_args.output_filters, stride=1)
            for _ in range(block_args.num_repeat - 1):
                blocks.append(MBConvBlock(block_args, global_params))
        return blocks

    @classmethod
    def from_name(cls, model_name, load_weights=False, advprop=False):
        model = cls(EfficientNetInfoContainer()[model_name])

        if load_weights:
            model.load_state_dict(
                model_zoo.load_url(
                    model.net_info.get_pretrained_url(advprop)))

        return model

    def extract_features(self, inputs):
        """ Returns output of the final convolution layer """

        # Stem
        x = self._swish(self._bn0(self._conv_stem(inputs)))

        # Blocks
        for idx, block in enumerate(self._blocks):
            drop_connect_rate = self._global_params.drop_connect_rate
            if drop_connect_rate:
                drop_connect_rate *= float(idx) / len(self._blocks)
            x = block(x, drop_connect_rate=drop_connect_rate)

        # Head
        x = self._swish(self._bn1(self._conv_head(x)))

        return x

    def forward(self, inputs):
        """ Calls extract_features to extract features, applies final linear layer, and returns logits. """
        bs = inputs.size(0)
        # Convolution layers
        x = self.extract_features(inputs)

        # Pooling and final linear layer
        x = self._avg_pooling(x)
        x = x.view(bs, -1)
        x = self._dropout(x)
        x = self._fc(x)
        return x

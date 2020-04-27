from torch import nn

from config.structure import EfficientNetInfoContainer
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

        Conv2d = get_same_padding_conv2d(image_size=global_params.image_size)

        out_channels = round_filters(32, net_info.network_params)
        bn_mom = 1 - global_params.batch_norm_momentum
        bn_eps = global_params.batch_norm_epsilon

        self._conv_stem = Conv2d(global_params.in_channels, out_channels, kernel_size=3, stride=2, bias=False)
        self._bn0 = nn.BatchNorm2d(num_features=out_channels, momentum=bn_mom, eps=bn_eps)

        self._blocks = self.generate_blocks()

        # Head
        in_channels = self._blocks[-1]._project_conv.out_channels
        out_channels = round_filters(1280, net_info.network_params)
        self._conv_head = Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self._bn1 = nn.BatchNorm2d(num_features=out_channels, momentum=bn_mom, eps=bn_eps)

        # Final linear layer
        self._avg_pooling = nn.AdaptiveAvgPool2d(1)
        self._dropout = nn.Dropout(global_params.dropout_rate)
        self._fc = nn.Linear(out_channels, global_params.num_classes)

    def generate_blocks(self):
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

from torch import nn
from torch.utils import model_zoo

from models.efficient_net.Swish import Swish
from models.efficient_net.block_decoder import BlockDecoder
from models.efficient_net.block_params import round_filters
from models.efficient_net.conv_2d import get_same_padding_conv2d
from models.efficient_net.mb_conv_block import MBConvBlock
from training.cars_dataset_lightning_module import StanfordCarsDatasetLightningModule
from training.trial_info import TrialInfo


class EfficientNet(StanfordCarsDatasetLightningModule):

    def __init__(self, trial_info: TrialInfo):
        net_info = trial_info.model_info
        self.image_size = net_info.network_params.compound_scalars.resolution

        super().__init__(trial_info)

        global_params = net_info.network_params.global_params
        Conv2d = get_same_padding_conv2d(image_size=self.image_size)

        out_channels = round_filters(32, net_info.network_params)
        self._conv_stem = Conv2d(in_channels=trial_info.in_channels, kernel_size=3, stride=2,
                                 out_channels=out_channels, bias=False)
        self._bn0 = nn.BatchNorm2d(num_features=out_channels, momentum=global_params.batch_norm_momentum,
                                   eps=global_params.batch_norm_epsilon)

        self._blocks = self.build_blocks()

        out_channels = round_filters(1280, net_info.network_params)
        self._conv_head = Conv2d(in_channels=self._blocks[-1]._project_conv.out_channels, out_channels=out_channels,
                                 kernel_size=1, stride=1, bias=False)
        self._bn1 = nn.BatchNorm2d(num_features=out_channels, momentum=global_params.batch_norm_momentum,
                                   eps=global_params.batch_norm_epsilon)

        self._avg_pooling = nn.AdaptiveAvgPool2d(1)
        if trial_info.custom_dropout_rate is not None:
            self._dropout = nn.Dropout(trial_info.custom_dropout_rate)
        else:
            self._dropout = nn.Dropout(global_params.dropout_rate)
        self._classification = nn.Linear(out_channels, trial_info.num_classes)
        self._swish = Swish()

        if trial_info.load_weights:
            self.weights_update(trial_info.advprop, trial_info.freeze_pretrained_weights)

    def weights_update(self, advprop, freeze_pretrained_weights):
        model_dict = self.state_dict()

        pretrained_dict = model_zoo.load_url(self.trial_info.model_info.get_pretrained_url(advprop))
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        if freeze_pretrained_weights:
            self.freeze_weights(pretrained_dict)
        model_dict.update(pretrained_dict)
        self.load_state_dict(model_dict)

    def freeze_weights(self, pretrained_dict):
        for name, param in self.named_parameters():
            if name in pretrained_dict.keys():
                param.requires_grad = False

    def build_blocks(self):
        global_params = self.trial_info.model_info.network_params.global_params

        blocks = nn.ModuleList([])
        for block_args in BlockDecoder.decode(self.trial_info.model_info.block_args):
            block_args = block_args.round_block(network_params=self.trial_info.model_info.network_params)

            blocks.append(MBConvBlock(block_args, global_params, self.image_size))
            if block_args.num_repeat > 1:
                block_args = block_args.update_parameters(input_filters=block_args.output_filters, stride=1)
            for _ in range(block_args.num_repeat - 1):
                blocks.append(MBConvBlock(block_args, global_params, self.image_size))
        return blocks

    def extract_features(self, inputs):
        """ Returns output of the final convolution layer """

        # Stem
        x = self._swish(self._bn0(self._conv_stem(inputs)))

        # Blocks
        for idx, block in enumerate(self._blocks):
            drop_connect_rate = self.trial_info.model_info.network_params.global_params.drop_connect_rate
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
        # x = self._swish(self._fc(x))
        x = self._classification(x)
        return x

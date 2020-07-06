import pytorch_lightning as pl
import torch
from fastai.layers import LabelSmoothingCrossEntropy
from torch import nn
from torch.utils import model_zoo
from torch.utils.data import DataLoader, random_split

from config.structure import get_data_sources
from model.Swish import MemoryEfficientSwish
from model.conv_2d import get_same_padding_conv2d
from model.mb_conv_block import MBConvBlock
from structure.block_decoder import BlockDecoder
from structure.block_params import round_filters
from structure.efficient_net_info import EfficientNetInfo
from training.cars_dataset import CarsDataset
from utils.default_logging import configure_default_logging
import numpy as np

log = configure_default_logging(__name__)


class EfficientNetLightning(pl.LightningModule):

    def __init__(self, net_info: EfficientNetInfo, batch_size, load_weights=False, advprop=False):
        super().__init__()
        self.batch_size = batch_size
        self.net_info = net_info
        self.loss = LabelSmoothingCrossEntropy()

        global_params = net_info.network_params.global_params
        Conv2d = get_same_padding_conv2d(image_size=global_params.image_size)

        out_channels = round_filters(32, net_info.network_params)
        self._conv_stem = Conv2d(in_channels=global_params.in_channels, kernel_size=3, stride=2,
                                 out_channels=out_channels, bias=False)
        self._bn0 = nn.BatchNorm2d(num_features=out_channels,
                                   momentum=global_params.batch_norm_momentum,
                                   eps=global_params.batch_norm_epsilon)

        self._blocks = self.build_blocks()

        out_channels = round_filters(1280, net_info.network_params)
        self._conv_head = Conv2d(in_channels=self._blocks[-1]._project_conv.out_channels,
                                 out_channels=out_channels,
                                 kernel_size=1, stride=1, bias=False)
        self._bn1 = nn.BatchNorm2d(num_features=out_channels,
                                   momentum=global_params.batch_norm_momentum,
                                   eps=global_params.batch_norm_epsilon)

        self._avg_pooling = nn.AdaptiveAvgPool2d(1)
        self._dropout = nn.Dropout(global_params.dropout_rate)
        self._fc = nn.Linear(out_channels, global_params.num_classes)
        self._swish = MemoryEfficientSwish()

        if load_weights:
            self.load_state_dict(
                model_zoo.load_url(
                    self.net_info.get_pretrained_url(advprop)))

    def forward(self, inputs):
        """ Calls extract_features to extract features, applies final linear layer, and returns
        logits. """
        bs = inputs.size(0)
        # Convolution layers
        x = self.extract_features(inputs)

        # Pooling and final linear layer
        x = self._avg_pooling(x)
        x = x.view(bs, -1)
        x = self._dropout(x)
        x = self._fc(x)
        return x

    def build_blocks(self):
        global_params = self.net_info.network_params.global_params

        blocks = nn.ModuleList([])
        for block_args in BlockDecoder.decode(self.net_info.block_args):
            block_args = block_args.round_block(network_params=self.net_info.network_params)

            blocks.append(MBConvBlock(block_args, global_params))
            if block_args.num_repeat > 1:
                block_args = block_args.update_parameters(input_filters=block_args.output_filters,
                                                          stride=1)
            for _ in range(block_args.num_repeat - 1):
                blocks.append(MBConvBlock(block_args, global_params))
        return blocks

    def extract_features(self, inputs):
        """ Returns output of the final convolution layer """

        # Stem
        x = self._swish(self._bn0(self._conv_stem(inputs)))

        # Blocks
        for idx, block in enumerate(self._blocks):
            drop_connect_rate = self.net_info.network_params.global_params.drop_connect_rate
            if drop_connect_rate:
                drop_connect_rate *= float(idx) / len(self._blocks)
            x = block(x, drop_connect_rate=drop_connect_rate)

        # Head
        x = self._swish(self._bn1(self._conv_head(x)))

        return x

    def training_step(self, train_batch):
        x, y = train_batch
        logits = self.forward(x)
        loss = self.loss(logits, y)

        logs = {'train_loss': loss}
        return {'loss': loss, 'log': logs}

    def validation_step(self, val_batch):
        x, y = val_batch
        logits = self.forward(x)
        loss = self.loss(logits, y)
        return {'val_loss': loss}

    def prepare_data(self):
        dataset_info = get_data_sources()['stanford']
        dataset_type = 'train'
        image_size = 300,
        dataset_location = dataset_info[dataset_type]['location']

        log.info(
            f'Loading data from: {dataset_location}; image size: {image_size}')

        dataset = CarsDataset(dataset_location, dataset_info, image_size)

        split_sizes = (len(dataset) * np.array([.8, .1, .1])).astype(np.int)
        # if integer was badly rounded we need to add or substract some data
        split_sizes[-1] = split_sizes[-1] + (len(dataset) - sum(split_sizes))

        self.train, self.val, self.test = random_split(dataset, split_sizes)

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.batch_size, num_workers=8)

    def val_dataloader(self):
        return DataLoader(self.val, batch_size=self.batch_size, num_workers=8)

    # def test_dataloader(self):
    #     return DataLoader(self.val, batch_size=self.batch_size, num_workers=8)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

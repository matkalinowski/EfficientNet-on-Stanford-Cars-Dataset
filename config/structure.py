from pathlib import Path

import re
from dataclasses import dataclass
from typing import List, Tuple, Optional

from model.round import round_filters, round_repeats

project_main_dir = Path('.')
data_dir = project_main_dir / 'data'

input_data = data_dir / 'input'
output_data = data_dir / 'output'
models_location = output_data / 'models'

# train_location = output_data / 'cars_train'

project_structure = dict(
    input_data=input_data,
    # output_data=output_data,
    stanford_data_source=input_data / 'stanford',
)

data_sources = dict(
    stanford=dict(
        data_source='https://ai.stanford.edu/~jkrause/cars/car_dataset.html',
        train=dict(
            location=project_structure['stanford_data_source'] / 'cars_train',
            source='http://imagenet.stanford.edu/internal/car196/cars_train.tgz'
        ),
        test=dict(
            location=project_structure['stanford_data_source'] / 'cars_test',
            source='http://imagenet.stanford.edu/internal/car196/cars_test.tgz'
        ),
        labels=dict(
            location=project_structure['stanford_data_source'] / 'devkit' / 'labels_df.csv',
            source='https://raw.githubusercontent.com/morganmcg1/stanford-cars/master/labels_df.csv'
        ),
    ),
)


@dataclass
class BlockParams:
    kernel_size: int
    num_repeat: int
    input_filters: int
    output_filters: int
    expand_ratio: int
    id_skip: Tuple
    se_ratio: Optional[float]
    stride: List[int]

    def update_block(self, input_filters, output_filters, num_repeat, network_params):
        self.input_filters = round_filters(input_filters, network_params)
        self.output_filters = round_filters(output_filters, network_params)
        self.num_repeat = round_repeats(num_repeat, network_params.compound_scalars)
        return self

    def update(self, **kwargs):
        self.__dict__.update(kwargs)
        return self


@dataclass
class CompoundScalars:
    width_coefficient: float
    depth_coefficient: float
    resolution: int


@dataclass
class GlobalParams:
    dropout_rate: float
    num_classes: int = 1000
    image_size: int = 224

    batch_norm_momentum: float = 0.99
    batch_norm_epsilon: float = 1e-3
    drop_connect_rate: float = 0.2
    depth_divisor: int = 8
    min_depth: int = None


@dataclass
class NetworkParams:
    compound_scalars: CompoundScalars
    global_params: GlobalParams



class BlockDecoder(object):
    """ Block Decoder for readability, straight from the official TensorFlow repository """

    @staticmethod
    def _decode_block_string(block_string):
        """ Gets a block through a string notation of arguments. """
        assert isinstance(block_string, str)

        ops = block_string.split('_')
        options = {}
        for op in ops:
            splits = re.split(r'(\d.*)', op)
            if len(splits) >= 2:
                key, value = splits[:2]
                options[key] = value

        # Check stride
        assert (('s' in options and len(options['s']) == 1) or
                (len(options['s']) == 2 and options['s'][0] == options['s'][1]))

        return BlockParams(
            kernel_size=int(options['k']),
            num_repeat=int(options['r']),
            input_filters=int(options['i']),
            output_filters=int(options['o']),
            expand_ratio=int(options['e']),
            id_skip=('noskip' not in block_string),
            se_ratio=float(options['se']) if 'se' in options else None,
            stride=[int(options['s'][0])])

    @staticmethod
    def _encode_block_string(block: BlockParams):
        """Encodes a block to a string."""
        args = [
            'r%d' % block.num_repeat,
            'k%d' % block.kernel_size,
            's%d%d' % (block.stride[0], block.stride[1]),
            'e%s' % block.expand_ratio,
            'i%d' % block.input_filters,
            'o%d' % block.output_filters
        ]
        if 0 < block.se_ratio <= 1:
            args.append('se%s' % block.se_ratio)
        if block.id_skip is False:
            args.append('noskip')
        return '_'.join(args)

    @staticmethod
    def decode(string_list) -> List[BlockParams]:
        """
        Decodes a list of string notations to specify blocks inside the network.

        :param string_list: a list of strings, each string is a notation of block
        :return: a list of BlockArgs namedtuples of block args
        """
        assert isinstance(string_list, list)
        blocks_paramms = []
        for block_string in string_list:
            blocks_paramms.append(BlockDecoder._decode_block_string(block_string))
        return blocks_paramms

    @staticmethod
    def encode(blocks_args):
        """
        Encodes a list of BlockArgs to a list of strings.

        :param blocks_args: a list of BlockArgs namedtuples of block args
        :return: a list of strings, each string is a notation of block
        """
        block_strings = []
        for block in blocks_args:
            block_strings.append(BlockDecoder._encode_block_string(block))
        return block_strings


@dataclass
class EfficientNetInfo:
    name: str
    network_params: NetworkParams
    pretrained_url: str
    advprop_pretrained_src: str
    _block_args: List[str] = None
    block_params: List[BlockParams] = None

    def __post_init__(self):
        if not self._block_args:
            self._block_args = [
                'r1_k3_s11_e1_i32_o16_se0.25', 'r2_k3_s22_e6_i16_o24_se0.25',
                'r2_k5_s22_e6_i24_o40_se0.25', 'r3_k3_s22_e6_i40_o80_se0.25',
                'r3_k5_s11_e6_i80_o112_se0.25', 'r4_k5_s22_e6_i112_o192_se0.25',
                'r1_k3_s11_e6_i192_o320_se0.25',
            ]
            self.block_params = BlockDecoder.decode(self._block_args)

    def get_pretrained_url(self, advprop):
        if advprop:
            return self.advprop_pretrained_src
        else:
            return self.pretrained_url


@dataclass
class EfficientNetInfoContainer:
    """
    Get efficientnet params based on model name.
    Based on:
        https://github.com/tensorflow/tpu/blob/master/models/official/efficientnet/efficientnet_builder.py
        https://github.com/lukemelas/EfficientNet-PyTorch/blob/master/efficientnet_pytorch/utils.py
    """
    values: List[EfficientNetInfo]

    def __init__(self):
        self.values = [
            EfficientNetInfo(name='efficientnet-b0',
                             network_params=NetworkParams(CompoundScalars(1.0, 1.0, 224),
                                                          GlobalParams(dropout_rate=0.2)),
                             pretrained_url='https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/efficientnet-b0-355c32eb.pth',
                             advprop_pretrained_src='https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/adv-efficientnet-b0-b64d5a18.pth'),
            EfficientNetInfo(name='efficientnet-b1',
                             network_params=NetworkParams(CompoundScalars(1.0, 1.0, 224),
                                                          GlobalParams(dropout_rate=0.2)),
                             pretrained_url='https://publicmodels.blob.core.windows.net/container/advprop/efficientnet-b1-0f3ce85a.pth',
                             advprop_pretrained_src='https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/adv-efficientnet-b0-b64d5a18.pth'),
            EfficientNetInfo(name='efficientnet-b2',
                             network_params=NetworkParams(CompoundScalars(1.0, 1.1, 224),
                                                          GlobalParams(dropout_rate=0.2)),
                             pretrained_url='https://publicmodels.blob.core.windows.net/container/advprop/efficientnet-b2-6e9d97e5.pth',
                             advprop_pretrained_src='https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/adv-efficientnet-b1-0f3ce85a.pth'),
            EfficientNetInfo(name='efficientnet-b0',
                             network_params=NetworkParams(CompoundScalars(1.1, 1.2, 260),
                                                          GlobalParams(dropout_rate=0.3)),
                             pretrained_url='https://publicmodels.blob.core.windows.net/container/advprop/efficientnet-b0-b64d5a18.pth',
                             advprop_pretrained_src='https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/adv-efficientnet-b2-6e9d97e5.pth'),
            EfficientNetInfo(name='efficientnet-b3',
                             network_params=NetworkParams(CompoundScalars(1.2, 1.4, 300),
                                                          GlobalParams(dropout_rate=0.3)),
                             pretrained_url='https://publicmodels.blob.core.windows.net/container/advprop/efficientnet-b3-cdd7c0f4.pth',
                             advprop_pretrained_src='https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/adv-efficientnet-b3-cdd7c0f4.pth'),
            EfficientNetInfo(name='efficientnet-b4',
                             network_params=NetworkParams(CompoundScalars(1.4, 1.8, 380),
                                                          GlobalParams(dropout_rate=0.4)),
                             pretrained_url='https://publicmodels.blob.core.windows.net/container/advprop/efficientnet-b4-44fb3a87.pth',
                             advprop_pretrained_src='https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/adv-efficientnet-b4-44fb3a87.pth'),
            EfficientNetInfo(name='efficientnet-b5',
                             network_params=NetworkParams(CompoundScalars(1.6, 2.2, 456),
                                                          GlobalParams(dropout_rate=0.4)),
                             pretrained_url='https://publicmodels.blob.core.windows.net/container/advprop/efficientnet-b5-86493f6b.pth',
                             advprop_pretrained_src='https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/adv-efficientnet-b5-86493f6b.pth'),
            EfficientNetInfo(name='efficientnet-b6',
                             network_params=NetworkParams(CompoundScalars(1.8, 2.6, 528),
                                                          GlobalParams(dropout_rate=0.5)),
                             pretrained_url='https://publicmodels.blob.core.windows.net/container/advprop/efficientnet-b6-ac80338e.pth',
                             advprop_pretrained_src='https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/adv-efficientnet-b6-ac80338e.pth'),
            EfficientNetInfo(name='efficientnet-b7',
                             network_params=NetworkParams(CompoundScalars(2.0, 3.1, 600),
                                                          GlobalParams(dropout_rate=0.5)),
                             pretrained_url='https://publicmodels.blob.core.windows.net/container/advprop/efficientnet-b7-4652b6dd.pth',
                             advprop_pretrained_src='https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/adv-efficientnet-b7-4652b6dd.pth'),
            EfficientNetInfo(name='efficientnet-b8',
                             network_params=NetworkParams(CompoundScalars(2.2, 3.6, 672),
                                                          GlobalParams(dropout_rate=0.5)),
                             pretrained_url='https://publicmodels.blob.core.windows.net/container/advprop/efficientnet-b8-22a8fe65.pth',
                             advprop_pretrained_src='https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/adv-efficientnet-b8-22a8fe65.pth'),
            EfficientNetInfo(name='efficientnet-l2',
                             network_params=NetworkParams(CompoundScalars(4.3, 5.3, 800),
                                                          GlobalParams(dropout_rate=0.5)),
                             pretrained_url='', advprop_pretrained_src='')
        ]

    def __getitem__(self, net_name):
        for net in self.values:
            if net.name == net_name:
                return net
        raise KeyError(f"Do not have configured network with name {net_name}")

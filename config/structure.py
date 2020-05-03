from pathlib import Path

from dataclasses import dataclass
from typing import List

from structure.efficient_net_info import EfficientNetInfo
from structure.network_params import CompoundScalars, GlobalParams, NetworkParams

project_main_dir = Path('.')
data_dir = project_main_dir / 'data'

input_data = data_dir / 'input'
output_data = data_dir / 'output'
training_trials = output_data / 'trials'

# train_location = output_data / 'cars_train'

project_structure = dict(
    input_data=input_data,
    output_data=output_data,
    logging_dir=output_data / 'logs',
    training_trials=training_trials,
    stanford_data_source=input_data / 'stanford',
)

telegram = dict(
    TELEGRAM_TOKEN='1069361426:AAH21f3L9g1PD_CJKe3ckKcVqlC00JdAI6c',
    CHAT_ID='368109717'
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

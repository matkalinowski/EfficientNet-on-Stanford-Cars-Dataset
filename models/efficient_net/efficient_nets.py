from enum import Enum

from models.efficient_net.efficient_net_info import EfficientNetInfo
from models.efficient_net.network_params import NetworkParams, CompoundScalars, GlobalParams


class EfficientNets(Enum):
    b0 = EfficientNetInfo(name='efficientnet-b0',
                          network_params=NetworkParams(CompoundScalars(1.0, 1.0, 224),
                                                       GlobalParams(dropout_rate=0.2)),
                          pretrained_url='https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/efficientnet-b0-355c32eb.pth',
                          advprop_pretrained_src='https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/adv-efficientnet-b0-b64d5a18.pth')
    b1 = EfficientNetInfo(name='efficientnet-b1',
                          network_params=NetworkParams(CompoundScalars(1.0, 1.1, 240),
                                                       GlobalParams(dropout_rate=0.2)),
                          pretrained_url='https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/efficientnet-b1-f1951068.pth',
                          advprop_pretrained_src='https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/adv-efficientnet-b1-0f3ce85a.pth')

    b2 = EfficientNetInfo(name='efficientnet-b2',
                          network_params=NetworkParams(CompoundScalars(1.1, 1.2, 260),
                                                       GlobalParams(dropout_rate=0.3)),
                          pretrained_url='https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/efficientnet-b2-8bb594d6.pth',
                          advprop_pretrained_src='https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/adv-efficientnet-b2-6e9d97e5.pth')
    b3 = EfficientNetInfo(name='efficientnet-b3',
                          network_params=NetworkParams(CompoundScalars(1.2, 1.4, 300),
                                                       GlobalParams(dropout_rate=0.3)),
                          pretrained_url='https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/efficientnet-b3-5fb5a3c3.pth',
                          advprop_pretrained_src='https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/adv-efficientnet-b3-cdd7c0f4.pth')
    b4 = EfficientNetInfo(name='efficientnet-b4',
                          network_params=NetworkParams(CompoundScalars(1.4, 1.8, 380),
                                                       GlobalParams(dropout_rate=0.4)),
                          pretrained_url='https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/efficientnet-b4-6ed6700e.pth',
                          advprop_pretrained_src='https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/adv-efficientnet-b4-44fb3a87.pth')
    b5 = EfficientNetInfo(name='efficientnet-b5',
                          network_params=NetworkParams(CompoundScalars(1.6, 2.2, 456),
                                                       GlobalParams(dropout_rate=0.4)),
                          pretrained_url='https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/efficientnet-b5-b6417697.pth',
                          advprop_pretrained_src='https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/adv-efficientnet-b5-86493f6b.pth')
    b6 = EfficientNetInfo(name='efficientnet-b6',
                          network_params=NetworkParams(CompoundScalars(1.8, 2.6, 528),
                                                       GlobalParams(dropout_rate=0.5)),
                          pretrained_url='https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/efficientnet-b6-c76e70fd.pth',
                          advprop_pretrained_src='https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/adv-efficientnet-b6-ac80338e.pth')
    b7 = EfficientNetInfo(name='efficientnet-b7',
                          network_params=NetworkParams(CompoundScalars(2.0, 3.1, 600),
                                                       GlobalParams(dropout_rate=0.5)),
                          pretrained_url='https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/efficientnet-b7-dcc49843.pth',
                          advprop_pretrained_src='https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/adv-efficientnet-b7-4652b6dd.pth')
    b8 = EfficientNetInfo(name='efficientnet-b8',
                          network_params=NetworkParams(CompoundScalars(2.2, 3.6, 672),
                                                       GlobalParams(dropout_rate=0.5)),
                          pretrained_url='',
                          advprop_pretrained_src='https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/adv-efficientnet-b8-22a8fe65.pth')
    l2 = EfficientNetInfo(name='efficientnet-l2',
                          network_params=NetworkParams(CompoundScalars(4.3, 5.3, 800),
                                                       GlobalParams(dropout_rate=0.5)),
                          pretrained_url='', advprop_pretrained_src='')

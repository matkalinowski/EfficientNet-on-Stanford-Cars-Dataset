# MobileNetV2
mobile2_params = dict(
    out_channels=[32, 16, 24, 32, 64, 96, 160, 320],
    strides=[2, 1, 2, 2, 2, 1, 2, 1],
    n=[1, 1, 2, 3, 4, 3, 3, 1],
    expansion=[1, 1, 6, 6, 6, 6, 6, 6]
)

# MobileNetV3
mobile3_large = dict(
    out_channels=[16, 24, 24, 40, 40, 40, 80, 80, 80, 80, 112, 112, 160, 160, 160],
    expansion=[1, 4, 3, 3, 3, 3, 6, 2.5, 2.3, 2.3, 6, 6, 6, 6, 6],
    kernel_size=[3, 3, 3, 5, 5, 5, 3, 3, 3, 3, 3, 3, 5, 5, 5],
    stride=[1, 2, 1, 2, 1, 1, 2, 1, 1, 1, 1, 1, 2, 1, 1],
    squeeze_excitation=[0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1],
    hard_swish=[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1],
)

mobile3_small = dict(
    out_channels=[16, 24, 24, 40, 40, 40, 48, 48, 96, 96, 96],
    expansion=[1, 4.5, 3.67, 4, 6, 6, 3, 3, 6, 6, 6],
    kernel_size=[3, 3, 3, 5, 5, 5, 5, 5, 5, 5, 5],
    stride=[2, 2, 1, 2, 1, 1, 1, 1, 2, 1, 1],
    squeeze_excitation=[1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1],
    hard_swish=[0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1]
)
from dataclasses import dataclass


@dataclass
class CompoundScalars:
    width_coefficient: float
    depth_coefficient: float
    resolution: int


@dataclass
class GlobalParams:
    dropout_rate: float

    # https://github.com/lukemelas/EfficientNet-PyTorch/issues/3
    batch_norm_momentum: float = 0.01

    batch_norm_epsilon: float = 1e-3
    drop_connect_rate: float = 0.2
    depth_divisor: int = 8
    min_depth: int = None


@dataclass
class NetworkParams:
    compound_scalars: CompoundScalars
    global_params: GlobalParams

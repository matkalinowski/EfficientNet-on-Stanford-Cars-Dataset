import math
from dataclasses import dataclass
from typing import Tuple, Optional, List

from models.efficient_net.network_params import NetworkParams


def round_filters(filter_size, network_params: NetworkParams):
    global_params = network_params.global_params
    """ Calculate and round number of filters based on depth multiplier. """
    multiplier = network_params.compound_scalars.width_coefficient
    if not multiplier:
        return filter_size
    divisor = global_params.depth_divisor
    min_depth = global_params.min_depth
    filter_size *= multiplier
    min_depth = min_depth or divisor
    new_filters = max(min_depth, int(filter_size + divisor / 2) // divisor * divisor)
    if new_filters < 0.9 * filter_size:  # prevent rounding by more than 10%
        new_filters += divisor
    return int(new_filters)


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

    def round_block(self, network_params):
        """Update block input and output filters based on depth multiplier."""
        self.input_filters = round_filters(self.input_filters, network_params)
        self.output_filters = round_filters(self.output_filters, network_params)
        self.num_repeat = self._round_repeats(network_params.compound_scalars)
        return self

    def update_parameters(self, **kwargs):
        self.__dict__.update(kwargs)
        return self

    def _round_repeats(self, compound_scalars):
        """ Round number of filters based on depth multiplier. """
        multiplier = compound_scalars.depth_coefficient
        if not multiplier:
            return self.num_repeat
        return int(math.ceil(multiplier * self.num_repeat))
import math


def round_repeats(repeats, compound_scalars):
    """ Round number of filters based on depth multiplier. """
    multiplier = compound_scalars.depth_coefficient
    if not multiplier:
        return repeats
    return int(math.ceil(multiplier * repeats))


def round_filters(filters, network_params):
    global_params = network_params.global_params
    """ Calculate and round number of filters based on depth multiplier. """
    multiplier = network_params.compound_scalars.width_coefficient
    if not multiplier:
        return filters
    divisor = global_params.depth_divisor
    min_depth = global_params.min_depth
    filters *= multiplier
    min_depth = min_depth or divisor
    new_filters = max(min_depth, int(filters + divisor / 2) // divisor * divisor)
    if new_filters < 0.9 * filters:  # prevent rounding by more than 10%
        new_filters += divisor
    return int(new_filters)
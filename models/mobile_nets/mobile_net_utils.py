"""
This module contains few methods that will be handy in creation of MobileNet architectures.
"""


# MobileNet 1
def create_next_layer_calculator(input_n, output_n, scaling_parameter):
    """
    This closure will be used to calculate and pass channels between layers.
    During creation we need to specify what is starting input and output channel

    Scaling parameter can be used to increase number of output channels across all layers but maintaining the
    proportion between them.
    """

    assert scaling_parameter > 0, "scaling_parameter must be larger then 0"

    input_channels = input_n
    output_channels = int(output_n * scaling_parameter)

    def get_next_layer_channels(new_output_channels=None, as_is=False):
        """
        This function is created when calling create_next_layer_calculator
        It has two ways of working. If as_is is set to True it will return
        channels for layer that is currently calculated. If we want to calculate
        new ones we need to pass an int into new_output_channels
        """

        nonlocal input_channels, output_channels, scaling_parameter

        if not as_is:
            input_channels, output_channels = \
                output_channels, int(new_output_channels * scaling_parameter)

        return {'in_channels': input_channels,
                'out_channels': output_channels}

    return get_next_layer_channels


# MobileNet 2
def parameter_generator(parameters: dict):
    """
    Generator that is used to fill specific bottleneck blocks with used parameters.
    During iteration, this generator will also held and yield information about in_channels.

    :param parameters: dictionary with three keys:
        out_channels - out_channels at the end of the bottleneck,
        strides - strides used in the specific bottleneck block
        n - number of blocks used in the section. Blocks parameters vary between each other. Only last layer of this
        n-block changes the channel number on its output. Every layer in between is an identity

    """

    # It will be easier to work with those params if we transform them to list of tuples and make an
    # generator afterwards.
    parameters = list(zip(
        *[value for key, value in parameters.items()]
    ))

    in_channels = 3  # First layer in channels -> RGB

    for layer_parameter in parameters:
        out_channels, stride, n, expansion = layer_parameter
        for i in range(1, n + 1):
            yield dict(
                in_channels=in_channels,
                out_channels=out_channels if i == 1 else in_channels,
                stride=stride if i == n else 1,
                expansion=expansion)
            if i == 1:
                in_channels = out_channels  # Replacing in_channels for next iteration/layer
import pytest

from model.modules import EfficientNet


def test_init_from_name():
    model = EfficientNet.from_name('efficientnet-b0', load_weights=True)
    print(model)

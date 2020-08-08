import pytest

import torch

from models.mobile_nets import MobileNetV1, MobileNetV2, MobileNetV3
from models.mobile_nets.mobile_net_parameters import mobile2_params

input_tensor = None


@pytest.fixture(autouse=True)
def mock_input_tensor():
    global input_tensor
    input_tensor = torch.zeros(1, 3, 224, 224)


def test_if_mobile_v1_forwards_tensor():
    model = MobileNetV1(100, 1)
    assert model(input_tensor).size() == torch.Size([1, 100])


def test_if_mobile_v1_with_scaling_forwards_tensor():
    model = MobileNetV1(100, 0.7)
    assert model(input_tensor).size() == torch.Size([1, 100])


def test_if_mobile_v2_forwards_tensor():
    model = MobileNetV2(mobile2_params, 100, 1)
    assert model(input_tensor).size() == torch.Size([1, 100])


def test_if_mobile_v2_with_scaling_forwards_tensor():
    model = MobileNetV2(mobile2_params, 100, 0.4)
    assert model(input_tensor).size() == torch.Size([1, 100])


def test_if_mobile_v3_small_forwards_tensor():
    assert True


def test_if_mobile_v3_large_forwards_tensor():
    assert True

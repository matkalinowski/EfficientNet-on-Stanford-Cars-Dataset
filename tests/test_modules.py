import pytest

from model.modules import EfficientNet
from config.structure import models_location
import torch

# model_path = models_location / 'base_model.pth'

model_path = "C:\\files\code\projects\dnn-studia\project\data\output\models\\base_model.pth"
# torch.save(model.state_dict(), model_path)


def test_init_from_name():
    model = EfficientNet.from_name('efficientnet-b0', load_weights=True)
    disc_model = torch.load(model_path)

    assert len(model.state_dict()) == len(disc_model)

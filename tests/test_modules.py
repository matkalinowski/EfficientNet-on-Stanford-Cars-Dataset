from model.efficient_net import EfficientNet
import torch

# model_path = models_location / 'base_model.pth'

from pathlib import Path

model_path = Path(r"C:\files\code\projects\dnn-studia\project\data\output\models\\base_model.pth")


def test_init_from_name():
    model = EfficientNet.from_name('efficientnet-b0', load_weights=True)
    # torch.save(model.state_dict(), model_path)
    disc_model = torch.load(model_path)

    assert len(model.state_dict()) == len(disc_model)
    print(model)

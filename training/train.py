import os

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import NeptuneLogger
from torch.autograd import Variable

from models.efficient_net.efficient_net import EfficientNet
from models.efficient_net.efficient_nets import EfficientNets
from training.trial_info import TrialInfo


def perform_training(
        model_info: EfficientNets,
        load_weights=True,
        advprop=False
):
    model = EfficientNet(
        batch_size=10,
        net_info=model_info.value,
        load_weights=load_weights,
        advprop=advprop)

    neptune_logger = NeptuneLogger(
        project_name="matkalinowski/sandbox",
        experiment_name="e0"
    )

    trial_info = TrialInfo(model_info, load_weights, advprop)

    checkpoint = ModelCheckpoint(filepath=str(trial_info.output_folder), period=2, mode='min')
    trainer = pl.Trainer(max_epochs=2, gpus=1, logger=neptune_logger, checkpoint_callback=checkpoint, fast_dev_run=True)
    trainer.fit(model)
    trainer.test(model)

    torch_out = torch.onnx.export(model,
                                  Variable(torch.randn(10, 3, 300, 300)).cuda(),
                                  "EfficientNets_b0.onnx",
                                  export_params=False)

    for item in os.listdir(trial_info.output_folder):
        neptune_logger.experiment.log_artifact(os.path.join(trial_info.output_folder, item))

    neptune_logger.experiment.stop()


def main():
    perform_training(EfficientNets.b0)


if __name__ == '__main__':
    main()

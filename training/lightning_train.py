import os

import neptune
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.logging.neptune import NeptuneLogger

from model.efficient_net_lightning import EfficientNetLightning
from structure.efficient_nets import EfficientNets
from structure.trial_info import TrialInfo


def perform_training(
        model_info: EfficientNets,
        load_weights=True,
        advprop=False
):
    model = EfficientNetLightning(model_info.value,
                                  batch_size=25,
                                  load_weights=load_weights,
                                  advprop=advprop)

    neptune_logger = NeptuneLogger(
        project_name="matkalinowski/sandbox",
        experiment_name="e0"
    )

    trial_info = TrialInfo(model_info, load_weights, advprop)

    checkpoint = ModelCheckpoint(filepath=trial_info.output_folder, period=2, mode='min')
    trainer = pl.Trainer(max_epochs=3, gpus=1, logger=neptune_logger, checkpoint_callback=checkpoint,
                         # fast_dev_run=True
                         )
    trainer.fit(model)
    trainer.test(model)

    for item in os.listdir(trial_info.output_folder):
        neptune_logger.experiment.log_artifact(os.path.join(trial_info.output_folder, item))

    neptune_logger.experiment.stop()


def main():
    perform_training(EfficientNets.b0)


if __name__ == '__main__':
    main()

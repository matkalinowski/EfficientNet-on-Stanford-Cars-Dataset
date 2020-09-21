import sys
from typing import Optional, List

sys.path.append('.')

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateLogger
from pytorch_lightning.loggers import NeptuneLogger

from datasets.stanford.stanford_cars_data_module import StanfordCarsDataModule
from models.efficient_net.efficient_net import EfficientNet
from models.efficient_net.efficient_nets import EfficientNets
from training.cars_dataset_callback import StanfordCarsDatasetCallback
from training.trial_info import TrialInfo
from utils.default_logging import configure_default_logging

log = configure_default_logging(__name__)


def perform_training(
        trial_info: TrialInfo,
        training_data=None,
        model=None,
        logger_tags: Optional[List[str]] = None,

):
    if model is None:
        model = EfficientNet(trial_info=trial_info)
    if training_data is None:
        training_data = StanfordCarsDataModule(batch_size=trial_info.batch_size,
                                               in_channels=trial_info.in_channels,
                                               image_size=model.image_size)

    neptune_logger = NeptuneLogger(
        project_name="matkalinowski/sandbox",
        experiment_name=f"{str(trial_info)}",
        tags=logger_tags
    )

    early_stop_callback = pl.callbacks.early_stopping.EarlyStopping(
        min_delta=1e-3,
        patience=7
    )

    checkpoint_callback = ModelCheckpoint(filepath=str(trial_info.output_folder))

    callback = StanfordCarsDatasetCallback(trial_info)
    lrl = LearningRateLogger()

    trainer = pl.Trainer(max_epochs=trial_info.epochs,
                         gpus=1,
                         # fast_dev_run=True,
                         logger=neptune_logger,
                         callbacks=[callback, lrl],
                         checkpoint_callback=checkpoint_callback,
                         early_stop_callback=early_stop_callback
                         )
    trainer.fit(model, datamodule=training_data)


if __name__ == '__main__':
    in_channels_grid = [3, 1]
    load_weights_grid = [True, False]
    optimizer_settings_weight_decay_grid = [1e-3, 1e-2, 1e-1, 2e-1]
    custom_dropout_rate_grid = [0, 0.1, 0.2, 0.3]

    for in_channels in in_channels_grid:
        for load_weights in load_weights_grid:
            for weight_decay in optimizer_settings_weight_decay_grid:
                for dropout_rate in custom_dropout_rate_grid:
                    if (in_channels == 3 and load_weights is True
                        and weight_decay in [1e-3, 1e-2, 1e-1, 2e-1]
                        and dropout_rate in [0, 0.1, 0.2, 0.3]) or (
                            in_channels == 3 and load_weights is False and weight_decay == 0.001 and dropout_rate == 0):
                        continue
                    trial_info = TrialInfo(in_channels=in_channels,
                                           load_weights=load_weights,
                                           optimizer_settings=dict(weight_decay=weight_decay),
                                           custom_dropout_rate=dropout_rate,
                                           # remaining values stay the same:
                                           optimizer=torch.optim.AdamW,
                                           model_info=EfficientNets.b0.value,
                                           advprop=False,
                                           freeze_pretrained_weights=False,
                                           epochs=150,
                                           batch_size=96,
                                           initial_lr=1e-3,
                                           scheduler_settings=dict(patience=3),
                                           num_classes=196,
                                           )
                    try:
                        perform_training(trial_info, logger_tags=['grid_search'])
                    except Exception as e:
                        log.error('Error in trial.')
                        log.error(e)

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

):
    if model is None:
        model = EfficientNet(trial_info=trial_info)
    if training_data is None:
        training_data = StanfordCarsDataModule(batch_size=trial_info.batch_size,
                                               in_channels=trial_info.in_channels,
                                               image_size=model.image_size)

    neptune_logger = NeptuneLogger(
        project_name="matkalinowski/sandbox",
        experiment_name=f"{str(trial_info)}"
    )

    early_stop_callback = pl.callbacks.early_stopping.EarlyStopping(
        monitor='val_acc',
        min_delta=5e-3,
        patience=5,
        mode='min'
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
    perform_training(trial_info=TrialInfo(model_info=EfficientNets.b0.value,
                                          load_weights=True,
                                          advprop=False,
                                          freeze_pretrained_weights=False,
                                          epochs=100,
                                          batch_size=32,
                                          initial_lr=1e-3,
                                          optimizer=torch.optim.AdamW,
                                          optimizer_settings=dict(),
                                          scheduler_settings=dict(patience=3),
                                          custom_dropout_rate=None,
                                          num_classes=196,
                                          in_channels=3,
                                          ))

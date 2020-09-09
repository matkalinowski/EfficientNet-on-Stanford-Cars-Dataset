import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
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

):
    model = EfficientNet(trial_info=trial_info)
    if training_data is None:
        training_data = StanfordCarsDataModule(batch_size=trial_info.batch_size, image_size=model.image_size)

    neptune_logger = NeptuneLogger(
        project_name="matkalinowski/sandbox",
        experiment_name=f"{str(trial_info)}_custom_linear_unit_freezed_pretrained_params",
        tags=['test']
    )

    early_stop_callback = pl.callbacks.early_stopping.EarlyStopping(
        monitor='val_loss',
        min_delta=0.0,
        patience=10,
        mode='min'
    )

    checkpoint_callback = ModelCheckpoint(filepath=str(trial_info.output_folder))
    callback = StanfordCarsDatasetCallback(trial_info)
    trainer = pl.Trainer(max_epochs=trial_info.epochs,
                         gpus=1,
                         fast_dev_run=True,
                         logger=neptune_logger,
                         # callbacks=[callback],
                         checkpoint_callback=checkpoint_callback,
                         early_stop_callback=early_stop_callback
                         )
    trainer.fit(model, datamodule=training_data)


if __name__ == '__main__':
    perform_training(trial_info=TrialInfo(model_info=EfficientNets.b0.value,
                                          load_weights=False,
                                          advprop=False,
                                          freeze_pretrained_weights=False,
                                          epochs=20,
                                          batch_size=20,
                                          initial_lr=1e-4,
                                          num_classes=196,
                                          in_channels=3,
                                          ))

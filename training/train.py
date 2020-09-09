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
        epochs,
        batch_size,
        model_info: EfficientNets,
        training_data=None,
        load_weights=False,
        freeze_pretrained_weights=False,
        advprop=False,
):
    model = EfficientNet(
        net_info=model_info.value,
        load_weights=load_weights,
        freeze_pretrained_weights=freeze_pretrained_weights,
        advprop=advprop,
        num_classes=196,
        in_channels=3,
    )
    if training_data is None:
        training_data = StanfordCarsDataModule(batch_size=batch_size, image_size=model.image_size)

    trial_info = TrialInfo(model_info, load_weights, advprop, freeze_pretrained_weights, epochs, batch_size)
    neptune_logger = NeptuneLogger(
        project_name="matkalinowski/sandbox",
        experiment_name=f"{str(trial_info)}_custom_linear_unit_freezed_pretrained_params",
        tags=['test']
    )

    checkpoint = ModelCheckpoint(filepath=str(trial_info.output_folder))
    callback = StanfordCarsDatasetCallback(trial_info)
    trainer = pl.Trainer(max_epochs=epochs,
                         gpus=1,
                         # fast_dev_run=True,
                         logger=neptune_logger,
                         callbacks=[callback],
                         checkpoint_callback=checkpoint
                         )
    trainer.fit(model, datamodule=training_data)


if __name__ == '__main__':
    perform_training(10, 16, EfficientNets.b1)

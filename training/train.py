import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import NeptuneLogger

from models.efficient_net.efficient_net import EfficientNet
from models.efficient_net.efficient_nets import EfficientNets
from training.cars_dataset_callback import StanfordCarsDatasetCallback
from training.trial_info import TrialInfo


def perform_training(
        model_info: EfficientNets,
        load_weights=True,
        freeze_pretrained_weights=False,
        advprop=False
):
    model = EfficientNet(
        batch_size=24,
        net_info=model_info.value,
        load_weights=load_weights,
        freeze_pretrained_weights=freeze_pretrained_weights,
        advprop=advprop
    )

    trial_info = TrialInfo(model_info, load_weights, advprop, freeze_pretrained_weights)
    neptune_logger = NeptuneLogger(
        project_name="matkalinowski/sandbox",
        experiment_name=f"{str(trial_info)}_custom_linear_unit_freezed_pretrained_params",
        tags=['debug']
    )

    checkpoint = ModelCheckpoint(filepath=str(trial_info.output_folder))
    trainer = pl.Trainer(max_epochs=20,
                         gpus=1,
                         fast_dev_run=True,
                         logger=neptune_logger,
                         callbacks=[(StanfordCarsDatasetCallback(trial_info))],
                         checkpoint_callback=checkpoint
                         )
    trainer.fit(model)
    trainer.test(model)


if __name__ == '__main__':
    perform_training(EfficientNets.b0, load_weights=True, freeze_pretrained_weights=True)

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
        advprop=False
):
    model = EfficientNet(
        batch_size=20,
        net_info=model_info.value,
        load_weights=load_weights,
        advprop=advprop
    )

    trial_info = TrialInfo(model_info, load_weights, advprop)
    neptune_logger = NeptuneLogger(
        project_name="matkalinowski/sandbox",
        experiment_name=f"{str(trial_info)}"
    )

    trainer = pl.Trainer(max_epochs=20, gpus=1,
                         fast_dev_run=True,
                         logger=neptune_logger,
                         # save_last=True,
                         callbacks=[(StanfordCarsDatasetCallback(trial_info))],
                         checkpoint_callback=ModelCheckpoint(filepath=str(trial_info.output_folder), period=2,
                                                             mode='min')
                         )
    trainer.fit(model)
    trainer.test(model)


if __name__ == '__main__':
    perform_training(EfficientNets.b0, load_weights=False)

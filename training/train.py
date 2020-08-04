import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

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
        image_size=model_info.value.network_params.global_params.image_size,
        net_info=model_info.value,
        load_weights=load_weights,
        advprop=advprop)

    trial_info = TrialInfo(model_info, load_weights, advprop)

    checkpoint = ModelCheckpoint(filepath=str(trial_info.output_folder), period=2, mode='min')
    trainer = pl.Trainer(max_epochs=20, checkpoint_callback=checkpoint, fast_dev_run=True)
    trainer.fit(model)
    trainer.test(model)


if __name__ == '__main__':
    perform_training(EfficientNets.b0)

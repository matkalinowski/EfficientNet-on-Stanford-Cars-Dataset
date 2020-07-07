import pytorch_lightning as pl

from model.efficient_net_lightning import EfficientNetLightning
from structure.efficient_nets import EfficientNets


def perform_training(
        model_info: EfficientNets,
        load_weights=True,
        advprop=False
):
    model = EfficientNetLightning(model_info.value,
                                  load_weights=load_weights,
                                  advprop=advprop)

    trainer = pl.Trainer(max_epochs=2, gpus=1, auto_scale_batch_size=True)
    trainer.fit(model)


def main():
    perform_training(EfficientNets.b0)


if __name__ == '__main__':
    main()

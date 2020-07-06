import pytorch_lightning as pl

from model.efficient_net_lightning import EfficientNetLightning
from structure.efficient_nets import EfficientNets


def perform_training(
        model_info: EfficientNets,
        load_weights=True,
        advprop=False
):
    early_stop_callback = pl.callbacks.early_stopping.EarlyStopping(
        monitor='val_loss',
        min_delta=0.0,
        patience=10,
        verbose=False,
        mode='min'
    )
    model = EfficientNetLightning(model_info.value,
                                  batch_size=32,
                                  load_weights=load_weights,
                                  advprop=advprop)

    trainer = pl.Trainer(early_stop_callback=early_stop_callback,
                         max_epochs=1)
    trainer.fit(model)


def main():
    perform_training(EfficientNets.b0)


if __name__ == '__main__':
    main()

from abc import ABC

import pytorch_lightning as pl
from pytorch_lightning.metrics import functional as FM
from torch.optim.lr_scheduler import ReduceLROnPlateau

from training.label_smoothing_cross_entropy import LabelSmoothingCrossEntropy
from utils.default_logging import configure_default_logging

log = configure_default_logging(__name__)


class StanfordCarsDatasetLightningModule(pl.LightningModule, ABC):

    def __init__(self, trial_info):
        super().__init__()
        self.trial_info = trial_info
        self.loss = trial_info.loss

    def training_step(self, batch, batch_idx):
        x, y = batch
        pred = self(x)
        loss = self.loss(pred, y)

        # acc = FM.accuracy(pred, y, num_classes=self.trial_info.num_classes)
        # result = pl.TrainResult(loss)
        # self.log({'train_loss': loss, 'train_acc': acc})
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        pred = self(x)
        loss = self.loss(pred, y)
        # acc = FM.accuracy(pred, y, num_classes=self.trial_info.num_classes)
        # self.log({'val_loss': loss, 'val_acc': acc})

        return loss

    def configure_optimizers(self):
        optimizer = self.trial_info.optimizer(self.parameters(), lr=self.trial_info.initial_lr,
                                              **self.trial_info.optimizer_settings)
        # fix according to: https://github.com/PyTorchLightning/pytorch-lightning/issues/2976
        scheduler = {
            'scheduler': ReduceLROnPlateau(optimizer, verbose=True, **self.trial_info.scheduler_settings),
            'reduce_on_plateau': True,
            'monitor': 'val_loss',
            'name': 'lr'
        }
        return [optimizer], [scheduler]

from abc import ABC

import pytorch_lightning as pl
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau

from training.label_smoothing_cross_entropy import LabelSmoothingCrossEntropy
from utils.default_logging import configure_default_logging

log = configure_default_logging(__name__)


class StanfordCarsDatasetLightningModule(pl.LightningModule, ABC):

    def __init__(self, trial_info):
        super().__init__()
        self.trial_info = trial_info
        self.loss = LabelSmoothingCrossEntropy()

    def training_step(self, batch, batch_idx):
        x, y = batch
        pred = self(x)
        loss = self.loss(pred, y)

        result = pl.TrainResult(loss)
        result.log_dict({'train_loss': loss})
        return result

    def validation_step(self, batch, batch_idx):
        x, y = batch
        pred = self(x)
        loss = self.loss(pred, y)
        # pred_class = pred.max(axis=1).indices
        # acc = FM.accuracy(pred_class, y, num_classes=self.num_classes)

        result = pl.EvalResult(checkpoint_on=loss)
        result.log_dict({'val_loss': loss})
        return result

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.trial_info.initial_lr)
        # fix according to: https://github.com/PyTorchLightning/pytorch-lightning/issues/2976
        scheduler = {
            'scheduler': ReduceLROnPlateau(optimizer),
            'reduce_on_plateau': True,
            # val_checkpoint_on is val_loss passed in as checkpoint_on
            'monitor': 'val_checkpoint_on'
        }
        return [optimizer], [scheduler]

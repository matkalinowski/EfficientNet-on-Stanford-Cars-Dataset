from abc import ABC

import pytorch_lightning as pl
import torch
from sklearn.metrics import classification_report
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader

from training.label_smoothing_cross_entropy import LabelSmoothingCrossEntropy
from utils.default_logging import configure_default_logging
from utils.metrics import top_k_accuracy
from pytorch_lightning.metrics import functional as FM

log = configure_default_logging(__name__)


class StanfordCarsDatasetLightningModule(pl.LightningModule, ABC):

    def __init__(self, image_size, num_classes, in_channels):
        super().__init__()
        self.image_size = image_size
        self.num_classes = num_classes
        self.in_channels = in_channels
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

    def test_step(self, batch, batch_idx):
        result = self.validation_step(batch, batch_idx)
        result.rename_keys({'val_acc': 'test_acc', 'val_loss': 'test_loss'})
        return result

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-4)

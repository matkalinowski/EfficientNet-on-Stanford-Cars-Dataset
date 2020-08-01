import numpy as np
import pytorch_lightning as pl
import torch
from fastai.layers import LabelSmoothingCrossEntropy
from torch.utils.data import DataLoader, random_split

from config.structure import get_data_sources
from datasets.stanford.cars_dataset import StanfordCarsDataset

from utils.default_logging import configure_default_logging

log = configure_default_logging(__name__)


class CarsDatasetLightningModule(pl.LightningModule):

    def __init__(self, batch_size):
        super().__init__()
        self.batch_size = batch_size
        self.loss = LabelSmoothingCrossEntropy()

    def single_step(self, batch, batch_idx, loss_type):
        x, y = batch
        loss = self.loss(self(x), y)

        logs = {loss_type: loss}
        return {loss_type: loss, 'log': logs}

    def training_step(self, train_batch, batch_idx):
        return self.single_step(train_batch, batch_idx, 'loss')

    def validation_step(self, val_batch, batch_idx):
        return self.single_step(val_batch, batch_idx, 'val_loss')

    def test_step(self, test_batch, batch_idx):
        return self.single_step(test_batch, batch_idx, 'test_loss')

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        tensorboard_logs = {'val_loss': avg_loss}
        return {'avg_val_loss': avg_loss, 'log': tensorboard_logs}

    def test_epoch_end(self, outputs):
        avg_loss = torch.stack([x['test_loss'] for x in outputs]).mean()
        tensorboard_logs = {'test_loss': avg_loss}
        return {'avg_test_loss': avg_loss, 'log': tensorboard_logs}

    def prepare_data(self):
        dataset_info = get_data_sources()['stanford']
        dataset_type = 'train'
        image_size = 300
        dataset_location = dataset_info[dataset_type]['location']

        log.info(
            f'Loading data from: {dataset_location}; image size: {image_size}')

        dataset = StanfordCarsDataset(dataset_location, dataset_info, image_size)

        split_sizes = (len(dataset) * np.array([.8, .1, .1])).astype(np.int)
        split_sizes[-1] = split_sizes[-1] + (len(dataset) - sum(split_sizes))
        self.train_data, self.val, self.test = random_split(dataset, split_sizes.tolist())

    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test, batch_size=self.batch_size)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)

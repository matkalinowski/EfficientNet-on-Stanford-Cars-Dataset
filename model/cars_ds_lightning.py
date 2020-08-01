import numpy as np
import pytorch_lightning as pl
import torch
from fastai.layers import LabelSmoothingCrossEntropy
from torch.utils.data import DataLoader, random_split

from config.structure import get_data_sources
from training.cars_dataset import CarsDataset
from utils.default_logging import configure_default_logging

log = configure_default_logging(__name__)


class CarsDatasetLightningModule(pl.LightningModule):

    def __init__(self, batch_size):
        super().__init__()
        self.loss = LabelSmoothingCrossEntropy()
        self.batch_size = batch_size

    def training_step(self, train_batch):
        x, y = train_batch
        logits = self.forward(x)
        loss = self.loss(logits, y)

        logs = {'train_loss': loss}
        return {'loss': loss, 'log': logs}

    def validation_step(self, val_batch):
        x, y = val_batch
        logits = self.forward(x)
        loss = self.loss(logits, y)
        return {'val_loss': loss}

    def prepare_data(self):
        dataset_info = get_data_sources()['stanford']
        dataset_type = 'train'
        image_size = 300,
        dataset_location = dataset_info[dataset_type]['location']

        log.info(
            f'Loading data from: {dataset_location}; image size: {image_size}')

        dataset = CarsDataset(dataset_location, dataset_info, image_size)

        split_sizes = (len(dataset) * np.array([.8, .1, .1])).astype(np.int)
        # if integer was badly rounded we need to add or substract some data
        split_sizes[-1] = split_sizes[-1] + (len(dataset) - sum(split_sizes))

        self.train, self.val, self.test = random_split(dataset, split_sizes)

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.batch_size, num_workers=8)

    def val_dataloader(self):
        return DataLoader(self.val, batch_size=self.batch_size, num_workers=8)

    # def test_dataloader(self):
    #     return DataLoader(self.val, batch_size=self.batch_size, num_workers=8)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

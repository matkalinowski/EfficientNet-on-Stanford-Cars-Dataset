from abc import ABC

import numpy as np
import pytorch_lightning as pl
import pytorch_lightning.metrics.sklearns as plm
import torch
from sklearn.metrics import multilabel_confusion_matrix, classification_report
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader, random_split

from config.structure import get_data_sources
from datasets.stanford.cars_dataset import StanfordCarsDataset
from utils.default_logging import configure_default_logging
from utils.metrics import top_k_accuracy

log = configure_default_logging(__name__)


class StanfordCarsDatasetLightningModule(pl.LightningModule, ABC):

    def __init__(self, batch_size, image_size):
        super().__init__()
        self.batch_size = batch_size
        self.image_size = image_size
        self.loss = CrossEntropyLoss()

    def single_step(self, batch, batch_idx, loss_type):
        x, y = batch

        pred = self(x)
        loss = self.loss(pred, y)
        logs = {loss_type: loss}

        pred_class = torch.argmax(pred, dim=1)

        # logs['confusion_matrix'] = plm.ConfusionMatrix()(pred_class, y)
        # logs['top_k_acc'] = top_k_accuracy(pred, y, (1, 3, 5, 10))

        pred_class_numpy = pred_class.cpu().numpy()
        y_numpy = y.cpu().numpy()

        # logs['multi_label_confusion_matrix_results'] = multilabel_confusion_matrix(y_numpy, pred_class_numpy)

        # TODO: If you have time do it properly, using lightning
        # There is a problem with converting because classification_report outputs dictionary,
        # and lightning has problem with converting it to tensor.
        # class_report = ClassificationReport()(pred_class, y)

        # to know more: https://towardsdatascience.com/multi-class-metrics-made-simple-part-ii-the-f1-score-ebe8b2c2ca1
        classification_report_results = classification_report(pred_class_numpy, y_numpy, output_dict=True)

        # acc = accuracy(pred_class, y) also works here in case of replacement needed
        logs['overall_accuracy'] = classification_report_results['accuracy']

        for type_of_avg in ['macro avg', 'weighted avg']:
            averaged_results = classification_report_results[type_of_avg]
            for metric in averaged_results.keys():
                logs[f"{type_of_avg.replace(' ', '_')}_{metric}"] = averaged_results[metric]

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

        dataset_location = dataset_info[dataset_type]['location']

        log.info(f'Loading data from: {dataset_location}; image size: {self.image_size}')

        dataset = StanfordCarsDataset(dataset_location, dataset_info, self.image_size)

        split_sizes = (len(dataset) * np.array([.8, .1, .1])).astype(np.int)
        split_sizes[-1] = split_sizes[-1] + (len(dataset) - sum(split_sizes))
        self.train_data, self.val_data, self.test_data = random_split(dataset, split_sizes.tolist())

    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_data, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test_data, batch_size=self.batch_size)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)

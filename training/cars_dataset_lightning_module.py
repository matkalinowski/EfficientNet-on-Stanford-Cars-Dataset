from abc import ABC

import pytorch_lightning as pl
import torch
from sklearn.metrics import classification_report
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader

from utils.default_logging import configure_default_logging
from utils.metrics import top_k_accuracy

log = configure_default_logging(__name__)


class StanfordCarsDatasetLightningModule(pl.LightningModule, ABC):

    def __init__(self, image_size):
        super().__init__()
        self.image_size = image_size
        self.loss = CrossEntropyLoss()

    def single_step(self, batch, batch_idx, loss_type):
        x, y = batch

        pred = self(x)
        loss = self.loss(pred, y)
        logs = {loss_type: loss.item()}

        return {loss_type: loss, 'log': logs}

    def training_step(self, train_batch, batch_idx):
        return self.single_step(train_batch, batch_idx, 'loss')

    def validation_step(self, val_batch, batch_idx):
        return self.single_step(val_batch, batch_idx, 'val_loss')

    def test_step(self, test_batch, batch_idx):
        return self.single_step(test_batch, batch_idx, 'test_loss')

    def predict(self, data_loader):
        self.eval()

        y_true = torch.tensor([], dtype=torch.long, device=self.device)
        y_pred = torch.tensor([], device=self.device)

        with torch.no_grad():
            for data in data_loader:
                inputs = [i.to(self.device) for i in data[:-1]]
                labels = data[-1].to(self.device)

                outputs = self(*inputs)
                y_true = torch.cat((y_true, labels), 0)
                y_pred = torch.cat((y_pred, outputs), 0)

        _, y_pred_class = torch.max(y_pred, 1)

        return y_true, y_pred, y_pred_class

    def calculate_metrics(self, data, loss, prefix):
        y_true, y_pred, y_pred_class = self.predict(DataLoader(data, batch_size=150))

        metrics = top_k_accuracy(y_pred, y_true, (1, 3, 5, 10))

        pred_class_numpy = y_pred_class.cpu().numpy()
        y_numpy = y_true.cpu().numpy()

        # logs['multi_label_confusion_matrix_results'] = multilabel_confusion_matrix(y_numpy, pred_class_numpy)
        # logs['confusion_matrix'] = plm.ConfusionMatrix()(pred_class, y)

        # TODO: check this out
        # from pytorch_lightning.metrics.functional import multiclass_precision_recall_curve

        # to know more: https://towardsdatascience.com/multi-class-metrics-made-simple-part-ii-the-f1-score-ebe8b2c2ca1
        classification_report_results = classification_report(pred_class_numpy, y_numpy, output_dict=True)

        for type_of_avg in ['macro avg', 'weighted avg']:
            averaged_results = classification_report_results[type_of_avg]
            for metric in averaged_results.keys():
                metrics[f"{type_of_avg.replace(' ', '_')}_{metric}"] = averaged_results[metric]

        metrics = {f'{prefix}_{k}': v for k, v in metrics.items()}
        metrics[f'{prefix}_loss'] = loss
        return metrics

    def training_epoch_end(self, outputs):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        metrics = self.calculate_metrics(self.trainer.datamodule.train_data, avg_loss, prefix='train')

        return {'loss': avg_loss, 'log': metrics}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        metrics = self.calculate_metrics(self.trainer.datamodule.val_data, avg_loss, prefix='val')

        return {'val_loss': avg_loss, 'log': metrics}

    def test_epoch_end(self, outputs):
        avg_loss = torch.stack([x['test_loss'] for x in outputs]).mean()
        metrics = self.calculate_metrics(self.trainer.datamodule.test_data, avg_loss, prefix='test')

        return {'test_loss': avg_loss, 'log': metrics}

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)

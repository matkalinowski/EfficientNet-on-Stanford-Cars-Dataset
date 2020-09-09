from time import time

import torch
from pytorch_lightning import Callback
from sklearn.metrics import classification_report
from torch.utils.data import DataLoader

from training.trial_info import TrialInfo
from utils.default_logging import configure_default_logging
from utils.environment import is_on_colab
from utils.metrics import top_k_accuracy
from utils.misc import calculate_model_info

log = configure_default_logging(__name__)


# onnx_file_name = "EfficientNet_b0.onnx"
# torch_out = torch.onnx.export(model, example_batch_input, onnx_file_name, export_params=True)

# example_batch_input = torch.rand([1, 3, 224, 224], requires_grad=True)
# with torch.autograd.profiler.profile() as prof:
#     model(example_batch_input)
# # NOTE: some columns were removed for brevity
# print(prof.key_averages().table(sort_by="self_cpu_time_total"))

def _predict(model, data_loader):
    model.eval()

    y_true = torch.tensor([], dtype=torch.long, device=model.device)
    y_pred = torch.tensor([], device=model.device)

    with torch.no_grad():
        for data in data_loader:
            inputs = [i.to(model.device) for i in data[:-1]]
            labels = data[-1].to(model.device)

            outputs = model(*inputs)
            y_true = torch.cat((y_true, labels), 0)
            y_pred = torch.cat((y_pred, outputs), 0)

    _, y_pred_class = torch.max(y_pred, 1)

    log.debug(f'Head of predicted classes: {y_pred_class[:10]}')

    model.train()

    return y_true, y_pred, y_pred_class


def _log_metrics(metrics, trainer):
    log.info(f'{metrics}')
    if trainer.logger is not None:
        for name, metric in metrics.items():
            trainer.logger.experiment.log_metric(name, metric)


def _calculate_metrics(trainer, data, prefix):
    log.debug(f'Calculating metrics for {prefix}')
    y_true, y_pred, y_pred_class = _predict(trainer.model, DataLoader(data, batch_size=150))

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
    _log_metrics(metrics, trainer)
    return metrics


class StanfordCarsDatasetCallback(Callback):

    def __init__(self, trial_info: TrialInfo):
        self.lap_times = []
        self.trial_info = trial_info
        self.lap_start: time = time()

    def on_train_start(self, trainer, pl_module):
        """Called when the train begins."""
        log.info(f'Training started. Assigned id: {self.trial_info.trial_id}')
        self.trial_info.drop_trial_info()
        if trainer.logger is not None:
            trainer.logger.experiment.log_metric('on_colab', is_on_colab())
            trial_info = self.trial_info.get_trial_info()
            self.log_dictionary(trial_info, trainer)
            model_info = calculate_model_info(trainer.model, image_size=trainer.model.image_size)
            self.log_dictionary(model_info, trainer)

    def log_dictionary(self, dictionary, trainer):
        for k, v in dictionary.items():
            trainer.logger.experiment.log_metric(k, v)

    def on_train_end(self, trainer, pl_module):
        """Called when the train ends."""
        log.info(f'Training with id: {self.trial_info.trial_id} ended.'
                 f' Results are stored in: {self.trial_info.output_folder}')
        if trainer.logger is not None:
            log.info('Uploading model to logger.')
            trainer.logger.experiment.log_artifact(str(self.trial_info.output_folder))
            trainer.logger.experiment.stop()

    def on_epoch_start(self, trainer, pl_module):
        """Called when the epoch begins."""
        self.lap_start = time()

    def on_epoch_end(self, trainer, pl_module):
        """Called when the epoch ends."""
        self.lap_times.append(time() - self.lap_start)
        if trainer.logger is not None:
            trainer.logger.experiment.log_metric('lap_time', self.lap_times[-1])

    def on_train_epoch_end(self, trainer, pl_module):
        _calculate_metrics(trainer, trainer.datamodule.train_data, prefix='train')

    def on_validation_epoch_end(self, trainer, pl_module):
        _calculate_metrics(trainer, trainer.datamodule.val_data, prefix='val')

    # def on_test_epoch_end(self, trainer, pl_module):
    #     _calculate_metrics(trainer, trainer.datamodule.test_data, prefix='test')

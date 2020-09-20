from time import time

import torch
from neptune.exceptions import NoChannelValue
from pytorch_lightning import Callback
from sklearn.metrics import classification_report
from torch.utils.data import DataLoader

from datasets.stanford.stanford_cars_data_module import DatasetTypes, StanfordCarsDataset
from training.trial_info import TrialInfo
from utils.default_logging import configure_default_logging
from utils.environment import is_on_colab
from utils.files import save_csv
from utils.metrics import top_k_accuracy
from utils.misc import calculate_model_info

log = configure_default_logging(__name__)


def _predict(model, data_loader):
    """
    Make predictions; additional info:
    - model.eval() will notify all your layers that you are in eval mode, that way, batchnorm or dropout layers
     will work in eval mode instead of training mode.
    - torch.no_grad() impacts the autograd engine and deactivate it. It will reduce memory usage and speed up
     computations but you won’t be able to backprop (which you don’t want in an eval script).
    """
    model.eval()

    target = torch.tensor([], dtype=torch.long, device=model.device)
    pred = torch.tensor([], device=model.device)

    with torch.no_grad():
        for data in data_loader:
            inputs = [i.to(model.device) for i in data[:-1]]
            labels = data[-1].to(model.device)

            outputs = model(*inputs)
            target = torch.cat((target, labels), 0)
            pred = torch.cat((pred, outputs), 0)

    _, pred_class = torch.max(pred, 1)

    log.debug(f'Head of predicted classes: {pred_class[:10]}')

    model.train()
    return target, pred, pred_class


def _log_metrics(metrics, trainer):
    log.info(f'Metrics results:\n{metrics}')
    if trainer.logger is not None:
        for name, metric in metrics.items():
            trainer.logger.experiment.log_metric(name, metric)


def _calculate_store_metrics(trainer, dataset: StanfordCarsDataset):
    prefix = dataset.dataset_type.name

    log.debug(f'Calculating metrics on whole {prefix} set')
    target, pred, pred_class = _predict(trainer.model, DataLoader(dataset, batch_size=150))

    pred_class_numpy = pred_class.cpu().numpy()
    target_numpy = target.cpu().numpy()

    # calculating metrics
    metrics = top_k_accuracy(pred, target, (1, 3, 5, 10))
    metrics = {**metrics, **get_classification_report_results(pred_class_numpy, target_numpy)}
    metrics = add_prefix_to_dictionary_key(metrics, prefix)

    # saving results
    _log_metrics(metrics, trainer)
    save_predictions(dataset, pred_class_numpy, trainer)

    log.debug(f'Metrics calculation for {prefix} set ended')
    return metrics


def add_prefix_to_dictionary_key(dictionary, prefix):
    return {f'{prefix}_{k}': v for k, v in dictionary.items()}


def get_classification_report_results(pred_class_numpy, target_numpy):
    # to know more: https://towardsdatascience.com/multi-class-metrics-made-simple-part-ii-the-f1-score-ebe8b2c2ca1
    results = {}
    classification_report_results = classification_report(pred_class_numpy, target_numpy, output_dict=True)
    for type_of_avg in ['macro avg', 'weighted avg']:
        averaged_results = classification_report_results[type_of_avg]
        for metric in averaged_results.keys():
            results[f"{type_of_avg.replace(' ', '_')}_{metric}"] = averaged_results[metric]
    return results


def save_predictions(dataset, pred_class_numpy, trainer):
    annotations = dataset.annotations
    is_validation = dataset.dataset_type == DatasetTypes.val

    df = annotations.loc[annotations.test == is_validation, ['relative_im_path', 'class']]
    df['pred'] = pred_class_numpy

    save_csv(df, trainer.model.trial_info.output_folder,
             filename=f'{dataset.dataset_type.name}_predictions_{str(trainer.model.trial_info)}',
             compression=None)


def log_dictionary(dictionary, trainer):
    for k, v in dictionary.items():
        try:
            trainer.logger.experiment.log_metric(k, v)
        except TypeError:
            trainer.logger.experiment.log_text(k, str(v))
        except NoChannelValue:
            trainer.logger.experiment.log_text(k, 'None')


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
            log_dictionary(trial_info, trainer)
            model_info = calculate_model_info(trainer.model, image_size=trainer.model.image_size,
                                              color_channels=self.trial_info.in_channels)
            log_dictionary(model_info, trainer)

    def on_train_end(self, trainer, pl_module):
        """Called when the train ends."""
        log.info(f'Training with id: {self.trial_info.trial_id} ended.'
                 f' Results are stored in: {self.trial_info.output_folder}')
        _calculate_store_metrics(trainer, trainer.datamodule.train_data)
        _calculate_store_metrics(trainer, trainer.datamodule.val_data)
        if trainer.logger is not None:
            try:
                log.info('Uploading model to logger.')
                trainer.logger.experiment.log_artifact(str(self.trial_info.output_folder))
                trainer.logger.experiment.stop()
            except:
                log.error('Uploading model failed.')

    def on_epoch_start(self, trainer, pl_module):
        """Called when the epoch begins."""
        self.lap_start = time()

    def on_epoch_end(self, trainer, pl_module):
        """Called when the epoch ends."""
        self.lap_times.append(time() - self.lap_start)
        if trainer.logger is not None:
            trainer.logger.experiment.log_metric('lap_time', self.lap_times[-1])

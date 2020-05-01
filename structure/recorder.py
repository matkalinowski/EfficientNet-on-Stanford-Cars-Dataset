from dataclasses import dataclass

import torch
from fastai.basic_train import LearnerCallback, Learner, plt

from config.structure import project_structure
from utils.telegram import TelegramUpdater, send_learning_update
import time
import logging

log = logging.getLogger(__name__)


def save_model(model):
    # model_id = f'{model.net_info.name}-{time.strftime("%Y_%m_%d-%H_%M_%S")}'
    # model_path = project_structure['models_location'] / model_id
    # torch.save(model.state_dict(), model_path)
    # return model_path
    log.warn('here i will implement saving model when i grow up.')


@dataclass
class SimpleRecorder(LearnerCallback):
    learn: Learner
    # telegram_updater: TelegramUpdater = TelegramUpdater()

    # def __init__(self):
        # super().__init__()
        # log.info('performing init of SimpleRecorder')

    def on_train_begin(self, **kwargs) -> None:
        self.losses = []

    def on_step_end(self, iteration: int, last_loss, **kwargs):
        self.losses.append(last_loss)

    def on_epoch_end(self, last_loss, smooth_loss, **kwarg):
        log.info('Epoch ended', last_loss, smooth_loss)
        save_model(self._learn)
        # send_learning_update(self.telegram_updater, self.learn)

    def plot(self, **kwargs):
        losses = self.losses
        iterations = range(len(losses))
        fig, ax = plt.subplots(1, 1)
        ax.set_ylabel('Loss')
        ax.set_xlabel('Iteration')
        ax.plot(iterations, losses)
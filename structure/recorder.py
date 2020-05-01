from fastai.basic_train import Recorder
from utils.default_logging import configure_default_logging

log = configure_default_logging(__name__)


class CustomRecorder(Recorder):

    def on_train_begin(self, **kwargs) -> None:
        Recorder.on_train_begin(self, **kwargs)
        log.info('Train begin')
        self.losses = []

    def on_step_end(self, iteration: int, last_loss, **kwargs):
        Recorder.on_step_end(self, **kwargs)
        log.info('Step ended')
        self.losses.append(last_loss)

    def on_epoch_end(self, last_loss, smooth_loss, **kwargs):
        Recorder.on_epoch_end(self, smooth_loss=smooth_loss, **kwargs)
        log.info(f'Epoch ended, last_loss: {last_loss}, smooth_loss: {smooth_loss}')

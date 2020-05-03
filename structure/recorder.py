from time import time
from typing import List

from fastai.basic_train import Recorder, Learner, Union

from structure.train_info import TrialInfo
from utils.default_logging import configure_default_logging
from utils.files import get_file_path_with_timestamp
from utils.telegram import TelegramUpdater

log = configure_default_logging(__name__)


class CustomRecorder(Recorder):
    trial_info: TrialInfo
    telegram_updater = TelegramUpdater()
    lap_times: List = []
    lap_start: time = time()

    def __init__(self, learn: Learner, trial_info: TrialInfo, add_time: bool = True, silent: bool = False):
        self.trial_info = trial_info
        self.losses = []
        super().__init__(learn, add_time=add_time, silent=silent)

    def on_train_begin(self, **kwargs) -> None:
        Recorder.on_train_begin(self, **kwargs)
        self._log_execution(f'Training started. Assigned id: {self.trial_info.trial_id}')
        self.losses = []

    def on_step_end(self, iteration: int, last_loss, **kwargs):
        Recorder.on_step_end(self, **kwargs)
        self.losses.append(last_loss)

    def on_epoch_begin(self, **kwargs) -> None:
        Recorder.on_epoch_begin(self, **kwargs)
        self.lap_start = time()

    def on_epoch_end(self, last_loss, smooth_loss, **kwargs):
        Recorder.on_epoch_end(self, smooth_loss=smooth_loss, **kwargs)
        self.lap_times.append(time() - self.lap_start)
        self._log_execution(f"Epoch {kwargs['epoch'] + 1}/{kwargs['n_epochs']} ended. "
                            f"Train loss: {round(smooth_loss.item(), 3)} "
                            f"Valid loss: {round(self.learn.recorder.val_losses[-1], 3)} "
                            f"Accuracy: {round(self.learn.recorder.metrics[-1][0].item(), 3)} "
                            f"Took: {round(self.lap_times[-1], 2)} seconds.")

    def on_train_end(self, exception: Union[bool, Exception], **kwargs) -> None:
        Recorder.on_train_end(self, **kwargs)
        if exception:
            self._log_execution(f'Training failed. Exception: {exception}')
        else:
            self._log_execution(f'Training successful. Results are stored in: {self.trial_info.output_folder}')
            self.save_and_send_image(img=self.learn.recorder.plot_losses(return_fig=True),
                                     filename=f'{self.learn.model.net_info.name}_losses')
            self.save_and_send_image(img=self.learn.recorder.plot_metrics(return_fig=True),
                                     filename=f'{self.learn.model.net_info.name}_metrics')

    def save_and_send_image(self, img, filename):
        img_path = get_file_path_with_timestamp(directory=self.learn.path,
                                                filename=filename,
                                                extension='jpg')
        img.savefig(img_path)
        if not self.learn.silent:
            self.telegram_updater.send_photo(open(img_path, 'rb'))

    def _log_execution(self, msg):
        log.info(msg)
        if not self.learn.silent:
            self.telegram_updater.send_message(msg)

import sys
from time import time
from typing import List

from fastai.basic_train import Recorder, Learner, Union

from training.trial_info import TrialInfo
from utils.default_logging import configure_default_logging
from utils.files import get_file_path_with_timestamp
from utils.telegram import TelegramUpdater

log = configure_default_logging(__name__)


class CustomRecorder(Recorder):
    trial_info: TrialInfo
    telegram_updater = TelegramUpdater()
    lap_times: List = []
    lap_start: time = time()
    in_colab: bool = 'google.colab' in sys.modules

    def __init__(self, learn: Learner, trial_info: TrialInfo, add_time: bool = True, silent: bool = False,
                 send_images_every=10):
        super().__init__(learn, add_time=add_time, silent=silent)
        self.send_images_every = send_images_every
        self.trial_info = trial_info

    def on_train_begin(self, **kwargs) -> None:
        Recorder.on_train_begin(self, **kwargs)
        self._log_execution(f'Training started. Assigned id: {self.trial_info.trial_id}')
        self.trial_info.drop_trial_info()

    def on_epoch_begin(self, **kwargs) -> None:
        self.lap_start = time()

    def on_epoch_end(self, last_loss, smooth_loss, **kwargs):
        self.lap_times.append(time() - self.lap_start)
        self._log_execution(f"Epoch {kwargs['epoch'] + 1}/{kwargs['n_epochs']} ended. "
                            f"Train loss: {smooth_loss.item():.3f} "
                            f"Valid loss: {self.learn.recorder.val_losses[-1]:.3f} "
                            f"Accuracy: {self.learn.recorder.metrics[-1][0].item():.3f} "
                            f"Took: {self.lap_times[-1]:.3f} seconds.")
        if kwargs['epoch'] % self.send_images_every == 0:
            self.send_images()

    def on_train_end(self, exception: Union[bool, Exception], **kwargs) -> None:
        if exception:
            self._log_execution(f'Training failed. Exception: {exception}')
        else:
            self._log_execution(f'Training successful. Results are stored in: {self.trial_info.output_folder}')
            self.send_images()

    def send_images(self):
        self.save_and_send_image(img=self.learn.recorder.plot_losses(return_fig=True),
                                 filename=f'{self.learn.model.net_info.name}_losses')
        self.save_and_send_image(img=self.learn.recorder.plot_metrics(return_fig=True),
                                 filename=f'{self.learn.model.net_info.name}_metrics')

    def save_and_send_image(self, img, filename):
        title = f'{self.trial_info.trial_id}_{filename}'
        img.suptitle(title)
        img_path = get_file_path_with_timestamp(directory=self.learn.path,
                                                filename=filename,
                                                extension='jpg')
        img.savefig(img_path)
        if not self.learn.silent:
            self.telegram_updater.send_photo(open(img_path, 'rb'))

    def _log_execution(self, msg):
        prefix = self.build_prefix()
        log.info(msg)
        if not self.learn.silent:
            self.telegram_updater.send_message(f'[{prefix}] {msg}')

    def build_prefix(self):
        prefix = str(self.trial_info.trial_id).split('-')[0]
        if self.in_colab:
            prefix = 'colab-' + prefix
        else:
            prefix = 'local-' + prefix
        return prefix

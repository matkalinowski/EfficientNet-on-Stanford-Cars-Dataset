
from time import time
from pytorch_lightning import Callback
from pytorch_lightning.core.memory import ModelSummary

from training.trial_info import TrialInfo
from utils.default_logging import configure_default_logging

log = configure_default_logging(__name__)


class StanfordCarsDatasetCallback(Callback):

    def __init__(self, trial_info: TrialInfo):
        self.lap_times = []
        self.trial_info = trial_info
        self.lap_start: time = time()

    def on_train_start(self, trainer, pl_module):
        """Called when the train begins."""
        log.info(f'Training started. Assigned id: {self.trial_info.trial_id}')
        self.trial_info.drop_trial_info()

    def on_train_end(self, trainer, pl_module):
        """Called when the train ends."""
        log.info(f'Training with id: {self.trial_info.trial_id} ended.'
                 f' Results are stored in: {self.trial_info.output_folder}')

    def on_epoch_start(self, trainer, pl_module):
        """Called when the epoch begins."""
        self.lap_start = time()

    def on_epoch_end(self, trainer, pl_module):
        """Called when the epoch ends."""
        self.lap_times.append(time() - self.lap_start)
        # param_count = sum(ModelSummary(trainer.model).param_nums)



    def on_validation_end(self, trainer, pl_module):
        """Called when the validation loop ends."""
        pass

    def on_test_end(self, trainer, pl_module):
        """Called when the test ends."""
        pass

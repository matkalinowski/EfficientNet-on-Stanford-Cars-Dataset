from time import time

from pytorch_lightning import Callback

from training.trial_info import TrialInfo
from utils.default_logging import configure_default_logging
from utils.misc import calculate_model_info

log = configure_default_logging(__name__)


# onnx_file_name = "EfficientNet_b0.onnx"
# torch_out = torch.onnx.export(model, example_batch_input, onnx_file_name, export_params=True)

# example_batch_input = torch.rand([1, 3, 224, 224], requires_grad=True)
# with torch.autograd.profiler.profile() as prof:
#     model(example_batch_input)
# # NOTE: some columns were removed for brevity
# print(prof.key_averages().table(sort_by="self_cpu_time_total"))


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

        model_info = calculate_model_info(trainer.model, image_size=trainer.model.image_size)
        if trainer.logger is not None:
            for k, v in model_info.items():
                trainer.logger.experiment.log_metric(k, v)


    def on_epoch_start(self, trainer, pl_module):
        """Called when the epoch begins."""
        self.lap_start = time()

    def on_epoch_end(self, trainer, pl_module):
        """Called when the epoch ends."""
        self.lap_times.append(time() - self.lap_start)

    def on_validation_end(self, trainer, pl_module):
        """Called when the validation loop ends."""
        pass

    def on_test_end(self, trainer, pl_module):
        """Called when the test ends."""
        pass

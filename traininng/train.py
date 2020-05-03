from fastai.basic_train import Learner
from fastai.callbacks import CSVLogger, SaveModelCallback
from fastai.metrics import accuracy, LabelSmoothingCrossEntropy

from config.structure import data_sources
from model.efficient_net import EfficientNet
from structure.recorder import CustomRecorder
from structure.train_info import TrialInfo
from traininng.data import load_data
from utils.default_logging import configure_default_logging

log = configure_default_logging(__name__)


def main():
    data = load_data(dataset_info=data_sources['stanford'], batch_size=48)

    def perform_efficient_net_training(model_name, epochs=40):
        model = EfficientNet.from_name(model_name, load_weights=True)
        trial_info = TrialInfo(model_type=model.net_info.name)

        learn = Learner(data=data,
                        model=model,
                        wd=1e-3,
                        bn_wd=False,
                        true_wd=True,
                        metrics=[accuracy],
                        loss_func=LabelSmoothingCrossEntropy(),
                        callback_fns=[CSVLogger, SaveModelCallback],
                        path=trial_info.output_folder,
                        ).to_fp16()

        learn.fit(epochs=epochs, lr=15e-4, wd=1e-3,
                  callbacks=[CustomRecorder(learn, trial_info)])

        return learn

    learn = perform_efficient_net_training('efficientnet-b0', epochs=3)
    print(learn.csv_logger.read_logged_file())


if __name__ == '__main__':
    main()

from fastai.basic_train import Learner
from fastai.callbacks import CSVLogger
from fastai.metrics import accuracy, LabelSmoothingCrossEntropy
from model.efficient_net import EfficientNet

import pandas as pd
from config.structure import data_sources
from fastai.vision import (get_transforms, ImageList, ResizeMethod)

from structure.recorder import SimpleRecorder
from utils.telegram import training_updater

from utils.default_logging import configure_default_logging

log = configure_default_logging(__name__)


def load_data(dataset_location, labels, image_size=300, batch_size=48):
    log.info(f'Loading data from: {dataset_location}; image size: {image_size}; batch size: {batch_size}.')

    data = ImageList.from_df(labels[labels.is_test == 0], dataset_location,
                             cols='filename').split_by_rand_pct(.2).label_from_df(cols='class_name')

    data = (data.transform(get_transforms(),
                           size=image_size,
                           resize_method=ResizeMethod.SQUISH,
                           padding_mode='reflection')
            .databunch())
    data.batch_size = batch_size

    log.info('Data loaded.')
    return data


def main():
    log.info('Started training.')
    dataset_info = data_sources['stanford']
    dataset_location = dataset_info['train']['location']

    labels = pd.read_csv(dataset_info['labels']['location'])
    data = load_data(dataset_location, labels)

    @training_updater
    def perform_EfficientNet_training(model_name, epochs=40):
        model = EfficientNet.from_name(model_name, load_weights=True)

        learn = Learner(data=data,
                        model=model,
                        wd=1e-3,
                        bn_wd=False,
                        true_wd=True,
                        metrics=[accuracy],
                        loss_func=LabelSmoothingCrossEntropy(),
                        callback_fns=[CSVLogger, SimpleRecorder]
                        ).to_fp16()

        learn.fit(epochs=epochs, lr=15e-4, wd=1e-3)

        return learn

    learn = perform_EfficientNet_training('efficientnet-b0', epochs=1)
    learn.csv_logger.read_logged_file()
    log.info('Ended training.')


if __name__ == '__main__':
    main()

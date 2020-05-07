from typing import Tuple

import pandas as pd
from fastai.vision import ImageList, get_transforms, ResizeMethod

from utils.default_logging import configure_default_logging

log = configure_default_logging(__name__)


def load_data(dataset_info, image_size=300, batch_size=48, dataset_type='train') -> Tuple[ImageList, pd.DataFrame]:
    dataset_location = dataset_info[dataset_type]['location']

    log.info(f'Loading data from: {dataset_location}; image size: {image_size}; batch size: {batch_size}.')

    labels = pd.read_csv(dataset_info['labels']['location'])

    train_data = (ImageList.
                  from_df(labels[labels.is_test == 0], dataset_location, cols='filename').
                  split_by_rand_pct(.2).
                  label_from_df(cols='class_name'))

    train_data = (train_data.transform(get_transforms(),
                                       size=image_size,
                                       resize_method=ResizeMethod.SQUISH,
                                       padding_mode='reflection')
                  .databunch())
    train_data.batch_size = batch_size

    log.info('Data loaded.')
    return train_data, labels

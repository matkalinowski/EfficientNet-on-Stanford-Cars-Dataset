import os

import numpy as np
import pandas as pd
import torch
from PIL import Image
from pytorch_lightning import LightningDataModule
from sklearn import preprocessing
from torch.utils.data import random_split, DataLoader, Dataset
from torchvision import transforms

from config.structure import get_data_sources
from utils.default_logging import configure_default_logging

log = configure_default_logging(__name__)


def _fit_label_encoder(labels):
    le = preprocessing.LabelEncoder()
    labels = le.fit_transform(labels.class_name)
    return le, torch.as_tensor(labels, dtype=torch.long)


class StanfordCarsDataset(Dataset):
    def __init__(self, data_directory, labels, image_size):
        self.image_dir = data_directory
        self.image_file_names = os.listdir(data_directory)
        self.image_size = image_size
        self.label_encoder, self.labels = _fit_label_encoder(labels)
        self.data = self.read_all_images()

    def transform(self, image):
        transform_ops = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor()
        ])
        return transform_ops(image)

    def __len__(self):
        return len(self.image_file_names)

    def load_transform(self, image_name):
        image_fp = os.path.join(self.image_dir, image_name)
        image = Image.open(image_fp).convert('RGB')
        image = self.transform(image)
        return image

    def __getitem__(self, index):
        return self.data[index], self.labels[index]

    def read_all_images(self):
        return [self.load_transform(file_name) for file_name in self.image_file_names]


class StanfordCarsDataModule(LightningDataModule):

    def __init__(self, image_size, batch_size, dataset_type='train'):
        super().__init__()
        self.dataset_info = get_data_sources()['stanford']

        self.image_size = image_size
        self.batch_size = batch_size
        self.dataset_type = dataset_type

    def setup(self, stage='fit'):
        dataset_info = get_data_sources()['stanford']
        labels = pd.read_csv(self.dataset_info['labels']['location'])

        if stage == 'fit':
            log.info(f"Loading train data from: {dataset_info['train']['location']}; image size: {self.image_size}")
            dataset = StanfordCarsDataset(dataset_info['train']['location'], labels, self.image_size)

            split_sizes = (len(dataset) * np.array([.9, .1])).astype(np.int)
            split_sizes[-1] = split_sizes[-1] + (len(dataset) - sum(split_sizes))
            self.train_data, self.val_data = random_split(dataset, split_sizes.tolist())

        if stage == 'test':
            log.info(f"Loading test data from: {dataset_info['test']['location']}")
            self.test_data = StanfordCarsDataset(dataset_info['test']['location'], labels, self.image_size)

    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=self.batch_size, shuffle=True, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val_data, batch_size=self.batch_size, pin_memory=True)

    def test_dataloader(self):
        return DataLoader(self.test_data, batch_size=self.batch_size, pin_memory=True)

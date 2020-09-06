import os
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from pytorch_lightning import LightningDataModule
from sklearn import preprocessing
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from config.structure import get_data_sources
from utils.default_logging import configure_default_logging

log = configure_default_logging(__name__)


def _fit_label_encoder(annotations):
    le = preprocessing.LabelEncoder()
    annotations = le.fit_transform(annotations.class_name)
    return le, torch.as_tensor(annotations, dtype=torch.long)


class StanfordCarsDataset(Dataset):
    def __init__(self, data_directory, annotations, image_size):
        self.data_directory = data_directory
        self.image_file_names = os.listdir(data_directory)
        self.image_size = image_size
        self.label_encoder, self.labels = _fit_label_encoder(annotations)

    def transform(self, image):
        transform_ops = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor()
        ])
        return transform_ops(image)

    def load_transform(self, image_file_name):
        image_fp = os.path.join(self.data_directory, image_file_name)
        image = Image.open(image_fp).convert('RGB')
        image = self.transform(image)
        return image

    def __len__(self):
        return len(self.image_file_names)


class StanfordCarsInMemory(StanfordCarsDataset):
    def __init__(self, data_directory, annotations, image_size):
        super().__init__(data_directory, annotations, image_size)
        self.data = self.read_all_images()

    def __getitem__(self, index):
        return self.data[index], self.labels[index]

    def read_all_images(self):
        return [self.load_transform(file_name) for file_name in self.image_file_names]


class StanfordCarsOutOfMemory(StanfordCarsDataset):
    def __init__(self, data_directory, annotations, image_size):
        super().__init__(data_directory, annotations, image_size)

    def __getitem__(self, index):
        image = self.load_transform(image_file_name=self.image_file_names[index])
        return image, self.labels[index]


class StanfordCarsDataModule(LightningDataModule):

    def __init__(self, image_size, batch_size, dataset_type='train', root_path=Path('.')):
        super().__init__()
        self.dataset_info = get_data_sources(root_path)['stanford']

        self.image_size = image_size
        self.batch_size = batch_size
        self.dataset_type = dataset_type

    def setup(self, stage='fit'):
        annotations = pd.read_csv(self.dataset_info['labels']['location'])

        if stage == 'fit':
            log.info(
                f"Loading train data from: {self.dataset_info['train']['location']}; image size: {self.image_size}")
            self.train_data = StanfordCarsInMemory(self.dataset_info['train']['location'], annotations, self.image_size)
            self.val_data = StanfordCarsOutOfMemory(self.dataset_info['test']['location'], annotations, self.image_size)

    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=self.batch_size, shuffle=True, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val_data, batch_size=self.batch_size, pin_memory=True)

    # def test_dataloader(self):
    #     return DataLoader(self.test_data, batch_size=self.batch_size, pin_memory=True)

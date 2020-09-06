import os
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from config.structure import get_data_sources
from utils.default_logging import configure_default_logging

log = configure_default_logging(__name__)


class StanfordCarsDataset(Dataset):
    def __init__(self, data_directory, annotations, image_size, is_test):
        self.data_directory = data_directory
        self.annotations = annotations
        self.image_size = image_size

        self.image_file_names = annotations[annotations.test == is_test].relative_im_path

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
    def __init__(self, data_directory, annotations, image_size, is_test):
        super().__init__(data_directory, annotations, image_size, is_test)
        self.data = self.read_all_images()
        self.labels = self.read_all_labels()

    def __getitem__(self, index):
        return self.data[index], self.labels[index]

    def read_all_images(self):
        return [self.load_transform(file_name) for file_name in self.image_file_names]

    def read_all_labels(self):
        return torch.as_tensor([self.annotations[self.annotations['relative_im_path'] == file_name]['class'].values[0]
                                for file_name in self.image_file_names])


class StanfordCarsOutOfMemory(StanfordCarsDataset):
    def __init__(self, data_directory, annotations, image_size, is_test):
        super().__init__(data_directory, annotations, image_size, is_test)

    def __getitem__(self, index):
        file_name = self.image_file_names.iloc[index]
        image = self.load_transform(image_file_name=file_name)
        return image, torch.as_tensor(
            self.annotations[self.annotations['relative_im_path'] == file_name]['class'].values[0])


class StanfordCarsDataModule(LightningDataModule):

    def __init__(self, image_size, batch_size, dataset_type='train', root_path=Path('.')):
        super().__init__()
        self.dataset_info = get_data_sources(root_path)['stanford']
        self.annotations = pd.read_csv(self.dataset_info['annotations']['csv_file_path'])
        # self.class_names = pd.read_csv(self.dataset_info['class_names']['csv_file_path']).class_names

        self.image_size = image_size
        self.batch_size = batch_size
        self.dataset_type = dataset_type

    def setup(self, stage=None):
        log.info(
            f"Loading train data from: {self.dataset_info['data_dir']}; image size: {self.image_size}")
        self.train_data = StanfordCarsInMemory(self.dataset_info['data_dir'], self.annotations,
                                               self.image_size, is_test=0)
        self.val_data = StanfordCarsInMemory(self.dataset_info['data_dir'], self.annotations,
                                             self.image_size, is_test=1)

    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=self.batch_size, shuffle=True, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val_data, batch_size=self.batch_size, pin_memory=True)

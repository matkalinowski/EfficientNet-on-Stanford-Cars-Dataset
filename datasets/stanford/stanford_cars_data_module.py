import os
from enum import auto, Enum
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


class DatasetTypes(Enum):
    train = auto()
    val = auto()


class StanfordCarsDataset(Dataset):
    def __init__(self, data_directory, annotations, image_size, dataset_type: DatasetTypes, in_channels):
        self.data_directory = data_directory
        self.annotations = annotations
        self.image_size = image_size
        self.dataset_type = dataset_type
        self.greyscale_conversion = in_channels == 1

        is_test = int(dataset_type != DatasetTypes.train)
        self.image_file_names = annotations[annotations.test == is_test].relative_im_path

    def transform(self, image):
        if self.dataset_type is DatasetTypes.train:
            transform_ops = [
                transforms.Resize((self.image_size, self.image_size)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomAffine(25, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=8),
                transforms.ToTensor(),
                transforms.RandomErasing(p=0.5, scale=(0.02, 0.25)),
            ]
        else:
            transform_ops = [
                transforms.Resize((self.image_size, self.image_size)),
                transforms.ToTensor(),
            ]
        # numbers calculated in notebooks/normalization_calculation.ipynb
        if self.greyscale_conversion:
            transform_ops = [transforms.Grayscale(), *transform_ops, transforms.Normalize(
                mean=[0.462],
                std=[0.270]
            )]
        else:
            transform_ops = [*transform_ops, transforms.Normalize(
                mean=[0.470, 0.460, 0.455],
                std=[0.267, 0.266, 0.270]
            )]

        return transforms.Compose(transform_ops)(image)

    def load_transform(self, image_file_name):
        image_fp = os.path.join(self.data_directory, image_file_name)
        image = Image.open(image_fp).convert('RGB')
        image = self.transform(image)
        return image

    def __len__(self):
        return len(self.image_file_names)

    def __getitem__(self, index):
        file_name = self.image_file_names.iloc[index]
        image = self.load_transform(image_file_name=file_name)
        return image, torch.as_tensor(
            self.annotations[self.annotations['relative_im_path'] == file_name]['class'].values[0], dtype=torch.long)


class StanfordCarsDataModule(LightningDataModule):

    def __init__(self, image_size, batch_size, in_channels, root_path=Path('.')):
        super().__init__()
        self.dataset_info = get_data_sources(root_path)['stanford']
        self.annotations = pd.read_csv(self.dataset_info['annotations']['csv_file_path'])
        self.in_channels = in_channels
        # self.class_names = pd.read_csv(self.dataset_info['class_names']['csv_file_path']).class_names

        self.image_size = image_size
        self.batch_size = batch_size

    def setup(self, stage=None):
        log.info(
            f"Loading train data from: {self.dataset_info['data_dir']}; image size: {self.image_size}")
        self.train_data = StanfordCarsDataset(self.dataset_info['data_dir'], self.annotations, self.image_size,
                                              DatasetTypes.train, self.in_channels)
        self.val_data = StanfordCarsDataset(self.dataset_info['data_dir'], self.annotations, self.image_size,
                                            DatasetTypes.val, self.in_channels)

    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=self.batch_size, shuffle=True, pin_memory=True, num_workers=4)

    def val_dataloader(self):
        return DataLoader(self.val_data, batch_size=self.batch_size, pin_memory=True, num_workers=4)

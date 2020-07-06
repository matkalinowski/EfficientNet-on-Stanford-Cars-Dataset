import os

import numpy as np
import pandas as pd
import torch
from PIL import Image
from fastai.vision import get_transforms
from torch.utils.data import Dataset
from torchvision import transforms
from sklearn import preprocessing


class CarsDataset(Dataset):
    def __init__(self, dataset_location, dataset_info, image_size):
        self.image_dir = dataset_location
        self.image_fns = os.listdir(dataset_location)
        self.image_size = image_size
        self.label_encoder, self.labels = self._fit_label_encoder(pd.read_csv(dataset_info['labels']['location']))

    def _fit_label_encoder(self, labels):
        le = preprocessing.LabelEncoder()
        labels = le.fit_transform(labels.class_name)
        return le, torch.as_tensor(labels, dtype=torch.long)

    def transform(self, image):
        transform_ops = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor()
        ])
        return transform_ops(image)

    def __len__(self):
        return len(self.image_fns)

    def __getitem__(self, index):
        image_fn = self.image_fns[index]
        image_fp = os.path.join(self.image_dir, image_fn)
        image = Image.open(image_fp).convert('RGB')
        image = self.transform(image)

        return image, self.labels[index]

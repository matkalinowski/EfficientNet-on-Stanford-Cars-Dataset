import os

import pandas as pd
import torch
from PIL import Image
from sklearn import preprocessing
from torch.utils.data import Dataset
from torchvision import transforms


def _fit_label_encoder(labels):
    le = preprocessing.LabelEncoder()
    labels = le.fit_transform(labels.class_name)
    return le, torch.as_tensor(labels, dtype=torch.long)


class StanfordCarsDataset(Dataset):
    def __init__(self, dataset_location, dataset_info, image_size):
        self.image_dir = dataset_location
        self.image_file_names = os.listdir(dataset_location)
        self.image_size = image_size
        self.label_encoder, self.labels = _fit_label_encoder(pd.read_csv(dataset_info['labels']['location']))
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

import os

import numpy as np
import pandas as pd
import torch
from PIL import Image
from fastai.vision import get_transforms
from torch.utils.data import Dataset
from torchvision import transforms


class CarsDataset(Dataset):
    def __init__(self, dataset_location, dataset_info, image_size):
        self.image_dir = dataset_location
        self.image_fns = os.listdir(dataset_location)
        self.image_size = image_size
        self.labels = pd.read_csv(dataset_info['labels']['location'])

    def transform(self, image):
        transform_ops = transforms.Compose([
            # get_transforms(),
            transforms.Resize((self.image_size,self.image_size))
        ])
        return transform_ops(image)

    def __len__(self):
        return len(self.image_fns)

    def __getitem__(self, index):
        image_fn = self.image_fns[index]
        image_fp = os.path.join(self.image_dir, image_fn)
        image = Image.open(image_fp).convert('RGB')
        # image = self.transform(np.array(image))

        label = self.labels.loc[index].class_name

        return image, label
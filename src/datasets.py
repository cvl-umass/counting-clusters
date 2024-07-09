import os
import csv
import numpy as np
from scipy.io import loadmat
from os.path import join
from skimage import io
from skimage.transform import resize

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

import torch
from torch.utils.data import Dataset

from utils_ import load_dataset

import matplotlib.pyplot as plt

from PIL import Image
import torchvision.transforms as T

class Datasets(Dataset):
    """Supports datasets in utils > load_dataset()."""

    def __init__(self,
                 root_dir,
                 dataset, # dataset name (['CTai', 'MacaqueFaces', etc.])
                 classes=100): # number of categories (individuals/identities) in the dataset

        files, labels = load_dataset(dataset, root_dir, classes=classes)
        print('number of classes: %d,'%classes)

        # Sort dataset by class
        sorted_indices = np.argsort(labels)
        self.files = [files[i] for i in sorted_indices]
        self.labels = [labels[i] for i in sorted_indices]

        self.root_dir = root_dir
        self.transforms = T.Compose([T.Resize((384, 384)),
                          T.ToTensor(),
                          T.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5])])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = join(self.root_dir, self.files[idx])

        label = [self.labels[idx]]
        image = Image.open(img_name).convert('RGB')

        image = self.transforms(image) 

        return {'image': torch.FloatTensor(image),
                'label': self.labels[idx],
                'filename': self.files[idx]}


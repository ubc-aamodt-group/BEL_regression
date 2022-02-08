import logging
import os
import pickle
from collections import defaultdict
from collections import OrderedDict

import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from torchvision import transforms
from PIL import Image

import json_tricks as json
import numpy as np

class AFADDataset(Dataset):
    def __init__(self, csv_path, img_dir, transform=None, output_transform=None):
        df = pd.read_csv(csv_path, index_col=0)
        self.img_dir = img_dir
        self.csv_path = csv_path
        self.img_paths = df['path']
        self.y = df['age'].values
        self.transform = transform
        self.output_transform = output_transform
    
    def __getitem__(self, index):
        img = Image.open(os.path.join(self.img_dir,
                                      self.img_paths[index]))

        if self.transform is not None:
            img = self.transform(img)

        label = self.y[index] # minimum age is 15
        # levels = [1]*label + [0]*(NUM_CLASSES - 1 - label)
        if self.output_transform is not None:
            levels = self.output_transform(label)
        else:
            levels = torch.tensor([label], dtype=torch.float32)

        levels = torch.tensor(levels, dtype=torch.float32)
        
        return img, label, levels
    
    def __len__(self):
        return self.y.shape[0]


class AgeMSELoss(nn.Module):
    def __init__(self):
        super(AgeMSELoss, self).__init__()
        self.criterion = nn.MSELoss(size_average=True)

    def forward(self, output, target):
        loss = self.criterion(output, target)
        return loss

class AgeL1Loss(nn.Module):
    def __init__(self):
        super(AgeL1Loss, self).__init__()
        self.criterion = nn.L1Loss(size_average=True)

    def forward(self, output, target):
        loss = self.criterion(output, target)
        return loss

class AgeBCELoss(nn.Module):
    def __init__(self):
        super(AgeBCELoss, self).__init__()
        self.criterion = nn.BCEWithLogitsLoss(size_average=True, reduction='sum')

    def forward(self, output, target):
        loss = self.criterion(output, target)
        return loss

class AgeCELoss(nn.Module):
    def __init__(self):
        super(AgeCELoss, self).__init__()
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, output, target):
        loss = self.criterion(output, target)
        return loss

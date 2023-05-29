import datetime
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Union
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from torchvision import utils as vutils
from torchvision.transforms import InterpolationMode


bonusList = [10,11,13,14,15,16,17,23,26,27,29]

class EMnistDataset(Dataset):
    def __init__(self, datapath, transform=None, bonus = False):
        self.datapath = datapath
        self.transform = transform

        if bonus:
            allLabels = np.load(datapath['label'])
            allImages = np.load(datapath['img'])
            getID = []
            for i in bonusList:
                found = np.where(allLabels == i)
                getID.extend(found[0])
            self.labels = allLabels[getID]
            self.imgs = allImages[getID]
            
        else:
            self.imgs = np.load(datapath['img'])
            self.labels = np.load(datapath['label'])

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):

        image = self.imgs[idx]
        if self.transform:
            image = self.transform(image)
        label = self.labels[idx]
        return image, label
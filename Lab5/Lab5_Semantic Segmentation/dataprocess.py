import datetime
import os
import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import Dataset
from torchvision import utils as vutils


class CCAgT_Dataset(Dataset):
    def __init__(self, root_dir, transforms=None):
        self.dir = root_dir
        self.transform = transforms
        self.img_folder = os.path.join(self.dir, 'images')
        self.mask_folder = os.path.join(self.dir, 'masks')

    def __len__(self):
        return len(os.listdir(self.img_folder))

    def __getitem__(self, idx):
        img = os.listdir(self.img_folder)
        mask = os.listdir(self.mask_folder)
        img.sort()
        mask.sort()

        img = cv2.imread(os.path.join(self.img_folder, img[idx]))
        mask = cv2.imread(os.path.join(self.mask_folder,
                          mask[idx]), cv2.COLOR_BGR2GRAY)

        toTensor = transforms.Compose([
                                        transforms.ToTensor(),
                                        transforms.Normalize(0.5, 0.5)
                                        ])
        img = toTensor(img)
        mask = torch.from_numpy(mask).unsqueeze(dim=0).clamp(min=0, max=7)

        cat_image = torch.cat((img, mask), dim=0)
        if self.transform:
            cat_image = self.transform(cat_image)

        img, mask = torch.split(cat_image, 3, dim=0)

        return img, mask


def postprocessing(batch, IMG_SIZE, device):

    threshold = 0.45
    classes_num = 8

    result = []
    for img in batch:  # img c*h*w

        one_hot_img = torch.zeros(1, IMG_SIZE, IMG_SIZE).to(device)
        for cls in range(1, classes_num):
            one_hot_img = torch.where(
                img[cls] > threshold, cls, one_hot_img).to(device)

        result.append(one_hot_img.unsqueeze(0))

    return torch.cat(result, dim=0)

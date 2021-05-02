import torch
import torch.nn as nn
import torchvision
from torchvision import datasets, transforms
from torchvision.utils import save_image
import numpy as np
from pathlib import Path
from tqdm import tqdm
import os
import shutil

data_path = 'data/tiny-imagenet-200/train'
data_dir = Path(data_path)

data_transforms = transforms.Compose([
        transforms.ToTensor(),
        # transforms.Normalize((0, 0, 0), (1, 1, 1)),
    ])
train_set = torchvision.datasets.ImageFolder(data_dir, data_transforms)
train_loader = torch.utils.data.DataLoader(train_set, batch_size=1000, shuffle=False, pin_memory=True)
train_loader = iter(train_loader)

total_mean = torch.zeros(3)
total_var = torch.zeros(3)
for i in range(100):
    print(i)
    item = next(train_loader)
    total_mean += torch.mean(item[0], dim=(0, 2, 3))
    total_var += torch.var(item[0], dim=(0, 2, 3))

print(total_mean / 100)
print(total_var / 100)

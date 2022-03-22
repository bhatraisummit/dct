#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 20 13:51:56 2020

@author: kbiren
"""

import os
import random
import argparse
import numpy as np
from tensorboardX import SummaryWriter
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torchvision
import torchvision.utils as utils
import torchvision.transforms as transforms
from modelBefore import AttnVGG_before
from modelAfter import AttnVGG_after
from utilities import *
from torch.utils.data.sampler import SubsetRandomSampler


class NWPUDataset(torch.utils.data.Dataset):
    
    def __init__(self, data_path='../NWPU-RESISC45/', transform=None):
        self.dataset = torchvision.datasets.ImageFolder(root=data_path)
        self.im_size = im_size
        self.transform = transform
        
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        if self.transform:
            image = self.transform(image)
        return image, label



im_size = 128
train_split = 0.8
num_classes = 45
num_images_per_class = 700

transform_train = transforms.Compose([
    transforms.Resize((im_size,im_size)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

transform_test = transforms.Compose([
    transforms.Resize((im_size,im_size)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

train_data = NWPUDataset(data_path='../NWPU-RESISC45/', transform=transform_train)
test_data = NWPUDataset(data_path='../NWPU-RESISC45/', transform=transform_test)

split = int(np.floor(train_split * num_images_per_class))
indices = np.arange(num_classes*num_images_per_class).reshape(-1, num_images_per_class)
train_indices, test_indices = indices[:,:split].flatten(), indices[:,split:].flatten()


train_loader = torch.utils.data.DataLoader(train_data, batch_size=128, shuffle=False, num_workers=0, sampler=SubsetRandomSampler(train_indices))
test_loader = torch.utils.data.DataLoader(test_data, batch_size=100, shuffle=False, num_workers=0, sampler=SubsetRandomSampler(test_indices))

import matplotlib.pyplot as plt
img_cnt = 0
label_list = []
for i, data in enumerate(train_loader, 0):
    # if i == 2:
    #     break
    inputs, labels = data
    img_cnt += inputs.shape[0]
    label_list += list(labels.numpy())
    # print(inputs.shape)
    # plt.imshow(np.transpose(inputs[0].numpy(), (1,2,0)))
    # plt.show()
 
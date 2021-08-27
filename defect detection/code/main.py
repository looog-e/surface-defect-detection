# -*- coding: utf-8 -*-
"""
Created on Tue Aug 17 10:18:49 2021

@author: 零
"""

import cv2
import os 
import torch
import json
import copy
import random
import numpy as np
import pandas as pd
from PIL import Image
import torch.nn as nn
import torch.functional as F
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader, random_split, Subset, ConcatDataset
import matplotlib.pyplot as plt
from collections import OrderedDict
from dataset import dataset_defectDetection
from LBG import LBG_layer
from CBG import CBG_layer
from DBG import DBG_layer
from Scope import Scope_layer
from Residual import Res_layer
from SegNet import SegNet
from Evaluator import evaluator
from modelTrain import model_train

torch.manual_seed(7)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

path = r"C:\Users\风\Desktop\表面缺陷检测\BSData-main\BSData-main"
state = ["train", "eval"]
input_size = (256, 512)
batch_size = 2
shuffle = True
num_workers = 0
pin_memory = True
drop_last = True

lr = 1e-3
beta1 = 0.9
beta2 = 0.999
epsilon = 1e-8
weight_decay = 0.
amsgrad = False
num_epoch = 50

transform = transforms.Compose([
        transforms.Resize(input_size),
        transforms.ToTensor()
])

image_datasets = dataset_defectDetection(path, transform=transform)
torch.manual_seed(7)
image_datasets_split = random_split(image_datasets, [len(image_datasets)//5, len(image_datasets)-len(image_datasets)//5])

loader = {
    "train": DataLoader(image_datasets_split[1], batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=pin_memory, drop_last=drop_last),
    "eval": DataLoader(image_datasets_split[0], batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=pin_memory, drop_last=drop_last), 
}

model = SegNet(input_size, device).to(device)
optimizer = torch.optim.Adam(filter(lambda param: param.requires_grad, model.parameters()), lr=lr, betas=[beta1, beta2], eps=epsilon, weight_decay=weight_decay, amsgrad=amsgrad)
model = model_train(model, loader, evaluator, optimizer, num_epoch, state, device)
model, Acc = model.train()

plt.title("Validation Accuracy vs Number of Training Epochs")
plt.xlabel("Training Epochs")
plt.ylabel("Validation Accuracy")
plt.plot(range(1, len(Acc[0])+1), Acc[0])
plt.plot(range(1, len(Acc[1])+1), Acc[1])
plt.ylim((0, 1))
plt.xticks(np.arange(1, len(Acc[0])+1, 5))
plt.legend()
plt.show()
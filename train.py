from pickletools import optimize
import torch, torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from model import *
from dataset import *


## filter labels
filtered_labels = filter_labels("det_train_shortened.json")

## data loader
transformed_train_data = DetectionDataset(
    label_dict=filtered_labels,
    root_dir='images/',
    transform=transforms.Compose([
        transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.255]),
        Rescale(256),
        ToTensor()
    ])
)

train_loader = DataLoader(
    transformed_train_data,
    batch_size=2,
    shuffle=True,
    num_workers=0
)

net = Net(cfgfile="cfg/model.cfg")
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)


## training loop
for epoch in range(1):
    
    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        inputs, labels = data

        # define optimiser

        # forward + backward + optimise
        outputs = net()
        loss = None
        loss.backward()
        optimizer.step

        # print basic stats
        running_loss += loss.item()
        if i % 2000 == 1999: # print every 2000 mini batch
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
            running_loss = 0.0

print("Training finished.")
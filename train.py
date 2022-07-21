from pickletools import optimize
import torch
import torchvision.transforms as transforms
import numpy as np
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from model import *
from dataset import DetectionDataset, Pad, ToTensor, Normalise
from loss import Yolo_Loss


## Load data
label_dict = "D:/Datasets/bdd100k/labels/det_20/det_val.json"   # labels json 
root_dir = "D:/Datasets/bdd100k/images/100k/val"                # images file

# batch size
bs = 5

# set rgb mean and std for normalise
rgb_mean = [92.11938007161459, 102.83839236762152, 104.90335580512152]
rgb_std = [66.09941202519124, 70.6808655565459, 75.05305001603533]

## Load custom dataset + transforms
transformed_train_data = DetectionDataset(
    label_dict=label_dict,                      # labels corresponding to images
    root_dir=root_dir,                          # images root dir
    classes_file="data/bdd100k.names",          # class names
    grid_sizes=[13, 26, 52],                    # grid sizes for detection
    anchors = np.array([                        # anchor box sizes per grid size
            [[116,90], [156,198], [373,326]],   
            [[30, 61], [62, 45], [59,119]],
            [[10, 13], [16, 30], [33, 23]],
        ]),
    transform=transforms.Compose([              # transforms
        Normalise(                              # 1. normalise
            mean=rgb_mean,                      
            std=rgb_std
        ),
        Pad(416),                               # 2. padding
        ToTensor()                              # 3. convert to tensor
    ])
)

# separate transformed dataset into batches
train_loader = DataLoader(
    transformed_train_data,
    batch_size=bs,
    shuffle=True,
    num_workers=0
)

## Define network
net = Net(cfgfile="cfg/model.cfg")

## Define Loss Function and Optimiser
criterion = Yolo_Loss()
optimizer = optim.SGD(
    params=net.parameters(), 
    lr=0.0000001, 
    momentum=0.9
    )

## Train network
CUDA = torch.cuda.is_available()
all_losses = []

n_epoch = 1

print("Training...")
for epoch in range(n_epoch): # each image gets 3 detections, this happens n_epoch times

    running_loss = 0.0
    for i, data in enumerate(train_loader):
        input_img, labels = data.values()
        optimizer.zero_grad()

        # forward pass
        outputs = net(input_img, CUDA)
        # compute loss
        loss = criterion(outputs, labels).float()
        
        # back prop        
        loss.backward()
        optimizer.step()

        # print stats
        running_loss +=loss.item()
        if i % bs == bs-1: # print every bs mini-batches
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / bs:.3f}')
            all_losses.append(running_loss / bs)

            running_loss = 0.0

print("Training complete.")

# save weights
torch.save(net.state_dict(), "weights/100_images.weights")
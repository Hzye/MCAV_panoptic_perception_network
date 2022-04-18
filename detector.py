from __future__ import division

import time
import torch 
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import cv2 
import argparse
import os 
import os.path as osp
import pickle as pkl
import pandas as pd
import random

from model import Net
from utils import *

def arg_parse():
    """
    Use to parse arguments to the 'detect' module.
    """
    parser = argparse.ArgumentParser(description="YOLOv3 Detection Module")

    parser.add_argument(
        "--images",
        dest="images",
        help="Indiv image or directory of images to perform detection on.",
        default="imgs",
        type=str
    )
    parser.add_argument(
        "--dets",
        dest="dets",
        help="Indiv image or directory to save detections to.",
        default="dets",
        type=str
    )
    parser.add_argument(
        "--bs",
        dest="bs",
        help="Batch size.",
        default=1
    )
    parser.add_argument(
        "--confidence",
        dest="confidence",
        help="Object confidence to filter predictions.",
        default=0.5
    )
    parser.add_argument(
        "--nms_thresh",
        dest="nms_thresh",
        help="Non-max suppression threshold",
        default=0.4
    )
    parser.add_argument(
        "--cfg",
        dest="cfgfile",
        help="Config file including model parameters and architecture.",
        default="cfg/model.cfg",
        type=str
    )
    parser.add_argument(
        "--weights",
        dest="weights",
        help="YOLOv3 pre-trained weights.",
        default="weights/yolov3.weights",
        type=str
    )
    parser.add_argument(
        "--res",
        dest="res",
        help="Input resolution of network. Increase for better acc, but slower runtime.",
        default="416",
        type=str
    )

    return parser.parse_args()

args = arg_parse()
images = args.images
batch_size = int(args.bs)
confidence = float(args.confidence)
nms_thresh = float(args.nms_thresh)
start = 0
CUDA = torch.cuda.is_available()

n_classes = 80
classes = load_classes("data/coco.names")


## init network and load weights
print("Loading network...")
model = Net(args.cfgfile)
model.load_weights(args.weightsfile)
print("Network loaded!")

model.net_info["height"] = args.res
in_dims = int(model.net_info["height"])
assert in_dims % 32 == 0
assert in_dims > 32

# check for CUDA availability
if CUDA:
    model.cuda()

# set model in eval mode
model.eval()

## read input images
read_dir = time.time() # checkpoint used to measure time
# Detection
try:
    img_list = [osp.join(osp.realpath('.'), images, img) for img in os.listdir(images)]
except NotADirectoryError:
    img_list = []
    img_list.append(osp.join(osp.realpath('.'), images))
except FileNotFoundError:
    print("No file or directory with the name {}".format(images))
    exit()

# if directory to save detections doesnt exist, create it
if not os.path.exists(args.dets):
    os.makedirs(args.dets)

# use opencv to load images
load_batch = time.time() # another checkpoint
loaded_imgs = [cv2.imread(img) for img in img_list]

# opencv loads images as np array with dims BxGxR
# keep list of original images and list im_dim_list containing dims of original images
img_batches = list(map(prep_image, loaded_imgs, [in_dims for img in range(len(img_list))]))

# list containing dims of original images
im_dims_list = [(img.shape[1], img.shape[0]) for img in loaded_imgs]
im_dims_list = torch.FloatTensor(im_dims_list).repeat(1,2)

if CUDA:
    im_dims_list = im_dims_list.cuda()

## create batches
leftover = 0
if (len(im_dims_list) % batch_size):
    leftover = 1

if batch_size != 1:
    n_batches = len(img_list)//batch_size + leftover
    img_batches = [torch.cat((img_batches[i*batch_size:min((i+1)*batch_size, len(img_batches))])) for i in range(n_batches)]


## detection loop
# 1. iterate over batches
# 2. make prediction
# 3. cat prediction tensors (Dx8 from write_results()) of all images we have to perform detections on

# measure time taken for det as time between taking input and producing output of write_results()
#   - one attr in write_prediction() is the index of img in batch
#   - transform this attr to rep index of image in img_list, which contains addresses of all imgs
#   - print time taken for each det + obj detected
write = 0
start_det_loop = time.time()
for idx, batch in enumerate(img_batches):
    # load image
    start = time.time()
    if CUDA:
        batch = batch.cuda()
    
    prediction = model(Variable(batch, volatile=True), CUDA)
    prediction = write_results(
        prediction=prediction,
        confidence=confidence,
        n_classes=n_classes,
        nms_conf=nms_thresh
    )
    end = time.time()

    if type(prediction) == int:
        for img_num, image in enumerate(img_list[idx*batch_size:min((idx+1)*batch_size, len(img_list))]):
            img_id = idx*batch_size + img_num
            print("{0:20s} predicted in {1:6.3f} seconds".format(image.split("/")[-1], (end - start)/batch_size))
            print("{0:20s} {1:s}".format("Objects Detected:", ""))
            print("----------------------------------------------------------")
        continue
    
    # transform attr from index in batch to index in img_list
    prediction[:,0] += idx*batch_size

    # if output hasnt been init
    if not write:
        output = prediction
        write = 1
    else:
        output = torch.cat((output, prediction))
    
    for img_num, image in enumerate(img_list[idx*batch_size:min((idx+1)*batch_size, len(img_list))]):
        img_id = idx*batch_size + img_num
        objs = [classes[int(x[-1])] for x in output if int(x[0])==img_id]
        print("{0:20s} predicted in {1:6.3f} seconds".format(image.split("/")[-1], (end - start)/batch_size))
        print("{0:20s} {1:s}".format("Objects Detected:", " ".join(objs)))
        print("----------------------------------------------------------")
    
    # make sure CUDA kernel is synchronised with CPU
    # avoids misleading end = time.time()
    if CUDA:
        torch.cuda.synchronize()

## drawing bboxes on images
# check whether there has been a single det has been made or not
# if no, exit program
try:
    output
except NameError:
    print("No detections were made")
    exit()

# 
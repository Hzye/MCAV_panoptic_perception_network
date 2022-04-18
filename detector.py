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
read_dir = time.time()
# Detection
try:
    im_list = [osp.join(osp.realpath('.'), images, img) for img in os.listdir(images)]
except NotADirectoryError:
    imlist = []
    imlist.append(osp.join(osp.realpath('.'), images))
except FileNotFoundError:
    print("No file or directory with the name {}".format(images))
    exit()
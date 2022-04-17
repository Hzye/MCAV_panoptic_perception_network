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
        "--dest",
        dest="dest",
        help="Indiv image or directory to save detections to.",
        default="dest",
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
        "--reso",
        dest="reso",
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



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
    Use to parse arguments to the 'detection_video' module.
    """
    parser = argparse.ArgumentParser(description="YOLOv3 Detection Module")

    parser.add_argument(
        "--video",
        dest="video",
        help="Video to perform detection on.",
        default="test_video.mp4",
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
vid = args.video
batch_size = int(args.bs)
confidence = float(args.confidence)
nms_thresh = float(args.nms_thresh)
start = 0
CUDA = torch.cuda.is_available()

n_classes = 80
classes = load_classes("data/coco.names")
colours = pkl.load(open("pallete", "rb"))

## init network and load weights
print("Loading network...")
model = Net(args.cfgfile)
model.load_weights(args.weights)
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

def draw_test_bbox(x, results):
    c1 = tuple(x[1:3].int())
    c2 = tuple(x[3:5].int())
    img = results
    img_class = int(x[-1])
    colour = random.choice(colours)
    label = "{0}".format(classes[img_class]) # display detected class
    cv2.rectangle(img, (int(c1[0]), int(c1[1])), (int(c2[0]), int(c2[1])), colour, 1)
    t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 1, 1)[0]
    c2 = c1[0]+t_size[0]+3, c1[1]+t_size[1]+4
    cv2.rectangle(img, (int(c1[0]), int(c1[1])), (int(c2[0]), int(c2[1])), colour, -1)
    cv2.putText(img, label, (int(c1[0]), int(c1[1])+t_size[1]+4), cv2.FONT_HERSHEY_PLAIN, 1, [255,255,255], 1)
    return img

## detection
vid = "test_video.mp4"
#print(vid)
cap = cv2.VideoCapture(0) # VideoCapture(0) for webcam | VideoCapture(vid) for video
assert cap.isOpened(), "Cannot capture source"

frames = 0
start = time.time()

# iterate over frames instead of batches
# simplifies code alot compared to batches of images
# replace im_dim_list with tuple 

# every iter, keep track of n_frames captured in frames
# divide n_frames by time elapsed for fps

# use cv2.imshow instead of cv2.imwrite to display frame live with bbox drawn on it

while cap.isOpened():
    ret, frame = cap.read()

    if ret:
        img = prep_image(frame, in_dims)
        cv2.imshow("a", frame)
        img_dims = frame.shape[1], frame.shape[0]
        img_dims = torch.FloatTensor(img_dims).repeat(1,2)
        
        if CUDA:
            img_dims = img_dims.cuda()
            img = img.cuda()
        
        with torch.no_grad():
            prediction = model(Variable(img), CUDA)

        prediction = write_results(
            prediction=prediction,
            confidence=confidence,
            n_classes=n_classes,
            nms_conf=nms_thresh
        )

        if type(prediction) == int:
            frames += 1
            print("FPS: {:5.4f}".format(frames/(time.time() - start)))
            cv2.imshow("frame", frame)
            key = cv2.waitKey(1)
            if key & 0xFF == ord('q'):
                break
            continue
    

        img_dims = img_dims.repeat(prediction.size(0), 1)
        scaling_factor = torch.min(416/img_dims,1)[0].view(-1,1)

        # transform bbox corner attrs to match padded images
        prediction[:,[1,3]] -= (in_dims - scaling_factor*img_dims[:,0].view(-1,1))/2
        prediction[:,[2,4]] -= (in_dims - scaling_factor*img_dims[:,1].view(-1,1))/2

        # undo rescaling to get coords of bbox on original image
        prediction[:,1:5] /= scaling_factor

        # clip bboxes that may have boundaries outside image
        for i in range(prediction.shape[0]):
            prediction[i, [1,3]] = torch.clamp(prediction[i, [1,3]], 0.0, img_dims[i,0])
            prediction[i, [2,4]] = torch.clamp(prediction[i, [2,4]], 0.0, img_dims[i,1])

        list(map(lambda x: draw_test_bbox(x, frame), prediction))

        # show live detection
        cv2.imshow("frame", frame)
        key = cv2.waitKey(1)
        if key & 0xFF == ord('q'):
            break
        frames += 1
        print(time.time() - start)
        print("FPS: {:5.2f}".format(frames/(time.time() - start)))
    else:
        break
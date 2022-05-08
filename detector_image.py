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
    Use to parse arguments to the 'detector_image' module.
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

# store appropirate argument parses into variables
# define other important variables
args = arg_parse()
images = args.images # dir/image 
batch_size = int(args.bs) 
confidence = float(args.confidence)
nms_thresh = float(args.nms_thresh)
mean = [92.11938007161459, 102.83839236762152, 104.90335580512152]
std = [66.09941202519124, 70.6808655565459, 75.05305001603533]
start = 0 # for timing
CUDA = torch.cuda.is_available()

n_classes = 12
classes = load_classes("data/coco.names")


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

## read input images
read_dir = time.time() # checkpoint used to measure time

# store list of image names in img_List
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


load_batch = time.time() # another checkpoint

# load image pixel values into list loaded_imgs
loaded_imgs = [cv2.imread(img) for img in img_list]

# opencv loads images as np array with dims BxGxR
# use prep_image to:
#   1. normalise the image
#   2. pad the image  
#   3. swap dims to correct order
# save transformed images to list img_batches
img_batches = list(map(
    prep_image, 
    loaded_imgs, 
    [in_dims for img in range(len(img_list))], 
    [mean for img in range(len(img_list))], 
    [std for img in range(len(img_list))]
))

# save original dimensions of images in list img_dims_list
img_dims_list = [(img.shape[1], img.shape[0]) for img in loaded_imgs]
img_dims_list = torch.FloatTensor(img_dims_list).repeat(1,2)

# print("img_list:")
# for img in img_list:
#     print(img)
# print("loaded_imgs")
# for img in loaded_imgs:
#     print(img)
# print("img_batches")
# for img in img_batches:
#     print(img.shape)
# print("in_dims: {}".format(in_dims))
# print("img_dims_list")
# for img in img_dims_list:
#     print(img)

if CUDA:
    img_dims_list = img_dims_list.cuda()

## create batches
leftover = 0
# (n of images)/(size of each batch) = remainder?
if (len(img_dims_list) % batch_size):
    leftover = 1

# ensure even batches given n_images and bs
if batch_size != 1:
    n_batches = len(img_list)//batch_size + leftover
    img_batches = [torch.cat((img_batches[i*batch_size:min((i+1)*batch_size, len(img_batches))])) for i in range(n_batches)]

# print("img_batches post")
# for img in img_batches:
#     print(img.shape)

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
    
    # feed batch into the model and predict classes, bboxes
    with torch.no_grad():
        torch.save(batch, "ex_tensors/model_input_{}".format(idx))
        prediction = model(Variable(batch), CUDA)
    
    #torch.save(prediction, "{}.pt".format(idx))

    # separate results of the prediction
    prediction = write_results(
        prediction=prediction,
        confidence=confidence,
        n_classes=n_classes,
        nms_conf=nms_thresh
    )
    end = time.time()
    
    #print(prediction)

    if type(prediction) == int:
        for img_num, image in enumerate(img_list[idx*batch_size:min((idx+1)*batch_size, len(img_list))]):
            img_id = idx*batch_size + img_num
            #print("here: {}".format(image))
            print("{0:20s} predicted in {1:6.3f} seconds".format(image.split("\\")[-1], (end - start)/batch_size))
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
        #print("here: {}".format(image))
        print("{0:20s} predicted in {1:6.3f} seconds".format(image.split("\\")[-1], (end - start)/batch_size))
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
    print("No detections were made.")
    exit()

# transform bbox corner attrs to match padded images
img_dims_list = torch.index_select(img_dims_list, 0, output[:,0].long())
scaling_factor = torch.min(in_dims/img_dims_list,1)[0].view(-1,1)

output[:,[1,3]] -= (in_dims - scaling_factor*img_dims_list[:,0].view(-1,1))/2
output[:,[2,4]] -= (in_dims - scaling_factor*img_dims_list[:,1].view(-1,1))/2

# undo rescaling to get coords of bbox on original image
output[:,1:5] /= scaling_factor

# clip bboxes that may have boundaries outside image
for i in range(output.shape[0]):
    output[i, [1,3]] = torch.clamp(output[i, [1,3]], 0.0, img_dims_list[i,0])
    output[i, [2,4]] = torch.clamp(output[i, [2,4]], 0.0, img_dims_list[i,1])

# colours for bboxes
output_recast = time.time()
class_load = time.time()
colours = pkl.load(open("pallete", "rb"))

# function to draw boxes
draw = time.time()

def draw_test_bbox(x, results):
    c1 = tuple(x[1:3].int())
    c2 = tuple(x[3:5].int())
    img = results[int(x[0])]
    img_class = int(x[-1])
    colour = random.choice(colours)
    label = "{0}".format(classes[img_class]) # display detected class
    cv2.rectangle(img, (int(c1[0]), int(c1[1])), (int(c2[0]), int(c2[1])), colour, 1)
    t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 1, 1)[0]
    c2 = c1[0]+t_size[0]+3, c1[1]+t_size[1]+4
    cv2.rectangle(img, (int(c1[0]), int(c1[1])), (int(c2[0]), int(c2[1])), colour, -1)
    cv2.putText(img, label, (int(c1[0]), int(c1[1])+t_size[1]+4), cv2.FONT_HERSHEY_PLAIN, 1, [255,255,255], 1)
    return img

# draw bboxes ON image
# modify loaded_imgs inplace
#list(map(lambda x: draw_test_bbox(x, loaded_imgs, colours), output))
list(map(lambda x: draw_test_bbox(x, loaded_imgs), output))

# each img saved by prefixing "det_" infront of image name
# create a list of addresses to which we sasve detection images to
det_names = pd.Series(img_list).apply(lambda x: "{}/det_{}".format(args.dets, x.split("\\")[-1]))

# write images with dets to address in det_names
list(map(cv2.imwrite, det_names, loaded_imgs))
print(det_names)
end = time.time()

# prit a summary of times
print("SUMMARY")
print("----------------------------------------------------------")
print("{:25s}: {}".format("Task", "Time Taken (in seconds)"))
print()
print("{:25s}: {:2.3f}".format("Reading addresses", load_batch - read_dir))
print("{:25s}: {:2.3f}".format("Loading batch", start_det_loop - load_batch))
print("{:25s}: {:2.3f}".format("Detection (" + str(len(img_list)) +  " images)", output_recast - start_det_loop))
print("{:25s}: {:2.3f}".format("Output Processing", class_load - output_recast))
print("{:25s}: {:2.3f}".format("Drawing Boxes", end - draw))
print("{:25s}: {:2.3f}".format("Average time_per_img", (end - load_batch)/len(img_list)))
print("----------------------------------------------------------")


torch.cuda.empty_cache()
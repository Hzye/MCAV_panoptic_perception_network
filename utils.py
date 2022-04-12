from __future__ import division
from matplotlib.pyplot import grid
from pyparsing import null_debug_action

import torch 
import torch.nn as nn
import torch.nn.functional as F 
from torch.autograd import Variable
import numpy as np
import cv2 

def predict_transform(prediction, in_dims, anchors, n_classes, CUDA=True):
    """
    Converts detection feature map into 2D tensor, where each row are attributes of bbox.
    """
    batch_size = prediction.size(0)
    stride = in_dims // prediction.size(2)
    grid_size = in_dims // stride
    bbox_attrs = 5 + n_classes
    n_anchors = len(anchors)

    # reshaping
    prediction = prediction.view(batch_size, bbox_attrs*n_anchors, grid_size*grid_size)
    prediction = prediction.transpose(1,2).contiguous()
    prediction = prediction.view(batch_size, grid_size*grid_size*n_anchors, bbox_attrs)

    # divide anchors by stride of detection feature map as input image is 
    # larger than detection map by a factor of stride
    anchors = [(a[0]/stride, a[1]/stride) for a in anchors]

    ## transform output
    # apply sigmoid to x,y coords and object confidence score
    prediction[:,:,0] = torch.sigmoid(prediction[:,:,0])
    prediction[:,:,1] = torch.sigmoid(prediction[:,:,1])
    prediction[:,:,4] = torch.sigmoid(prediction[:,:,4])

    # add grid offset to centre coord predictions
    grid = np.arange(grid_size)
    a,b = np.meshgrid(grid,grid)

    # convert to float and reshape into column vectors
    x_offset = torch.FloatTensor(a).view(-1,1)
    y_offset = torch.FloatTensor(b).view(-1,1)

    if CUDA:
        x_offset = x_offset.cuda()
        y_offset = y_offset.cuda()
    
    x_y_offset = torch.cat((x_offset, y_offset), 1).repeat(1, n_anchors).view(-1,2).unsqueeze(0)

    prediction[:,:,:2] += x_y_offset

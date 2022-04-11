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

    
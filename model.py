from __future__ import division

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

def parse_cfg(cfgfile):
    """
    Parses config file and stores every block as a dicts.
    
    Attributes and their values are stored as key-value pairs.
    
    As the cfg file is parsed through, append these dicts, denoted by variable 'block' in code
    to a list 'blocks'.
    """
    ## Preprocessing
    file = open(cfgfile, 'r')
    lines = file.read().split('\n') # read each line, separated by newline
    lines = [x for x in lines if len(x) > 0] # dont read empty lines
    lines = [x for x in lines if x[0] != '#'] # dont read comment lines
    lines = [x.rstrip().lstrip() for x in lines] # remove fringe spaces

    ## Loop over resultant list to get blocks
    block = {}
    blocks = []

    for line in lines:
        # the square bracket marks a start of a block and that line includes block name
        if line[0] == "[": 

            # if block dict is NOT empty, it must be storing info from prev block
            if len(block) != 0: 
                blocks.append(block) # therefore add it to blocks list
                block = {} # reset block

            # if block is already empty, start filling it in
            block["type"] = line[1:-1].rstrip() # type is the string starting the first index in line
                                                # remove/strip the last character, which should be "]"
        # if there is no square bracket, it must include values of that block
        else:
            # extract attribute and value pair and store in dict
            key, value = line.split("=") # split by the equal sign -> "key=" , "=value"
            block[key.rstrip()] = value.lstrip() # remove "=" from both
    blocks.append(block)

    return blocks             

class EmptyLayer(nn.Module):
    def __init__(self):
        super(EmptyLayer, self).__init__()

class DetectionLayer(nn.Module):
    def __init__(self, anchors):
        super(DetectionLayer, self).__init__()
        self.anchors = anchors

def create_modules(blocks):
    net_info = blocks[0] # save info from [net] in cfg
    module_list = nn.ModuleList()
    prev_filters = 3 # 3 filters for each of the RGB channels
    output_filters = []

    # iterate over list of blocks and create pytorch module for each one
    for index, block in enumerate(blocks[1:]): # start from entry 1 -> end, so skip the [net] entry
        module = nn.Sequential()

        # ------------------------------
        # 1. check type of block
        # 2. create new module for block
        # 3. append to module_list
        # ------------------------------

        # ------------------------------
        # 1.1 Layer types:
        #   [convolutional]
        #   [upsample]
        #   [route]
        #   [shortcut]
        # ------------------------------

        # [convolutional] : (1)conv -> (2)batch_norm -> (3)leaky
        # take from pytorch.nn.
        if (block["type"] == "convolutional"):

            # get the info about the layer
            activation = block["activation"]
            try:
                batch_normalize = int(block["batch_normalize"])
                bias = False
            except:
                batch_normalize = 0
                bias = True

            filters = int(block["filters"])
            padding = int(block["pad"])
            kernel_size = int(block["size"])
            stride = int(block["stride"])

            if padding:
                pad = (kernel_size - 1) // 2
            else:
                pad = 0

            # (1) add conv layer
            conv = nn.Conv2d(
                in_channels=prev_filters,
                out_channels=filters,
                kernel_size=kernel_size,
                stride=stride,
                padding=pad, 
                bias=bias
            )
            module.add_module("conv_{0}".format(index), conv) # name and actual layer

            # (2) add batch norm layer
            if batch_normalize:
                bn = nn.BatchNorm2d(filters)
                module.add_module("batch_norm_{0}".format(index), bn)

            # check activation
            # it is either linear or learky relu or yolo
            # (3) add leaky layer
            if activation == "leaky":
                act = nn.LeakyReLU(0.1, inplace=True)
                module.add_module("leaky_{0}".format(index), act)

        # [upsample] 
        # take from pytorch.nn.Upsample()
        elif (block["type"] == "upsample"):
            stride = int(block["stride"])
            upsample = nn.Upsample(scale_factor=2, mode='bilinear')
            module.add_module("upsample_{0}".format(index), upsample)

        # [route] - make manually 
        elif (block["type"] == "route"):
            block["layers"] = block["layers"].split(',')
            # start of route
            start = int(block["layers"][0])
            # end, there it exists
            try:
                end = int(block["layers"][1])
            except:
                end = 0

            # positive annotation
            if start > 0:
                start = start - index
            if end > 0:
                end = end - index
            route = EmptyLayer()
            module.add_module("route_{0}".format(index), route)

            if end < 0:
                filters = output_filters[index + start] + output_filters[index + end]
            else:
                filters = output_filters[index + start]
        
        # [shortcut]
        elif (block["type"] == "shortcut"):
            shortcut = EmptyLayer()
            module.add_module("short_cut_{0}".format(index), shortcut)

        # [yolo]
        elif (block["type"] == "yolo"):
            # extract masks into list 
            mask = block["mask"].split(',')
            mask = [int(x) for x in mask] # make sure they are int

            # extract anchors into list
            anchors = block["anchors"].split(',')
            anchors = [int(a) for a in anchors] # make sure they are int
            anchors = [(anchors[i], anchors[i+1]) for i in range(0, len(anchors), 2)] # iterate through every second anchor and save pairs
            anchors = [anchors[i] for i in mask] # now only save anchors corresponding to mask indices

            detection = DetectionLayer(anchors)
            module.add_module("detection_{0}".format(index), detection)
        
        module_list.append(module)
        prev_filters = filters
        output_filters.append(filters)
    
    return (net_info, module_list)
from __future__ import division

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

from utils import *

def test_forward_pass():
    """
    Simple function to test forward pass.
    """
    img = cv2.imread("images/0000f77c-62c2a288.jpg")
    img = cv2.resize(img, (416,416)) # resize to input dims
    reshaped_img = img[:,:,::-1].transpose((2,0,1)) # H x W x C -> C x H x W
    reshaped_img = reshaped_img[np.newaxis,:,:,:]/255.0 # add channel at 0 for batch norm
    reshaped_img = torch.from_numpy(reshaped_img).float() # convert to float
    reshaped_img = Variable(reshaped_img) # convert to variable
    return reshaped_img

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

class Net(nn.Module):
    def __init__(self, cfgfile):
        super(Net, self).__init__()
        self.blocks = parse_cfg(cfgfile)
        self.net_info, self.module_list = create_modules(self.blocks)

    ## Forward pass
    # 1. calculate output
    # 2. transform output detection feature maps so that it can be 
    #    processed easier 
    def forward(self, x, CUDA):
        modules = self.blocks[1:]
        outputs = {} # cache outputs for route layer
        #print("hello")

        # iterate over module_list, pass input through each module
        # in -> [module[0]] -> [module[1]] -> ...
        write = 0 # used as flag to inidicate if we've encountered first detection yet for yolo 
        for idx, module in enumerate(modules):
            module_type = (module["type"])
            #print(module_type)

            ## CONV and UPSAMPLE LAYERS:
            # pass input -> conv/upsample module -> output
            if (module_type == "convolutional") or (module_type == "upsample"):
                x = self.module_list[idx](x) # pass in 
            
            ## ROUTE LAYERS
            # case 1: layer = n
            # - output feature map from the layer n-layers backward
            # case 2: layer = n, m
            # - concat(feature map from n-layers back + mth layer) along depth dim
            elif (module_type == "route"):
                layers = module["layers"]
                layers = [int(n) for n in layers]

                if (layers[0]) > 0:
                    layers[0] = layers[0] - idx # refer to current layer/module idx 

                # for case 1
                if len(layers) == 1:
                    x = outputs[idx + (layers[0])] # pull from cache n layers ago
                
                # for case 2
                else:
                    if (layers[1]) > 0:
                        layers[1] = layers[1] - idx # refer to current layer/module idx
                    
                    feature_map_1 = outputs[idx + layers[0]] # take feature map from n-layers back
                    feature_map_2 = outputs[idx + layers[1]] # take feature map from mth layer     

                    # concat feature maps along depth dim
                    x = torch.cat((feature_map_1, feature_map_2), 1)

            ## SHORTCUT LAYERS
            # from = n
            # output = (feature map from prev layer) + (feature layer from n-layers back)
            elif (module_type == "shortcut"):
                from_ = int(module["from"])
                x = outputs[idx-1] + outputs[idx+from_]           

            ## YOLO LAYERS
            # output = conv feature map containing bbox attributes along depth of 
            #          feature map (attribute bboxes predicted are stacjed 1 by 1
            #          along each other)
            # so to access 3rd bbox at cell (6,9) requires index:
            #          map[5, 6, 2*(5+C): 3*(5+C)], where C is n_classes
            # this sucks
            # another issue: detection happens at 3 different scales
            #              ->dims of pred maps different
            #              ->use predict_transform
            elif (module_type == "yolo"):
                # concat detection maps at three diff scales into one bit tensor (possible post transform)

                # cannot init empty tensor, therefore:
                # 1. delay collector init until first detection map
                # 2. concat maps to it after subsequent detections             

                anchors = self.module_list[idx][0].anchors
                # get input dims and n_classes
                in_dims = int(self.net_info["height"])
                n_classes = int(module["classes"])

                # transform
                x = x.data
                x = predict_transform(
                    prediction=x, 
                    in_dims=in_dims, 
                    anchors=anchors, 
                    n_classes=n_classes,
                    CUDA=CUDA
                )
                # recall write=0 means collector hasnt been initatlised
                if not write:
                    detections = x
                    write = 1
                else:
                    detections = torch.cat((detections, x), 1)
            # save current output
            outputs[idx] = x
        return detections
from __future__ import division

import torch 
import torch.nn as nn
import torch.nn.functional as F 
from torch.autograd import Variable
import numpy as np
import cv2 

def predict_transform(prediction, in_dims, anchors, n_classes, CUDA=True):
    """
    Converts detection feature map from convolution layer into 2D tensor, where each row are attributes of bbox.

    Inputs:
    =prediction=    output from previous conv layer. size (n_batches, n_conv_filters, grid_size, grid_size)
                        -> we expect 3 different predictions to come in for EACH IMAGE, with grid_sizes:
                            - [n_batches, n_conv_filters, 13, 13]
                            - [n_batches, n_conv_filters, 26, 26]
                            - [n_batches, n_conv_filters, 52, 52]
    =in_dims=       model height
    =anchors=       widths, heights of anchor boxes. size (n_anchors, 2) -> [(a1_w, a1_h), (a2_w, a2_h), (a3_w, a3_h)]
    =n_classes=     number of object classes in image dataset

    Output:
    =prediction=    output from yolo detection layer. size (n_batches, (grid_size*grid_size*n_anchors), (4+1+n_classes))
                        -> we expect 3 different predictions to output for EACH IMAGE, with grid_sizes:
                            - [n_batches, (13*13*n_anchors), (4+1+n_classes)]
                            - [n_batches, (26*26*n_anchors), (4+1+n_classes)]
                            - [n_batches, (52*52*n_anchors), (4+1+n_classes)]
"""
    # print(prediction.shape)
    # print(prediction[0])
    #torch.save(prediction, "yolo_layer_input.pt")
    batch_size = prediction.shape[0]
    stride = in_dims // prediction.shape[2]
    grid_size = prediction.shape[2]
    bbox_attrs = 5 + n_classes
    n_anchors = len(anchors)

    # reshaping
    prediction = prediction.view(batch_size, bbox_attrs*n_anchors, grid_size*grid_size) # size (n_batches, n_conv_filters, (grid_w*grid_h))
    prediction = prediction.transpose(1,2).contiguous()                                 # size (n_batches, (grid_w*grid_h), n_conv_filters)
    prediction = prediction.view(batch_size, grid_size*grid_size*n_anchors, bbox_attrs) # size (n_batches, (grid_w*grid_h*n_anchors), 4+1+n_classes)

    # divide anchors by stride of detection feature map as input image is larger than detection map by a factor of stride
    anchors = [(a[0]/stride, a[1]/stride) for a in anchors]

    ## transform output
    # apply sigmoid to x,y coords and object confidence score
    prediction[:,:,0] = torch.sigmoid(prediction[:,:,0]) # centre x coord - first index of each bbox attr row
    prediction[:,:,1] = torch.sigmoid(prediction[:,:,1]) # centre y coord
    prediction[:,:,4] = torch.sigmoid(prediction[:,:,4]) # obj confidence score

    # add grid offset to centre coord predictions
    grid = np.arange(grid_size)
    a,b = np.meshgrid(grid,grid)

    # convert to float and reshape into column vectors
    x_offset = torch.FloatTensor(a).view(-1,1)      # size (grid_size*grid_size, 1)    (0, 1, ..., grid_size) * grid_size times
    y_offset = torch.FloatTensor(b).view(-1,1)      # size (grid_size*grid_size, 1)    (0) * grid_size times, (1) * grid_size times, ... (grid_size) * grid_size times

    if CUDA:
        x_offset = x_offset.cuda()
        y_offset = y_offset.cuda()
    
    x_y_offset = torch.cat((x_offset, y_offset), 1) # size (grid_size*grid_size, 2)
    x_y_offset = x_y_offset.repeat(1, n_anchors)    # size (grid_size*grid_size, 2*n_anchors)
    x_y_offset = x_y_offset.view(-1,2)              # size (grid_size*grid_size*n_anchors, 2)
    x_y_offset = x_y_offset.unsqueeze(0)            # size (1, grid_size*grid_size*n_anchors, 2)

    # apply x y grid offset to x y centre coords (indices 0, 1 of bbox attrs row)
    prediction[:,:,:2] = prediction[:,:,:2] + x_y_offset

    ## apply anchors to dimensions of bbox
    # log transform height and width

    # convert to tensor
    anchors = torch.FloatTensor(anchors) # size (n_anchors, 2)

    if CUDA:
        anchors = anchors.cuda()

    anchors = anchors.repeat(grid_size*grid_size, 1).unsqueeze(0) # size (1, grid_size*grid_size, 2)
    
    # apply anchors to bbox width and height (indices 2, 3 of bbox attrs row)
    prediction[:,:,2:4] = torch.exp(prediction[:,:,2:4])*anchors

    ## apply sigmoid activation to class scores
    prediction[:,:,5:5+n_classes] = torch.sigmoid((prediction[:,:,5:5+n_classes]))

    # resize detection map to size of input image
    # bbox attributes are sized according to feature map e.g. 13x13
    # if input image is 416x416, multiply attributes by 32, or stride

    # only need to apply to x, y centres, height and width
    prediction[:,:,:4] = prediction[:,:,:4]*stride

    return prediction

def unique(tensor):
    """
    Returns tensor list of unique class indices detected.
    """
    # convert to np array
    tensor_np = tensor.cpu().numpy()
    #
    unique_np = np.unique(tensor_np)
    unique_tensor = torch.from_numpy(unique_np)

    tensor_res = tensor.new(unique_tensor.shape)
    tensor_res.copy_(unique_tensor)
    return tensor_res

def bbox_iou(box1, box2):
    """
    Returns intersection over union of two bounding boxes.
    """
    # get coords of bboxes
    b1_x1, b1_y1, b1_x2, b1_y2 = box1[:,0], box1[:,1], box1[:,2], box1[:,3]
    b2_x1, b2_y1, b2_x2, b2_y2 = box2[:,0], box2[:,1], box2[:,2], box2[:,3]

    # get coords of intersection
    intersect_x1 = torch.max(b1_x1, b2_x1)
    intersect_y1 = torch.max(b1_y1, b2_y1)
    intersect_x2 = torch.min(b1_x2, b2_x2)
    intersect_y2 = torch.min(b1_y2, b2_y2)

    # intersection area
    # clamp to > 0
    # this avoids areas being calculated for boxes with zero intersect
    intersect_area = torch.clamp(intersect_x2 - intersect_x1, min=0)*torch.clamp(intersect_y2 - intersect_y1, min=0)

    # union area
    b1_area = (b1_x2 - b1_x1)*(b1_y2 - b1_y1)
    b2_area = (b2_x2 - b2_x1)*(b2_y2 - b2_y1)
    union_area = b1_area + b2_area - intersect_area

    # compute iou
    iou = intersect_area/union_area

    return iou    

def write_results(prediction, confidence, n_classes, nms_conf=0.4):
    """
    Filters raw predictions made from network forward pass based on object confidence thresholding and NMS.

    Inputs:
    =prediction=    output prediction tensor from network, size (batch_size, n_bboxes, 4 bbox attrs + 1 obj score + n_classes)
    =confidence=    object confidence score threshold, default 0.5
    =n_classes=     number of classes within dataset
    =nms_conf=      non max suppression confidence threshold, default 0.4

    Outputs:
    =output=        returns number of tensors based on number of predicted objects (1, 8)
                    [n_in_batch, min_x, min_y, max_x, max_y, max_class_score, max_class_score_idx]
    """

    ## [1] Filter predictions based on OBJECT CONFIDENCE THRESHOLD

    # prediction tensor contains info about Bx10647 bboxes

    #   1. take the 4th index of all bbox attr rows - this contains objectness score (confidence that there is an object)
    #   2. convert tensor of T and F -> 0s and 1s so we can multiply to entire predictions tensor
    #   3. right now prediction.size(1, 22743, 85) but mask.size(1, 22743)
    #       -> unsqueeze() in the 3rd dim (or index 2) such that mask.size(1, 22743, 1)
    #   4. now shapes match so we can multiply to mask ENTIRE bbox attr rows which do not meet obj conf threshold

    conf_mask = (prediction[:,:,4] > confidence).float().unsqueeze(2) 
    prediction = prediction*conf_mask

    ## [2] Filter predictions based on HIGHEST CLASS SCORE

    # bbox attrs described by centre coords
    # convert (centre_x, centre_y, height, width) -> (min_x, min_y, max_x, max_y)
    box_corner = prediction.new(prediction.shape)
    box_corner[:,:,0] = (prediction[:,:,0] - prediction[:,:,2]/2)
    box_corner[:,:,1] = (prediction[:,:,1] - prediction[:,:,3]/2)
    box_corner[:,:,2] = (prediction[:,:,0] + prediction[:,:,2]/2)
    box_corner[:,:,3] = (prediction[:,:,1] + prediction[:,:,3]/2)
    # replace in the prediction tensor
    prediction[:,:,:4] = box_corner[:,:,:4]

    # cannot vectorise confidence thresholding and nms
    # loop over first dim of prediction which contains indexes of images in a batch
    batch_size = prediction.size(0)

    write = False # indicates if we have initialised 'output' 
    # output collects filtered detections across entire batch

    # loop through image predictions in batch
    for idx in range(batch_size):
        image_pred = prediction[idx] # ith image tensor

        # each bbox row has 85 attrs, last 80 are class scores (%)
        #   1. look at all n_classes class scores (aka remove 4 bbox attrs + obj score)
        #       -> image_pred[:, 5:5+n_classes] (n_bboxes, n_classes)
        #   2. find 'most likely class' by finding max class prob in FIRST DIM
        #       -> save tensor of max score and index of score
        #   3. unsqueeze to add appropriate dims ()
        #       -> (n_bboxes) -> (n_bboxes, 1)      
        
        max_class_score, max_class_score_idx = torch.max(image_pred[:,5:5+n_classes], 1) # max_class_score, max_class_score_idx (n_bboxes)
        max_class_score = max_class_score.float().unsqueeze(1) # max_class_score (n_bboxes, 1)
        max_class_score_idx = max_class_score_idx.float().unsqueeze(1) # max_class_score_score (n_bboxes, 1)

        # concatenate bbox_attrs + max_class_score + max_class_score_idx
        seq = (image_pred[:,:5], max_class_score, max_class_score_idx) 
        image_pred = torch.cat(seq, 1) # size (n_bboxes, (4+1)+(1)+(1)) = (n_bboxes, 7)

        # indices of all nonzero (valid) bbox rows
        non_zero_idx = (torch.nonzero(image_pred[:,4]))

        # if there are no detection, skip rest of loop body for current image
        try:
            image_pred_ = image_pred[non_zero_idx.squeeze(),:].view(-1,7)
        except:
            continue

        if image_pred_.shape[0] == 0:
            continue

        # we have now filtered bbox predictions down to those with obj confidence > 0.5, with info:
        # [min_x, min_y, max_x, max_y, Pr(object), max_class_score, max_class_score_idx], size (1, 7)
        # image_pred_ size (n_filtered_preds, 7)
        
        # obtain a tensor list of all classes.
        img_classes = unique(image_pred_[:,-1])
        
        ###########################################################################################
        
        # arrange image_pred_ from high to low based on Pr(obj)
        # loop through image_pred_:
        #   check IoU with all other 
        #   if > iou_thresh then delete
        
        conf_sort_idx = torch.sort(image_pred_[:,4], descending=True)[1]
        image_pred_ordered = image_pred_[conf_sort_idx]

        n_detections = image_pred_ordered.size(0)

        ## non max suppression
        for det in range(n_detections):
            # get IOUs of all boxes that come after the one we are looking at in loop
            # use try except because bboxes may be removed from image_pred_class
            try:
                # ious indexed by det with all bboxes having indices > det
                ious = bbox_iou(image_pred_ordered[det].unsqueeze(0), image_pred_ordered[det+1:])
            except ValueError:
                break

            except IndexError:
                break
            
            # ever iteration, if any bboxes that have indices > det also have iou 
            # (with box indexed by det) larger than threshold nms_thresh, eliminate it

            # zero out all detections that have iou > threshold
            iou_mask = (ious < nms_conf).float().unsqueeze(1)
            image_pred_ordered[det+1:] *= iou_mask

            # remove non-zero entries
            non_zero_idx = torch.nonzero(image_pred_ordered[:,4]).squeeze()
            image_pred_ordered = image_pred_ordered[non_zero_idx].view(-1,7)
        
            ###########################################################################################

        # output tensor (n_true_preds_in_all_images x 8)
        # 8 attrs are:
        #   - idx of image in batch to which detection belongs to
        #   - 4 corner coords
        #   - objectness score
        #   - score of class with max confidence
        #   - idx of class with max confidence

        # do not init output tensor unless we have a detection to assign to
        # check write flag
        batch_idx = image_pred_ordered.new(image_pred_ordered.size(0), 1).fill_(idx)
        
        # repeat batch_id for as many detections of class det in image
        seq = batch_idx, image_pred_ordered

        if not write:
            output = torch.cat(seq,1)
            write = True
        else:
            pre_output = torch.cat(seq,1)
            output = torch.cat((output, pre_output))

    # check whether otuput has been init at all
    # if no then there hasnt been a detection in any images in the batch  
    try:
        return output
    except:
        return 0

def load_classes(namesfile):
    fp = open(namesfile, "r")
    names = fp.read().split("\n")[:-1]
    return names

def norm_with_padding(img, in_dims):
    """
    Adds padding to image in order to adhere to size (in_dims, in_dims, 3).

    Does not change aspect ratio.
    """
    img_w, img_h = img.shape[1], img.shape[0]
    w, h = in_dims

    # calculate new width and height
    new_w = int(img_w * min(w/img_w, h/img_h))
    new_h = int(img_h * min(w/img_w, h/img_h))
    resized_image = cv2.resize(img, (new_w,new_h), interpolation = cv2.INTER_CUBIC)
    
    canvas = np.full((in_dims[1], in_dims[0], 3), 128)

    canvas[(h-new_h)//2:(h-new_h)//2 + new_h,(w-new_w)//2:(w-new_w)//2 + new_w,  :] = resized_image
    
    return canvas

def prep_image(img, in_dims):
    """
    Prepare image for inputting to the neural network. 
    
    Returns a Variable 
    """
    # first pad the image
    img = (norm_with_padding(img, (in_dims, in_dims)))
    # then transpose to correct dims order and tensorise 
    img = img[:,:,::-1].transpose((2,0,1)).copy()
    img = torch.from_numpy(img).float().div(255.0).unsqueeze(0)
    
    return img
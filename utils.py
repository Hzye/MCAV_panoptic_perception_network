from __future__ import division

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
    grid_size = prediction.shape[2]
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

    ## apply anchors to dimensions of bbox
    # log transform height and width
    anchors = torch.FloatTensor(anchors)

    if CUDA:
        anchors = anchors.cuda()

    anchors = anchors.repeat(grid_size*grid_size, 1).unsqueeze(0)
    prediction[:,:,2:4] = torch.exp(prediction[:,:,2:4])*anchors

    # apply sigmoid activation to class scores
    prediction[:,:,5:5+n_classes] = torch.sigmoid((prediction[:,:,5:5+n_classes]))

    # resize detection map to size of input image
    # bbox attributes are sized according to feature map e.g. 13x13
    # if input image is 416x416, multiply attributes by 32, or stride
    prediction[:,:,:4] *= stride

    return prediction

def unique(tensor):
    """
    Returns classes present in any given image.
    """
    tensor_np = tensor.cpu().numpy()
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
    intersect_x2 = torch.max(b1_x2, b2_x2)
    intersect_y2 = torch.max(b1_y2, b2_y2)

    # intersection area
    # clamp to > 0
    intersect_area = torch.clamp(intersect_x2 - intersect_x1+1, min=0)*torch.clamp(intersect_y2 - intersect_y1+1, min=0)

    # union area
    b1_area = (b1_x2 - b1_x1)*(b1_y2 - b1_y1)
    b2_area = (b2_x2 - b2_x1)*(b2_y2 - b2_y1)
    union_area = b1_area + b2_area - intersect_area

    # compute iou
    iou = intersect_area/union_area

    return iou    

def write_results(prediction, confidence, n_classes, nms_conf=0.4):
    """
    Objectness score thresholding and non-max suppression.

    Input:
    =prediction= tensor of size (batch_size) x (n_bboxes) x (4 bbox attrs + 1 obj score + 80 class scores)
    """

    ## object confidence thresholding

    # prediction tensor contains info about Bx10647 bboxes
    # for each bbox that has an objectness score below threshol:
    # we set the value of its attr *entire row representing bbox) to zero
    conf_mask = (prediction[:,:,4] > confidence).float().unsqueeze(2)
    prediction = prediction*conf_mask

    ## performing non-max suppression

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

    for idx in range(batch_size):
        image_pred = prediction[idx] # image tensor

        # each bbox row has 85 attrs, last 80 are class scores (%)
        # so remove all class scores for each row except one with max value
        # save indices + classes of max value classes
        max_conf, max_conf_score = torch.max(image_pred[:,5:5+n_classes], 1)
        max_conf = max_conf.float().unsqueeze(1)
        max_conf_score = max_conf_score.float().unsqueeze(1)
        seq = (image_pred[:,:5], max_conf, max_conf_score)
        image_pred = torch.cat(seq, 1)

        # remove filtered object confidences
        non_zero_ind = (torch.nonzer(image_pred[:,4]))

        # if there are no detection, skip rest of loop body for current image
        try:
            image_pred_ = image_pred[non_zero_ind.squeeze(),:].view(-1,7)
        except:
            continue

        if image_pred_.shape[0] == 0:
            continue

        # get classes detected in current image
        im_classes = unique(image_pred_[:,-1]) # all except last element - holds class index

        ## classwise non max suppression
        for classes in im_classes:
            # extract detections of particular class (classes)
            class_mask = image_pred_*(image_pred_[:,-1] == classes).float().unsqueeze(1)
            class_mask_idx = torch.nonzerO(class_mask[:-2]).unsqueeze() # all except last 2 elements
            image_pred_class = image_pred_[class_mask_idx].view(-1,7) # (best fit) x 7

            # sort such that first element has max objectness confidence
            conf_sort_idx = torch.sort(image_pred_class[:,4], descending=True)[1]
            image_pred_class = image_pred_class[conf_sort_idx]
            
            n_detections = image_pred_class.size(0)

            ## non max suppression
            for det in range(n_detections):
                # get IOUs of all boxes that come after the one we are looking at in loop
                # use try except because bboxes may be removed from image_pred_class
                try:
                    # ious indexed by det with all bboxes having indices > det
                    ious = bbox_iou(image_pred_class[det].unsqueeze(0), image_pred_class[det+1:])
                except ValueError:
                    break

                except IndexError:
                    break
                
                # ever iteration, if any bboxes that have indices > det also have iou 
                # (with box indexed by det) larger than threshold nms_thresh, eliminate it

                # zero out all detections that have iou > threshold
                iou_mask = (ious < nms_conf).float().unsqueeze(1)
                image_pred_class[det+1:] *= iou_mask

                # remove non-zero entries
                non_zero_idx = torch.nonzero(image_pred_class[:,4]).squeeze()
                image_pred_class = image_pred_class[non_zero_idx].view(-1,7)

            # output tensor (n_true_preds_in_all_images x 8)
            # 8 attrs are:
            #   - idx of image in batch to which detection belongs to
            #   - 4 corner coords
            #   - objectness score
            #   - score of class with max confidence
            #   - idx of class with max confidence

            # do not init output tensor unless we have a detection to assign to
            # check write flag
            batch_idx = image_pred_class.new(image_pred_class.size(0), 1).fill_(idx)
            
            # repeat batch_id for as many detections of class det in image
            seq = batch_idx, image_pred_class

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
    """
    Returns dictionary which maps indeces of each class to string of its name.
    """
    fp = open(namesfile, "r")
    names = fp.read().split("\n")[:-1]
    return names

def norm_with_padding(img, in_dims):
    """
    Resizes image with unchanged aspect ratio using padding (128,128,128).
    """
    img_w, img_h = img.shape[1], img.shape[0]
    w, h = in_dims
    new_w = int(img_w*min(w/img_w, h/img_h))
    new_h = int(img_h*min(w/img_w, h/img_h))
    resized_image = cv2.resize(img, (new_h, new_w), interpolation=cv2.INTER_CUBIC)

    canvas = np.full((in_dims[1], in_dims[0], 3), 128)
    canvas[(h-new_h)//2:(h-new_h)//2 + new_h, (w-new_w)//2:(w-new_w)//2 + new_w, :] = resized_image

    return canvas

def prep_image(img, in_dims):
    """
    Prep image for inputting into neural network
    """
    img= cv2.resize(img, (in_dims, in_dims))
    img = img[:,:,::-1].transpose((2,0,1)).copy()
    img = torch.from_numpy(img).float().div(255.0).unsqueeze(0)
    
    return img
from __future__ import print_function, division
import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skimage import io, transform
import cv2

def filter_labels(raw_json):
    """
    Re-organises json file into dictionary, such that:
        - keys =    image names.
        - values =  list of dictionaries which contain key value information for each labelled 
                    object in frame.
    
    For later down the track (segmentation, lane detection), other labels will be let thru.
    -----------------------------------------------------------------------------------------
    The structure is as follows:

    data_labels = {
                    'im_name1.jpg' : [ 
                        {obj1_category: car, obj1_bboxes: [{x1:x1},{y1:y1},{x2:x2},{y2:y2}]},
                        {obj2_category: car, obj2_bboxes: [{x1:x1},{y1:y1},{x2:x2},{y2:y2}]},
                        ...
                    ],
                    'im_name2.jpg' : [ 
                        {obj1_category: car, obj1_bboxes: [{x1:x1},{y1:y1},{x2:x2},{y2:y2}]},
                        {obj2_category: car, obj2_bboxes: [{x1:x1},{y1:y1},{x2:x2},{y2:y2}]},
                        ...
                    ],
                }
    -----------------------------------------------------------------------------------------
    Input:
    =raw_json=      raw json file containing all labels corresponding to image dataset.

    Output:
    =data_labels=   dictionary with appropriate labels corresponding to image name.
    """
    raw_df = pd.read_json(raw_json)
    data_labels = {}
    for idx, label in enumerate(raw_df["labels"]):
        # skip nan labels
        if type(label) == float:
            continue
        objects = []
        obj = {}

        for object in label:
            #obj["id"] = object["id"] # object id probs not useful?
            obj["category"] = object["category"]
            obj["box2d"] = object["box2d"]
            objects.append(obj)
            obj = {}

        data_labels[str(raw_df["name"][idx])] = objects
    
    return data_labels

def load_classes(namesfile):
    """
    Loads file containing the unique classes of objects within the BDD100k dataset.
    """
    fp = open(namesfile, "r")
    names = fp.read().split("\n")
    return names

def corners_to_centre_dims(bbox):
    """
    Converts bbox attributes of form [x1, y1, x2, y2] to form [x_centre, y_centre, width, height].

    This form is used for training and feeding into network.
    """
    width = bbox[2]-bbox[0]
    height = bbox[3]-bbox[1]
    x_centre = bbox[0]+(width/2)
    y_centre = bbox[1]+(height/2)
    new_bbox = np.array([x_centre, y_centre, width, height])
    return new_bbox

def centre_dims_to_corners(bbox):
    """
    Converts bbox attributes of form [x_centre, y_centre, width, height] to form [x1, y1, x2, y2]. 
    
    Use on an array of bboxes: [[bbox_1], [bbox_2], ... [bbox_n]].

    This form is used for easily calculating 2 bbox's IoU.
    """
    if len(bbox.shape) > 1:
        x_c, y_c, w, h = bbox[:,0], bbox[:,1], bbox[:,2], bbox[:,3]
    else:
        x_c, y_c, w, h = bbox[0], bbox[1], bbox[2], bbox[3]
    x1, x2 = x_c-(w/2), x_c+(w/2)
    y1, y2 = y_c-(h/2), y_c+(h/2)
    
    if len(bbox.shape) > 1:
        x1 = np.expand_dims(x1, 1)
        x2 = np.expand_dims(x2, 1)
        y1 = np.expand_dims(y1, 1)
        y2 = np.expand_dims(y2, 1)

        new_bbox = np.concatenate((x1, y1, x2, y2), axis=1)
    else:
        new_bbox = np.array([x1, y1, x2, y2])

    return new_bbox

def bbox_anchorbox_iou(bbox, anchor_boxes):
    """
    Returns intersection over union of two bounding boxes.

    Performed on np arrays.

    Inputs:

    Outputs:
    =iou=   array with iou between bbox and chosen anchor boxes.
    """
    # get coords of bboxes
    b1_x1, b1_y1, b1_x2, b1_y2 = bbox[0], bbox[1], bbox[2], bbox[3]
    b2_x1, b2_y1, b2_x2, b2_y2 = anchor_boxes[:,0], anchor_boxes[:,1], anchor_boxes[:,2], anchor_boxes[:,3]

    # reshape so that we can vectorise operations
    x1_comparisons = np.concatenate((np.expand_dims(b1_x1.repeat(3),1), np.expand_dims(b2_x1, 1)), axis=1)
    x2_comparisons = np.concatenate((np.expand_dims(b1_x2.repeat(3),1), np.expand_dims(b2_x2, 1)), axis=1)
    y1_comparisons = np.concatenate((np.expand_dims(b1_y1.repeat(3),1), np.expand_dims(b2_y1, 1)), axis=1)
    y2_comparisons = np.concatenate((np.expand_dims(b1_y2.repeat(3),1), np.expand_dims(b2_y2, 1)), axis=1)

    # get coords of intersection
    intersect_x1 = np.max(x1_comparisons, axis=1)
    intersect_x2 = np.min(x2_comparisons, axis=1)
    intersect_y1 = np.max(y1_comparisons, axis=1)
    intersect_y2 = np.min(y2_comparisons, axis=1)

    # intersection area
    # clamp to > 0
    # this avoids areas being calculated for boxes with zero intersect
    intersect_area = np.clip(intersect_x2 - intersect_x1, a_min=0, a_max=1e10)*np.clip(intersect_y2 - intersect_y1, a_min=0, a_max=1e10)

    # union area
    b1_area = (b1_x2 - b1_x1)*(b1_y2 - b1_y1)
    b2_area = (b2_x2 - b2_x1)*(b2_y2 - b2_y1)
    union_area = b1_area + b2_area - intersect_area

    # compute iou
    iou = intersect_area/union_area

    return iou    

def draw_bbox(image, bboxes):
    """
    Draw all bounding boxes over image.
    
    Inputs:
    =image=     image to be shown with bounding boxes.
    =bboxes=    bounding box coordinates as list.
    
    Outputs:
    =plt.plot=  image from dataset with bounding boxes drawn over it.
    """
    # display current image
    #print(image.shape)
    plt.imshow(image)

    for obj in bboxes:
        # label coords appropriately
        x_min, y_min, x_max, y_max = obj
        # arrange coords in order such that we can draw the box in full
        x_points = [x_min, x_min, x_max, x_max, x_min]
        y_points = [y_min, y_max, y_max, y_min, y_min]

        plt.plot(x_points, y_points, 'r-')




class DetectionDataset(Dataset):
    """
    Object detection dataset with bounding box, objectness score and class data.
    
    """
    def __init__(self, label_dict, root_dir, classes_file, grid_sizes, anchors, transform=None):
        """
        Args:
        =label_dict=    file location of bdd100k detection labels (.json).
        =root_dir=      directory with all images.
        =classes_file=  file location of unique classes within bdd100k (.names).
        =grid_sizes=    list of different grid sizes that detection will be performed on per image. 
                            Expected size: (1, grid_sizes)
        =anchors=       np array of anchors to be used per grid size. 
                            Expected size: (n_grid_sizes, n_anchors, 2) - 2 being size of (anchor_w, anchor_h)
        =transform=     optional transform to be applied on a sample.
        """
        self.labels = filter_labels(label_dict)
        self.root_dir = root_dir
        self.classes = np.array(load_classes(classes_file)) # loads a array of strings
        self.grid_sizes = np.array(grid_sizes)
        self.anchors = anchors
        self.transform = transform
    
    def __len__(self):
        return(len(self.labels))
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        ## collect all image names
        img_name = list(self.labels.keys())[idx]
        image = io.imread(os.path.join(self.root_dir, img_name))
        
        ## collect all labels per image
        category_labels = []
        for image_idx in range(len(self.labels)):
            curr_im_cat_labels = [obj["category"] for obj in list(self.labels.values())[image_idx]]
            category_labels.append(curr_im_cat_labels)
        categories = category_labels[idx]
        #categories = np.array(categories)
        
        ## collect all bounding box coords per image
        # first create list of list of dictionaries exclusively full of bbox coordinates
        box2d_labels = []
        for image_idx in range(len(self.labels)):
            curr_im_box_labels = [obj["box2d"] for obj in list(self.labels.values())[image_idx]]
            box2d_labels.append(curr_im_box_labels)

        # now convert the list of list of dictionaries into list of list of list
        bbox_labels = []
        for image_labels in box2d_labels: # iterate through each image label set
            cur_image = []
            for bbox in image_labels: # iterate through each set of coords in image label set
                coords = []
                for _, bbox_value in bbox.items(): # iterate through each indiv coord and pull out values
                    # fill list with coords [x1, y1, x2, y2]
                    coords.append(bbox_value)
                # add the [x1, y1, x2, y2] to current image list
                cur_image.append(coords)

            # add the current image list to full list of images
            bbox_labels.append(cur_image)
        bboxes = bbox_labels[idx]
        bboxes = np.array(bboxes) # size (n_bboxes, 4) -> [x1, y1, x2, y2]

        ## form labels to compare with output of forward pass
        # loop through each grid size
        grid_sizes = self.grid_sizes
        n_classes = self.classes.shape[0]
        anchors = self.anchors
        n_anchors = anchors.shape[1]

        img_w, img_h = image.shape[0], image.shape[1]

        write = 0 # flag for knowing which grid cell size we are up to
        
        ## Loop through and create labels for each grid within grid_sizes
        for g, grid_size in enumerate(grid_sizes):
            # create meshgrid to fit bbox centres
            grid = np.arange(1, grid_size+1)
            a,b = np.meshgrid(grid,grid)
            stride_x, stride_y = (img_w//grid_size), (img_h//grid_size)

            # initiate empty grid cells
            labels = np.zeros(shape=(grid_size, grid_size, n_anchors, 5+n_classes))

            # calculate grid cell centres
            a *= stride_x 
            b *= stride_y 
            a = a - stride_x/2
            b = b - stride_y/2

            # loop through each labelled obj in image
            for i, bbox in enumerate(bboxes):
                # reorganise bbox attrs from [x1, y1, x2, y2] to [x_c, y_c, w, h]
                new_bbox = corners_to_centre_dims(bbox)

                # convert grid to pairs of grid centre coords
                grid_coords = np.concatenate((b.reshape(-1,1), a.reshape(-1,1)), axis=1)
                dists = np.sum(np.square(grid_coords-new_bbox[:2]), axis=1).reshape(grid_size,grid_size)

                # find coord of grid cell that bbox centre belong to
                w_idx, h_idx = np.unravel_index(dists.argmin(), dists.shape)

                # make class type label
                class_array = (self.classes == categories[i]).astype(int) # not working 
                #print(class_array)

                # complete grid cell label for detected object!
                obj_cell_label = np.concatenate((new_bbox, np.array([1]), class_array))

                ## now find which of the 3 anchor boxes in corresponding grid size best describes bbox label 
                
                # repeat bbox label's centre coords and then append these to the 3 anchor box widths, heights
                bbox_tile = np.tile(new_bbox[:2], [3,1])
                anchor_boxes = np.concatenate((bbox_tile, anchors[g]), 1) # reference the g'th set of anchor dims corresponding to gth grid_size

                # convert these abox attrs from [x_c, y_c, w, h] -> [x1, y1, x2, y2]
                anchor_boxes = centre_dims_to_corners(anchor_boxes)

                # calculate IoUs between current bbox and the set of aboxes
                # then find idx of abox that best describes the current bbox (has highest IoU)
                anchorbox_ious = bbox_anchorbox_iou(bbox, anchor_boxes)
                anchorbox_idx = np.argmax(anchorbox_ious)

                # fill in object label
                # recall labels size (grid_size, grid_size, n_anchors, 5+n_classes)
                labels[w_idx][h_idx][anchorbox_idx] = obj_cell_label

            # reshape to size (grid_size*grid_size*n_anchors, 5+n_classes)
            # therefore, each label with an obj is now store at idx:
            #   (w_idx*grid_size + h_idx)*n_anchors + anchorbox_idx
            labels = labels.reshape(-1,5+n_classes)
            #print(labels.shape)

            # now create the full label with all grid sizes
            if not write:
                all_labels = labels
                write = 1
            else:
                all_labels = np.concatenate((all_labels, labels), axis=0)

        sample = {"image": image, "labels": all_labels} # can also return "bbox": bboxes, "category": categories,

        if self.transform:
            sample = self.transform(sample)

        return sample

## transforms
class Pad(object):
    """
    Add padding to image

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple)) # make sure output size is EITHER int or tuple
        self.output_size = output_size

    def __call__(self, sample):
        image, labels = sample["image"], sample["labels"]

        img_w, img_h = image.shape[1], image.shape[0]
        w, h = self.output_size

        # calculate new width and height
        new_w = int(img_w * min(w/img_w, h/img_h))
        new_h = int(img_h * min(w/img_w, h/img_h))
        resized_image = cv2.resize(image, (new_w,new_h), interpolation = cv2.INTER_CUBIC)
        
        canvas = np.full((self.output_size[1], self.output_size[0], 3), 128)

        canvas[(h-new_h)//2:(h-new_h)//2 + new_h,(w-new_w)//2:(w-new_w)//2 + new_w,  :] = resized_image

        return {"image": canvas, "labels": labels}
    
class Rescale(object):
    """
    Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple)) # make sure output size is EITHER int or tuple
        self.output_size = output_size

    def __call__(self, sample):
        image, categories, bboxes = sample["image"], sample["categories"], sample["bboxes"]

        h, w = image.shape[:2] # image.shape = (x, y, channels)
        if isinstance(self.output_size, int): # the int specifies the smaller dimension, other dim scales based on it
            if h > w:
                new_h, new_w = self.output_size*h/w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size*w/h
        else: # if tuple then completely specified dims
            new_h, new_w = self.output_size
        
        new_h, new_w = int(new_h), int(new_w)

        image = transform.resize(image, (new_h, new_w))

        # h and w are swapped for landmarks because for images,
        # x and y axes are axis 1 and 0 respectively
        # broadcast (7, 4) * (1, 4)
        bboxes = bboxes * [new_w/w, new_h/h, new_w/w, new_h/h]

        return {"image": image, "categories": categories, "bboxes": bboxes}

class Normalise(object):
    """
    Normalise pixel values.
    """
    def __init__(self, mean, std):
        # ensure input mean and std are integers or floats
        assert isinstance(mean, (int, float))
        assert isinstance(std, (int, float))
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        image, categories, bboxes = sample["image"], sample["categories"], sample["bboxes"]

        image = (image - self.mean)/self.std

        return {"image": image, "categories": categories, "bboxes": bboxes} 

class ToTensor(object):
    """
    Convert ndarrays in sample to Tensors.
    """

    def __call__(self, sample):
        image, categories, bboxes = sample["image"], sample["categories"], sample["bboxes"]

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C x H x W
        image = image.transpose((2, 0, 1))
        image = torch.from_numpy(image)
        
        # bboxes are simply H x W and axes stay the same
        bboxes = torch.from_numpy(bboxes)
        
        return {"image": image, "categories": categories, "bboxes": bboxes}



# ## filter labels
# filtered_labels = filter_labels("det_train_shortened.json")

# # ## load custom dataset
# # testing = DetectionDataset(
# #     label_dict=filtered_labels,
# #     root_dir="./images/",
# # )
# # fig = plt.figure()

# # # check shapes
# # print("# image_shape    n_objects")
# # for i in range(len(testing)):
# #     sample = testing[i]

# #     print(i, sample["image"].shape, len(sample["categories"]))

# #     ax = plt.subplot(1, 2, i+1)
# #     ax.set_title('Sample #{}'.format(i))
# #     ax.axis('off')
# #     draw_bbox(**sample)

# ## data loader
# transformed_train_data = DetectionDataset(
#     label_dict=filtered_labels,
#     root_dir='images/',
#     transform=transforms.Compose([
#         transforms.Normalize(
#         mean=[0.485, 0.456, 0.406],
#         std=[0.229, 0.224, 0.255]),
#         Rescale(256),
#         ToTensor()
#     ])
# )

# train_loader = DataLoader(
#     transformed_train_data,
#     batch_size=2,
#     shuffle=True,
#     num_workers=0
# )
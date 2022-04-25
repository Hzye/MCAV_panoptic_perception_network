from __future__ import print_function, division
import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skimage import io, transform

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

def draw_bbox(image, categories, bboxes):
    """
    Draw all bounding boxes over image.
    
    Inputs:
    =image=     image to be shown with bounding boxes.
    =bboxes=    bounding box coordinates as list.
    
    Outputs:
    =plt.plot=  image from dataset with bounding boxes drawn over it.
    """
    # display current image
    print(image.shape)
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
    Object detection dataset with bounding box and category data.
    
    """
    def __init__(self, label_dict, root_dir, transform=None):
        """
        Args:
        =label_dict=    dictionary with key (image name) and value (object categories and bboxes).
        =root_dir=      directory with all images.
        =transform=     optional transform to be applied on a sample.
        """
        self.labels = label_dict
        self.root_dir = root_dir
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
        bboxes = np.array(bboxes)

        sample = {"image": image, "categories": categories, "bboxes": bboxes}

        if self.transform:
            sample = self.transform(sample)

        return sample

## transforms
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
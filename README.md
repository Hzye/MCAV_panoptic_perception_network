# Panoptic Perception Network

Researching and implementing an end-to-end network which comprises of object detection, lane detection and image segmentation. Is it more beneficial to have all three tasks in one network?

---
## Setup
* For testing the network, download the YOLOv3 darknet53 pretrained weights [here](https://pjreddie.com/media/files/yolov3.weights).
* Save the weights into a new folder 'weights' within this cloned repo.
---
## Repository Composition

The files and what they do.

### dataloader.py
Prepares the dataset for the network to train and test on.

1. Reads in images and labels from BDD100k dataset.
2. ```filter_labels()```: Filters labels down to just the ones we want to use (e.g. bounding box coordinates and object labels)
3. ```Rescale()```, ```ToTensor()```: Transforms images.
4. ```DetectionDataset()```: Collects transformed images and filtered labels into a custom dataset.
5. Loads the custom dataset using a dataloader.

### model.py
Constructing the network.

1. ```parse_cfg()```: Parses the config file within 'cfg' folder to read model parameters and architecture.
2. ```create_modules()```: Creates modules for each type of layer (conv, upsample, route, shortcut, yolo).
3. ```Net()```: Loads pre-trained weights using ```load_weights()``` for testing purposes, and computes forward passes of the network using ```forward()```.

### utils.py
Helper functions.

### detector_image.py
Runs the detection network on images.

1. ```arg_parse()```: Creates arguments to pass into detector_image from terminal.
2. Runs network over batches of images.
3. Draws bounding on input images and saves them to new folder 'dets'.

### detector_video.py
Runs the detection network on videos or webcam for demonstration purposes.

1. ```arg_parse()```: Creates arguments to pass into detector_image from terminal.
2. Runs network over videos or webcam frames.
3. Outputs live bounding box predictions.

### Any Notebook (.ipynb)
Used for testing and debugging the specific python file that the notebook is named after. 

For example: model_testing.ipynb is used to test bits and pieces of the model.py file.
---
### How to run
1. Make sure you have downloaded the pretrained weights from above and have saved them into a new folder 'weights'.
2. Open terminal, and cd to this project's repository.

#### For image detection
3. Run: ```python detector_image.py --images images --dets dets```. This will output run times of various sections of the detection process and save output images to a new 'dets' folder.

#### For live webcam detection
3. Run: ```python detector_video.py```. This will output two live video feeds, one of the raw webcam input and one with detection bounding boxes.
4. Press 'q' on the keyboard to end detection.



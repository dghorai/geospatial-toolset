Object Detection
==================

Object detection consists of two sub-tasks: 
- localization, which is determining the location of an object in an image, and 
- classification, which is assigning a class to that object

[Find Most popular metrics used to evaluate object detection algorithms](https://github.com/rafaelpadilla/Object-Detection-Metrics).


Object Detection Competitions
==============================
1) COCO
2) Pascal VOC (Visual Object Classifier) 07/12
3) ImageNet

Bounding Box
=============

Create a rectangle. Take two coordinates or center coordinates with width/height of the box to construct bouding box. Either considereding top-left corner and bottom-right corner coordinates or consideirng center point and with width/height of the box. Loss function is being calculated between predicted coordinates and actual coordinates of these. Bounding boxes are used to `localize objects` within an image and are commonly annotated in object detection datasets to provide ground-truth information about the object’s location.

- Bounding boxes represent the actual regions in an image that enclose objects of interest.
- Bounding boxes are provided as annotations in object detection datasets, representing ground-truth object locations.
- The object's true boundaries are the basis for bounding boxes, which are fixed.
- Bounding boxes are used for object localization, and they are compared to predicted boxes to measure accuracy.

Anchor Boxes
=============

[Anchor boxes](https://www.thinkautonomous.ai/blog/anchor-boxes/) are pre-defined bounding boxes with specific sizes, aspect ratios, and positions that are used as reference templates during object detection. Anchor boxes are placed at various positions across an image to capture objects of different scales and shapes. The locations and shapes of objects relative to reference boxes are predicted using anchor boxes during training and inference. 

- Anchor boxes are reference bounding boxes used to predict object locations and shapes during object detection.
- During model training and inference, anchor boxes are used as templates. The model predicts adjustments to anchor boxes to match the actual locations of the objects.
- Different scales and aspect ratios can be used to handle various object sizes and shapes.
- [The anchor boxes are used to predict object locations and shapes](https://towardsdatascience.com/training-yolo-select-anchor-boxes-like-this-3226cb8d7f0b).

Bounding Box Regression
=======================

Bounding Box regression is a metric for measuring how well predicted bounding boxes captures objects. Determine quantities. 

This is done by calculating the mean-squared error (MSE) between the coordinates of predicted bounding boxes and the ground truth bounding boxes.

Example:

Actual:
(x, y) = (350, 350)
l = 50
h = 50

Predicted:
(x', y') = (310, 310)
l' = 10
h' = 10

Loss Function to use:
1) L1 loss 
	- L1 loss is also known as mean absolute loss or mean absolute error.
	- MAE = (1/n) * Σ|yᵢ - ȳ|
2) L2 loss 
	- The Mean Square Error(MSE) or L2 loss is a loss function that quantifies the magnitude of the error between a machine learning algorithm prediction and an actual output by taking the average of the squared difference between the predictions and the target values.
	- MSE = (1/n) * Σ(yᵢ - ȳ)²
3) Smooth-L1 loss
	- This version of loss depends on the value based on the value of the beta threshold. When the value is less than threshold It’s less sensitive to outliers than the mean squared error loss. But at the same time when the value is greater, it acts the same as the L1 loss which makes it more sensitive to the outliers.

Intersection Over Union (IOU)
=============================

Intersection over Union (IoU), also known as the Jaccard index, is the most popular evaluation metric for tasks such as segmentation, object detection and tracking. 

IOU evaluates the overlap between two bounding boxes. It requires a ground truth bounding box and a predicted bounding box. By applying IOU we can tell if a detection is valid (True +ve) or not (False +ve).

IOU = (area of overlap)/(area of union)

Precision and Recall
======================

Precision:
- Precision is the ability of a model to identify only the relevant objects. It is the percentage of correct +ve predictions and is given by the following eq.
- Precision = TP/(TP + FP) = TP/(all detection)
- TP -> True Positive: A correct detection
- FP -> False Positive: A wrong detection
- FN -> False Negative: A ground truth not detected
- TN -> True Negative: Not detected correctly  

Recall: 
- It is the ability of a model to find all the relevant cases (all ground truth bounding boxes). It is the percentage of true positive detected among all relevant ground truths and is given by:
- Recall = TP/(TP + FN) = TP/(all ground truths)

Average Precision (AP)
========================

The average precision (AP) is a way to summarize the precision-recall curve into a single value representing the average of all precisions. AP summarizes a precision-recall curve as the weighted mean of precisions achieved at each threshold, with the increase in recall from the previous threshold used as the weight:

- AP = Σ(Rn - Rn-1)*Pn
- Pn -> Precision at the n-th threshold
- Rn -> Recall at the n-th threshold

Mean Average Precision (mAP)
=============================


Object Detection Architectures
=================================
1) OD Family (3 family)
	- RCNN Family (Slower and more accurate)
		- RCNN
		- Fast RCNN
		- Faster RCNN
		- Mask RCNN
	- Single Shot Detector (SSD) Family (faster and less accurate)
		- YOLO
		- SSD
	- CenterNet Family (slow and better accuracy)
		- CENTERNET

RCNN
========

Paper: https://arxiv.org/pdf/1311.2524.pdf

R-CNN -> Region Proposals + CNN

Steps of RCNN/Architecture of RCNN:
1) Takes an input image
2) Extract around 2000 bottom-up region proposal using selective serach method
3) Warped regions
4) Computes features for each proposal using a large CNN (AlexNet)
5) 4.a: Classifies each region using class-specific linear SVMs (from CNN features).
6) 4.b: Bounding box prediction using linear regression (from CNN features) 

Stages of RCNN:
1) The first module generates category-independent region proposals.
2) The second module is a large convolutional neural netork that extracts a fixed-length feature vector from each region.
3) The third module is a set of class-specific linear SVMs.


Methods of Region Proposals:
1) Objectness
2) Selective search (selected for RCNN)
3) Category-independent object proposals
4) Constrained parametric min-cuts (CPMC)
5) Multi-scale combinational grouping
6) Ciresan et al.

Feature Extraction:
1) ~2000 region proposal extracted from selective search method
2) Extracted 4096-dimensional feature vector from each region proposal using Caffe implementation of the CNN described by Krizhevsky et al.
3) Features are computed by forward propagating a mean-subtracted 227 × 227 RGB image through five convolutional layers and two fully connected layers.


Fast RCNN
===============

Paper: https://arxiv.org/pdf/1504.08083.pdf

Source Code: https://github.com/rbgirshick/fast-rcnn

Architcture:
1) Input Image -> (Selective Search+Edge Box)
2) Input Image and (Selective Search+Edge Box) -> AlexNet -> Features -> RoI Pooling
-> FC -> FC
3) 2.a) Softmax (for classification object)
4) 2.b) Bounding Box Regression (L2 Loss and SmoothL1 Loss) for bounding box prediction of the object


Faster RCNN
=================

Paper: https://arxiv.org/pdf/1506.01497.pdf

Source Code: https://github.com/rbgirshick/py-faster-rcnn

Architcture:
1) Input Image -> VGG -> Features <-> RPN -> RoI Pooling -> FC -> FC -> a) Fine Tuning Softmax using Log Loss and b) Bounding Box Regression
2) RPN (Regional Proposal Network) -> 3x3 -> a) 1x1 for classification (FG/BG) and b) 1x1 for Region Bounding Box


YOLO
==============

Paper: https://arxiv.org/pdf/1506.02640.pdf

Website: https://pjreddie.com/darknet/yolo/

Architcture:
YOLO architecture is similar to GoogleNet. As illustrated below, it has overall 24 convolutional layers, four max-pooling layers, and two fully connected layers.

YOLO Flow:
1) Resizes the input image into 448x448 before going through the convolutional network.
2) A 1x1 convolution is first applied to reduce the number of channels, which is then followed by a 3x3 convolution to generate a cuboidal output.
3) The activation function under the hood is ReLU, except for the final layer, which uses a linear activation function.
4) Some additional techniques, such as batch normalization and dropout, respectively regularize the model and prevent it from overfitting.

The algorithm works based on the following four approaches:
1) Residual blocks
This first step starts by dividing the original image (A) into NxN grid cells of equal shape. Each cell in the grid is responsible for localizing and predicting the class of the object that it covers, along with the probability/confidence value. 

2) Bounding box regression
The next step is to determine the bounding boxes which correspond to rectangles highlighting all the objects in the image. We can have as many bounding boxes as there are objects within a given image. 
YOLO determines the attributes of these bounding boxes using a single regression module in the following format, where Y is the final vector representation for each bounding box. 
Y = [pc, bx, by, bh, bw, c1, c2]

3) Intersection Over Unions or IOU
4) Non-Maximum Suppression. 

Steps to follow:
1) Input image
2) Breaking an image into a grid
3) Each section of the grid is classified and localized
4) Predicts where to place bounding boxes based on regression-based algorithms

Requirements:
1) Yolov5 Repository
2) Python 3.8 or later
3) PyTorch (It is the most popular machine learning frameworks to define models, and perform training)
4) CUDA (It is NVIDIA's parallel computing platform for their GPUs)

Step-1: Install YOLOv5. Get YOLOv5 from here: 
- i) manual download https://github.com/ultralytics/yolov5
- ii) git clone https://github.com/ultralytics/yolov5.git

Step-2: Install Python. Get this from [here](https://www.python.org/downloads/)

Step-3: Install CUDA. Get this from [here](https://developer.nvidia.com/cuda-downloads)

Step-4: Install PyTorch as a Python module. Get the correct pip install version of this by selecting the preferences available at [here](https://pytorch.org/get-started/locally/)

Step-5: Install additional modules for YOLOv5 available in requirements.txt inside of this folder
- pip install -r requirements.txt
- If visual studio error occure then get it from [here](https://visualstudio.microsoft.com/downloads/) to install
- After that install PyCOCOTools:
    - pip install git+https://github.com/philferriere/cocoapi.git#subdirectory=PythonAPI
- Then further run this:
    - pip install -r requirements.txt
    - This is just for good measure.

Further details can be found from this links:
1) [YOLO Object Detection Explained](https://www.datacamp.com/blog/yolo-object-detection-explained)
2) [What Is YOLO Algorithm?](https://www.baeldung.com/cs/yolo-algorithm)
3) [YOLOv5 Object Detection on Windows (Step-By-Step Tutorial)](https://wandb.ai/onlineinference/YOLO/reports/YOLOv5-Object-Detection-on-Windows-Step-By-Step-Tutorial---VmlldzoxMDQwNzk4)
4) [Comparing YOLOv3, YOLOv4 and YOLOv5 for Autonomous Landing Spot Detection in Faulty UAVs](https://www.mdpi.com/1424-8220/22/2/464)


YOLO Practical
==============

## 1) Inference using pretrained YOLOv5 model with Google Colab
- Open Google Colab/Jupyter Notebook
- Cloning Github Repository (! git clone https://github.com/ultralytics/yolov5)
- Installling dependencies (! pip install -r requirements.txt)
- Dependency Checks

```
import torch

from IPython.display import Image, clear_output  # to display images

clear_output()

print(
    f"Setup complete. Using torch \
    {torch.__version__} \
    ( \
        {torch.cuda.get_device_properties(0).name if torch.cuda.is_available() else 'CPU'} \
    )" \
)

# inferencing with YoloV5

# inference with default model weights

! python detect.py --img 640 --conf 0.4 --source /full/path/of/the/test-image-folder

# inference with different model weights (see 'Pretrained Checkpoints' for model weights and pixel size/resolution at: https://github.com/ultralytics/yolov5)

! python detect.py --img 640 --conf 0.4 --weights yolov5m.pt --source /full/path/of/the/test image folder
```

## 2) Inference using pretrained YOLOv5 model on Local System
```
## Using Webcamp Data
#=====================
# in cmd terminal, cd to YOLOv5 directory: > cd tolov5
# type this line of code: python detect.py --source 0
# Note: If there is multiple webcams or applications that spoof them we may need to change this number. If there is only one, then source will be 0 as in the example code.
# To stop taking this simply click inside the open window.

## Using Image Data
#==================
# If you want to detect objects in a pre-existing image, simply place the image in a folder within the YOLO directory. 
# Opening up the YOLO Console again, we enter:
# python detect.py --source images/google_image.jpg
# Go to the location it's stored and see the new image with resultant bounding boxes

## Using YouTube Video Data
#==========================
# Install few more modules to create bounding boxes or just detect objects on a YouTube video
# pip install pafy
# pip install youtube_dl
# Get the youtube video id from video link (video id of this https://www.youtube.com/watch?v=jNQXAC9IVRw link is 'jNQXAC9IVRw')
# Construct source link of the video id: https://youtu.be/jNQXAC9IVRw
# Now in cmd terminal type this line to run: python detect.py --source https://youtu.be/jNQXAC9IVRw
# press ctrl+C in terminal to stop
# example: python detect.py --source https://youtu.be/AmarqUQzqZg

```

## 3) Custom training and inference of YOLO model

- prepare dataset

<pre>
## collect/download images from true sources

## download following tools for annotation and train-test split
# 1) https://github.com/HumanSignal/labelImg
# 2) https://github.com/ivangrov/ModifiedOpenLabelling

## for annotation use anyone of them; for this training, labelImg is consideirng
# install depandancy
# conda install pyqt=5
# conda install -c anaconda lxml
# pyrcc5 -o libs/resources.py resources.qrc
# run python labelImg.py
# it will open a GUI for image annotation
# Once it is opened, change the annotation save format from PascalVOC/CreateML to YOLO
# In View menu, tick Auto Save Mode
# Draw bounding boxes and assign class name and then save

## split the data into training and validation

# execute the train_test_split.py from 'ModifiedOpenLabelling' library by typing into the command prompt:

python train_test_split.py

# before execute the above commapnd, configure the input file paths in train_test_split.py file
# 1) move images folder inside ModifiedOpenLabelling folder
# 2) move classes.txt file in the same folder and rename it class_list.txt
# 3) create a folder called bbox_txt and move all annotation text files inside of this and then move this folder to ModifiedOpenLabelling folder
# 4) now run the above command
# Copy the custom_dataset folder from ModifiedOpenLabelling into Yolov5 folder
</pre>

- Open Google Colab

<pre>
%cd /content/drive/MyDrive/YOLOV5_CUSTOM_TRAINING
</pre>

- configuring development environment

<pre>
!git clone https://github.com/ultralytics/yolov5  # clone
%cd yolov5
!git reset --hard 68211f72c99915a15855f7b99bf5d93f5631330f (optional)
!pip install -qr requirements.txt  # install dependencies (ignore errors)
</pre>

- some imports

<pre>
import torch
from IPython.display import Image, clear_output  # to display images
from utils.google_utils import gdrive_download  # to download models/datasets
</pre>

- unzipping datasets

<pre>
%cd /content/drive/MyDrive/YOLOV5_CUSTOM_TRAINING/
!unzip train.zip
!unzip test.zip
</pre>

- create data.yaml file that contains the path of training and validation images and also the classes

<pre>
import yaml
config = {'train': 'custom_dataset/train/',
         'val': 'custom_dataset/valid/',
         'nc': 6,
         'names': ['dog', 'person', 'cat', 'laptop', 'bottol', 'animal']}

with open("custom_dataset.yaml", "w") as file:
   yaml.dump(config, file, default_flow_style=False)
</pre>

- checking the no. of classes in our datasets (open data.yaml file)

<pre>
%cat data.yaml
</pre>

- get the no. of classes from data.yaml file

<pre>
import yaml
with open("data.yaml", 'r') as stream:
    num_classes = str(yaml.safe_load(stream)['nc'])
</pre>

- see the existing model configuration of yolov5

<pre>
%cat /content/drive/MyDrive/YOLOV5_CUSTOM_TRAINING/yolov5/models/yolov5s.yaml
</pre>

- customize iPython writefile so we can write variables

<pre>
from IPython.core.magic import register_line_cell_magic

@register_line_cell_magic
def writetemplate(line, cell):
    with open(line, 'w') as f:
        f.write(cell.format(**globals()))
</pre>

- configure model for our dataset (edit 'model' yaml file and save with new name)

<pre>
# example:

%%writetemplate /content/drive/MyDrive/YOLOV5_CUSTOM_TRAINING/yolov5/models/custom_yolov5s.yaml

# parameters
nc: {num_classes}  # number of classes
depth_multiple: 0.33  # model depth multiple
width_multiple: 0.50  # layer channel multiple

# anchors
anchors:
  - [10,13, 16,30, 33,23]  # P3/8
  - [30,61, 62,45, 59,119]  # P4/16
  - [116,90, 156,198, 373,326]  # P5/32

# YOLOv5 backbone
backbone:
  # [from, number, module, args]
  [[-1, 1, Focus, [64, 3]],  # 0-P1/2
   [-1, 1, Conv, [128, 3, 2]],  # 1-P2/4
   [-1, 3, BottleneckCSP, [128]],
   [-1, 1, Conv, [256, 3, 2]],  # 3-P3/8
   [-1, 9, BottleneckCSP, [256]],
   [-1, 1, Conv, [512, 3, 2]],  # 5-P4/16
   [-1, 9, BottleneckCSP, [512]],
   [-1, 1, Conv, [1024, 3, 2]],  # 7-P5/32
   [-1, 1, SPP, [1024, [5, 9, 13]]],
   [-1, 3, BottleneckCSP, [1024, False]],  # 9
  ]

# YOLOv5 head
head:
  [[-1, 1, Conv, [512, 1, 1]],
   [-1, 1, nn.Upsample, [None, 2, 'nearest']],
   [[-1, 6], 1, Concat, [1]],  # cat backbone P4
   [-1, 3, BottleneckCSP, [512, False]],  # 13

   [-1, 1, Conv, [256, 1, 1]],
   [-1, 1, nn.Upsample, [None, 2, 'nearest']],
   [[-1, 4], 1, Concat, [1]],  # cat backbone P3
   [-1, 3, BottleneckCSP, [256, False]],  # 17 (P3/8-small)

   [-1, 1, Conv, [256, 3, 2]],
   [[-1, 14], 1, Concat, [1]],  # cat head P4
   [-1, 3, BottleneckCSP, [512, False]],  # 20 (P4/16-medium)

   [-1, 1, Conv, [512, 3, 2]],
   [[-1, 10], 1, Concat, [1]],  # cat head P5
   [-1, 3, BottleneckCSP, [1024, False]],  # 23 (P5/32-large)

   [[17, 20, 23], 1, Detect, [nc, anchors]],  # Detect(P3, P4, P5)
  ]
</pre>

- train.py code can edit based on need

- start training

<pre>
# train yolov5s on custom data for 100 epochs
# time its performance
%%time
%cd /content/drive/MyDrive/YOLOV5_CUSTOM_TRAINING/yolov5
!python train.py --img 640 --batch 16 --epochs 100 --data '/content/drive/MyDrive/YOLOV5_CUSTOM_TRAINING/data.yaml' --cfg /content/drive/MyDrive/YOLOV5_CUSTOM_TRAINING/yolov5/models/custom_yolov5s.yaml --weights '' --name yolov5s_results  --cache
</pre>

- view result

<pre>
# To see the results for the training at localhost:6006 in your browser using tensorboard, 
# run this command in another terminal tab

tensorboard --logdir=runs
</pre>

- inferencing with our custom trained model

<pre>
%cd /content/drive/MyDrive/YOLOV5_CUSTOM_TRAINING/yolov5
!python detect.py --weights /content/drive/MyDrive/YOLOV5_CUSTOM_TRAINING/yolov5/runs/train/yolov5s_results2/weights/best.pt --img 640 --conf 0.4 --source ../test/images
</pre>


SSD
===========

Paper: https://arxiv.org/pdf/1512.02325.pdf

Source Code: https://github.com/weiliu89/caffe/tree/ssd

Architcture:
1) A Backbone Model -> A pre-trained image classification network serves as the backbone model's feature extractor. Usually, this is a network similar to ResNet that was trained on ImageNet after the last fully connected classification layer was eliminated. 
2) SSD Head -> The outputs are interpreted as bounding boxes and classes of objects in the spatial location of the final layer activations. The SSD head is simply one or more convolutional layers added to this backbone.
3) SSD uses a grid to split the image, with each grid cell in charge of identifying objects in that area.
4) In SSD, multiple anchor/prior boxes can be assigned to each grid cell. Each of these pre-defined anchor boxes is in charge of a specific size and shape inside a grid cell. 
5) During training, SSD employs a matching phase to align the bounding boxes of each ground truth object in an image with the relevant anchor box. 
6) Predicting the class and location of an object is essentially the responsibility of the anchor box that overlaps with it the most.
7) This can be accommodated by the anchor boxes' pre-defined aspect ratios thanks to the SSD architecture. The various aspect ratios of the anchor boxes associated with each grid cell at each zoom/scale level can be specified using the ratios parameter.
8) The amount that the anchor boxes need to be scaled up or down in relation to each grid cell is specified by the zooms parameter.

Further details can be found from this links:
1) [How single-shot detector (SSD) works?](https://developers.arcgis.com/python/guide/how-ssd-works/)
2) [Single Shot Detectors (SSDs)](https://www.baeldung.com/cs/ssd)

SSD Flow:
1) Process the image
2) Pass the image through model
3) Apply anchor boxes
4) Run the classifier
5) Non-maximal suppression
6) Return the bounding boxes and class probabilities
7) Done

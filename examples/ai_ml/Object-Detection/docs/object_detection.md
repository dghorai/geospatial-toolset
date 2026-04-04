Object Detection
=================
1)	Object classification - > classify an image
2)	Object localization -> find bounding box of different objects
3)	Object detection -> find the localized object type
4)	Object recognition -> recognized the different object of same type 

Object detection = object classification + object localization

Approach of Object Detection:
=============================
1) General approach/classical approach/basic approach/Naive approach:
	-> Sliding window approach
	-> start a kernel/window from top-left corner on the image to find the object’s bounding box 
	-> this approach also called brute-force approach
	-> Drawback of this algorithm:
			Variable box/window size
			Computational expensive
			Having overlapping issue
			The process is very slow
2) R-CNN and its variants
3) YOLO (You Only Look Once) and its variants
4) SSD (Single Shot multi-box Detector) and its variants

Models:
==========
All available models can be obtained from here: https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_detection_zoo.md
-> select best model and then right click on that to copy the model link and paste to jupyter-notebook to download

CV Models (Detectron): https://github.com/facebookresearch/detectron2/blob/main/MODEL_ZOO.md

Training/Prediction with Pre-trained Model -> TFOD2, Detectron2, YOLO

Terminology
==============
Key terminology of object detection model:
1)	Region of Interest (RoI) -> Region Proposal
2)	Feature extraction or network prediction
3)	Non-maximum suppression (NMS)
4)	IOU -> Intersection over Union
5)	FPS -> Frame per second
6)	mAP -> mean average precision 

CNN Summary
================
![alt text](/static/image.png)

Non-ML OD Approach
==================
1) https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_imgproc/py_template_matching/py_template_matching.html
2) https://medium.com/@iTrendTV/object-detection-without-machine-learning-aed3c5b668f3#:~:text=When%20objects%20are%20of%20similar,image%20in%20a%20larger%20imag


End-to-End ML Process
==================
An end-to-end machine learning platform: https://www.tensorflow.org/

1) Prepare Data
2) Build ML Models
3) Deploy Models
4) Implement MLOps

Pretrained Deep Learning Models
==============================
https://www.kaggle.com/models

References
===========
1. https://colab.research.google.com/github/hemantha-kumara/tensorflow-objectdetection/blob/master/Object_detection_pets.ipynb#scrollTo=RGlhYlx4g_16
2. https://github.com/amanchauhan71/Oxford-IIIT-Pet-using-Detectron2/tree/main
3. https://github.com/yrodriguezmd/IceVision_miniprojects/blob/main/02_object_detection__fasterrcnn%2C_yolo5%2C_retinanet%2C_effdet_2021_8_23.ipynb
4. https://medium.com/geekculture/different-models-for-object-detection-9c5cda7863c1
5. https://www.indusmic.com/post/want-to-train-your-first-object-detection-model
6. https://colab.research.google.com/github/italojs/traffic-lights-detector/blob/master/traffic_lights_detector.ipynb
7. https://www.kaggle.com/datasets/tanlikesmath/the-oxfordiiit-pet-dataset/code
8. from scratch: https://github.com/arashazimi0032/Pascal-Voc-Object-Detection
9. tensorflow OD API: https://github.com/johntikas/pet-detection/blob/master/notebooks/tf2_obj_det_api_pets.ipynb
10. create label: https://www.indusmic.com/post/want-to-train-your-first-object-detection-model
11. Transfer Learning Oxford iiit pet Classification: https://www.kaggle.com/code/aledili/transfer-learning-oxford-iiit-pet-classification
12. conplete industry worflow: https://neptune.ai/blog/how-to-train-your-own-object-detector-using-tensorflow-object-detection-api
13. od explain: https://x-wei.github.io/notes/Ng_DLMooc_c4wk3.html
14. https://en.wikipedia.org/wiki/Object_detection
15. https://www.kaggle.com/code/infernop/object-detection-techniques/notebook
16. https://cs182sp21.github.io/static/slides/lec-8.pdf
17. [Awesome Readme Templates](https://awesomeopensource.com/project/elangosundar/awesome-README-templates)
18. [Awesome README](https://github.com/matiassingers/awesome-readme)
19. [How to write a Good readme](https://bulldogjob.com/news/449-how-to-write-a-good-readme-for-your-github-project)
20. [README Template](https://readme.so/)
21. [Object Detection in 2024: The Definitive Guide](https://viso.ai/deep-learning/object-detection/)
22. [Object Detection - Computer Vision Tutorial](https://pyimagesearch.com/category/object-detection/)
23. [Object-Detection-and-Image-Segmentation-with-Detectron2](https://github.com/TannerGilbert/Object-Detection-and-Image-Segmentation-with-Detectron2?ref=gilberttanner.com)
24. [Build Detectron2 from Source](https://detectron2.readthedocs.io/en/latest/tutorials/install.html)
25. [Detectron2 - Object Detection with PyTorch](https://gilberttanner.com/blog/detectron-2-object-detection-with-pytorch/)
26. [RCNN - Original Paper](https://arxiv.org/pdf/1311.2524.pdf)
27. R-CNN References: [Link-1](https://www.kaggle.com/code/ggsri123/objectdetection-using-rcnn), [Link-2: Video Tutorial](https://www.youtube.com/watch?v=eJ7now8Q-w4&t=789s), [Link-3](https://www.kaggle.com/code/mukeshmanral/r-cnn), [Link-4](https://towardsdatascience.com/r-cnn-for-object-detection-a-technical-summary-9e7bfa8a557c), [Link-5](https://ammarchalifah.wordpress.com/2020/09/13/rcnn-implementation-with-tensorflow-2-3-and-its-application-in-video-processing/), [Link-6](https://paperswithcode.com/paper/rich-feature-hierarchies-for-accurate-object#code), [Link-7](https://sahiltinky94.medium.com/object-detection-r-cnn-aa2b180bfb49), [Link-8](https://github.com/sahilsharma884/RCNN_Scratch), [Link-9](https://towardsdatascience.com/step-by-step-r-cnn-implementation-from-scratch-in-python-e97101ccde55), [Link-10](https://github.com/1297rohit/RCNN/blob/master/RCNN.ipynb), [Link-11](https://github.com/object-detection-algorithm/R-CNN/tree/master)
28. [Faster R-CNN Object Detector](https://developers.arcgis.com/python/guide/faster-rcnn-object-detector/)

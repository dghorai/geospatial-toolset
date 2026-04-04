# -*- coding: utf-8 -*-
"""
Created on Tue May 17 2022

@author: Debabrata Ghorai, Ph.D.

Mobile phone image augmentation.
"""

import cv2
import matplotlib.pyplot  as plt


from numpy import expand_dims
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import load_img
from keras.preprocessing.image import save_img
from keras.preprocessing.image import img_to_array



imgfile = 'scene00601.jpg'

img = load_img(imgfile)

# load the image
# convert to numpy array
data = img_to_array(img)

# expand dimension to one sample
samples = expand_dims(data, 0)

# create image data augmentation generator
datagen = ImageDataGenerator(
    featurewise_center=True,
    rotation_range=(0-30),
    width_shift_range=0.2,
    height_shift_range=0.2,
    brightness_range=[0.5,1.5],
    shear_range=0.2,
    zoom_range=0.2,
    channel_shift_range=0.2,
    horizontal_flip=True,
    vertical_flip=True,
    fill_mode='nearest'
    )

# prepare iterator
it = datagen.flow(samples, batch_size=1)

# generate samples and plot
plt.figure(figsize=(45,30))
for i in range(6):
    # define subplot
    plt.subplot(330 + 1 + i)
    # generate batch of images
    batch = it.next()
    # convert to unsigned integers for viewing
    image = batch[0].astype('uint8')
    filename = imgfile.split("\\")[-1].split(".")[0]+str(i+1)+".jpg"
    cv2.imwrite(filename, image)
    # plot raw pixel data
    plt.imshow(image)
# show the figure
plt.show()

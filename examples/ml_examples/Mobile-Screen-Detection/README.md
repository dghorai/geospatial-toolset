# detect-mobile-screen-on-or-off

Mobile pictures are collected from the internet and YouTube videos, and then data augmentation techniques are used to increase the training sample for ML model development. The collected pictures were renamed using the rename file script.

Considered 151 sample datasets with 3 channels for mobile phones' screen on/off detection ML model development. Every picture was augmented six times to increase the training samples using Keras's ImageDataGenerator function. After image data augmentation, the total training sample used for the model developed was about 900+.

Model validation accuracy: 94%

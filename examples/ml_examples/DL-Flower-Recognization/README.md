Flower Recognization using CNN
----------------------------------

Data source: [Link to Dataset](https://www.tensorflow.org/datasets/catalog/tf_flowers)

Total flowers images used for the work:  3670

Flowers names:  ['dandelion', 'tulips', 'sunflowers', 'roses', 'daisy']

- dandelion total count: 898
- tulips total count: 799
- sunflowers total count: 699
- roses total count: 641
- daisy total count: 633

I have trained a convolutional neural network (CNN) to predict the 5 types of flowers using the flower recognition dataset mentioned above.
The flower images were first preprocessed and resized into a fixed image size to train the machine learning model. After the model training, the best model training accuracy was about 68%, and the validation accuracy was about 66%. These accuracy levels can be further improved by adding more datasets and enhancing CNN layers, along with working on the following options: image enhancement and feature extraction, fine-tuning data augmentation, transfer learning on the VGG model, etc.

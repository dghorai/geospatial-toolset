import os
import numpy as np
import pandas as pd
import tensorflow as tf

from classification.cnn_classifier.constants import CONFIG_FILE_PATH, PARAMS_FILE_PATH
from classification.cnn_classifier.utils.utilities import read_yaml
from classification.cnn_classifier.config.configuration import ConfigurationManager


class PredictionPipeline:
    def __init__(self, filename):
        self.filename = filename
        # self.config = read_yaml(CONFIG_FILE_PATH)
        self.config = ConfigurationManager()
        self.inference_config = self.config.get_predict_config()
        self.params = read_yaml(PARAMS_FILE_PATH)

    def get_label(self, label_id):
        # df = pd.read_csv(os.path.join(self.config.data_ingestion.root_dir, 'cifar100_trainset_info.csv'))
        df = pd.read_csv(self.inference_config.metadata)
        label_name = df['fine_label_names'][df['fine_labels'] == label_id].values[0]
        return label_name

    def predict(self):
        # load model
        # model = tf.keras.models.load_model(os.path.join("artifacts", "training", "model.h5"))
        model = tf.keras.models.load_model(self.inference_config.path_of_model)
        x, y, _ = self.params.IMAGE_SIZE
        imagename = self.filename
        test_image = tf.keras.preprocessing.image.load_img(imagename, target_size=(x, y))
        test_image = tf.keras.preprocessing.image.img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis=0)
        result = np.argmax(model.predict(test_image), axis=1)
        # print(result)
        if len(result) > 0:
            prediction = self.get_label(result[0])
        else:
            prediction = ''
        return [{"image": prediction}]

import os
import pickle
import tensorflow as tf

from pathlib import Path
from CNNClassifier.entity import EvaluationConfig
from CNNClassifier.utils.utilities import save_json


class Evaluation:
    def __init__(self, config: EvaluationConfig):
        self.config = config

    def unpickle(self, file):
        with open(file, 'rb') as fo:
            dict = pickle.load(fo, encoding='latin1')
        return dict

    def extract_data(self, dicts, num_classes):
        images = dicts['data']
        n = len(images)
        images = images.reshape(n, 3, 32, 32).transpose(0, 2, 3, 1)
        # take coarse and fine labels of the images
        c_labels = dicts['coarse_labels']
        f_labels = dicts['fine_labels']
        # transform label to categorical
        c_labels = tf.keras.utils.to_categorical(c_labels, num_classes)
        f_labels = tf.keras.utils.to_categorical(f_labels, num_classes)
        # normalize data
        # images = images/255
        return images, f_labels, c_labels

    def _valid_generator(self):
        test_set = self.unpickle(os.path.join(self.config.training_data, self.config.testset_file))
        x_test, y_test, _ = self.extract_data(test_set, self.config.num_classes)

        datagenerator_kwargs = dict(
            rescale=1./255,
            validation_split=0.30
        )

        test_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(
            **datagenerator_kwargs
        )

        self.test_generator = test_datagenerator.flow(x_test, y_test, batch_size=1)

    @staticmethod
    def load_model(path: Path) -> tf.keras.Model:
        return tf.keras.models.load_model(path)

    def evaluation(self):
        self.model = self.load_model(self.config.path_of_model)
        self._valid_generator()
        self.score = self.model.evaluate(self.test_generator)

    def save_score(self):
        scores = {"loss": self.score[0], "accuracy": self.score[1]}
        save_json(path=Path("scores.json"), data=scores)

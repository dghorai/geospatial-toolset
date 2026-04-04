import os
import pickle
import tensorflow as tf

from pathlib import Path
from CNNClassifier.entity import TrainingConfig


class Training:
    def __init__(self, config: TrainingConfig):
        self.config = config

    def get_base_model(self):
        self.model = tf.keras.models.load_model(
            self.config.updated_base_model_path
        )

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

    def train_valid_generator(self):
        train_set = self.unpickle(os.path.join(self.config.training_data, self.config.trainset_file))
        x_train, y_train, _ = self.extract_data(train_set, self.config.num_classes)

        datagenerator_kwargs = dict(
            rescale=1./255,
            validation_split=0.20
        )

        if self.config.params_is_augmentation:
            train_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(
                rotation_range=20,
                zoom_range=0.15,
                width_shift_range=0.2,
                height_shift_range=0.2,
                shear_range=0.15,
                horizontal_flip=True,
                fill_mode="nearest",
                **datagenerator_kwargs
            )
        else:
            train_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(
                **datagenerator_kwargs
            )

        valid_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(
            **datagenerator_kwargs
        )

        batch_size = self.config.params_batch_size

        self.train_generator = train_datagenerator.flow(x_train, y_train, batch_size=batch_size, subset='training', shuffle=False)
        self.valid_generator = valid_datagenerator.flow(x_train, y_train, batch_size=batch_size, subset='validation', shuffle=False)
        self.samples = len(x_train)
        self.test_size = datagenerator_kwargs['validation_split']

    @staticmethod
    def save_model(path: Path, model: tf.keras.Model):
        model.save(path)

    def train(self, callback_list: list):
        self.steps_per_epoch = (self.samples*(1 - self.test_size)) // self.train_generator.batch_size
        self.validation_steps = (self.samples*self.test_size) // self.valid_generator.batch_size

        self.model.fit(
            self.train_generator,
            steps_per_epoch=self.steps_per_epoch,
            epochs=self.config.params_epochs,
            validation_data=self.valid_generator,
            validation_steps=self.validation_steps,
            callbacks=callback_list,
            verbose=1
        )

        self.save_model(
            path=self.config.trained_model_path,
            model=self.model
        )

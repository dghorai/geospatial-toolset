import tensorflow as tf

from pathlib import Path
from CNNClassifier.entity import PrepareBaseModelConfig


class PrepareBaseModel:
    def __init__(self, config: PrepareBaseModelConfig):
        self.config = config

    def pre_trained_model(self):
        # Compiles a model integrated with VGG16 pretrained layers
        # input_shape: tuple - the shape of input images (width, height, channels)
        # n_classes: int - number of classes for the output layer
        # optimizer: string - instantiated optimizer to use for training. Defaults to 'RMSProp'
        # fine_tune: int - The number of pre-trained layers to unfreeze. If set to 0, all pretrained layers will freeze during training
        # Include_top is set to False, in order to exclude the model's fully-connected layers.
        # Pretrained convolutional layers are loaded using the Imagenet weights.

        self.model = tf.keras.applications.vgg16.VGG16(
            include_top=self.config.params_include_top,
            weights=self.config.params_weights,
            input_shape=self.config.params_image_size
        )

        self.save_model(path=self.config.base_model_path, model=self.model)
        return

    def model_from_scratch1(self):
        input_shape = tuple(self.config.params_image_size)
        n_classes = self.config.params_classes

        self.model = tf.keras.Sequential(
            [
                # convolutional layer
                tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), input_shape=input_shape, activation='relu', padding='same'),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same'),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
                tf.keras.layers.Dropout(0.2),
                # convolutional layer
                tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding='same'),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding='same'),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
                tf.keras.layers.Dropout(0.2),
                # convolutional layer
                tf.keras.layers.Conv2D(filters=256, kernel_size=(3, 3), activation='relu', padding='same'),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Conv2D(filters=256, kernel_size=(3, 3), activation='relu', padding='same'),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
                tf.keras.layers.Dropout(0.2),
                # convolutional layer
                tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding='same'),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding='same'),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
                tf.keras.layers.Dropout(0.2),
                # flattening
                tf.keras.layers.Flatten(),
                # fully connected layers
                # tf.keras.layers.Dense(1024, activation='relu'),
                # tf.keras.layers.Dropout(0.2),
                tf.keras.layers.Dense(256, activation='relu'),
                tf.keras.layers.Dropout(0.2),
                # tf.keras.layers.BatchNormalization(
                #     momentum=0.95,
                #     epsilon=0.005,
                #     beta_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.05),
                #     gamma_initializer=tf.keras.initializers.Constant(value=0.9)),
                # tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Dense(n_classes, activation='softmax')
            ]
        )

        self.save_model(path=self.config.base_model_path, model=self.model)
        return

    def model_from_scratch2(self):
        input_shape = tuple(self.config.params_image_size)
        n_classes = self.config.params_classes

        self.model = tf.keras.Sequential(
            [
                # convolutional layer
                tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3), input_shape=input_shape, activation='relu', padding='same'),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding='same'),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
                tf.keras.layers.Dropout(0.2),
                # convolutional layer
                tf.keras.layers.Conv2D(filters=256, kernel_size=(3, 3), activation='relu', padding='same'),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Conv2D(filters=256, kernel_size=(3, 3), activation='relu', padding='same'),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
                tf.keras.layers.Dropout(0.2),
                # convolutional layer
                tf.keras.layers.Conv2D(filters=512, kernel_size=(3, 3), activation='relu', padding='same'),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Conv2D(filters=512, kernel_size=(3, 3), activation='relu', padding='same'),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
                tf.keras.layers.Dropout(0.2),
                # convolutional layer
                tf.keras.layers.Conv2D(filters=1024, kernel_size=(3, 3), activation='relu', padding='same'),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Conv2D(filters=1024, kernel_size=(3, 3), activation='relu', padding='same'),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
                tf.keras.layers.Dropout(0.2),
                # flattening
                tf.keras.layers.Flatten(),
                # fully connected layers
                tf.keras.layers.Dense(1000, activation='relu'),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Dense(n_classes, activation='softmax')
            ]
        )
        self.save_model(path=self.config.base_model_path, model=self.model)
        return

    @staticmethod
    def _prepare_full_model(model, n_classes, optimizer='RMSProp', freeze_till=0, is_pretrained=False):
        if is_pretrained:
            # Defines how many layers to freeze during training.
            # Layers in the convolutional base are switched from trainable to non-trainable
            # depending on the size of the fine-tuning parameter.
            if freeze_till > 0:
                for layer in model.layers[:-freeze_till]:
                    layer.trainable = False
            else:
                for layer in model.layers:
                    layer.trainable = False
            # Create a new 'top' of the model (i.e. fully-connected layers).
            # This is 'bootstrapping' a new top_model onto the pretrained layers.
            top_model = model.output
            top_model = tf.keras.layers.Flatten(name="flatten")(top_model)
            top_model = tf.keras.layers.Dense(4096, activation='relu')(top_model)
            top_model = tf.keras.layers.Dense(1072, activation='relu')(top_model)
            top_model = tf.keras.layers.Dropout(0.2)(top_model)
            output_layer = tf.keras.layers.Dense(n_classes, activation='softmax')(top_model)
            # Group the convolutional base and new fully-connected layers into a Model object.
            full_model = tf.keras.models.Model(inputs=model.input, outputs=output_layer)
        else:
            full_model = model

        # Compiles the model for training.
        full_model.compile(
            optimizer=optimizer,
            loss=tf.keras.losses.CategoricalCrossentropy(),
            metrics=["accuracy"]
        )
        full_model.summary()

        return full_model

    def update_base_model(self):
        learning_rate = self.config.params_learning_rate
        optimizer = tf.keras.optimizers.RMSprop(learning_rate=learning_rate)
        # optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        # model = self.model_from_scratch2()
        self.full_model = self._prepare_full_model(
            model=self.model,
            n_classes=self.config.params_classes,
            optimizer=optimizer,
            freeze_till=0
        )

        self.save_model(path=self.config.updated_base_model_path, model=self.full_model)
        return

    @staticmethod
    def save_model(path: Path, model: tf.keras.Model):
        model.save(path)
        return

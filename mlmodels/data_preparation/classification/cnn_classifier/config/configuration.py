import os
import shutil

from pathlib import Path

from classification.cnn_classifier.constants import *
from classification.cnn_classifier.utils.utilities import read_yaml, create_directory
from classification.cnn_classifier.entity.config_entity import (
    DataIngestionConfig, 
    PrepareBaseModelConfig, 
    PrepareCallbacksConfig, 
    TrainingConfig, 
    EvaluationConfig, 
    PredictConfig
)


class ConfigurationManager:
    def __init__(self, config_filepath=CONFIG_FILE_PATH, params_filepath=PARAMS_FILE_PATH):
        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)

        create_directory([self.config.artifacts_root])

    def get_data_ingestion_config(self):
        config = self.config.data_ingestion
        create_directory([config.root_dir])

        data_ingestion_config = DataIngestionConfig(
            root_dir=config.root_dir,
            source_url=config.source_url,
            local_data_file=config.local_data_file,
            unzip_dir=config.unzip_dir,
            trainset_file=config.trainset_file,
            meta_file=config.meta_file,
            metadata=config.metadata
        )

        return data_ingestion_config

    def get_prepare_base_model_config(self) -> PrepareBaseModelConfig:
        config = self.config.prepare_base_model
        create_directory([config.root_dir])

        prepare_base_model_config = PrepareBaseModelConfig(
            root_dir=Path(config.root_dir),
            base_model_path=Path(config.base_model_path),
            updated_base_model_path=Path(config.updated_base_model_path),
            params_image_size=self.params.IMAGE_SIZE,
            params_learning_rate=self.params.LEARNING_RATE,
            params_include_top=self.params.INCLUDE_TOP,
            params_weights=self.params.WEIGHTS,
            params_classes=self.params.CLASSES
        )

        return prepare_base_model_config

    def get_prepare_callback_config(self) -> PrepareCallbacksConfig:
        config = self.config.prepare_callbacks
        model_ckpt_dir = os.path.dirname(config.checkpoint_model_filepath)

        create_directory([
            Path(model_ckpt_dir),
            Path(config.tensorboard_root_log_dir)
        ])

        prepare_callback_config = PrepareCallbacksConfig(
            root_dir=Path(config.root_dir),
            tensorboard_root_log_dir=Path(config.tensorboard_root_log_dir),
            checkpoint_model_filepath=Path(config.checkpoint_model_filepath)
        )

        return prepare_callback_config

    def get_training_config(self) -> TrainingConfig:  # this is return value annotation
        training = self.config.training
        prepare_base_model = self.config.prepare_base_model
        params = self.params

        create_directory([Path(training.root_dir)])

        training_config = TrainingConfig(
            root_dir=Path(training.root_dir),
            trained_model_path=Path(training.trained_model_path),
            updated_base_model_path=Path(
                prepare_base_model.updated_base_model_path),
            training_data=Path(self.config.data_ingestion.unzip_dir),
            trainset_file=Path(self.config.data_ingestion.trainset_file),
            testset_file=Path(self.config.data_ingestion.testset_file),
            num_classes=params.CLASSES,
            params_epochs=params.EPOCHS,
            params_batch_size=params.BATCH_SIZE,
            params_is_augmentation=params.AUGMENTATION,
            params_image_size=params.IMAGE_SIZE
        )

        return training_config

    def get_validation_config(self) -> EvaluationConfig:
        eval_config = EvaluationConfig(
            path_of_model=Path(self.config.training.trained_model_path),
            training_data=Path(self.config.data_ingestion.unzip_dir),
            testset_file=Path(self.config.data_ingestion.testset_file),
            all_params=self.params,
            num_classes=self.params.CLASSES,
            params_image_size=self.params.IMAGE_SIZE,
            params_batch_size=self.params.BATCH_SIZE
        )

        return eval_config

    def get_predict_config(self) -> PredictConfig:
        inference = self.config.inference
        create_directory([Path(inference.root_dir)])

        src_model = Path(self.config.training.trained_model_path)
        dst_model = Path(self.config.inference.inference_model_path)

        if not os.path.isfile(dst_model):
            shutil.copy(src_model, dst_model)

        src_label = Path(self.config.data_ingestion.metadata)
        dst_label = Path(self.config.inference.metadata)

        if not os.path.isfile(dst_label):
            shutil.copy(src_label, dst_label)

        pred_config = PredictConfig(
            path_of_model=Path(self.config.inference.inference_model_path),
            metadata=Path(self.config.inference.metadata)
        )

        return pred_config

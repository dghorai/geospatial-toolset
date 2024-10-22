from logger import logger
from classification.cnn_classifier.config.configuration import ConfigurationManager
from classification.cnn_classifier.components.stage3_prepare_callbacks import PrepareCallback
from classification.cnn_classifier.components.stage4_training import Training


STAGE_NAME = "TRAINING"


class ModelTrainingPipeline:
    def __init__(self) -> None:
        pass

    def main(self):
        config = ConfigurationManager()
        callback_config = config.get_prepare_callback_config()
        callbacks = PrepareCallback(config=callback_config)
        callbacks = callbacks.callbacks()
        training_config = config.get_training_config()
        training = Training(config=training_config)
        training.get_base_model()
        training.train_valid_generator()
        training.train(callbacks)


if __name__ == '__main__':
    try:
        logger.info(f"*******************")
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = ModelTrainingPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e

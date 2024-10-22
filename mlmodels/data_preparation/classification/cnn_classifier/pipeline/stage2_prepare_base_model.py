from logger import logger
from classification.cnn_classifier.config.configuration import ConfigurationManager
from classification.cnn_classifier.components.stage2_prepare_base_model import PrepareBaseModel


STAGE_NAME = "PREPARE BASE MODEL"


class PrepareBaseModelPipeline:
    def __init__(self) -> None:
        pass

    def main(self):
        config = ConfigurationManager()
        prepare_base_model_config = config.get_prepare_base_model_config()
        prepare_base_model = PrepareBaseModel(config=prepare_base_model_config)
        # prepare_base_model.pre_trained_model()
        # prepare_base_model.model_from_scratch1()
        prepare_base_model.model_from_scratch2()
        prepare_base_model.update_base_model()


if __name__ == '__main__':
    try:
        logger.info(f"*******************")
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = PrepareBaseModelPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n")
    except Exception as e:
        logger.exception(e)
        raise e

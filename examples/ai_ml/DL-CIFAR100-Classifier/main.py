import sys
sys.path.append('src')
from CNNClassifier import logger
from CNNClassifier.pipeline.stage1_data_ingestion import DataIngestionPipeline
from CNNClassifier.pipeline.stage2_prepare_base_model import PrepareBaseModelPipeline
from CNNClassifier.pipeline.stage3_training import ModelTrainingPipeline
from CNNClassifier.pipeline.stage4_evaluation import ModelEvaluationPipeline


logger.info("Logger import successfull!")

STAGE_NAME = "DATA INGESTION"

try:
    logger.info(f">>>>> {STAGE_NAME} started <<<<<")
    obj = DataIngestionPipeline()
    obj.main()
    logger.info(f">>>>> {STAGE_NAME} completed <<<<<\n")
except Exception as e:
    logger.exception(e)
    raise e


STAGE_NAME = "PREPARE BASE MODEL"

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


STAGE_NAME = "TRAINING"

if __name__ == '__main__':
    try:
        logger.info(f"*******************")
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = ModelTrainingPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n")
    except Exception as e:
        logger.exception(e)
        raise e


STAGE_NAME = "MODEL EVALUATION"

if __name__ == '__main__':
    try:
        logger.info(f"*******************")
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = ModelEvaluationPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e

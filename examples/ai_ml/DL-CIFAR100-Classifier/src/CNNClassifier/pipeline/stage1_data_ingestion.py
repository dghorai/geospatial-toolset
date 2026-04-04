from CNNClassifier import logger
from CNNClassifier.config.configuration import ConfigurationManager
from CNNClassifier.components.stage1_data_ingestion import DataIngestion


STAGE_NAME = "DATA INGESTION STAGE"


class DataIngestionPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        data_ingestion_config = config.get_data_ingestion_config()
        data_ingestion = DataIngestion(config=data_ingestion_config)
        data_ingestion.download_file()
        data_ingestion.unzip_targzfile()
        data_ingestion.get_metadata()


if __name__ == "__main__":
    try:
        logger.info(f">>>>> {STAGE_NAME} started <<<<<")
        obj = DataIngestionPipeline()
        obj.main()
        logger.info(f">>>>> {STAGE_NAME} completed <<<<<\n")
    except Exception as e:
        logger.exception(e)
        raise e

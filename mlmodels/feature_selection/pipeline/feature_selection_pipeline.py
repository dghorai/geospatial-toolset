from logger import logger
from feature_selection.components.find_best_features import GetBestMLFeatures
from feature_selection.config import ConfigManager


STAGE_NAME = "Find Best ML Features"


class FeatureSelector:
    def __init__(self) -> None:
        pass

    def main(self):
        config = ConfigManager()
        get_dataingestion_config = config.get_dataingestion_config()
        get_mlfeatureselector_config = config.get_mlfeatureselector_config()

        best_feat = GetBestMLFeatures(
            diconfig=get_dataingestion_config, mlfsconfig=get_mlfeatureselector_config)
        final_features = best_feat.find_best_features()
        return final_features


if __name__ == "__main__":
    try:
        logger.info(f">>>>> {STAGE_NAME} started <<<<<")
        obj = FeatureSelector()
        ff = obj.main()
        logger.info(f'Final Features: {ff}')
        logger.info(f">>>>> {STAGE_NAME} completed <<<<<\n")
    except Exception as e:
        logger.exception(e)
        raise e

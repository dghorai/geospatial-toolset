import os

from pathlib import Path

from feature_selection.utils import read_yaml, create_directory
from feature_selection.config_entity import DataIngestionConfig, MLFeatureSelectorConfig
from feature_selection import CONFIG_FILE_PATH, PARAMS_FILE_PATH


class ConfigManager:
    def __init__(self, config_filepath=CONFIG_FILE_PATH, params_filepath=PARAMS_FILE_PATH):
        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)
        create_directory([self.config.artifacts_root])

    def get_dataingestion_config(self):
        config = self.config.data_ingestion
        create_directory([config.root_dir])
        data_ingestion_config = DataIngestionConfig(
            root_dir=config.root_dir,
            data_file=config.data_file
        )
        return data_ingestion_config
    
    def get_mlfeatureselector_config(self):
        mlfeature_selector_config = MLFeatureSelectorConfig(
            supervised_type=self.params.MLBF_SUPERVISED_TYPE,
            label_id_columns=self.params.MLBF_LABEL_ID_COLUMNS,
            label_name_columns=self.params.MLBF_LABEL_NAME_COLUMNS,
            drop_columns=self.params.MLBF_DROP_COLUMNS,
            feature_selection_option=self.params.MLBF_FEATURE_SELECTION_OPTION,
            n_feature_to_select=self.params.MLBF_N_FEATURE_TO_SELECT
        )
        return mlfeature_selector_config

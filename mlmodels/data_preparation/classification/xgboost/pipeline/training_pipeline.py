# import modules
from classification.xgboost.components.model_trainer import ModelTrainer
from classification.xgboost.components.data_transformation import DataTransformation
from classification.xgboost.components.data_ingestion import DataIngestion


def main():
    obj = DataIngestion()
    train_dp, test_dp = obj.initiate_data_ingestion()
    data_transformation = DataTransformation()
    train_arr, test_arr, _ = data_transformation.initiate_data_transformation(
        train_dp, test_dp)
    model_trainer = ModelTrainer()
    model_trainer.initiate_model_training(train_arr, test_arr)

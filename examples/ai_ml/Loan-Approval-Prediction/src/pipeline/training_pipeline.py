from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer


def main():
    obj = DataIngestion()
    train_data_path, test_data_path = obj.initiate_data_ingestion()
    data_transformation = DataTransformation()
    train_arr_cs, test_arr_cs, train_arr_ss, test_arr_ss = data_transformation.initiate_data_transformation(
        train_data_path, test_data_path)
    model_trainer = ModelTrainer()
    model_trainer.initiate_model_training(
        train_arr_cs, test_arr_cs, train_arr_ss, test_arr_ss)

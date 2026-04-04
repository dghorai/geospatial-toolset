import os
import sys
import pickle
import pandas as pd
import tensorflow as tf
import warnings

from sklearn.preprocessing import StandardScaler
from dataclasses import dataclass
from src.logger import logging, project_dir
from src.exception import CustomException


warnings.simplefilter("ignore", category=FutureWarning)


@dataclass
class DataIngestionConfig:
    sample_data = os.path.join(
        project_dir, "artifacts", "preprocessor_dataset.csv")
    dt_model = os.path.join(project_dir, "artifacts",
                            "best", "dt_model_73p.sav")
    rf_model = os.path.join(project_dir, "artifacts",
                            "best", "rf_model_80p_latest.sav")
    lr_model = os.path.join(project_dir, "artifacts",
                            "best", "lr_model_76p.sav")
    svm_model = os.path.join(project_dir, "artifacts",
                             "best", "svm_model_76p.sav")
    knn_model = os.path.join(project_dir, "artifacts",
                             "best", "knn_model_67p.sav")
    gnb_model = os.path.join(project_dir, "artifacts",
                             "best", "gnb_model_74p.sav")
    ann_model = os.path.join(project_dir, "artifacts",
                             "best", "ann_model_81p_latest.h5")


class PredictPipeline:

    def __init__(self):
        self.config = DataIngestionConfig()

    def predict(self, X_test_custom, X_test_standard):
        """predict result"""
        try:
            prediction = {0: 'N', 1: 'Y'}

            # DT : Load dt model
            model_dt = pickle.load(open(self.config.dt_model, 'rb'))
            y_pred1 = prediction[model_dt.predict(X_test_custom)[0]]

            # RF: Load rf model
            model_rf = pickle.load(open(self.config.rf_model, 'rb'))
            y_pred2 = prediction[model_rf.predict(X_test_custom)[0]]

            # LR: Load lr model
            model_lr = pickle.load(open(self.config.lr_model, 'rb'))
            y_pred3 = prediction[model_lr.predict(X_test_standard)[0]]

            # SVM: Load svm model
            model_svm = pickle.load(open(self.config.svm_model, 'rb'))
            y_pred4 = prediction[model_svm.predict(X_test_standard)[0]]

            # KNN: Load knn model
            model_knn = pickle.load(open(self.config.knn_model, 'rb'))
            y_pred5 = prediction[model_knn.predict(X_test_standard)[0]]

            # GNB: Load gnb model
            model_gnb = pickle.load(open(self.config.gnb_model, 'rb'))
            y_pred6 = prediction[model_gnb.predict(X_test_standard)[0]]

            # ANN : Load ann model
            model_ann = tf.keras.models.load_model(self.config.ann_model)
            y_pred = (model_ann.predict(X_test_standard) > 0.5)
            y_pred7 = prediction[0 if y_pred[0][0] == False else 1]

            # final prediction
            y_pred_list = [y_pred1, y_pred2, y_pred3,
                           y_pred4, y_pred5, y_pred6, y_pred7]
            n_count = y_pred_list.count("N")
            y_count = y_pred_list.count("Y")

            if y_count > n_count:
                result = 'Yes'
            else:
                result = 'No'
            return result
        except Exception as e:
            logging.info('Exception occured in prediction')
            raise CustomException(e, sys)


class CustomData:

    def __init__(self, **kwargs):
        self.config = DataIngestionConfig()
        # self.p1 = str(kwargs)
        self.p2 = str(kwargs['Property_Area'])
        self.p3 = str(kwargs['Married'])
        self.p4_str = str(kwargs['Dependents'])
        self.p5 = str(kwargs['Education'])
        self.p6 = float(kwargs['ApplicantIncome'])
        self.p7 = float(kwargs['CoapplicantIncome'])
        self.p8 = float(kwargs['LoanAmount'])
        self.p9 = int(kwargs['Loan_Amount_Term'])
        self.p10 = int(kwargs['Credit_History'])
        # convert str to int
        if self.p4_str == "3+":
            self.p4 = 4
        else:
            self.p4 = int(self.p4_str)

    def string_to_numeric(self, df):
        """convert string column to numeric"""
        df2 = df.dropna()  # drop nan
        for cols in df.columns.tolist():
            if str(df2[cols].values[0])[0].isdigit() == True:
                df[cols] = pd.to_numeric(df[cols])
        return df

    def category_to_numeric(self, df3, categorical_features):
        # 0 -> Married; 2 -> Education; 8 -> Property_Area
        disc = {0: {'No': 0, 'Yes': 1, 'None': 2},
                2: {'Graduate': 0, 'Not Graduate': 1},
                8: {'Rural': 0, 'Semiurban': 1, 'Urban': 2}}
        # loop over category variable
        for col in categorical_features:
            for ix, row in df3.iterrows():
                value = row[col]
                # print(col, value, disc[col][value])
                df3.loc[ix, col] = disc[col][value]
        return df3

    def feature_engineering(self, xdf, return_type=None):
        """feature engineering with different scale"""
        # statement
        if return_type == 'StandardScaler':
            # Perform Feature Scaling
            xtrain_df = pd.read_csv(self.config.sample_data)
            X_train = xtrain_df.to_numpy()
            sc = StandardScaler()
            X_train = sc.fit_transform(X_train)
            X_test = sc.transform(xdf.to_numpy())
        elif return_type == 'CustomScaler':
            for cols in xdf.columns.tolist():
                if cols == 0:
                    div1 = 1
                elif cols == 1:
                    div1 = 0.0
                elif cols == 2:
                    div1 = 0
                elif cols == 3:
                    div1 = 2500
                elif cols == 4:
                    div1 = 0.0
                elif cols == 5:
                    div1 = 120.0
                elif cols == 6:
                    div1 = 360.0
                elif cols == 7:
                    div1 = 1.0
                elif cols == 8:
                    div1 = 1
                else:
                    print('index out of range')
                # divide value
                if div1 > 0:
                    xdf[cols] = xdf[cols]/div1
            # split-data
            X_test = xdf.to_numpy()
        else:
            print('scale not define')
        return X_test

    def model_inputs(self, df):
        """feature engineering with different scale"""
        # feature engineering with LabelEncoder
        # replace header with 0-n values
        df3 = pd.DataFrame(df.iloc[:, 0:df.shape[-1]].values)
        df3 = self.string_to_numeric(df3)
        categorical_features = [
            feature for feature in df3.columns if df3[feature].dtypes == 'O']
        # apply LabelEncoder to categorical variable
        df3 = self.category_to_numeric(df3, categorical_features)
        # create final inputs
        X_test_custom = self.feature_engineering(
            df3, return_type='CustomScaler')
        X_test_standard = self.feature_engineering(
            df3, return_type='StandardScaler')
        return X_test_custom, X_test_standard

    def format_user_data(self, user_data_dict):
        # create a df
        df = pd.DataFrame(user_data_dict)
        x_test_custom, x_test_standard = self.model_inputs(df)
        return x_test_custom, x_test_standard

    def get_user_inputs(self):
        try:
            user_data_dict = {
                'Married': [self.p3],
                'Dependents': [self.p4],
                'Education': [self.p5],
                'ApplicantIncome': [self.p6],
                'CoapplicantIncome': [self.p7],
                'LoanAmount': [self.p8],
                'Loan_Amount_Term': [self.p9],
                'Credit_History': [self.p10],
                'Property_Area': [self.p2]
            }
            # create a df
            x_test_custom, x_test_standard = self.format_user_data(
                user_data_dict)
            logging.info('Dataframe gathered')
            return x_test_custom, x_test_standard
        except Exception as e:
            logging.info('Exception occured in prediction pipeline')
            raise CustomException(e, sys)

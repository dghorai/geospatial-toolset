import os
import sys
import pickle

from dataclasses import dataclass
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from keras.models import Sequential
from keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras.callbacks import EarlyStopping
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.model_selection import train_test_split

from src.exception import CustomException
from src.logger import logging, project_dir


@dataclass
class ModelTrainerConfig:
    dt_model_path = os.path.join(project_dir, "artifacts", "dt_model.sav")
    rf_model_path = os.path.join(project_dir, "artifacts", "rf_model.sav")
    lr_model_path = os.path.join(project_dir, "artifacts", "lr_model.sav")
    svm_model_path = os.path.join(project_dir, "artifacts", "svm_model.sav")
    knn_model_path = os.path.join(project_dir, "artifacts", "knn_model.sav")
    gnb_model_path = os.path.join(project_dir, "artifacts", "gnb_model.sav")
    ann_model_path = os.path.join(project_dir, "artifacts", "ann_model.h5")


class ModelTrainer:

    def __init__(self):
        self.config = ModelTrainerConfig()
        self.model_outpath = {
            'DecisionTreeClassifier': self.config.dt_model_path,
            'RandomForestClassifier': self.config.rf_model_path,
            'LogisticRegression': self.config.lr_model_path,
            'SVC': self.config.svm_model_path,
            'KNeighborsClassifier': self.config.knn_model_path,
            'GaussianNB': self.config.gnb_model_path,
            'ANNClassifier': self.config.ann_model_path
        }

    def ann_model(self, n_rows, n_cols):
        """binary classification with ANN model"""
        # ann model
        classifier = Sequential()
        # Add the input layer and the first hidden layer
        classifier.add(Dense(units=7, activation='relu'))
        classifier.add(Dense(units=7, activation='relu'))
        classifier.add(Dense(units=1, activation='sigmoid'))
        # optimizer set
        opt = Adam(learning_rate=0.0001)
        classifier.compile(
            optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
        classifier.build((n_rows, n_cols))
        classifier.summary()
        return classifier

    def evaluate_model(
            self,
            X_train_cs, y_train_cs, X_test_cs, y_test_cs,
            X_train_ss, y_train_ss, X_test_ss, y_test_ss,
            models
    ):
        try:
            report = {}
            for i in range(len(models)):
                model_name = list(models.keys())[i]
                model = list(models.values())[i]

                if model_name in ['DecisionTreeClassifier', 'RandomForestClassifier']:
                    # Train model
                    model.fit(X_train_cs, y_train_cs)
                    # Predict Testing data
                    y_test_pred = model.predict(X_test_cs)
                    # Get accuracy_score for test data
                    test_model_score = accuracy_score(y_test_cs, y_test_pred)
                    # get classification report for test data
                    test_classification_report = classification_report(
                        y_test_cs, y_test_pred)
                    # get confusion matrix
                    cm = confusion_matrix(y_test_cs, y_test_pred)
                    # save model
                    pickle.dump(model, open(
                        self.model_outpath[model_name], 'wb'))
                else:
                    if model_name == 'ANNClassifier':
                        # split train data
                        X_train1, X_valid, y_train1, y_valid = train_test_split(
                            X_train_ss, y_train_ss, test_size=0.3, random_state=1, shuffle=True)
                        # set callback parameters
                        mcp = ModelCheckpoint(
                            filepath=self.model_outpath[model_name], monitor='val_loss', save_best_only=True, mode='min', verbose=1)
                        es = EarlyStopping(monitor='val_loss', patience=5)
                        callbacks = [es, mcp]
                        model.fit(X_train1, y_train1, batch_size=10, validation_data=(
                            X_valid, y_valid), epochs=500, verbose=1, callbacks=callbacks)
                        # predict
                        y_test_pred = model.predict(X_test_ss)
                        y_test_pred = (y_test_pred > 0.5)
                    else:
                        # Train model
                        model.fit(X_train_ss, y_train_ss)
                        # Predict Testing data
                        y_test_pred = model.predict(X_test_ss)
                        # save model
                        pickle.dump(model, open(
                            self.model_outpath[model_name], 'wb'))

                    # Get accuracy_score for test data
                    test_model_score = accuracy_score(
                        y_test_ss, y_test_pred)
                    # get classification report for test data
                    test_classification_report = classification_report(
                        y_test_ss, y_test_pred)
                    # get confusion matrix
                    cm = confusion_matrix(y_test_ss, y_test_pred)

                # res dict
                res = {'accuracy_score': test_model_score,
                       'classification_report': test_classification_report,
                       'confusion_matrix': cm}
                # append
                report[model_name] = res

            return report
        except Exception as e:
            logging.info('Exception occured during model training')
            raise CustomException(e, sys)

    def initiate_model_training(self, train_arrc_cs, test_arrc_cs, train_arrc_ss, test_arrc_ss):
        try:
            logging.info(
                'Splitting dependent and independent variables from train and test data')

            X_train_cs, y_train_cs, X_test_cs, y_test_cs = (
                train_arrc_cs[:, :-1],
                train_arrc_cs[:, -1],
                test_arrc_cs[:, :-1],
                test_arrc_cs[:, -1]
            )
            X_train_ss, y_train_ss, X_test_ss, y_test_ss = (
                train_arrc_ss[:, :-1],
                train_arrc_ss[:, -1],
                test_arrc_ss[:, :-1],
                test_arrc_ss[:, -1]
            )

            n_rows, n_cols = X_train_ss.shape

            # change/add based on model
            models = {
                'DecisionTreeClassifier': DecisionTreeClassifier(criterion='entropy', random_state=123),
                'RandomForestClassifier': RandomForestClassifier(random_state=445, max_depth=8, n_estimators=10, min_samples_split=2, min_samples_leaf=6),
                'LogisticRegression': LogisticRegression(C=4.49),
                'SVC': SVC(C=1, gamma=0.1, kernel='rbf'),
                'KNeighborsClassifier': KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=1),
                'GaussianNB': GaussianNB(),
                'ANNClassifier': self.ann_model(n_rows, n_cols)
            }

            model_report: dict = self.evaluate_model(
                X_train_cs, y_train_cs, X_test_cs, y_test_cs,
                X_train_ss, y_train_ss, X_test_ss, y_test_ss,
                models
            )

            print(model_report)
            print('\n===========================\n')
            logging.info(f'Model Report: {model_report}')

            # get the best model score and name from dictionary
            # best_model_score = max(sorted(model_report.values()))
            best_model_score = max(
                sorted([model_report[mn]['accuracy_score'] for mn in model_report.keys()]))
            # best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)]
            best_model_name = list(model_report.keys())[
                [model_report[mn]['accuracy_score'] for mn in model_report.keys()].index(best_model_score)]

            # best_model = models[best_model_name]

            print(
                f'Best model found -> Model Name: {best_model_name}, accuracy report: {best_model_score}')
            print('\n===========================\n')
            logging.info(
                f'Best model found -> Model Name: {best_model_name}, accuracy report: {best_model_score}')
        except Exception as e:
            logging.info('Exception occured at model training stage')
            raise CustomException(e, sys)

# -*- coding: utf-8 -*-
"""
Created on Tue Sep 20 19:33:15 2022

@author: Debabrata Ghorai, Ph.D.

Create ensemble model.
"""

# https://trenton3983.github.io/files/projects/2019-07-19_fraud_detection_python/2019-07-19_fraud_detection_python.html

import numpy as np

from sklearn.ensemble import VotingClassifier

from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

import pandas as pd

# for data preparation
from sklearn.preprocessing import LabelEncoder #, OneHotEncoder
# from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

# for data split for model training
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
pd.options.mode.chained_assignment = None # to avoid pd warnings

#%%
def get_model_results(X_train: np.ndarray, y_train: np.ndarray,
                      X_test: np.ndarray, y_test: np.ndarray, model):
    """
    model: sklearn model (e.g. RandomForestClassifier)
    """
    # Fit your training model to your training set
    model.fit(X_train, y_train)

    # Obtain the predicted values and probabilities from the model 
    predicted = model.predict(X_test)
    
    try:
        probs = model.predict_proba(X_test)
        print('ROC Score:')
        print(roc_auc_score(y_test, probs[:,1]))
    except AttributeError:
        pass

    # Print the ROC curve, classification report and confusion matrix
    print('\nClassification Report:')
    print(classification_report(y_test, predicted))
    print('\nConfusion Matrix:')
    print(confusion_matrix(y_test, predicted))


def string_to_numeric(df):
    """convert string column to numeric"""
    df2=df.dropna() # drop nan
    for cols in df.columns.tolist():
        if str(df2[cols].values[0])[0].isdigit() == True:
            df[cols] = pd.to_numeric(df[cols])
    return df


def create_partial_balanced_data(df, th='15%'):
    """create partial balance data for model training"""
    t = int(th[:-1])
    target_yes = df[df['Loan_Status'] == 'Y']
    target_no = df[df['Loan_Status'] == 'N']
    # get counts
    n_yes = target_yes.shape[0]
    n_no = target_no.shape[0]
    # statements
    if n_yes > n_no:
        p = n_no + int((n_no*t)/100)
        if p < n_yes:
            df_yes = target_yes.sample(frac = p/n_yes)
        else:
            df_yes = target_yes.copy()
        # final df
        df_final = pd.concat([df_yes, target_no], axis=0)
    else:
        p = n_yes + int((n_yes*t)/100)
        if p < n_no:
            df_no = target_no.sample(frac = p/n_no)
        else:
            df_no = target_no.copy()
        # final df
        df_final = pd.concat([target_yes, df_no], axis=0)
    # shuffle the rows
    df_final = shuffle(df_final)
    return df_final

def input_dataframe(in_file):
    # read data
    df = pd.read_csv(in_file)
    # replace '3+' from ‘Dependents’ columns
    df['Dependents'].replace('3+', 4, inplace=True)
    df['Dependents'] = df['Dependents'].astype(float)
    df = string_to_numeric(df)
    # create partial balanced data
    df = create_partial_balanced_data(df, th='20%')
    return df

def feature_engineering(df, colslist=None, return_type='XY'):
    """feature engineering with different scale"""
    df = df[colslist]
    # print(df.columns)
    # feature engineering with LabelEncoder and StandardScaler
    df3 = pd.DataFrame(df.iloc[:, 0:df.shape[-1]].values) # replace header with 0-n values
    df3 = string_to_numeric(df3)
    categorical_features = [feature for feature in df3.columns if df3[feature].dtypes=='O']
    # print(df3.columns)
    # apply LabelEncoder to categorical variable
    for col in categorical_features:
        labelencoder_col = LabelEncoder()
        df3.loc[:, col] = labelencoder_col.fit_transform(df3.iloc[:, col])
        labelencoder_col = None
    
    # seperate inputs and label
    x_columns = [i for i in range(df3.shape[-1] - 1)]
    X = df3[x_columns]
    y = df3.iloc[:, df3.shape[-1]-1].values
    
    # fill missing values if any
    # interpolate backwardly across the column
    X.interpolate(method ='linear', limit_direction ='backward', inplace=True)
    # interpolate in forward order across the column
    X.interpolate(method ='linear', limit_direction ='forward', inplace=True)
    
    # statement
    if return_type == 'XY':
        result = {'X': X, 'y': y}
    elif return_type == 'StandardScaler':
        # Perform Feature Scaling
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0, shuffle=True)
        # https://towardsdatascience.com/what-and-why-behind-fit-transform-vs-transform-in-scikit-learn-78f915cf96fe
        sc = StandardScaler()
        X_train = sc.fit_transform(X_train)
        X_test = sc.transform(X_test)
        result = {'X_train': X_train, 'X_test': X_test, 'y_train': y_train, 'y_test': y_test}
    elif return_type == 'CustomScaler':
        for cols in X.columns.tolist():
            # div1 = X[cols].mode().values[0]
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
            print(cols, div1)
            if div1 > 0:
                X[cols] = X[cols]/div1
        # split-data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0, shuffle=True)
        result = {'X_train': X_train.to_numpy(), 'X_test': X_test.to_numpy(), 'y_train': y_train, 'y_test': y_test}
    else:
        print('scale not define')
    return result

    
#%%
in_file = "../training-samples/input_training_data.csv"

df = input_dataframe(in_file)

final_features = ['Married', 'Dependents', 'Education', 'ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term', 'Credit_History', 'Property_Area']
final_columns = final_features + ['Loan_Status']
df_final = df[final_columns]
xysplit = feature_engineering(df_final, colslist=final_columns, return_type='CustomScaler')
X_train = xysplit['X_train']
X_test = xysplit['X_test']
y_train = xysplit['y_train']
y_test = xysplit['y_test']

#%%
# Voting Classifier
# Define Models
# Define the three classifiers to use in the ensemble
clf1 = LogisticRegression(class_weight={0:1, 1:15},
                          random_state=5,
                          solver='liblinear')

clf2 = RandomForestClassifier(class_weight={0:1, 1:12}, 
                              criterion='gini', 
                              max_depth=8, 
                              max_features='log2',
                              min_samples_leaf=10, 
                              n_estimators=30, 
                              n_jobs=-1,
                              random_state=5)

clf3 = DecisionTreeClassifier(random_state=5,
                              class_weight="balanced")


# Combine the classifiers in the ensemble model
ensemble_model = VotingClassifier(estimators=[('lr', clf1), ('rf', clf2), ('dt', clf3)], voting='hard')

# Fit and predict as with other models
ensemble_model.fit(X_train, y_train)
ensemble_model.predict(X_test)

# the voting='hard' option uses the predicted class labels and takes the majority vote
# the voting='soft' option takes the average probability by combining the predicted probabilities of the individual models
# Weights can be assigned to the VotingClassifer with weights=[2,1,1] (Useful when one model significantly outperforms the others)

#%%
# Adjusting weights within the Voting Classifier
# Define the ensemble model
ensemble_model = VotingClassifier(estimators=[('lr', clf1), ('rf', clf2), ('gnb', clf3)], voting='soft', weights=[1, 4, 1], flatten_transform=True)

# Get results 
get_model_results(X_train, y_train, X_test, y_test, ensemble_model)

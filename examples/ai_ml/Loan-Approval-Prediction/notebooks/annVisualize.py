# -*- coding: utf-8 -*-
"""
Created on Mon Sep 19 20:17:15 2022

@author: Debabrata Ghorai, Ph.D.

Draw ANN Model Figure.
"""

# https://www.geeksforgeeks.org/how-to-visualize-a-neural-network-in-python-using-graphviz/
# https://towardsdatascience.com/visualizing-artificial-neural-networks-anns-with-just-one-line-of-code-b4233607209e
# https://neptune.ai/blog/visualizing-machine-learning-models


# import module
from keras.models import Sequential
from keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from sklearn.utils import shuffle
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from ann_visualizer.visualize import ann_viz

import pandas as pd
import numpy

#%%
# fix random seed for reproducibility
numpy.random.seed(7)

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
xysplit = feature_engineering(df_final, colslist=final_columns, return_type='StandardScaler')
X_train = xysplit['X_train']
X_test = xysplit['X_test']
y_train = xysplit['y_train']
y_test = xysplit['y_test']
X_train1, X_valid, y_train1, y_valid = train_test_split(X_train, y_train, test_size=0.3, random_state=1, shuffle=True)

#%%
# ann model
classifier = Sequential()
# Add the input layer and the first hidden layer
classifier.add(Dense(units=7, activation = 'relu'))
classifier.add(Dense(units=7, activation = 'relu'))
classifier.add(Dense(units=1, activation = 'sigmoid'))
# optimizer set
opt = Adam(learning_rate=0.0001)
classifier.compile(optimizer = opt, loss = 'binary_crossentropy', metrics = ['accuracy'])
classifier.build(X_train.shape)
classifier.summary()
# fit model
classifier.fit(X_train1, y_train1, batch_size = 10, validation_data=(X_valid, y_valid), epochs = 100)
# evaluate the model
scores = classifier.evaluate(X_train1, y_train1)
print("\n%s: %.2f%%" % (classifier.metrics_names[1], scores[1]*100))

#%%
# plot ann model
# step-1: install graphviz exe (https://graphviz.org/download/)
# step-2: Install python graphviz package
# step-3: Add C:\Program Files (x86)\Graphviz2.38\bin to User path
# step-4: Add C:\Program Files (x86)\Graphviz2.38\bin\dot.exe to System Path
# or do the following steps:
import os
os.environ["PATH"] += os.pathsep + 'C:/Program Files/Graphviz/bin/'

ann_viz(classifier, view=True, title="My first neural network")

# Online Plot ANN: http://alexlenail.me/NN-SVG/

# -*- coding: utf-8 -*-
"""
Created on Mon Sep 19 21:09:10 2022

@author: Debabrata Ghorai, Ph.D.

Draw Decesion Tree Model Figure.
"""

# https://neptune.ai/blog/visualizing-machine-learning-models


import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
import pandas as pd
from sklearn import tree

# for data preparation
from sklearn.preprocessing import LabelEncoder #, OneHotEncoder
# from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

# for data split for model training
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
pd.options.mode.chained_assignment = None # to avoid pd warnings


#%%
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
dt = DecisionTreeClassifier(criterion = 'entropy', random_state = 123)
dt.fit(X_train, y_train)
# plot dt
plt.figure(figsize=(45,35)) 
tree.plot_tree(dt, filled=True, fontsize=10)
plt.show()


#%%
# https://www.kaggle.com/code/mgabrielkerr/visualizing-knn-svm-and-xgboost-on-iris-dataset
# import warnings
from matplotlib.colors import ListedColormap
import numpy as np

# def versiontuple(v):
#     return tuple(map(int, (v.split("."))))


def plot_decision_regions(X, y, classifier, test_idx=None, resolution=0.02):
    # setup marker generator and color map
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution), np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1],
                    alpha=0.8, c=cmap(idx),
                    marker=markers[idx], label=cl)


#%%
svm = SVC(C=1, gamma=0.1, kernel='rbf')
svm.fit(X_train, y_train)
plot_decision_regions(X_train, y_train, svm)


knn = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 1)
knn.fit(X_train, y_train)


gnb = GaussianNB()
gnb.fit(X_train, y_train)


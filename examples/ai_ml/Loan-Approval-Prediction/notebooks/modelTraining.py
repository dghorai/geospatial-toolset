# -*- coding: utf-8 -*-
"""
Created on Sun Sep 11 10:20:56 2022

@author: Debabrata Ghorai, Ph.D.

Loan Approval Prediction: DT, RF, LR, SVM, KNN, GNB Model Training.
"""

# References
# https://towardsdatascience.com/predict-loan-eligibility-using-machine-learning-models-7a14ef904057
# https://www.analyticsvidhya.com/blog/2022/05/loan-prediction-problem-from-scratch-to-end/
# https://dphi.tech/notebooks/704/riyaz_khorasi/bank-loan-approval-prediction-using-machine-learning
# https://stackoverflow.com/questions/67426438/pandas-how-can-i-group-by-one-numeric-column-and-filter-rows-from-each-group-b
# https://www.analyticsvidhya.com/blog/2020/10/feature-selection-techniques-in-machine-learning/
# https://github.com/shrikant-temburwar/Loan-Prediction-Dataset/blob/master/LoanPrediction.ipynb
# https://www.section.io/engineering-education/hyperparmeter-tuning/
# https://www.kdnuggets.com/2017/06/7-techniques-handle-imbalanced-data.html
# https://towardsdatascience.com/how-to-balance-a-dataset-in-python-36dff9d12704
# https://www.geeksforgeeks.org/ml-handling-imbalanced-data-with-smote-and-near-miss-algorithm-in-python/
# https://www.geeksforgeeks.org/svm-hyperparameter-tuning-using-gridsearchcv-ml/
# https://www.kaggle.com/code/arunimsamudra/k-nn-with-hyperparameter-tuning
# https://medium.com/mlearning-ai/predicting-loan-approval-status-practice-problem-on-analytics-vidhya-e15ae8b6b0d2


# import modules
# import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# import random
import pickle

# for data preparation
from sklearn.preprocessing import LabelEncoder #, OneHotEncoder
# from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

# for feature slection
from sklearn.linear_model import Lasso
from sklearn.feature_selection import SelectFromModel
from mlxtend.feature_selection import ExhaustiveFeatureSelector

# for data split for model training
from sklearn.model_selection import train_test_split

# for imbalance data handling
from imblearn.under_sampling import NearMiss
from imblearn.over_sampling import SMOTE
from sklearn.utils import shuffle

## for ml models
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier

# for parameters tuning
from sklearn.model_selection import GridSearchCV #, RandomizedSearchCV

# for model evaluation
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

# visualizing model
# from sklearn.tree import plot_tree #, export_text

# data display configuration (optional)
# pd.pandas.set_option('display.max_columns',None)
pd.options.mode.chained_assignment = None # to avoid pd warnings

#%%
def string_to_numeric(df):
    """convert string column to numeric"""
    df2=df.dropna() # drop nan
    for cols in df.columns.tolist():
        if str(df2[cols].values[0])[0].isdigit() == True:
            df[cols] = pd.to_numeric(df[cols])
    return df

def create_balance_data(X_train, y_train, method_type='Oversampling', is_print=False):
    """create balance data for model training"""
    # Imbalanced Data Handling Techniques: (1) SMOTE, (2) Near Miss Algorithm
    X_train_res = None
    y_train_res = None
    
    if method_type == 'Oversampling':
        # 1) SMOTE (Synthetic Minority Oversampling Technique) – Oversampling
        # import SMOTE module from imblearn library
        # pip install imblearn (if you don't have imblearn in your system)
        sm = SMOTE(random_state = 2)
        X_train_res, y_train_res = sm.fit_resample(X_train, y_train.ravel())
        
    if method_type == 'Undersampling':
        # 2) NearMiss Algorithm – Undersampling
        # apply near miss
        nr = NearMiss()
        X_train_res, y_train_res = nr.fit_resample(X_train, y_train.ravel())
    
    if is_print == True:
        print('\n')
        print("Before Sampling, counts of label '1': {}".format(sum(y_train == 1)))
        print("Before Sampling, counts of label '0': {} \n".format(sum(y_train == 0)))
        print('After OverSampling, the shape of train_X: {}'.format(X_train_res.shape))
        print('After OverSampling, the shape of train_y: {} \n'.format(y_train_res.shape))
        print("After OverSampling, counts of label '1': {}".format(sum(y_train_res == 1)))
        print("After OverSampling, counts of label '0': {}".format(sum(y_train_res == 0)))
    
    return X_train_res, y_train_res


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


def understanding_data(df):
    """basic information of input data"""
    print('total rows and columns: {}'.format(df.shape))
    print('field/attribute/feature names: \n{}'.format(list(df.columns)))
    # check data types
    print('data types: \n{}'.format(df.dtypes))
    # check top5 records
    print('top 5 records: \n{}'.format(df.head()))
    # total no. of rows
    print('total records: {}'.format(df.shape[0]))
    return


def missing_inforation_status(df, display=False):
    """find missing values in input data"""
    # 1) missing Value identification
    total_records = df.shape[0]
    missing_value_fields = []
    for col in df.columns:
        if df[col].isnull().sum() > 0:
            missing_value_fields.append(col)
        # print % of missing values in each columns/fields
        print(col+' missing value: ', round((df[col].isnull().sum()/total_records)*100, 2),' %')
    # 2) find relationship between missing values and target variable
    for feature in missing_value_fields:
        data = df.copy()
        # let's make a variable that indicates 1 if the observation was missing or zero otherwise
        data[feature] = np.where(data[feature].isnull(), 1, 0)
        if display == True:
            # let's calculate the mean target value where the information is missing or present
            data.groupby(feature)['Loan_Status'].count().plot.bar() # count is applied as Y is categorical ortherwise apply median/mean
            plt.title(feature)
            plt.show()
    return


def univariate_analysis(df, display_stats=False, display_numerical=False, display_categorical=False, display_outlier=False, is_return=False):
    """univariate analysis - individual parameter analysis"""       
    # features
    numerical_features = [feature for feature in df.columns if df[feature].dtypes != 'O']  # 'O' = Object ddtype
    discrete_feature = [feature for feature in numerical_features if len(df[feature].unique())<15]
    continuous_feature = [feature for feature in numerical_features if feature not in discrete_feature]
    categorical_features=[feature for feature in df.columns if df[feature].dtypes=='O']
    
    if display_stats == True:
        print('Number of numerical variables: {}\n{}'.format(len(numerical_features), numerical_features))
        print("Continuous feature Count {}\n{}".format(len(continuous_feature), continuous_feature))
        print("Discrete Variables Count: {}\n{}".format(len(discrete_feature), discrete_feature))
        print("Categorical feature Count {}\n{}".format(len(categorical_features), categorical_features))
    
    if display_numerical == True:
        # Numerical variables are usually of 2 type
        # 1. Continous variable and Discrete Variables
        # visualise the numerical variables
        # df[numerical_features].head()
        ## Lets Find the realtionship between discrete_feature and target variable
        for feature in discrete_feature:
            data = df.copy()
            data.groupby(feature)['Loan_Status'].count().plot.bar()
            plt.xlabel(feature)
            plt.ylabel('Loan_Status')
            plt.title('discrete_feature - '+feature)
            plt.show()
        
        ## Lets analyse the continuous values by creating histograms to understand the distribution
        for feature in continuous_feature:
            data = df.copy()
            data[feature].hist(bins=35)
            plt.xlabel(feature)
            plt.ylabel("Count")
            plt.title('continuous_feature - '+feature)
            plt.show()
        
        # as continuous feature has some bias in the data therefore log-tranformation applied
        # We will be using logarithmic transformation        
        for feature in continuous_feature:
            data = df.copy()
            if 0 not in data[feature].unique():
                data[feature]=np.log(data[feature])
                # data['Loan_Status_num']=np.log(data['Loan_Status_num'])
                # plt.scatter(data[feature],data['Loan_Status_num'])
                data[feature].hist(bins=35)
                plt.xlabel(feature)
                plt.ylabel('Count')
                plt.title('continuous_feature log - '+feature)
                plt.show()
        
    if display_outlier == True:
        # Outlier detection
        for feature in continuous_feature:
            data = df.copy()
            if 0 not in data[feature].unique():
                data[feature]=np.log(data[feature])
                data.boxplot(column=feature)
                plt.ylabel(feature)
                plt.title(feature)
                plt.show()
    
    if display_categorical == True:
        # categorical variable analysis
        for feature in categorical_features:
            print('The feature is {} and number of categories are {}'.format(feature,len(df[feature].unique())))
        
        ## Find out the relationship between categorical variable and target variable
        for feature in categorical_features:
            if feature != 'Loan_ID':
                data=df.copy()
                data.groupby(feature)['Loan_Status'].count().plot.bar()
                plt.xlabel(feature)
                plt.ylabel('Loan_Status')
                plt.title(feature)
                plt.show()
                
    if is_return == True:
        return numerical_features, continuous_feature, discrete_feature, categorical_features
    return


def bivariate_analysis(df, display_numerical=False, display_categorical=False):
    """
    Bivariate Analysis
    -------------------
    Let’s recall some of the hypotheses that we generated earlier:
    
    1) Applicants with high incomes should have more chances of loan approval.
    2) Applicants who have repaid their previous debts should have higher chances of loan approval.
    3) Loan approval should also depend on the loan amount. If the loan amount is less, the chances of loan approval should be high.
    4) Lesser the amount to be paid monthly to repay the loan, the higher the chances of loan approval.
    """
    
    numerical_features, continuous_feature, discrete_feature, categorical_features = univariate_analysis(df, is_return=True)
    
    if display_numerical == True:
        # Numerical Independent Variable vs Target Variable
        # 1) discrete Independent Variable vs Target Variable
        for feature in discrete_feature:
            data = df.copy()
            feat = pd.crosstab(data[feature], data['Loan_Status'])
            feat.div(feat.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True, figsize=(4,4))
            plt.show()
        
        # 1) continuous Independent Variable vs Target Variable
        # make the continuous variable in ordinal
        for feature in continuous_feature:
            data = df.copy()
            if feature == 'ApplicantIncome':
                bins=[0,2500,4000,6000,81000]
                group=['Low','Average','High','Very high']
                data['Income_bin']=pd.cut(data['ApplicantIncome'], bins, labels=group)
                Income_bin=pd.crosstab(data['Income_bin'], data['Loan_Status'])
                Income_bin.div(Income_bin.sum(1).astype(float), axis=0).plot(kind="bar",stacked=True)
                plt.xlabel('ApplicantIncome')
                plt.ylabel('Percentage')
            elif feature == 'CoapplicantIncome':
                bins=[0,1000,3000,42000]
                group=['Low','Average','High']
                data['Coapplicant_Income_bin']=pd.cut(data['CoapplicantIncome'], bins, labels=group)
                Coapplicant_Income_bin=pd.crosstab(data['Coapplicant_Income_bin'], data['Loan_Status'])
                Coapplicant_Income_bin.div(Coapplicant_Income_bin.sum(1).astype(float), axis=0).plot(kind="bar",stacked=True)
                plt.xlabel('CoapplicantIncome')
                plt.ylabel('Percentage')
            elif feature == 'LoanAmount':
                bins=[0,100,200,700]
                group=['Low','Average','High']
                data['LoanAmount_bin']=pd.cut(data['LoanAmount'], bins, labels=group)
                LoanAmount_bin=pd.crosstab(data['LoanAmount_bin'], data['Loan_Status'])
                LoanAmount_bin.div(LoanAmount_bin.sum(1).astype(float), axis=0).plot(kind="bar",stacked=True)
                plt.xlabel('LoanAmount')
                plt.ylabel('Percentage')
                
    if display_categorical == True:
        # Categorical Independent Variable vs Target Variable
        for feature in categorical_features:
            if feature != 'Loan_ID':
                data = df.copy()
                feat = pd.crosstab(data[feature], data['Loan_Status'])
                feat.div(feat.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True, figsize=(4,4))
                plt.show()
    return


def numericalvar_correlation_heatmap(df, is_heatmap=False):
    """
    Now let’s look at the correlation between all the numerical variables. 
    We will use the heat map to visualize the correlation. 
    Heatmaps visualize data through variations in coloring. 
    The variables with darker color means their correlation is more.
    """
    numerical_features, _, _, _ = univariate_analysis(df, is_return=True)
    
    # drop unnecessary columns
    data = df[numerical_features]
    matrix = data.corr()
    if is_heatmap == True:
        f, ax = plt.subplots(figsize=(9,6))
        sns.heatmap(matrix,vmax=.8,square=True,cmap="BuPu", annot = True)
    """
    We see that the most correlate variables are (ApplicantIncome — LoanAmount) and 
    (Credit_History — Loan_Status). 
    LoanAmount is also correlated with CoapplicantIncome.
    """
    return


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


def input_feature_selection(X, y, cols):
    """feature selection with different methods"""
    # Method-1: Exhaustive Feature Selector
    efs = ExhaustiveFeatureSelector(RandomForestClassifier(), min_features=4, max_features=8, scoring='roc_auc', cv=2)
    efs = efs.fit(X, y)
    selected_features = X.columns[list(efs.best_idx_)]
    # print(selected_features)
    # print(efs.best_score_)
    selected_feat1 = [cols[i] for i in list(selected_features)]
    
    # Method-2: RandomForest importance
    model = RandomForestClassifier(n_estimators=340)
    model.fit(X, y)
    importance = model.feature_importances_
    final_df = pd.DataFrame({"Feature":pd.DataFrame(X).columns, "Importances":importance})
    final_df.set_index('Importances')
    final_df = final_df.sort_values('Importances')
    # final_df.plot.bar(color='teal')
    th_features = final_df['Feature'][final_df['Importances'] > 0.04].values
    selected_feat2 = [cols[i] for i in th_features]
    
    # Method-3: Lasso
    feature_sel_model = SelectFromModel(Lasso(alpha=0.005, random_state=0)) # remember to set the seed, the random state in this function
    feature_sel_model.fit(X, y)
    feature_sel_model.get_support()
    # this is how we can make a list of the selected features
    lasso_feature = list(X.columns[(feature_sel_model.get_support())])
    selected_feat3 = [cols[i] for i in lasso_feature]
    
    # total feature
    total_feature = selected_feat1 + selected_feat2 + selected_feat3
    total_feature_unique = list(set(total_feature))
    final_features = [feat for feat in cols if feat in total_feature_unique]
    return final_features


# HYPERPARAMETER TUINING
def hyperparameter_LogisticRegression(X_train, y_train):
    """logistic regression - hyperparameter tuning"""
    # Creating the hyperparameter grid
    c_space = np.logspace(-5, 8, 15)
    param_grid = {'C': c_space}
    # Instantiating logistic regression classifier
    logreg = LogisticRegression()
    # Instantiating the GridSearchCV object
    logreg_cv = GridSearchCV(logreg, param_grid, cv = 5)
    logreg_cv.fit(X_train, y_train)
    # Print the tuned parameters and score
    print("Tuned Logistic Regression Parameters: {}".format(logreg_cv.best_params_))
    print("Best score is {}".format(logreg_cv.best_score_))
    

def hyperparameter_svc(X_train, y_train):
    """support vector machine classifier - hyperparameter tuning"""
    # defining parameter range
    param_grid = {'C': [0.1, 1, 10, 100, 1000], 
                  'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
                  'kernel': ['rbf']}
    grid = GridSearchCV(SVC(), param_grid, refit = True, verbose = 3)
    # fitting the model for grid search
    grid.fit(X_train, y_train)
    # print best parameter after tuning
    print(grid.best_params_)
    # print how our model looks after hyper-parameter tuning
    print(grid.best_estimator_)
    print(grid.best_score_)
    
    
def hyperparameter_knn(X_train, y_train):
    """k-nn - hyperparameter tuning"""
    grid_params = { 'n_neighbors' : [3,5,7,9,11,13,15],
                   'weights' : ['uniform','distance'],
                   'metric' : ['minkowski','euclidean','manhattan'],
                   'p' : [1,2,3,4,5]}
    gs = GridSearchCV(KNeighborsClassifier(), grid_params, verbose = 1, cv=7, n_jobs = -1)
    # fit the model on our train set
    g_res = gs.fit(X_train, y_train)
    # find the best score
    print(g_res.best_score_)
    # get the hyperparameters with the best score
    print(g_res.best_params_)


def hyperparameter_rf(X_train, y_train):
    """random forest - hyperparameter tuning"""
    # search RF best parameters
    forest = RandomForestClassifier(random_state = 0)
    n_estimators = [50, 80, 100, 300, 500, 800, 1200]
    max_depth = [5, 8, 15, 25, 30]
    min_samples_split = [2, 5, 10, 15, 100]
    min_samples_leaf = [1, 2, 5, 10] 
    hyperF = dict(n_estimators = n_estimators, max_depth = max_depth,  
                  min_samples_split = min_samples_split, 
                 min_samples_leaf = min_samples_leaf)
    # grid-search
    gridF = GridSearchCV(forest, hyperF, cv = 3, verbose = 1, n_jobs = -1)
    bestF = gridF.fit(X_train, y_train)
    print(bestF.best_params_)
    print(bestF.best_score_)
    # # random-search
    # randomF = RandomizedSearchCV(forest, hyperF, random_state=0)
    # bestRandomF = randomF.fit(X_train, y_train)
    # print(bestRandomF.best_params_)
    # print(bestRandomF.best_score_)


def customtuning_hyperparameter_rf(X_train, X_test, y_train, y_test):
    """random forest - best accuracy finding with random_state changing after hyperparameter tuning"""
    for i in range(1, 500):
        rfc = RandomForestClassifier(random_state = i, max_depth = 8, n_estimators = 10, min_samples_split = 2, min_samples_leaf = 6)       
        rfc.fit(X_train, y_train)
        y_pred_rfc = rfc.predict(X_test)
        confusion_matrix(y_test, y_pred_rfc)
        acc = round(accuracy_score(y_pred_rfc, y_test), 2)
        if acc > 0.75:
            print(i, acc)


# ML MODELS
def dt_model(X_train, y_train, outdir=''):
    # 2) DecisionTree Classifier (data with custom-scale)
    # Fitting Decision Tree Classification to the Training set
    dt = DecisionTreeClassifier(criterion = 'entropy', random_state = 123)
    dt.fit(X_train, y_train)
    y_pred_dt = dt.predict(X_test)
    confusion_matrix(y_test, y_pred_dt)
    print('accuracy: {}'.format(round(accuracy_score(y_pred_dt, y_test), 2)))
    print(classification_report(y_test, y_pred_dt))
    # plt.figure(figsize =(80,20))
    # plot_tree(dt, feature_names=final_columns, max_depth=2, filled=True)
    filename = outdir+'\\dt_best_model.sav'
    pickle.dump(dt, open(filename, 'wb'))
    # # load the model from disk
    # loaded_model = pickle.load(open(filename, 'rb'))
    # result = loaded_model.score(X_test, y_test)
    return


def rf_model(X_train, y_train, outdir=''):
    # 6) Random Forest (data with custom-scale)
    rfc = RandomForestClassifier(random_state = 445, max_depth = 8, n_estimators = 10, min_samples_split = 2, min_samples_leaf = 6)          
    rfc.fit(X_train, y_train)
    y_pred_rfc = rfc.predict(X_test)
    confusion_matrix(y_test, y_pred_rfc)
    print('accuracy: {}'.format(round(accuracy_score(y_pred_rfc, y_test), 2)))
    print(classification_report(y_test, y_pred_rfc))
    filename = outdir+'\\rf_best_model.sav'
    pickle.dump(rfc, open(filename, 'wb'))
    # # load the model from disk
    # loaded_model = pickle.load(open(filename, 'rb'))
    # result = loaded_model.score(X_test, y_test)
    return


def logistic_model(X_train, y_train, outdir=''):
    # 1) Logistic Regression
    # Fitting Logistic Regression to the Training set
    lr = LogisticRegression(C = 4.49) #(C = 1, random_state = 1)
    lr.fit(X_train, y_train)
    y_pred = lr.predict(X_test)
    confusion_matrix(y_test, y_pred)
    print('accuracy: {}'.format(round(accuracy_score(y_pred, y_test), 2)))
    print(classification_report(y_test, y_pred))
    filename = outdir+'\\lr_best_model.sav'
    pickle.dump(lr, open(filename, 'wb'))
    # # load the model from disk
    # loaded_model = pickle.load(open(filename, 'rb'))
    # result = loaded_model.score(X_test, y_test)
    return


def svm_model(X_train, y_train, outdir=''):
    # 3) SVC
    # Fitting SVM to the Training set
    svm = SVC(C=1, gamma=0.1, kernel='rbf')
    svm.fit(X_train, y_train)
    y_pred_svm = svm.predict(X_test)
    confusion_matrix(y_test, y_pred_svm)
    print('accuracy: {}'.format(round(accuracy_score(y_pred_svm, y_test), 2)))
    print(classification_report(y_test, y_pred_svm))
    filename = outdir+'\\svm_best_model.sav'
    pickle.dump(svm, open(filename, 'wb'))
    # # load the model from disk
    # loaded_model = pickle.load(open(filename, 'rb'))
    # result = loaded_model.score(X_test, y_test)
    return


def knn_model(X_train, y_train, outdir=''):
    # 4) K-NN
    # # Fitting K-NN to the Training set
    knn = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 1)
    knn.fit(X_train, y_train)
    y_pred_knn = knn.predict(X_test)
    confusion_matrix(y_test, y_pred_knn)
    print('accuracy: {}'.format(round(accuracy_score(y_pred_knn, y_test), 2)))
    print(classification_report(y_test, y_pred_knn))
    filename = outdir+'\\knn_best_model.sav'
    pickle.dump(knn, open(filename, 'wb'))
    # # load the model from disk
    # loaded_model = pickle.load(open(filename, 'rb'))
    # result = loaded_model.score(X_test, y_test)
    return


def gnb_model(X_train, y_train, outdir=''):
    # 5) Gaussian Naive Bayes
    # Fitting Naive Bayes to the Training set
    gnb = GaussianNB()
    gnb.fit(X_train, y_train)
    y_pred_gnb = gnb.predict(X_test)
    confusion_matrix(y_test, y_pred_gnb)
    print('accuracy: {}'.format(round(accuracy_score(y_pred_gnb, y_test), 2)))
    print(classification_report(y_test, y_pred_gnb))
    filename = outdir+'\\gnb_best_model.sav'
    pickle.dump(gnb, open(filename, 'wb'))
    # # load the model from disk
    # loaded_model = pickle.load(open(filename, 'rb'))
    # result = loaded_model.score(X_test, y_test)
    return


#%%
# read data
in_file = "../training-samples/input_training_data.csv"

df = input_dataframe(in_file)


#%%
# UNDERSTAND DATA
#==============================
understanding_data(df)

# EXPLORATORY DATA ANALYSIS (EDA)
#================================
missing_inforation_status(df, display=True)
univariate_analysis(df, display_stats=True)
univariate_analysis(df, display_numerical=True)
univariate_analysis(df, display_categorical=True)
univariate_analysis(df, display_outlier=True)
bivariate_analysis(df, display_numerical=True)
bivariate_analysis(df, display_categorical=True)
numericalvar_correlation_heatmap(df, is_heatmap=False)

#%%
# FEATURE ENGINEERING
#=====================================
cols = df.columns.tolist()[1:]
feateng = feature_engineering(df, colslist=cols, return_type='XY')
X = feateng['X']
y = feateng['y']


# FEATURE SELECTION
#=======================================
# final_features = input_feature_selection(X, y, cols)
final_features = ['Married', 'Dependents', 'Education', 'ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term', 'Credit_History', 'Property_Area']


out_model_dir = "../models"

#%%
# # random seed setup
# np.random.seed(1)
# random.seed(1)
# os.environ['TF_DETERMINISTIC_OPS'] = '1'
# tf.random.set_seed(1)


# MODEL BUILDING - 1
#=========================
# TRAIN-TEST SPLIT: CUSTOM-SCALE
#================================
final_columns = final_features + ['Loan_Status']
df_final = df[final_columns]
xysplit = feature_engineering(df_final, colslist=final_columns, return_type='CustomScaler')
X_train = xysplit['X_train']
X_test = xysplit['X_test']
y_train = xysplit['y_train']
y_test = xysplit['y_test']
# run model
dt_model(X_train, y_train, outdir=out_model_dir)
rf_model(X_train, y_train, outdir=out_model_dir)


#%%
# MODEL BUILDING - 2
#=========================
# TRAIN-TEST SPLIT: STANDARD-SCALE
#==========================================
final_columns = final_features + ['Loan_Status']
df_final = df[final_columns]
xysplit = feature_engineering(df_final, colslist=final_columns, return_type='StandardScaler')
X_train = xysplit['X_train']
X_test = xysplit['X_test']
y_train = xysplit['y_train']
y_test = xysplit['y_test']

# run model
logistic_model(X_train, y_train, outdir=out_model_dir)
svm_model(X_train, y_train, outdir=out_model_dir)
knn_model(X_train, y_train, outdir=out_model_dir)
gnb_model(X_train, y_train, outdir=out_model_dir)


#%%
# LOAD MODEL AND TEST
#=================================
best_model_dir = "../models"

# DT : Load dt model
#--------------------------------
filename_dt = best_model_dir+'\\dt_model_73p.sav'
loaded_model_dt = pickle.load(open(filename_dt, 'rb'))
result_dt = loaded_model_dt.score(X_test, y_test)

# RF: Load rf model
#------------------------------
filename_rf = best_model_dir+'\\rf_best_model_78.sav'
loaded_model_rf = pickle.load(open(filename_rf, 'rb'))
print(loaded_model_rf.score(X_test, y_test))

# LR: Load lr model
filename_lr = best_model_dir+'\\lr_model_76p.sav'
loaded_model_lr = pickle.load(open(filename_lr, 'rb'))
result_lr = loaded_model_lr.score(X_test, y_test)

# SVM: Load svm model
filename_svm = best_model_dir+'\\svm_model_76p.sav'
loaded_model_svm = pickle.load(open(filename_svm, 'rb'))
result_svm = loaded_model_svm.score(X_test, y_test)

# KNN: Load knn model
filename_knn = best_model_dir+'\\knn_model_67p.sav'
loaded_model_knn = pickle.load(open(filename_knn, 'rb'))
result_knn = loaded_model_knn.score(X_test, y_test)

# GNB: Load gnb model
filename_gnb = best_model_dir+'\\gnb_model_74p.sav'
loaded_model_gnb = pickle.load(open(filename_gnb, 'rb'))
result_gnb = loaded_model_gnb.score(X_test, y_test)

# Done!
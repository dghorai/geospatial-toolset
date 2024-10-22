import pandas as pd

from imblearn.under_sampling import NearMiss
from imblearn.over_sampling import SMOTE
from sklearn.utils import shuffle
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split


def string_to_numeric(df):
    """convert string column to numeric"""
    df2 = df.dropna()  # drop nan
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
        sm = SMOTE(random_state=2)
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
        print('After OverSampling, the shape of train_y: {} \n'.format(
            y_train_res.shape))
        print("After OverSampling, counts of label '1': {}".format(
            sum(y_train_res == 1)))
        print("After OverSampling, counts of label '0': {}".format(
            sum(y_train_res == 0)))

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
            df_yes = target_yes.sample(frac=p/n_yes)
        else:
            df_yes = target_yes.copy()
        # final df
        df_final = pd.concat([df_yes, target_no], axis=0)
    else:
        p = n_yes + int((n_yes*t)/100)
        if p < n_no:
            df_no = target_no.sample(frac=p/n_no)
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
    # replace header with 0-n values
    df3 = pd.DataFrame(df.iloc[:, 0:df.shape[-1]].values)
    df3 = string_to_numeric(df3)
    categorical_features = [
        feature for feature in df3.columns if df3[feature].dtypes == 'O']
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
    X.interpolate(method='linear', limit_direction='backward', inplace=True)
    # interpolate in forward order across the column
    X.interpolate(method='linear', limit_direction='forward', inplace=True)

    # statement
    if return_type == 'XY':
        result = {'X': X, 'y': y}
    elif return_type == 'StandardScaler':
        # Perform Feature Scaling
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=0, shuffle=True)
        # https://towardsdatascience.com/what-and-why-behind-fit-transform-vs-transform-in-scikit-learn-78f915cf96fe
        sc = StandardScaler()
        X_train = sc.fit_transform(X_train)
        X_test = sc.transform(X_test)
        result = {'X_train': X_train, 'X_test': X_test,
                  'y_train': y_train, 'y_test': y_test}
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
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=0, shuffle=True)
        result = {'X_train': X_train.to_numpy(), 'X_test': X_test.to_numpy(),
                  'y_train': y_train, 'y_test': y_test}
    else:
        print('scale not define')
    return result

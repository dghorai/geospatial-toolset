# -*- coding: utf-8 -*-
"""
Created on Fri Sep 16 21:11:22 2022

@author: Debabrata Ghorai, Ph.D.

GUI Application: Loan Approval Prediction.
"""

# GUI Reference-1: https://www.tutorialsteacher.com/python/create-gui-using-tkinter-python
# GUI Reference-2: https://www.knowledgehut.com/tutorials/python-tutorial/python-tkinter
# GUI Reference-3: https://python-textbok.readthedocs.io/en/1.0/Introduction_to_GUI_Programming.html

# Import Module
import os
os.chdir(r"../Loan-Approval-Prediction")
import pandas as pd
pd.options.mode.chained_assignment = None # to avoid pd warnings
# import numpy as np
# import random
import tensorflow as tf
import pickle

from tkinter import Tk, Label, StringVar, Entry, Button, E
from tkinter.ttk import Combobox
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle

#%%
class LoanApprovalPrediction:
    
    """
    A Machine Learning Model for Loan Approval Prediction:
    
    CustomScaler - division values
    ------------------------------
    0 Married --> 1
    1 Dependents --> 0 (ignore division)
    2 Education --> 0 (ignore division)
    3 ApplicantIncome --> 2500
    4 CoapplicantIncome --> 0 (ignore division)
    5 LoanAmount --> 120.0
    6 Loan_Amount_Term --> 360
    7 Credit_History --> 1
    8 Property_Area --> 1
    
    Categorical to Numeric
    ----------------------
    0 Married: No --> 0; Yes --> 1; blank --> 2
    2 Education: Graduate --> 0; Not Graduate --> 1
    8 Property_Area: Urban --> 2; Rural --> 0; Semiurban --> 1
    """
    
    def __init__(self, master):
        """initilizer"""
        self.master = master
        master.geometry("665x165+300+300") #length x height + xposition + yposition
        master.title("Loan Approval Prediction")
        
        # GUI Layout
        # Label (widget) on window and Give input to Label
        #--------------------------------------------------
        # Applicant Name
        an = Label(root, text="Applicant Name: ", width=25)
        an.grid(row=0, column=0)
        v1 = StringVar()
        v1.set("applicant name")
        self.input1 = Entry(root, textvariable=v1, width=23)
        self.input1.grid(row=0, column=1)
        
        # Property Area
        pa = Label(root, text="Property Area: ", width=25)
        pa.grid(row=0, column=2)
        self.input2 = StringVar()
        data2=("Rural", "Semiurban", "Urban")
        cb2=Combobox(root, values=data2, textvariable=self.input2)
        cb2.grid(row=0, column=3)
        
        # Married
        mr = Label(root, text="Married: ", width=25)
        mr.grid(row=2, column=0)
        self.input3 = StringVar()
        data3=("No", "Yes", "None")
        cb3=Combobox(root, values=data3, textvariable=self.input3)
        cb3.grid(row=2, column=1)
        
        # Dependents
        dd = Label(root, text="Dependents: ", width=25)
        dd.grid(row=2, column=2)
        self.input4 = StringVar()
        data4=("0", "1", "2", "3+")
        cb4=Combobox(root, values=data4, textvariable=self.input4)
        cb4.grid(row=2, column=3)
        
        # Education
        ed = Label(root, text="Education: ", width=25)
        ed.grid(row=4, column=0)
        self.input5 = StringVar()
        data5=("Graduate", "Not Graduate")
        cb5=Combobox(root, values=data5, textvariable=self.input5)
        cb5.grid(row=4, column=1)
        
        # Applicant Income
        ai = Label(root, text="Applicant Income: ", width=25)
        ai.grid(row=4, column=2)
        v2 = StringVar()
        self.input6 = Entry(root, textvariable=v2, width=23)
        self.input6.grid(row=4, column=3)
        
        # Coapplicant Income
        ci = Label(root, text="Coapplicant Income: ", width=25)
        ci.grid(row=6, column=0)
        v3 = StringVar()
        self.input7 = Entry(root, textvariable=v3, width=23)
        self.input7.grid(row=6, column=1)
        
        # Loan Amount
        la = Label(root, text="Loan Amount: ", width=25)
        la.grid(row=6, column=2)
        v4 = StringVar()
        self.input8 = Entry(root, textvariable=v4, width=23)
        self.input8.grid(row=6, column=3)
        
        # Loan Amount Term
        lat = Label(root, text="Loan Amount Term: ", width=25)
        lat.grid(row=8, column=0)
        self.input9 = StringVar()
        data9=(12, 26, 60, 84, 120, 180, 240, 300, 360, 480)
        cb9=Combobox(root, values=data9, textvariable=self.input9)
        cb9.grid(row=8, column=1)
        
        # Credit History
        ch = Label(root, text="Credit History: ", width=25)
        ch.grid(row=8, column=2)
        self.input10 = StringVar()
        data10=(0, 1)
        cb10=Combobox(root, values=data10, textvariable=self.input10)
        cb10.grid(row=8, column=3)
        
        # Set output to label
        #-----------------------------------------
        # Predict Result
        pred = Label(root, text="Prediction Result: ", width=25)
        pred.grid(row=10, column=0)
        self.out = StringVar()
        out_entry = Entry(root, textvariable=self.out, width=23)
        out_entry.grid(row=10, column=1)
        
        # Create Button for Operation
        #-------------------------------------------
        # Predict button
        cal = Button(root, text = "Predict", command = self.ModelPrediction)
        cal.grid(row=12, column=1, sticky=E, pady=4)
        # Reset value button
        reset = Button(root, text = "Reset Values!", command = self.ResVal)
        reset.grid(row=12, column=2, sticky=E, pady=4)
    
    def ModelPrediction(self):
        """get inputs and predict result"""
        try:
            p1 = str(self.input1.get())
            print(p1)
            p2 = str(self.input2.get())
            p3 = str(self.input3.get())
            p4_str = str(self.input4.get())
            p5 = str(self.input5.get())
            p6 = float(self.input6.get())
            p7 = float(self.input7.get())
            p8 = float(self.input8.get())
            p9 = int(self.input9.get())
            p10 = int(self.input10.get())
            # convert str to int
            if p4_str == "3+":
                p4 = 4
            else:
                p4 = int(p4_str)
            # create dataframe
            disc = {'Married':[p3], 'Dependents':[p4], 'Education': [p5], 'ApplicantIncome':[p6],
                    'CoapplicantIncome':[p7], 'LoanAmount':[p8], 'Loan_Amount_Term':[p9],
                    'Credit_History':[p10], 'Property_Area':[p2]}
            df = pd.DataFrame(disc)
            X_test_custom, X_test_standard = self.model_inputs(df)
            new_val = self.load_models(X_test_custom, X_test_standard)
            self.out.set(new_val)
        except ValueError:
            pass
    
    def ResVal(self):
        """reset gui inputs/results"""
        self.input1.delete(0, 'end')
        self.input2.set('')
        self.input3.set('')
        self.input4.set('')
        self.input5.set('')
        self.input6.delete(0, 'end')
        self.input7.delete(0, 'end')
        self.input8.delete(0, 'end')
        self.input9.set('')
        self.input10.set('')
        self.out.set("None")
        
    def create_partial_balanced_data(self, df, th='15%'):
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
    
    def input_dataframe(self, in_file):
        # read data
        df = pd.read_csv(in_file)
        # replace '3+' from ‘Dependents’ columns
        df['Dependents'].replace('3+', 4, inplace=True)
        df['Dependents'] = df['Dependents'].astype(float)
        df = self.string_to_numeric(df)
        # create partial balanced data
        df = self.create_partial_balanced_data(df, th='20%')
        return df
    
    def string_to_numeric(self, df):
        """convert string column to numeric"""
        df2=df.dropna() # drop nan
        for cols in df.columns.tolist():
            if str(df2[cols].values[0])[0].isdigit() == True:
                df[cols] = pd.to_numeric(df[cols])
        return df
    
    def category_to_numeric(self, df3, categorical_features):
        # 0 -> Married; 2 -> Education; 8 -> Property_Area
        disc = {0:{'No':0, 'Yes':1, 'None':2},
                2:{'Graduate':0, 'Not Graduate':1},
                8:{'Rural':0, 'Semiurban':1, 'Urban':2}}
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
            xtrain_df = pd.read_csv("training-samples/xtrain_ann_sc81p.csv")
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
        df3 = pd.DataFrame(df.iloc[:, 0:df.shape[-1]].values) # replace header with 0-n values
        df3 = self.string_to_numeric(df3)
        categorical_features = [feature for feature in df3.columns if df3[feature].dtypes=='O']
        # apply LabelEncoder to categorical variable
        df3 = self.category_to_numeric(df3, categorical_features)
        # create final inputs
        X_test_custom = self.feature_engineering(df3, return_type='CustomScaler')
        X_test_standard = self.feature_engineering(df3, return_type='StandardScaler')
        return X_test_custom, X_test_standard
    
    def load_models(self, X_test_custom, X_test_standard):
        """predict result"""
        prediction = {0:'N', 1:'Y'}
        # DT : Load dt model
        model_dt = pickle.load(open('models/dt_model_73p.sav', 'rb'))
        y_pred1 = prediction[model_dt.predict(X_test_custom)[0]]
        # RF: Load rf model
        model_rf = pickle.load(open('models/rf_model_80p_latest.sav', 'rb'))
        y_pred2 = prediction[model_rf.predict(X_test_custom)[0]]
        # LR: Load lr model
        model_lr = pickle.load(open('models/lr_model_76p.sav', 'rb'))
        y_pred3 = prediction[model_lr.predict(X_test_standard)[0]]
        # SVM: Load svm model
        model_svm = pickle.load(open('models/svm_model_76p.sav', 'rb'))
        y_pred4 = prediction[model_svm.predict(X_test_standard)[0]]
        # KNN: Load knn model
        model_knn = pickle.load(open('models/knn_model_67p.sav', 'rb'))
        y_pred5 = prediction[model_knn.predict(X_test_standard)[0]]
        # GNB: Load gnb model
        model_gnb = pickle.load(open('models/gnb_model_74p.sav', 'rb'))
        y_pred6 = prediction[model_gnb.predict(X_test_standard)[0]]
        # ANN : Load ann model
        model_ann = tf.keras.models.load_model('models/ann_model_81p_latest.h5')
        y_pred = (model_ann.predict(X_test_standard) > 0.5)
        y_pred7 = prediction[0 if y_pred[0][0] == False else 1]
        # final prediction
        y_pred_list = [y_pred1, y_pred2, y_pred3, y_pred4, y_pred5, y_pred6, y_pred7]
        n_count = y_pred_list.count("N")
        y_count = y_pred_list.count("Y")
        if y_count > n_count:
            result = 'Yes'
        else:
            result = 'No'
        return result


if __name__ == '__main__':
    # Main window
    root = Tk()
    mygui = LoanApprovalPrediction(root)
    root.mainloop()


# -*- coding: utf-8 -*-
"""
Created on Thu Oct 20 11:01:24 2022

@author: gitan
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import RFE
import os
path = "C:/Users/gitan/College/3402 Semester 3/Ai/Assignment 3/Exercose#2_Gitansh/"
filename = 'Ecom Expense.csv'
fullpath = os.path.join(path,filename)

ecom_exp_gitansh = pd.read_csv(fullpath,sep=',')


ecom_exp_gitansh.head(3)
ecom_exp_gitansh.tail()
ecom_exp_gitansh.shape
ecom_exp_gitansh.columns
ecom_exp_gitansh.dtypes
ecom_exp_gitansh.isnull().sum()

gender_dummies = pd.get_dummies(ecom_exp_gitansh.Gender, prefix="gender")
cityTier_dummies= pd.get_dummies(ecom_exp_gitansh["City Tier"], prefix="City")
ecom_exp_gitansh = pd.concat([ecom_exp_gitansh, gender_dummies, cityTier_dummies])
ecom_exp_gitansh=ecom_exp_gitansh.drop(columns=['Transaction ID'])
ecom_exp_gitansh=ecom_exp_gitansh.drop(columns=['Gender', 'City Tier'])


def min_max(df):
    df_norm = df.copy()
    for col in df_norm.columns:
        df_norm[col] = (df[col] - df[col].min()) / (df[col].max() - df[col].min())
        return df_norm

df_ecom_norm = min_max(ecom_exp_gitansh)
df_ecom_norm.head(2)

df_ecom_norm.hist(figsize=(9,10), bins=(30))
pd.plotting.scatter_matrix(df_ecom_norm[['Age ', 'Monthly Income', 'Transaction Time', 'Total Spend']], figsize=[13, 15],hist_kwds={'bins':30}, alpha=0.4, diagonal='kde')
df_ecom_norm = df_ecom_norm.rename(columns={'Age ': 'Age', ' Items ':'Items', 'Monthly Income': 'Monthly_Income', 'Transaction Time':'Transaction_Time', 'Total Spend':'Total_Spend','gender_Male':'gender Male','gender_Female': 'gender Female', 'City_Tier 1': 'City_Tier 1', 'City_Tier 2':'City_Tier 2', 'City_Tier 3':'City_Tier 3'})

df_x_gitansh = df_ecom_norm[['Monthly_Income', 'Transaction_Time', 'gender Female', 'gender Male', 'City_Tier 1', 'City_Tier 2', 'City_Tier 3']]


df_y_gitansh = df_ecom_norm['Total_Spend']

X_train_gitansh, X_test_gitansh, Y_train_gitansh, Y_test_gitansh = train_test_split(df_x_gitansh, df_y_gitansh, test_size=0.35, random_state=17)

model = LinearRegression()

model.fit(X_train_gitansh, Y_train_gitansh)
print(model.intercept_)
print(model.coef_)
model.score(X_test_gitansh, Y_test_gitansh)

df_ecom_norm = df_ecom_norm.rename(columns={'Age ': 'Age', ' Items ':'Items', 'Monthly Income': 'Monthly_Income', 'Transaction Time':'Transaction_Time', 'Total Spend':'Total_Spend','gender_Male':'gender Male','gender_Female': 'gender Female', 'City_Tier 1': 'City_Tier 1', 'City_Tier 2':'City_Tier 2', 'City_Tier 3':'City_Tier 3'})

df_x_gitansh = df_ecom_norm[['Monthly_Income', 'Transaction_Time', 'Record', 'gender Female', 'gender Male', 'City_Tier 1', 'City_Tier 2', 'City_Tier 3']]


df_y_gitansh = df_ecom_norm['Total_Spend']

X_train_gitansh1, X_test_gitansh1, Y_train_gitansh1, Y_test_gitansh1 = train_test_split(df_x_gitansh, df_y_gitansh, test_size=0.35, random_state=17)

mode1 = LinearRegression()

model.fit(X_train_gitansh1, Y_train_gitansh1)
print(model.intercept_)
print(model.coef_)
model.score(X_test_gitansh1, Y_test_gitansh1)



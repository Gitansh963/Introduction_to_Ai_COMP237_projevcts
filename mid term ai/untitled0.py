# -*- coding: utf-8 -*-
"""
Created on Wed Oct 26 13:10:20 2022

@author: gitan
"""

import pandas as pd
import os
import matplotlib.pyplot as plt
path = "C:/Users/gitan/OneDrive/Desktop/mid term ai/"
filename = 'Fish.csv'
fullpath = os.path.join(path,filename)
df_gitansh = pd.read_csv(fullpath)
#1
df_gitansh.columns
df_gitansh.dtypes
df_gitansh['Species'].unique()

df_gitansh.head(5)
df_gitansh.isnull()
df_gitansh.isnull().sum()

#2

df_gitansh.hist(figsize= (9,10), bins=8)

plt.scatter(df_gitansh['Weight'],df_gitansh['Length1'], alpha=0.5)
plt.title('gitansh_WL_scatter')
plt.xlabel("Length1")
plt.ylabel("Weight")

avg_length_col = df_gitansh[['Length1', 'Length2','Length3']].mean(axis=1)
df_gitansh['Avg_len_gitansh '] = avg_length_col

df_gitansh.drop('Length1', axis=1,inplace=True)
df_gitansh.drop('Length2', axis=1,inplace=True)
df_gitansh.drop('Length3', axis=1,inplace=True)



df_gitansh.drop(columns= ['Height'], axis= 1, inplace = True)
species_dummies= pd.get_dummies(df_gitansh["Species"], prefix="specie")
column_name=df_gitansh.columns.values.tolist()
df_gitansh=df_gitansh[column_name].join(species_dummies)

df_gitansh.drop(columns= ['Species'], axis= 1, inplace = True)

df_gitansh.head(5)

df_gitansh_numeric = df_gitansh


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

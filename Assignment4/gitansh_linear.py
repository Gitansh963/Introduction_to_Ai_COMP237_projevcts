# -*- coding: utf-8 -*-
"""
Created on Sat Oct 29 02:02:51 2022

@author: gitan
"""

import pandas as pd
import os
path = "C:/Users/gitan/College/3402 Semester 3/Ai/Assignment4"
filename = 'titanic.csv'
fullpath = os.path.join(path,filename)
titanic_gitansh = pd.read_csv(fullpath)

titanic_gitansh.head(3)
titanic_gitansh.shape
titanic_gitansh.info()
titanic_gitansh.dtypes

"""
Columns that are not usefull
pasengerId
Name
ticket
cabin
"""

titanic_gitansh['Sex'].unique()
titanic_gitansh['Pclass'].unique()


#c
import matplotlib.pyplot as plt
pd.crosstab(titanic_gitansh.Survived,titanic_gitansh.Pclass).plot(kind='bar')
plt.title('bar_gitansh')
plt.xlabel('Survived')
plt.ylabel('Passenger class')


pd.crosstab(titanic_gitansh.Survived,titanic_gitansh.Sex).plot(kind='bar')
plt.title('bar_gitansh')
plt.xlabel('Survived')
plt.ylabel('Sex')


pd.plotting.scatter_matrix(titanic_gitansh[['Sex', 'Pclass', 'Fare', 'SibSp','Parch']], figsize=[9, 10],hist_kwds={'bins':30}, alpha=0.4, diagonal='kde')

#d

titanic_gitansh.drop(columns=['PassengerId'], axis= 1, inplace = True)
titanic_gitansh.drop(columns=['Name'], axis= 1, inplace = True)
titanic_gitansh.drop(columns=['Ticket'], axis= 1, inplace = True)
titanic_gitansh.drop(columns=['Cabin'], axis= 1, inplace = True)

sex_dummies = pd.get_dummies(titanic_gitansh['Sex'],prefix="Sex")

embarked_dummies = pd.get_dummies(titanic_gitansh['Embarked'],prefix="Emb")

column_name=titanic_gitansh.columns.values.tolist()
titanic_gitansh=titanic_gitansh[column_name].join(sex_dummies)


column_name=titanic_gitansh.columns.values.tolist()
titanic_gitansh=titanic_gitansh[column_name].join(embarked_dummies)

titanic_gitansh.head(1)
titanic_gitansh.drop(columns=['Sex'], axis= 1, inplace = True)
titanic_gitansh.drop(columns=['Embarked'], axis= 1, inplace = True)

titanic_gitansh['Age'].fillna(int(titanic_gitansh['Age'].mean()), inplace=True)
titanic_gitansh.dtypes
titanic_gitansh = titanic_gitansh.astype(float)
titanic_gitansh.dtypes
titanic_gitansh.info()

def min_max(df):
    df_norm = df.copy()
    for col in df_norm.columns:
        df_norm[col] = (df[col] - df[col].min()) / (df[col].max() - df[col].min())
        return df_norm

numeric_titanic_gitansh = min_max(titanic_gitansh)

numeric_titanic_gitansh.head(2)
numeric_titanic_gitansh.hist(figsize = (9,10))


from sklearn.model_selection import train_test_split
features_cols = ['Pclass','Age','SibSp','Parch','Fare','Sex_female','Sex_male','Emb_C','Emb_Q','Emb_S']
x_gitansh = numeric_titanic_gitansh[features_cols]
y_gitansh =numeric_titanic_gitansh['Survived']


X_train_gitansh, X_test_gitansh, Y_train_gitansh, Y_test_gitansh = train_test_split(x_gitansh,y_gitansh, test_size=0.30, random_state=17)

#e
from sklearn import linear_model
import numpy as np
gitansh_model = linear_model.LogisticRegression(solver='lbfgs')
gitansh_model.fit(X_train_gitansh, Y_train_gitansh)

pd.DataFrame(zip(X_train_gitansh.columns, np.transpose(gitansh_model.coef_)))


from sklearn.model_selection import cross_val_score
scores = cross_val_score(linear_model.LogisticRegression(solver='lbfgs'), X_train_gitansh, Y_train_gitansh, scoring='accuracy', cv=10)
print(scores)
print(scores.mean())
print(scores.min())
print(scores.max())

y_pred_train_gitansh = gitansh_model.predict_proba(X_train_gitansh)
print(y_pred_train_gitansh)
type(y_pred_train_gitansh)

y_pred_train_gitansh_flag = y_pred_train_gitansh[:,1]>0.5
print(y_pred_train_gitansh_flag)

from sklearn import metrics
from sklearn.metrics import confusion_matrix
accuracy_score_train_gitansh = metrics.accuracy_score(Y_train_gitansh, y_pred_train_gitansh_flag)
print(accuracy_score_train_gitansh)

confusion_matrix_train_gitansh = confusion_matrix(Y_train_gitansh, y_pred_train_gitansh_flag)
print (confusion_matrix_train_gitansh)

classification_report_train_gitansh = metrics.classification_report(Y_train_gitansh, y_pred_train_gitansh_flag)
print(classification_report_train_gitansh)


# loop for changing the test size 


features_cols = ['Pclass','Age','SibSp','Parch','Fare','Sex_female','Sex_male','Emb_C','Emb_Q','Emb_S']
x_loop_gitansh = numeric_titanic_gitansh[features_cols]
y_loop_gitansh =numeric_titanic_gitansh['Survived']

loop_scores_mean = []
loop_scores = []
loop_scores_min = []

loop_scores_max = []

for i in np.arange(0.10, 0.51, 0.05):
    print(i)
    X_loop_train_gitansh, X_loop_test_gitansh, Y_loop_train_gitansh, Y_loop_test_gitansh = train_test_split(x_loop_gitansh,y_loop_gitansh, test_size=i, random_state=17)

    gitansh_loop_model = linear_model.LogisticRegression(solver='lbfgs')
    gitansh_loop_model.fit(X_loop_train_gitansh, Y_loop_train_gitansh)
    
    loopscores = cross_val_score(linear_model.LogisticRegression(solver='lbfgs'), X_loop_train_gitansh, Y_loop_train_gitansh, scoring='accuracy', cv=10)
    loop_scores.append(i)
    loop_scores.append(loopscores)
    loop_scores_mean.append(loopscores.mean())
    loop_scores_max.append(loopscores.max())
    loop_scores_min.append(loopscores.min())
    print(loopscores.mean())
    print(loopscores.min())
    print(loopscores.max())
    
print(loop_scores)
print(loop_scores_mean)
print(loop_scores_min)
print(loop_scores_max)







# testing agian at last
#1
features_cols = ['Pclass','Age','SibSp','Parch','Fare','Sex_female','Sex_male','Emb_C','Emb_Q','Emb_S']
x_re_gitansh = numeric_titanic_gitansh[features_cols]
y_re_gitansh =numeric_titanic_gitansh['Survived']

X_re_train_gitansh, X_re_test_gitansh, Y_re_train_gitansh, Y_re_test_gitansh = train_test_split(x_re_gitansh,y_re_gitansh, test_size=0.30, random_state=17)

gitansh_re_model = linear_model.LogisticRegression(solver='lbfgs')
gitansh_re_model.fit(X_re_train_gitansh, Y_re_train_gitansh)

# pd.DataFrame(zip(X_re_train_gitansh.columns, np.transpose(gitansh_re_model.coef_)))


from sklearn.model_selection import cross_val_score
rescores = cross_val_score(linear_model.LogisticRegression(solver='lbfgs'), X_re_test_gitansh, Y_re_test_gitansh, scoring='accuracy', cv=10)
print(rescores)
print(rescores.mean())
print(rescores.min())
print(rescores.max())
#2
y_pred_gitansh = gitansh_re_model.predict_proba(X_re_test_gitansh)
print(y_pred_gitansh)
type(y_pred_gitansh)

#3

y_pred_gitansh_flag = y_pred_gitansh[:,1]>0.5
print(y_pred_gitansh_flag)

#4
from sklearn import metrics
from sklearn.metrics import confusion_matrix
#5
accuracy_score_gitansh = metrics.accuracy_score(Y_re_test_gitansh, y_pred_gitansh_flag)
print(accuracy_score_gitansh)

#6
confusion_matrix_gitansh = confusion_matrix(Y_re_test_gitansh, y_pred_gitansh_flag)
print (confusion_matrix_gitansh)

#7
classification_report_gitansh = metrics.classification_report(Y_re_test_gitansh, y_pred_gitansh_flag)
print(classification_report_gitansh)

#8 on report

#9 


y_pred_a_gitansh_flag = y_pred_gitansh[:,1]>0.75
print(y_pred_a_gitansh_flag)

accuracy_score_a_gitansh = metrics.accuracy_score(Y_re_test_gitansh, y_pred_a_gitansh_flag)
print(accuracy_score_a_gitansh)

confusion_matrix_a_gitansh = confusion_matrix(Y_re_test_gitansh, y_pred_a_gitansh_flag)
print (confusion_matrix_a_gitansh)

classification_report_a_gitansh = metrics.classification_report(Y_re_test_gitansh, y_pred_a_gitansh_flag)
print(classification_report_a_gitansh)

#10 
# used in the above loop line 112

# y_pred_train_gitansh = gitansh_re_model.predict_proba(X_train_gitansh)
# print(y_pred_train_gitansh)
# type(y_pred_train_gitansh)

# y_pred_train_gitansh_flag = y_pred_train_gitansh[:,1]>0.5
# print(y_pred_train_gitansh_flag)s

# accuracy_score_train_gitansh = metrics.accuracy_score(Y_re_train_gitansh, y_pred_train_gitansh_flag)
# print(accuracy_score_train_gitansh)

# confusion_matrix_train_gitansh = confusion_matrix(Y_re_train_gitansh, y_pred_train_gitansh_flag)
# print (confusion_matrix_train_gitansh)

# classification_report_train_gitansh = metrics.classification_report(Y_re_train_gitansh, y_pred_train_gitansh_flag)
# print(classification_report_train_gitansh)

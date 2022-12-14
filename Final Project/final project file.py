# -*- coding: utf-8 -*-
"""
Created on Tue Nov 29 16:55:44 2022

@author: gitan
"""
import pandas as pd
import os
path = "C:/Users/gitan/College/3402 Semester 3/Ai/Final Project/"

filename = 'Youtube03-LMFAO.csv'

fullpath = os.path.join(path,filename)

project_gitansh = pd.read_csv(fullpath)

import numpy as np
posts = project_gitansh[['CONTENT','CLASS']]
posts_input = posts['CONTENT']
posts_input=pd.DataFrame(posts_input)
import string
punctuation  = list(string.punctuation)
print(punctuation)
def remove_punctuation(text): 
    for punc in punctuation:
        if punc in text:
            text = text.replace(punc, '')
    return text

posts_input['CONTENT'] = posts_input['CONTENT'].apply(remove_punctuation)
posts_output = project_gitansh['CLASS']
posts_output=pd.DataFrame(posts_output, columns=['CLASS'])

posts_features = pd.concat([posts_input,posts_output], axis=1)

posts_features = posts_features.sample(frac=1).reset_index(drop=True)

posts_train=posts_features.sample(frac=0.75,random_state=250)
posts_train = posts_train.reset_index(drop=True)
posts_test=posts_features.drop(posts_train.index)
posts_test = posts_test.reset_index(drop=True)

posts_train_Y = posts_train.drop(columns = ['CONTENT'])
posts_test_Y = posts_test.drop(columns = ['CONTENT'])

posts_train_X = posts_train.drop(columns=['CLASS'])
posts_test_X = posts_test.drop(columns = ['CLASS'])

import nltk
from sklearn.feature_extraction.text import CountVectorizer
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
stopwords2  = STOP_WORDS
print (stopwords2)
count_vectorizer = CountVectorizer(stop_words = stopwords2)
features_cv_train = count_vectorizer.fit_transform(posts_train_X.CONTENT)
from sklearn.feature_extraction.text import TfidfTransformer
tfidf = TfidfTransformer()
tfidf_group = tfidf.fit_transform(features_cv_train)
####
from sklearn.naive_bayes import MultinomialNB
classifier = MultinomialNB().fit( tfidf_group, posts_train_Y)
######
from sklearn.model_selection import cross_val_score
scores = cross_val_score(classifier, tfidf_group, posts_train_Y, cv=5)
print(scores)
print(scores.mean())
print(scores.min())
print(scores.max())

import nltk

from sklearn.feature_extraction.text import CountVectorizer


features_cv_test = count_vectorizer.transform(posts_test_X.CONTENT)

from sklearn.feature_extraction.text import TfidfTransformer

tfidf_test = tfidf.transform(features_cv_test)

predictions = classifier.predict(tfidf_test)
print(predictions)

from sklearn.metrics import classification_report
print(classification_report(posts_test_Y, predictions))

from sklearn.metrics import confusion_matrix
matrix = confusion_matrix(posts_test_Y, predictions)
print(matrix)

final_test_X = ['I love your songs',
                'check this on youtube',
                'Awesome song',
                'really liked the song',
                'LMFAO is best', 
                'one subscriber',
                 ]




#Set the labels for the newly created test set
final_test_Y = [0,1,0,0,0,1]

#Transform the new test set with Count Vectorizer and TfIdf
final_test = count_vectorizer.transform(final_test_X)

final_tfidf= tfidf.transform(final_test)

#Second round of predictions
final_pred = classifier.predict(final_tfidf)

print(final_pred)

#Classification report
print(classification_report(final_test_Y,final_pred))

#Confusion Matrix
final_matrix = confusion_matrix(final_test_Y, final_pred)
print(final_matrix)

########################

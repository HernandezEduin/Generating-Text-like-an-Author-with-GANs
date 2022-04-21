# -*- coding: utf-8 -*-
"""
Created on Fri Apr 22 00:01:31 2022

@author: Eduin Hernandez
"""

from sklearn.neighbors import NearestCentroid
from sklearn.pipeline import Pipeline
from sklearn import metrics
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.datasets import fetch_20newsgroups

from authors import get_train_path, get_train_path
from tools import open_list, shuffle
import numpy as np



x_train_path, y_train = get_train_path()
x_train = open_list(x_train_path)

x_test_path, y_test = get_train_path()
x_test = open_list(x_test_path)

text_clf = Pipeline([('vect', CountVectorizer()),
                      ('tfidf', TfidfTransformer()),
                      ('clf', NearestCentroid()),
                      ])

text_clf.fit(x_train, y_train)

x_shuff, y_shuff = shuffle(x_test, y_test)
predicted = text_clf.predict(x_shuff)

print(metrics.classification_report(y_shuff, predicted))

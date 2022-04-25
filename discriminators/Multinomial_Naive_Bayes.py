# -*- coding: utf-8 -*-
"""
Created on Tue Apr 26 00:31:23 2022

@author: Eduin Hernandez
"""

from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn import metrics
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer


from authors import get_train_path, get_test_path
from tools import open_authors_list, author_text_prepocess, shuffle

x_train_path, y_train = get_train_path()
x_train = open_authors_list(x_train_path)
x_train, y_train = author_text_prepocess(x_train, y_train)
x_train, y_train = shuffle(x_train, y_train)

x_test_path, y_test = get_test_path()
x_test = open_authors_list(x_test_path)
x_test, y_test = author_text_prepocess(x_test, y_test)
x_test, y_test = shuffle(x_test, y_test)

text_clf = Pipeline([('vect', CountVectorizer()),
                     ('tfidf', TfidfTransformer()),
                     ('clf', MultinomialNB()),
                     ])

text_clf.fit(x_train, y_train)


predicted = text_clf.predict(x_test)

print(metrics.classification_report(y_test, predicted))
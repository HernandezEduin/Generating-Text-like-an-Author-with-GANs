# -*- coding: utf-8 -*-
"""
Created on Tue Apr 26 00:38:16 2022

@author: Eduin Hernandez
"""

from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

from discriminator import Discriminator
from authors import get_train_path, get_test_path
from tools import open_authors_list, author_text_prepocess, shuffle

class KNearestNeighbors(Discriminator):
    def __init__(self, name='k_nearest_neighbors', filepath = './models/'):
        super().__init__(name, filepath) 
        self.classifier = Pipeline([('vect', CountVectorizer()),
                             ('tfidf', TfidfTransformer()),
                             ('clf', KNeighborsClassifier()),
                             ])

if __name__ == '__main__':    
    x_train_path, y_train = get_train_path()
    x_train = open_authors_list(x_train_path)
    x_train, y_train = author_text_prepocess(x_train, y_train)
    x_train, y_train = shuffle(x_train, y_train)
    
    x_test_path, y_test = get_test_path()
    x_test = open_authors_list(x_test_path)
    x_test, y_test = author_text_prepocess(x_test, y_test)
    x_test, y_test = shuffle(x_test, y_test)
    
    model = KNearestNeighbors()
    if model.save_exists():
        model.load_model()
    else:
        model.fit(x_train, y_train)
        model.save_model()
    
    predicted = model.predict(x_test)
    print(model.classification_report(y_test, predicted))
    print('Accuracy: %1.2f%%\n' % (100*model.accuracy(y_test, predicted)))
    print('Confusion Matrix:\n', model.confusion_matrix(y_test, predicted))
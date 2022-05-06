# -*- coding: utf-8 -*-
"""
Created on Wed Apr 27 23:58:14 2022

@author: Eduin Hernandez
"""
import numpy as np
from sklearn import metrics

import pickle
import os

class Discriminator():
    def __init__(self, name = 'base', filepath= './models/'):
        self.name = 'base'
        self.filepath = filepath + name + '.sav'
        self.classifier = None
    
    def save_model(self):
        pickle.dump(self.classifier, open(self.filepath, 'wb'))
    
    def load_model(self):
        self.classifier = pickle.load(open(self.filepath, 'rb'))
        
    def save_exists(self):
        return os.path.exists(self.filepath)
      
    def fit(self, x_train, y_train):
        self.classifier.fit(x_train, y_train)
        
    def predict(self, x_test):
        return self.classifier.predict(x_test)
    
    def classification_report(self, truth_label, predicted_label):
        return metrics.classification_report(truth_label, predicted_label)
    
    def confusion_matrix(self, truth_label, predicted_label, labels):
        return metrics.confusion_matrix(truth_label, predicted_label, labels=labels)
    
    def confusion2f1(self, matrix):
        tp = np.diag(matrix)
        precision = tp/matrix.sum(axis=0)
        recall = tp/matrix.sum(axis=1)
        f1_vec = 2/(1/recall + 1/precision)
        f1_weighted = (f1_vec*matrix.sum(axis=0)/matrix.sum()).sum()
        return f1_vec, f1_weighted
    
    def accuracy(self, truth_label, predicted_label):
        return metrics.accuracy_score(truth_label, predicted_label)
    
    def f1(self, truth_label, predicted_label):
        return metrics.f1_score(truth_label, predicted_label, average='weighted')
    
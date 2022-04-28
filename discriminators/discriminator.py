# -*- coding: utf-8 -*-
"""
Created on Wed Apr 27 23:58:14 2022

@author: Eduin Hernandez
"""
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
    
    def confusion_matrix(self, truth_label, predicted_label):
        return metrics.confusion_matrix(truth_label, predicted_label)
    
    def accuracy(self, truth_label, predicted_label):
        return metrics.accuracy_score(truth_label, predicted_label)
    
    
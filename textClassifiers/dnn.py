# -*- coding: utf-8 -*-
"""
Created on Thu May  5 15:09:38 2022

@author: Eduin Hernandez
"""
import numpy as np
from sklearn import metrics

import torch
import torch.nn as nn

import os

class DNN(nn.Module):
    def __init__(self, name='dnn', filepath = './models/'):
        super(DNN, self).__init__() 
        self.name = 'dnn'
        self.filepath = filepath + name + '.pt'
        self.model = None
        self.device = None

    def save_exists(self):
        return os.path.exists(self.filepath)
     
    def forward(self, x):
        pass
    
    def fit(self, dataset_train, epochs=10, batch_size=128):
        pass
        
    def predict(self, x):
        self.model.eval()
        return self.forward(x.to(self.device)).argmax(dim=1).cpu().numpy()
    
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
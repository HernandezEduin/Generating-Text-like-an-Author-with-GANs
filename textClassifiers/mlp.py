# -*- coding: utf-8 -*-
"""
Created on Tue Apr 26 01:00:46 2022

@author: Eduin Hernandez
"""
import os
import pickle

import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer

import torch
import torch.nn as nn
from collections import OrderedDict
from torch.utils.data import TensorDataset, DataLoader
from dnn import DNN

from authors import get_train_path, get_test_path
from tools import open_authors_list, author_text_prepocess, shuffle


class MLP(DNN):
    def __init__(self, in_features: int, out_features: int, hidden: int, layers: int, bias: bool = True,
                 relu: bool = True, dropout: bool = True, name='mlp', filepath='./models/'): 
        super(MLP, self).__init__(name, filepath)
        self.layers = layers
        self.dropout = 0.5
        self.flat = nn.Flatten()
        self.model = nn.Sequential(self._create(in_features, out_features, hidden, layers, bias, dropout))

        self.criterion = nn.CrossEntropyLoss()
        self.criterion_sum = nn.CrossEntropyLoss(reduction='sum')
        self.optimizer = torch.optim.Adam(self.model.parameters())
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
    def _create(self, in_features, out_features, hidden, layers, bias = True, mem_state = None, relu = False, dropout = False):
        if layers == 1:
            d = OrderedDict()
            d['linear0'] = nn.Linear(in_features, out_features, bias=bias)
            return d
        
        d = OrderedDict()
        for i in range(layers):
            if i == 0:
                d['linear' + str(i)] = nn.Linear(in_features, hidden, bias=bias)
                if relu:
                    d['relu' + str(i)] = nn.ReLU()
                if dropout:
                    d['dropout' + str(i)] = nn.Dropout(p=self.dropout)
            elif i == layers - 1:
                d['linear' + str(i)] = nn.Linear(hidden, out_features, bias=bias)
            else:
                d['linear' + str(i)] = nn.Linear(hidden, hidden, bias=bias)
                if relu:
                    d['relu' + str(i)] = nn.ReLU()
                if dropout:
                    d['dropout' + str(i)] = nn.Dropout(p=self.dropout)
        return d
    
    def forward(self, x):
        x = self.flat(x)
        x = self.model(x)
        return x
    
    def fit(self, dataset_train, epochs=10, batch_size=128):
        trainloader = DataLoader(dataset_train, batch_size = batch_size, shuffle=True)
        losses = np.zeros(epochs)
        'Train and Validate Network'
        for epoch in range(epochs):  # loop over the dataset multiple times
            print('Epoch#', epoch + 1)              
            'Training Phase'
            self.model.train()
            running_loss = 0.0
            for inputs, labels in trainloader:
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = inputs.to(self.device), labels.to(self.device)
            
                # zero the parameter gradients
                self.optimizer.zero_grad()
        
                'Foward Propagation'
                output = self.forward(inputs)
                loss = self.criterion(output, labels)
                
                running_loss += self.criterion_sum(output, labels).item()
                
                'Backward Propagation'
                #Automatically calculates the gradients for trainable weights, access as weight.grad
                loss.backward()
        
                #Performs the weight Update
                self.optimizer.step()
            
            losses[epoch] = running_loss / len(trainloader)
        self.optimizer.zero_grad()
        return losses
    
    


def save(model):
    save_filename = './models/mlp.pt'
    torch.save(model, save_filename)
    print('Saved as %s' % save_filename)

def load(load_filename='./models/mlp.pt'):
    return torch.load(load_filename)
    
if __name__ == '__main__': 
    x_train_path, y_train = get_train_path()
    x_train = open_authors_list(x_train_path)
    x_train, y_train = author_text_prepocess(x_train, y_train)
    x_train, y_train = shuffle(x_train, y_train)
    
    if os.path.exists('./models/vectorizer.pkl'):
        vectorizer = pickle.load(open('./models/vectorizer.pkl','rb'))
    else:
        vectorizer = TfidfVectorizer(max_features=30000)
        vectorizer.fit(x_train)
        pickle.dump(vectorizer, open('./models/vectorizer.pkl', 'wb'))
    
    x_train = vectorizer.transform(x_train).toarray()

    x_train = torch.Tensor(x_train)
    y_train = torch.Tensor(y_train).long()

    
    dataset_train = TensorDataset(x_train, y_train)
    
    if os.path.exists('./models/mlp.pt'):
        model = load()
    else:
        model = MLP(x_train.size(1), 6, 512, 4)
        loss = model.fit(dataset_train)
        save(model)
        
        plt.figure()
        plt.plot(loss)
        plt.grid()
        plt.title('Loss')
        
    del x_train
    del y_train

    x_test_path, y_test = get_test_path()
    x_test = open_authors_list(x_test_path)
    x_test, y_test = author_text_prepocess(x_test, y_test)
    x_test, y_test = shuffle(x_test, y_test)
    
    x_test = vectorizer.transform(x_test).toarray()
    x_test = torch.Tensor(x_test)
    
    predicted = model.predict(x_test)
    
    print(model.classification_report(y_test, predicted))
    print('Accuracy: %1.2f%%\n' % (100*model.accuracy(y_test, predicted)))
    print('Confusion Matrix:\n', model.confusion_matrix(y_test, predicted, labels=[0,1,2,3,4,5]))

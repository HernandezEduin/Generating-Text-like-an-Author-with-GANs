# -*- coding: utf-8 -*-
"""
Created on Tue Apr 26 01:00:46 2022

@author: Eduin Hernandez
"""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import metrics

from authors import get_train_path, get_test_path
from tools import open_authors_list, author_text_prepocess, shuffle

import torch
import torch.nn as nn
from collections import OrderedDict
from torch.utils.data import TensorDataset, DataLoader

def TFIDF(X_train, X_test, MAX_NB_WORDS=45000):
    vectorizer_x = TfidfVectorizer(max_features=MAX_NB_WORDS)
    X_train = vectorizer_x.fit_transform(X_train).toarray()
    X_test = vectorizer_x.transform(X_test).toarray()
    return (X_train,X_test)

class MLP(nn.Module):
    def __init__(self, in_features: int, out_features: int, hidden: int, layers: int, bias: bool = True,
                 relu: bool = True, dropout: bool = True):
        super(MLP, self).__init__()
        self.layers = layers
        self.dropout = 0.5
        self.flat = nn.Flatten()
        self.model = nn.Sequential(self._create(in_features, out_features, hidden, layers, bias, dropout))

        self.criterion = nn.CrossEntropyLoss()
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
    
    def predict(self, x):
        self.model.eval()
        return self.forward(x.to(self.device)).argmax(dim=1).cpu().numpy()
    
    def fit(self, dataset_train, epochs=10, batch_size=128):
        trainloader = DataLoader(dataset_train, batch_size = batch_size, shuffle=True)
        
        'Train and Validate Network'
        for epoch in range(epochs):  # loop over the dataset multiple times
            print('Epoch#', epoch + 1)              
            'Training Phase'
            self.model.train()
            for inputs, labels in trainloader:
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = inputs.to(self.device), labels.to(self.device)
            
                # zero the parameter gradients
                self.optimizer.zero_grad()
        
                'Foward Propagation'
                output = self.forward(inputs)
                loss = self.criterion(output, labels)
                
                'Backward Propagation'
                #Automatically calculates the gradients for trainable weights, access as weight.grad
                loss.backward()
        
                #Performs the weight Update
                self.optimizer.step()
        
        self.optimizer.zero_grad()


if __name__ == '__main__': 
    x_train_path, y_train = get_train_path()
    x_train = open_authors_list(x_train_path)
    x_train, y_train = author_text_prepocess(x_train, y_train)
    x_train, y_train = shuffle(x_train, y_train)
    
    x_test_path, y_test = get_test_path()
    x_test = open_authors_list(x_test_path)
    x_test, y_test = author_text_prepocess(x_test, y_test)
    x_test, y_test = shuffle(x_test, y_test)
    
    x_train, x_test = TFIDF(x_train, x_test)
    
    x_train = torch.Tensor(x_train)
    x_test = torch.Tensor(x_test)
    y_train = torch.Tensor(y_train).long()
    y_test = torch.Tensor(y_test).long()
    
    dataset_train = TensorDataset(x_train, y_train)
    
    model = MLP(x_train.size(1), 6, 512, 4)
    model.fit(dataset_train)
    predicted = model.predict(x_test)
    
    print(metrics.classification_report(y_test, predicted))
    print(metrics.confusion_matrix(y_test.numpy(), predicted))
    print(sum(predicted == y_test.numpy())/len(predicted))

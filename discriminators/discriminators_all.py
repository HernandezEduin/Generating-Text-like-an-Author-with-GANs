# -*- coding: utf-8 -*-
"""
Created on Thu Apr 28 00:57:27 2022

@author: Eduin Hernandez
"""
from Bagging import Bagging 
from Decision_Tree import DecisionTree
from K_Nearest_Neighbors import KNearestNeighbors
from Multinomial_Naive_Bayes import NaiveBayes
from Random_Forest import RandomForest
from Rocchio import Rocchio
from SVM import SVM

from mlp import MLP

discriminator_models = {'bagging': Bagging,
                        'bag': Bagging,
                        'decisiontree': DecisionTree,
                        'dt': DecisionTree,
                        'knearestneighbors': KNearestNeighbors,
                        'knn': KNearestNeighbors,
                        'naivebayes': NaiveBayes,
                        'nb': NaiveBayes,
                        'randomforest': RandomForest,
                        'rf': RandomForest,
                        'rocchio': Rocchio,
                        'svm': SVM,
                        }

discriminator_dnn_models = {'mlp': MLP}

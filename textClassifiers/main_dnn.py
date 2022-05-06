# -*- coding: utf-8 -*-
"""
Created on Thu Apr 28 02:17:49 2022

@author: Eduin Hernandez
"""

import argparse
import os
import pickle

from sklearn.feature_extraction.text import TfidfVectorizer

import torch
from torch.utils.data import TensorDataset

from mlp import MLP

from discriminators_all import discriminator_dnn_models as dm
from authors import get_train_path, get_test_path
from authors import authors as authors_dict
from tools import open_authors_list, author_text_prepocess, shuffle, read_file

def str2bool(string):
    if isinstance(string, bool):
       return string
   
    if string.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif string.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def parse_args():
    parser = argparse.ArgumentParser(description='Author Text Classifier')

    'Model Details'
    parser.add_argument('--model-name', type=str, default='mlp', help='Model to use. Can be ' + str(list(dm.keys()))) #120
    parser.add_argument('--char-paragraph', type=int, default=200, help='Number of characters required per paragraph.')
    
    parser.add_argument('--model-filepath', type=str, default='./models/mlp.pt', help='Model load file.')

    parser.add_argument('--show-train-metrics', type=str2bool, default='False', help='Calculate and Show Train metrics')
    parser.add_argument('--show-val-metrics', type=str2bool, default='True', help='Calculate and Show Train metrics')
    parser.add_argument('--show-test-metrics', type=str2bool, default='True', help='Calculate and Show Train metrics')

    parser.add_argument('--test-filepath', type=str, default="../Dataset/HG Wells/wells_test_seqgan.txt", help='Test Text to Evaluate')
    parser.add_argument('--test-author', type=str, default='wells', help='Test Author being evaluated')
    args = parser.parse_args()
    return args

def save(model, filepath):
    torch.save(model, filepath)
    print('Saved as %s' % filepath)

def load(filepath):
    return torch.load(filepath)

if __name__ == '__main__':
    args = parse_args()
    
    assert args.model_name.lower() in list(dm.keys()), 'Error, ' + args.model_name + ' not in the list of models.'
    
    x_train_path, y_train = get_train_path()
    x_train = open_authors_list(x_train_path)
    x_train, y_train = author_text_prepocess(x_train, y_train, args.char_paragraph)
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
    
    if not(args.model_filepath):
        assert False, 'Training option is not available in this file!'
    else:
        model = load(args.model_filepath)
    
    
    if args.show_train_metrics:
        predicted = model.predict(x_train)
        print('Train Metric Results:')
        print(model.classification_report(y_train, predicted))
        print('Accuracy: %1.2f%%\n' % (100*model.accuracy(y_train, predicted)))
        print('Confusion Matrix:\n', model.confusion_matrix(y_train, predicted, labels=[0,1,2,3,4,5]))
    
    del x_train
    del y_train
    
    if args.show_val_metrics:
        x_val_path, y_val = get_test_path()
        x_val = open_authors_list(x_val_path)
        x_val, y_val = author_text_prepocess(x_val, y_val, args.char_paragraph)
        x_val, y_val = shuffle(x_val, y_val)
    
        x_val = vectorizer.transform(x_val).toarray()
        x_val = torch.Tensor(x_val)
        
        predicted = model.predict(x_val)
        print('Validation Metric Results:')
        print(model.classification_report(y_val, predicted))
        print('Accuracy: %1.2f%%\n' % (100*model.accuracy(y_val, predicted)))
        print('Confusion Matrix:\n', model.confusion_matrix(y_val, predicted, labels=[0,1,2,3,4,5]))
        print('\n')
        
        del x_val
        del y_val
        
    if args.show_test_metrics and os.path.exists(args.test_filepath):
        x_test, _ = read_file(args.test_filepath)
        _,_, class_label = authors_dict[args.test_author.lower()]
        x_test, y_test = [x_test], [class_label]
        
        x_test, y_test = author_text_prepocess(x_test, y_test, args.char_paragraph)
        
        x_test = vectorizer.transform(x_test).toarray()
        x_test = torch.Tensor(x_test)
        predicted = model.predict(x_test)
        
        print('Test Metrics Results:')
        print('Accuracy: %1.2f%%' % (100*model.accuracy(y_test, predicted)))
        print('f1-score: %1.2f%%' % (100*model.f1(y_test, predicted)))
        print('Confusion Matrix:\n', model.confusion_matrix(y_test, predicted, labels=[0,1,2,3,4,5]))

        del x_test
        del y_test
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 28 02:17:49 2022

@author: Eduin Hernandez
"""

import argparse
import os

from discriminators_all import discriminator_models as dm
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
    parser.add_argument('--model-name', type=str, default='svm', help='Model to use. Can be ' + str(list(dm.keys()))) #120
    parser.add_argument('--train', type=str2bool, default='False', help='Whether to retrain model if it exists')
    parser.add_argument('--save', type=str2bool, default='True', help='Whether to save model.')

    parser.add_argument('--show-train-metrics', type=str2bool, default='False', help='Calculate and Show Train metrics')
    parser.add_argument('--show-val-metrics', type=str2bool, default='False', help='Calculate and Show Train metrics')
    parser.add_argument('--show-test-metrics', type=str2bool, default='True', help='Calculate and Show Train metrics')

    parser.add_argument('--test-filepath', type=str, default="../Dataset/William Shakespeare/shakespeare_valid.txt", help='Test Text to Evaluate')
    parser.add_argument('--test-author', type=str, default='shakespeare', help='Test Author being evaluated')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    
    assert args.model_name.lower() in list(dm.keys()), 'Error, ' + args.model_name + ' not in the list of models.'
    
    x_train_path, y_train = get_train_path()
    x_train = open_authors_list(x_train_path)
    x_train, y_train = author_text_prepocess(x_train, y_train)
    x_train, y_train = shuffle(x_train, y_train)
    
    model = dm[args.model_name.lower()]()
    
    if args.train or not(model.save_exists()):
        model.fit(x_train, y_train)
    else:
        model.load_model()
    
    if (args.save and not(model.save_exists())) or (args.train and args.save):
        model.save_model()
    
    if args.show_train_metrics:
        predicted = model.predict(x_train)
        print('Train Metric Results:')
        print(model.classification_report(y_train, predicted))
    
    if args.show_val_metrics:
        x_val_path, y_val = get_test_path()
        x_val = open_authors_list(x_val_path)
        x_val, y_val = author_text_prepocess(x_val, y_val)
        x_val, y_val = shuffle(x_val, y_val)
    
        predicted = model.predict(x_val)
        print('Validation Metric Results:')
        print(model.classification_report(y_val, predicted))
    
    if args.show_test_metrics and os.path.exists(args.test_filepath):
        x_test, _ = read_file(args.test_filepath)
        _,_, y_test = authors_dict[args.test_author.lower()]
        x_test, y_test = [x_test], [y_test]
        
        x_test, y_test = author_text_prepocess(x_test, y_test)
        predicted = model.predict(x_test)
        
        print('Test Metrics Results')
        print('Accuracy: ', 100*sum(y_test == predicted)/len(y_test))
        print('Correct: ', sum(y_test == predicted))
        print('Incorrect: ', sum(y_test != predicted))
        print('Sample size: ', len(y_test))
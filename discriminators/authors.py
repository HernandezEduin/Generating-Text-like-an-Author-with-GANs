# -*- coding: utf-8 -*-
"""
Created on Fri Apr 22 00:39:22 2022

@author: Eduin Hernandez
"""
authors = {'shakespeare': ["../Dataset/William Shakespeare/shakespeare_train.txt", "../Dataset/William Shakespeare/shakespeare_valid.txt", 0],
            'dickens': ["../Dataset/Charles Dickens/dickens_train.txt", "../Dataset/Charles Dickens/dickens_val.txt", 1],
            'doyle': ["../Dataset/Arthur Conan Doyle/arthur_train_v2.txt", "../Dataset/Arthur Conan Doyle/arthur_val_v2.txt", 2],
            'eliot': ["../Dataset/George Eliot/eliot_train_v2.txt", "../Dataset/George Eliot/eliot_val_v2.txt", 3],
            'wells': ["../Dataset/HG Wells/wells_train_v2.txt", "../Dataset/HG Wells/wells_val_v2.txt", 4],
            'austen': ["../Dataset/Jane Austen/austen_train_v2.txt", "../Dataset/Jane Austen/austen_val_v2.txt", 5]}

def get_train_path():
    paths = []
    labels = []
    for a0 in authors:
        p0, _, l0 = authors[a0]
        paths.append(p0)
        labels.append(l0)
    return paths, labels

def get_test_path():
    paths = []
    labels = []
    for a0 in authors:
        _, p0, l0 = authors[a0]
        paths.append(p0)
        labels.append(l0)
    return paths, labels
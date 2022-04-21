# -*- coding: utf-8 -*-
"""
Created on Fri Apr 22 00:37:08 2022

@author: Eduin Hernandez
"""

import unidecode
import numpy as np
import random

def read_file(filename):
    file = unidecode.unidecode(open(filename, encoding="utf8").read())
    return file, len(file)

def open_list(authors):
    works = []
    for a0 in authors:
        w0, _ = read_file(a0)
        works.append(w0)
    return works

def shuffle(a,b):
    c = list(zip(a, b))
    random.shuffle(c)
    a, b = zip(*c)
    return list(a), np.array(b)
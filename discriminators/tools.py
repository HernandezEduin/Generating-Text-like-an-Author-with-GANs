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

def open_authors_list(authors):
    works = []
    for a0 in authors:
        w0, _ = read_file(a0)
        works.append(w0)
    return works

def shuffle(a,b):
    'Shuffles both the train and label set together'
    c = list(zip(a, b))
    random.shuffle(c)
    a, b = zip(*c)
    return list(a), list(b)

def author_text_prepocess(x, y, thrs = 200):
    'Separate entire text into list of paragraphs. Paragraph size must be bigger than the thrs'
    x_new = []
    y_new = []
    for i0 in range(len(x)):
        text = x[i0].split('\n\n')
        text = [t0  for t0 in text if (len(t0) > thrs)]
        y_new += [y[i0]] * len(text)
        x_new += text
    return x_new, y_new
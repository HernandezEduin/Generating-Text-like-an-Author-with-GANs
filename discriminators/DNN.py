# -*- coding: utf-8 -*-
"""
Created on Tue Apr 26 01:00:46 2022

@author: Eduin Hernandez
"""

from keras.layers import Dropout, Dense
from keras.models import Sequential
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from sklearn import metrics

from authors import get_train_path, get_test_path
from tools import open_authors_list, author_text_prepocess, shuffle

def TFIDF(X_train, X_test, MAX_NB_WORDS=75000):
    vectorizer_x = TfidfVectorizer(max_features=MAX_NB_WORDS)
    X_train = vectorizer_x.fit_transform(X_train).toarray()
    X_test = vectorizer_x.transform(X_test).toarray()
    # print("tf-idf with",str(np.array(X_train).shape[1]),"features")
    return (X_train,X_test)


def Build_Model_DNN_Text(shape, nClasses, dropout=0.5):
    """
    buildModel_DNN_Tex(shape, nClasses,dropout)
    Build Deep neural networks Model for text classification
    Shape is input feature space
    nClasses is number of classes
    """
    model = Sequential()
    node = 512 # number of nodes
    nLayers = 4 # number of  hidden layer

    model.add(Dense(node,input_dim=shape,activation='relu'))
    model.add(Dropout(dropout))
    for i in range(0,nLayers):
        model.add(Dense(node,input_dim=node,activation='relu'))
        model.add(Dropout(dropout))
    model.add(Dense(nClasses, activation='softmax'))

    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    return model


x_train_path, y_train = get_train_path()
x_train = open_authors_list(x_train_path)
x_train, y_train = author_text_prepocess(x_train, y_train)
x_train, y_train = shuffle(x_train, y_train)

x_test_path, y_test = get_test_path()
x_test = open_authors_list(x_test_path)
x_test, y_test = author_text_prepocess(x_test, y_test)
x_test, y_test = shuffle(x_test, y_test)

x_train_tfidf, x_test_tfidf = TFIDF(x_train, x_test)


model_DNN = Build_Model_DNN_Text(x_train_tfidf.shape[1], 6)
model_DNN.summary()
exit(1)
model_DNN.fit(x_train_tfidf, y_train,
                              validation_data=(x_test_tfidf, y_test),
                              epochs=10,
                              batch_size=128,
                              verbose=2)

predicted = model_DNN.predict_classes(x_test_tfidf)

print(metrics.classification_report(y_test, predicted))
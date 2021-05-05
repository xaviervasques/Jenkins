#!/usr/bin/python3
# train-nn.py
# Xavier Vasques 13/04/2021

import platform
import sys
import scipy

import os
import numpy as np
from sklearn.neural_network import MLPClassifier
import pandas as pd
from joblib import dump, load
from sklearn import preprocessing
from sklearn.model_selection import GridSearchCV

def train():

    # Load directory paths for persisting model

    MODEL_DIR = os.environ["MODEL_DIR"]
    MODEL_FILE_NN = os.environ["MODEL_FILE_NN"]
    MODEL_PATH_NN = os.path.join(MODEL_DIR, MODEL_FILE_NN)

    # Load, read and normalize training data
    training = "./train.csv"
    data_train = pd.read_csv(training)

    y_train = data_train['# Letter'].values
    X_train = data_train.drop(data_train.loc[:, 'Line':'# Letter'].columns, axis = 1)

    # Data normalization (0,1)
    X_train = preprocessing.normalize(X_train, norm='l2')

    # Models training

    # Load, read and normalize training data
    testing = "test.csv"
    data_test = pd.read_csv(testing)

    y_test = data_test['# Letter'].values
    X_test = data_test.drop(data_test.loc[:, 'Line':'# Letter'].columns, axis = 1)

    # Data normalization (0,1)
    X_test = preprocessing.normalize(X_test, norm='l2')

    # Neural Networks multi-layer perceptron (MLP) algorithm that trains using Backpropagation
    param_grid = [
     {
         'activation' : ['identity', 'logistic', 'tanh', 'relu'],
         'solver' : ['lbfgs', 'sgd', 'adam'],
         'hidden_layer_sizes': [(300,),(500,)],
         'max_iter': [1000],
         'alpha': [1e-5, 0.001, 0.01, 0.1, 1, 10],
         'random_state':[0]
     }
    ]

    clf_neuralnet = GridSearchCV(MLPClassifier(), param_grid,scoring='accuracy')
    clf_neuralnet.fit(X_train, y_train)
    print("The Neural Net (few parameters) best prediction is ...")
    print(clf_neuralnet.score(X_test, y_test))
    print("Best parameters set found on development set:")
    print(clf_neuralnet.best_params_)

if __name__ == '__main__':
    train()

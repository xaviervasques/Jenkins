#!/usr/bin/python3
# tain.py
# Xavier Vasques 13/04/2021

import platform
import sys
import numpy
import scipy

import os
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import pandas as pd
from joblib import dump, load
from sklearn import preprocessing

def train():

    # Load directory paths for persisting model

    MODEL_DIR = os.environ["MODEL_DIR"]
    MODEL_FILE_LDA = os.environ["MODEL_FILE_LDA"]
    MODEL_PATH_LDA = os.path.join(MODEL_DIR, MODEL_FILE_LDA)

    # Load, read and normalize training data
    training = "./train.csv"
    data_train = pd.read_csv(training)

    y_train = data_train['# Letter'].values
    X_train = data_train.drop(data_train.loc[:, 'Line':'# Letter'].columns, axis = 1)
    
    # Data normalization (0,1)
    X_train = preprocessing.normalize(X_train, norm='l2')

    # Models training

    # Linear Discrimant Analysis (Default parameters)
    clf_lda = LinearDiscriminantAnalysis()
    clf_lda.fit(X_train, y_train)

    # Serialize model
    from joblib import dump
    dump(clf_lda, MODEL_PATH_LDA)

    # Load, read and normalize testing data
    testing = "test.csv"
    data_test = pd.read_csv(testing)

    y_test = data_test['# Letter'].values
    X_test = data_test.drop(data_test.loc[:, 'Line':'# Letter'].columns, axis = 1)

    # Data normalization (0,1)
    X_test = preprocessing.normalize(X_test, norm='l2')

    # load and Run model
    clf_lda = load(MODEL_PATH_LDA)
    
    print(int(clf_lda.score(X_test, y_test)*100))

if __name__ == '__main__':
    train()

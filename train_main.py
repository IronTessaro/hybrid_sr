import numpy as np
import scipy.io as sio
from sklearn.model_selection import train_test_split
from model_train_lib import train_models


def train_main_models(dataset='data/Dataset_PCP_WHTC.mat', kfolds=2, repeats=1, n_search=1, score='r2'):

    # Load data from .mat file
    data1 = sio.loadmat(dataset)
    X1 = data1['X']
    y1 = data1['y']

    # Ensure y2 is a 1D array
    y1 = y1.flatten()

    # Load data from .mat file
    data2 = sio.loadmat(dataset)
    X2 = data2['X']
    y2 = data2['y']

    # Ensure y2 is a 1D array
    y2 = y2.flatten()

    X = np.concatenate((X1, X2), axis=0)
    y = np.concatenate((y1, y2), axis=0).ravel()

    np.random.seed(31415)

    # --- Training parameters
    train_size = X1.shape[0]/X.shape[0]  # train size

    # ------------- split data, train and evaluate model
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_size, random_state=0, shuffle=True)
    print(X_train.shape)
    print(y_train.shape)

    print(X_test.shape)
    print(y_test.shape)

    train_models(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, output_dir='main_models',
                 kfolds=kfolds, repeats=repeats, n_search=n_search, score=score)

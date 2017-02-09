"""
Data utils
"""
import pickle
import numpy as np
import os
from scipy.misc import imread

def load_cifar10_batch(filename):
    """Load a batch of cifar data """
    with open(filename, "rb") as f:
        datadict = pickle.load(f, encoding="latin-1")
        X = datadict["data"]
        y = datadict["labels"]
        X = np.asarray(X.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1), dtype=np.float64)
        y = np.asarray(y, dtype=np.int32)
        return X, y

def load_cifar10(root):
    """Load all cifar10 data"""
    xs = []
    ys = []
    for b in range(1, 6):
        f = os.path.join(root, "data_batch_{0}".format(b))
        X, y = load_cifar10_batch(f)
        xs.append(X)
        ys.append(y)
    X_train = np.concatenate(xs)
    y_train = np.concatenate(ys)
    X_test, y_test = load_cifar10_batch(os.path.join(root, "test_batch"))
    return X_train, y_train, X_test, y_test


        
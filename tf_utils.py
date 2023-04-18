import numpy as np
import tensorflow as tf


def train_test_split(X, y, train_ratio=0.7, val_ratio=0.15):
   train_split = int(len(X)*train_ratio)
   val_split = int(len(X)*val_ratio)

   X_train = X[:train_split]
   y_train = y[:train_split]

   X_val = X[train_split:train_split+val_split]
   y_val = y[train_split:train_split+val_split]

   X_test = X[train_split+val_split:]
   y_test = y[train_split+val_split:]

   return X_train, X_val, X_test, y_train, y_val, y_test


def prepare_dataset(X, y):
    # Distribute slip and no slip samples
    X_trains = []
    X_vals = []
    X_tests = []
    y_trains = []
    y_vals = []
    y_tests = []
    sample_indices = np.arange(0, len(X), len(X)/6)
    sample_indices = np.append(sample_indices, len(X))
    for i in range(0, len(sample_indices)-1):
        X_train, X_val, X_test, y_train, y_val, y_test = train_test_split(X[int(sample_indices[i]):int(sample_indices[i+1])], y[int(sample_indices[i]):int(sample_indices[i+1])])
        X_trains.append(X_train)
        X_vals.append(X_val)
        X_tests.append(X_test)
        y_trains.append(y_train)
        y_vals.append(y_val)
        y_tests.append(y_test)

    X_trains = np.array(X_trains)
    X_trains = X_trains.reshape(X_trains.shape[0]*X_trains.shape[1], X_trains.shape[2], X_trains.shape[3])
    X_vals = np.array(X_vals)
    X_vals = X_vals.reshape(X_vals.shape[0] * X_vals.shape[1], X_vals.shape[2], X_vals.shape[3])
    X_tests = np.array(X_tests)
    X_tests = X_tests.reshape(X_tests.shape[0]*X_tests.shape[1], X_tests.shape[2], X_tests.shape[3])
    y_trains = np.array(y_trains)
    y_trains = y_trains.reshape(y_trains.shape[0]*y_trains.shape[1], y_trains.shape[2])
    y_vals = np.array(y_vals)
    y_vals = y_vals.reshape(y_vals.shape[0] * y_vals.shape[1], y_vals.shape[2])
    y_tests = np.array(y_tests)
    y_tests = y_tests.reshape(y_tests.shape[0]*y_tests.shape[1], y_tests.shape[2])

    return X_trains, X_vals, X_tests, y_trains, y_vals, y_tests

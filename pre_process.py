"""
Pre processing
"""
import numpy as np
from sklearn.utils import shuffle
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
import pandas as pd
from keras.utils import np_utils
from sklearn.preprocessing import scale


def process_iris_data():
    """
    Process UCI Iris Petal data
    """
    iris_data = pd.read_csv("Iris.csv")
    iris_data = shuffle(iris_data)
    labelencoder = LabelEncoder()
    for col in iris_data.columns:
        iris_data[col] = labelencoder.fit_transform(iris_data[col])
    labels = np.array(iris_data["Species"])
    labelencoder.fit(labels)
    labels = labelencoder.transform(labels)
    labels = np_utils.to_categorical(labels)
    y_train = labels[:150]
    y_test = labels[100:150]
    X = iris_data.iloc[:, 1:5]
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    X_train = np.array(X[:150])
    X_test = np.array(X[100:150])

    return [X_train, y_train, X_test, y_test]


def process_breast_cancer_data():
    Breast_Cancer = pd.read_csv("breastcancer.csv")
    Breast_Cancer.drop(Breast_Cancer.columns[[0, 5]], axis=1, inplace=True)
    X = np.array(Breast_Cancer.iloc[:, [0, 1, 2, 3, 5, 6, 7]])
    X = scale(X)
    labelencoder = LabelEncoder()
    y = np.array(Breast_Cancer.iloc[:, [8]])

    y = y.ravel()
    labelencoder.fit(y)
    y = labelencoder.transform(y)
    y = np_utils.to_categorical(y)

    X_train = X[:550, :]
    X_test = X[550:, :]
    y_train = y[:550, :]
    y_test = y[550:, :]

    return [X_train, y_train, X_test, y_test]

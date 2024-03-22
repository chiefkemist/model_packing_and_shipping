#!/usr/bin/env python3

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder


iris = datasets.load_iris()
X, y = iris.data, iris.target


# One-hot encode the targets since we are dealing with a multi-class classification problem
encoder = OneHotEncoder(sparse_output=False)
y_encoded = encoder.fit_transform(y.reshape(-1, 1))


# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=44
)

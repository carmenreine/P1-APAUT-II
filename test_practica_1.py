# test_practica_1.py
import pytest
import numpy as np
from KNNClassifier import KNNClassifier
from mRMR import mRMR


# ------------------------------
# Datos de juguete para KNN
# ------------------------------
def make_toy_dataset():
    X = np.array([
        [0.0, 0.0],
        [1.0, 1.0],
        [0.0, 1.0],
        [1.0, 0.0],
        [10.0, 10.0],
        [9.0, 9.0],
        [10.0, 9.0],
        [9.0, 10.0],
    ])
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    return X, y


# distancia Manhattan (L1)
def manhattan(a, b):
    return np.sum(np.abs(a - b))


# ------------------------------
# Tests para KNNClassifier
# ------------------------------
def test_knn_fit_and_predict_uniform():
    X, y = make_toy_dataset()
    clf = KNNClassifier(k=3, distance_metric=manhattan)
    clf.fit(X, y)

    x_test = np.array([[0.2, 0.1]])  # cerca de clase 0
    y_pred = clf.predict(x_test)
    assert y_pred[0] == 0

    x_test = np.array([[9.5, 11.0]])  # cerca de clase 1
    y_pred = clf.predict(x_test)
    assert y_pred[0] == 1


def test_knn_predict_without_fit():
    clf = KNNClassifier(k=3)
    X_test = np.array([[1, 2]])
    # tu implementación lanza RuntimeError, no ValueError
    with pytest.raises(RuntimeError):
        clf.predict(X_test)


# ------------------------------
# Datos de juguete para mRMR
# ------------------------------
@pytest.fixture
def data_mrmr():
    X = np.array([
        [0, 2, 1],
        [1, 5, 0],
        [1, 8, 0],
        [0, 0, 1],
        [0, 1, 1]
    ])
    y = np.array([0, 1, 1, 0, 0])
    return X, y


# ------------------------------
# Tests para mRMR
# ------------------------------
def test_mRMR_basic(data_mrmr):
    """
    Test para verificar si el método mRMR selecciona correctamente los atributos.
    """
    X, y = data_mrmr
    my_mRMR = mRMR(2)
    my_mRMR.fit(X, y)
    X_transformed = my_mRMR.transform(X)
    assert X_transformed.shape[1] == 2
    assert np.array_equal(X_transformed, X[:, [0, 2]])


def test_mRMR_all_features():
    """
    Si pides todas las columnas, deben devolverse todas.
    """
    X = np.array([
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9],
        [2, 3, 4]
    ])
    y = np.array([0, 1, 0, 1])
    my_mRMR = mRMR(3)
    my_mRMR.fit(X, y)
    X_transformed = my_mRMR.transform(X)
    assert X_transformed.shape[1] == 3
